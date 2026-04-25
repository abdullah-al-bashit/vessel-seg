import numpy as np
import torch
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes
import sknw
import networkx as nx
from torch_geometric.nn import GATConv
import torch.nn.functional as F


# ── Rule-based postprocess ─────────────────────────────────────────────────────

def postprocess(mask, min_size=100):
    """
    Clean up predicted binary mask.
    mask: (H, W) bool

    Steps:
      1. Remove isolated blobs smaller than min_size pixels
      2. Fill small holes inside vessels
      3. Light morphological closing to smooth boundaries
    """
    mask = remove_small_objects(mask, min_size=min_size)
    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, footprint=disk(2))
    return mask.astype(bool)


# ── GAT topology refinement ────────────────────────────────────────────────────

class VesselGAT(torch.nn.Module):
    """
    2-layer GAT for vessel edge classification.
    Task: classify each edge as real / false positive / broken
    Input node features: (x, y, cos_theta, sin_theta, diameter, curvature)  = 6
    """
    def __init__(self, in_ch=6, hidden=32, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_ch,  hidden, heads=heads, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False)
        self.edge_head = torch.nn.Linear(hidden * 2, 3)   # 3 classes

    def forward(self, x, edge_index):
        """
        x:          (N, 6) node features
        edge_index: (2, E)
        Returns:    (E, 3) edge logits  [real, false_pos, broken]
        """
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)           # (N, hidden)

        # Per-edge feature: concat both endpoint embeddings
        src, dst = edge_index
        e_feat   = torch.cat([x[src], x[dst]], dim=1)  # (E, hidden*2)
        return self.edge_head(e_feat)            # (E, 3)


def build_graph_from_mask(mask):
    """
    Skeletonize mask and extract vessel graph.
    Returns networkx graph or None.
    """
    skel  = skeletonize(mask)
    graph = sknw.build_sknw(skel.astype(np.uint16))
    return graph


def node_features(graph, mask):
    """
    Extract 6-dim anisotropic features per node.
    Returns (N, 6) float32 array and node list.
    """
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(mask)

    nodes = list(graph.nodes())
    feats = []

    for n in nodes:
        y, x = map(int, graph.nodes[n]['o'])
        y = np.clip(y, 0, mask.shape[0]-1)
        x = np.clip(x, 0, mask.shape[1]-1)

        angles = []
        for nb in graph.neighbors(n):
            pts = graph[n][nb]['pts']
            if len(pts) >= 2:
                dy = pts[-1][0] - pts[0][0]
                dx = pts[-1][1] - pts[0][1]
                angles.append(np.arctan2(dy, dx))
        theta = np.mean(angles) if angles else 0.0

        d     = float(dist[y, x]) * 2

        kappa = 0.0
        for nb in graph.neighbors(n):
            pts = graph[n][nb]['pts']
            if len(pts) >= 3:
                v1 = pts[1] - pts[0]
                v2 = pts[-1] - pts[-2]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                norm  = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                kappa += abs(cross / norm)

        feats.append([
            x / mask.shape[1],
            y / mask.shape[0],
            np.cos(theta),
            np.sin(theta),
            d / max(mask.shape),
            kappa,
        ])

    return np.array(feats, dtype=np.float32), nodes


def gnn_topology_refine(mask, gat_model=None, device='cpu', conf_thresh=0.7):
    """
    Optional GNN topology refinement stage.
    If gat_model is None, returns mask unchanged (safe fallback).

    Steps:
      1. Skeletonize mask → vessel graph
      2. Extract node features
      3. Run GAT edge classifier
      4. Remove edges classified as false positive
      5. Reconnect edges classified as broken

    mask:      (H, W) bool
    Returns:   (H, W) bool  refined mask
    """
    if gat_model is None:
        return mask                              # no GNN — pass through

    graph = build_graph_from_mask(mask)
    if graph is None or len(graph.nodes()) < 2:
        return mask

    feats_np, nodes = node_features(graph, mask)
    if len(feats_np) < 2:
        return mask

    node_idx = {n: i for i, n in enumerate(nodes)}
    edges    = list(graph.edges())
    if len(edges) == 0:
        return mask

    src = [node_idx[u] for u, v in edges]
    dst = [node_idx[v] for u, v in edges]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long).to(device)
    x          = torch.from_numpy(feats_np).to(device)

    gat_model.eval()
    with torch.no_grad():
        edge_logits = gat_model(x, edge_index)  # (E, 3)
        edge_probs  = torch.softmax(edge_logits, dim=1)
        edge_labels = edge_probs.argmax(dim=1)  # 0=real, 1=false_pos, 2=broken

    # Remove false positive edges from the graph
    refined_graph = graph.copy()
    for i, (u, v) in enumerate(edges):
        label = edge_labels[i].item()
        conf  = edge_probs[i, label].item()
        if label == 1 and conf > conf_thresh:    # false positive — remove
            if refined_graph.has_edge(u, v):
                refined_graph.remove_edge(u, v)

    # Reconnect broken edges: find disconnected endpoints and bridge them
    # (simple heuristic: connect nearest endpoints within max_gap pixels)
    max_gap = 20
    endpoints = [n for n in refined_graph.nodes()
                 if refined_graph.degree(n) == 1]
    used = set()
    for ep in endpoints:
        if ep in used:
            continue
        y1, x1 = map(int, refined_graph.nodes[ep]['o'])
        best_dist, best_ep = np.inf, None
        for other in endpoints:
            if other == ep or other in used:
                continue
            y2, x2 = map(int, refined_graph.nodes[other]['o'])
            dist = np.sqrt((y1-y2)**2 + (x1-x2)**2)
            if dist < best_dist and dist < max_gap:
                best_dist = dist
                best_ep   = other
        if best_ep is not None:
            refined_graph.add_edge(ep, best_ep)
            used.add(ep); used.add(best_ep)

    # Convert refined graph back to mask (draw skeleton edges on image)
    from skimage.draw import line as draw_line
    refined_mask = mask.copy()
    for u, v, data in refined_graph.edges(data=True):
        pts = data.get('pts', None)
        if pts is not None and len(pts) > 0:
            for pt in pts:
                ry, rx = int(pt[0]), int(pt[1])
                if 0 <= ry < mask.shape[0] and 0 <= rx < mask.shape[1]:
                    refined_mask[ry, rx] = True
        else:
            y1, x1 = map(int, refined_graph.nodes[u]['o'])
            y2, x2 = map(int, refined_graph.nodes[v]['o'])
            rr, cc = draw_line(y1, x1, y2, x2)
            rr = np.clip(rr, 0, mask.shape[0]-1)
            cc = np.clip(cc, 0, mask.shape[1]-1)
            refined_mask[rr, cc] = True

    return refined_mask.astype(bool)

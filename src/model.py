import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx                                    # connected-component analysis for component_size feature
from torch_geometric.nn import ChebConv                 # Chebyshev spectral graph convolution
from skimage.morphology import skeletonize               # morphological thinning → 1px centerline
from torchvision.transforms.functional import to_pil_image   # tensor → PIL for SAM2 processor
import sknw                                              # skeleton → networkx graph (nodes=junctions/tips, edges=vessel segments)

HF_MODEL_ID = "facebook/sam2.1-hiera-large"             # SAM2.1 Hiera-Large — newest checkpoint on HuggingFace

# ── Feature dimensions ─────────────────────────────────────────────────────────
# Node features (42 total):
#   [0]   x_norm              normalized x position ∈ [0,1]
#   [1]   y_norm              normalized y position ∈ [0,1]
#   [2]   cos(θ)              vessel orientation cos ∈ [-1,1]
#   [3]   sin(θ)              vessel orientation sin ∈ [-1,1]  (anisotropic pair)
#   [4]   diameter            local vessel width ∈ [0,1]        fixes broken thin vessels
#   [5]   curvature κ         bending rate at node              fixes bifurcation errors
#   [6]   degree              number of branches at node        detects endpoints/bifurcations/false blobs
#   [7]   component_size      normalized size of vessel network isolates false positive blobs
#   [8]   endpoint_distance   normalized dist to nearest tip    locates break-prone positions
#   [9]   diameter_consistency std of diameter along all edges  flags sudden width changes
#   [10-41] CNN features (32) local appearance from decoder     pixel-level evidence
#
# Edge features (4 total):
#   [0]   length_norm         normalized path length            vessel segment size
#   [1]   gap_intensity_mean  mean image brightness along path  detects real vs phantom vessels
#   [2]   gap_intensity_std   std of brightness along path      consistent signal = real vessel
#   [3]   delta_theta         angle difference between nodes    detects false branches at bifurcations
NODE_FEAT_DIM = 42   # 10 geometric/topological + 32 CNN
EDGE_FEAT_DIM = 4    # 4 edge descriptors


# ── SAM2 Wrapper — HuggingFace ─────────────────────────────────────────────────

class SAM2Encoder(nn.Module):
    """
    Frozen Hiera ViT-L encoder via HuggingFace transformers.
    Auto-downloads weights from facebook/sam2.1-hiera-large on first run.
    No manual checkpoint file needed.
    Outputs 4-scale FPN feature maps at 256 channels each.
    FPN (Feature Pyramid Network): multi-scale feature maps at different resolutions
    (e.g. H/4, H/8, H/16, H/32) so both fine detail and broad context are captured.
    feats[0] is the finest (highest resolution), feats[3] is the coarsest.
    """
    def __init__(self):
        super().__init__()
        from transformers import Sam2Model, Sam2Processor  # lazy import — avoids load at module level

        self.processor = Sam2Processor.from_pretrained(HF_MODEL_ID)  # handles image pre-processing (resize, normalize, pad)
        self.sam2      = Sam2Model.from_pretrained(HF_MODEL_ID)       # full SAM2 model (encoder + prompt + decoder)

        # Freeze entire vision encoder — Hiera ViT weights never updated during training
        for name, p in self.sam2.named_parameters():
            if 'vision_encoder' in name:       # only the Hiera trunk, not prompt/mask decoder
                p.requires_grad = False        # no gradient → no weight update → no optimizer memory

        frozen    = sum(p.numel() for p in self.sam2.parameters() if not p.requires_grad)   # scalar int — total frozen param count
        trainable = sum(p.numel() for p in self.sam2.parameters() if p.requires_grad)       # scalar int — total trainable param count
        print(f'SAM2 encoder  frozen:    {frozen:,}')
        print(f'SAM2 decoder  trainable: {trainable:,}')

    def forward(self, x):
        """
        x:  (B, 1, H, W)  float32 in [0, 1]   single-channel tile
        Returns list of 4 FPN feature maps, each 256 channels.
        """
        B = x.shape[0]   # scalar int — batch size

        # Processor expects PIL images — convert 1-channel tensor to 3-channel PIL
        pil_images = [
            to_pil_image(x[i].repeat(3, 1, 1).clamp(0, 1).cpu())   # (1,H,W) → (3,H,W) → PIL RGB
            for i in range(B)
        ]
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"          # dict: {'pixel_values': (B,3,1024,1024), ...}
        ).to(x.device)

        with torch.no_grad():            # frozen — no gradients, saves memory and compute
            vision_out = self.sam2.vision_encoder(**inputs)

        return vision_out.feature_maps   # list of 4 tensors: [(B,256,H/4,W/4), ..., (B,256,H/32,W/32)]


# ── CNN Upsampling Path ────────────────────────────────────────────────────────

class CNNDecoder(nn.Module):
    """
    Lightweight upsampling decoder.
    Takes highest-resolution FPN feature (256ch) and upsamples 4×.
    Outputs F_pixel: (B, 32, H, W)
    """
    def __init__(self, in_ch=256):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 64, kernel_size=2, stride=2),  # 2× upsample: (B,256,H/4,W/4)→(B,64,H/2,W/2)
            nn.BatchNorm2d(64),    # normalizes across (B,H,W) per channel → stable training
            nn.ReLU(inplace=True), # inplace=True: overwrites input tensor → saves one tensor allocation
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),     # 2× upsample: (B,64,H/2,W/2)→(B,32,H,W)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        """feats: list of 4 FPN feature maps; use finest scale feats[0]"""
        x = self.up1(feats[0])   # (B, 256, H/4, W/4) → (B, 64, H/2, W/2)
        x = self.up2(x)          # (B, 64, H/2, W/2)  → (B, 32, H, W)
        return x                 # (B, 32, H, W) — pixel-aligned feature map F_pixel


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_vessel_graph(mask_np, img_np, pixel_feats_np):
    """
    Build vessel graph from binary mask with topology-aware node and edge features.

    mask_np:        (H, W)     bool    — binary vessel prediction
    img_np:         (H, W)     float32 — raw image [0,1] for intensity evidence
    pixel_feats_np: (H, W, 32) float32 — CNN decoder features for appearance context

    Coordinate system (image pixel grid):
        col (x) →  0                W-1
        row (y) ↓  (0,0)─────────────┐
                   │                 │
                   │   image tile    │
                   │                 │
                   └─────────────(W-1, H-1)
                   H-1

        node_pos stores (x=col, y=row): x increases right, y increases downward.
        e.g. top-left corner = (0, 0),  bottom-right = (W-1, H-1)

    Returns dict with keys:
        node_pos:   (N, 2)             (x=col, y=row) pixel position of each skeleton node
                                       e.g. N=3 nodes → [[512, 200], [530, 350], [480, 500]]
                                            x=512,y=200  x=530,y=350  x=480,y=500

        node_feats: (N, NODE_FEAT_DIM) 42-dim feature vector per node
                                       e.g. N=3 → shape (3, 42); row 0 = features of node 0

        edge_index: (2, E)             directed edge list for all E edges (each undirected edge stored twice)
                                       row 0 = source indices, row 1 = destination indices
                                       e.g. 2 undirected edges A-B, B-C → E=4 directed edges:
                                            [[0, 1, 1, 2],   ← sources:      A→B, B→A, B→C, C→B
                                             [1, 0, 2, 1]]   ← destinations: A→B, B→A, B→C, C→B

        edge_feats: (E, EDGE_FEAT_DIM) 4-dim feature vector per directed edge
                                       e.g. E=4 → shape (4, 4); rows 0,1 are A-B (forward/reverse)
    Returns None if graph is degenerate (< 2 nodes or no edges).
    """
    H, W = mask_np.shape   # scalar ints — image dimensions

    # ── Skeletonize: 2D vessel region → 1-pixel-wide centerline ───────────
    # Repeatedly erodes the vessel mask inward until only a 1-pixel-wide spine remains.
    # True pixels = centerline; False = background or vessel interior that was eroded away.
    # e.g. a 10px-wide vessel region becomes a single-pixel chain running down its center.
    skel  = skeletonize(mask_np) 
    # (H, W) bool — same spatial size as mask_np; True only on the 1px centerline

    # Convert the skeleton pixel image into a graph of vessel topology.
    # sknw scans the skeleton for pixels with unusual connectivity:
    #
    #   degree = number of distinct vessel segments meeting at a skeleton pixel
    #
    #   degree=1 → TIP/ENDPOINT
    #     skeleton pixel with only one neighbour — the vessel ends here
    #
    #     · · T · ·   ← tip (degree=1): background above, X below — only ONE neighbour
    #     · · X · ·        'X' = interior pixel — neighbour above AND below → degree=2, NOT a node
    #     · · X · ·        'T' = tip node — only ONE neighbour (the pixel directly next to it)
    #     · · T · ·   ← tip (degree=1): X above, background below — only ONE neighbour
    #     · · · · ·        '·' = background pixel (False in skeleton)
    #
    #   degree≥3 → JUNCTION/BIFURCATION
    #     skeleton pixel with 3 or more neighbours — the vessel forks here
    #
    #     · · X · ·        three segments radiate from J:
    #     · · X · ·          up, down-left, down-right
    #     · · J · ·   ← J = junction node (degree=3)
    #     · X · X ·
    #     X · · · X
    #
    #   degree=2 → NOT a node — just a straight-run pixel stored in the edge 'pts' path
    #
    #     · · · · ·        graph.nodes[u]['o'] = ( 9, 20)   u = sknw integer ID of top    T node
    #     · · T · ·   T (deg=1) ──── p ──── p ──── T (deg=1)
    #     · · p · ·   pts = [[10, 20],   ← interior pixel 1: row=10, col=20
    #     · · p · ·          [11, 20]]   ← interior pixel 2: row=11, col=20   L=len(pts)=2
    #     · · T · ·        graph.nodes[v]['o'] = (12, 20)   v = sknw integer ID of bottom T node
    #                 u, v: sknw integer node IDs, assigned 0,1,2,... in order of discovery
    #                   straight vessel (2 nodes)  → u=0 (top T), v=1 (bottom T), 1 edge
    #                   Y-shaped vessel (4 nodes)  → u=0,1,2 (tips), v=3 (junction), 3 edges
    #                 sknw graph is UNDIRECTED: graph[u][v] and graph[v][u] return the same pts
    #                 we store BOTH directions in edge_index so ChebConv can pass messages u→v AND v→u
    #
    # e.g. a straight vessel  → 2 nodes (2 tips deg=1),       1 edge  with 1 pts array of L pixels
    #      a Y-shaped vessel  → 4 nodes (1 junction + 3 tips), 3 edges each with its own pts array
    #                           branch 0: tip0 ── pts_0 (L0 pixels) ── junction
    #                           branch 1: tip1 ── pts_1 (L1 pixels) ── junction
    #                           branch 2: tip2 ── pts_2 (L2 pixels) ── junction
    graph = sknw.build_sknw(skel.astype(np.uint16))
    # sknw requires uint16 input; bool skel cast to 0/1 uint16
    # returns networkx.Graph; node IDs are integers 0,1,2,... assigned by sknw
    #
    #   n, u, v are all the SAME type: sknw integer node IDs from the same pool {0,1,2,...}
    #   n   = used alone  → look up one node's pixel position
    #   u,v = used as pair → look up the edge that connects node u to node v
    #
    #   Straight vessel example (nodes = [0, 1]):
    #
    #     n=0 (top T):    graph.nodes[0]['o'] = ( 9, 20)   ← pixel position of node 0
    #     n=1 (bottom T): graph.nodes[1]['o'] = (12, 20)   ← pixel position of node 1
    #
    #     u=0, v=1 share an edge (node 0 is connected to node 1 by a vessel segment):
    #       graph[0][1]['pts'] = [[10, 20],   ← interior pixel between node 0 and node 1
    #                             [11, 20]]   ← (L=2 interior pixels; nodes themselves NOT in pts)

    nodes = list(graph.nodes())   # list of N node IDs, e.g. [0, 1] (straight) or [0,1,2,3] (Y-shape)
    if len(nodes) < 2:
        return None               # need ≥2 nodes to have any edge; 0 or 1 node → no message passing possible

    # ── Pre-compute distance transform once (reused for diameter + consistency) ──
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(mask_np)
    # (H, W) float64 — each pixel's Euclidean distance to the nearest background (False) pixel
    #
    # 2-D patch example — a 5-pixel-wide horizontal vessel (rows 2-4 are vessel):
    #
    #   F = background (mask=False, dist=0)
    #   T = vessel     (mask=True,  dist = Euclidean dist to nearest F pixel)
    #   * = skeleton centerline pixel (local dist maximum along the vessel axis)
    #
    #   row\col  0    1    2    3    4    5    6    7    8
    #      0     F    F    F    F    F    F    F    F    F    ← all background
    #      1     F    F    F    F    F    F    F    F    F    ← all background
    #      2     F    F    T    T    T    T    T    F    F    ← vessel top edge
    #      3     F    F    T    T    T    T    T    F    F    ← vessel middle row (skeleton)
    #      4     F    F    T    T    T    T    T    F    F    ← vessel bottom edge
    #      5     F    F    F    F    F    F    F    F    F    ← all background
    #      6     F    F    F    F    F    F    F    F    F    ← all background
    #
    #   dist values for the same patch (Euclidean distance to nearest F):
    #
    #   row\col  0    1    2    3    4    5    6    7    8
    #      0     0    0    0    0    0    0    0    0    0
    #      1     0    0    0    0    0    0    0    0    0
    #      2     0    0    1    1    1    1    1    0    0    ← 1 step from background
    #      3     0    0    1    2    2    2    1    0    0    ← center row: dist=2 (vessel radius=2)
    #      4     0    0    1    1    1    1    1    0    0    ← 1 step from background
    #      5     0    0    0    0    0    0    0    0    0
    #      6     0    0    0    0    0    0    0    0    0
    #
    #   skeleton runs along row=3 (the row of maximum dist values);
    #   skeleton pixels at (3,3), (3,4), (3,5) all have dist=2 = vessel radius
    #   diameter = 2 × dist = 4 pixels  (≈ vessel width, here exactly 4 rows wide including edges)
    #
    #   corner pixels such as (2,2) get dist=1 because the nearest background pixel is
    #   directly above (1,2) or directly left (2,1), both 1 step away — Euclidean distance
    #   takes the shortest path in 2D, not just along rows or columns

    # ── Pre-compute connected components for component_size feature ────────
    # component_size: how large is the vessel network this node belongs to?
    # Large component = real vessel network. Small isolated component = likely false positive.
    components      = list(nx.connected_components(graph))           # list of sets of node IDs
    node_to_comp    = {}                                             # dict: node_id → component index
    for ci, comp in enumerate(components):
        for nid in comp:
            node_to_comp[nid] = ci
    comp_sizes      = [len(c) for c in components]                  # list of int — node count per component
    max_comp_size   = max(comp_sizes) if comp_sizes else 1          # scalar int — largest component size for normalization

    # ── Pre-compute endpoint nodes for endpoint_distance feature ───────────
    # Endpoints = nodes with degree 1 (vessel tips). These are break-prone positions.
    # Nodes close to an endpoint are more likely to be near a real break.
    endpoint_nodes  = [n for n in nodes if graph.degree(n) == 1]    # list of node IDs with degree=1
    endpoint_pos    = np.array(
        [graph.nodes[n]['o'] for n in endpoint_nodes], dtype=np.float32
    ) if endpoint_nodes else None                                    # (n_endpoints, 2) float32 or None

    # ── Build node features ────────────────────────────────────────────────
    node_pos   = []    # will become (N, 2)              float32 — (x, y) pixel coords
    node_feats = []    # will become (N, NODE_FEAT_DIM)  float32 — feature vectors

    for n in nodes:
        y, x = graph.nodes[n]['o']                   # (2,) — skeleton node position as (row, col)
        y, x = int(y), int(x)
        y = np.clip(y, 0, H - 1)                     # scalar int — row clamped to valid range
        x = np.clip(x, 0, W - 1)                     # scalar int — col clamped to valid range

        # ── Feature 1-2: normalized position ──────────────────────────────
        x_norm = x / W                               # scalar float ∈ [0,1] — where in image horizontally
        y_norm = y / H                               # scalar float ∈ [0,1] — where in image vertically

        # ── Feature 3-4: vessel direction θ as (cos θ, sin θ) ─────────────
        # Stored as cos/sin pair — avoids discontinuity at ±π that raw angle has.
        # Anisotropic: allows ChebConv to distinguish vessel direction at bifurcations.
        angles = []                                  # list of float — one angle per connected edge
        for nb in graph.neighbors(n):
            pts = graph[n][nb]['pts']                # (L, 2) int — pixel path along this edge
            if len(pts) >= 2:
                dy = float(pts[-1][0] - pts[0][0])  # scalar float — row displacement
                dx = float(pts[-1][1] - pts[0][1])  # scalar float — col displacement
                angles.append(np.arctan2(dy, dx))   # scalar float ∈ [-π, π]
        theta = float(np.mean(angles)) if angles else 0.0  # scalar float — mean orientation; 0 if isolated

        # ── Feature 5: diameter ────────────────────────────────────────────
        # dist[y,x] = radius (distance to nearest background pixel).
        # Multiply by 2 = diameter. Normalize by max image dimension.
        # Key failure fix: sudden diameter changes along a vessel → false positive or break.
        d_norm = float(dist[y, x]) * 2 / max(H, W)  # scalar float ∈ [0,1]

        # ── Feature 6: curvature κ ─────────────────────────────────────────
        # Cross product of start/end tangent vectors of each edge at this node.
        # |cross / (|v1|·|v2|)| = |sin(bend angle)| ≈ curvature.
        # High κ = sharp bend. Low κ = straight vessel.
        kappa = 0.0                                  # scalar float — accumulated bend across all edges
        for nb in graph.neighbors(n):
            pts = graph[n][nb]['pts']                # (L, 2) int
            if len(pts) >= 3:
                v1    = pts[1]  - pts[0]             # (2,) int — tangent at start
                v2    = pts[-1] - pts[-2]            # (2,) int — tangent at end
                cross = float(v1[0]*v2[1] - v1[1]*v2[0])          # scalar — z-component of 2D cross product
                norm  = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
                kappa += abs(cross / norm)           # accumulate |sin(angle)| per edge

        # ── Feature 7: degree ──────────────────────────────────────────────
        # Number of vessel branches at this node.
        # degree=1 → endpoint (vessel tip, break-prone)
        # degree=2 → straight segment (interior node)
        # degree=3+ → bifurcation (branch point)
        # degree=0 → isolated node (strong false positive signal)
        # Normalized by 4 — most vessels have at most 3-way bifurcations.
        deg_norm = float(graph.degree(n)) / 4.0     # scalar float ∈ [0, ~1]

        # ── Feature 8: component_size ──────────────────────────────────────
        # Normalized size (node count) of the connected vessel network this node belongs to.
        # Real vessel: part of a large network → high component_size.
        # False positive blob: isolated small component → low component_size.
        # This is the primary feature for detecting spurious isolated blobs.
        ci        = node_to_comp[n]                  # scalar int — component index
        comp_norm = float(comp_sizes[ci]) / float(max_comp_size)  # scalar float ∈ [0,1]

        # ── Feature 9: endpoint_distance ──────────────────────────────────
        # Distance from this node to the nearest vessel endpoint (degree-1 node).
        # Nodes near an endpoint are at positions where vessels commonly break.
        # The GAT uses this to identify break-prone regions for reconnection.
        if endpoint_pos is not None and len(endpoint_pos) > 0:
            node_yx   = np.array([y, x], dtype=np.float32)         # (2,) float32 — current node position
            dists_to_eps = np.linalg.norm(endpoint_pos - node_yx, axis=1)  # (n_endpoints,) — Euclidean dist to each endpoint
            ep_dist   = float(dists_to_eps.min()) / max(H, W)      # scalar float ∈ [0,1] — normalized nearest endpoint dist
        else:
            ep_dist = 1.0                            # no endpoints in graph → far from any endpoint

        # ── Feature 10: diameter_consistency ──────────────────────────────
        # Standard deviation of vessel diameter sampled along all edges at this node.
        # Consistent diameter along a segment → real vessel (low std).
        # Sudden diameter change → false positive joining two different structures (high std).
        diameters_along_edges = []
        for nb in graph.neighbors(n):
            pts = graph[n][nb]['pts']                # (L, 2) int — pixel path
            if len(pts) > 0:
                ys_e = np.clip(pts[:, 0].astype(int), 0, H-1)   # (L,) int
                xs_e = np.clip(pts[:, 1].astype(int), 0, W-1)   # (L,) int
                diameters_along_edges.extend(dist[ys_e, xs_e].tolist())  # list of floats — radius at each path pixel
        diam_std = float(np.std(diameters_along_edges)) / max(H, W) if diameters_along_edges else 0.0
        # scalar float — low = consistent width (good vessel), high = variable width (suspicious)

        # ── Feature 11-42: CNN features ────────────────────────────────────
        # 32-dim appearance vector from the CNN decoder at this node's pixel position.
        # Provides local texture, intensity, and learned vessel context.
        cnn_feat = pixel_feats_np[y, x]             # (32,) float32 — sampled from (H, W, 32) feature map

        # ── Assemble 42-dim node feature vector ────────────────────────────
        node_pos.append([x, y])                     # list grows to (N, 2)
        node_feats.append(np.concatenate([
            [x_norm,                # [0]  position x          ∈ [0,1]
             y_norm,                # [1]  position y          ∈ [0,1]
             np.cos(theta),         # [2]  direction cos       ∈ [-1,1]
             np.sin(theta),         # [3]  direction sin       ∈ [-1,1]
             d_norm,                # [4]  diameter            ∈ [0,1]   fixes broken thin vessels
             kappa,                 # [5]  curvature           ∈ [0,∞]  fixes bifurcation errors
             deg_norm,              # [6]  degree              ∈ [0,~1]  detects endpoints/blobs
             comp_norm,             # [7]  component_size      ∈ [0,1]   isolates false blobs
             ep_dist,               # [8]  endpoint_distance   ∈ [0,1]   locates break positions
             diam_std],             # [9]  diameter_std        ∈ [0,1]   flags width changes
            cnn_feat                # [10-41] CNN features     (32,)     local appearance
        ]))                         # (42,) float32 — NODE_FEAT_DIM

    # ── Build edge features ────────────────────────────────────────────────
    node_idx            = {n: i for i, n in enumerate(nodes)}   # dict: networkx node ID → integer index 0..N-1
    edge_src, edge_dst  = [], []                                 # will become (E,) each
    edge_feats_list     = []                                     # will become (E, EDGE_FEAT_DIM)

    for u, v, data in graph.edges(data=True):
        if u not in node_idx or v not in node_idx:
            continue
        i, j = node_idx[u], node_idx[v]             # scalar ints — integer indices

        pts    = data.get('pts', np.array([]))       # (L, 2) int — pixel path along edge; empty if direct neighbour
        length = float(len(pts)) if len(pts) > 0 else 1.0   # scalar float — path length in pixels

        # ── Edge feature 1: normalized length ─────────────────────────────
        length_norm = length / max(H, W)             # scalar float ∈ [0,1]

        # ── Edge feature 2: gap_intensity_mean ────────────────────────────
        # Mean image brightness along the vessel path.
        # High mean → fluorescence signal present → real vessel.
        # Low mean  → dark region between two predicted segments → likely a false prediction.
        if len(pts) > 0:
            ys_e = np.clip(pts[:, 0].astype(int), 0, H-1)   # (L,) int — row coords
            xs_e = np.clip(pts[:, 1].astype(int), 0, W-1)   # (L,) int — col coords
            intensities  = img_np[ys_e, xs_e]               # (L,) float32 — pixel intensities along path
            gap_mean     = float(intensities.mean())         # scalar float ∈ [0,1]
            # ── Edge feature 3: gap_intensity_std ─────────────────────────
            # Standard deviation of brightness along the path.
            # Low std  → uniform signal → continuous vessel.
            # High std → patchy signal → broken or uncertain vessel segment.
            gap_std  = float(intensities.std())              # scalar float ∈ [0,1]
        else:
            gap_mean = 0.0                                   # no path pixels → unknown intensity
            gap_std  = 0.0

        # ── Edge feature 4: delta_theta ────────────────────────────────────
        # Angular difference between the orientations of the two endpoint nodes.
        # Low Δθ  → both nodes point in the same direction → straight consistent vessel.
        # High Δθ → nodes point in different directions → sharp bend or false branch at bifurcation.
        fi          = node_feats[i]                          # (42,) float32 — source node features
        fj          = node_feats[j]                          # (42,) float32 — destination node features
        ti          = np.arctan2(fi[3], fi[2])               # scalar float — recover θ_i from (sin=index3, cos=index2)
        tj          = np.arctan2(fj[3], fj[2])               # scalar float — recover θ_j
        delta_theta = float(abs(ti - tj))                    # scalar float ∈ [0, 2π]

        # Add edge in both directions (undirected graph)
        for src, dst in [(i, j), (j, i)]:
            edge_src.append(src)
            edge_dst.append(dst)
            edge_feats_list.append([
                length_norm,    # [0] normalized length       ∈ [0,1]  vessel segment size
                gap_mean,       # [1] mean intensity          ∈ [0,1]  real vs phantom vessel
                gap_std,        # [2] intensity std           ∈ [0,1]  signal consistency
                delta_theta,    # [3] angular difference      ∈ [0,2π] bend / false branch
            ])                  # (4,) — EDGE_FEAT_DIM

    if len(edge_src) == 0:
        return None             # no edges → cannot do message passing

    return {
        'node_pos':   np.array(node_pos,        dtype=np.float32),   # (N, 2)
        'node_feats': np.array(node_feats,      dtype=np.float32),   # (N, 42)
        'edge_index': np.array([edge_src, edge_dst], dtype=np.int64),# (2, E)
        'edge_feats': np.array(edge_feats_list, dtype=np.float32),   # (E, 4)
    }


# ── ChebConv Graph Network ─────────────────────────────────────────────────────

class VesselGraphNet(nn.Module):
    """
    3-layer ChebConv with K=10 polynomial hops.
    Input:  node_feats (N, NODE_FEAT_DIM=42)  anisotropic + topological features
    Output: node_emb   (N, 64)               enriched embeddings encoding vessel topology

    K=10: each node aggregates information from up to 10 hops away per layer.
    3 layers → 30 hops total → covers full vessel length regardless of pixel distance.
    """
    def __init__(self, in_ch=NODE_FEAT_DIM, hidden=64, out_ch=64, K=10):
        super().__init__()
        self.conv1 = ChebConv(in_ch,  hidden, K=K)   # (N, 42) → (N, 64): aggregates 10-hop neighbourhood
        self.conv2 = ChebConv(hidden, hidden, K=K)   # (N, 64) → (N, 64): deeper topology context
        self.conv3 = ChebConv(hidden, out_ch, K=K)   # (N, 64) → (N, 64): final topology embedding
        self.bn1   = nn.BatchNorm1d(hidden)           # normalizes 64 features across N nodes → (N, 64)
        self.bn2   = nn.BatchNorm1d(hidden)           # normalizes 64 features across N nodes → (N, 64)

    def forward(self, x, edge_index):
        """
        x:          (N, 42)  node feature matrix
        edge_index: (2, E)   COO-format edge list
        Returns:    (N, 64)  enriched node embeddings
        """
        x = F.relu(self.bn1(self.conv1(x, edge_index)))   # (N,42)→conv1→(N,64)→bn1→relu→(N,64)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))   # (N,64)→conv2→(N,64)→bn2→relu→(N,64)
        x = self.conv3(x, edge_index)                     # (N,64)→conv3→(N,64) — no activation before scatter
        return x                                           # (N, 64)


# ── Scatter Node Features → Pixel Grid ────────────────────────────────────────

def scatter_to_pixels(node_pos, node_emb, H, W, device, sigma=2.0):
    """
    Scatter node embeddings (on 1-pixel skeleton) back to full H×W pixel grid.
    Gaussian spread fills vessel width so every pixel inside a vessel gets graph features.

    node_pos: (N, 2)  (x, y) pixel coordinates of skeleton nodes
    node_emb: (N, D)  node embeddings from ChebConv
    Returns:  (1, D, H, W)  scattered + blurred feature map
    """
    D      = node_emb.shape[1]                           # scalar int — embedding dim, e.g. 64
    F_scat = torch.zeros(1, D, H, W, device=device)     # (1, D, H, W) — blank canvas, all zeros

    xs = node_pos[:, 0].long().clamp(0, W-1)   # (N,) int64 — x (col) coords clamped to [0, W-1]
    ys = node_pos[:, 1].long().clamp(0, H-1)   # (N,) int64 — y (row) coords clamped to [0, H-1]

    # Place each node's D-dim embedding at its pixel position
    F_scat[0, :, ys, xs] = node_emb.T          # node_emb.T: (D, N) → writes to (1, D, H, W)

    # Gaussian spread σ=2 — fills vessel width around 1-pixel skeleton nodes
    # Without this, only the exact skeleton pixels get graph features;
    # nearby vessel pixels would have zeros and see no topology information.
    kernel_size = int(6 * sigma + 1) | 1        # odd number; e.g. σ=2 → ks=13
    padding     = kernel_size // 2              # same-padding keeps H, W unchanged
    blur_kernel = _gaussian_kernel(kernel_size, sigma, D).to(device)   # (D, 1, ks, ks)
    F_scat      = F.conv2d(F_scat, blur_kernel, padding=padding, groups=D)  # depthwise blur per channel

    return F_scat                               # (1, D, H, W) float32


def _gaussian_kernel(ks, sigma, groups):
    """
    Separable 2D Gaussian kernel for depthwise convolution.
    groups=D means each of the D channels gets its own kernel (identical) — efficient.
    Returns (D, 1, ks, ks) float32.
    """
    coords = torch.arange(ks, dtype=torch.float32) - ks // 2   # (ks,) e.g. [-6,...,0,...,6]
    g      = torch.exp(-0.5 * (coords / sigma) ** 2)            # (ks,) 1D Gaussian values
    g      = g / g.sum()                                         # (ks,) normalized so values sum to 1
    k2d    = g.unsqueeze(0) * g.unsqueeze(1)                     # (ks, ks) outer product → separable 2D Gaussian
    return k2d.view(1, 1, ks, ks).repeat(groups, 1, 1, 1)       # (groups, 1, ks, ks)


# ── Joint Model ────────────────────────────────────────────────────────────────

class VesselSegNet(nn.Module):
    """
    Joint SAM2 encoder + CNN decoder + ChebConv graph decoder.

    Architecture:
        Raw tile (B,1,H,W)
            │
            ▼
        Frozen SAM2.1 encoder → 4-scale FPN [(B,256,H/4,W/4),...,(B,256,H/32,W/32)]
            │
            ├──────────────────────────────────────┐
            │                                      │
        CNN upsampling                       Graph path
        feats[0] → ConvT×2                   coarse mask → skeletonize → sknw
        (B,256,H/4,W/4)→(B,32,H,W)          build 42-dim node features:
        F_pixel                                position, direction, diameter,
                                               curvature, degree, component_size,
                                               endpoint_distance, diameter_std, CNN(32)
                                             ChebConv(K=10) × 3 layers
                                             (N,42)→(N,64)
                                             scatter+Gaussian → (B,64,H,W)
                                             F_graph
            │                                      │
            └──────────────── cat ─────────────────┘
                        (B,96,H,W)
                            │
                        fuse_proj Conv1×1
                        (B,32,H,W)
                            │
                        head Conv1×1
                        (B,1,H,W) logits
    """
    def __init__(self):
        super().__init__()
        self.encoder   = SAM2Encoder()                                          # frozen Hiera ViT-L
        self.cnn_dec   = CNNDecoder(in_ch=256)                                  # (B,256,H/4,W/4)→(B,32,H,W)
        self.graph_net = VesselGraphNet(in_ch=NODE_FEAT_DIM, hidden=64,         # (N,42)→(N,64)
                                        out_ch=64, K=10)
        self.fuse_proj = nn.Conv2d(32 + 64, 32, kernel_size=1)                 # (B,96,H,W)→(B,32,H,W)
        self.head      = nn.Conv2d(32, 1, kernel_size=1)                        # (B,32,H,W)→(B,1,H,W)

    def forward(self, x, img_np=None, mask_coarse_np=None):
        """
        x:               (B, 1, H, W)  float32 tile ∈ [0,1]
        img_np:          (H, W)        float32 raw image for graph intensity features
        mask_coarse_np:  (H, W)        bool    coarse mask for skeleton and graph construction

        Returns: logits (B, 1, H, W)  — pass through sigmoid for probabilities
        """
        B, C, H, W = x.shape   # e.g. B=2, C=1, H=1300, W=1024

        # ── Encoder (frozen) — no gradients ─────────────────────────────────
        feats   = self.encoder(x)         # [(B,256,H/4,W/4), ..., (B,256,H/32,W/32)]

        # ── CNN path ────────────────────────────────────────────────────────
        F_pixel = self.cnn_dec(feats)     # (B, 32, H, W) — pixel-aligned local features

        # ── Graph path (runs only when coarse mask is provided) ─────────────
        # First training pass: no coarse mask → CNN-only (logits from first pass become the mask)
        # Subsequent passes: coarse mask from sigmoid(logits)>0.5 feeds the graph path
        if mask_coarse_np is not None and img_np is not None:
            try:
                # Sample CNN pixel features at skeleton node positions
                pf_np = F_pixel[0].permute(1, 2, 0).detach().cpu().numpy()
                # (B,32,H,W)[0] → (32,H,W) → permute → (H,W,32) float32 numpy

                graph_data = build_vessel_graph(mask_coarse_np, img_np, pf_np)
                # dict with node_pos(N,2), node_feats(N,42), edge_index(2,E), edge_feats(E,4)
                # or None if graph is degenerate

                if graph_data is not None:
                    nf  = torch.from_numpy(graph_data['node_feats']).to(x.device)  # (N, 42) float32
                    ei  = torch.from_numpy(graph_data['edge_index']).to(x.device)  # (2, E)  int64
                    np_ = torch.from_numpy(graph_data['node_pos']).to(x.device)    # (N, 2)  float32

                    node_emb = self.graph_net(nf, ei)                  # (N, 42) → (N, 64) topology embeddings
                    F_graph  = scatter_to_pixels(np_, node_emb, H, W, x.device)   # (1, 64, H, W)
                    F_graph  = F_graph.expand(B, -1, -1, -1)           # (B, 64, H, W) — broadcast to batch

                    # Fuse: concatenate CNN local features + graph topology features
                    F_fused  = torch.cat([F_pixel, F_graph], dim=1)    # (B, 32+64, H, W) = (B, 96, H, W)
                    F_fused  = self.fuse_proj(F_fused)                 # (B, 96, H, W) → (B, 32, H, W)
                else:
                    F_fused = F_pixel   # degenerate graph → CNN-only fallback
            except Exception:
                F_fused = F_pixel       # any graph error → silent CNN-only fallback (never crashes training)
        else:
            F_fused = F_pixel           # no mask provided → CNN-only forward pass

        logits = self.head(F_fused)     # (B, 32, H, W) → (B, 1, H, W) raw logits
        return logits                   # (B, 1, H, W) float32 — passed to VesselLoss or sigmoid
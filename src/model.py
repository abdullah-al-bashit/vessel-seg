import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import ChebConv          # Chebyshev spectral graph convolution
from skimage.morphology import skeletonize        # morphological thinning → 1px centerline
from torchvision.transforms.functional import to_pil_image  # tensor → PIL for SAM2 processor
import sknw                                       # skeleton → networkx graph (nodes = junctions/tips, edges = vessel segments)

HF_MODEL_ID = "facebook/sam2.1-hiera-large"      # SAM2.1 Hiera-Large — newest checkpoint on HuggingFace


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
                p.requires_grad = False        # no gradient → no weight update → no memory for optimizer state

        frozen    = sum(p.numel() for p in self.sam2.parameters() if not p.requires_grad)   # scalar int — total frozen param count, e.g. 308,000,000
        trainable = sum(p.numel() for p in self.sam2.parameters() if p.requires_grad)       # scalar int — total trainable param count, e.g. ~4,000,000
        print(f'SAM2 encoder  frozen:    {frozen:,}')      # e.g. "SAM2 encoder  frozen:    308,278,272"
        print(f'SAM2 decoder  trainable: {trainable:,}')   # e.g. "SAM2 decoder  trainable: 3,876,864"

    def forward(self, x):
        """
        x:  (B, 1, H, W)  float32 in [0, 1]   single-channel tile
        Returns list of 4 FPN feature maps, each 256 channels.
        """
        B = x.shape[0]   # scalar int — batch size, e.g. 2

        # Processor expects PIL images — convert 1-channel tensor to 3-channel PIL
        pil_images = [
            to_pil_image(x[i].repeat(3, 1, 1).clamp(0, 1).cpu())  # (1,H,W) → (3,H,W) → PIL RGB image
            for i in range(B)                                        # list of B PIL images
        ]
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"         # dict: {'pixel_values': (B, 3, 1024, 1024), ...} — SAM2 resizes to 1024×1024
        ).to(x.device)                  # move all tensors in dict to same device as x

        # Run vision encoder only — frozen, so no_grad saves memory and compute
        with torch.no_grad():
            vision_out = self.sam2.vision_encoder(**inputs)  # Hiera ViT forward pass → vision_out.feature_maps

        return vision_out.feature_maps   # list of 4 tensors: [(B,256,H/4,W/4), (B,256,H/8,W/8), (B,256,H/16,W/16), (B,256,H/32,W/32)]


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
            nn.ConvTranspose2d(in_ch, 64, kernel_size=2, stride=2),  # 2× upsample: (B, 256, H/4, W/4) → (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),                                        # normalizes across (B, H, W) per channel → stable training
            nn.ReLU(inplace=True),                                     # non-linearity; inplace saves one tensor allocation
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),     # 2× upsample: (B, 64, H/2, W/2) → (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        """feats: list of 4 FPN feature maps; use finest scale feats[0]"""
        x = self.up1(feats[0])   # (B, 256, H/4, W/4) → (B, 64, H/2, W/2)
        x = self.up2(x)          # (B, 64, H/2, W/2)  → (B, 32, H, W)
        return x                 # (B, 32, H, W)  — pixel-aligned feature map


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_vessel_graph(mask_np, img_np, pixel_feats_np):
    """
    Build vessel graph from binary mask.
    mask_np:        (H, W)     bool
    img_np:         (H, W)     float32  raw image [0,1]
    pixel_feats_np: (H, W, 32) float32  CNN features at each pixel

    Returns:
        node_pos:   (N, 2)  (x, y) positions
        node_feats: (N, 38) anisotropic features per node
        edge_index: (2, E)  graph connectivity
        edge_feats: (E, 2)  edge features
    """
    # Morphological thinning: reduces vessel mask to 1-pixel-wide centerline
    skel = skeletonize(mask_np)                          # (H, W) bool — 1px skeleton

    # Convert skeleton image to a networkx graph:
    # nodes = junction points and tip points; edges = vessel segments between them
    graph = sknw.build_sknw(skel.astype(np.uint16))     # networkx.Graph — N nodes, E edges

    nodes = list(graph.nodes())      # list of N node IDs (ints), e.g. [0, 1, 2, ..., N-1]
    if len(nodes) < 2:               # need at least 2 nodes to form an edge
        return None

    # ── Node features ──────────────────────────────────────────────────────
    node_pos   = []   # will become (N, 2)  float32 — (x, y) pixel coords per node
    node_feats = []   # will become (N, 38) float32 — feature vector per node

    for n in nodes:
        y, x = graph.nodes[n]['o']               # (2,) — skeleton node position as (row, col)
        y, x = int(y), int(x)                    # scalar ints
        y = np.clip(y, 0, mask_np.shape[0] - 1)  # scalar int — row clamped to [0, H-1]
        x = np.clip(x, 0, mask_np.shape[1] - 1)  # scalar int — col clamped to [0, W-1]

        # Direction θ — vessel orientation at this node
        # Average angle from all edges connected to this node
        angles = []                               # list of floats, one per connected edge
        for nb in graph.neighbors(n):
            pts = graph[n][nb]['pts']             # (L, 2) int array — pixel path along this edge
            if len(pts) >= 2:
                dy = pts[-1][0] - pts[0][0]       # scalar float — row displacement along path
                dx = pts[-1][1] - pts[0][1]       # scalar float — col displacement along path
                angles.append(np.arctan2(dy, dx)) # scalar float — angle in [-π, π]
        theta = np.mean(angles) if angles else 0.0  # scalar float — mean orientation across all edges; 0 if isolated node

        # Diameter d — local vessel width via Euclidean distance transform
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask_np)   # (H, W) float64 — each pixel's distance to nearest background pixel
        d = float(dist[y, x]) * 2                # scalar float — diameter = 2 × radius at this skeleton point

        # Curvature κ — bending rate from edge path (cross-product of endpoint tangent vectors)
        kappa = 0.0                               # scalar float — accumulated bending across all edges at this node
        for nb in graph.neighbors(n):
            pts = graph[n][nb]['pts']             # (L, 2) int array — pixel path
            if len(pts) >= 3:
                v1 = pts[1]  - pts[0]             # (2,) int — tangent vector at start of path
                v2 = pts[-1] - pts[-2]            # (2,) int — tangent vector at end of path
                cross = v1[0]*v2[1] - v1[1]*v2[0]   # scalar float — z-component of 2D cross product, proportional to sin(bend angle)
                norm  = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8  # scalar float — product of magnitudes + eps to avoid divide-by-zero
                kappa += abs(cross / norm)           # scalar float — ≈ |sin(bend angle)|; accumulate over all edges

        # CNN features sampled at node's pixel position
        cnn_feat = pixel_feats_np[y, x]          # (32,) float32 — decoder features at this skeleton point

        node_pos.append([x, y])                  # [col, row] = [x, y]; list grows to length N
        node_feats.append(np.concatenate([
            [x / mask_np.shape[1],               # scalar float — normalized x ∈ [0, 1]
             y / mask_np.shape[0],               # scalar float — normalized y ∈ [0, 1]
             np.cos(theta),                      # scalar float — direction cos ∈ [-1, 1]
             np.sin(theta),                      # scalar float — direction sin ∈ [-1, 1]
             d / max(mask_np.shape),             # scalar float — normalized diameter ∈ [0, 1]
             kappa],                             # scalar float — curvature (unbounded; larger = sharper bend)
            cnn_feat                             # (32,) float32 — learned pixel context
        ]))                                      # (38,) float32 — total: 6 anisotropic + 32 CNN = 38 features per node

    # ── Edge index + features ──────────────────────────────────────────────
    node_idx = {n: i for i, n in enumerate(nodes)}   # dict: networkx node ID → integer index 0..N-1
    edge_src, edge_dst, edge_feats = [], [], []        # will become (E,), (E,), (E, 2) after both directions added

    for u, v, data in graph.edges(data=True):
        if u not in node_idx or v not in node_idx:   # skip edges with missing endpoints
            continue
        i, j = node_idx[u], node_idx[v]              # scalar ints — integer indices for source and destination

        # Edge length — number of pixels along the path between nodes
        pts    = data.get('pts', np.array([]))        # (L, 2) int array — pixel path; empty array if direct neighbour
        length = float(len(pts)) if len(pts) > 0 else 1.0  # scalar float — path length in pixels

        # Gap intensity — mean image brightness along the vessel path
        if len(pts) > 0:
            ys = np.clip(pts[:, 0].astype(int), 0, img_np.shape[0]-1)  # (L,) int — row coords clamped to [0, H-1]
            xs = np.clip(pts[:, 1].astype(int), 0, img_np.shape[1]-1)  # (L,) int — col coords clamped to [0, W-1]
            gap_int = float(img_np[ys, xs].mean())   # scalar float — mean intensity ∈ [0, 1] along edge path
        else:
            gap_int = 0.0                             # scalar float — no path pixels to sample

        # Angle difference Δθ between connected nodes (unused in edge_feats but available for extension)
        fi = node_feats[i]                            # (38,) float32 — feature vector of source node
        fj = node_feats[j]                            # (38,) float32 — feature vector of destination node
        ti = np.arctan2(fi[3], fi[2])                 # scalar float — recover θ_i from sin (index 3) and cos (index 2)
        tj = np.arctan2(fj[3], fj[2])                 # scalar float — recover θ_j
        delta_theta = abs(ti - tj)                    # scalar float — angular difference ∈ [0, 2π], how much vessel bends at this edge

        edge_feats.append([length / max(mask_np.shape), gap_int])  # (2,) — [normalized length, brightness]; forward edge
        edge_src.append(i); edge_dst.append(j)        # forward direction: i → j
        edge_src.append(j); edge_dst.append(i)        # reverse direction: j → i  (undirected graph)
        edge_feats.append([length / max(mask_np.shape), gap_int])  # (2,) — same features for reverse edge

    if len(edge_src) == 0:    # graph has nodes but no edges — cannot do message passing
        return None

    return {
        'node_pos':   np.array(node_pos,   dtype=np.float32),          # (N, 2)
        'node_feats': np.array(node_feats, dtype=np.float32),          # (N, 38)
        'edge_index': np.array([edge_src, edge_dst], dtype=np.int64),  # (2, E)
        'edge_feats': np.array(edge_feats, dtype=np.float32),          # (E, 2)
    }


# ── ChebConv Graph Network ─────────────────────────────────────────────────────

class VesselGraphNet(nn.Module):
    """
    3-layer ChebConv with K=10 polynomial hops.
    Input:  node_feats (N, 38)  anisotropic features
    Output: node_emb   (N, 64)  enriched node embeddings
    """
    def __init__(self, in_ch=38, hidden=64, out_ch=64, K=10):
        super().__init__()
        self.conv1 = ChebConv(in_ch,   hidden, K=K)   # spectral conv: aggregates from K-hop neighbourhood; (N, 38) → (N, 64)
        self.conv2 = ChebConv(hidden,  hidden, K=K)   # (N, 64) → (N, 64)
        self.conv3 = ChebConv(hidden,  out_ch, K=K)   # (N, 64) → (N, 64)  final embedding
        self.bn1   = nn.BatchNorm1d(hidden)            # normalizes 64 features across N nodes; input/output (N, 64)
        self.bn2   = nn.BatchNorm1d(hidden)            # normalizes 64 features across N nodes; input/output (N, 64)

    def forward(self, x, edge_index):
        """
        x:          (N, 38)   node feature matrix
        edge_index: (2, E)    COO-format edge list
        Returns:    (N, 64)   enriched node embeddings
        """
        x = F.relu(self.bn1(self.conv1(x, edge_index)))   # (N, 38) → conv1 → (N, 64) → bn1 → (N, 64) → relu → (N, 64)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))   # (N, 64) → conv2 → (N, 64) → bn2 → (N, 64) → relu → (N, 64)
        x = self.conv3(x, edge_index)                     # (N, 64) → conv3 → (N, 64) — no activation before fuse
        return x                                           # (N, 64)


# ── Scatter Node Features → Pixel Grid ────────────────────────────────────────

def scatter_to_pixels(node_pos, node_emb, H, W, device, sigma=2.0):
    """
    Scatter node embeddings back to image pixel space.
    node_pos: (N, 2)  (x, y) pixel coordinates
    node_emb: (N, D)  node embeddings
    Returns:  (1, D, H, W)  scattered feature map
    """
    D       = node_emb.shape[1]                          # scalar int — embedding dimension, e.g. 64
    F_scat  = torch.zeros(1, D, H, W, device=device)    # (1, D, H, W) float32 — blank canvas; all zeros

    xs = node_pos[:, 0].long().clamp(0, W-1)   # (N,) int64 — x (col) coords of all N nodes, clamped to [0, W-1]
    ys = node_pos[:, 1].long().clamp(0, H-1)   # (N,) int64 — y (row) coords of all N nodes, clamped to [0, H-1]

    F_scat[0, :, ys, xs] = node_emb.T          # node_emb.T: (D, N); writes each node's D-dim embedding to its (row, col) pixel location → (1, D, H, W)

    # Gaussian spread — blurs point embeddings over vessel width so embeddings cover more than a single pixel
    kernel_size = int(6 * sigma + 1) | 1        # scalar int — e.g. sigma=2 → 6*2+1=13; bitwise OR 1 ensures odd
    padding     = kernel_size // 2              # scalar int — same-padding: (ks-1)/2 keeps H, W unchanged
    blur_kernel = _gaussian_kernel(kernel_size, sigma, D).to(device)   # (D, 1, ks, ks) float32 — depthwise kernel
    F_scat = F.conv2d(F_scat, blur_kernel, padding=padding, groups=D)  # (1, D, H, W) → depthwise blur → (1, D, H, W)

    return F_scat                               # (1, D, H, W) float32


def _gaussian_kernel(ks, sigma, groups):
    """Per-channel depthwise Gaussian kernel for scatter blurring."""
    coords = torch.arange(ks, dtype=torch.float32) - ks // 2   # (ks,) float32 — e.g. [-3,-2,-1,0,1,2,3] for ks=7
    g      = torch.exp(-0.5 * (coords / sigma) ** 2)            # (ks,) float32 — 1D Gaussian values
    g      = g / g.sum()                                         # (ks,) float32 — normalize so values sum to 1
    k2d    = g.unsqueeze(0) * g.unsqueeze(1)                     # (ks, ks) float32 — outer product → separable 2D Gaussian
    return k2d.view(1, 1, ks, ks).repeat(groups, 1, 1, 1)       # (groups, 1, ks, ks) float32 — one kernel per channel for depthwise conv


# ── Joint Model ────────────────────────────────────────────────────────────────

class VesselSegNet(nn.Module):
    """
    Joint SAM2 encoder + CNN decoder + ChebConv graph decoder.

    Training mode:  full forward pass with graph path if graph available
    Inference mode: same

    Architecture:
        Image → frozen Hiera encoder → 4-scale FPN features
                                            │
                    ┌───────────────────────┤
                    │                       │
               CNN upsample           Graph construct
               F_pixel (H×W×32)       → ChebConv(K=10)×3
                    │                  → F_graph (N×64)
                    │                       │
                    └──────── fuse ─────────┘
                    F_fused = F_pixel + scatter(F_graph)
                                    │
                              Conv 1×1 → logits → sigmoid → mask
    """
    def __init__(self):
        super().__init__()
        self.encoder   = SAM2Encoder()                            # frozen Hiera ViT-L; auto-downloads from HuggingFace
        self.cnn_dec   = CNNDecoder(in_ch=256)                    # 2× ConvTranspose upsample path; (B,256,H/4,W/4)→(B,32,H,W)
        self.graph_net = VesselGraphNet(in_ch=38, hidden=64, out_ch=64, K=10)  # 3-layer ChebConv; (N,38)→(N,64)
        self.fuse_proj = nn.Conv2d(32 + 64, 32, kernel_size=1)   # 1×1 conv: (B, 96, H, W) → (B, 32, H, W)
        self.head      = nn.Conv2d(32, 1, kernel_size=1)          # 1×1 conv: (B, 32, H, W) → (B, 1, H, W) logit map

    def forward(self, x, img_np=None, mask_coarse_np=None):
        """
        x:               (B, 1, H, W)  float32 tile
        img_np:          (H, W)        float32  original image (for graph feat)
        mask_coarse_np:  (H, W)        bool     coarse mask for skeleton

        Returns: logits (B, 1, H, W)
        """
        B, C, H, W = x.shape   # B = batch size (e.g. 2), C = 1 (greyscale), H = 1300, W = 1024

        # ── Encoder (frozen) ────────────────────────────────────────────────
        feats = self.encoder(x)          # list of 4 FPN tensors: [(B,256,H/4,W/4), (B,256,H/8,W/8), (B,256,H/16,W/16), (B,256,H/32,W/32)]

        # ── CNN upsampling path ─────────────────────────────────────────────
        F_pixel = self.cnn_dec(feats)    # (B, 32, H, W) — pixel-aligned feature map upsampled from feats[0]

        # ── Graph path (if coarse mask available) ───────────────────────────
        if mask_coarse_np is not None and img_np is not None:
            try:
                # Extract CNN pixel features at native (H, W) resolution for graph node sampling
                pf_np = F_pixel[0].permute(1, 2, 0).detach().cpu().numpy()  # (32, H, W) → (H, W, 32) float32 numpy

                graph_data = build_vessel_graph(mask_coarse_np, img_np, pf_np)  # dict or None

                if graph_data is not None:
                    nf  = torch.from_numpy(graph_data['node_feats']).to(x.device)   # (N, 38) float32 — node features
                    ei  = torch.from_numpy(graph_data['edge_index']).to(x.device)   # (2, E)  int64  — edge list
                    np_ = torch.from_numpy(graph_data['node_pos']).to(x.device)     # (N, 2)  float32 — node pixel positions

                    node_emb = self.graph_net(nf, ei)                # (N, 38) → ChebConv × 3 → (N, 64) float32

                    F_graph  = scatter_to_pixels(np_, node_emb, H, W, x.device)  # (1, 64, H, W) — point embeddings spread to pixel grid
                    F_graph  = F_graph.expand(B, -1, -1, -1)         # (1, 64, H, W) → (B, 64, H, W) — same graph features broadcast to all items in batch

                    # Fuse CNN pixel features and graph topology features channel-wise
                    F_fused  = torch.cat([F_pixel, F_graph], dim=1)  # (B, 32, H, W) + (B, 64, H, W) → (B, 96, H, W)
                    F_fused  = self.fuse_proj(F_fused)               # (B, 96, H, W) → (B, 32, H, W) via 1×1 conv
                else:
                    F_fused = F_pixel    # (B, 32, H, W) — no valid graph → skip graph branch
            except Exception:
                F_fused = F_pixel        # (B, 32, H, W) — graph build failed (e.g. empty mask) → fall back to CNN

        else:
            F_fused = F_pixel            # (B, 32, H, W) — graph inputs not provided → CNN-only forward pass

        logits = self.head(F_fused)      # (B, 32, H, W) → (B, 1, H, W) — raw logits (before sigmoid)
        return logits                    # (B, 1, H, W) float32 — passed to VesselLoss or sigmoid for inference

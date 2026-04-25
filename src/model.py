import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import ChebConv
from skimage.morphology import skeletonize
from torchvision.transforms.functional import to_pil_image
import sknw

HF_MODEL_ID = "facebook/sam2.1-hiera-large"     # SAM2.1 — newest version


# ── SAM2 Wrapper — HuggingFace ─────────────────────────────────────────────────

class SAM2Encoder(nn.Module):
    """
    Frozen Hiera ViT-L encoder via HuggingFace transformers.
    Auto-downloads weights from facebook/sam2.1-hiera-large on first run.
    No manual checkpoint file needed.
    Outputs 4-scale FPN feature maps at 256 channels each.
    """
    def __init__(self):
        super().__init__()
        from transformers import Sam2Model, Sam2Processor

        self.processor = Sam2Processor.from_pretrained(HF_MODEL_ID)
        self.sam2      = Sam2Model.from_pretrained(HF_MODEL_ID)

        # Freeze entire vision encoder — Hiera ViT weights never updated
        for name, p in self.sam2.named_parameters():
            if 'vision_encoder' in name:
                p.requires_grad = False

        frozen    = sum(p.numel() for p in self.sam2.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.sam2.parameters() if p.requires_grad)
        print(f'SAM2 encoder  frozen:    {frozen:,}')
        print(f'SAM2 decoder  trainable: {trainable:,}')

    def forward(self, x):
        """
        x:  (B, 1, H, W)  float32 in [0, 1]   single-channel tile
        Returns list of 4 FPN feature maps, each 256 channels.
        """
        B = x.shape[0]

        # Processor expects PIL images — convert 1ch tensor → 3ch PIL
        pil_images = [
            to_pil_image(x[i].repeat(3, 1, 1).clamp(0, 1).cpu())
            for i in range(B)
        ]
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        ).to(x.device)

        # Run vision encoder only (frozen)
        with torch.no_grad():
            vision_out = self.sam2.vision_encoder(**inputs)

        return vision_out.feature_maps               # list of 4 feature maps


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
            nn.ConvTranspose2d(in_ch, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        """feats: list of 4 feature maps, use finest scale feats[0]"""
        x = self.up1(feats[0])
        x = self.up2(x)
        return x                                 # (B, 32, H, W)


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_vessel_graph(mask_np, img_np, pixel_feats_np):
    """
    Build vessel graph from binary mask.
    mask_np:        (H, W)  bool
    img_np:         (H, W)  float32  raw image [0,1]
    pixel_feats_np: (H, W, 32)  CNN features at each pixel

    Returns:
        node_pos:   (N, 2)  (x, y) positions
        node_feats: (N, F)  anisotropic features per node
        edge_index: (2, E)  graph connectivity
        edge_feats: (E, 2)  edge features
    """
    # Skeletonize the binary mask → 1px centerline
    skel = skeletonize(mask_np)

    # Extract graph from skeleton using sknw
    graph = sknw.build_sknw(skel.astype(np.uint16))

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return None

    # ── Node features ──────────────────────────────────────────────────────
    node_pos   = []
    node_feats = []

    for n in nodes:
        y, x = graph.nodes[n]['o']               # skeleton node position (row, col)
        y, x = int(y), int(x)
        y = np.clip(y, 0, mask_np.shape[0] - 1)
        x = np.clip(x, 0, mask_np.shape[1] - 1)

        # Direction θ — vessel orientation at this node
        # Computed from edges connected to this node
        angles = []
        for nb in graph.neighbors(n):
            pts = graph[n][nb]['pts']            # pixel path between nodes
            if len(pts) >= 2:
                dy = pts[-1][0] - pts[0][0]
                dx = pts[-1][1] - pts[0][1]
                angles.append(np.arctan2(dy, dx))
        theta = np.mean(angles) if angles else 0.0

        # Diameter d — local vessel width via distance transform
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask_np)
        d = float(dist[y, x]) * 2               # diameter = 2 × radius

        # Curvature κ — bending rate from edge path
        kappa = 0.0
        for nb in graph.neighbors(n):
            pts = graph[n][nb]['pts']
            if len(pts) >= 3:
                v1 = pts[1] - pts[0]
                v2 = pts[-1] - pts[-2]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                norm  = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
                kappa += abs(cross / norm)

        # CNN features sampled at node position
        cnn_feat = pixel_feats_np[y, x]         # (32,)

        node_pos.append([x, y])
        node_feats.append(np.concatenate([
            [x / mask_np.shape[1],              # normalized x
             y / mask_np.shape[0],              # normalized y
             np.cos(theta),                     # direction cos
             np.sin(theta),                     # direction sin (anisotropic)
             d / max(mask_np.shape),            # normalized diameter
             kappa],                            # curvature
            cnn_feat                            # CNN features (32,)
        ]))                                     # total: 38 features

    # ── Edge index + features ──────────────────────────────────────────────
    node_idx = {n: i for i, n in enumerate(nodes)}
    edge_src, edge_dst, edge_feats = [], [], []

    for u, v, data in graph.edges(data=True):
        if u not in node_idx or v not in node_idx:
            continue
        i, j = node_idx[u], node_idx[v]

        # Edge length (pixel distance along path)
        pts    = data.get('pts', np.array([]))
        length = float(len(pts)) if len(pts) > 0 else 1.0

        # Gap intensity — mean intensity along the path
        if len(pts) > 0:
            ys = np.clip(pts[:, 0].astype(int), 0, img_np.shape[0]-1)
            xs = np.clip(pts[:, 1].astype(int), 0, img_np.shape[1]-1)
            gap_int = float(img_np[ys, xs].mean())
        else:
            gap_int = 0.0

        # Angle difference Δθ between connected nodes (anisotropic key feature)
        fi = node_feats[i]
        fj = node_feats[j]
        ti = np.arctan2(fi[3], fi[2])
        tj = np.arctan2(fj[3], fj[2])
        delta_theta = abs(ti - tj)

        edge_feats.append([length / max(mask_np.shape), gap_int])
        edge_src.append(i); edge_dst.append(j)
        edge_src.append(j); edge_dst.append(i)   # undirected
        edge_feats.append([length / max(mask_np.shape), gap_int])

    if len(edge_src) == 0:
        return None

    return {
        'node_pos':   np.array(node_pos,   dtype=np.float32),
        'node_feats': np.array(node_feats, dtype=np.float32),
        'edge_index': np.array([edge_src, edge_dst], dtype=np.int64),
        'edge_feats': np.array(edge_feats, dtype=np.float32),
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
        self.conv1 = ChebConv(in_ch,   hidden, K=K)
        self.conv2 = ChebConv(hidden,  hidden, K=K)
        self.conv3 = ChebConv(hidden,  out_ch, K=K)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)

    def forward(self, x, edge_index):
        """
        x:          (N, 38)
        edge_index: (2, E)
        Returns:    (N, 64)
        """
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.conv3(x, edge_index)
        return x


# ── Scatter Node Features → Pixel Grid ────────────────────────────────────────

def scatter_to_pixels(node_pos, node_emb, H, W, device, sigma=2.0):
    """
    Scatter node embeddings back to image pixel space.
    node_pos: (N, 2)  (x, y) pixel coordinates
    node_emb: (N, D)  node embeddings
    Returns:  (1, D, H, W)  scattered feature map
    """
    D       = node_emb.shape[1]
    F_scat  = torch.zeros(1, D, H, W, device=device)

    xs = node_pos[:, 0].long().clamp(0, W-1)
    ys = node_pos[:, 1].long().clamp(0, H-1)

    F_scat[0, :, ys, xs] = node_emb.T              # scatter

    # Gaussian spread — fills vessel width around skeleton nodes
    kernel_size = int(6 * sigma + 1) | 1            # odd
    padding     = kernel_size // 2
    blur_kernel = _gaussian_kernel(kernel_size, sigma, D).to(device)
    F_scat = F.conv2d(F_scat, blur_kernel, padding=padding, groups=D)

    return F_scat                                   # (1, D, H, W)


def _gaussian_kernel(ks, sigma, groups):
    """Per-channel depthwise Gaussian kernel for scatter blurring."""
    coords = torch.arange(ks, dtype=torch.float32) - ks // 2
    g      = torch.exp(-0.5 * (coords / sigma) ** 2)
    g      = g / g.sum()
    k2d    = g.unsqueeze(0) * g.unsqueeze(1)        # (ks, ks)
    return k2d.view(1, 1, ks, ks).repeat(groups, 1, 1, 1)  # (G, 1, ks, ks)


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
        self.encoder   = SAM2Encoder()           # auto-downloads from HuggingFace
        self.cnn_dec   = CNNDecoder(in_ch=256)
        self.graph_net = VesselGraphNet(in_ch=38, hidden=64, out_ch=64, K=10)
        self.fuse_proj = nn.Conv2d(32 + 64, 32, kernel_size=1)  # project fused
        self.head      = nn.Conv2d(32, 1, kernel_size=1)         # logit head

    def forward(self, x, img_np=None, mask_coarse_np=None):
        """
        x:               (B, 1, H, W)  float32 tile
        img_np:          (H, W)        float32  original image (for graph feat)
        mask_coarse_np:  (H, W)        bool     coarse mask for skeleton

        Returns: logits (B, 1, H, W)
        """
        B, C, H, W = x.shape

        # ── Encoder (frozen) ────────────────────────────────────────────────
        feats = self.encoder(x)                  # list of 4 feature maps

        # ── CNN upsampling path ─────────────────────────────────────────────
        F_pixel = self.cnn_dec(feats)            # (B, 32, H, W)

        # ── Graph path (if coarse mask available) ───────────────────────────
        if mask_coarse_np is not None and img_np is not None:
            try:
                # Extract pixel features at native resolution
                pf_np = F_pixel[0].permute(1, 2, 0).detach().cpu().numpy()  # (H,W,32)

                graph_data = build_vessel_graph(mask_coarse_np, img_np, pf_np)

                if graph_data is not None:
                    nf  = torch.from_numpy(graph_data['node_feats']).to(x.device)
                    ei  = torch.from_numpy(graph_data['edge_index']).to(x.device)
                    np_ = torch.from_numpy(graph_data['node_pos']).to(x.device)

                    node_emb = self.graph_net(nf, ei)   # (N, 64)

                    F_graph  = scatter_to_pixels(np_, node_emb, H, W, x.device)
                    F_graph  = F_graph.expand(B, -1, -1, -1)

                    # Fuse CNN + Graph features
                    F_fused  = torch.cat([F_pixel, F_graph], dim=1)  # (B, 96, H, W)
                    F_fused  = self.fuse_proj(F_fused)               # (B, 32, H, W)
                else:
                    F_fused = F_pixel
            except Exception:
                F_fused = F_pixel
        else:
            F_fused = F_pixel

        logits = self.head(F_fused)              # (B, 1, H, W)
        return logits

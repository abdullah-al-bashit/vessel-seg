import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx                                    # connected-component analysis for component_size feature
from torch_geometric.nn import ChebConv                 # Chebyshev spectral graph convolution
from skimage.morphology import skeletonize               # morphological thinning → 1px centerline
from torchvision.transforms.functional import to_pil_image   # tensor → PIL for SAM2 processor
import skan                                              # skeleton → graph; replaces sknw which had unfixed C-level heap-corruption bug
                                                          # (sknw issue #1: out-of-bounds writes in fill/trace buf on 1024+px inputs)
from scipy.ndimage import distance_transform_edt         # vessel radius map; one module-level import avoids re-importing on every forward pass

HF_MODEL_ID = "facebook/sam2.1-hiera-tiny"  # default; overridden per-experiment via config sam2_model
                                            # options: sam2.1-hiera-{tiny,small,base-plus,large}

# ── Feature dimensions ─────────────────────────────────────────────────────────
# Node features (39 total):
#   [0]   cos(θ)              vessel orientation cos ∈ [-1,1]
#   [1]   sin(θ)              vessel orientation sin ∈ [-1,1]  (anisotropic pair)
#   [2]   diameter            local vessel width ∈ [0,1]        fixes broken thin vessels
#   [3]   curvature κ         bending rate per edge ∈ [0,1]     fixes bifurcation errors
#   [4]   degree              number of branches at node        detects endpoints/bifurcations/false blobs
#   [5]   component_size      normalized size of vessel network isolates false positive blobs
#   [6]   diameter_consistency std of diameter along all edges  flags sudden width changes
#   [7-38] CNN features (32)  local appearance from decoder     pixel-level evidence
#
# ── Worked example: 3-way junction node on a 1024×1024 tile ──────────────────
#   Vessel is ~20 px wide; three edges point right, up, and down.
#
#   idx  feature            value    derivation
#   [0]  cos_theta         +0.330    mean(cos) of 3 edges: right(+1)+up(0)+dn(0) / 3
#   [1]  sin_theta          0.000    mean(sin) of 3 edges: 0 + (−1) + (+1) / 3 = 0
#   [2]  d_norm             0.020    dist[y,x]=10 → 10×2/1024
#   [3]  kappa              0.570    see κ example above (sum=1.71, degree=3 → 1.71/3)
#   [4]  deg_norm           0.750    degree=3 → 3/4
#   [5]  comp_norm          1.000    node belongs to the largest component
#   [6]  diam_std           0.050    CoV of radii along all edges
#   [7-38] 32 CNN floats            pixel_feats_np[y, x]  (local appearance)
#
# Edge features (4 total):
#   [0]   length_norm         normalized path length            vessel segment size
#   [1]   gap_intensity_mean  mean image brightness along path  detects real vs phantom vessels
#   [2]   gap_intensity_std   std of brightness along path      consistent signal = real vessel
#   [3]   delta_theta         angle difference between nodes    detects false branches at bifurcations
NODE_FEAT_DIM = 39   # 7 geometric/topological + 32 CNN
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
    def __init__(self, hf_model_id=HF_MODEL_ID):
        super().__init__()
        from transformers import Sam2Model, Sam2Processor  # lazy import — avoids load at module level

        self.processor = Sam2Processor.from_pretrained(hf_model_id)  # handles image pre-processing (resize, normalize, pad)
        self.sam2      = Sam2Model.from_pretrained(hf_model_id)       # full SAM2 model (encoder + prompt + decoder)

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

        # Processor expects PIL images — convert to 3-channel PIL.
        # x may be (B,1,H,W) grayscale or (B,3,H,W) with sharpness as 3rd channel.
        pil_images = [
            to_pil_image(
                (x[i] if x.shape[1] == 3 else x[i].repeat(3, 1, 1)).clamp(0, 1).cpu()
            )
            for i in range(B)
        ]
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"          # dict: {'pixel_values': (B,3,1024,1024), ...}
        ).to(x.device)

        with torch.no_grad():            # frozen — no gradients, saves memory and compute
            vision_out = self.sam2.vision_encoder(**inputs)

        # feature_maps was renamed to fpn_hidden_states in transformers ≥4.47 (same structure in 4.x and 5.x).
        # SAM2 processor rescales every input to 1024×1024, so for any tile size:
        #   fpn_hidden_states[0]: (B, 256, 256, 256)  — stride 4  (1024/4),  finest scale
        #   fpn_hidden_states[1]: (B, 256, 128, 128)  — stride 8  (1024/8)
        #   fpn_hidden_states[2]: (B, 256,  64,  64)  — stride 16 (1024/16), coarsest scale
        # CNNDecoder only uses index [0] (finest), the coarser scales are available for future use.
        return vision_out.fpn_hidden_states


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
        """feats: tuple of 3 FPN feature maps from SAM2 (always 1024×1024 after processor rescale); use finest scale feats[0]"""
        x = self.up1(feats[0])   # (B, 256, 256, 256) → (B, 64, 512, 512)  [SAM2 stride-4 feature, 4× upsample]
        x = self.up2(x)          # (B, 64, 512, 512)  → (B, 32, 1024, 1024)
        return x                 # (B, 32, 1024, 1024) — pixel-aligned at SAM2 resolution


# ── Skeleton → networkx graph (skan-based replacement for sknw) ───────────────

def _skel_to_graph(skel):
    """
    Convert a binary skeleton image into a networkx graph in the SAME shape as
    sknw.build_sknw used to produce, but using skan's stable Cython backend
    (no segfault on 1024+px images — sknw issue #1 is unfixed for years).

    skel: (H, W) bool — 1-pixel skeleton from skimage.morphology.skeletonize
    Returns: networkx.Graph or None (when skeleton has < 2 pixels / no edges)

    Output graph contract (matches what the rest of build_vessel_graph expects):
      graph.nodes[n]['o']   = (row, col) int pixel position of node n
      graph[u][v]['pts']    = (L, 2) int64 array of INTERIOR path pixels
                              (the endpoints u, v themselves are NOT in pts —
                               same convention as sknw)
      graph.nodes()         = integer IDs (skan's pixel-into-coordinates index)
      graph.degree(n)       = number of edges at node n
    """
    skel_bool = np.ascontiguousarray(skel.astype(bool))
    if skel_bool.sum() < 2:
        return None                                           # nothing to graph

    skel_obj = skan.Skeleton(skel_bool)
    if skel_obj.n_paths == 0:
        return None

    # skan.summarize returns one row per branch (= one edge in the graph), with
    # node-id-{src,dst} pointing into skel_obj.coordinates. separator='_' makes
    # the column names stable across skan versions (e.g. 'node_id_src').
    summary = skan.summarize(skel_obj, separator='_')
    if len(summary) == 0:
        return None

    coords = skel_obj.coordinates                             # (N_skel, 2) int — all skeleton pixel coords (row, col)

    G = nx.Graph()
    for i in range(len(summary)):
        src = int(summary['node_id_src'].iloc[i])
        dst = int(summary['node_id_dst'].iloc[i])

        # Add nodes (idempotent in networkx). 'o' attribute mirrors sknw's
        # node-position field used by the feature extraction below.
        G.add_node(src, o=tuple(int(c) for c in coords[src]))
        G.add_node(dst, o=tuple(int(c) for c in coords[dst]))

        # path_coordinates returns the full pixel chain INCLUDING both endpoint
        # nodes. Strip the endpoints so 'pts' contains only interior pixels —
        # that matches sknw's convention. int64 to dodge any int16 overflow.
        path = skel_obj.path_coordinates(i).astype(np.int64)
        interior = path[1:-1] if len(path) > 2 else path

        G.add_edge(src, dst, pts=interior)

    return G


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

        node_feats: (N, NODE_FEAT_DIM) 41-dim feature vector per node
                                       e.g. N=3 → shape (3, 41); row 0 = features of node 0

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
    # Downsample 4× before skeletonizing: ~16× fewer skeleton pixels → ~16× faster
    # skeletonize + skan, which is the dominant cost per batch. Vessel topology
    # (junctions, tips, branches) is preserved at quarter resolution. Node positions
    # are scaled ×4 back to full-res before returning, so they stay pixel-aligned
    # with F_pixel and img_np (±3 px error max — acceptable given vessel widths of 10–50 px).
    # dist is still computed on the full-res mask so vessel-radius values are accurate.
    mask_small = mask_np[::4, ::4]                               # (H/4, W/4) bool — 4× downsampled
    skel  = skeletonize(np.ascontiguousarray(mask_small))
    # (H/2, W/2) bool — True only on the 1px centerline of the downsampled mask

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
    # Use the skan-based wrapper instead of sknw.build_sknw — same networkx
    # output contract, but built on skan's stable Cython backend rather than
    # sknw's numba-jitted fill/trace which segfault on 1024+px images.
    graph = _skel_to_graph(skel)
    if graph is None or len(graph.nodes()) < 2:
        return None

    # ── Prune spurious stub branches ───────────────────────────────────────
    # A noisy coarse mask generates many short leaf branches at junctions —
    # 1-pixel protrusions that skan traces as degree-1 nodes connected by a
    # short edge. These add nodes without topological meaning and dominate
    # the graph size on early-training coarse masks.
    #
    # How edge length is measured:
    #   skan stores only the INTERIOR pixels of each edge in pts (endpoints excluded).
    #   e.g. a path  L(row=0,col=7) → ●(row=1,col=7) → J(row=2,col=7)
    #        has total 3 pixels, but pts = [[1,7]]  shape (1, 2)  → len(pts)=1
    #   We use len(pts) as the edge-length proxy (number of interior pixels).
    #
    # Example — main vessel with two stubs hanging off J (all at col=7):
    #
    #   col →  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
    #   row 0  .    .    .    .    .    .    .    L    .    .    .    .    .    .    .
    #   row 1  .    .    .    .    .    .    .    ●    .    .    .    .    .    .    .   ← 1 interior pixel
    #   row 2  .    n────p────p────p────p────p────J────p────p────p────p────p────p────n
    #   row 3  .    .    .    .    .    .    .    ●    .    .    .    .    .    .    .   ← 1 interior pixel
    #   row 4  .    .    .    .    .    .    .    L    .    .    .    .    .    .    .
    #
    #   n  = real endpoint node  (degree=1, row=2)
    #   J  = junction node       (degree=3, row=2, col=7)
    #   L  = STUB leaf node      (degree=1)
    #   ●  = single interior pixel of the stub edge
    #   p  = interior pixel of a real vessel edge (not a node)
    #
    #   Top stub    L(row=0,col=7) ── J(row=2,col=7):   pts=[[1,7]]            len(pts)=1 < 5 → PRUNE
    #   Bottom stub L(row=4,col=7) ── J(row=2,col=7):   pts=[[3,7]]            len(pts)=1 < 5 → PRUNE
    #   Left edge   n(row=2,col=1) ── J(row=2,col=7):   pts=[[2,2]..[2,6]]    len(pts)=5 ≥ 5 → keep
    #   Right edge  J(row=2,col=7) ── n(row=2,col=14):  pts=[[2,8]..[2,13]]   len(pts)=6 ≥ 5 → keep
    #
    # Why iterate? Removing a stub can expose a NEW leaf from its former neighbour.
    # Example — chain of two stubs both at col=7:
    #
    #   col →  0    1    2    3    4    5    6    7    8    9   10   11
    #   row 0  .    .    .    .    .    .    .    L₂   .    .    .    .
    #   row 1  .    .    .    .    .    .    .    ●    .    .    .    .
    #   row 2  .    .    .    .    .    .    .    L₁   .    .    .    .
    #   row 3  .    .    .    .    .    .    .    ●    .    .    .    .
    #   row 4  .    n────p────p────p────p────p────J────p────p────p────n
    #
    #   before:  L₂(deg=1)──L₁(deg=2)──J(deg=3)──n
    #   iter 1:  L₂ leaf, L₂──L₁ pts=[[1,7]] len=1 < 5 → remove L₂
    #            → L₁ now degree=1 (was 2: one edge to L₂ gone, one to J remains)
    #   iter 2:  L₁ leaf, L₁──J  pts=[[3,7]] len=1 < 5 → remove L₁
    #            → J.degree drops 3→2, not a leaf → stop
    #
    # MIN_STUB_PTS=5 interior px in 4× downsampled space ≈ 20 full-res pixels along the path.
    # Real vessel tips don't terminate after <20 px; anything shorter is boundary noise.
    MIN_STUB_PTS = 5
    # Repeatedly scan all nodes; remove any leaf whose connecting edge is shorter than
    # MIN_STUB_PTS interior pixels. Loop until a full pass finds nothing to prune —
    # needed because each removal can turn a non-leaf into a new leaf (chain effect above).
    pruned = True
    while pruned:
        pruned = False
        for n in list(graph.nodes()):       # list() snapshot — graph mutates inside loop; iterating live view raises RuntimeError
            if graph.degree(n) == 1:        # leaf: exactly one edge at n
                                            #   degree=2 → straight interior node (skip)
                                            #   degree≥3 → junction (skip)
                nb  = next(iter(graph.neighbors(n)))            # scalar int — the single neighbour of leaf n
                                                                #   e.g. n=L(row=0,col=7), nb=J(row=2,col=7)
                pts = graph[n][nb].get('pts', np.array([]))     # (len(pts), 2) int64 — interior pixels on n──nb
                                                                #   top stub: pts=[[1,7]] shape (1,2), len=1
                                                                #   right real edge: pts=[[2,8]..[2,13]] shape (6,2), len=6
                if len(pts) < MIN_STUB_PTS: # 1 < 5 → stub → prune
                    graph.remove_node(n)    # removes n AND its single edge n──nb
                    pruned = True           # nb may now be a new leaf → re-scan
    # returns networkx.Graph; node IDs are skeleton-pixel indices into
    # skan.Skeleton.coordinates (not necessarily 0,1,2,...)
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
    # Always on full-res mask so vessel-radius values are accurate (not halved by downsampling).
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
    #      2     F    F  [ T    T    T    T    T ]  F    F    ← vessel top edge
    #      3     F    F  [ T    T    T    T    T ]  F    F    ← vessel middle row (skeleton)
    #      4     F    F  [ T    T    T    T    T ]  F    F    ← vessel bottom edge
    #      5     F    F    F    F    F    F    F    F    F    ← all background
    #      6     F    F    F    F    F    F    F    F    F    ← all background
    #
    #   dist values for the same patch (Euclidean distance to nearest F):
    #
    #   row\col  0    1    2    3    4    5    6    7    8
    #      0     0    0    0    0    0    0    0    0    0
    #      1     0    0    0    0    0    0    0    0    0
    #      2     0    0  [ 1    1    1    1    1 ]  0    0    ← 1 step from background
    #      3     0    0  [ 1    2    2    2    1 ]  0    0    ← center row: dist=2 (vessel radius=2)
    #      4     0    0  [ 1    1    1    1    1 ]  0    0    ← 1 step from background
    #      5     0    0    0    0    0    0    0    0    0
    #      6     0    0    0    0    0    0    0    0    0
    #
    #   skeleton runs along row=3 (the row of maximum dist values);
    #   skeleton pixels at (3,3), (3,4), (3,5) all have dist=2 = rows to nearest background
    #   diameter = 2 × dist = 4  (background-row gap: row 5 − row 1 = 4; actual vessel
    #   pixel count = 3 rows 2-4; ×2 mirrors the radius symmetrically to both sides)
    #
    #   corner pixels such as (2,2) get dist=1 because the nearest background pixel is
    #   directly above (1,2) or directly left (2,1), both 1 step away — Euclidean distance
    #   takes the shortest path in 2D, not just along rows or columns

    # ── Pre-compute connected components for component_size feature ────────
    # Each connected component = one isolated vessel subgraph in this tile.
    # Large → real vessel network; small / isolated → likely spurious blob.
    components   = list(nx.connected_components(graph))    # list[set[int]], length C ≤ N — e.g. [{0,1,2}, {3}]
    node_to_comp = {}                                       # dict[int,int], N entries — e.g. {0:0, 1:0, 2:0, 3:1}
    for ci, comp in enumerate(components):  # ci = component index 0,1,...,C-1; comp = set of node IDs in it
        for nid in comp:                     # flatten: stamp every node in this component with its index ci
            node_to_comp[nid] = ci           # e.g. nodes 0,1,2 → ci=0; node 3 → ci=1
    comp_sizes    = [len(c) for c in components]           # list[int], length C — e.g. [3, 1]
    max_comp_size = max(comp_sizes) if comp_sizes else 1   # int — e.g. 3; comp_norm = comp_sizes[ci] / 3 → [0,1]

    # ── Build node features ────────────────────────────────────────────────
    node_pos   = []    # list[list[float,float]]; one [x=col, y=row] per node → np.array → (N, 2) float32
    node_feats = []    # list[ndarray(39,)];     one 39-dim vec  per node → np.array → (N, 39) float32

    for n in nodes:
        y, x = graph.nodes[n]['o']                   # (2,) — skeleton node position as (row, col) in downsampled space
        # The graph was built on mask_small = mask_np[::4, ::4], so every coordinate
        # lives in [0, H/4) × [0, W/4). Multiplying by 4 maps it back onto the
        # full-res pixel grid — e.g. downsampled row=100 → full-res row=400.
        # This ensures dist[], img_np[], and pixel_feats_np[] are indexed at the correct
        # full-resolution pixel rather than a 4× smaller location.
        y = np.clip(int(y) * 4, 0, H - 1)            # full-res row ∈ [0, H-1]
        x = np.clip(int(x) * 4, 0, W - 1)            # full-res col ∈ [0, W-1]

        # ── Feature 1-2: vessel direction θ as (cos θ, sin θ) ───────────────
        # pts = ordered interior pixel path of edge n──nb; pts[0] sits next to n, pts[-1] next to nb.
        # The direction of that edge, seen from n, is the vector pts[0] → pts[-1].
        #
        #   col →  0    1    2    3    4    5    6
        #   row r  n   [p0]  p1   p2   p3  [p4]  nb        ← horizontal vessel
        #                ↑                   ↑
        #              pts[0]             pts[-1]
        #           pixel(r,1)         pixel(r,5)
        #
        #   each pts entry is [row, col]:
        #     pts[0]  = [r, 1]   →  pts[0][0]  = r  (row),  pts[0][1]  = 1  (col)
        #     pts[-1] = [r, 5]   →  pts[-1][0] = r  (row),  pts[-1][1] = 5  (col)
        #
        #   dy = pts[-1][0] − pts[0][0] = r − r = 0    (same row  → no vertical displacement)
        #   dx = pts[-1][1] − pts[0][1] = 5 − 1 = +4   (col 1→5  → 4 steps rightward)
        #   θ  = arctan2(dy=0, dx=+4) = 0.0   →  pointing right →
        #
        #   Diagonal vessel (45°, down-right), 3 interior pts:
        #
        #   col →  0    1    2    3    4
        #   row r  n   [p0]                    ← pts[0]  at pixel (r,   1)
        #   r+1          p1
        #   r+2               [p2]  nb         ← pts[-1] at pixel (r+2, 3);  nb at (r+3, 4)
        #
        #   dy = pts[-1][0] − pts[0][0] = (r+2) − r = +2  (moved 2 rows downward)
        #   dx = pts[-1][1] − pts[0][1] =   3   − 1 = +2  (moved 2 cols rightward)
        #   θ  = arctan2(dy=+2, dx=+2) = π/4  →  pointing down-right ↘
        #
        #   arctan2(dy, dx) — angle of displacement vector from positive x-axis →.
        #   Uses sign of BOTH dy and dx to pick the correct quadrant (all 8 directions):
        #
        #          dy=−4 dx=0   up ↑  θ=−π/2
        #          dy=−2 dx=+2  ↗    θ=−π/4
        #                          │
        #   dy=0 dx=−4  ← ─────────┼───────── → dy=0 dx=+4   θ=0  /  θ=±π
        #                          │
        #          dy=+2 dx=+2  ↘    θ=+π/4
        #          dy=+4 dx=0  down ↓  θ=+π/2
        #
        #   Unlike plain arctan(dy/dx): arctan(1)=π/4 for BOTH (dy=+2,dx=+2) ↘ and
        #   (dy=−2,dx=−2) ↖ since (+2)/(+2)=(−2)/(−2)=1.  arctan2 sees the signs
        #   separately: arctan2(+2,+2)=+π/4 (↘) vs arctan2(−2,−2)=−3π/4 (↖).
        #
        #   Angle → (cos θ, sin θ) table:
        #
        #     direction      dy    dx    θ (rad)   cos θ    sin θ
        #     →  right        0    +4     0.00     +1.00     0.00
        #     ↓  down        +4     0    +π/2       0.00    +1.00
        #     ←  left         0    −4    ±π        −1.00     0.00
        #     ↑  up          −4     0    −π/2       0.00    −1.00
        #     ↘  down-right  +2    +2    +π/4      +0.71    +0.71
        #
        #   Why (cos θ, sin θ) and not raw θ — concrete failure example:
        #
        #   Junction J with two edges both pointing roughly left (one slightly up, one slightly down):
        #
        #     nb_up ·····
        #               \    dy=−1, dx=−4  →  θ = arctan2(−1,−4) ≈ −2.90 rad
        #                J
        #               /    dy=+1, dx=−4  →  θ = arctan2(+1,−4) ≈ +2.90 rad
        #     nb_dn ·····
        #
        #   Raw θ average:   (−2.90 + 2.90) / 2 = 0.0 rad  →  "pointing right" →   WRONG ✗
        #     The two angles straddle the ±π boundary and cancel to zero.
        #
        #   (cos θ, sin θ) average:
        #     edge J──nb_up:  (cos(−2.90), sin(−2.90)) ≈ (−0.97, −0.24)
        #     edge J──nb_dn:  (cos(+2.90), sin(+2.90)) ≈ (−0.97, +0.24)
        #     mean:                                        (−0.97,  0.00)
        #     → arctan2(0.00, −0.97) ≈ ±π  →  "pointing left"  ←   CORRECT ✓
        #
        #   Storing as a 2-D unit vector (cos θ, sin θ) also lets ChebConv weight
        #   the horizontal (cos) and vertical (sin) components independently, giving
        #   it the power to learn orientation-sensitive vessel filters.
        # ── Single neighbor loop — computes direction, curvature, and diameter
        #    consistency in one pass instead of three, eliminating 2× redundant
        #    graph[n][nb] dict lookups and neighbour-list iterations per node.
        cos_vals = []              # accumulates cos(θ) per edge leaving n
        sin_vals = []              #   e.g. tip (deg=1) → len=1; junction (deg=3) → len=3
        kappa    = 0.0             # accumulated |sin(bend angle)| across all edges
        diameters_along_edges = [] # radius samples from every interior path pixel at this node
        for nb in graph.neighbors(n):               # nb = node ID of each direct neighbour
            pts = graph[n][nb]['pts']               # (L, 2) int — ordered [row,col] path of interior
                                                    #   pixels between n and nb; does NOT include n or nb
                                                    #   e.g. horizontal edge, L=5:
                                                    #   pts = [[r,1],[r,2],[r,3],[r,4],[r,5]]
            L = len(pts)
            # ── direction (cos/sin) ────────────────────────────────────────
            if L >= 2:              # need ≥2 distinct pixels to form a direction vector pts[-1]−pts[0];
                                    #   L=0 → index error;  L=1 → pts[-1] is pts[0] → dy=dx=0 → arctan2(0,0)=0 (undefined direction)
                dy    = float(pts[-1][0] - pts[0][0])  # scalar float — row displacement (+ = downward)
                dx    = float(pts[-1][1] - pts[0][1])  # scalar float — col displacement (+ = rightward)
                angle = np.arctan2(dy, dx)             # scalar float ∈ [−π, π]
                cos_vals.append(np.cos(angle))         # circular mean: accumulate unit-vector components
                sin_vals.append(np.sin(angle))         #   avoids ±π wrap-around of plain angle mean
            # ── curvature κ ───────────────────────────────────────────────
            # direction and curvature use relative pixel differences (pts[-1]-pts[0],
            # pts[1]-pts[0], etc.) which are scale-invariant — multiplying all coords
            # by 2 cancels in the ratio, so downsampled pts give identical results.
            if L >= 3:
                v1    = (pts[1]  - pts[0]).astype(np.int64)         # (2,) int64 — tangent at start
                v2    = (pts[-1] - pts[-2]).astype(np.int64)        # (2,) int64 — tangent at end
                cross = float(v1[0]*v2[1] - v1[1]*v2[0])            # scalar — z-component of 2D cross product
                norm  = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
                kappa += abs(cross / norm)             # accumulate |sin(angle)| per edge
            # ── diameter consistency ───────────────────────────────────────
            # pts are in downsampled space → multiply by 4 to index dist[] which
            # is full-res, so we sample the accurate vessel-radius value.
            if L > 0:
                ys_e = np.clip(pts[:, 0].astype(int) * 4, 0, H - 1)
                xs_e = np.clip(pts[:, 1].astype(int) * 4, 0, W - 1)
                diameters_along_edges.extend(dist[ys_e, xs_e].tolist())
        # cos_vals / sin_vals after loop examples:
        #   tip  (deg=1, down-right ↘):    cos_vals=[+0.71], sin_vals=[+0.71]
        #   junction (deg=3, right/up/dn): cos_vals=[+1.00, 0.00, 0.00], sin_vals=[0.00, −1.00, +1.00]
        cos_theta = float(np.mean(cos_vals)) if cos_vals else 1.0   # scalar float ∈ [−1,1]
        sin_theta = float(np.mean(sin_vals)) if sin_vals else 0.0   # scalar float ∈ [−1,1]
        #   tip  example:      cos_theta=+0.71, sin_theta=+0.71  (down-right ↘)
        #   junction example:  cos_theta=+0.33, sin_theta= 0.00  (right →; up/dn sin values cancel)
        #   isolated node:     cos_theta= 1.00, sin_theta= 0.00  (default; no edges → no direction)

        # ── Feature 5: diameter ────────────────────────────────────────────
        # dist[y,x] = radius (distance to nearest background pixel).
        # Multiply by 2 = diameter. Normalize by max image dimension.
        # Key failure fix: sudden diameter changes along a vessel → false positive or break.
        d_norm = float(dist[y, x]) * 2 / max(H, W)
        # scalar float ∈ [0,1] — normalized vessel diameter at this node's skeleton pixel
        #   dist[y,x] = radius = how many pixels the skeleton lies from the nearest background pixel
        #   "N-px-wide" = the vessel cross-section spans N True pixels in the mask
        #   skeleton sits at the center → dist ≈ N/2 (background is ~N/2 steps away)
        #   e.g. 3-px-wide vessel  (as in the dist example above): dist[y,x]=2.0 → d_norm = 2.0×2/1024 ≈ 0.004
        #        20-px-wide vessel on a 1024×1024 tile:            dist[y,x]=10.0 → d_norm = 10.0×2/1024 ≈ 0.020

        # ── Feature 6: curvature κ ─────────────────────────────────────────
        # κ = (sum of |sin(bend angle)| over all edges) / degree  ∈ [0, 1]
        #   each edge contributes |sin| ∈ [0,1]; dividing by degree normalizes to per-edge rate
        #   measures how sharply the vessel bends at this node, independent of branch count
        #   computed per edge as: |cross(v1,v2)| / (|v1|·|v2|) = |sin(θ)|
        #     v1 = tangent at START of edge: pts[1]−pts[0]   (first-step direction)
        #     v2 = tangent at END   of edge: pts[-1]−pts[-2] (last-step direction)
        #     θ  = bend angle between v1 and v2
        #
        #   Why sine: the 2D cross product gives |v1|·|v2|·|sin(θ)| directly,
        #   so dividing by |v1|·|v2| isolates |sin(θ)| without any arcsin call.
        #     sin(0°)  = 0  →  v1 ∥ v2  →  straight  →  κ += 0
        #     sin(90°) = 1  →  v1 ⊥ v2  →  right-angle bend  →  κ += 1  (max per edge)
        #   Result ∈ [0,1] per edge regardless of step size.
        #
        #   Straight edge, L=5, pts = [[r,c+1],[r,c+2],[r,c+3],[r,c+4],[r,c+5]]:
        #
        #   col →  c  c+1  c+2  c+3  c+4  c+5  c+6
        #   row r  n  [p0]  p1   p2   p3  [p4]  nb      all on same row
        #
        #      v1 = pts[1]−pts[0]   = [r,c+2]−[r,c+1] = [0,+1]  (rightward)
        #      v2 = pts[-1]−pts[-2] = [r,c+5]−[r,c+4] = [0,+1]  (rightward)
        #      cross = 0×1 − 1×0 = 0   →  κ += 0
        #
        #   90° bend, L=4, pts = [[r,c+1],[r,c+2],[r+1,c+2],[r+2,c+2]]:
        #
        #   col →  c  c+1  c+2  c+3
        #   row r  n  [p0]  p1             v1 = [r,c+2]−[r,c+1]     = [0,+1]  (rightward →)
        #   r+1          p2
        #   r+2         [p3]  nb           v2 = [r+2,c+2]−[r+1,c+2] = [+1,0]  (downward ↓)
        #
        #      cross = v1[0]×v2[1] − v1[1]×v2[0] = 0×0 − 1×1 = −1
        #      norm  = |[0,+1]| × |[+1,0]| = 1 × 1 = 1
        #      κ += |−1/1| = 1.0   (|sin(90°)| = 1)
        #
        #   Junction node (deg=3): accumulate over 3 edges, then divide by 3
        #     edge 1 (straight, v1=[0,+1], v2=[0,+1]):     κ += 0.00
        #     edge 2 (45° bend, v1=[0,+1], v2=[+1,+1]):    cross=−1, norm=√2  →  κ += 0.71
        #     edge 3 (90° bend, v1=[0,+1], v2=[+1, 0]):    cross=−1, norm=1   →  κ += 1.00
        #     sum=1.71 → kappa = 1.71/3 = 0.57   ← normalized; straight node≈0, tight elbow≈1
        #
        # High κ = sharp bend. Low κ = straight vessel.
        kappa = kappa / max(graph.degree(n), 1)      # mean |sin| per edge ∈ [0,1] — accumulated in the single neighbor loop above

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
        ci        = node_to_comp[n]                  # n = current sknw node ID; ci = index of the component n belongs to; e.g. n=3 → ci=1
        comp_norm = float(comp_sizes[ci]) / float(max_comp_size)  # comp_sizes[ci] = node count of ci's component; max_comp_size = largest count; e.g. 1/3 ≈ 0.33

        # ── Feature 9: diameter_consistency ───────────────────────────────
        # Standard deviation of vessel diameter sampled along all edges at this node.
        # Consistent diameter along a segment → real vessel (low std).
        # Sudden diameter change → false positive joining two different structures (high std).
        # diameters_along_edges was filled (with full-res ×4 coords) in the single neighbor loop above.
        if diameters_along_edges:
            r     = np.array(diameters_along_edges)
            diam_std = float(r.std() / (r.mean() + 1e-8))   # coefficient of variation: std / mean ∈ [0, ~1]
            # scale-free: measures relative width variability regardless of absolute vessel size
            #   consistent vessel (radii ≈ [1.0, 1.1, 0.9]):  std≈0.1, mean≈1.0 → CoV≈0.10  (low → real vessel)
            #   sudden change     (radii ≈ [1, 1, 5, 9, 10]): std≈4.0, mean≈5.2 → CoV≈0.77  (high → suspicious)
        else:
            diam_std = 0.0

        # ── Feature 10-41: CNN features ────────────────────────────────────
        # 32-dim appearance vector from the CNN decoder at this node's pixel position.
        # Provides local texture, intensity, and learned vessel context.
        cnn_feat = pixel_feats_np[y, x]             # (32,) float32 — sampled from (H, W, 32) feature map

        # ── Assemble 39-dim node feature vector ────────────────────────────
        node_pos.append([x, y])                     # list grows to (N, 2)
        node_feats.append(np.concatenate([
            [cos_theta,             # [0]  direction cos       ∈ [-1,1]
             sin_theta,             # [1]  direction sin       ∈ [-1,1]
             d_norm,                # [2]  diameter            ∈ [0,1]   fixes broken thin vessels
             kappa,                 # [3]  curvature           ∈ [0,1]  fixes bifurcation errors
             deg_norm,              # [4]  degree              ∈ [0,~1]  detects endpoints/blobs
             comp_norm,             # [5]  component_size      ∈ [0,1]   isolates false blobs
             diam_std],             # [6]  diameter_std        ∈ [0,1]   flags width changes
            cnn_feat                # [7-38] CNN features      (32,)     local appearance
        ]))                         # (39,) float32 — NODE_FEAT_DIM

    # ── Build edge features ────────────────────────────────────────────────
    node_idx            = {n: i for i, n in enumerate(nodes)}    # dict: networkx node ID → integer index 0..N-1
    edge_src, edge_dst  = [], []                                 # will become (E,) each
    edge_feats_list     = []                                     # will become (E, EDGE_FEAT_DIM)

    for u, v, data in graph.edges(data=True):
        # u, v: sknw node IDs of the two endpoints; data: edge attribute dict with key 'pts'
        if u not in node_idx or v not in node_idx:
            continue
        i, j = node_idx[u], node_idx[v]             # scalar ints — integer indices

        pts    = data.get('pts', np.array([]))       # (L, 2) int — pixel path in downsampled coords; empty if direct neighbour
        # ×4: pts are in downsampled space (same reason as node positions above).
        # Multiply pixel count by 4 to approximate the full-res path length.
        length = float(len(pts)) * 4 if len(pts) > 0 else 4.0   # scalar float — approx full-res path length in pixels

        # ── Edge feature 1: normalized length ─────────────────────────────
        length_norm = length / max(H, W)             # scalar float ∈ [0,1]

        # ── Edge feature 2-3: gap_intensity_mean, gap_intensity_std ────────────────────────────
        # Mean image brightness along the vessel path.
        # High mean → fluorescence signal present → real vessel.
        # Low mean  → dark region between two predicted segments → likely a false prediction.
        if len(pts) > 0:
            # pts are in downsampled coords → multiply by 4 to index img_np which is full-res.
            # Each downsampled pixel maps to the corresponding full-res pixel at 4× the position.
            ys_e = np.clip(pts[:, 0].astype(int) * 4, 0, H-1)   # (L,) int — full-res row coords
            xs_e = np.clip(pts[:, 1].astype(int) * 4, 0, W-1)   # (L,) int — full-res col coords
            intensities  = img_np[ys_e, xs_e]               # (L,) float32 — pixel intensities along path         img_np: (H, W) - float32 raw image for graph intensity features
            gap_mean     = float(intensities.mean())         # scalar float ∈ [0,1]
            # Standard deviation of brightness along the path.
            # Low std  → uniform signal → continuous vessel.
            # High std → patchy signal → broken or uncertain vessel segment.
            gap_std  = float(intensities.std())              # scalar float ∈ [0,1]
        else:
            gap_mean = 0.0                                   # no path pixels → unknown intensity
            gap_std  = 0.0

        # ── Edge feature 4: delta_theta ────────────────────────────────────
        # Angular difference between the mean orientations of the two endpoint nodes.
        # Each node's orientation = mean(cos,sin) over ALL its edges, not just this one.
        # Low Δθ  → both nodes face the same direction → straight consistent vessel.
        # High Δθ → nodes face different directions → false connection between two structures.
        #
        #   i, j: compact integer indices into node_feats (0..N-1), derived from node_idx[u] and node_idx[v]
        #         node_idx maps sknw node ID → row position in node_feats
        #         i = node_idx[u] → row of source node u's 39-dim feature vector in node_feats
        #         j = node_idx[v] → row of destination node v's 39-dim feature vector in node_feats
        #   Stored in node_feats (set in the node-feature loop above):
        #     node_feats[i][0] = cos θ_i,   node_feats[i][1] = sin θ_i
        #     node_feats[j][0] = cos θ_j,   node_feats[j][1] = sin θ_j
        #
        #   Recovery — invert (cos, sin) back to θ using arctan2:
        #     θ = arctan2(sin θ, cos θ)   ∈ [−π, π]
        #     arctan2(y, x) = angle of vector (x, y) from the positive x-axis
        #       e.g. arctan2(sin=0, cos=+1) = 0        (→ rightward)
        #            arctan2(sin=+1, cos=0) = π/2       (↓ downward)
        #            arctan2(sin=0,  cos=−1) = ±π       (← leftward)
        #
        #   θ_i = arctan2(node_feats[i][1], node_feats[i][0])  =  arctan2(sin θ_i, cos θ_i)
        #   θ_j = arctan2(node_feats[j][1], node_feats[j][0])  =  arctan2(sin θ_j, cos θ_j)
        #   Both are the mean orientation of that node across ALL its edges (not just this edge).
        #
        #   Case 1 — real straight vessel, tip i to tip j (Δθ = 0):
        #
        #   col →  0    1    2    3    4    5    6
        #   row r  i    p    p    p    p    p    j        pts ordered left→right
        #
        #     θ_i = arctan2(0, +4) = 0   (1 edge, pts[0]=col1→pts[-1]=col5, dx=+4)
        #     θ_j = arctan2(0, +4) = 0   (sknw stores pts in one fixed order; graph[j][i]['pts']
        #                                  returns the same array as graph[i][j]['pts'], so j also
        #                                  sees pts[0]=col1→pts[-1]=col5, dx=+4 → also →)
        #     Δθ = |0 − 0| = 0.0   → consistent direction → real vessel ✓
        #
        #   Case 2 — false connection between two perpendicular structures (Δθ = π/2):
        #
        #   col →  0    1    2    3
        #   row r  T    p    p   J_A          ← J_A: tip of horizontal vessel
        #                        |            ← false edge: directly adjacent, 0 interior pts
        #   r+1                 J_B          ← J_B: tip of vertical vessel
        #   r+2                  p
        #   r+3                  p
        #   r+4                  T
        #
        #     J_A: tip with 1 real (horizontal) edge; false edge is direct → 0 pts → skipped
        #       real edge pts: [[r,1],[r,2]] → dx = 2−1 = +1, dy = 0 → angle = 0
        #       cos_vals=[+1], sin_vals=[0]  →  θ_J_A = arctan2(0, +1) = 0   (→ rightward)
        #
        #     J_B: tip with 1 real (vertical) edge; false edge is direct → 0 pts → skipped
        #       real edge pts: [[r+2,3],[r+3,3]] → dy = +1, dx = 0 → angle = π/2
        #       cos_vals=[0], sin_vals=[+1]  →  θ_J_B = arctan2(+1, 0) = π/2  (↓ downward)
        #
        #     Δθ = |0 − π/2| = 1.57   → direction mismatch → false connection ✗
        #       Why: a real vessel segment joins two nodes that face the SAME direction (both
        #       aligned along the vessel axis). J_A faces horizontally; J_B faces vertically.
        #       They belong to two different structural elements — the edge between them is a
        #       phantom bridge predicted across a gap, not a real continuous vessel.
        fi          = node_feats[i]                          # (39,) float32 — source node features
        fj          = node_feats[j]                          # (39,) float32 — destination node features
        ti          = np.arctan2(fi[1], fi[0])               # scalar float — recover θ_i from (sin=index1, cos=index0)
        tj          = np.arctan2(fj[1], fj[0])               # scalar float — recover θ_j
        delta_theta = float(abs(ti - tj))
        # abs(): sign is discarded because the feature measures direction disagreement,
        # which is symmetric — whether i is +30° or −30° relative to j, the mismatch is 30°.
        # Range is [0, 2π] rather than [0, π] because plain subtraction does not wrap: two
        # "leftward" nodes at ti≈+π and tj≈−π give |ti−tj|≈2π despite matching directions.
        # ChebConv learns to treat both near-0 and near-2π as "consistent"; values near π
        # are the true mismatches (perpendicular structures).

        # sknw gives one undirected edge per vessel segment (u──v), but ChebConv does message
        # passing along directed edges: node u reads messages arriving from its neighbours, so
        # it needs an explicit u←v entry, not just u→v.  Storing both (i→j) and (j→i) lets
        # every node receive topology signals from both ends of every connected segment.
        # The feature vector is identical for both directions: length, intensity, and delta_theta
        # all describe the segment itself, not which end you're standing on.
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
        'node_pos':   np.array(node_pos,        dtype=np.float32),   # (N, 2) — each row is [x=col, y=row]
        'node_feats': np.array(node_feats,      dtype=np.float32),   # (N, 39)
        'edge_index': np.array([edge_src, edge_dst], dtype=np.int64),# (2, E)
        'edge_feats': np.array(edge_feats_list, dtype=np.float32),   # (E, 4)
    }


# ── ChebConv Graph Network ─────────────────────────────────────────────────────

class VesselGraphNet(nn.Module):
    """
    3-layer ChebConv with K=10 polynomial hops.
    Input:  node_feats (N, NODE_FEAT_DIM=39)  anisotropic + topological features
    Output: node_emb   (N, 64)               enriched embeddings encoding vessel topology

    K=10: each node aggregates information from up to 10 hops away per layer.
    3 layers → 30 hops total → covers full vessel length regardless of pixel distance.
    """
    def __init__(self, in_ch=NODE_FEAT_DIM, hidden=64, out_ch=64, K=10):
        super().__init__()
        self.conv1 = ChebConv(in_ch,  hidden, K=K)   # (N, 39) → (N, 64): aggregates 10-hop neighbourhood
        self.conv2 = ChebConv(hidden, hidden, K=K)   # (N, 64) → (N, 64): deeper topology context
        self.conv3 = ChebConv(hidden, out_ch, K=K)   # (N, 64) → (N, 64): final topology embedding
        self.bn1   = nn.BatchNorm1d(hidden)           # normalizes 64 features across N nodes → (N, 64)
        self.bn2   = nn.BatchNorm1d(hidden)           # normalizes 64 features across N nodes → (N, 64)

    def forward(self, x, edge_index):
        """
        x:          (N, 39)  node feature matrix
        edge_index: (2, E)   COO-format edge list
        Returns:    (N, 64)  enriched node embeddings
        """
        x = F.relu(self.bn1(self.conv1(x, edge_index)))   # (N,39)→conv1→(N,64)→bn1→relu→(N,64)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))   # (N,64)→conv2→(N,64)→bn2→relu→(N,64)
        x = self.conv3(x, edge_index)                     # (N,64)→conv3→(N,64) — no activation before scatter
        return x                                           # (N, 64)


# ── Scatter Node Features → Pixel Grid ────────────────────────────────────────

def scatter_to_pixels(node_pos, node_emb, H, W, device, sigma=2.0):
    """
    Scatter node embeddings (on 1-pixel skeleton) back to full H×W pixel grid.
    Gaussian spread fills vessel width so every pixel inside a vessel gets graph features.

    Operates on a single image (not a batch).  Output is (1, D, H, W).

    node_pos: (N, 2)  [x=col, y=row] pixel coordinates of skeleton nodes
    node_emb: (N, D)  node embeddings from ChebConv
    Returns:  (1, D, H, W)  scattered + blurred feature map
    """
    D      = node_emb.shape[1]                          # scalar int — embedding dim, e.g. 64
    F_scat = torch.zeros(1, D, H, W, device=device)     # (1, D, H, W) — blank canvas, all zeros

    xs = node_pos[:, 0].long().clamp(0, W-1)   # (N,) int64 — x (col) coords clamped to [0, W-1]
    ys = node_pos[:, 1].long().clamp(0, H-1)   # (N,) int64 — y (row) coords clamped to [0, H-1]

    # Place each node's D-dim embedding at its pixel position
    F_scat[0, :, ys, xs] = node_emb.T          # node_emb.T: (D, N) → writes to (1, D, H, W)

    # Gaussian spread σ=2 — fills vessel width around 1-pixel skeleton nodes.
    # Without this, only the exact skeleton pixels get graph features;
    # nearby vessel pixels would have zeros and see no topology information.
    #
    # Kernel equation:  G(r,c) = exp(−(r²+c²) / (2σ²))
    #   r = row offset from centre, c = col offset from centre.
    #
    #   Normalisation:  G_norm(r,c) = G(r,c) / ΣG
    #     where ΣG = Σ_{r,c} G(r,c) summed over all ks×ks grid positions
    #     ensures Σ G_norm = 1 — convolution preserves mean signal level.
    #
    #   e.g. σ=2, ks=13 (r,c ∈ {−6,…,+6}):
    #     Unnormalised:  G(0,0)=exp(0)=1.000,  G(0,±1)=exp(−1/8)≈0.882,  G(0,±2)=exp(−4/8)≈0.607
    #     ΣG ≈ 25.08  (sum of all 169 kernel entries)
    #     Normalised:    G_norm(0,0)=1.000/25.08≈0.040,  G_norm(0,±1)≈0.035,  G_norm(0,±2)≈0.024
    #
    # Before blur — F_scat has point masses at ALL N skeleton node positions simultaneously
    # (F_scat[0, :, ys, xs] = node_emb.T writes every node in one vectorised op):
    #
    #   col →   0     1     2     3     4
    #   row 0  0.00  0.00  0.00  0.00  0.00
    #   row 1  0.00  V_A   0.00  0.00  0.00   ← node A: embedding V_A at pixel (1,1)
    #   row 2  0.00  0.00  0.00  0.00  0.00
    #   row 3  0.00  0.00  0.00  V_B   0.00   ← node B: embedding V_B at pixel (3,3)
    #   row 4  0.00  0.00  0.00  0.00  0.00
    #
    # After blur — each output pixel sums contributions from ALL nodes weighted by G
    # (G = Gaussian kernel defined above: G(r,c) = exp(−(r²+c²)/(2σ²)), normalised to ΣG=1):
    #
    #   Variables:
    #     r, c       — row and col index of the output pixel being evaluated (0..H-1, 0..W-1)
    #     n          — index over all N skeleton nodes
    #     V_n        — D-dim embedding vector of node n (output of ChebConv, shape (D,))
    #     r_n, c_n   — row and col pixel position of node n in the skeleton (from node_pos)
    #     Δr, Δc     — offset from output pixel to node: Δr = r_n − r,  Δc = c_n − c
    #     G_norm(Δr,Δc) — normalised Gaussian weight at offset (Δr,Δc): how strongly
    #                     node n contributes to pixel (r,c) based on their distance
    #
    #   Output formula (one value per pixel, per embedding channel):
    #     output(r,c) = Σ_n  V_n × G_norm(r_n − r,  c_n − c)
    #   For the two-node example above:
    #     output(r,c) = V_A×G_norm(1−r, 1−c) + V_B×G_norm(3−r, 3−c)
    #
    #   Why this formula is correct — F.conv2d computes cross-correlation (kernel NOT flipped):
    #     dr, dc     — kernel offset variables in the cross-correlation sum, ranging over ±(ks//2)
    #     output(r,c) = Σ_{dr,dc} F_scat[r+dr, c+dc] × G_norm(dr, dc)
    #   F_scat is zero everywhere except V_n at (r_n, c_n), so only one term per node survives:
    #     r+dr = r_n  and  c+dc = c_n  →  dr = r_n−r,  dc = c_n−c
    #     output(r,c) = V_n × G_norm(r_n − r,  c_n − c)
    #   G is symmetric: G_norm(Δr,Δc) = G_norm(−Δr,−Δc); cross-correlation = convolution here.
    #
    #   Unnormalised values (σ=2): G(Δr,Δc) = exp(−(Δr²+Δc²)/8)
    #     G(0, 0) = exp(0)    = 1.000
    #     G(0,±1) = exp(−1/8) ≈ 0.882   G(±1, 0) = exp(−1/8) ≈ 0.882
    #     G(±1,±1)= exp(−2/8) ≈ 0.779   G(0, ±2) = exp(−4/8) ≈ 0.607
    #     G(±2, 0)= exp(−4/8) ≈ 0.607   G(±2,±2) = exp(−8/8) ≈ 0.368
    #   After normalising by ΣG≈25.08 the absolute values scale down; relative shape is preserved.
    #
    #   Contribution of node A (r_A=1, c_A=1, embedding V_A) to each output pixel:
    #   Δr = r_A−r = 1−r,  Δc = c_A−c = 1−c  (how far pixel (r,c) is from the node)
    #
    #   col →      0              1              2              3              4
    #   row 0  G(+1,+1)×V_A  G(+1, 0)×V_A  G(+1,−1)×V_A  G(+1,−2)×V_A  G(+1,−3)×V_A
    #   row 1  G( 0,+1)×V_A      V_A        G( 0,−1)×V_A  G( 0,−2)×V_A  G( 0,−3)×V_A  ← peak (Δr=Δc=0)
    #   row 2  G(−1,+1)×V_A  G(−1, 0)×V_A  G(−1,−1)×V_A  G(−1,−2)×V_A  G(−1,−3)×V_A
    #   row 3  G(−2,+1)×V_A  G(−2, 0)×V_A  G(−2,−1)×V_A  G(−2,−2)×V_A  G(−2,−3)×V_A
    #   row 4  G(−3,+1)×V_A  G(−3, 0)×V_A  G(−3,−1)×V_A  G(−3,−2)×V_A  G(−3,−3)×V_A
    #
    #   Node B (r_B=3, c_B=3, embedding V_B) adds V_B×G_norm(3−r, 3−c) to every cell.
    #   Full output = contribution from A + contribution from B (summed over all N nodes).
    #
    # groups=D: each of the D embedding channels is convolved independently with the
    # same kernel — no cross-channel mixing.  Equivalent to D separate 2D blurs.
    kernel_size = int(6 * sigma + 1) | 1   # int scalar (always odd) — e.g. σ=2.0 → ks=13, σ=1.5 → ks=11
    # kernel_size must be ODD so the kernel has a unique centre pixel.
    # F.conv2d slides the kernel across the input: at each step it places the kernel
    # centre over one input pixel and computes a weighted sum. With an odd kernel (ks=13),
    # the centre is at index ks//2 = 6, sitting exactly on an input pixel → no spatial shift.
    # With an even kernel (ks=12) there is no integer centre: the "centre" falls between two
    # pixels, shifting the output by 0.5 px and breaking same-size output (output would be
    # H-1 × W-1 instead of H × W with padding = ks//2 = 6).
    #
    # How the formula is derived:
    #   Truncate at ±3σ — G(±3σ) = exp(−(3σ)²/(2σ²)) = exp(−4.5) ≈ 0.011, negligible tail.
    #   Span from −3σ to +3σ = 6σ pixels.  Add 1 for the centre pixel → total = 6σ+1.
    #   int() truncates the float to an integer.
    #   | 1 forces the result odd via bitwise-OR with 1 (sets the least-significant bit):
    #     odd  input: 13 in binary = ...01101 → 13 | 1 = ...01101 = 13  (LSB already 1, unchanged)
    #     even input: 10 in binary = ...01010 → 10 | 1 = ...01011 = 11  (LSB flipped 0→1, adds 1)
    #
    #   σ=2.0 → 6×2.0+1=13.0 → int=13 (odd)  → 13|1=13 → ks=13  (covers −6..0..+6 pixels)
    #   σ=1.5 → 6×1.5+1=10.0 → int=10 (even) → 10|1=11 → ks=11  (covers −5..0..+5 pixels)
    padding     = kernel_size // 2
    # same-size padding: F.conv2d with padding=ks//2 produces the same (H, W) output as input.
    # e.g. ks=13 → padding=6: each border row/column gets 6 zeros added → convolution output stays H×W.
    blur_kernel = _gaussian_kernel(kernel_size, sigma, D).to(device)   # (D, 1, ks, ks)
    F_scat      = F.conv2d(F_scat, blur_kernel, padding=padding, groups=D)  # (1,D,H,W) depthwise blur

    return F_scat                               # (1, D, H, W) float32


def _gaussian_kernel(ks, sigma, groups):
    """
    Separable 2D Gaussian kernel for depthwise convolution.
    groups=D means each of the D channels gets its own kernel (identical) — efficient.
    Returns (D, 1, ks, ks) float32.
    """
    coords = torch.arange(ks, dtype=torch.float32) - ks // 2
    # (ks,) float32 — pixel offsets from kernel centre: 0,1,...,ks-1 shifted by ks//2
    # e.g. ks=13: arange=[0..12], ks//2=6 → coords=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]

    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    # (ks,) float32 — 1D Gaussian: G(x) = exp(−x²/(2σ²))
    # evaluates the standard Gaussian bell curve at each integer offset in coords
    # e.g. σ=2: G(0)=exp(0)=1.000, G(±1)=exp(−0.125)≈0.882, G(±3)=exp(−1.125)≈0.325, G(±6)=exp(−4.5)≈0.011
    g = g / g.sum()
    # (ks,) float32 — normalise so Σg = 1 (the 13 values sum to 1)
    # required so the convolution preserves mean signal level — without this,
    # applying the kernel multiplies every pixel by ΣG_2D_unnorm ≈ 25× its value.
    # Where 25 comes from (σ=2, ks=13):
    #   1D unnormalized sum Σg_1D = g(-6)+...+g(0)+...+g(+6)
    #     = 2×(0.011+0.044+0.135+0.325+0.607+0.882) + 1.000 ≈ 5.008
    #   k2d (built two lines below) is the (ks,ks) 2D kernel matrix where k2d[r,c]=g[r]*g[c]
    #   — the unnormalized 2D Gaussian.  Centre 7×7 slice (σ=2, offsets −3..+3):
    #
    #     r\c   −3      −2      −1       0      +1      +2      +3
    #      −3  0.106   0.197   0.287   0.325   0.287   0.197   0.106
    #      −2  0.197   0.368   0.535   0.607   0.535   0.368   0.197
    #      −1  0.287   0.535   0.778   0.882   0.778   0.535   0.287
    #       0  0.325   0.607   0.882   1.000   0.882   0.607   0.325   ← centre row (r=0)
    #      +1  0.287   0.535   0.778   0.882   0.778   0.535   0.287
    #      +2  0.197   0.368   0.535   0.607   0.535   0.368   0.197
    #      +3  0.106   0.197   0.287   0.325   0.287   0.197   0.106
    #
    #   Each cell follows:  k2d[r,c] = g[r] × g[c] = exp(−r²/8) × exp(−c²/8) = exp(−(r²+c²)/8)
    #   e.g. k2d[−1, 0] = exp(−1/8) × exp(0)   = 0.882 × 1.000 = 0.882  (one row above centre)
    #        k2d[−2,+1] = exp(−4/8) × exp(−1/8) = 0.607 × 0.882 = 0.535  (two up, one right)
    #   The full 13×13 k2d has 169 cells; total sum derived by factoring the double sum:
    #     Σk2d = Σ_r Σ_c k2d[r,c]
    #          = Σ_r Σ_c g[r]*g[c]      (substitute definition k2d[r,c]=g[r]*g[c])
    #          = Σ_r g[r] * (Σ_c g[c])  (g[r] is constant w.r.t. the inner sum over c)
    #          = Σ_r g[r] * Σg_1D       (Σ_c g[c] = Σg_1D, same 1D sum for every row)
    #          = Σg_1D * Σg_1D          (pull out the constant Σg_1D from the outer sum)
    #          = (Σg_1D)² = 5.008² ≈ 25.08

    # A 2D Gaussian is separable: G2D(r,c) = exp(−(r²+c²)/(2σ²))
    #                                       = exp(−r²/(2σ²)) × exp(−c²/(2σ²))
    #                                       = G1D(r) × G1D(c)
    # So every cell (r,c) of the 2D kernel is just two independent 1D lookups multiplied.
    # The outer product of g with itself fills the whole matrix in one operation:
    #
    #   Outer product: take g as a column vector and g as a row vector, multiply:
    #
    #              g[0]   g[1]   g[2]      ← row vector  (1, ks)  g.unsqueeze(0)
    #   g[0]  [ g[0]×g[0]  g[0]×g[1]  g[0]×g[2] ]
    #   g[1]  [ g[1]×g[0]  g[1]×g[1]  g[1]×g[2] ]   ← k2d, shape (ks, ks)
    #   g[2]  [ g[2]×g[0]  g[2]×g[1]  g[2]×g[2] ]
    #     ↑
    #   column vector (ks, 1)  g.unsqueeze(1)
    #
    #   k2d[i,j] = g[i] × g[j] = G1D(offset_i) × G1D(offset_j) = G2D(offset_i, offset_j)
    #   Every cell lands on the correct 2D Gaussian value — no loop needed.
    k2d = g.unsqueeze(0) * g.unsqueeze(1)
    # g.unsqueeze(0): (ks,) → (1, ks)  — row vector
    # g.unsqueeze(1): (ks,) → (ks, 1)  — column vector
    # broadcasting (1,ks)×(ks,1) → (ks,ks): k2d[r,c] = g[r] * g[c]
    # e.g. ks=13, σ=2:
    #   k2d[0,0] = g[-6]*g[-6] = 0.011*0.011 ≈ 0.000  (corner — very far from centre)
    #   k2d[6,6] = g[ 0]*g[ 0] = 1.000*1.000 = 1.000  (centre — peak, before normalisation)
    #   k2d[6,7] = g[ 0]*g[+1] = 1.000*0.882 ≈ 0.882  (one step right of centre)

    return k2d.view(1, 1, ks, ks).repeat(groups, 1, 1, 1)
    # F.conv2d weight format requires (out_channels, in_channels/groups, kH, kW).
    # With groups=D the input (1,D,H,W) is split into D single-channel groups, so
    # in_channels/groups = 1 and out_channels = D → weight must be (D, 1, ks, ks).
    #
    # Step 1 — view(1,1,ks,ks): insert the two leading channel dims:
    #
    #   k2d (ks,ks)         →   view   →   (1, 1, ks, ks)≠
    #   ┌───────────┐               out[0], in[0]
    #   │ · · · · · │               ┌───────────┐
    #   │ · · G · · │               │ · · · · · │
    #   │ · · · · · │               │ · · G · · │
    #   └───────────┘               │ · · · · · │
    #   single 2D matrix            └───────────┘
    #                               one kernel, one in-channel slot
    #
    # Step 2 — repeat(D,1,1,1): stack D identical copies along dim 0 (out-channel dim):
    #
    #   repeat(D, 1, 1, 1): each argument is how many times to repeat along that dim
#     dim 0 (out-channels): repeat D times → size 1 → size D  (stack D copies)
#     dim 1 (in-channels) : repeat 1 time  → size 1 → size 1  (unchanged)
#     dim 2 (kH)          : repeat 1 time  → size ks→ size ks (unchanged)
#     dim 3 (kW)          : repeat 1 time  → size ks→ size ks (unchanged)
#
#   (1,1,ks,ks)  →  repeat(D,1,1,1)  →  (D, 1, ks, ks)
    #
    #   kernel[0]  kernel[1]  ...  kernel[D-1]     ← D identical copies
    #   ┌───────┐  ┌───────┐       ┌───────┐
    #   │ · G · │  │ · G · │  ...  │ · G · │
    #   └───────┘  └───────┘       └───────┘
    #      ↓           ↓                ↓
    #   channel 0  channel 1  ...  channel D-1   of F_scat
    #
    # F.conv2d with groups=D: kernel[d] is applied only to F_scat[0,d,:,:].
    # Each embedding channel is blurred independently with the same Gaussian —
    # no mixing between channels 0..D-1.


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
        (B,256,H/4,W/4)→(B,32,H,W)          build 39-dim node features:
        F_pixel                                direction(cos,sin), diameter,
                                               curvature, degree, component_size,
                                               diameter_std, CNN(32)
                                             ChebConv(K=10) × 3 layers
                                             (N,39)→(N,64)
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
    def __init__(self, sam2_model=HF_MODEL_ID):
        super().__init__()
        self.encoder   = SAM2Encoder(hf_model_id=sam2_model)                   # frozen Hiera ViT-L
        self.cnn_dec   = CNNDecoder(in_ch=256)                                  # (B,256,H/4,W/4)→(B,32,H,W)
        self.graph_net = VesselGraphNet(in_ch=NODE_FEAT_DIM, hidden=64,         # (N,39)→(N,64)
                                        out_ch=64, K=10)
        self.fuse_proj = nn.Conv2d(32 + 64, 32, kernel_size=1)                 # (B,96,H,W)→(B,32,H,W)
        self.head      = nn.Conv2d(32, 1, kernel_size=1)                        # (B,32,H,W)→(B,1,H,W)

    def forward(self, x, use_graph=True, sharpness=None, grad_mag=None):
        """
        Standard two-pass self-feedback forward for graph-augmented segmentation.

        Why two-pass? The graph path needs a binary skeleton/graph as input, but
        we don't have ground truth at inference time. The standard solution is
        self-distillation: the model produces a coarse mask in pass 1 (CNN-only)
        and uses *its own* coarse mask in pass 2 to build the graph. This keeps
        train and inference behaviour identical — both use the model's coarse
        prediction as the graph source — eliminating any train/test mismatch.

        Pass 1 (no_grad): CNN → coarse_logits → sigmoid → threshold → coarse_mask.
        Pass 2 (with grad): per-sample graph constructed from coarse_mask, GAT
            produces topology features F_graph, fused with F_pixel via 1×1 conv,
            head produces refined logits. Loss is computed on the refined logits.

        x:          (B, 1, H, W)  float32 tile ∈ [0,1]
        use_graph:  if False, skip pass 2 → CNN-only logits (ablation / fallback).
                    Default True.

        Returns: (logits, coarse_logits)
            logits:        (B, 1, H, W)  refined logits  (use_graph=True)  or CNN-only logits (use_graph=False)
            coarse_logits: (B, 1, H, W)  CNN-only logits for deep-supervision loss (use_graph=True only),
                           or None when use_graph=False.
        """
        B, _, H_in, W_in = x.shape   # input tile shape, e.g. (B, 1, 1300, 1024)

        # ── Encoder (frozen) + CNN decoder — feature extraction ─────────────
        # 3-channel input to SAM2:
        #   ch0 = grayscale, ch1 = gradient magnitude (vessel edges), ch2 = sharpness (focus)
        # Without both maps, falls back to repeating grayscale 3×.
        if sharpness is not None and grad_mag is not None:
            x_enc = torch.cat([x, grad_mag, sharpness], dim=1)
        else:
            x_enc = x
        feats   = self.encoder(x_enc)
        F_pixel = self.cnn_dec(feats)             # (B, 32, H_sam, W_sam) ≈ (B, 32, 1024, 1024)

        if not use_graph:
            # CNN-only path (used for ablation studies or when graph is disabled).
            coarse_logits = None
            logits = self.head(F_pixel)
        else:
            # ── Pass 1: coarse logits for deep supervision + mask for graph ──
            # coarse_logits retains gradient so deep-supervision loss trains the
            # CNN head directly. The mask used for graph construction is detached
            # (.detach() + .numpy()) — graph construction is non-differentiable,
            # so no gradient flows through it regardless.
            coarse_logits     = self.head(F_pixel)                   # (B, 1, H_sam, W_sam) — gradient ON
            mask_coarse_batch = (torch.sigmoid(coarse_logits.detach()) > 0.5).cpu().numpy()  # (B, 1, H_sam, W_sam)

            # Resize input image to F_pixel's resolution once for the whole batch.
            # build_vessel_graph uses raw image intensities for some node features,
            # so img and mask must be at the same scale as F_pixel.
            H_sam, W_sam = F_pixel.shape[-2:]
            if (H_in, W_in) != (H_sam, W_sam):
                x_sam = F.interpolate(x, size=(H_sam, W_sam),
                                      mode='bilinear', align_corners=False)
            else:
                x_sam = x

            # ── Pass 2: per-sample graph features ─────────────────────────
            # Built one graph per sample (correct for any batch size). Earlier
            # versions used sample 0's graph for the whole batch — that was a
            # bug at B>1.
            #
            # All arrays are forced contiguous via np.ascontiguousarray BEFORE
            # passing to skeletonize/sknw. Slicing and .astype() can yield
            # non-contiguous views in some numpy versions, and the C
            # extensions in skimage/sknw assume contiguous C-order memory —
            # passing a view triggers heap corruption (`malloc(): invalid next
            # size`) at unpredictable points later in the run.
            F_graph_list = []
            for b in range(B):
                try:
                    mask_b = np.ascontiguousarray(mask_coarse_batch[b, 0]).astype(bool, copy=True)
                    img_b  = np.ascontiguousarray(
                        x_sam[b, 0].detach().cpu().numpy().astype(np.float32, copy=True))
                    pf_b   = np.ascontiguousarray(
                        F_pixel[b].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32, copy=True))

                    graph_data = build_vessel_graph(mask_b, img_b, pf_b)
                    if graph_data is None or len(graph_data.get('node_pos', [])) < 2:
                        raise ValueError('degenerate graph')

                    nf      = torch.from_numpy(graph_data['node_feats']).to(x.device)  # (N, 39)
                    ei      = torch.from_numpy(graph_data['edge_index']).to(x.device)  # (2, E)
                    np_pos  = torch.from_numpy(graph_data['node_pos']).to(x.device)    # (N, 2)

                    node_emb = self.graph_net(nf, ei)                            # (N, 64) topology embed
                    fg       = scatter_to_pixels(np_pos, node_emb,
                                                 H_sam, W_sam, x.device)         # (1, 64, H_sam, W_sam)
                    F_graph_list.append(fg[0])                                   # (64, H_sam, W_sam)
                except Exception:
                    # Degenerate graph (e.g. coarse mask mostly empty in early
                    # training, or a tile with no vessels). Use zero topology
                    # features so this sample falls back to CNN-only behaviour.
                    F_graph_list.append(torch.zeros(64, H_sam, W_sam, device=x.device))

            F_graph = torch.stack(F_graph_list, dim=0)                # (B, 64, H_sam, W_sam)
            F_fused = torch.cat([F_pixel, F_graph], dim=1)            # (B, 96, H_sam, W_sam)
            F_fused = self.fuse_proj(F_fused)                          # (B, 32, H_sam, W_sam)
            logits  = self.head(F_fused)                               # (B, 1,  H_sam, W_sam)

        # SAM2 always works at 1024×1024 internally, so logits come out at that
        # resolution. Resize both outputs back to the original tile shape so
        # loss / mask alignment is correct.
        if logits.shape[-2:] != (H_in, W_in):
            logits = F.interpolate(logits, size=(H_in, W_in),
                                   mode='bilinear', align_corners=False)
        # coarse_logits.shape[-2:] is the SAM2-internal (H_sam, W_sam) = (1024, 1024).
        # (H_in, W_in) is the original input tile resolution, e.g. (1300, 1024).
        # The check fails (and resize is skipped) only when the input was already 1024×1024.
        # Skip entirely when coarse_logits is None (use_graph=False path).
        if coarse_logits is not None and coarse_logits.shape[-2:] != (H_in, W_in):
            coarse_logits = F.interpolate(coarse_logits, size=(H_in, W_in),
                                          mode='bilinear', align_corners=False)

        return logits, coarse_logits
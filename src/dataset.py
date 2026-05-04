import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tifffile import imread
from skimage.transform import resize
from scipy.ndimage import map_coordinates, gaussian_filter

TILE_W  = 1024
TILE_H  = 1300   # full strip height — no vertical cuts
STRIDE  = 512    # horizontal stride only


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize(img):
    """Per-image min-max normalization. No pixel removal."""
    img  = img.astype(np.float32)
    lo   = img.min()
    hi   = img.max()
    img  = (img - lo) / (hi - lo + 1e-8)       # [0, 1]
    return (img * 255).astype(np.uint8)          # uint8


# ── Hanning boundary weight map ────────────────────────────────────────────────

def hanning_weight(h, w):
    """
    2D Hann window:  center = 1.0,  edges = 0.0
    Gates the loss — boundary pixels near tile edges
    contribute nothing, preventing broken-vessel learning.
    shape: (h, w)  float32
    """
    wy  = np.hanning(h).astype(np.float32)      # (h,)
    wx  = np.hanning(w).astype(np.float32)      # (w,)
    W   = np.outer(wy, wx)                       # (h, w)
    W  /= W.mean() + 1e-8                        # normalize mean to ~1
    return W


# ── Tiling ─────────────────────────────────────────────────────────────────────

def tile_image(img, mask=None):
    """
    Strip tiling: horizontal only, full image height preserved.
    No vertical vessel cuts.

    img:  (H, W)  uint8
    mask: (H, W)  uint8  or None

    Returns list of:
        (img_tile, mask_tile, x_offset)   if mask provided
        (img_tile, x_offset)              if mask is None
    """
    H, W = img.shape[:2]
    tw   = min(TILE_W, W)

    # Horizontal tile positions
    xs = list(range(0, max(W - tw, 1), STRIDE))
    if not xs or xs[-1] + tw < W:
        xs.append(W - tw)               # flush-right tile

    tiles = []
    for x in xs:
        x2       = min(x + tw, W)
        img_tile = img[:, x:x2]
        img_tile = _pad_tile(img_tile, H, tw)    # reflect-pad if edge tile

        if mask is not None:
            msk_tile = mask[:, x:x2]
            msk_tile = _pad_tile(msk_tile, H, tw)
            tiles.append((img_tile, msk_tile, x))
        else:
            tiles.append((img_tile, x))

    return tiles


def _pad_tile(arr, H, W):
    """Reflect-pad to exactly (H, W). Never zero-pad."""
    ph = H - arr.shape[0]
    pw = W - arr.shape[1]
    if ph > 0 or pw > 0:
        arr = np.pad(arr, ((0, ph), (0, pw)), mode='reflect')
    return arr


# ── Stitch predictions back ────────────────────────────────────────────────────

def stitch_tiles(tile_probs, tile_xs, img_h, img_w, tile_w=TILE_W):
    """
    Average overlapping tile probability maps.
    tile_probs: list of (H, W) float32 arrays   (sigmoid output)
    tile_xs:    list of int  (x offsets)
    Returns:    (img_h, img_w) bool  binary mask
    """
    accum = np.zeros((img_h, img_w), dtype=np.float32)
    count = np.zeros((img_h, img_w), dtype=np.float32)

    for prob, x in zip(tile_probs, tile_xs):
        w = min(tile_w, img_w - x)
        h = min(prob.shape[0], img_h)
        accum[:h, x:x+w] += prob[:h, :w]
        count[:h, x:x+w] += 1.0

    count = np.maximum(count, 1.0)
    return (accum / count) > 0.5


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_pairs(input_dir, output_dir):
    """
    Match input and output .tif files by leading integer ID.
    Returns sorted list of (img_path, mask_path) tuples.
    """
    def by_id(folder):
        # ** with recursive=True descends into subdirectories (D7/, D14/, D21/)
        # so all timepoint folders are searched without flattening the data layout.
        files = glob.glob(os.path.join(folder, '**', '*.tif'), recursive=True)
        d = {}
        for f in files:
            try:
                key = int(os.path.basename(f).split('_')[0])
                d[key] = f
            except ValueError:
                pass
        return d

    inputs  = by_id(input_dir)
    outputs = by_id(output_dir)
    matched = sorted(set(inputs) & set(outputs))
    return [(inputs[k], outputs[k]) for k in matched]


class VesselDataset(Dataset):
    """
    Yields one tile at a time with Hanning weight map.
    Each (image, mask) pair is pre-tiled; all tiles stored in memory.
    For 30 images × ~30 tiles = ~900 items.
    """
    def __init__(self, pairs, augment=False, seed=None):
        self.augment = augment
        self.seed    = seed   # None → random each call; int → reproducible per idx
        self.items   = []                        # (img_tile, msk_tile, hann)

        hann = hanning_weight(TILE_H, TILE_W).astype(np.float32)

        for img_path, msk_path in pairs:
            img  = normalize(imread(img_path))   # (H, W) uint8
            msk  = imread(msk_path)              # (H, W) uint8

            # vessel = 255 in your masks → bool
            msk  = (msk > 0).astype(np.uint8)

            for img_tile, msk_tile, _ in tile_image(img, msk):
                # Resize tile to (TILE_H, TILE_W) if needed for SAM2
                if img_tile.shape != (TILE_H, TILE_W):
                    img_tile = _pad_tile(img_tile, TILE_H, TILE_W)
                    msk_tile = _pad_tile(msk_tile, TILE_H, TILE_W)

                self.items.append((
                    img_tile.copy(),
                    msk_tile.copy(),
                    hann.copy()
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, msk, hann = self.items[idx]

        if self.augment:
            # Seed derived from base seed + idx so each sample is independently
            # reproducible; None falls through to a random RNG inside _augment.
            rng = np.random.default_rng(self.seed + idx) if self.seed is not None else None
            img, msk = _augment(img, msk, rng=rng)

        # img: (H, W) uint8 → float32 tensor (1, H, W)
        img_t  = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        msk_t  = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)
        hann_t = torch.from_numpy(hann).unsqueeze(0)

        return img_t, msk_t, hann_t


# ── Augmentation ───────────────────────────────────────────────────────────────

def _augment(img, msk, rng=None):
    """Geometric and intensity augmentations for vessel segmentation.

    rng: np.random.Generator — pass a seeded Generator for reproducibility,
         or None to draw a fresh random one each call.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape[:2]

    # ── Geometric ──────────────────────────────────────────────────────────────
    # Random horizontal flip
    if rng.random() > 0.5:
        img = np.fliplr(img).copy()
        msk = np.fliplr(msk).copy()

    # Random vertical flip
    if rng.random() > 0.5:
        img = np.flipud(img).copy()
        msk = np.flipud(msk).copy()

    # Random 90° rotation (k ∈ {1, 2, 3})
    # k=1 or k=3 swaps H and W (e.g. 1300×1024 → 1024×1300), which mismatches the
    # pre-computed Hanning weight map; resize back to the original (h, w) to keep
    # all tensors the same shape while still benefiting from rotational augmentation.
    k = rng.integers(0, 4)
    if k > 0:
        img = np.rot90(img, k).copy()
        msk = np.rot90(msk, k).copy()
        if img.shape[:2] != (h, w):
            img = resize(img, (h, w), preserve_range=True, anti_aliasing=True).astype(np.uint8)
            msk = resize(msk, (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(msk.dtype)

    # Random zoom-in: crop a random sub-region then resize back to original shape.
    # order=0 for mask preserves binary labels without interpolation artifacts.
    if rng.random() > 0.5:
        scale = rng.uniform(0.75, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0 = rng.integers(0, h - ch + 1)
        x0 = rng.integers(0, w - cw + 1)
        img = resize(img[y0:y0+ch, x0:x0+cw], (h, w),
                     preserve_range=True, anti_aliasing=True).astype(np.uint8)
        msk = resize(msk[y0:y0+ch, x0:x0+cw], (h, w),
                     order=0, preserve_range=True, anti_aliasing=False).astype(msk.dtype)

    # Elastic deformation: smooth random displacement field mimics vessel curvature variation.
    # alpha controls deformation magnitude, sigma controls smoothness.
    # Mask uses order=0 to avoid introducing fractional label values.
    if rng.random() > 0.7:
        alpha = h * rng.uniform(0.5, 2.0)
        sigma = h * 0.08
        dx = gaussian_filter(rng.standard_normal((h, w)), sigma) * alpha
        dy = gaussian_filter(rng.standard_normal((h, w)), sigma) * alpha
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        coords = [np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)]
        img = map_coordinates(img, coords, order=1, mode='reflect').astype(img.dtype)
        msk = map_coordinates(msk, coords, order=0, mode='reflect').astype(msk.dtype)

    # ── Intensity ──────────────────────────────────────────────────────────────
    # Gaussian noise: simulates sensor noise; only applied to image, not mask.
    if rng.random() > 0.5:
        sigma_n = rng.uniform(2, 12)
        noise = rng.normal(0, sigma_n, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Brightness / contrast jitter: alpha scales contrast, beta shifts brightness.
    if rng.random() > 0.5:
        alpha = rng.uniform(0.7, 1.4)          # contrast
        beta  = rng.integers(-25, 25)           # brightness
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Gamma correction: nonlinear brightness shift, robust to illumination changes.
    if rng.random() > 0.5:
        gamma = rng.uniform(0.5, 1.8)
        img = (255.0 * (img.astype(np.float32) / 255.0) ** gamma).astype(np.uint8)

    return img, msk

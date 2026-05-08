import os
import glob
import json
import argparse
import yaml
import random
import numpy as np
import torch
from tifffile import imread, imwrite
from tqdm import tqdm

from dataset     import normalize, tile_image, stitch_tiles, load_pairs, compute_sharpness, compute_gradient_magnitude
from model       import AttentionUNet, visualize_attention_maps
import wandb

# ── Prediction filtering parameters ────────────────────────────────────────
NUM_TRAINVAL_SAMPLES = 2  # number of random train-val images to predict
NUM_TEST_SAMPLES = 5      # number of test images (all of them)


def predict_image(model, img_path, device):
    """
    Full inference pipeline for one image.
    1. Load + normalize
    2. Tile horizontally
    3. Forward pass per tile
    4. Stitch overlapping tiles

    Returns: (mask, img_tile)
      mask:     (H, W) bool
      img_tile: last tile numpy array; model.last_psi matches it
    """
    img_raw  = imread(img_path)
    img_u8   = normalize(img_raw)               # (H, W) uint8
    H, W     = img_u8.shape[:2]

    tiles = tile_image(img_u8)
    probs = []
    tile_xs = []

    model.eval()
    with torch.no_grad():
        for img_tile, x_off in tqdm(tiles, desc='tiles', leave=False):
            t       = torch.from_numpy(img_tile.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            sharp_t = torch.from_numpy(compute_sharpness(img_tile)).unsqueeze(0).unsqueeze(0).to(device)
            grad_t  = torch.from_numpy(compute_gradient_magnitude(img_tile)).unsqueeze(0).unsqueeze(0).to(device)
            prob    = torch.sigmoid(model(t, sharpness=sharp_t, grad_mag=grad_t))
            probs.append(prob.squeeze().cpu().numpy())
            tile_xs.append(x_off)
    # img_tile retains last tile value after loop; model.last_psi matches it

    mask = stitch_tiles(probs, tile_xs, H, W)
    return mask, img_tile


def main(args):
    # MPS = Apple Silicon GPU (Metal); falls back to CPU on Intel Macs.
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    seed = 42
    config_path = os.path.join(os.path.dirname(args.ckpt_path), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
            if 'seed' in cfg:
                seed = cfg['seed']

    model = AttentionUNet().to(device)
    print(f'AttentionUNet loaded: trainable ResNet34 encoder + UNet decoder with attention gates')
    ckpt  = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) else ckpt)
    model.eval()

    # ** with recursive=True descends into subdirectories (D7/, D14/, D21/),
    # matching how train.py's load_pairs finds images.
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, '**', '*.tif'), recursive=True))

    # Load split labels written by train.py so each predicted image is labelled
    # "test" or "trainval" in the W&B Media panel.
    splits_path = os.path.join(os.path.dirname(args.ckpt_path), "data_splits.json")
    splits_info = json.load(open(splits_path)) if os.path.exists(splits_path) else {}

    # Filter to NUM_TRAINVAL_SAMPLES random train-val + NUM_TEST_SAMPLES test images
    if splits_info:
        trainval_paths = [p for p in img_paths if os.path.basename(p) in splits_info and splits_info[os.path.basename(p)] == "trainval"]
        test_paths = [p for p in img_paths if os.path.basename(p) in splits_info and splits_info[os.path.basename(p)] == "test"]

        random.seed(seed)
        selected_trainval = random.sample(trainval_paths, min(NUM_TRAINVAL_SAMPLES, len(trainval_paths)))
        selected_test = test_paths[:NUM_TEST_SAMPLES]

        img_paths = sorted(selected_trainval + selected_test)
        print(f'Filtered to: {len(selected_trainval)} train-val + {len(selected_test)} test images = {len(img_paths)} total')

    # Build {img_path → mask_path} lookup so we can compare predictions to ground truth.
    # load_pairs matches by leading integer ID, so it works even if filenames differ
    # (e.g. 17_..._Crop.tif paired with 17_..._Crop_Process_Binary.tif).
    img_to_mask = dict(load_pairs(args.input_dir, args.mask_dir)) if args.mask_dir else {}

    # log which checkpoint and which images are being predicted — links this
    # prediction run back to the exact training run that produced the checkpoint
    wandb.init(
        entity   = "eeebashit",
        project  = "vessel-seg",
        job_type = "predict",            # shown as a separate job type in the W&B dashboard
        config   = {
            "ckpt_path":   args.ckpt_path,
            "input_dir":   args.input_dir,
            "out_dir":     args.out_dir,
            "n_images":    len(img_paths),
            "input_files": img_paths,    # exact list of images being predicted
        }
    )
    print(f'Using checkpoint: {args.ckpt_path}')
    print(f'Images to predict: {len(img_paths)}')

    os.makedirs(args.out_dir, exist_ok=True)

    for img_path in tqdm(img_paths, desc='images'):
        fname    = os.path.basename(img_path).replace('.tif', '_pred.tif')
        out_path = os.path.join(args.out_dir, fname)

        mask, img_tile = predict_image(model, img_path, device)
        imwrite(out_path, (mask.astype(np.uint8) * 255))
        print(f'  saved → {out_path}')

        # W&B Media panels per image: originals, prediction, gt, tp, fp, fn,
        # combined. Each panel has its own step slider so you can isolate one
        # error type at a time, or compare predictions vs ground truth, or see
        # everything together. Caption tags from data_splits.json show whether
        # the image was test / trainval / no-GT.
        img_raw = imread(img_path)
        fname   = os.path.basename(img_path)
        # Label hierarchy: split assignment from training, or "no_gt" if the
        # image has no matching mask, or "extra" otherwise.
        if fname in splits_info:
            label = splits_info[fname]                # "test" / "trainval"
        elif img_path not in img_to_mask:
            label = "no_gt"
        else:
            label = "extra"

        # Percentile (1–99%) contrast stretch — outlier-robust so vessel
        # structures stay visible. Plain min-max compresses the histogram when
        # a few extreme pixels sit at the ends of the range.
        lo, hi   = np.percentile(img_raw, (1, 99))
        img_norm = np.clip((img_raw.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)
        img_u8   = (img_norm * 255).astype(np.uint8)
        img_rgb  = np.stack([img_u8] * 3, axis=-1)            # (H, W, 3) grayscale → RGB

        # Color palette (high contrast on dark microscopy):
        #   red    = "model predicted vessel" (prediction, FP)
        #   green  = "actual vessel"          (ground truth, TP)
        #   yellow = "missed vessel"          (FN)
        RED, GREEN, YELLOW = (
            np.array([255,   0,   0]),
            np.array([  0, 220,   0]),
            np.array([255, 220,   0]),
        )
        alpha = 0.6

        def overlay(base, m, color):
            """Blend `color` onto `base` wherever boolean mask `m` is True."""
            out = base.copy()
            out[m] = (alpha * color + (1 - alpha) * out[m]).astype(np.uint8)
            return out

        def make_combined(m, gt_):
            """Build one TP/FP/FN composite view for a given mask vs ground truth."""
            tp = m  &  gt_
            fp = m  & ~gt_
            fn = ~m &  gt_
            c  = img_rgb.copy()
            c[tp] = (alpha * GREEN  + (1 - alpha) * c[tp]).astype(np.uint8)
            c[fp] = (alpha * RED    + (1 - alpha) * c[fp]).astype(np.uint8)
            c[fn] = (alpha * YELLOW + (1 - alpha) * c[fn]).astype(np.uint8)
            return c

        grad_u8  = (compute_gradient_magnitude(img_u8) * 255).astype(np.uint8)
        sharp_u8 = (compute_sharpness(img_u8)          * 255).astype(np.uint8)

        cap = f"[{label}] {fname}"
        log_payload = {
            "originals":  wandb.Image(img_u8,   caption=f"{cap}  |  grayscale"),
            "ch_grad":    wandb.Image(grad_u8,  caption=f"{cap}  |  gradient magnitude"),
            "ch_sharp":   wandb.Image(sharp_u8, caption=f"{cap}  |  sharpness (VoL)"),
            "prediction": wandb.Image(overlay(img_rgb, mask, RED),
                                      caption=f"{cap}  |  prediction"),
        }

        # If GT mask exists, log isolated TP/FP/FN panels
        gt_path = img_to_mask.get(img_path)
        if gt_path:
            gt = imread(gt_path) > 0                          # (H, W) bool ground truth
            log_payload.update({
                "gt":       wandb.Image(overlay(img_rgb, gt, GREEN),
                                        caption=f"{cap}  |  green = ground-truth vessel"),
                "tp":       wandb.Image(overlay(img_rgb, mask  &  gt, GREEN),
                                        caption=f"{cap}  |  TP"),
                "fp":       wandb.Image(overlay(img_rgb, mask  & ~gt, RED),
                                        caption=f"{cap}  |  FP"),
                "fn":       wandb.Image(overlay(img_rgb, ~mask &  gt, YELLOW),
                                        caption=f"{cap}  |  FN"),
                "combined": wandb.Image(make_combined(mask, gt),
                                        caption=f"{cap}  |  green=TP  red=FP  yellow=FN"),
            })

        # Attention maps: use the last tile's image — last_psi was saved during its forward pass.
        # Using the full img_rgb would stretch one tile's attention map across the whole image.
        tile_rgb    = np.stack([img_tile] * 3, axis=-1)
        attn_panels = visualize_attention_maps(model, tile_rgb)
        log_payload[f"attn/input_tile"] = wandb.Image(tile_rgb, caption=f"Last tile: {fname}")
        log_payload.update({f"attn/{k}": v for k, v in attn_panels.items()})

        wandb.log(log_payload)


    wandb.finish()   # marks predict run complete in the dashboard


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True)
    parser.add_argument('--ckpt_path',  required=True)
    parser.add_argument('--out_dir',    default='../predictions')
    parser.add_argument('--mask_dir',   default=None,
                        help='Optional ground-truth mask folder. If given, log a per-image '
                             'TP/FP/FN error map alongside the prediction.')
    args = parser.parse_args()
    main(args)

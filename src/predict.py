import os
import glob
import json
import argparse
import numpy as np
import torch
from tifffile import imread, imwrite
from tqdm import tqdm

from dataset     import normalize, tile_image, stitch_tiles, load_pairs, compute_sharpness, compute_gradient_magnitude
from model       import VesselSegNet
from postprocess import postprocess           # rule-based mask cleanup
import wandb


def predict_image(model, img_path, device):
    """
    Full inference pipeline for one image.
    1. Load + normalize
    2. Tile horizontally
    3. Forward pass per tile
    4. Stitch overlapping tiles
    5. Postprocess

    Returns: (H, W) bool binary mask
    """
    img_raw  = imread(img_path)
    img_u8   = normalize(img_raw)               # (H, W) uint8
    H, W     = img_u8.shape[:2]

    tiles    = tile_image(img_u8)               # list of (img_tile, x_offset)
    tile_probs = []
    tile_xs    = []

    model.eval()
    with torch.no_grad():
        for img_tile, x_off in tqdm(tiles, desc='tiles', leave=False):
            t = torch.from_numpy(
                    img_tile.astype(np.float32) / 255.0
                ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

            sharp   = compute_sharpness(img_tile)
            grad    = compute_gradient_magnitude(img_tile)
            sharp_t = torch.from_numpy(sharp).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
            grad_t  = torch.from_numpy(grad).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,H,W)
            logits, _ = model(t, use_graph=False, sharpness=sharp_t, grad_mag=grad_t)
            prob   = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)
            tile_probs.append(prob)
            tile_xs.append(x_off)

    # Stitch + threshold
    mask = stitch_tiles(tile_probs, tile_xs, H, W)
    # mask = postprocess(mask)
    return mask


def main(args):
    # MPS = Apple Silicon GPU (Metal); falls back to CPU on Intel Macs.
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    model = VesselSegNet().to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    # ** with recursive=True descends into subdirectories (D7/, D14/, D21/),
    # matching how train.py's load_pairs finds images.
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, '**', '*.tif'), recursive=True))

    # Load split labels written by train.py so each predicted image is labelled
    # "test" or "trainval" in the W&B Media panel.
    splits_path = os.path.join(os.path.dirname(args.ckpt_path), "data_splits.json")
    splits_info = json.load(open(splits_path)) if os.path.exists(splits_path) else {}

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
    print(f'Images to predict: {len(img_paths)}')

    os.makedirs(args.out_dir, exist_ok=True)

    for img_path in tqdm(img_paths, desc='images'):
        fname = os.path.basename(img_path).replace('.tif', '_pred.tif')
        out_path = os.path.join(args.out_dir, fname)

        mask    = predict_image(model, img_path, device)        # raw model output
        mask_pp = postprocess(mask)                              # cleaned: small blobs removed, holes filled, gaps closed

        # Save the postprocessed mask (cleaner, what you'd actually use downstream).
        imwrite(out_path, (mask_pp.astype(np.uint8) * 255))
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
        #   red   = "model predicted vessel" (prediction, FP)
        #   green = "actual vessel"          (ground truth, TP)
        #   cyan  = "missed vessel"          (FN)
        RED, GREEN, CYAN = (
            np.array([255,   0,   0]),
            np.array([  0, 220,   0]),
            np.array([  0, 200, 255]),
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
            c[tp] = (alpha * GREEN + (1 - alpha) * c[tp]).astype(np.uint8)
            c[fp] = (alpha * RED   + (1 - alpha) * c[fp]).astype(np.uint8)
            c[fn] = (alpha * CYAN  + (1 - alpha) * c[fn]).astype(np.uint8)
            return c

        cap = f"[{label}] {fname}"
        log_payload = {
            "originals":     wandb.Image(img_rgb, caption=cap),
            "prediction":    wandb.Image(overlay(img_rgb, mask,    RED),
                                         caption=f"{cap}  |  raw prediction"),
            "prediction_pp": wandb.Image(overlay(img_rgb, mask_pp, RED),
                                         caption=f"{cap}  |  postprocessed prediction"),
        }

        # If GT mask exists, log isolated TP/FP/FN panels + combined views, both
        # for the raw mask and the postprocessed mask, so you can directly compare.
        gt_path = img_to_mask.get(img_path)
        if gt_path:
            gt = imread(gt_path) > 0                          # (H, W) bool ground truth
            log_payload.update({
                "gt":          wandb.Image(overlay(img_rgb, gt, GREEN),
                                           caption=f"{cap}  |  green = ground-truth vessel"),
                "tp":          wandb.Image(overlay(img_rgb, mask    &  gt, GREEN),
                                           caption=f"{cap}  |  TP (raw)"),
                "fp":          wandb.Image(overlay(img_rgb, mask    & ~gt, RED),
                                           caption=f"{cap}  |  FP (raw)"),
                "fn":          wandb.Image(overlay(img_rgb, ~mask   &  gt, CYAN),
                                           caption=f"{cap}  |  FN (raw)"),
                "combined":    wandb.Image(make_combined(mask, gt),
                                           caption=f"{cap}  |  raw  |  green=TP  red=FP  cyan=FN"),
                "tp_pp":       wandb.Image(overlay(img_rgb, mask_pp &  gt, GREEN),
                                           caption=f"{cap}  |  TP (postprocessed)"),
                "fp_pp":       wandb.Image(overlay(img_rgb, mask_pp & ~gt, RED),
                                           caption=f"{cap}  |  FP (postprocessed)"),
                "fn_pp":       wandb.Image(overlay(img_rgb, ~mask_pp &  gt, CYAN),
                                           caption=f"{cap}  |  FN (postprocessed)"),
                "combined_pp": wandb.Image(make_combined(mask_pp, gt),
                                           caption=f"{cap}  |  postprocessed  |  green=TP  red=FP  cyan=FN"),
            })

        wandb.log(log_payload)


    wandb.finish()   # marks predict run complete in the dashboard


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True)
    parser.add_argument('--ckpt_path',  required=True)
    parser.add_argument('--out_dir',    default='../predictions')
    parser.add_argument('--mask_dir',   default=None,
                        help='Optional ground-truth mask folder. If given, log a per-image '
                             'TP/FP/FN error map (green/red/yellow) alongside the prediction.')
    args = parser.parse_args()
    main(args)

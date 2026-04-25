import os
import glob
import argparse
import numpy as np
import torch
from tifffile import imread, imwrite
from tqdm import tqdm

from dataset  import normalize, tile_image, stitch_tiles
from model    import VesselSegNet
from postprocess import postprocess


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
    img_f32  = img_u8.astype(np.float32) / 255.0
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

            logits = model(t)
            prob   = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)
            tile_probs.append(prob)
            tile_xs.append(x_off)

    # Stitch + threshold
    mask = stitch_tiles(tile_probs, tile_xs, H, W)
    mask = postprocess(mask)
    return mask


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = VesselSegNet().to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    img_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.tif')))
    print(f'Images to predict: {len(img_paths)}')

    os.makedirs(args.out_dir, exist_ok=True)

    for img_path in tqdm(img_paths, desc='images'):
        fname = os.path.basename(img_path).replace('.tif', '_pred.tif')
        out_path = os.path.join(args.out_dir, fname)

        mask = predict_image(model, img_path, device)

        # Save as uint8 binary (0 / 255) to match your existing mask format
        imwrite(out_path, (mask.astype(np.uint8) * 255))
        print(f'  saved → {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True)
    parser.add_argument('--ckpt_path',  required=True)
    parser.add_argument('--out_dir',    default='../predictions')
    args = parser.parse_args()
    main(args)

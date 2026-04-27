import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

from dataset import load_pairs, VesselDataset
from model   import VesselSegNet
from loss    import VesselLoss


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dice metric ────────────────────────────────────────────────────────────────

def dice_score(pred_bin, target):
    """pred_bin, target: (B, 1, H, W) bool tensors"""
    inter = (pred_bin & target).float().sum()
    denom = pred_bin.float().sum() + target.float().sum()
    return (2.0 * inter / (denom + 1e-6)).item()


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, msk, hann in tqdm(loader, leave=False):
            img  = img.to(device)
            msk  = msk.to(device)
            hann = hann.to(device)

            # First pass without graph (graph requires coarse mask)
            logits = model(img)

            loss, loss_dict = criterion(logits, msk, hann)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            pred_bin = (torch.sigmoid(logits) > 0.5)
            total_dice += dice_score(pred_bin, msk.bool())
            total_loss += loss.item()

    n = len(loader)
    return total_loss / n, total_dice / n


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load all annotated pairs
    pairs = load_pairs(args.input_dir, args.output_dir)
    print(f'Annotated pairs found: {len(pairs)}')

    # Build full dataset (pre-tiled)
    full_ds = VesselDataset(pairs, augment=False)
    print(f'Total tiles: {len(full_ds)}')

    # 5-fold CV on image-level pairs (not tile-level)
    # We split pairs first to avoid data leakage across folds
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    pair_indices = list(range(len(pairs)))

    best_dice_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(pair_indices)):
        print(f'\n═══ Fold {fold+1}/5 ═══')

        train_pairs = [pairs[i] for i in train_idx]
        val_pairs   = [pairs[i] for i in val_idx]

        train_ds = VesselDataset(train_pairs, augment=True,  seed=42)
        val_ds   = VesselDataset(val_pairs,   augment=False, seed=42)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        model     = VesselSegNet().to(device)
        criterion = VesselLoss(lambda_cldice=0.5)

        # Only train decoder + graph net — encoder is frozen inside model
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)

        best_dice = 0.0
        patience  = 20
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_dice = run_epoch(model, train_loader, criterion,
                                         optimizer, device, train=True)
            va_loss, va_dice = run_epoch(model, val_loader,   criterion,
                                         optimizer, device, train=False)
            scheduler.step()

            print(f'Epoch {epoch:3d} | '
                  f'train loss {tr_loss:.4f} dice {tr_dice:.4f} | '
                  f'val loss {va_loss:.4f} dice {va_dice:.4f}')

            # Save best checkpoint for this fold
            if va_dice > best_dice:
                best_dice  = va_dice
                no_improve = 0
                ckpt_path  = os.path.join(args.ckpt_dir, f'fold{fold+1}_best.pth')
                torch.save(model.state_dict(), ckpt_path)
                print(f'  ✓ saved {ckpt_path}  (dice {best_dice:.4f})')
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'  Early stopping at epoch {epoch}')
                    break

        best_dice_folds.append(best_dice)
        print(f'Fold {fold+1} best val Dice: {best_dice:.4f}')

    print(f'\n5-fold CV Dice: {np.mean(best_dice_folds):.4f} '
          f'± {np.std(best_dice_folds):.4f}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--ckpt_dir',   default='../checkpoints')
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr',         type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)

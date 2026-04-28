import os
import argparse
import random  # used in set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
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
    torch.backends.cudnn.deterministic = True  # reproducible GPU ops
    torch.backends.cudnn.benchmark     = False # disable auto-tuner (picks different algos each run)


# ── Dice metric ────────────────────────────────────────────────────────────────

def dice_score(pred_bin, target):
    """pred_bin, target: (B, 1, H, W) bool tensors"""
    inter = (pred_bin & target).float().sum()
    denom = pred_bin.float().sum() + target.float().sum()
    return (2.0 * inter / (denom + 1e-6)).item()


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    total_loss  = 0.0
    total_dice  = 0.0
    total_parts = {'dice': 0.0, 'bce': 0.0, 'cldice': 0.0}

    use_amp = device.type == 'cuda'
    # Select gradient context: enable for training (backprop), disable for val/test (saves memory).
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, msk, hann in tqdm(loader, leave=False):
            img  = img.to(device)
            msk  = msk.to(device)
            hann = hann.to(device)

            with autocast(enabled=use_amp):
                logits = model(img)
                loss, loss_dict = criterion(logits, msk, hann)

            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()       # scale loss to prevent float16 underflow, then backprop
                    scaler.unscale_(optimizer)          # restore true gradient magnitudes before clipping
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent gradient explosion
                    scaler.step(optimizer)              # skips update if gradients contain inf/nan
                    scaler.update()                     # adjust scale factor for next iteration
                else:
                    loss.backward()                                        # compute gradients
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent gradient explosion
                    optimizer.step()                                       # update weights

            pred_bin = (torch.sigmoid(logits) > 0.5)
            total_dice += dice_score(pred_bin, msk.bool())
            total_loss += loss.item()
            for k in total_parts:
                total_parts[k] += loss_dict[k]

    n = len(loader)
    parts = {k: v / n for k, v in total_parts.items()}
    return total_loss / n, total_dice / n, parts


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load all annotated pairs
    pairs = load_pairs(args.input_dir, args.output_dir)
    print(f'Annotated pairs found: {len(pairs)}')

    # Hold out 20% as a final test set — evaluated once after all CV folds,
    # never used during training or model selection to avoid optimism bias.
    trainval_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    print(f'Train+val pairs: {len(trainval_pairs)}  |  Test pairs: {len(test_pairs)}')

    # 5-fold CV on image-level pairs (not tile-level) to avoid data leakage
    kf           = KFold(n_splits=5, shuffle=True, random_state=42)
    pair_indices = list(range(len(trainval_pairs)))
    best_dice_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(pair_indices)):
        # Reset seed per fold so each fold's model initialises from the same state.
        set_seed(42 + fold)
        print(f'\n═══ Fold {fold+1}/5 ═══')

        train_pairs = [trainval_pairs[i] for i in train_idx]
        val_pairs   = [trainval_pairs[i] for i in val_idx]

        train_ds = VesselDataset(train_pairs, augment=True,  seed=42)
        val_ds   = VesselDataset(val_pairs,   augment=False, seed=42)
        print(f'  Train tiles: {len(train_ds)}  |  Val tiles: {len(val_ds)}')

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=4, pin_memory=True,
                                  persistent_workers=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True,
                                  persistent_workers=True)

        model     = VesselSegNet().to(device)
        criterion = VesselLoss(lambda_cldice=0.5)
        scaler    = GradScaler() if device.type == 'cuda' else None

        # Only train decoder + graph net — encoder is frozen inside model
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)

        best_dice  = 0.0
        # Patience of 30: cosine LR needs time to decay before judging convergence.
        patience   = 30
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            train_ds.seed = 42 + epoch  # vary augmentations each epoch

            tr_loss, tr_dice, tr_parts = run_epoch(model, train_loader, criterion,
                                                    optimizer, scaler, device, train=True)
            va_loss, va_dice, va_parts = run_epoch(model, val_loader,   criterion,
                                                    None,      None,   device, train=False)
            scheduler.step()

            print(f'Epoch {epoch:3d} | '
                  f'train loss {tr_loss:.4f} dice {tr_dice:.4f} '
                  f'(d={tr_parts["dice"]:.3f} b={tr_parts["bce"]:.3f} cl={tr_parts["cldice"]:.3f}) | '
                  f'val   loss {va_loss:.4f} dice {va_dice:.4f} '
                  f'(d={va_parts["dice"]:.3f} b={va_parts["bce"]:.3f} cl={va_parts["cldice"]:.3f})')

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

    # ── Final test evaluation ──────────────────────────────────────────────────
    # Load the best fold's checkpoint and evaluate once on the held-out test set.
    best_fold = int(np.argmax(best_dice_folds)) + 1
    print(f'\nEvaluating fold {best_fold} (best CV) on held-out test set ...')
    set_seed(42)

    test_ds     = VesselDataset(test_pairs, augment=False, seed=42)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True,
                             persistent_workers=True)

    model = VesselSegNet().to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f'fold{best_fold}_best.pth'),
                                     map_location=device))
    criterion = VesselLoss(lambda_cldice=0.5)

    te_loss, te_dice, te_parts = run_epoch(model, test_loader, criterion,
                                            None, None, device, train=False)
    print(f'Test  loss {te_loss:.4f} dice {te_dice:.4f} '
          f'(d={te_parts["dice"]:.3f} b={te_parts["bce"]:.3f} cl={te_parts["cldice"]:.3f})')


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

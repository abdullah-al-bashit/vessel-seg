import os
import json
import time
import argparse
import random  # used in set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from tifffile import imread as tif_imread

from dataset import load_pairs, VesselDataset
from model   import VesselSegNet
from loss    import VesselLoss
from predict import predict_image  # full-image stitched inference for test Dice
import wandb

SEED          = 42   # single source of truth for all random seeds throughout training
N_FOLDS       = 5    # number of cross-validation folds
TEST_SPLIT    = 0.2  # fraction of pairs held out as final test set
PATIENCE      = 30   # early stopping: epochs without val loss improvement before stopping
LAMBDA_CLDICE = 0.5  # weight of clDice term in VesselLoss (total = soft_dice + bce + λ·cldice)
NUM_WORKERS   = 4    # DataLoader worker processes for parallel data loading


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # reproducible GPU ops
    torch.backends.cudnn.benchmark     = False # disable auto-tuner (picks different algos each run)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    total_loss  = 0.0
    total_parts = {'dice': 0.0, 'bce': 0.0, 'cldice': 0.0}

    use_amp = device.type == 'cuda'
    # Select gradient context: enable for training (backprop), disable for val/test (saves memory).
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, msk, hann in tqdm(loader):  # batch loop — each iteration yields (B, 1, H, W) tensors
            img  = img.to(device)
            msk  = msk.to(device)
            hann = hann.to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(img)
                loss, loss_dict = criterion(logits, msk, hann)

            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()       # multiply loss by scale factor, then backprop to keep float16 gradients from flushing to zero
                    scaler.unscale_(optimizer)          # divide gradients back by scale factor so clip operates on true magnitudes
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # cap gradient norm to avoid exploding updates
                    scaler.step(optimizer)              # update weights; skips entire step if any gradient is inf/nan
                    scaler.update()                     # increase scale factor if step was taken, decrease if inf/nan was detected
                else:
                    loss.backward()                                        # compute gradients
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent gradient explosion
                    optimizer.step()                                       # update weights

            total_loss += loss.item()                      # scalar float — .item() detaches from graph
            for k in total_parts:
                total_parts[k] += loss_dict[k]  # scalar float — loss_dict values are already Python floats via .item() in VesselLoss

    n = len(loader)                                        # number of batches in the epoch
    parts = {k: v / n for k, v in total_parts.items()}    # per-sub-loss epoch averages
    return total_loss / n, parts                           # epoch-averaged total loss and sub-losses


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(SEED)
    # MPS = Apple Silicon GPU (Metal); falls back to CPU on Intel Macs.
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    # Load all annotated pairs
    pairs = load_pairs(args.input_dir, args.output_dir)  # e.g. [('data/1_img.tif', 'data/1_msk.tif'), ('data/2_img.tif', 'data/2_msk.tif'), ...]
    print(f'Annotated pairs found: {len(pairs)}')

    # Hold out 20% as a final test set — evaluated once after all CV folds,
    # never used during training or model selection to avoid optimism bias.
    trainval_pairs, test_pairs = train_test_split(pairs, test_size=TEST_SPLIT, random_state=SEED)
    # e.g. pairs=30 → trainval_pairs=[('1_img.tif','1_msk.tif'), ...] (24 items), test_pairs=[('7_img.tif','7_msk.tif'), ...] (6 items)
    print(f'Train+val pairs: {len(trainval_pairs)}  |  Test pairs: {len(test_pairs)}')

    # ── W&B run — one run covers all folds + final test ───────────────────────
    wandb.init(
        entity  = "eeebashit",
        project = "vessel-seg",
        config  = {
            "seed":           SEED,
            "n_folds":        N_FOLDS,
            "test_split":     TEST_SPLIT,
            "patience":       PATIENCE,
            "lambda_cldice":  LAMBDA_CLDICE,
            "epochs":         args.epochs,
            "batch_size":     args.batch_size,
            "lr":             args.lr,
        }
    )

    # 5-fold CV on image-level pairs (not tile-level) to avoid data leakage
    kf           = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)  # splitter object — actual split happens at kf.split() below
    pair_indices = list(range(len(trainval_pairs)))  # e.g. [0, 1, 2, ..., 23] for 24 trainval pairs
    best_loss_folds = []                             # best val loss per fold — used to select fold for test evaluation

    # Log every file's role (fold number + train/val/test) in one Table.
    # wandb.Table stores string data properly — wandb.log() with strings or lists
    # would be treated as metrics and silently dropped or misrepresented.
    # fold=0 marks the held-out test set (never used during training).
    split_table = wandb.Table(columns=["fold", "split", "file"])
    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(pair_indices)):
        for i in tr_idx:
            split_table.add_data(fold_i + 1, "train", trainval_pairs[i][0])
        for i in va_idx:
            split_table.add_data(fold_i + 1, "val",   trainval_pairs[i][0])
    for p in test_pairs:
        split_table.add_data(0, "test", p[0])
    wandb.log({"data_splits": split_table})

    # Save filename → split label so predict.py can label each image in the W&B
    # Media panel without needing to know which folder is test vs trainval.
    splits_info = {os.path.basename(p[0]): "trainval" for p in trainval_pairs}
    splits_info.update({os.path.basename(p[0]): "test" for p in test_pairs})
    with open(os.path.join(args.ckpt_dir, "data_splits.json"), "w") as f:
        json.dump(splits_info, f, indent=2)

    for fold, (train_idx, val_idx) in enumerate(kf.split(pair_indices)):
        # --folds lets you stop after the first N folds (still uses 5-way split → 80/20 ratio).
        # e.g. --folds 1 trains exactly one fold on 80% of trainval, validated on the remaining 20%.
        if fold >= args.folds:
            break
        # kf.split yields 5 rounds; each round rotates which 1/5 of pair_indices is val_idx,
        # the other 4/5 become train_idx — e.g. fold 0: val_idx=[0..4], train_idx=[5..23]

        # reset RNG so every fold's model starts from the same weight initialisation
        set_seed(SEED + fold)
        print(f'\n═══ Fold {fold+1}/{N_FOLDS} ═══')

        # map integer indices back to actual (img_path, msk_path) tuples
        train_pairs = [trainval_pairs[i] for i in train_idx]  # e.g. 19 pairs
        val_pairs   = [trainval_pairs[i] for i in val_idx]    # e.g. 5 pairs

        # tile each pair; train set gets augmentation, val set does not
        train_ds = VesselDataset(train_pairs, augment=True,  seed=SEED)
        val_ds   = VesselDataset(val_pairs,   augment=False, seed=SEED)
        print(f'  Train tiles: {len(train_ds)}  |  Val tiles: {len(val_ds)}')

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=NUM_WORKERS,
                                  pin_memory=True,         # allocates batches in CPU memory that the GPU can read directly, avoiding an extra copy
                                  persistent_workers=True) # keeps worker processes alive between epochs so they don't need to be spawned again each epoch
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=True,
                                  persistent_workers=True)

        model     = VesselSegNet().to(device)
        criterion = VesselLoss(lambda_cldice=LAMBDA_CLDICE)
        scaler    = GradScaler() if device.type == 'cuda' else None

        # Only train decoder + graph net — encoder is frozen inside model
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)

        best_loss  = float('inf')  # best val loss this fold (full VesselLoss = soft_dice + bce + cldice)
        no_improve = 0             # counter — resets to 0 whenever val loss improves, increments otherwise

        for epoch in range(1, args.epochs + 1):
            train_ds.seed = SEED + epoch  # vary augmentations each epoch

            # Wall-clock per-phase timing — used for accurate runtime estimation
            t_train_start = time.perf_counter()
            tr_loss, tr_parts = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, device, train=True)
            t_train = time.perf_counter() - t_train_start

            t_val_start = time.perf_counter()
            va_loss, va_parts = run_epoch(model, val_loader,   criterion,
                                          None,    None,       device, train=False)
            t_val = time.perf_counter() - t_val_start
            scheduler.step()

            t_epoch = t_train + t_val

            print(f'Epoch {epoch:3d} | '
                  f'train loss {tr_loss:.4f} '
                  f'(soft_dice={tr_parts["dice"]:.3f} bce={tr_parts["bce"]:.3f} cldice={tr_parts["cldice"]:.3f}) | '
                  f'val   loss {va_loss:.4f} '
                  f'(soft_dice={va_parts["dice"]:.3f} bce={va_parts["bce"]:.3f} cldice={va_parts["cldice"]:.3f}) | '
                  f'time {t_epoch:.1f}s (train {t_train:.1f}s, val {t_val:.1f}s)')

            # fold-prefixed metrics so all 5 folds are visible as separate curves in W&B
            wandb.log({
                "epoch":                         epoch,
                f"fold{fold+1}/train_loss":      tr_loss,
                f"fold{fold+1}/train_dice":      tr_parts["dice"],
                f"fold{fold+1}/train_bce":       tr_parts["bce"],
                f"fold{fold+1}/train_cldice":    tr_parts["cldice"],
                f"fold{fold+1}/val_loss":        va_loss,
                f"fold{fold+1}/val_dice":        va_parts["dice"],
                f"fold{fold+1}/val_bce":         va_parts["bce"],
                f"fold{fold+1}/epoch_time_s":    t_epoch,
                f"fold{fold+1}/train_time_s":    t_train,
                f"fold{fold+1}/val_time_s":      t_val,
                f"fold{fold+1}/val_cldice":      va_parts["cldice"],
            })

            # model selection and early stopping based on full val loss (VesselLoss)
            if va_loss < best_loss:
                best_loss  = va_loss
                no_improve = 0
                ckpt_path  = os.path.join(args.ckpt_dir, f'fold{fold+1}_best.pth')
                torch.save(model.state_dict(), ckpt_path)
                print(f'  ✓ saved {ckpt_path}  (val loss {best_loss:.4f})')
                # Artifact: uploads the checkpoint file to W&B so it's stored with
                # this run and downloadable by name — no need to track file paths manually
                artifact = wandb.Artifact(f"fold{fold+1}_best", type="model",
                                          description=f"fold{fold+1} best checkpoint, val_loss={best_loss:.4f}")
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f'  Early stopping at epoch {epoch}')
                    break

        best_loss_folds.append(best_loss)
        print(f'Fold {fold+1} best val loss: {best_loss:.4f}')
        wandb.log({f"fold{fold+1}/best_val_loss": best_loss})

    print(f'\n5-fold CV val loss: {np.mean(best_loss_folds):.4f} '
          f'± {np.std(best_loss_folds):.4f}')

    # ── Final test evaluation ──────────────────────────────────────────────────
    # Load the fold with the lowest val loss and evaluate once on the held-out test set.
    best_fold = int(np.argmin(best_loss_folds)) + 1
    print(f'\nEvaluating fold {best_fold} (lowest val loss) on held-out test set ...')
    set_seed(SEED)

    test_ds     = VesselDataset(test_pairs, augment=False, seed=SEED)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=True)

    model = VesselSegNet().to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f'fold{best_fold}_best.pth'),
                                     map_location=device))
    criterion = VesselLoss(lambda_cldice=LAMBDA_CLDICE)

    te_loss, te_parts = run_epoch(model, test_loader, criterion,
                                   None, None, device, train=False)
    print(f'Test loss {te_loss:.4f} '
          f'(soft_dice={te_parts["dice"]:.3f} bce={te_parts["bce"]:.3f} cldice={te_parts["cldice"]:.3f})')

    # ── Stitched-image Dice on test set (interpretable 0-1 metric) ────────────
    # te_loss above is computed on tile-level patches with Hanning weighting,
    # so its components fall outside [0, 1]. To report a standard Dice score,
    # run inference on each full test image (predict_image stitches tiles back
    # into a (H, W) binary mask) and compare against the ground-truth mask.
    print('\nStitched-image Dice on test set:')
    dice_per_image = []
    for img_path, msk_path in test_pairs:
        pred_mask = predict_image(model, img_path, device)        # (H, W) bool
        gt_mask   = tif_imread(msk_path) > 0                       # (H, W) bool
        inter     = float(np.logical_and(pred_mask, gt_mask).sum())
        denom     = float(pred_mask.sum() + gt_mask.sum())
        dice      = (2.0 * inter) / (denom + 1e-8)                 # standard Dice ∈ [0, 1]
        dice_per_image.append(dice)
        print(f'  {os.path.basename(img_path):40s} dice = {dice:.4f}')
    mean_test_dice = float(np.mean(dice_per_image))
    print(f'Mean test Dice (stitched, unweighted): {mean_test_dice:.4f}')

    wandb.log({
        "test/loss":             te_loss,
        "test/dice":             te_parts["dice"],
        "test/bce":              te_parts["bce"],
        "test/cldice":           te_parts["cldice"],
        "test/dice_stitched":    mean_test_dice,        # standard Dice ∈ [0, 1] — clinically interpretable
        "best_fold":             best_fold,
        "cv_mean_val_loss":      float(np.mean(best_loss_folds)),
        "cv_std_val_loss":       float(np.std(best_loss_folds)),
    })
    wandb.finish()   # marks the run complete and uploads any remaining data


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--ckpt_dir',   default='../checkpoints')
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--folds',      type=int, default=N_FOLDS,
                        help='Train at most this many folds (still uses 5-way 80/20 split)')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr',         type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)

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

# All training hyperparameters are loaded from the YAML config file (--config).
# See configs/exp_*.yaml for the full parameter set and documentation.


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # reproducible GPU ops
    torch.backends.cudnn.benchmark     = False # disable auto-tuner (picks different algos each run)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True,
              feat_capture=None, use_graph=None, lambda_coarse=0.4):
    """
    feat_capture: optional dict populated by a forward hook (see main()) that
                  holds the decoder features under key 'feats'. Passed to
                  criterion only during training so the contrastive loss can
                  use them. Val/test pass None to skip that compute.
    """
    if use_graph is None:
        use_graph = train  # default: graph during training, disabled during val
    model.train(train)
    total_loss  = 0.0
    total_parts = {'dice': 0.0, 'bce': 0.0, 'cldice': 0.0, 'boundary': 0.0, 'contrast': 0.0, 'coarse': 0.0}

    use_amp = device.type == 'cuda'
    # Select gradient context: enable for training (backprop), disable for val/test (saves memory).
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, msk, hann, sharp, grad in tqdm(loader):
            img   = img.to(device)
            msk   = msk.to(device)
            hann  = hann.to(device)
            sharp = sharp.to(device)
            grad  = grad.to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                logits, coarse_logits = model(img, use_graph=use_graph, sharpness=sharp, grad_mag=grad)
                # Pass captured features to criterion only during training; val
                # / test gets feats=None so contrastive is skipped.
                feats  = feat_capture.get('feats') if (train and feat_capture is not None) else None
                loss, loss_dict = criterion(logits, msk, hann, sharpness=sharp, feats=feats)
                if train and coarse_logits is not None:
                    loss_coarse, _ = criterion(coarse_logits, msk, hann, sharpness=sharp)
                    loss = loss + lambda_coarse * loss_coarse
                    loss_dict['coarse'] = loss_coarse.item()
                else:
                    loss_dict['coarse'] = 0.0

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
    set_seed(args.seed)
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

    # Hold out test_split fraction as final test — evaluated once, never touches training.
    trainval_pairs, test_pairs = train_test_split(pairs, test_size=args.test_split, random_state=args.seed)
    # e.g. pairs=30 → trainval_pairs=[('1_img.tif','1_msk.tif'), ...] (24 items), test_pairs=[('7_img.tif','7_msk.tif'), ...] (6 items)
    print(f'Train+val pairs: {len(trainval_pairs)}  |  Test pairs: {len(test_pairs)}')

    # ── W&B run — one run covers all folds + final test ───────────────────────
    # vars(args) logs every YAML/CLI parameter so each W&B run is fully reproducible.
    wandb.init(
        entity  = args.wandb_entity,
        project = args.wandb_project,
        config  = vars(args),
    )

    # n_folds-way CV on image-level pairs (not tile-level) to avoid data leakage
    kf           = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)  # splitter object — actual split happens at kf.split() below
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
        set_seed(args.seed + fold)
        print(f'\n═══ Fold {fold+1}/{args.n_folds} ═══')

        # map integer indices back to actual (img_path, msk_path) tuples
        train_pairs = [trainval_pairs[i] for i in train_idx]  # e.g. 19 pairs
        val_pairs   = [trainval_pairs[i] for i in val_idx]    # e.g. 5 pairs

        # tile each pair; train set gets augmentation, val set does not
        train_ds = VesselDataset(train_pairs, augment=True,  seed=args.seed,
                                 sharp_hann=not args.plain_hann,
                                 blur_prob=args.blur_prob,
                                 blur_sigma_max=args.blur_sigma_max)
        val_ds   = VesselDataset(val_pairs,   augment=False, seed=args.seed,
                                 sharp_hann=not args.plain_hann)
        print(f'  Train tiles: {len(train_ds)}  |  Val tiles: {len(val_ds)}')

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.num_workers,
                                  pin_memory=True,         # allocates batches in CPU memory that the GPU can read directly, avoiding an extra copy
                                  persistent_workers=True) # keeps worker processes alive between epochs so they don't need to be spawned again each epoch
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True,
                                  persistent_workers=True)

        model     = VesselSegNet(sam2_model=args.sam2_model).to(device)

        # Forward hook registered before compile so it fires on the underlying module.
        captured = {}
        def _capture_features(_module, inputs, _output):
            captured['feats'] = inputs[0]
        feat_hook = model.head.register_forward_hook(_capture_features)  # removed at end of fold

        criterion = VesselLoss(lambda_cldice=args.lambda_cldice,
                               lambda_boundary=args.lambda_boundary,
                               lambda_contrast=args.lambda_contrast,
                               hard_neg_factor=args.hard_neg_factor)
        scaler    = GradScaler() if device.type == 'cuda' else None

        # Only train decoder + graph net — encoder is frozen inside model
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)

        best_loss  = float('inf')  # best val loss this fold (full VesselLoss = soft_dice + bce + cldice)
        no_improve = 0             # counter — resets to 0 whenever val loss improves, increments otherwise

        for epoch in range(1, args.epochs + 1):
            train_ds.seed = args.seed + epoch  # vary augmentations each epoch

            # CNN-only: graph path is slower and does not improve val Dice over pure CNN.
            use_graph = False

            # Wall-clock per-phase timing — used for accurate runtime estimation
            t_train_start = time.perf_counter()
            tr_loss, tr_parts = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, device, train=True,
                                          use_graph=use_graph,
                                          feat_capture=captured,
                                          lambda_coarse=args.lambda_coarse)
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
                f"fold{fold+1}/train_boundary":  tr_parts["boundary"],
                f"fold{fold+1}/train_contrast":  tr_parts["contrast"],
                f"fold{fold+1}/train_coarse":    tr_parts["coarse"],
                f"fold{fold+1}/val_loss":        va_loss,
                f"fold{fold+1}/val_dice":        va_parts["dice"],
                f"fold{fold+1}/val_bce":         va_parts["bce"],
                f"fold{fold+1}/val_cldice":      va_parts["cldice"],
                f"fold{fold+1}/epoch_time_s":    t_epoch,
                f"fold{fold+1}/train_time_s":    t_train,
                f"fold{fold+1}/val_time_s":      t_val,
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
                if no_improve >= args.patience:
                    print(f'  Early stopping at epoch {epoch}')
                    break

        best_loss_folds.append(best_loss)
        print(f'Fold {fold+1} best val loss: {best_loss:.4f}')
        wandb.log({f"fold{fold+1}/best_val_loss": best_loss})
        feat_hook.remove()  # detach the forward hook before next fold rebuilds the model

    print(f'\n{args.n_folds}-fold CV val loss: {np.mean(best_loss_folds):.4f} '
          f'± {np.std(best_loss_folds):.4f}')

    # ── Final test evaluation ──────────────────────────────────────────────────
    # Load the fold with the lowest val loss and evaluate once on the held-out test set.
    best_fold = int(np.argmin(best_loss_folds)) + 1
    print(f'\nEvaluating fold {best_fold} (lowest val loss) on held-out test set ...')
    set_seed(args.seed)

    test_ds     = VesselDataset(test_pairs, augment=False, seed=args.seed,
                                sharp_hann=not args.plain_hann)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True)

    model = VesselSegNet(sam2_model=args.sam2_model).to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f'fold{best_fold}_best.pth'),
                                     map_location=device))
    criterion = VesselLoss(lambda_cldice=args.lambda_cldice,
                           lambda_boundary=args.lambda_boundary,
                           lambda_contrast=args.lambda_contrast,
                           hard_neg_factor=args.hard_neg_factor)

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
    import yaml

    # Two-pass parse: first read --config path, load YAML, then set those values
    # as argparse defaults so any explicit CLI flag still overrides the config.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, _ = pre.parse_known_args()

    cfg = {}
    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser()
    # ── Paths (cluster-specific; always set via CLI or submit.sh) ─────────────
    parser.add_argument('--config',       default=None,  help='Path to YAML experiment config')
    parser.add_argument('--input_dir',    required=True)
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--ckpt_dir',     default='../checkpoints')
    # ── W&B ───────────────────────────────────────────────────────────────────
    parser.add_argument('--wandb_entity',  default='eeebashit')
    parser.add_argument('--wandb_project', default='vessel-seg')
    # ── Reproducibility ───────────────────────────────────────────────────────
    parser.add_argument('--seed',         type=int,   default=42)
    # ── Cross-validation ──────────────────────────────────────────────────────
    parser.add_argument('--n_folds',      type=int,   default=5)
    parser.add_argument('--test_split',   type=float, default=0.2)
    parser.add_argument('--folds',        type=int,   default=5,
                        help='Train at most this many folds (still uses n_folds-way 80/20 split)')
    parser.add_argument('--patience',     type=int,   default=30)
    # ── Training loop ─────────────────────────────────────────────────────────
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--batch_size',   type=int,   default=2)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--num_workers',  type=int,   default=8)
    # ── Loss weights ──────────────────────────────────────────────────────────
    parser.add_argument('--lambda_cldice',   type=float, default=1.0)
    parser.add_argument('--lambda_boundary', type=float, default=0.5)
    parser.add_argument('--lambda_contrast', type=float, default=0.1)
    parser.add_argument('--lambda_coarse',   type=float, default=0.4)
    parser.add_argument('--hard_neg_factor', type=float, default=2.0)
    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--sam2_model',   default='facebook/sam2.1-hiera-tiny')
    # ── Dataset / augmentation ────────────────────────────────────────────────
    parser.add_argument('--plain_hann',     action='store_true')
    parser.add_argument('--blur_prob',      type=float, default=0.3)
    parser.add_argument('--blur_sigma_max', type=float, default=4.0)
    # YAML config values become defaults; explicit CLI flags still override.
    # Exclude path args — those are always cluster-specific and set via CLI.
    parser.set_defaults(**{k: v for k, v in cfg.items()
                           if k not in ('input_dir', 'output_dir', 'ckpt_dir')})
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)

import os
import json
import time
import gc
import argparse
import random
import shutil
import numpy as np
import timm
import torch
import torch.nn as nn

from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from tifffile import imread as tif_imread

from dataset      import load_pairs, VesselDataset, collate_fn_with_filenames
from model        import AttentionUNet, visualize_attention_maps
from loss         import VesselLoss
from predict      import predict_image
from summarize_cv import print_cv_summary
import wandb


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # reproducible GPU ops
    torch.backends.cudnn.benchmark     = False # disable auto-tuner (picks different algos each run)


# ── Shared helpers ─────────────────────────────────────────────────────────────

_RESNET34_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'weights', 'resnet34_pretrained.pth'
)


def create_model(device, ckpt_path=None):
    model = AttentionUNet()
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) else ckpt)
        print(f'AttentionUNet loaded from checkpoint: {ckpt_path}')
    else:
        if os.path.exists(_RESNET34_WEIGHTS):
            print(f'Loading ResNet34 weights from {_RESNET34_WEIGHTS}')
            model.encoder.load_state_dict(torch.load(_RESNET34_WEIGHTS, map_location='cpu', weights_only=True))
        else:
            print('weights/resnet34_pretrained.pth not found — downloading from HuggingFace and saving locally...')
            enc = timm.create_model('resnet34', pretrained=True, features_only=True, in_chans=3, out_indices=(0,1,2,3,4))
            os.makedirs(os.path.dirname(_RESNET34_WEIGHTS), exist_ok=True)
            torch.save(enc.state_dict(), _RESNET34_WEIGHTS)
            model.encoder.load_state_dict(enc.state_dict())
            print(f'Saved → {_RESNET34_WEIGHTS}')
        print('AttentionUNet created: trainable ResNet34 encoder + UNet decoder with attention gates')
    return model.to(device)


def _cache_clear(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


def _save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, val_loss,
                     artifact_name, description, args=None):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict() if scaler else None,
        'val_loss':             val_loss,
        'config':               vars(args) if args is not None else {},
    }, path)
    artifact = wandb.Artifact(artifact_name, type="model", description=description)
    artifact.add_file(os.path.abspath(path))
    wandb.log_artifact(artifact)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True,
              epoch=None, fold=None):
    model.train(train)
    total_loss  = 0.0
    total_parts = {'tversky': 0.0, 'cldice': 0.0}
    dice_num    = 0.0
    dice_den    = 0.0

    use_amp = device.type == 'cuda'
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, batch_data in enumerate(tqdm(loader)):
            img    = batch_data[0].to(device)
            msk    = batch_data[1].to(device)
            hann   = batch_data[2].to(device)
            sharp  = batch_data[3].to(device)
            grad   = batch_data[4].to(device)
            fnames = batch_data[5]

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(img, sharpness=sharp, grad_mag=grad)
                loss, loss_dict, pred = criterion(logits, msk, hann)

            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()       # multiply loss by scale factor, then backprop to keep float16 gradients from flushing to zero
                    scaler.unscale_(optimizer)          # divide gradients back by scale factor so clip operates on true magnitudes
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)              # update weights; skips entire step if any gradient is inf/nan
                    scaler.update()                     # increase scale factor if step was taken, decrease if inf/nan was detected
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item()
            for k in total_parts:
                total_parts[k] += loss_dict[k]
            with torch.no_grad():
                pred_bin  = (pred.detach() > 0.5)
                dice_num += 2.0 * (pred_bin & msk.bool()).float().sum().item()
                dice_den += pred_bin.float().sum().item() + msk.float().sum().item()

            # Log prediction + attention maps every 10 val epochs on first batch
            if (not train) and (epoch is not None) and (fold is not None) and \
               (epoch % 10 == 0) and (batch_idx == 0):
                try:
                    fname   = fnames[0]
                    cap     = f"[val] {fname} | fold {fold+1} epoch {epoch}"
                    img_np  = img[0, 0].cpu().numpy()
                    lo, hi  = np.percentile(img_np, (1, 99))
                    img_u8  = np.clip((img_np - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
                    img_rgb = np.stack([img_u8] * 3, axis=-1)
                    prob_np  = pred[0, 0].detach().cpu().numpy()
                    gt_u8    = (msk[0, 0].cpu().numpy()   * 255).astype(np.uint8)
                    grad_u8  = (grad[0, 0].cpu().numpy()  * 255).astype(np.uint8)
                    sharp_u8 = (sharp[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    log_dict = {
                        f"fold{fold+1}/input_image": wandb.Image(img_u8,                                 caption=f"Input    | {cap}"),
                        f"fold{fold+1}/ch_grad":     wandb.Image(grad_u8,                                caption=f"Grad mag | {cap}"),
                        f"fold{fold+1}/ch_sharp":    wandb.Image(sharp_u8,                               caption=f"Sharpness| {cap}"),
                        f"fold{fold+1}/gt_mask":     wandb.Image(gt_u8,                                  caption=f"GT       | {cap}"),
                        f"fold{fold+1}/pred_prob":   wandb.Image((prob_np * 255).astype(np.uint8),       caption=f"P(vessel)| {cap}"),
                        f"fold{fold+1}/pred_binary": wandb.Image((prob_np > 0.5).astype(np.uint8) * 255, caption=f"Pred>0.5 | {cap}"),
                    }
                    attn_panels = visualize_attention_maps(model, img_rgb)
                    log_dict.update({f"fold{fold+1}/{k}": v for k, v in attn_panels.items()})
                    wandb.log(log_dict)
                    print(f'  ✓ logged val visualizations ({cap})')
                except Exception as e:
                    print(f'  ⚠ visualization logging failed: {e}')

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in total_parts.items()}, dice_num / (dice_den + 1e-8)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    pairs = load_pairs(args.input_dir, args.output_dir)
    print(f'Annotated pairs found: {len(pairs)}')

    trainval_pairs, test_pairs = train_test_split(
        pairs, test_size=args.test_split, random_state=args.seed)
    print(f'Train+val pairs: {len(trainval_pairs)}  |  Test pairs: {len(test_pairs)}')

    wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=vars(args))

    kf           = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pair_indices = list(range(len(trainval_pairs)))
    fold_splits  = list(kf.split(pair_indices))

    # Log every file's role (fold + train/val/test) in one Table.
    split_table = wandb.Table(columns=["fold", "split", "file"])
    for fold_i, (tr_idx, va_idx) in enumerate(fold_splits):
        for i in tr_idx:
            split_table.add_data(fold_i + 1, "train", trainval_pairs[i][0])
        for i in va_idx:
            split_table.add_data(fold_i + 1, "val",   trainval_pairs[i][0])
    for p in test_pairs:
        split_table.add_data(0, "test", p[0])
    wandb.log({"data_splits": split_table})

    splits_info = {**{os.path.basename(p[0]): "trainval" for p in trainval_pairs},
                   **{os.path.basename(p[0]): "test"     for p in test_pairs}}
    with open(os.path.join(args.ckpt_dir, "data_splits.json"), "w") as f:
        json.dump(splits_info, f, indent=2)

    best_loss_folds  = []
    best_epoch_folds = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        if fold >= args.folds:
            break

        wandb.define_metric(f"fold{fold+1}/epoch", hidden=True)
        wandb.define_metric(f"fold{fold+1}/*",      step_metric=f"fold{fold+1}/epoch")

        set_seed(args.seed + fold)
        print(f'\n═══ Fold {fold+1}/{args.folds} ═══')

        train_pairs = [trainval_pairs[i] for i in train_idx]
        val_pairs   = [trainval_pairs[i] for i in val_idx]

        train_ds = VesselDataset(train_pairs, augment=True,  seed=args.seed,
                                 sharp_hann=not args.plain_hann,
                                 blur_prob=args.blur_prob,
                                 blur_sigma_max=args.blur_sigma_max)
        val_ds   = VesselDataset(val_pairs,   augment=False, seed=args.seed,
                                 sharp_hann=not args.plain_hann)
        print(f'  Train tiles: {len(train_ds)}  |  Val tiles: {len(val_ds)}')

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,          # allocates batches in CPU memory that the GPU can read directly
                                  persistent_workers=True,  # keeps workers alive between epochs
                                  collate_fn=collate_fn_with_filenames)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  collate_fn=collate_fn_with_filenames)

        model = create_model(device, ckpt_path=args.warmstart_ckpt or None)
        criterion = VesselLoss(lambda_tversky=args.lambda_tversky,
                               lambda_cldice=args.lambda_cldice,
                               tversky_beta=args.tversky_beta)
        scaler    = GradScaler() if device.type == 'cuda' else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)

        best_loss  = float('inf')
        best_epoch = 0
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            train_ds.seed = args.seed + epoch

            t0 = time.perf_counter()
            tr_loss, tr_parts, tr_dice = run_epoch(model, train_loader, criterion,
                                                    optimizer, scaler, device, train=True)
            t1 = time.perf_counter()
            va_loss, va_parts, va_dice = run_epoch(model, val_loader, criterion,
                                                    None, None, device, train=False,
                                                    epoch=epoch, fold=fold)
            scheduler.step()
            t_epoch = time.perf_counter() - t0

            print(f'Epoch {epoch:3d} | '
                  f'train {tr_loss:.4f} dice={tr_dice:.4f} '
                  f'(tv={tr_parts["tversky"]:.3f} cl={tr_parts["cldice"]:.3f}) | '
                  f'val {va_loss:.4f} dice={va_dice:.4f} '
                  f'(tv={va_parts["tversky"]:.3f} cl={va_parts["cldice"]:.3f}) | '
                  f'{t_epoch:.1f}s (train {t1-t0:.1f}s)')

            wandb.log({
                f"fold{fold+1}/epoch":         epoch,
                f"fold{fold+1}/train_loss":    tr_loss,
                f"fold{fold+1}/train_dice":    tr_dice,
                f"fold{fold+1}/train_tversky": tr_parts["tversky"],
                f"fold{fold+1}/train_cldice":  tr_parts["cldice"],
                f"fold{fold+1}/val_loss":      va_loss,
                f"fold{fold+1}/val_dice":      va_dice,
                f"fold{fold+1}/val_tversky":   va_parts["tversky"],
                f"fold{fold+1}/val_cldice":    va_parts["cldice"],
            })

            if va_loss < best_loss:
                best_loss  = va_loss
                best_epoch = epoch
                no_improve = 0
                ckpt_path  = os.path.join(args.ckpt_dir, f'fold{fold+1}_best.pth')
                _save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, scaler,
                                 best_loss, f"fold{fold+1}_best",
                                 f"fold{fold+1} best checkpoint, val_loss={best_loss:.4f}", args)
                print(f'  ✓ saved {ckpt_path}  (val loss {best_loss:.4f})')
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f'  Early stopping at epoch {epoch}')
                    break

        best_loss_folds.append(best_loss)
        best_epoch_folds.append(best_epoch)
        print(f'Fold {fold+1} best val loss: {best_loss:.4f} (epoch {best_epoch})')
        wandb.log({f"fold{fold+1}/best_val_loss": best_loss,
                   f"fold{fold+1}/best_epoch":    best_epoch})

        _cache_clear(device)

    # ── Select best fold and promote checkpoint ────────────────────────────────
    best_fold = int(np.argmin(best_loss_folds)) + 1
    print(f'\nBest fold: {best_fold}  (val loss {best_loss_folds[best_fold-1]:.4f}, '
          f'epoch {best_epoch_folds[best_fold-1]})')

    shutil.copy(os.path.join(args.ckpt_dir, f'fold{best_fold}_best.pth'),
                os.path.join(args.ckpt_dir, 'best_model.pth'))
    for fold_num in range(1, args.folds + 1):
        p = os.path.join(args.ckpt_dir, f'fold{fold_num}_best.pth')
        if os.path.exists(p):
            os.remove(p)

    # ── Final test evaluation ──────────────────────────────────────────────────
    set_seed(args.seed)
    test_ds     = VesselDataset(test_pairs, augment=False, seed=args.seed,
                                sharp_hann=not args.plain_hann)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True, collate_fn=collate_fn_with_filenames)

    model = create_model(device, ckpt_path=os.path.join(args.ckpt_dir, 'best_model.pth'))
    criterion = VesselLoss(lambda_tversky=args.lambda_tversky,
                           lambda_cldice=args.lambda_cldice,
                           tversky_beta=args.tversky_beta)

    te_loss, te_parts, te_dice = run_epoch(model, test_loader, criterion,
                                           None, None, device, train=False)
    print(f'Test loss {te_loss:.4f} dice={te_dice:.4f} '
          f'(tv={te_parts["tversky"]:.3f} cl={te_parts["cldice"]:.3f})')

    print('\nStitched-image Dice on test set:')
    dice_per_image = []
    for img_path, msk_path in test_pairs:
        mask, _ = predict_image(model, img_path, device)
        gt_mask  = tif_imread(msk_path) > 0
        inter    = float(np.logical_and(mask, gt_mask).sum())
        denom    = float(mask.sum() + gt_mask.sum())
        dice     = (2.0 * inter) / (denom + 1e-8)
        dice_per_image.append(dice)
        print(f'  {os.path.basename(img_path):40s}  dice={dice:.4f}')

    mean_dice = float(np.mean(dice_per_image))
    print(f'Mean test Dice: {mean_dice:.4f}')

    wandb.log({
        "test/loss":          te_loss,
        "test/dice":          te_dice,
        "test/tversky":       te_parts["tversky"],
        "test/cldice":        te_parts["cldice"],
        "test/dice_stitched": mean_dice,
        "best_fold":          best_fold,
        "cv_mean_val_loss":   float(np.mean(best_loss_folds)),
        "cv_std_val_loss":    float(np.std(best_loss_folds)),
    })

    wandb_run_dir = os.path.dirname(wandb.run.dir)
    wandb.finish()
    shutil.rmtree(wandb_run_dir, ignore_errors=True)

    print_cv_summary(
        folds            = args.folds,
        best_fold        = best_fold,
        best_loss_folds  = best_loss_folds,
        best_epoch_folds = best_epoch_folds,
        n_folds          = args.n_folds,
        te_loss          = te_loss,
        dice_per_image   = dice_per_image,
        ckpt_dir         = args.ckpt_dir,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import yaml

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, _ = pre.parse_known_args()

    cfg = {}
    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser()
    # ── Paths (cluster-specific; always set via CLI or submit.sh) ─────────────
    parser.add_argument('--config',       default=None)
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
                        help='Train at most this many folds (still uses n_folds-way split)')
    parser.add_argument('--patience',     type=int,   default=30)
    # ── Training loop ─────────────────────────────────────────────────────────
    parser.add_argument('--epochs',       type=int,   default=150)
    parser.add_argument('--batch_size',   type=int,   default=2)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--num_workers',  type=int,   default=8)
    # ── Loss weights ──────────────────────────────────────────────────────────
    parser.add_argument('--warmstart_ckpt',  default=None,
                        help='Checkpoint to warmstart all folds from (e.g. best_model.pth from a prior run).')
    parser.add_argument('--lambda_tversky',      type=float, default=1.0)
    parser.add_argument('--lambda_cldice',  type=float, default=0.0)   # disabled; enable for topology
    parser.add_argument('--tversky_beta',        type=float, default=0.5,
                        help='FN weight β; α=1−β derived. 0.5=Dice, >0.5 penalises FN more.')
    # ── Dataset / augmentation ────────────────────────────────────────────────
    parser.add_argument('--plain_hann',     action='store_true')
    parser.add_argument('--blur_prob',      type=float, default=0.3)
    parser.add_argument('--blur_sigma_max', type=float, default=4.0)

    parser.set_defaults(**{k: v for k, v in cfg.items()
                           if k not in ('input_dir', 'output_dir', 'ckpt_dir')})
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)

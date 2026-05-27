# Vessel Segmentation — Project Context for Claude Code

## Project Overview
Retinal vessel segmentation from TIFF microscopy images.
Architecture: **AttentionUNet** — trainable ResNet34 encoder + UNet decoder with 3 AttentionGate modules.
3-channel input: `[grayscale, gradient_magnitude, sharpness_map]`.

## Model Architecture (`src/model.py`)
- `AttentionGate`: soft attention on skip connections guided by decoder query
- `AttentionUNet`: ResNet34 (timm, pretrained) → 4-level UNet decoder with attention gates + dilated refinement head
- `visualize_attention_maps`: logs all 3 gate outputs (heatmap / overlay / alpha) to W&B


## Configuration (`configs/config.yaml`)
Single config file — all hyperparameters live here, never hardcoded in `.py` files.
`train.py` does two-pass argparse: loads YAML first, then CLI flags override.
Each run copies its YAML into `checkpoints/<job_id>/config.yaml` for self-documentation.

## HPC / SLURM Workflow
- **Cluster**: Explorer at NEU — `ssh a.bashit@login.explorer.northeastern.edu`
- **Project path**: `/home/a.bashit/vessel_seg/`
- **Sync command**:
  ```bash
  rsync -avz --exclude='data/' --exclude='checkpoints/' --exclude='predictions/' --exclude='weights/*.pth' --exclude='logs/' --exclude='wandb/' --exclude='__pycache__/' --exclude='inference_data/' --delete \
      /Users/bashit.a/Downloads/vessel_seg_github/ a.bashit@login.explorer.northeastern.edu:/home/a.bashit/vessel_seg/
  ```
- **Submit training job**:
  ```bash
  sbatch scripts/submit_train.sh                                   # full run (100 epochs, 5 folds)
  sbatch --export=ALL,EPOCHS=5,FOLDS=1 scripts/submit_train.sh    # 1-fold timing test
  sbatch --export=ALL,USE_WANDB=0 scripts/submit_train.sh         # disable W&B
  sbatch --export=ALL,WARMSTART_CKPT=checkpoints/<id>/best_model.pth scripts/submit_train.sh  # warmstart
  ```
- **Submit predict on inference data** (no masks, mirrors subfolder structure):
  ```bash
  sbatch --export=ALL,CKPT_PATH=checkpoints/<id>/best_model.pth,INPUT_DIR=inference_data,MASK_DIR=,USE_WANDB=0 scripts/submit_predict.sh
  ```
- Predict job is auto-queued as a dependent job (`--dependency=afterok`). When `INPUT_DIR` is not `data/images`, `INFERENCE_MODE=1` is set automatically.
- Each job writes to `checkpoints/<SLURM_JOB_ID>/` — no checkpoint collisions.
- **Do NOT** load the system CUDA module — PyTorch ships its own CUDA/cuDNN.

## Known Issues / Decisions
- **numpy/scikit-image binary incompatibility**: different nodes load different PyTorch versions. `submit_predict.sh` guards against this with a `pip install --force-reinstall scikit-image` fallback.
- **torch_geometric / SAM2 not imported**: graph path imports removed from `model.py` and `predict.py` — these packages are not needed and not installed on all nodes.
- **NUM_WORKERS=8** matches `--cpus-per-task=8` in SLURM scripts.
- **Memory fix**: explicit `gc.collect()` + `torch.cuda.empty_cache()` between folds prevents exponential slowdown.

## Source File Map
| File | Purpose |
|------|---------|
| `src/train.py` | Training loop, CV folds, W&B logging, calls `print_cv_summary` at end |
| `src/model.py` | `AttentionUNet` + `AttentionGate` + `visualize_attention_maps` |

| `src/dataset.py` | `VesselDataset` — tiling, sharpness, gradient magnitude, augmentation |
| `src/loss.py` | `VesselLoss` — Tversky + clDice (λ=0); independently weighted |
| `src/predict.py` | Full-image inference: tile → forward → stitch → W&B media log; `--inference_mode` predicts all images and mirrors subfolder structure |
| `src/summarize_cv.py` | `print_cv_summary()` — publication-ready CV table, called directly from train.py |
| `configs/config.yaml` | Single experiment config (AttentionUNet, 5-fold CV, 100 epochs, patience 30) |
| `scripts/submit_train.sh` | SLURM train job — override with `EPOCHS=`, `FOLDS=`, `BATCH_SIZE=`, `USE_WANDB=0`, `WARMSTART_CKPT=` |
| `scripts/submit_predict.sh` | SLURM predict job — auto-queued after training; override `INPUT_DIR=`, `MASK_DIR=`, `USE_WANDB=0` |

## W&B
- Entity: `eeebashit`, Project: `vessel-seg`
- Per-epoch metrics: `fold{N}/train_{loss,dice,tversky,cldice}`, `fold{N}/val_{loss,dice,tversky,cldice}`
- Test metrics: `test/{loss,dice,tversky,cldice,dice_stitched}`
- Predict run: originals, ch_grad, ch_sharp, prediction overlay, gt, tp/fp/fn, combined, attention maps
- Disable with `USE_WANDB=0` in sbatch `--export` — saves ~7 min startup time per job

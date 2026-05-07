# Vessel Segmentation — Project Context for Claude Code

## Project Overview
Retinal vessel segmentation from TIFF microscopy images.
Architecture: **AttentionUNet** — trainable ResNet34 encoder + UNet decoder with 3 AttentionGate modules.
3-channel input: `[grayscale, gradient_magnitude, sharpness_map]`.

## Model Architecture (`src/model.py`)
- `AttentionGate`: soft attention on skip connections guided by decoder query
- `AttentionUNet`: ResNet34 (timm, pretrained) → 4-level UNet decoder with attention gates + dilated refinement head
- `visualize_attention_maps`: logs all 3 gate outputs (heatmap / overlay / alpha) to W&B
- Graph/SAM2 code archived in `src/model_graph.py` — not imported anywhere

## Configuration (`configs/config.yaml`)
Single config file — all hyperparameters live here, never hardcoded in `.py` files.
`train.py` does two-pass argparse: loads YAML first, then CLI flags override.
Each run copies its YAML into `checkpoints/<job_id>/config.yaml` for self-documentation.

## HPC / SLURM Workflow
- **Cluster**: Explorer at NEU — `ssh a.bashit@login.explorer.northeastern.edu`
- **Project path**: `/home/a.bashit/vessel_seg/`
- **Sync command**:
  ```bash
  rsync -avz --exclude='data/' --exclude='checkpoints/' --exclude='predictions/' --exclude='__pycache__/' --delete \
      /Users/bashit.a/Downloads/vessel_seg_github/ a.bashit@login.explorer.northeastern.edu:/home/a.bashit/vessel_seg/
  ```
- **Submit training job**:
  ```bash
  sbatch scripts/submit_train.sh                        # full run (200 epochs, 2 folds)
  sbatch --export=ALL,EPOCHS=1,FOLDS=1 scripts/submit_train.sh  # 1-epoch timing test
  ```
- Predict job is auto-queued as a dependent job (`--dependency=afterok`).
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
| `src/model_graph.py` | Archived VesselSegNet + graph path — not used |
| `src/dataset.py` | `VesselDataset` — tiling, sharpness, gradient magnitude, augmentation |
| `src/loss.py` | `VesselLoss` — Dice + BCE hard-neg + clDice + sharpness boundary + contrastive |
| `src/predict.py` | Full-image inference: tile → forward → stitch → W&B media log |
| `src/postprocess.py` | Rule-based mask cleanup (blob removal, hole fill, gap close) |
| `src/summarize_cv.py` | `print_cv_summary()` — publication-ready CV table, called directly from train.py |
| `configs/config.yaml` | Single experiment config (AttentionUNet, 2-fold CV, 200 epochs) |
| `scripts/submit_train.sh` | SLURM train job — override with `EPOCHS=`, `FOLDS=`, `BATCH_SIZE=` |
| `scripts/submit_predict.sh` | SLURM predict job — auto-queued after training |

## W&B
- Entity: `eeebashit`, Project: `vessel-seg`
- Per-epoch metrics: `fold{N}/train_{loss,dice,bce,cldice,boundary,contrast}`
- Every 10 val epochs: attention maps (heatmap, overlay, alpha) for all 3 gates
- Predict run: originals, prediction overlay, gt, tp/fp/fn, combined, attention maps

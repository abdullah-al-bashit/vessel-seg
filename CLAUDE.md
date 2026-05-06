# Vessel Segmentation — Project Context for Claude Code

## Project Overview
Retinal vessel segmentation from TIFF microscopy images.
Architecture: frozen SAM2.1 Hiera-Tiny encoder → CNN decoder → segmentation head.
Graph path (skeletonize + ChebConv) is implemented but **disabled** — CNN-only converges faster and achieves better val Dice.

## Current Model Input Channels
SAM2 receives a **3-channel input** built in `VesselSegNet.forward`:
- ch0 = grayscale tile
- ch1 = gradient magnitude (`compute_gradient_magnitude` — Sobel edges, vessel walls)
- ch2 = sharpness map (`compute_sharpness` — VoL, focus quality)

Both maps are computed per tile in `dataset.py` and passed as `grad_mag` and `sharpness` tensors.

## Blurry Region Suppression — Design
Three complementary mechanisms to stop the model predicting vessels in blurry/out-of-focus regions:

1. **Sharpness-weighted Hanning loss** (`plain_hann=false`) — blurry pixels contribute less to loss
2. **Sharpness boundary loss** (`lambda_boundary=0.5`) — penalises predictions crossing sharp→blurry edges
3. **Feature-level gating** (exp-specific, see below)

## 4-Experiment Design
Currently running jobs **6577725–6577728** (200 epochs, patience=30, ~105s/epoch without graph):

| Job | Exp | plain_hann | sharp_gate | use_focus_gate | blur_aug |
|-----|-----|-----------|------------|----------------|----------|
| 6577725 | A | false | false | false | default (0.3, σ4) |
| 6577726 | B | true | **true** (learnable) | false | default |
| 6577727 | C | false | false | **true** | aggressive (0.6, σ8) |
| 6577728 | D | true | false | false | aggressive (0.6, σ8) |

- **Exp B — learnable sharp gate**: `gate = sigmoid(scale * sharpness + bias)` per decoder channel (32 scale + 32 bias params). Raw sharpness gate collapsed predictions (negative dice) — learnable version fixes this.
- **Exp C — focus head**: separate `focus_head` conv predicts in-focus map from features, supervised by VoL sharpness. Gates `F_pixel` by `sigmoid(focus_logits)`. Focus loss logged as `fold1/train_focus`.

## Configuration Management
- **ALL hyperparameters live in `configs/exp_*.yaml`** — never hardcoded in `.py` files.
- `train.py` does two-pass argparse: loads YAML first via `--config`, then CLI flags override.
- Each run copies its YAML into `checkpoints/<job_id>/config.yaml` for self-documentation.
- Override epochs for timing tests: `sbatch --export=ALL,EPOCHS=1 scripts/submit.sh`

## HPC / SLURM Workflow
- **Cluster**: Explorer at NEU — `ssh a.bashit@login.explorer.northeastern.edu`
- **Project path on cluster**: `/home/a.bashit/vessel_seg/`
- **Sync command**: `rsync -avz --exclude='data/' --exclude='checkpoints/' --exclude='predictions/' --exclude='__pycache__/' /Users/bashit.a/Downloads/vessel_seg_github/ a.bashit@login.explorer.northeastern.edu:/home/a.bashit/vessel_seg/`
- **Submit 4 overnight jobs**:
  ```bash
  sbatch scripts/submit.sh                                          # A: baseline
  sbatch --export=ALL,CONFIG=configs/exp_B.yaml scripts/submit.sh  # B: learnable sharp gate
  sbatch --export=ALL,CONFIG=configs/exp_C.yaml scripts/submit.sh  # C: focus head gate
  sbatch --export=ALL,CONFIG=configs/exp_D.yaml scripts/submit.sh  # D: plain hann + aggressive blur
  ```
- Each job writes to `checkpoints/<SLURM_JOB_ID>/` — no checkpoint collisions.
- Predict job is automatically queued as a dependent job (`--dependency=afterok`).
- **Do NOT** load the system CUDA module (`module load cuda`) — PyTorch ships its own CUDA/cuDNN.
- Calibrate epoch count with a 2-epoch timing test before committing an 8-hour job.

## Known Incompatibilities / Decisions
- **`torch.compile` is incompatible with HuggingFace SAM2** — do not add it. Causes `NameError: name 'torch' is not defined` inside `transformers/output_capturing.py`.
- **Graph path is disabled** (`use_graph=False` always). Adds 4× runtime with no Dice improvement.
- **`use_graph` must match between train and predict** — if ever re-enabled, update both files.
- **Raw sharpness gate collapses training** — exp B previously used `F_pixel * sharpness` directly, causing negative dice/cldice (features suppressed to ~0). Fixed with learnable per-channel gate.
- **NUM_WORKERS=8** matches `--cpus-per-task=8` in submit.sh.

## Source File Map
| File | Purpose |
|------|---------|
| `src/train.py` | Training loop, CV folds, W&B logging. All constants come from config. |
| `src/model.py` | `VesselSegNet` — SAM2 encoder + CNN decoder + learnable sharp gate + focus head. |
| `src/dataset.py` | `VesselDataset` — tiling, `compute_sharpness`, `compute_gradient_magnitude`, augmentation. |
| `src/loss.py` | `VesselLoss` — Dice + BCE hard-neg + clDice + sharpness boundary + contrastive. |
| `src/predict.py` | Full-image inference: tile → forward → stitch → postprocess → W&B media log. |
| `src/postprocess.py` | Rule-based mask cleanup (small blob removal, hole fill, gap close). |
| `configs/exp_*.yaml` | Per-experiment configs. A=baseline, B=learnable gate, C=focus head, D=aggressive blur. |
| `scripts/submit.sh` | SLURM train job. Set `CONFIG=` and optionally `EPOCHS=` env vars. |
| `scripts/submit_predict.sh` | SLURM predict job. Auto-queued after training via `--dependency=afterok`. |

## W&B
- Entity: `eeebashit`, Project: `vessel-seg`
- Per-epoch metrics: `fold1/train_{loss,dice,bce,cldice,boundary,contrast,coarse,focus}`
- Predict run logs: originals, prediction, prediction_pp, gt, tp/fp/fn, combined overlays

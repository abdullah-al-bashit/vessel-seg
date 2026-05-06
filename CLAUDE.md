# Vessel Segmentation — Project Context for Claude Code

## Project Overview
Retinal vessel segmentation from TIFF microscopy images.
Architecture: frozen SAM2.1 Hiera encoder → CNN decoder → segmentation head.
Graph path (skeletonize + ChebConv) is implemented but **disabled** — CNN-only converges faster and achieves better val Dice.

## Configuration Management
- **ALL hyperparameters live in `configs/exp_*.yaml`** — never hardcoded in `.py` files.
- Includes: epochs, lr, batch_size, loss weights, W&B entity/project, SAM2 model ID, blur aug settings.
- `train.py` does two-pass argparse: loads YAML first via `--config`, then CLI flags override.
- Each run copies its YAML into `checkpoints/<job_id>/config.yaml` for self-documentation.

## HPC / SLURM Workflow
- **Cluster**: Explorer at NEU — `ssh a.bashit@login.explorer.northeastern.edu`
- **Project path on cluster**: `/home/a.bashit/vessel_seg/`
- **Sync command**: `rsync -avz --exclude='data/' --exclude='checkpoints/' --exclude='predictions/' --exclude='__pycache__/' /Users/bashit.a/Downloads/vessel_seg_github/ a.bashit@login.explorer.northeastern.edu:/home/a.bashit/vessel_seg/`
- **Submit 4 overnight jobs**:
  ```bash
  sbatch scripts/submit.sh                                          # A: baseline
  sbatch --export=ALL,CONFIG=configs/exp_B.yaml scripts/submit.sh  # B: plain hann
  sbatch --export=ALL,CONFIG=configs/exp_C.yaml scripts/submit.sh  # C: more blur aug
  sbatch --export=ALL,CONFIG=configs/exp_D.yaml scripts/submit.sh  # D: both
  ```
- Each job writes to `checkpoints/<SLURM_JOB_ID>/` — no checkpoint collisions.
- Predict job is automatically queued as a dependent job (`--dependency=afterok`).
- **Notifications**: Use Slack webhooks — cluster email and Mac background notifications are unreliable.
- **Do NOT** load the system CUDA module (`module load cuda`) — PyTorch ships its own CUDA/cuDNN; loading the system module causes `CUDNN_STATUS_NOT_INITIALIZED`.
- Calibrate epoch count with a 2-epoch timing test before committing an 8-hour job.

## Known Incompatibilities / Decisions
- **`torch.compile` is incompatible with HuggingFace SAM2** — do not add it.
- **Graph path is disabled** (`use_graph = False` always in train.py, `use_graph=False` in predict.py). The graph architecture (skan + ChebConv) is in the code for reference but adds 4× runtime with no Dice improvement.
- **`use_graph` must match between train and predict** — if graph is ever re-enabled in training, predict.py must also use `use_graph=True`, otherwise untrained `fuse_proj` weights produce noise.

## Source File Map
| File | Purpose |
|------|---------|
| `src/train.py` | Training loop, CV folds, W&B logging. All constants come from config. |
| `src/model.py` | `VesselSegNet(sam2_model=...)` — SAM2 encoder + CNN decoder + graph net (disabled). |
| `src/dataset.py` | `VesselDataset(sharp_hann=, blur_prob=, blur_sigma_max=)` — tiling, augmentation. |
| `src/loss.py` | `VesselLoss` — Dice + BCE hard-neg + clDice + sharpness boundary + contrastive. |
| `src/predict.py` | Full-image inference: tile → forward → stitch → postprocess → W&B media log. |
| `src/postprocess.py` | Rule-based mask cleanup (small blob removal, hole fill, gap close). |
| `configs/exp_*.yaml` | Per-experiment configs. A=baseline, B=plain hann, C=more blur, D=both. |
| `scripts/submit.sh` | SLURM train job. Set `CONFIG=` env var to pick experiment. |
| `scripts/submit_predict.sh` | SLURM predict job. Reads `CKPT_PATH` from env (set by submit.sh). |

## Blur Detection — 4-Experiment Design
The four overnight configs are a 2×2 factorial experiment:
- **Factor 1** — Loss gate: `plain_hann=false` (sharpness-gated, model ignores blur) vs `plain_hann=true` (plain Hanning, model trains on blurry regions)
- **Factor 2** — Blur aug: default (prob=0.3, sigma_max=4) vs aggressive (prob=0.6, sigma_max=8)

Goal: identify whether blur detection improves from the loss side (B), the augmentation side (C), or both (D).

## W&B
- Entity: `eeebashit` (set in YAML, not hardcoded)
- Project: `vessel-seg`
- Each run logs full config via `wandb.init(config=vars(args))` — every YAML param is tracked.

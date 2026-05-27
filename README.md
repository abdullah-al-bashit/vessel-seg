# Vessel Segmentation — AttentionUNet

Binary segmentation of blood vessels in single-channel fluorescence TIFF microscopy images using an Attention UNet with a trainable ResNet34 encoder.

---

## Architecture

```
Raw fluorescence image  (H × W · uint16)
        │
        ▼
Normalize → tile horizontally → [grayscale, gradient_magnitude, sharpness_map]
        │
        ▼
ResNet34 encoder (timm, ImageNet pretrained, fully trainable)
  e0: (B,  64, H/2,  W/2)   ← stem, finest scale
  e1: (B,  64, H/4,  W/4)   ← layer1
  e2: (B, 128, H/8,  W/8)   ← layer2
  e3: (B, 256, H/16, W/16)  ← layer3
  e4: (B, 512, H/32, W/32)  ← layer4, coarsest scale
        │
        ▼
UNet decoder with AttentionGates on all skip connections
  d4: up4(e4)       + att1(d4, e3)    → (B, 256, H/16, W/16)
  d3: up3(d4+att1)  + att2(d3, e2)   → (B, 128, H/8,  W/8)
  d2: up2(d3+att2)  + att3(d2, e1)   → (B,  64, H/4,  W/4)
  d1: up1(d2+att3)  + att0(d1, e0)   → (B,  32, H/2,  W/2)
  d0: up0(d1+att0)                   → (B,  16, H,     W)
        │
        ▼
Dilated refinement head (dilation=2 → 9×9 receptive field)
        │
        ▼
Conv 1×1 → logits (B, 1, H, W)
        │
        ▼
Loss: Tversky (β=0.5 default, configurable) + optional clDice
        │
        ▼
Binary mask (H × W)
```

---

## Key Design Choices

| Choice | Reason |
|---|---|
| 3-channel input [gray, grad_mag, sharpness] | Explicit texture cues beyond raw intensity |
| Trainable ResNet34 encoder | Adapts to fluorescence domain; ImageNet weights as starting point |
| AttentionGate on all 4 skip connections | Suppresses irrelevant background features at every decoder scale |
| Dilated refinement head (d=2) | 9×9 effective receptive field fills vessel gaps before final projection |
| Tversky loss (β configurable) | β=0.5 → Dice; β=0.7 → penalises missed vessels 2.3× more than FP |
| Hanning loss gate | Tile-boundary pixels never contribute to loss — prevents broken-vessel learning |
| Horizontal tiling + overlap stitching | No vertical vessel cuts on panoramic ~15800×1300 px images |
| Per-image min-max normalization | Handles any bit depth or acquisition settings |

---

## Dataset

- Fluorescence `.tif` images, ~15800 × 1300 px, uint16
- Binary masks: `255` = vessel lumen, `0` = background/walls

Expected layout:

```
data/
├── images/    ← raw fluorescence images (.tif)
└── masks/     ← binary masks matching images by filename
```

---

## Installation (NEU Explorer Cluster)

**Step 1 — Create the conda environment (one time only):**
```bash
bash scripts/setup_env.sh
```

**Step 2 — Edit your email in the SLURM scripts:**
```bash
# In scripts/submit_train.sh and scripts/submit_predict.sh:
#SBATCH --mail-user=YOUR_EMAIL@northeastern.edu
```

---

## Configuration

All hyperparameters live in [configs/config.yaml](configs/config.yaml) — nothing is hardcoded in `.py` files.

```yaml
n_folds:    5
epochs:     100
batch_size: 12
lr:         0.0001
patience:   30

lambda_tversky: 1.0
lambda_cldice:  0.0   # enable for topology-aware training
tversky_beta:   0.5   # 0.5 = Dice; 0.7 = FN penalises missed vessels more
```

Each run copies its config into `checkpoints/<SLURM_JOB_ID>/config.yaml`.

---

## Usage

### 1 — Train (K-fold cross-validation)

```bash
sbatch scripts/submit_train.sh
```

A predict job is **automatically queued** as a dependent SLURM job and runs on completion.

Common overrides:
```bash
# 1-fold timing test (5 epochs)
sbatch --export=ALL,EPOCHS=5,FOLDS=1 scripts/submit_train.sh

# Disable W&B (skips ~7 min startup overhead)
sbatch --export=ALL,USE_WANDB=0 scripts/submit_train.sh

# Warmstart every fold from a prior checkpoint
sbatch --export=ALL,WARMSTART_CKPT=checkpoints/<job_id>/best_model.pth scripts/submit_train.sh

# Enable clDice topology loss
sbatch --export=ALL,LAMBDA_CLDICE=0.3 scripts/submit_train.sh
```

Checkpoints are saved per job: `checkpoints/<SLURM_JOB_ID>/best_model.pth`.  
The config used is copied there too: `checkpoints/<SLURM_JOB_ID>/config.yaml`.

---

### 2 — Predict on training data (test/trainval split-aware)

Automatically queued after training. To re-run manually against an existing checkpoint:
```bash
sbatch --export=ALL,CKPT_PATH=checkpoints/<job_id>/best_model.pth scripts/submit_predict.sh
```

- Predicts a subset of images (2 trainval + up to 5 test) and logs them to W&B
- TP/FP/FN overlays and per-image Dice are logged when ground-truth masks are present
- Output masks saved as `predictions/<name>_pred.tif`

---

### 3 — Inference on new images in `inference_data/`

Place `.tif` images on the cluster under `inference_data/` (subfolders are fine), then:

```bash
# No ground-truth masks available
sbatch --export=ALL,CKPT_PATH=checkpoints/<job_id>/best_model.pth,INPUT_DIR=inference_data,MASK_DIR= scripts/submit_predict.sh

# With ground-truth masks — logs per-image Dice + TP/FP/FN panels to W&B
sbatch --export=ALL,CKPT_PATH=checkpoints/<job_id>/best_model.pth,INPUT_DIR=inference_data,MASK_DIR=inference_masks scripts/submit_predict.sh
```

- Processes **all** `.tif` files recursively — no split filtering
- `INFERENCE_MODE=1` is set automatically when `INPUT_DIR` is not `data/images`
- Output masks mirror the input subfolder structure: `predictions/inference/<subfolder>/<name>_mask.tif`
- W&B run logs originals, gradient/sharpness channels, prediction overlays, and attention maps
- When `MASK_DIR` is provided, a `dice_per_image` table and `mean_dice` scalar are also logged

---

### Monitor

```bash
squeue -u $USER
tail -f logs/train_<JOBID>.out
tail -f logs/predict_<JOBID>.out
```

---

## Repository Structure

```
vessel_seg/
├── src/
│   ├── model.py           AttentionUNet + AttentionGate + visualize_attention_maps
│   ├── dataset.py         VesselDataset — tiling, sharpness, gradient magnitude, augmentation
│   ├── loss.py            VesselLoss — Tversky + clDice (Hanning-gated); independently weighted
│   ├── train.py           K-fold CV, AMP, W&B logging, early stopping
│   ├── predict.py         tile → forward → stitch → W&B media log; --inference_mode for new data
│   └── summarize_cv.py    print_cv_summary — publication-ready CV table
├── configs/
│   └── config.yaml        single experiment config (all hyperparameters)
├── scripts/
│   ├── setup_env.sh       one-time conda env setup on Explorer
│   ├── submit_train.sh    SLURM train job (auto-queues predict on success)
│   ├── submit_predict.sh  SLURM predict job; handles inference via INFERENCE_MODE=1
│   └── overlay_inspector.ipynb         interactive overlay viewer for predictions
├── inference_data/        new images for inference — cluster only, git-ignored
├── vessel-seg-report/     LaTeX report source + compiled PDF
├── weights/               cached ResNet34 pretrained weights (git-ignored)
├── checkpoints/           saved model weights per job ID (git-ignored)
├── predictions/           output masks (git-ignored)
├── logs/                  SLURM logs (git-ignored)
├── requirements.txt
└── README.md
```

---

## W&B Logging

- **Entity**: `eeebashit` · **Project**: `vessel-seg`
- **Training** — per-epoch: `fold{N}/train_{loss,dice,tversky,cldice}` and matching `val_*`; attention maps every 10 val epochs
- **Predict / Inference** — per image: originals, gradient, sharpness, prediction overlay, attention maps; when masks are available: ground truth, TP/FP/FN panels, `dice` scalar, and an end-of-run `dice_per_image` table + `mean_dice`
- Disable with `USE_WANDB=0` in `--export` — saves ~7 min startup time per job

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Dice coefficient | Pixel-level overlap |
| clDice | Centerline topology — vessel connectivity |

---

## Output Format

Predicted masks saved as uint8 TIFF:
- `255` = vessel lumen
- `0` = background / vessel walls

Filename format:
- Training data predict: `{original_name}_pred.tif`
- Inference mode: `{original_name}_mask.tif` (subfolder structure mirrored)

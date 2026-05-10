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
Loss: Tversky (β=0.7 penalises missed vessels) + optional clDice + skel_density
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
| Tversky loss (α=0.3, β=0.7) | Penalises missed vessels 2.3× more than false positives |
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

lambda_tversky:      1.0
lambda_cldice:       0.0   # enable for topology-aware training
lambda_skel_density: 0.0   # enable for blob-penalty training
tversky_beta:        0.5   # 0.5 = Dice; 0.7 = FN penalised 2.3× more
```

Each run copies its config into `checkpoints/<SLURM_JOB_ID>/config.yaml`.

---

## Usage

### Train (K-fold cross-validation)

```bash
sbatch scripts/submit_train.sh
```

Override config values via environment variables:
```bash
# Quick 1-epoch timing test
sbatch --export=ALL,EPOCHS=1,FOLDS=1 scripts/submit_train.sh

# Larger batch + more workers
sbatch --export=ALL,BATCH_SIZE=16,NUM_WORKERS=8 scripts/submit_train.sh

# Enable clDice loss
sbatch --export=ALL,LAMBDA_CLDICE=0.3 scripts/submit_train.sh
```

The predict job is automatically queued as a dependent SLURM job after training completes.

### Predict

```bash
sbatch scripts/submit_predict.sh
```

Or manually:
```bash
python src/predict.py \
    --config     configs/config.yaml \
    --input_dir  data/images \
    --output_dir data/masks \
    --ckpt_path  checkpoints/<job_id>/best_model.pth
```

### Monitor

```bash
squeue -u $USER
tail -f logs/train_<JOBID>.out
```

---

## Repository Structure

```
vessel_seg/
├── src/
│   ├── model.py           AttentionUNet + AttentionGate + visualize_attention_maps
│   ├── dataset.py         VesselDataset — tiling, sharpness, gradient magnitude, augmentation
│   ├── loss.py            VesselLoss — Tversky + clDice + skel_density (Hanning-gated)
│   ├── train.py           K-fold CV, AMP, W&B logging, early stopping
│   ├── predict.py         tile → forward → stitch → W&B media log
│   └── summarize_cv.py    print_cv_summary — publication-ready CV table
├── configs/
│   └── config.yaml        single experiment config (all hyperparameters)
├── scripts/
│   ├── setup_env.sh       one-time conda env setup on Explorer
│   ├── submit_train.sh    SLURM train job (auto-queues predict on success)
│   ├── submit_predict.sh  SLURM predict job
│   └── visualize_d7_interactive.ipynb  interactive overlay viewer
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
- Per-epoch: `fold{N}/train_{loss,tversky,cldice,boundary}` and validation metrics
- Every 10 val epochs: attention maps (heatmap, overlay, alpha) for all 3 gates (att1, att2, att3)
- Predict run: originals, prediction overlay, ground truth, TP/FP/FN panels, attention maps

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Dice coefficient | Pixel-level overlap |
| clDice | Centerline topology — vessel connectivity |
| Hausdorff (95%) | Worst-case boundary error |

---

## Output Format

Predicted masks saved as uint8 TIFF:
- `255` = vessel lumen
- `0` = background / vessel walls

Filename format: `{original_name}_pred.tif`

# Vessel Segmentation — SAM2.1 + ChebConv Graph Network

Robust binary segmentation of blood vessels in single-channel fluorescence microscopy images using a joint CNN + Graph Neural Network decoder built on top of a frozen SAM2.1 (Hiera ViT-L) image encoder.

---

## Architecture

```
Raw fluorescence image  (H × W · uint16)
        │
        ▼
Normalize  →  Strip tile 1024×1300  →  Hanning weight map
        │
        ▼
Frozen SAM2.1 Hiera ViT-L encoder  (facebook/sam2.1-hiera-large)
  Stage 1 · 2×  local attn
  Stage 2 · 2×  + pool
  Stage 3 · 11× + pool
  Stage 4 · 2×  + pool
  FPN neck · 4 scales · 256ch
        │
        ├──────────────────────────────────┐
        │                                  │
   CNN upsampling                    Graph construction
   ConvT 2× · 256→64ch               coarse mask → skeletonize
   ConvT 2× · 64→32ch                extract G=(V,E) via sknw
   F_pixel (H×W×32)                  anisotropic node features:
                                        (x, y, cosθ, sinθ, d, κ)
                                      edge features: (Δθ, length, gap_int)
                                      ChebConv(K=10) × 3 layers
                                      F_graph (N×64)
        │                                  │
        └──────── F_fused = F_pixel + scatter(F_graph) ──────┘
                                    │
                              Conv 1×1 → logits
                                    │
                        Loss: Dice + BCE + clDice  (Hanning-gated)
                                    │
                              Binary mask (H × W)
                                    │
                        Postprocess: rm small obj · fill holes
                                    │
                        GAT topology refinement  (optional)
```

---

## Key Design Choices

| Choice | Reason |
|---|---|
| Frozen SAM2.1 encoder | 11M image pretrain — illumination invariant, no preprocessing needed |
| HuggingFace `from_pretrained` | No manual checkpoint download, auto-cached to scratch |
| Strip tiling `1024 × full_height` | No vertical vessel cuts — panoramic images are ~15800×1300px |
| Hanning loss gate | Boundary pixels at tile edges never contribute to loss — prevents broken-vessel learning |
| 50% tile overlap + average stitching | Every pixel seen near-center in at least one tile |
| ChebConv K=10 | 10-hop receptive field captures full vessel length regardless of pixel distance |
| Anisotropic node features (θ, d, κ) | Direction-aware — fixes GCN isotropy problem for tubular structures |
| Dice + BCE + clDice | Pixel accuracy + topology correctness dual objective |
| Per-image min-max normalization | No pixel removal, handles any bit depth or acquisition |

---

## Dataset

- 45 single-channel fluorescence `.tif` images
- 30 annotated with binary masks (`255` = vessel lumen, `0` = vessel walls)
- 15 unannotated (used for pseudo-labeling after training)
- Image resolution: ~15800 × 1300 px · uint16 · 12-bit effective range

Expected directory layout:

```
project/
├── Input/
│   ├── D7/     ← annotated raw images
│   ├── D14/
│   └── D21/
├── Output/
│   ├── D7/     ← binary masks matching Input/D7 by leading ID
│   ├── D14/
│   └── D21/
├── Input_No_Masks/   ← 15 unlabeled images
└── vessel_seg/       ← this repo
```

---

## Installation (NEU Explorer Cluster)

**Step 1 — Start an interactive GPU session:**
```bash
srun --partition=gpu-interactive --gres=gpu:v100-sxm2:1 \
     --cpus-per-task=4 --mem=16GB --time=01:00:00 --pty /bin/bash
```

**Step 2 — Run setup script (one time only):**
```bash
bash scripts/setup_env.sh
```

This creates the `vessel_seg` conda environment, installs all dependencies, and pre-downloads the SAM2.1 weights (~2.5 GB) into `/scratch/$USER/.cache/huggingface/`.

**Step 3 — Edit email in SLURM scripts:**
```bash
# In scripts/submit.sh and scripts/submit_predict.sh:
#SBATCH --mail-user=YOUR_EMAIL@northeastern.edu
```

---

## Usage

### Train (5-fold cross-validation)

```bash
sbatch scripts/submit.sh
```

Or manually:
```bash
cd src
python train.py \
    --input_dir  ../../Input/D7  \
    --output_dir ../../Output/D7 \
    --ckpt_dir   ../checkpoints  \
    --epochs     200             \
    --batch_size 2               \
    --lr         1e-4
```

### Predict (15 unlabeled images)

```bash
sbatch scripts/submit_predict.sh
```

Or manually:
```bash
cd src
python predict.py \
    --input_dir ../../Input_No_Masks    \
    --ckpt_path ../checkpoints/fold1_best.pth \
    --out_dir   ../predictions
```

### Monitor training

```bash
squeue -u $USER
tail -f logs/train_JOBID.out
```

---

## Repository Structure

```
vessel_seg/
├── src/
│   ├── dataset.py       normalize · strip tiling · Hanning weight · VesselDataset
│   ├── model.py         SAM2.1 encoder · CNN decoder · ChebConv graph decoder
│   ├── loss.py          Dice + BCE + clDice · Hanning-gated
│   ├── train.py         5-fold CV · AdamW · cosine LR · early stopping
│   ├── predict.py       tile → forward → stitch → postprocess
│   └── postprocess.py   cleanup + GAT topology refinement
├── scripts/
│   ├── setup_env.sh     one-time env setup on Explorer
│   ├── submit.sh        SLURM training job
│   └── submit_predict.sh SLURM prediction job
├── notebooks/
│   └── visualize.ipynb  interactive overlay viewer (ipywidgets)
├── checkpoints/         saved model weights (git-ignored)
├── predictions/         output masks (git-ignored)
├── logs/                SLURM logs (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Output Format

Predicted masks saved as uint8 TIFF files:
- `255` = vessel lumen
- `0` = vessel walls / background

Filename format: `{original_name}_pred.tif`

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Dice coefficient | Pixel-level overlap |
| clDice | Centerline topology — vessel connectivity |
| Hausdorff (95%) | Worst-case boundary error |
| AUC-PR | Threshold-independent performance |

---

## Citation

If you use SAM2.1:
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

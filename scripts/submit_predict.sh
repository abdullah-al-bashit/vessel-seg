#!/bin/bash
#SBATCH --job-name=vessel_predict
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=../logs/predict_%j.out
#SBATCH --error=../logs/predict_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@northeastern.edu

# ── Explorer modules ───────────────────────────────────────────────────────────
module purge
module load explorer
module load anaconda3/2024.06
module load cuda/12.1.1

source activate vessel_seg

# ── HuggingFace cache → scratch ────────────────────────────────────────────────
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/vessel_seg

echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')"
echo "Start: $(date)"

# ── Paths — EDIT THESE ─────────────────────────────────────────────────────────
INPUT_DIR="../Input_No_Masks"        # 15 unlabeled images
CKPT_PATH="../checkpoints/fold1_best.pth"
OUT_DIR="../predictions"

# ── Predict (no --sam2_ckpt needed) ───────────────────────────────────────────
python src/predict.py \
    --input_dir  $INPUT_DIR  \
    --ckpt_path  $CKPT_PATH  \
    --out_dir    $OUT_DIR

echo "Done: $(date)"

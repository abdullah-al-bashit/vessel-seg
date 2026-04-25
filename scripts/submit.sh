#!/bin/bash
#SBATCH --job-name=vessel_seg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/train_%j.out
#SBATCH --error=../logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@northeastern.edu

# ── Explorer modules ───────────────────────────────────────────────────────────
module purge
module load explorer
module load anaconda3/2024.06
module load cuda/12.1.1

source activate vessel_seg

# ── HuggingFace cache → scratch (more space than home) ────────────────────────
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/vessel_seg

# ── Verify GPU + CUDA ──────────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA:      $(nvcc --version | grep release)"
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}  {torch.cuda.get_device_name(0)}')"
echo "Start:     $(date)"

# ── Paths — EDIT THESE ─────────────────────────────────────────────────────────
INPUT_DIR="../Input/D7"
OUTPUT_DIR="../Output/D7"
CKPT_DIR="../checkpoints"

# ── Training (no --sam2_ckpt needed — HuggingFace auto-downloads) ─────────────
python src/train.py \
    --input_dir   $INPUT_DIR  \
    --output_dir  $OUTPUT_DIR \
    --ckpt_dir    $CKPT_DIR   \
    --epochs      200         \
    --batch_size  2           \
    --lr          1e-4

echo "Done: $(date)"

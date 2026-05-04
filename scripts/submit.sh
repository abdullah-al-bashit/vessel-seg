#!/bin/bash
#SBATCH --job-name=vessel_seg
#SBATCH --output=../logs/train_%j.out
#SBATCH --error=../logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a.bashit@northeastern.edu

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
INPUT_DIR="../data/images"
OUTPUT_DIR="../data/masks"
CKPT_DIR="../checkpoints"

# ── Training (1-epoch sanity check; bump --epochs once it runs cleanly) ───────
python src/train.py \
    --input_dir   $INPUT_DIR  \
    --output_dir  $OUTPUT_DIR \
    --ckpt_dir    $CKPT_DIR   \
    --epochs      1           \
    --batch_size  2           \
    --lr          1e-4

echo "Done: $(date)"

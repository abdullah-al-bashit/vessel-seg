#!/bin/bash
#SBATCH --job-name=vessel_seg
#SBATCH --output=/home/a.bashit/vessel_seg/logs/train_%j.out
#SBATCH --error=/home/a.bashit/vessel_seg/logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a.bashit@northeastern.edu

# ── Explorer modules ───────────────────────────────────────────────────────────
module purge
module load explorer
module load anaconda3/2024.06
# (Don't load system cuda — PyTorch 2.5.1+cu121 ships its own CUDA/cuDNN.
#  Loading the system module puts an older cuDNN in LD_LIBRARY_PATH and
#  causes `CUDNN_STATUS_NOT_INITIALIZED` on the first forward pass.)

source activate pytorch_env

# ── HuggingFace cache → scratch (more space than home) ────────────────────────
export HF_HOME=/scratch/$USER/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/$USER/vessel_seg

# ── Verify GPU + CUDA ──────────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA:      $(python -c 'import torch; print(torch.version.cuda)')"
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}  {torch.cuda.get_device_name(0)}')"
echo "Start:     $(date)"

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_DIR="data/images"
OUTPUT_DIR="data/masks"
CKPT_DIR="checkpoints"

# ── Train ─────────────────────────────────────────────────────────────────────
python src/train.py \
    --input_dir   $INPUT_DIR  \
    --output_dir  $OUTPUT_DIR \
    --ckpt_dir    $CKPT_DIR   \
    --folds       1           \
    --epochs      1           \
    --batch_size  8           \
    --lr          1e-4

echo "Training done: $(date)"

# ── Predict: run on all images using best checkpoint ──────────────────────────
python src/predict.py \
    --input_dir  $INPUT_DIR          \
    --mask_dir   $OUTPUT_DIR         \
    --ckpt_path  $CKPT_DIR/fold1_best.pth \
    --out_dir    predictions

echo "Done: $(date)"

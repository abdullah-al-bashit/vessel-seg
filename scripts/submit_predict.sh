#!/bin/bash
#SBATCH --job-name=vessel_predict
#SBATCH --output=/home/a.bashit/vessel_seg/logs/predict_%j.out
#SBATCH --error=/home/a.bashit/vessel_seg/logs/predict_%j.err
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
# Predicts on all .tif images in INPUT_DIR (test + trainval, recursively).
# data_splits.json (next to CKPT_PATH) tags each W&B Media caption as [test] or [trainval].
INPUT_DIR="data/images"
MASK_DIR="data/masks"      # for TP/FP/FN error overlay; omit to skip
# CKPT_PATH is set by submit.sh via --export=ALL,CKPT_PATH=checkpoints/<job_id>/fold1_best.pth
# The fallback is for manual runs only — in normal usage this is always set by the training job.
CKPT_PATH="${CKPT_PATH:-checkpoints/fold1_best.pth}"
OUT_DIR="predictions"

# ── Predict ────────────────────────────────────────────────────────────────────
python src/predict.py \
    --input_dir  $INPUT_DIR  \
    --mask_dir   $MASK_DIR   \
    --ckpt_path  $CKPT_PATH  \
    --out_dir    $OUT_DIR

echo "Done: $(date)"

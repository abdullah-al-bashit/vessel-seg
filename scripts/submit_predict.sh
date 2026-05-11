#!/bin/bash
#SBATCH --job-name=vessel_pred
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
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

# Use absolute path to conda env so the correct Python is always used,
# regardless of which node SLURM lands on. 'source activate' silently
# falls back to the system Python on some nodes (different CUDA stacks).
PYTHON=/home/$USER/.conda/envs/pytorch_env/bin/python
PIP=/home/$USER/.conda/envs/pytorch_env/bin/pip

# ── Guard against numpy/scikit-image binary incompatibility ───────────────────
# Different nodes on Explorer can load different CUDA stacks, producing a numpy
# mismatch. Reinstall scikit-image against the active numpy if the import fails.
$PYTHON -c "from skimage.transform import resize" 2>/dev/null || {
    echo "scikit-image binary incompatible with current numpy — reinstalling..."
    $PIP install --force-reinstall scikit-image -q
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Forces Python to write each line to the log immediately.
# Without it, SLURM holds output back and the log stays empty until the job finishes.
export PYTHONUNBUFFERED=1

cd /home/$USER/vessel_seg

# ── Verify GPU + CUDA ──────────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA:      $($PYTHON -c 'import torch; print(torch.version.cuda)')"
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}  {torch.cuda.get_device_name(0)}')"
echo "Start:     $(date)"

# ── Paths ──────────────────────────────────────────────────────────────────────
# Predicts on all .tif images in INPUT_DIR (test + trainval, recursively).
# data_splits.json (next to CKPT_PATH) tags each W&B Media caption as [test] or [trainval].
# Defaults are used when running normally after training.
# Override via --export to point at a different folder, e.g. for inference on new data:
#   sbatch --export=ALL,CKPT_PATH=...,INPUT_DIR=inference_data/batch1,OUT_DIR=predictions/inference/batch1,INFERENCE_MODE=1 scripts/submit_predict.sh
INPUT_DIR="${INPUT_DIR:-data/images}"
MASK_DIR="${MASK_DIR:-data/masks}"     # for TP/FP/FN error overlay; omit to skip
# CKPT_PATH is set by submit_train.sh via --export=ALL,CKPT_PATH=checkpoints/<job_id>/best_model.pth
# The fallback is for manual runs only — in normal usage this is always set by the training job.
CKPT_PATH="${CKPT_PATH:-checkpoints/best_model.pth}"
OUT_DIR="${OUT_DIR:-predictions}"

# ── Predict ────────────────────────────────────────────────────────────────────
$PYTHON src/predict.py \
    --input_dir  $INPUT_DIR  \
    --mask_dir   $MASK_DIR   \
    --ckpt_path  $CKPT_PATH  \
    --out_dir    $OUT_DIR    \
    ${INFERENCE_MODE:+--inference_mode}

echo "Done: $(date)"

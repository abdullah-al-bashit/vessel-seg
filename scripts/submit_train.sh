#!/bin/bash
#SBATCH --job-name=vessel_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
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

# Use absolute path to conda env so the correct Python is always used,
# regardless of which node SLURM lands on. 'source activate' silently
# falls back to the system Python on some nodes (different CUDA stacks).
PYTHON=/home/$USER/.conda/envs/pytorch_env/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Forces Python to write each line to the log immediately.
# Without it, SLURM holds output back and the log stays empty until the job finishes.
export PYTHONUNBUFFERED=1

cd /home/$USER/vessel_seg

# ── W&B toggle (default: enabled) ─────────────────────────────────────────────
# Pass USE_WANDB=0 to disable: sbatch --export=ALL,USE_WANDB=0 scripts/submit_train.sh
[ "${USE_WANDB:-1}" = "0" ] && export WANDB_MODE=disabled && echo "W&B logging disabled"

# ── Clear wandb cache to prevent disk quota issues ────────────────────────────
rm -rf /home/$USER/.cache/wandb/
echo "Cleared ~/.cache/wandb/"

# ── Verify GPU + CUDA ──────────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA:      $($PYTHON -c 'import torch; print(torch.version.cuda)')"
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}  {torch.cuda.get_device_name(0)}')"
echo "Start:     $(date)"

# ── Paths ──────────────────────────────────────────────────────────────────────
# PREDICT_INPUT_DIR: where predict.py looks for images — overridable via --export.
# Training always uses data/images; predict defaults to the same unless overridden.
PREDICT_INPUT_DIR="${INPUT_DIR:-data/images}"
PREDICT_MASK_DIR="${MASK_DIR:-data/masks}"
# Auto-enable inference mode when predict target is not the training data dir
# (skips splits filtering so all images are predicted, not just the 7-image subset)
[ "$PREDICT_INPUT_DIR" != "data/images" ] && INFERENCE_MODE=1 || true
INPUT_DIR="data/images"
OUTPUT_DIR="data/masks"

# Job-specific checkpoint dir — each run gets its own folder so concurrent or
# sequential submits never overwrite each other's best model.
CKPT_DIR="checkpoints/$SLURM_JOB_ID"
mkdir -p $CKPT_DIR

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/config.yaml}"

# Copy config into checkpoint dir so the run is self-documenting — inspecting
# checkpoints/<job_id>/config.yaml tells you exactly what was run.
cp $CONFIG $CKPT_DIR/config.yaml

echo "Config:    $CONFIG"
echo "CKPT_DIR:  $CKPT_DIR"

# ── Train ─────────────────────────────────────────────────────────────────────
# Environment vars override config values — useful for quick tests and GPU optimization:
#   sbatch --export=ALL,EPOCHS=1 scripts/submit_train.sh
#   sbatch --export=ALL,BATCH_SIZE=96,NUM_WORKERS=32 scripts/submit_train.sh
#   sbatch --export=ALL,EPOCHS=20,FOLDS=1 scripts/submit_train.sh
$PYTHON src/train.py \
    --config     $CONFIG    \
    --input_dir  $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --ckpt_dir   $CKPT_DIR  \
    ${EPOCHS:+--epochs $EPOCHS} \
    ${FOLDS:+--folds $FOLDS} \
    ${BATCH_SIZE:+--batch_size $BATCH_SIZE} \
    ${NUM_WORKERS:+--num_workers $NUM_WORKERS} \
    ${WARMSTART_CKPT:+--warmstart_ckpt $WARMSTART_CKPT} \
    ${LAMBDA_TVERSKY:+--lambda_tversky $LAMBDA_TVERSKY} \
    ${LAMBDA_CLDICE:+--lambda_cldice $LAMBDA_CLDICE} \
    ${TVERSKY_BETA:+--tversky_beta $TVERSKY_BETA} \

echo "Training done: $(date)"

# ── Submit predict as a separate job dependent on this job's success ───────────
# --dependency=afterok:$SLURM_JOB_ID ensures predict only starts after successful exit.
# CKPT_PATH is passed via --export so predict.py uses this run's checkpoint, not
# a hardcoded path that could point to a different run's model.
PREDICT_JOB=$(sbatch --parsable \
    --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,CKPT_PATH=$CKPT_DIR/best_model.pth,INPUT_DIR=$PREDICT_INPUT_DIR,MASK_DIR=$PREDICT_MASK_DIR \
    scripts/submit_predict.sh)
echo "Predict job submitted: $PREDICT_JOB  (checkpoint: $CKPT_DIR/best_model.pth)"

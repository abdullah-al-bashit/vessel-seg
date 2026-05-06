#!/bin/bash
#SBATCH --job-name=vessel_train
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

# Job-specific checkpoint dir — each run gets its own folder so concurrent or
# sequential submits never overwrite each other's best model.
CKPT_DIR="checkpoints/$SLURM_JOB_ID"
mkdir -p $CKPT_DIR

# ── Config ─────────────────────────────────────────────────────────────────────
# Set CONFIG when submitting to pick the experiment YAML.
# All hyperparameters (epochs, lr, loss weights, blur aug, sam2_model, …) live
# in the YAML file — this script only handles cluster-specific paths.
#
# Current default: exp_A with base-plus model
#   sbatch scripts/submit.sh                  # runs exp_A (base-plus SAM2 encoder)
#
# Other experiments (commented out, previously tested blur/gate configurations):
#   # sbatch --export=ALL,CONFIG=configs/exp_B.yaml scripts/submit.sh  # B: plain hann + learnable sharp gate
#   # sbatch --export=ALL,CONFIG=configs/exp_C.yaml scripts/submit.sh  # C: aggressive blur + focus head
#   # sbatch --export=ALL,CONFIG=configs/exp_D.yaml scripts/submit.sh  # D: plain hann + aggressive blur
CONFIG="${CONFIG:-configs/exp_A.yaml}"

# Copy config into checkpoint dir so the run is self-documenting — inspecting
# checkpoints/<job_id>/config.yaml tells you exactly what was run.
cp $CONFIG $CKPT_DIR/config.yaml

echo "Config:    $CONFIG"
echo "CKPT_DIR:  $CKPT_DIR"

# ── Train ─────────────────────────────────────────────────────────────────────
# Environment vars override config values — useful for quick tests and GPU optimization:
#   sbatch --export=ALL,EPOCHS=1 scripts/submit.sh
#   sbatch --export=ALL,BATCH_SIZE=96,NUM_WORKERS=32 scripts/submit.sh
python src/train.py \
    --config     $CONFIG    \
    --input_dir  $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --ckpt_dir   $CKPT_DIR  \
    ${EPOCHS:+--epochs $EPOCHS} \
    ${BATCH_SIZE:+--batch_size $BATCH_SIZE} \
    ${NUM_WORKERS:+--num_workers $NUM_WORKERS}

echo "Training done: $(date)"

# ── Submit predict as a separate job dependent on this job's success ───────────
# --dependency=afterok:$SLURM_JOB_ID ensures predict only starts after successful exit.
# CKPT_PATH is passed via --export so predict.py uses this run's checkpoint, not
# a hardcoded path that could point to a different run's model.
PREDICT_JOB=$(sbatch --parsable \
    --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,CKPT_PATH=$CKPT_DIR/best_model.pth \
    scripts/submit_predict.sh)
echo "Predict job submitted: $PREDICT_JOB  (ckpt: $CKPT_DIR/best_model.pth)"

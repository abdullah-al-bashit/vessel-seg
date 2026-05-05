#!/bin/bash
#SBATCH --job-name=vessel_predict
#SBATCH --output=/home/a.bashit/vessel_seg/logs/predict_%j.out
#SBATCH --error=/home/a.bashit/vessel_seg/logs/predict_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a.bashit@northeastern.edu

module purge
module load explorer
module load anaconda3/2024.06
# (no system cuda — PyTorch ships its own)

source activate pytorch_env
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /home/$USER/vessel_seg

echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

# Predicts on all .tif images in INPUT_DIR (test + trainval, recursively).
# data_splits.json (next to CKPT_PATH) tags each W&B Media caption as [test] or [trainval].
INPUT_DIR="data/images"
MASK_DIR="data/masks"      # for TP/FP/FN error overlay; omit to skip
CKPT_PATH="checkpoints/fold1_best.pth"
OUT_DIR="predictions"

python src/predict.py \
    --input_dir  $INPUT_DIR  \
    --mask_dir   $MASK_DIR   \
    --ckpt_path  $CKPT_PATH  \
    --out_dir    $OUT_DIR

echo "Done: $(date)"

#!/bin/bash
#SBATCH --job-name=test_skeleton
#SBATCH --output=/home/a.bashit/vessel_seg/logs/test_skeleton_%j.out
#SBATCH --error=/home/a.bashit/vessel_seg/logs/test_skeleton_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

module purge
module load explorer
module load anaconda3/2024.06
source activate pytorch_env

cd /home/a.bashit/vessel_seg

echo "=== Test 1: Direct import ==="
python test_skeletonize.py

echo ""
echo "=== Test 2: Via model.py import ==="
python -c "from src.model import VesselSegNet; print('✓ model.py imports successfully')" || echo "✗ model.py import failed"

echo ""
echo "Done: $(date)"

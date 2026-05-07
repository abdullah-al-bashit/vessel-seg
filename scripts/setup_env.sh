#!/bin/bash
# Run ONCE interactively on Explorer to set up the conda environment.
#
# Usage:
#   srun --partition=gpu-interactive --gres=gpu:v100-sxm2:1 \
#        --cpus-per-task=4 --mem=16GB --time=01:00:00 --pty /bin/bash
#   bash scripts/setup_env.sh

module purge
module load explorer
module load anaconda3/2024.06
# Do NOT load system cuda — PyTorch ships its own CUDA/cuDNN.

# Create environment
conda create -n pytorch_env python=3.12 -y
source activate pytorch_env

# PyTorch 2.5.1 + CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# AttentionUNet backbone
pip install timm

# Data / image processing
pip install tifffile numpy scikit-image scipy \
            opencv-python-headless matplotlib tqdm

# Experiment tracking & config
pip install wandb pyyaml

# sklearn for cross-validation splits
pip install scikit-learn

# ── Verify ─────────────────────────────────────────────────────────────────────
python - << 'PYEOF'
import torch
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

import timm
m = timm.create_model('resnet34', pretrained=False, features_only=True, in_chans=3, out_indices=(0,1,2,3,4))
print(f'timm ResNet34 OK — output channels: {[f.num_chs for f in m.feature_info]}')

import skimage, numpy, wandb, yaml, sklearn
print(f'skimage={skimage.__version__}  numpy={numpy.__version__}  wandb={wandb.__version__}')
print('Environment OK')
PYEOF

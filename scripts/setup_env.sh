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
module load cuda/12.1.1

# Create environment
conda create -n vessel_seg python=3.12.4 -y
source activate vessel_seg

# PyTorch 2.4.1 + CUDA 12.1 — exact versions from NEU RC docs
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# HuggingFace — SAM2.1 support added in transformers 4.47
pip install transformers>=4.47.0 accelerate

# torch-geometric
pip install torch-geometric \
    torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

# Remaining deps
pip install tifffile numpy scikit-image scipy networkx sknw \
            opencv-python-headless matplotlib tqdm

# ── Verify ─────────────────────────────────────────────────────────────────────
python - << 'PYEOF'
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA avail:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device:  {torch.cuda.get_device_name(0)}')

import transformers
print(f'Transformers: {transformers.__version__}')

# Test HuggingFace SAM2 load (downloads ~2.5GB on first run)
from transformers import Sam2Model, Sam2Processor
print('Downloading SAM2.1 weights from HuggingFace...')
processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
model     = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")
print('SAM2.1 loaded successfully')

import torch_geometric
print(f'PyG:          {torch_geometric.__version__}')
PYEOF

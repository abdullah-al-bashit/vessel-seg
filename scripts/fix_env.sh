#!/bin/bash
# Fix numpy/scikit-image incompatibility in pytorch_env

module purge
module load explorer
module load anaconda3/2024.06

source activate pytorch_env

echo "Fixing scikit-image/numpy incompatibility..."
echo "Current numpy version:"
python -c "import numpy; print(numpy.__version__)" 2>&1 || echo "(import failed)"

# Uninstall scikit-image and reinstall to match current numpy
pip uninstall -y scikit-image
pip install --force-reinstall scikit-image

echo "Testing import..."
python -c "import numpy; import skimage; print(f'✓ Fixed! numpy={numpy.__version__}, scikit-image={skimage.__version__}')" && echo "SUCCESS" || echo "FAILED"

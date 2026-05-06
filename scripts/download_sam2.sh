#!/bin/bash
# One-time download of SAM2 weights to sam2_weights/ inside the project directory.
# Run this ONCE on the login node before submitting training jobs:
#
#   cd /home/a.bashit/vessel_seg
#   bash scripts/download_sam2.sh
#
# All subsequent train/predict/eval runs load from sam2_weights/ without
# touching HF Hub, so compute nodes don't need internet access or a warm cache.
# To download a different variant: bash scripts/download_sam2.sh facebook/sam2.1-hiera-small

SAM2_MODEL="${1:-facebook/sam2.1-hiera-tiny}"
SAM2_DIR="sam2_weights"   # relative to project root (wherever you run this from)

if [ -d "$SAM2_DIR" ] && [ "$(ls -A $SAM2_DIR 2>/dev/null)" ]; then
    echo "SAM2 weights already present at $SAM2_DIR — skipping download."
    ls -lh "$SAM2_DIR"
    exit 0
fi

mkdir -p "$SAM2_DIR"
echo "Downloading $SAM2_MODEL → $SAM2_DIR ..."

module purge
module load explorer
module load anaconda3/2024.06
source activate pytorch_env

python - "$SAM2_MODEL" "$SAM2_DIR" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
model_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
print(f"Done. Weights saved to {local_dir}")
PYEOF

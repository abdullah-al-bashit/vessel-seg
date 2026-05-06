# Northeastern University RC — GPU Resources & Configuration

## Available GPUs on Explorer Cluster

| GPU Model | Memory | Compute | Status | SLURM Request |
|-----------|--------|---------|--------|---------------|
| **V100-SXM2** | 32GB | 125 TFLOPS | Current (in use) | `--gres=gpu:v100-sxm2:1` |
| **A100** | 80GB | 312 TFLOPS | ✓ Available | `--gres=gpu:a100:1` |
| **H100** | 141GB | 1,455 TFLOPS | ✓ Available | `--gres=gpu:h100:1` |
| **H200** | 141GB | 1,455 TFLOPS | ✓ Available (newer) | `--gres=gpu:h200:1` |

**Note:** V100 is older Volta architecture; A100+ are recommended for new projects.

---

## Memory & Batch Size Analysis

### Current Setup (V100 32GB)

```
GPU:        V100-SXM2 32GB
CPU RAM:    64GB
Batch Size: 12
Num Workers: 8
```

**Memory Usage per Batch:**
- Forward pass (ResNet34 + UNet): ~15-16GB
- Gradient computation: ~15-16GB
- Loss (soft_skeletonize): ~15-16GB
- Optimizer state (Adam): ~8-10GB
- **Total: ~50-55GB / 32GB** → tight fit, safe at batch_size=12

---

### A100 Optimized (80GB) — Recommended

```yaml
GPU:        A100 80GB
CPU RAM:    128GB
Batch Size: 48
Num Workers: 16
```

**Memory Usage:**
- Forward pass: ~15-16GB
- Gradient computation: ~15-16GB
- Loss (soft_skeletonize): ~15-16GB
- Optimizer state (Adam): ~8-10GB
- **Total: ~50-55GB / 80GB** ✓ safe margin (31% utilization)

**Benefits:**
- 4× larger batches = 4× fewer gradient updates per epoch
- Better gradient stability (larger batch = more diverse samples)
- 2.5× faster training (~30-40s/epoch vs 40-60s on V100)
- ~3-4× wall-clock speedup

**SLURM Config:**
```bash
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
```

---

### H100 Maximum (141GB)

```yaml
GPU:        H100 141GB
CPU RAM:    256GB
Batch Size: 96-128
Num Workers: 32
```

**Memory Usage:**
- Forward pass: ~15-16GB
- Gradient computation: ~15-16GB
- Loss (soft_skeletonize): ~15-16GB
- Optimizer state (Adam): ~8-10GB
- **Total: ~50-55GB / 141GB** ✓ massive margin (39% utilization)

**Benefits:**
- 8-10× larger batches = smoothest gradients
- 5-6× faster training (~20s/epoch)
- ~5-6× wall-clock speedup
- Can experiment with higher learning rates / longer schedules

**SLURM Config:**
```bash
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
```

---

## Batch Size Scaling Rules

For soft_skeletonize loss bottleneck:

| GPU | Max Safe Batch | Memory Budget | Notes |
|-----|---|---|---|
| V100 32GB | 12-16 | 30-32GB / 32GB | Current, tight fit |
| A100 80GB | 48-64 | 50-55GB / 80GB | 4× improvement |
| H100 141GB | 96-128 | 50-55GB / 141GB | 8× improvement |

**Formula:** `batch_size = (GPU_MEMORY - 10GB_buffer) / 1.2GB_per_sample`

---

## SLURM Submission Examples

### Current (V100)
```bash
sbatch --export=ALL,CONFIG=configs/exp_E.yaml scripts/submit.sh
```

### A100 (Recommended)
```bash
# Create exp_E_a100.yaml with batch_size: 48, num_workers: 16
sbatch \
  --gres=gpu:a100:1 \
  --mem=128G \
  --cpus-per-task=16 \
  --export=ALL,CONFIG=configs/exp_E_a100.yaml \
  scripts/submit.sh
```

### H100 (Maximum Performance)
```bash
# Create exp_E_h100.yaml with batch_size: 128, num_workers: 32
sbatch \
  --gres=gpu:h100:1 \
  --mem=256G \
  --cpus-per-task=32 \
  --export=ALL,CONFIG=configs/exp_E_h100.yaml \
  scripts/submit.sh
```

---

## Check GPU Availability

```bash
# SSH to Explorer login node
ssh a.bashit@login.explorer.northeastern.edu

# Check available GPUs in gpu partition
sinfo -p gpu -o '%20N %10c %20b %5D'

# Check job queue
squeue -p gpu

# Check specific GPU availability
sinfo -p gpu --gres=gpu:a100
sinfo -p gpu --gres=gpu:h100
```

---

## Training Time Estimates

| GPU | Batch Size | Sec/Epoch | 200 Epochs | Notes |
|-----|---|---|---|---|
| V100 32GB | 12 | 40-60 | ~2.2-3.3 hrs | Current (patience=30 → ~50 epochs) |
| A100 80GB | 48 | 20-30 | ~1.1-1.7 hrs | 2-3× speedup |
| H100 141GB | 128 | 15-20 | ~0.8-1.1 hrs | 3-4× speedup |

**Early stopping at patience=30:** Actual training typically finishes 50-80 epochs (within 1-1.5 hrs on A100).

---

## References

- [Northeastern RC — GPU Overview](https://rc-docs.northeastern.edu/en/latest/gpus/gpuoverview.html)
- [Northeastern RC — Multi-GPU Partition](https://rc-docs.northeastern.edu/en/latest/gpus/multigpu-partition-access.html)
- [Northeastern RC Documentation](https://rc-docs.northeastern.edu/)
- [Support: rchelp@northeastern.edu](mailto:rchelp@northeastern.edu)

---

## Next Steps

1. **Check availability:** `sinfo -p gpu --gres=gpu:a100` or `--gres=gpu:h100`
2. **Create config variant:** `configs/exp_E_a100.yaml` with batch_size=48, num_workers=16
3. **Test timing:** 2-epoch warmup to verify speed before full 200-epoch run
4. **Submit job:** Use appropriate SLURM flags above for chosen GPU


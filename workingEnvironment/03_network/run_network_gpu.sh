#!/bin/bash
#SBATCH --job-name=CLR_GPU
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --partition=gpu        # Use GPU partition (adjust to your cluster)
#SBATCH --output=clr_gpu_%j.out
#SBATCH --error=clr_gpu_%j.err

# ============================================================================
# GPU-Accelerated Gene Network Analysis
# 
# This script runs the MI+CLR pipeline with GPU acceleration for CLR step
# Expected runtime: 2-3 hours (MI on CPU) + 10-60 seconds (CLR on GPU)
# ============================================================================

echo "================================================"
echo "Gene Network Analysis - GPU Accelerated CLR"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "================================================"

# Activate virtual environment
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate

# Set temporary directory to scratch
export TMPDIR=/scratch/$USER/pip_tmp_$$
mkdir -p $TMPDIR

# GPU Information
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# Check if GPU libraries are available
echo "Checking GPU libraries..."
python -c "
try:
    import cupy as cp
    print(f'✓ CuPy installed: {cp.__version__}')
    print(f'  CUDA version: {cp.cuda.runtime.runtimeGetVersion()}')
except ImportError:
    print('✗ CuPy not installed')

try:
    import torch
    print(f'✓ PyTorch installed: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('✗ PyTorch not installed')
"
echo ""

# Run the analysis with GPU CLR
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network

echo "Starting network analysis with GPU-accelerated CLR..."
python generate_net_python_gpu.py

# Clean up
rm -rf $TMPDIR

echo ""
echo "================================================"
echo "Job completed: $(date)"
echo "================================================"

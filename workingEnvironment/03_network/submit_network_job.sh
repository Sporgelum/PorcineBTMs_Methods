#!/bin/bash
#SBATCH --job-name=gene_network
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
##SBATCH --mem=527G
#SBATCH --mem-per-cpu=6G
#SBATCH --output=slurm_%j_network_analysis.out
#SBATCH --error=slurm_%j_network_analysis.err
#SBATCH --partition=pgpu       
#SBATCH --gres=gpu:1          # Request 1 GPU (optional, for GPU-accelerated CLR)


# ============================================================================
# Gene Network Inference Analysis - SLURM Job Script
# This script runs the complete network inference pipeline with logging
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "Gene Network Inference: MI+CLR vs GRNBoost2"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "============================================================================"
echo ""

# Navigate to working directory
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network

# Activate Python environment
echo "[INFO] Activating Python environment..."
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate
echo "[INFO] Using Python: $(which python)"
echo "[INFO] Python version: $(python --version)"
echo ""

# Set up scratch space for temporary files
echo "[INFO] Setting up temporary directory..."
export TMPDIR="/scratch/$USER/network_tmp_$$"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"
echo "[INFO] Temporary directory: $TMPDIR"
echo ""

# Clean pip cache to avoid disk quota issues
echo "[INFO] Cleaning pip cache..."
pip cache purge 2>/dev/null || true
echo ""

# System information
echo "[INFO] System Information:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "  Disk space (home): $(df -h ~ | tail -1 | awk '{print $4}' | head -1)"
echo "  Disk space (scratch): $(df -h /scratch | tail -1 | awk '{print $4}')"
echo ""

# Check for GPU (optional, for GPU-accelerated CLR)
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Load PyTorch module for GPU acceleration
# IMPORTANT: Must be loaded BEFORE running Python script
echo "[INFO] Loading PyTorch module for GPU acceleration..."
if module load PyTorch 2>&1; then
    echo "[INFO] ✓ PyTorch module loaded successfully"
    module list 2>&1 | grep PyTorch || true
else
    echo "[WARNING] Could not load PyTorch module - GPU acceleration will not be available"
fi
echo ""

# Re-activate the virtualenv AFTER module loads to ensure its packages take priority.
# 'module load PyTorch' contaminates PYTHONPATH with system packages compiled against
# NumPy 1.x (e.g. numexpr in SciPy-bundle). When dask spawns worker sub-processes
# for GRNBoost2 those workers inherit PYTHONPATH and crash with NumPy 2.x in the venv.
# Solution: reset PYTHONPATH to ONLY the venv site-packages.
echo "[INFO] Re-activating virtualenv and resetting PYTHONPATH to venv only..."
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate
VENV_SITE="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages"
export PYTHONPATH="${VENV_SITE}"
# Disable numexpr as an extra safety net (pandas treats it as optional anyway)
export NUMEXPR_DISABLED=1
echo "[INFO] Using Python: $(which python)"
echo "[INFO] PYTHONPATH: $PYTHONPATH"
echo "[INFO] NUMEXPR_DISABLED: $NUMEXPR_DISABLED"
echo ""

# Verify PyTorch is importable from Python
echo "[INFO] Verifying PyTorch import..."
if python -c "import torch; print(f'✓ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')" 2>&1; then
    echo "[INFO] ✓ PyTorch verification successful"
else
    echo "[WARNING] PyTorch import failed - installing pkg_resources compatibility..."
    pip install -q setuptools
    
    # Try again
    if python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>&1; then
        echo "[INFO] ✓ PyTorch now working after setuptools install"
    else
        echo "[WARNING] PyTorch still not working - GPU acceleration may not be available"
        echo "[INFO] Analysis will continue with CPU-only CLR"
    fi
fi
echo ""

# Run the analysis
echo "============================================================================"
echo "Starting Network Inference Analysis"
echo "============================================================================"
echo ""

# Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=1

# Run with timing
START_TIME=$(date +%s)

python -u generate_net_python.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Calculate runtime
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================================"
echo "Job Complete!"
echo "============================================================================"
echo "Ended: $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================================================"
echo ""

# Clean up temporary files
echo "[INFO] Cleaning up temporary files..."
rm -rf "$TMPDIR"
echo "[INFO] Cleanup complete"
echo ""

# List output files
echo "Output files generated:"
ls -lh *.mtx *.txt *.tsv *.graphml *.log 2>/dev/null | tail -20 || echo "  (no files found)"
echo ""

echo "============================================================================"
echo "Check the following files for results:"
echo "  - network_analysis_*.log     : Complete execution log"
echo "  - analysis_report_*.txt      : Summary report with timings"
echo "  - slurm_${SLURM_JOB_ID}_*.out/err : SLURM job output"
echo "============================================================================"

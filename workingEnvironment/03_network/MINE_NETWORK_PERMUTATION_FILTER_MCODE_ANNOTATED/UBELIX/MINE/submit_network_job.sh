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
echo "Gene Network Inference: MINE + MODULE"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "============================================================================"
echo ""

# Navigate to working directory
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED

# Explicit input/output paths
COUNTS_FILE="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/logCPM_matrix_filtered_samples.csv"
META_FILE="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/metadata_with_sample_annotations.csv"
OUTPUT_DIR="$(pwd)/output"

# Optional hub prefiltering (recommended for cleaner network topology).
# Set to 1 to enable, 0 to disable.
FILTER_RIBOSOMAL=0
FILTER_MIRNA=0
EXCLUDE_GENE_REGEX=""
EXCLUDE_GENES_FILE=""

GENE_FILTER_ARGS=()
if [[ "$FILTER_RIBOSOMAL" -eq 1 ]]; then
    GENE_FILTER_ARGS+=("--filter-ribosomal")
fi
if [[ "$FILTER_MIRNA" -eq 1 ]]; then
    GENE_FILTER_ARGS+=("--filter-mirna")
fi
if [[ -n "$EXCLUDE_GENE_REGEX" ]]; then
    GENE_FILTER_ARGS+=("--exclude-gene-regex" "$EXCLUDE_GENE_REGEX")
fi
if [[ -n "$EXCLUDE_GENES_FILE" ]]; then
    GENE_FILTER_ARGS+=("--exclude-genes-file" "$EXCLUDE_GENES_FILE")
fi

echo "[INFO] Counts file: $COUNTS_FILE"
echo "[INFO] Metadata file: $META_FILE"
echo "[INFO] Output directory: $OUTPUT_DIR"

if [[ ! -f "$COUNTS_FILE" ]]; then
    echo "[ERROR] Counts file not found: $COUNTS_FILE"
    exit 1
fi

if [[ ! -f "$META_FILE" ]]; then
    echo "[ERROR] Metadata file not found: $META_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo ""

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

# Re-activate virtualenv after module load. Keep only venv + PyTorch module
# site-packages in PYTHONPATH to avoid pulling incompatible NumPy-1.x bundles.
echo "[INFO] Re-activating virtualenv and setting clean PYTHONPATH (venv + torch module)..."
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate
VENV_SITE="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages"
if [[ -n "${EBROOTPYTORCH:-}" ]]; then
    TORCH_SITE="${EBROOTPYTORCH}/lib/python3.9/site-packages"
    export PYTHONPATH="${VENV_SITE}:${TORCH_SITE}"
else
    # Fallback if module env var is unavailable on this cluster node.
    export PYTHONPATH="${VENV_SITE}:${PYTHONPATH:-}"
fi
# Disable numexpr as an extra safety net (pandas treats it as optional anyway)
export NUMEXPR_DISABLED=1
echo "[INFO] Using Python: $(which python)"
echo "[INFO] PYTHONPATH: $PYTHONPATH"
echo "[INFO] NUMEXPR_DISABLED: $NUMEXPR_DISABLED"
echo ""

# Ensure pkg_resources is available (required by cluster PyTorch module).
echo "[INFO] Checking setuptools/pkg_resources availability..."
if python -c "import pkg_resources" 2>/dev/null; then
    echo "[INFO] ✓ pkg_resources available"
else
    echo "[WARNING] pkg_resources missing - installing compatible setuptools (<81) in venv..."
    pip install -q "setuptools<81"
    python -c "import pkg_resources" || {
        echo "[ERROR] pkg_resources still unavailable after setuptools install"
        exit 1
    }
    echo "[INFO] ✓ pkg_resources available after setuptools install"
fi
echo ""

# PyTorch/1.10 module is compiled against NumPy 1.x.
echo "[INFO] Checking NumPy compatibility for cluster PyTorch..."
NUMPY_MAJOR=$(python -c "import numpy as np; print(np.__version__.split('.')[0])")
if [[ "$NUMPY_MAJOR" -ge 2 ]]; then
    echo "[WARNING] NumPy >=2 detected; installing numpy<2 for PyTorch 1.10 compatibility..."
    pip install -q "numpy<2"
fi
echo "[INFO] NumPy version: $(python -c 'import numpy as np; print(np.__version__)')"
echo ""

# Verify PyTorch is importable from Python (hard requirement)
echo "[INFO] Verifying PyTorch import..."
if python -c "import torch; print(f'✓ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')" 2>&1; then
    echo "[INFO] ✓ PyTorch verification successful"
else
    echo "[ERROR] PyTorch import failed after module load + venv activation."
    echo "[ERROR] Debugging Python path and module visibility:"
    python - <<'PY'
import sys
print("Python executable:", sys.executable)
print("First 10 sys.path entries:")
for p in sys.path[:10]:
    print("  ", p)
PY
    exit 1
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

python run_pipeline.py \
    --counts "$COUNTS_FILE" \
    --meta "$META_FILE" \
    --output "$OUTPUT_DIR" \
    --device cuda \
    --perms 10000 \
    --mode global \
    --pval 0.001 \
    --epochs 200 \
    --batch-pairs 512 \
    --prescreen-threshold 0.3 \
    --max-pairs 50000000 \
    --prescreen-method "spearman" \
    "${GENE_FILTER_ARGS[@]}"
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
find "$OUTPUT_DIR" -maxdepth 2 -type f | tail -20 || echo "  (no files found)"
echo ""

echo "============================================================================"
echo "Check the following files for results:"
echo "  - network_analysis_*.log     : Complete execution log"
echo "  - analysis_report_*.txt      : Summary report with timings"
echo "  - output/                    : Pipeline result files"
echo "  - slurm_${SLURM_JOB_ID}_*.out/err : SLURM job output"
echo "============================================================================"

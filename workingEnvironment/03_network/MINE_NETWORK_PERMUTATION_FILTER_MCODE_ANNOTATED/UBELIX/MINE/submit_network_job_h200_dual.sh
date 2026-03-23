#!/bin/bash
#SBATCH --job-name=MINE_H200_dual
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6G
#SBATCH --output=mine_h200_dual_%j.log
#SBATCH --error=mine_h200_dual_%j.err
#SBATCH --partition=pgpu
#SBATCH --nodelist=gnode36,gnode37        # Target H200 nodes
#SBATCH --gres=gpu:h200:2                 # Request 2 H200 GPUs on same node
#SBATCH --nodes=1

# ============================================================================
# MINE Gene Network Inference - H200 Dual-GPU Optimized Job
# Date: 2026-03-22
# Config: Auto-batch sizing + Mixed precision + Dual-GPU parallelism
#
# Expected runtime: ~1.5-2 hours for full dataset
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "MINE GENE NETWORK INFERENCE - H200 DUAL-GPU OPTIMIZED"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs Requested: gpu:h200:2"
echo "Started: $(date)"
echo "============================================================================"
echo ""

# Navigate to working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Paths (update these to your environment)
COUNTS_FILE="/storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/workingEnvironment/02_counts/logCPM_matrix_filtered_samples.csv"
META_FILE="/storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/workingEnvironment/02_counts/metadata_with_sample_annotations.csv"
OUTPUT_DIR="$(pwd)/output/h200_dual_$(date +%Y%m%d_%H%M%S)"

# Runtime knobs (override at submit time, e.g. MAX_PAIRS=1000000 sbatch ...)
MAX_PAIRS="${MAX_PAIRS:-500000000}"
BATCH_PAIRS="${BATCH_PAIRS:-auto}"   # keep "auto" for VRAM-aware sizing, or set fixed int
USE_MIXED_PRECISION="${USE_MIXED_PRECISION:-1}"

echo "[INFO] Counts file: $COUNTS_FILE"
echo "[INFO] Metadata file: $META_FILE"
echo "[INFO] Output directory: $OUTPUT_DIR"
echo "[INFO] Max pairs: $MAX_PAIRS"
echo "[INFO] Batch pairs: $BATCH_PAIRS"
echo "[INFO] Mixed precision: $USE_MIXED_PRECISION"

# Verify inputs exist
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
source /storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/bin/activate
echo "[INFO] Python: $(which python)"
echo "[INFO] Python version: $(python --version)"
echo ""

# Setup temporary directory
echo "[INFO] Setting up temporary directory..."
export TMPDIR="/scratch/$USER/mine_tmp_$$"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"
echo "[INFO] Temporary directory: $TMPDIR"
echo ""

# System information
echo "[INFO] Computing Resources:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "  Disk (home): $(df -h ~ 2>/dev/null | tail -1 | awk '{print $4}' || echo 'N/A')"
echo "  Disk (scratch): $(df -h /scratch 2>/dev/null | tail -1 | awk '{print $4}' || echo 'N/A')"
echo ""

# GPU information
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] GPU Information:"
    echo "  GPUs detected: $(nvidia-smi --list-gpus | wc -l)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "[WARNING] nvidia-smi not found; GPU detection skipped"
fi

# Load PyTorch module (if available on this cluster)
echo "[INFO] Loading PyTorch environment..."
if module load PyTorch 2>&1 | grep -q "PyTorch"; then
    echo "[INFO] ✓ PyTorch module loaded"
else
    echo "[INFO] ℹ PyTorch module not available (using venv PyTorch)"
fi
echo ""

# Re-activate venv after module load
source /storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/bin/activate
echo "[INFO] Python after module load: $(which python)"
echo ""

# Verify PyTorch and CUDA
echo "[INFO] Verifying PyTorch + CUDA setup..."
python << 'PYEOF'
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"    Device {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
else:
    print("  ERROR: CUDA not available!")
    import sys
    sys.exit(1)
PYEOF
echo ""

# Run compatibility test (optional but recommended)
echo "[INFO] Running PyTorch compatibility validation..."
if python test_pytorch_compat.py 2>&1 | head -50; then
    echo "[INFO] ✓ Environment validation passed"
else
    echo "[WARNING] Environment validation had issues; continuing anyway..."
fi
echo ""

# ============================================================================
# Pipeline Execution - H200 Optimized Configuration
# ============================================================================

echo "============================================================================"
echo "Starting MINE Pipeline"
echo "Configuration:"
echo "  - Batch sizing: $BATCH_PAIRS"
echo "  - Mixed precision flag: $USE_MIXED_PRECISION"
echo "  - Study GPU workers: 2 (concurrent studies on 2 GPUs)"
echo "  - Permutations: 30000"
echo "  - Max pairs: $MAX_PAIRS"
echo "============================================================================"
echo ""

export PYTHONUNBUFFERED=1
START_TIME=$(date +%s)

MP_ARGS=()
if [[ "$USE_MIXED_PRECISION" -eq 1 ]]; then
    MP_ARGS+=("--mixed-precision")
fi

# PRIMARY: Full-feature pipeline preserving previously used arguments.
python run_pipeline.py \
    --counts "$COUNTS_FILE" \
    --meta "$META_FILE" \
    --output "$OUTPUT_DIR" \
    --device cuda \
    --batch-pairs "$BATCH_PAIRS" \
    "${MP_ARGS[@]}" \
    --study-gpu-workers 2 \
    --study-gpu-devices cuda:0 cuda:1 \
    --epochs 150 \
    --perms 30000 \
    --mode global \
    --pval 0.05 \
    --prescreen-threshold 0.3 \
    --max-pairs "$MAX_PAIRS" \
    --prescreen-method spearman \
    --mad-top-genes 32763 \
    --qc-preplot \
    --qc-postplot \
    --min-studies 2 \
    --module-method leiden \
    --module-leiden-resolution 1.2 \
    --module-min-size 10 \
    --master-edge-weight mean_neglog10p \
    --normalize-weights \
    --submodule-method mcode \
    --submodule-size-threshold 200 \
    --submodule-min-size 10 \
    --submodule-mcode-score-threshold 0.2 \
    --submodule-mcode-min-density 0.3 \
    --ortholog-map ../gene_id_mapping.tsv \
    --ortholog-source-col ensembl_gene_id \
    --ortholog-target-col external_gene_name \
    --download-gmt \
    --module-export-map ../gene_id_mapping.tsv \
    --module-export-key-col ensembl_gene_id \
    --module-export-cols external_gene_name entrezgene_id \
    --save-per-gmt-enrichments \
    --include-network-visualization \
    2>&1

PIPELINE_EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================================"
if [[ $PIPELINE_EXIT_CODE -eq 0 ]]; then
    echo "✓ PIPELINE COMPLETED SUCCESSFULLY"
else
    echo "✗ PIPELINE FAILED (exit code: $PIPELINE_EXIT_CODE)"
fi
echo "============================================================================"
echo "Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Completed: $(date)"
echo "============================================================================"
echo ""

# List output files
echo "[INFO] Output files:"
if [[ -d "$OUTPUT_DIR" ]]; then
    find "$OUTPUT_DIR" -maxdepth 2 -type f 2>/dev/null | head -20 || echo "  (directory exists but no files found)"
else
    echo "  (output directory not created)"
fi
echo ""

# Cleanup
echo "[INFO] Cleaning up temporary files..."
rm -rf "$TMPDIR"
echo "[INFO] Temporary directory removed"
echo ""

# Final report
echo "============================================================================"
echo "SUMMARY"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2× H200 (141GB each)"
echo "Output directory: $OUTPUT_DIR"
echo "Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Status: $([ $PIPELINE_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "============================================================================"

exit $PIPELINE_EXIT_CODE

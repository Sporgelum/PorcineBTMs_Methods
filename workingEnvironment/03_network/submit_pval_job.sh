#!/bin/bash
#SBATCH --job-name=gene_network_pval
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6G
#SBATCH --output=slurm_%j_network_pval.out
#SBATCH --error=slurm_%j_network_pval.err
#SBATCH --partition=pgpu          # CPU-only job – no GPU needed

# ============================================================================
# Gene Network Inference: Permutation-based MI Significance + MCODE modules
# Runs generate_net_python_pval.py via SLURM
#
# Key differences from submit_network_job.sh:
#   - No GPU / PyTorch sections  (no CLR matrix, no GRNBoost2)
#   - No PYTHONPATH reset        (no dask worker subprocesses)
#   - No NUMEXPR workaround      (not imported by this pipeline)
#   - Outputs go to 03_network/NETS_MI_PVAL/  (set inside the Python script)
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "Gene Network Inference: Permutation MI + MCODE modules"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "============================================================================"
echo ""

# Navigate to working directory
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network

# Activate Python virtual environment
echo "[INFO] Activating Python environment..."
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate
echo "[INFO] Using Python: $(which python)"
echo "[INFO] Python version: $(python --version)"
echo ""

# Set up scratch space for temporary files (joblib uses TMPDIR for memmap)
echo "[INFO] Setting up temporary directory..."
export TMPDIR="/scratch/$USER/network_pval_tmp_$$"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"
echo "[INFO] Temporary directory: $TMPDIR"
echo ""

# System information
echo "[INFO] System Information:"
echo "  CPU cores available : $(nproc)"
echo "  Memory total        : $(free -h | grep Mem | awk '{print $2}')"
echo "  Disk (scratch)      : $(df -h /scratch | tail -1 | awk '{print $4}')"
echo ""

# ============================================================================
# Run the analysis
# ============================================================================
echo "============================================================================"
echo "Starting Permutation MI Network Inference"
echo "============================================================================"
echo ""

export PYTHONUNBUFFERED=1     # real-time log output

START_TIME=$(date +%s)

python -u generate_net_python_pval.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
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

# List output files from the NETS_MI_PVAL output directory
OUTPUT_DIR="NETS_MI_PVAL"
echo "Output files in ${OUTPUT_DIR}/:"
ls -lh "${OUTPUT_DIR}"/*.{tsv,txt,mtx,graphml,log,npy} 2>/dev/null \
    | awk '{print "  " $5, $9}' \
    || echo "  (no files found)"
echo ""

echo "============================================================================"
echo "Check the following for results:"
echo "  ${OUTPUT_DIR}/network_pval_*.log          Complete execution log"
echo "  ${OUTPUT_DIR}/analysis_report_pval_*.txt  Summary report with timings"
echo "  ${OUTPUT_DIR}/master_network.graphml      Master network (Cytoscape)"
echo "  ${OUTPUT_DIR}/master_BTM_modules.tsv      MCODE module membership"
echo "  ${OUTPUT_DIR}/master_network_edgelist.tsv Edge list with study counts"
echo "  slurm_${SLURM_JOB_ID}_network_pval.out/err  SLURM job output"
echo "============================================================================"

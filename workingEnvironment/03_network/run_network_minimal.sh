#!/bin/bash
#
# Run MI+CLR Network Inference (Minimal Version - No Disk Issues)
# Skips GRNBoost2 to avoid large package downloads
#

set -e  # Exit on error

echo "=========================================================================="
echo "Gene Network Inference: MI+CLR (Minimal Install)"
echo "=========================================================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check quota
echo "[INFO] Checking disk quota..."
quota -s 2>/dev/null || echo "[INFO] Cannot check quota, proceeding anyway"
echo ""

# Step 1: Set up SCRATCH for temporary files (avoid /tmp quota)
echo "[INFO] Setting up temporary directory in /scratch..."
export TMPDIR="/scratch/$USER/pip_tmp_$$"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"
echo "[INFO] Using temporary directory: $TMPDIR"
echo ""

# Step 2: Clean pip cache to free up space
echo "[INFO] Cleaning pip cache..."
pip cache purge || pip cache remove '*' 2>/dev/null || echo "[INFO] Cache already clean"
echo ""

# Step 3: Check if Python environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "[WARNING] No Python virtual environment detected."
    echo "[INFO] Attempting to activate environment..."
    
    PARENT_ENV="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+"
    if [[ -f "$PARENT_ENV/bin/activate" ]]; then
        source "$PARENT_ENV/bin/activate"
        echo "[INFO] Environment activated: $VIRTUAL_ENV"
    else
        echo "[ERROR] Cannot find virtual environment. Please activate it manually:"
        echo "        source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate"
        exit 1
    fi
fi

# Step 4: Install minimal packages (no GRNBoost2 to save space)
echo ""
echo "[INFO] Installing minimal required Python packages..."
echo "[INFO] Note: Skipping GRNBoost2 to avoid disk quota issues"
echo ""

# Install with --no-cache-dir to avoid filling /tmp
pip install --no-cache-dir -r requirements_minimal.txt

echo ""
echo "[INFO] Packages installed successfully!"

# Step 5: Check CPU cores
N_CORES=$(nproc 2>/dev/null || echo "Unknown")
echo ""
echo "[INFO] System has $N_CORES CPU cores available"
echo "[INFO] The script will use all cores for parallel MI+CLR computation"

# Step 6: Run the analysis (MI+CLR only)
echo ""
echo "=========================================================================="
echo "Starting MI+CLR Network Inference"
echo "=========================================================================="
echo ""

python generate_net_python.py

# Step 7: Clean up temporary files
echo ""
echo "[INFO] Cleaning up temporary files from /scratch..."
rm -rf "$TMPDIR"

# Step 8: Show results
echo ""
echo "=========================================================================="
echo "Analysis Complete!"
echo "=========================================================================="
echo ""
echo "Results saved in: $SCRIPT_DIR"
echo ""
echo "Output files:"
echo "  - CLR_adjacency_matrix_mi_clr_python.mtx"
echo "  - CLR_network_edgelist_mi_clr_python.txt"
echo "  - CLR_network_weighted_mi_clr_python.txt"
echo "  - CLR_network_mi_clr_python.graphml (for Cytoscape)"
echo "  - BTM_modules_mi_clr_python.tsv"
echo ""
echo "Note: GRNBoost2 was skipped due to disk quota limitations."
echo "      MI+CLR analysis provides the same results as your R code."
echo ""

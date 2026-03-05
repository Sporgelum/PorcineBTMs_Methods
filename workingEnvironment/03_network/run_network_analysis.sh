#!/bin/bash
#
# Run Gene Network Inference Analysis
# This script installs dependencies and runs both MI+CLR and GRNBoost2
#

set -e  # Exit on error

echo "=========================================================================="
echo "Gene Network Inference: MI+CLR vs GRNBoost2"
echo "=========================================================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Check if Python environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "[WARNING] No Python virtual environment detected."
    echo "[INFO] Attempting to activate environment..."
    
    # Try to activate the parent environment
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

# Step 2: Set up SCRATCH for temporary files (avoid /tmp quota)
echo ""
echo "[INFO] Setting up temporary directory in /scratch..."
export TMPDIR="/scratch/$USER/pip_tmp_$$"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"
echo "[INFO] Using temporary directory: $TMPDIR"

# Step 3: Install required packages
echo ""
echo "[INFO] Checking and installing required Python packages..."
pip install -q --upgrade pip --no-cache-dir

if [[ -f requirements_network.txt ]]; then
    echo "[INFO] Installing from requirements_network.txt..."
    pip install -r requirements_network.txt --no-cache-dir
else
    echo "[INFO] Installing packages individually..."
    pip install numpy pandas scipy scikit-learn joblib networkx python-igraph --no-cache-dir
    
    # Try to install arboreto (may fail on some systems)
    echo "[INFO] Attempting to install arboreto (for GRNBoost2)..."
    pip install arboreto dask[complete] distributed --no-cache-dir || {
        echo "[WARNING] Failed to install arboreto. GRNBoost2 will be skipped."
        echo "[INFO] You can run MI+CLR only, which is still very useful!"
    }
fi

# Step 4: Check CPU cores
N_CORES=$(nproc)
echo ""
echo "[INFO] System has $N_CORES CPU cores available"
echo "[INFO] The script will use all cores for parallel processing"

# Step 5: Run the analysis
echo ""
echo "=========================================================================="
echo "Starting Network Inference Analysis"
echo "=========================================================================="
echo ""

export PYTHONUNBUFFERED=1
python -u generate_net_python.py

#adding u  allows to see live unbuffered progrees on screens, full log saved aand no buffering delays..


# Step 6: Clean up temporary files
echo ""
echo "[INFO] Cleaning up temporary files from /scratch..."
rm -rf "$TMPDIR"

# Step 7: Show results
echo ""
echo "=========================================================================="
echo "Analysis Complete!"
echo "=========================================================================="
echo ""
echo "Results saved in: $SCRIPT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check network_comparison.txt for overlap between methods"
echo "  2. Load .graphml files into Cytoscape for visualization"
echo "  3. Compare BTM_modules_*_python.tsv files between methods"
echo ""

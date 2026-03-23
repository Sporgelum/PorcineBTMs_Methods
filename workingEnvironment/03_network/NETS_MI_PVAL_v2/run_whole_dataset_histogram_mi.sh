#!/bin/bash
set -euo pipefail

# Full-feature wrapper for NETS_MI_PVAL_v2 (classical MI, no GPU required)
#
# Examples:
#   ./run_whole_dataset_histogram_mi.sh
#   MAX_PAIRS=50000000 N_JOBS=-1 ./run_whole_dataset_histogram_mi.sh
#   N_BINS=5 MI_STRATEGY=quantile ./run_whole_dataset_histogram_mi.sh
#   VERBOSE=1 ./run_whole_dataset_histogram_mi.sh

PROJECT_DIR="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/NETS_MI_PVAL_v2"
COUNTS_FILE="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/logCPM_matrix_filtered_samples.csv"
META_FILE="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/metadata_with_sample_annotations.csv"

# Pre-screening (correlation-based candidate filtering)
PRESCREEN_THRESHOLD="${PRESCREEN_THRESHOLD:-0.3}"
PRESCREEN_METHOD="${PRESCREEN_METHOD:-spearman}"
MAX_PAIRS="${MAX_PAIRS:-300000000}"
MAD_TOP_GENES="${MAD_TOP_GENES:-32763}"
MIN_STUDIES="${MIN_STUDIES:-2}"
MIN_STUDIES_FRACTION="${MIN_STUDIES_FRACTION:-}"

# Classical MI parameters
N_BINS="${N_BINS:-5}"
MI_STRATEGY="${MI_STRATEGY:-quantile}"
N_JOBS="${N_JOBS:--1}"  # -1 = all CPU cores

# Output and resume behavior
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/wholde_dataset_histogram_mi_${MAX_PAIRS}}"
NO_RESUME_STUDIES="${NO_RESUME_STUDIES:-0}"
NO_REUSE_MI_SCORES="${NO_REUSE_MI_SCORES:-0}"
NO_SAVE_MI_CACHE="${NO_SAVE_MI_CACHE:-0}"
VERBOSE="${VERBOSE:-0}"  # Set to 1 for verbose MI progress output

# Permutation testing
PERMS="${PERMS:-30000}"
MODE="${MODE:-global}"
PVAL="${PVAL:-0.001}"

# Network and module detection
MODULE_METHOD="${MODULE_METHOD:-leiden}"
MODULE_LEIDEN_RESOLUTION="${MODULE_LEIDEN_RESOLUTION:-1.2}"
MODULE_MIN_SIZE="${MODULE_MIN_SIZE:-10}"
MASTER_EDGE_WEIGHT="${MASTER_EDGE_WEIGHT:-mean_neglog10p}"
NORMALIZE_WEIGHTS="${NORMALIZE_WEIGHTS:-1}"

# Submodule refinement
SUBMODULE_METHOD="${SUBMODULE_METHOD:-mcode}"
SUBMODULE_SIZE_THRESHOLD="${SUBMODULE_SIZE_THRESHOLD:-200}"
SUBMODULE_MIN_SIZE="${SUBMODULE_MIN_SIZE:-10}"
SUBMODULE_MCODE_SCORE_THRESHOLD="${SUBMODULE_MCODE_SCORE_THRESHOLD:-0.2}"
SUBMODULE_MCODE_MIN_DENSITY="${SUBMODULE_MCODE_MIN_DENSITY:-0.3}"

# Gene ID mapping (pig to human)
ORTHOLOG_MAP="${ORTHOLOG_MAP:-${PROJECT_DIR}/gene_id_mapping.tsv}"
ORTHOLOG_SOURCE_COL="${ORTHOLOG_SOURCE_COL:-ensembl_gene_id}"
ORTHOLOG_TARGET_COL="${ORTHOLOG_TARGET_COL:-external_gene_name}"
MODULE_EXPORT_MAP="${MODULE_EXPORT_MAP:-${ORTHOLOG_MAP}}"
MODULE_EXPORT_KEY_COL="${MODULE_EXPORT_KEY_COL:-ensembl_gene_id}"
MODULE_EXPORT_COLS="${MODULE_EXPORT_COLS:-external_gene_name entrezgene_id}"

# Gene-set enrichment
DOWNLOAD_GMT="${DOWNLOAD_GMT:-1}"
SAVE_PER_GMT_ENRICHMENTS="${SAVE_PER_GMT_ENRICHMENTS:-1}"
INCLUDE_NETWORK_VISUALIZATION="${INCLUDE_NETWORK_VISUALIZATION:-1}"

# Activate virtual environment
cd "$PROJECT_DIR"
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate

export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages
export PYTHONPATH=${VENV_SITE}
export NUMEXPR_DISABLED=1

# Build command-line arguments
EXTRA_ARGS=()
if [[ "$NORMALIZE_WEIGHTS" -eq 1 ]]; then EXTRA_ARGS+=(--normalize-weights); fi
if [[ "$DOWNLOAD_GMT" -eq 1 ]]; then EXTRA_ARGS+=(--download-gmt); fi
if [[ "$SAVE_PER_GMT_ENRICHMENTS" -eq 1 ]]; then EXTRA_ARGS+=(--save-per-gmt-enrichments); fi
if [[ "$INCLUDE_NETWORK_VISUALIZATION" -eq 1 ]]; then EXTRA_ARGS+=(--include-network-visualization); fi
if [[ "$NO_RESUME_STUDIES" -eq 1 ]]; then EXTRA_ARGS+=(--no-resume-studies); fi
if [[ "$NO_REUSE_MI_SCORES" -eq 1 ]]; then EXTRA_ARGS+=(--no-reuse-mi-scores); fi
if [[ "$NO_SAVE_MI_CACHE" -eq 1 ]]; then EXTRA_ARGS+=(--no-save-mi-cache); fi
if [[ "$VERBOSE" -eq 1 ]]; then EXTRA_ARGS+=(--verbose); fi
if [[ -n "$MIN_STUDIES_FRACTION" ]]; then EXTRA_ARGS+=(--min-studies-fraction "$MIN_STUDIES_FRACTION"); fi

echo "[INFO] ═══════════════════════════════════════════════════════════════"
echo "[INFO] NETS_MI_PVAL_v2: Classical Histogram MI Full Feature Run"
echo "[INFO] ═══════════════════════════════════════════════════════════════"
echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] N_JOBS=$N_JOBS (CPU cores)"
echo "[INFO] N_BINS=$N_BINS MI_STRATEGY=$MI_STRATEGY"
echo "[INFO] PERMS=$PERMS MODE=$MODE PVAL=$PVAL"
echo "[INFO] PRESCREEN_METHOD=$PRESCREEN_METHOD THRESHOLD=$PRESCREEN_THRESHOLD"
echo "[INFO]"

# Run the pipeline
python run_pipeline.py \
  --counts "$COUNTS_FILE" \
  --meta "$META_FILE" \
  --output "$OUTPUT_DIR" \
  --prescreen-method "$PRESCREEN_METHOD" \
  --prescreen-threshold "$PRESCREEN_THRESHOLD" \
  --max-pairs "$MAX_PAIRS" \
  --mad-top-genes "$MAD_TOP_GENES" \
  --min-studies "$MIN_STUDIES" \
  --perms "$PERMS" \
  --mode "$MODE" \
  --pval "$PVAL" \
  --module-method "$MODULE_METHOD" \
  --module-leiden-resolution "$MODULE_LEIDEN_RESOLUTION" \
  --module-min-size "$MODULE_MIN_SIZE" \
  --master-edge-weight "$MASTER_EDGE_WEIGHT" \
  --submodule-method "$SUBMODULE_METHOD" \
  --submodule-size-threshold "$SUBMODULE_SIZE_THRESHOLD" \
  --submodule-min-size "$SUBMODULE_MIN_SIZE" \
  --submodule-mcode-score-threshold "$SUBMODULE_MCODE_SCORE_THRESHOLD" \
  --submodule-mcode-min-density "$SUBMODULE_MCODE_MIN_DENSITY" \
  --ortholog-map "$ORTHOLOG_MAP" \
  --ortholog-source-col "$ORTHOLOG_SOURCE_COL" \
  --ortholog-target-col "$ORTHOLOG_TARGET_COL" \
  --module-export-map "$MODULE_EXPORT_MAP" \
  --module-export-key-col "$MODULE_EXPORT_KEY_COL" \
  --module-export-cols ${MODULE_EXPORT_COLS} \
  "${EXTRA_ARGS[@]}"

echo "[INFO]"
echo "[INFO] ✓ Pipeline complete. Results in: $OUTPUT_DIR"
echo "[INFO]"

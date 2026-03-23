#!/bin/bash
set -euo pipefail

# Reproducible full-feature run wrapper for this project.
#
# Examples:
#   ./run_whole_dataset_in_gpu_2.sh
#   MAX_PAIRS=50000000 PERMS=10000 ./run_whole_dataset_in_gpu_2.sh
#   STUDY_GPU_WORKERS=1 STUDY_GPU_DEVICES="cuda:0" ./run_whole_dataset_in_gpu_2.sh
#   NO_RESUME_STUDIES=1 ./run_whole_dataset_in_gpu_2.sh

PROJECT_DIR="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED"
COUNTS_FILE="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/logCPM_matrix_filtered_samples.csv"
META_FILE="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/metadata_with_sample_annotations.csv"


PRESCREEN_THRESHOLD="${PRESCREEN_THRESHOLD:-0.3}"
PRESCREEN_METHOD="${PRESCREEN_METHOD:-spearman}"
MAX_PAIRS="${MAX_PAIRS:-300000000}"
MAD_TOP_GENES="${MAD_TOP_GENES:-32763}"
MIN_STUDIES="${MIN_STUDIES:-2}"
MIN_STUDIES_FRACTION="${MIN_STUDIES_FRACTION:-}"


OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/wholde_dataset_in_gpu_2_${MAX_PAIRS}}"
DEVICE="${DEVICE:-cuda}"
STUDY_GPU_WORKERS="${STUDY_GPU_WORKERS:-2}"
STUDY_GPU_DEVICES="${STUDY_GPU_DEVICES:-cuda:0 cuda:1}"

PERMS="${PERMS:-30000}"
MODE="${MODE:-global}"
PVAL="${PVAL:-0.001}"
HIDDEN="${HIDDEN:-64}"
EPOCHS="${EPOCHS:-100}"
BATCH_PAIRS="${BATCH_PAIRS:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-0}"


MODULE_METHOD="${MODULE_METHOD:-leiden}"
MODULE_LEIDEN_RESOLUTION="${MODULE_LEIDEN_RESOLUTION:-1.2}"
MODULE_MIN_SIZE="${MODULE_MIN_SIZE:-10}"
MASTER_EDGE_WEIGHT="${MASTER_EDGE_WEIGHT:-mean_neglog10p}"
NORMALIZE_WEIGHTS="${NORMALIZE_WEIGHTS:-1}"

SUBMODULE_METHOD="${SUBMODULE_METHOD:-mcode}"
SUBMODULE_SIZE_THRESHOLD="${SUBMODULE_SIZE_THRESHOLD:-200}"
SUBMODULE_MIN_SIZE="${SUBMODULE_MIN_SIZE:-10}"
SUBMODULE_MCODE_SCORE_THRESHOLD="${SUBMODULE_MCODE_SCORE_THRESHOLD:-0.2}"
SUBMODULE_MCODE_MIN_DENSITY="${SUBMODULE_MCODE_MIN_DENSITY:-0.3}"

ORTHOLOG_MAP="${ORTHOLOG_MAP:-${PROJECT_DIR}/UBELIX/MINE/gene_id_mapping.tsv}"
ORTHOLOG_SOURCE_COL="${ORTHOLOG_SOURCE_COL:-ensembl_gene_id}"
ORTHOLOG_TARGET_COL="${ORTHOLOG_TARGET_COL:-external_gene_name}"
MODULE_EXPORT_MAP="${MODULE_EXPORT_MAP:-${ORTHOLOG_MAP}}"
MODULE_EXPORT_KEY_COL="${MODULE_EXPORT_KEY_COL:-ensembl_gene_id}"
MODULE_EXPORT_COLS="${MODULE_EXPORT_COLS:-external_gene_name entrezgene_id}"

DOWNLOAD_GMT="${DOWNLOAD_GMT:-1}"
SAVE_PER_GMT_ENRICHMENTS="${SAVE_PER_GMT_ENRICHMENTS:-1}"
INCLUDE_NETWORK_VISUALIZATION="${INCLUDE_NETWORK_VISUALIZATION:-1}"

NO_RESUME_STUDIES="${NO_RESUME_STUDIES:-0}"
NO_REUSE_MINE_SCORES="${NO_REUSE_MINE_SCORES:-0}"
NO_SAVE_MINE_CACHE="${NO_SAVE_MINE_CACHE:-0}"

cd "$PROJECT_DIR"
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate
module load PyTorch

export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages
export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages
export PYTHONPATH=${VENV_SITE}:${TORCH_SITE}
export NUMEXPR_DISABLED=1

GPU_WORKER_ARGS=(--study-gpu-workers "$STUDY_GPU_WORKERS")
if [[ -n "$STUDY_GPU_DEVICES" ]]; then
  read -r -a _GPU_DEVICES_ARR <<< "$STUDY_GPU_DEVICES"
  if [[ "${#_GPU_DEVICES_ARR[@]}" -gt 0 ]]; then
    GPU_WORKER_ARGS+=(--study-gpu-devices "${_GPU_DEVICES_ARR[@]}")
  fi
fi

EXTRA_ARGS=()
if [[ "$MIXED_PRECISION" -eq 1 ]]; then EXTRA_ARGS+=(--mixed-precision); fi
if [[ "$NORMALIZE_WEIGHTS" -eq 1 ]]; then EXTRA_ARGS+=(--normalize-weights); fi
if [[ "$DOWNLOAD_GMT" -eq 1 ]]; then EXTRA_ARGS+=(--download-gmt); fi
if [[ "$SAVE_PER_GMT_ENRICHMENTS" -eq 1 ]]; then EXTRA_ARGS+=(--save-per-gmt-enrichments); fi
if [[ "$INCLUDE_NETWORK_VISUALIZATION" -eq 1 ]]; then EXTRA_ARGS+=(--include-network-visualization); fi
if [[ "$NO_RESUME_STUDIES" -eq 1 ]]; then EXTRA_ARGS+=(--no-resume-studies); fi
if [[ "$NO_REUSE_MINE_SCORES" -eq 1 ]]; then EXTRA_ARGS+=(--no-reuse-mine-scores); fi
if [[ "$NO_SAVE_MINE_CACHE" -eq 1 ]]; then EXTRA_ARGS+=(--no-save-mine-cache); fi
if [[ -n "$MIN_STUDIES_FRACTION" ]]; then EXTRA_ARGS+=(--min-studies-fraction "$MIN_STUDIES_FRACTION"); fi

echo "[INFO] Running full-feature whole_dataset_in_gpu_2"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] DEVICE=$DEVICE"
echo "[INFO] STUDY_GPU_WORKERS=$STUDY_GPU_WORKERS"
echo "[INFO] STUDY_GPU_DEVICES=$STUDY_GPU_DEVICES"
echo "[INFO] PERMS=$PERMS MODE=$MODE PVAL=$PVAL"
echo "[INFO] BATCH_PAIRS=$BATCH_PAIRS MIXED_PRECISION=$MIXED_PRECISION"
echo "[INFO] MAX_PAIRS=$MAX_PAIRS PRESCREEN_METHOD=$PRESCREEN_METHOD PRESCREEN_THRESHOLD=$PRESCREEN_THRESHOLD"
echo "[INFO] Resume toggles: NO_RESUME_STUDIES=$NO_RESUME_STUDIES NO_REUSE_MINE_SCORES=$NO_REUSE_MINE_SCORES NO_SAVE_MINE_CACHE=$NO_SAVE_MINE_CACHE"

python run_pipeline.py \
  --counts "$COUNTS_FILE" \
  --meta "$META_FILE" \
  --output "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "${GPU_WORKER_ARGS[@]}" \
  --hidden "$HIDDEN" \
  --epochs "$EPOCHS" \
  --batch-pairs "$BATCH_PAIRS" \
  --perms "$PERMS" \
  --mode "$MODE" \
  --pval "$PVAL" \
  --prescreen-threshold "$PRESCREEN_THRESHOLD" \
  --max-pairs "$MAX_PAIRS" \
  --prescreen-method "$PRESCREEN_METHOD" \
  --mad-top-genes "$MAD_TOP_GENES" \
  --qc-preplot \
  --qc-postplot \
  --min-studies "$MIN_STUDIES" \
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
  --module-export-cols $MODULE_EXPORT_COLS \
  "${EXTRA_ARGS[@]}"

echo "[INFO] Run complete"

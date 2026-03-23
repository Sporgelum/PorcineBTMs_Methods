#!/bin/bash
set -euo pipefail

# Quick full-feature run:
# - Keeps the same argument family as the full historical run.
# - Only MAX_PAIRS is intended to change for faster smoke/quick tests.
#
# Example:
#   MAX_PAIRS=1000000 ./run_quick_full_feature_test.sh
#   MAX_PAIRS=50000000 BATCH_PAIRS=auto USE_MIXED_PRECISION=1 ./run_quick_full_feature_test.sh
#   MAX_PAIRS=300000000 STUDY_GPU_WORKERS=1 STUDY_GPU_DEVICES="cuda:0" ./run_quick_full_feature_test.sh

PROJECT_DIR="/storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/workingEnvironment/03_network/MINE"
COUNTS_FILE="/storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/workingEnvironment/02_counts/logCPM_matrix_filtered_samples.csv"
META_FILE="/storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/workingEnvironment/02_counts/metadata_with_sample_annotations.csv"

MAX_PAIRS="${MAX_PAIRS:-300000000}"
BATCH_PAIRS="${BATCH_PAIRS:-auto}"
USE_MIXED_PRECISION="${USE_MIXED_PRECISION:-1}"
DEVICE="${DEVICE:-cuda}"
STUDY_GPU_WORKERS="${STUDY_GPU_WORKERS:-1}"
STUDY_GPU_DEVICES="${STUDY_GPU_DEVICES:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/quick_full_features_maxpairs_${MAX_PAIRS}}"

cd "$PROJECT_DIR"
source /storage/homefs/mb23h197/Environments/radian_env_2025/bin/activate

MP_ARGS=()
if [[ "$USE_MIXED_PRECISION" -eq 1 ]]; then
    MP_ARGS+=("--mixed-precision")
fi

GPU_WORKER_ARGS=("--study-gpu-workers" "$STUDY_GPU_WORKERS")
if [[ -n "$STUDY_GPU_DEVICES" ]]; then
  read -r -a _GPU_DEVICES_ARR <<< "$STUDY_GPU_DEVICES"
  if [[ "${#_GPU_DEVICES_ARR[@]}" -gt 0 ]]; then
    GPU_WORKER_ARGS+=("--study-gpu-devices" "${_GPU_DEVICES_ARR[@]}")
  fi
fi

echo "[INFO] Running quick full-feature test"
echo "[INFO] MAX_PAIRS=$MAX_PAIRS"
echo "[INFO] BATCH_PAIRS=$BATCH_PAIRS"
echo "[INFO] USE_MIXED_PRECISION=$USE_MIXED_PRECISION"
echo "[INFO] DEVICE=$DEVICE"
echo "[INFO] STUDY_GPU_WORKERS=$STUDY_GPU_WORKERS"
echo "[INFO] STUDY_GPU_DEVICES=$STUDY_GPU_DEVICES"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"

python run_pipeline.py \
  --counts "$COUNTS_FILE" \
  --meta "$META_FILE" \
  --output "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "${GPU_WORKER_ARGS[@]}" \
  --perms 30000 \
  --mode global \
  --pval 0.05 \
  --epochs 150 \
  --batch-pairs "$BATCH_PAIRS" \
  "${MP_ARGS[@]}" \
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
  --ortholog-map gene_id_mapping.tsv \
  --ortholog-source-col ensembl_gene_id \
  --ortholog-target-col external_gene_name \
  --download-gmt \
  --module-export-map gene_id_mapping.tsv \
  --module-export-key-col ensembl_gene_id \
  --module-export-cols external_gene_name entrezgene_id \
  --save-per-gmt-enrichments \
  --include-network-visualization

echo "[INFO] Quick full-feature run complete"

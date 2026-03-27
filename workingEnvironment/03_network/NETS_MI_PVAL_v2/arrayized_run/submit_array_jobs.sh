#!/bin/bash
###############################################################################
# SLURM Array Job Submission - MINE per Study
#
# Usage:
#   bash submit_array_jobs.sh [max_pairs] [perms] [output_tag]
#   bash submit_array_jobs.sh 300000000 30000 "my_run"
#   bash submit_array_jobs.sh  # Uses defaults
#
# Environment variables (optional):
#   MAX_PAIRS, PERMS, PARTITION, SLURM_ARRAY_LIMIT, etc.
###############################################################################

set -euo pipefail

# Paths
PROJECT_DIR="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+"
VENV_DIR="${PROJECT_DIR}"
WORK_DIR="${PROJECT_DIR}/workingEnvironment/03_network/NETS_MI_PVAL_v2"
ARRAYIZED_DIR="${WORK_DIR}/arrayized_run"
COUNTS_FILE="${PROJECT_DIR}/workingEnvironment/02_counts/logCPM_matrix_filtered_samples.csv"
META_FILE="${PROJECT_DIR}/workingEnvironment/02_counts/metadata_with_sample_annotations.csv"

# Parameters from arguments or environment
MAX_PAIRS="${1:-${MAX_PAIRS:-3000}}"
PERMS="${2:-${PERMS:-30000}}"
OUTPUT_TAG="${3:-${OUTPUT_TAG:-array_run_$(date +%s)}}"

# Pre-screening (correlation-based candidate filtering)
PRESCREEN_THRESHOLD="${PRESCREEN_THRESHOLD:-0.3}"
PRESCREEN_METHOD="${PRESCREEN_METHOD:-spearman}"
MAD_TOP_GENES="${MAD_TOP_GENES:-32763}"
MIN_STUDIES_FRACTION="${MIN_STUDIES_FRACTION:-}"

# MINE / runtime parameters
HIDDEN_DIM="${HIDDEN_DIM:-64}"
EPOCHS="${EPOCHS:-200}"
BATCH_PAIRS="${BATCH_PAIRS:-512}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
MIXED_PRECISION="${MIXED_PRECISION:-0}"
N_JOBS="${N_JOBS:--1}"

# Output and resume behavior
NO_RESUME_STUDIES="${NO_RESUME_STUDIES:-0}"
NO_REUSE_MI_SCORES="${NO_REUSE_MI_SCORES:-0}"
NO_SAVE_MI_CACHE="${NO_SAVE_MI_CACHE:-0}"
VERBOSE="${VERBOSE:-0}"

# Permutation testing
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
ORTHOLOG_MAP="${ORTHOLOG_MAP:-${PROJECT_DIR}/workingEnvironment/03_network/gene_id_mapping.tsv}"
ORTHOLOG_SOURCE_COL="${ORTHOLOG_SOURCE_COL:-ensembl_gene_id}"
ORTHOLOG_TARGET_COL="${ORTHOLOG_TARGET_COL:-external_gene_name}"
MODULE_EXPORT_MAP="${MODULE_EXPORT_MAP:-${ORTHOLOG_MAP}}"
MODULE_EXPORT_KEY_COL="${MODULE_EXPORT_KEY_COL:-ensembl_gene_id}"
MODULE_EXPORT_COLS="${MODULE_EXPORT_COLS:-external_gene_name entrezgene_id}"

# Gene-set enrichment
DOWNLOAD_GMT="${DOWNLOAD_GMT:-1}"
SAVE_PER_GMT_ENRICHMENTS="${SAVE_PER_GMT_ENRICHMENTS:-1}"
INCLUDE_NETWORK_VISUALIZATION="${INCLUDE_NETWORK_VISUALIZATION:-1}"

# Network baseline minimums
MIN_STUDIES="${MIN_STUDIES:-2}"

# SLURM settings
PARTITION="${PARTITION:-pibu_el8}"
SLURM_ARRAY_LIMIT="${SLURM_ARRAY_LIMIT:-50}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
MEM_PER_TASK="${MEM_PER_TASK:-64G}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"

# Paths for this run
OUTPUT_DIR="${ARRAYIZED_DIR}/output/${OUTPUT_TAG}"
STUDIES_DIR="${OUTPUT_DIR}/studies"
JOBS_DIR="${ARRAYIZED_DIR}/job_${OUTPUT_TAG}"

mkdir -p "${OUTPUT_DIR}" "${STUDIES_DIR}" "${JOBS_DIR}"

cat << EOF

╔════════════════════════════════════════════════════════════════════════════╗
║             SLURM Array Submission - MINE per Study (BioProject)            ║
╚════════════════════════════════════════════════════════════════════════════╝

Configuration:
  Output directory : ${OUTPUT_DIR}
  Studies directory: ${STUDIES_DIR}
  Counts file      : ${COUNTS_FILE}
  Metadata file    : ${META_FILE}

Pre-screening:
  PRESCREEN METHOD : ${PRESCREEN_METHOD} (threshold=${PRESCREEN_THRESHOLD})
  MAX_PAIRS        : ${MAX_PAIRS}
  MAD_TOP_GENES    : ${MAD_TOP_GENES}

MINE:
  HIDDEN_DIM       : ${HIDDEN_DIM}
  EPOCHS           : ${EPOCHS}
  BATCH_PAIRS      : ${BATCH_PAIRS}
  LEARNING_RATE    : ${LEARNING_RATE}
  MIXED_PRECISION  : ${MIXED_PRECISION}
  N_JOBS           : ${N_JOBS}

Permutation Testing:
  PERMS            : ${PERMS}
  MODE             : ${MODE}
  PVAL             : ${PVAL}

Network & Modules:
  MIN_STUDIES      : ${MIN_STUDIES}
  MIN_STUDIES_FRAC : ${MIN_STUDIES_FRACTION:-none}
  MODULE_METHOD    : ${MODULE_METHOD}
  MODULE_LEIDEN_RES: ${MODULE_LEIDEN_RESOLUTION}
  MODULE_MIN_SIZE  : ${MODULE_MIN_SIZE}
  MASTER_EDGE_WGT  : ${MASTER_EDGE_WEIGHT}
  NORMALIZE_WEIGHTS: ${NORMALIZE_WEIGHTS}

Submodule Refinement:
  SUBMODULE_METHOD : ${SUBMODULE_METHOD}
  SUBMODULE_SIZE   : ${SUBMODULE_SIZE_THRESHOLD}
  SUBMODULE_MIN    : ${SUBMODULE_MIN_SIZE}
  MCODE_SCORE_THR  : ${SUBMODULE_MCODE_SCORE_THRESHOLD}
  MCODE_MIN_DENS   : ${SUBMODULE_MCODE_MIN_DENSITY}

Annotation & Export:
  ORTHOLOG_MAP     : ${ORTHOLOG_MAP##*/}
  DOWNLOAD_GMT     : ${DOWNLOAD_GMT}
  SAVE_PER_GMT     : ${SAVE_PER_GMT_ENRICHMENTS}
  NETWORK_VIZ      : ${INCLUDE_NETWORK_VISUALIZATION}

Resume/Cache:
  NO_RESUME        : ${NO_RESUME_STUDIES}
  NO_REUSE_SCORES  : ${NO_REUSE_MI_SCORES}
  NO_SAVE_CACHE    : ${NO_SAVE_MI_CACHE}
  VERBOSE          : ${VERBOSE}

SLURM Resources:
  Partition        : ${PARTITION}
  Cpus/task        : ${CPUS_PER_TASK}
  Memory/task      : ${MEM_PER_TASK}
  Time limit       : ${TIME_LIMIT}
  Array limit      : ${SLURM_ARRAY_LIMIT}

EOF

# Step 1: Create config file
echo "[1] Creating config file..."
CONFIG_FILE="${JOBS_DIR}/config.json"
python3 << PYCODE
import json, sys
sys.path.insert(0, '.') 
from mine_network.config import PipelineConfig

def as_bool(v):
  return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

cfg = PipelineConfig()
cfg.counts_path = "${COUNTS_FILE}"
cfg.metadata_path = "${META_FILE}"
cfg.n_jobs = int(${N_JOBS})

cfg.mine.hidden_dim = int(${HIDDEN_DIM})
cfg.mine.n_epochs = int(${EPOCHS})
_batch_pairs_raw = "${BATCH_PAIRS}".strip()
cfg.mine.batch_pairs = (
  _batch_pairs_raw if _batch_pairs_raw.lower() == "auto"
  else int(_batch_pairs_raw)
)
cfg.mine.learning_rate = float(${LEARNING_RATE})
cfg.mine.mixed_precision = as_bool("${MIXED_PRECISION}")

cfg.prescreen.threshold = ${PRESCREEN_THRESHOLD}
cfg.prescreen.method = "${PRESCREEN_METHOD}"
cfg.prescreen.max_pairs = ${MAX_PAIRS}

cfg.permutation.n_permutations = ${PERMS}
cfg.permutation.mode = "${MODE}"
cfg.permutation.p_value_threshold = ${PVAL}

cfg.network.min_study_count = ${MIN_STUDIES}
cfg.network.min_samples_per_study = 3
if "${MIN_STUDIES_FRACTION}".strip():
  cfg.network.min_study_fraction = float("${MIN_STUDIES_FRACTION}")

cfg.module.method = "${MODULE_METHOD}"
cfg.module.module_leiden_resolution = float(${MODULE_LEIDEN_RESOLUTION})
cfg.module.module_min_size = int(${MODULE_MIN_SIZE})
cfg.module.master_edge_weight = "${MASTER_EDGE_WEIGHT}"
cfg.module.normalize_weights = as_bool("${NORMALIZE_WEIGHTS}")
cfg.module.submodule_method = "${SUBMODULE_METHOD}"
cfg.module.submodule_size_threshold = int(${SUBMODULE_SIZE_THRESHOLD}) if "${SUBMODULE_SIZE_THRESHOLD}".strip() else None
cfg.module.submodule_min_size = int(${SUBMODULE_MIN_SIZE})
cfg.module.submodule_mcode_score_threshold = float(${SUBMODULE_MCODE_SCORE_THRESHOLD})
cfg.module.submodule_mcode_min_density = float(${SUBMODULE_MCODE_MIN_DENSITY})

cfg.annotation.download_enrichr = as_bool("${DOWNLOAD_GMT}")
cfg.annotation.ortholog_map_path = "${ORTHOLOG_MAP}" if "${ORTHOLOG_MAP}".strip() else None
cfg.annotation.ortholog_source_col = "${ORTHOLOG_SOURCE_COL}"
cfg.annotation.ortholog_target_col = "${ORTHOLOG_TARGET_COL}"
cfg.annotation.module_export_map_path = "${MODULE_EXPORT_MAP}" if "${MODULE_EXPORT_MAP}".strip() else None
cfg.annotation.module_export_key_col = "${MODULE_EXPORT_KEY_COL}"
cfg.annotation.module_export_cols = "${MODULE_EXPORT_COLS}".split() if "${MODULE_EXPORT_COLS}".strip() else []
cfg.annotation.save_per_gmt_results = as_bool("${SAVE_PER_GMT_ENRICHMENTS}")

cfg.visualization.enabled = as_bool("${INCLUDE_NETWORK_VISUALIZATION}")

cfg.resume_completed_studies = not as_bool("${NO_RESUME_STUDIES}")
cfg.reuse_mine_scores = not as_bool("${NO_REUSE_MI_SCORES}")
cfg.save_mine_score_cache = not as_bool("${NO_SAVE_MI_CACHE}")

# Export as JSON
import dataclasses
def asdict_recursive(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: asdict_recursive(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    elif isinstance(obj, list):
        return [asdict_recursive(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: asdict_recursive(v) for k, v in obj.items()}
    else:
        return obj

with open("${CONFIG_FILE}", 'w') as f:
    json.dump(asdict_recursive(cfg), f, indent=2)
print(f"Config saved: ${CONFIG_FILE}")
PYCODE

cd "${WORK_DIR}"

# Step 2: Discover number of studies
echo "[2] Discovering studies..."
STUDIES_OUTPUT=$(python3 << PYCODE
import sys
sys.path.insert(0, '.')
from mine_network.data_loader import load_expression, load_metadata, discover_studies
expr = load_expression("${COUNTS_FILE}")
meta = load_metadata("${META_FILE}")
studies = discover_studies(expr, meta, min_samples=3)
print(len(studies))
for i, study_name in enumerate(studies):
  print(study_name if isinstance(study_name, str) else study_name.get('name', f'study_{i}'))
PYCODE
)

# Extract just the count (should be a number on its own line)
N_STUDIES=$(echo "$STUDIES_OUTPUT" | grep -E '^[0-9]+$' | head -1)
# Extract study names (should start with PRJNA followed by digits)
STUDY_NAMES=$(echo "$STUDIES_OUTPUT" | grep '^PRJNA')

if [ -z "$N_STUDIES" ]; then
    echo "[ERROR] Could not discover studies"
    exit 1
fi

echo "  Found $N_STUDIES studies:"
echo "$STUDY_NAMES" | while read study; do
    echo "    ✓ $study"
done

# Step 3: Create SLURM script
echo "[3] Creating SLURM script..."
SLURM_SCRIPT="${JOBS_DIR}/array_job.slurm"

cat > "$SLURM_SCRIPT" << 'SLURMSCRIPT'
#!/bin/bash
#SBATCH --job-name=MI_arrays
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --array=0-NSTUDIES_PLACEHOLDER%LIMIT_PLACEHOLDER
#SBATCH --output=JOBS_DIR_PLACEHOLDER/slurm_logs/job_%A_%a.log
#SBATCH --error=JOBS_DIR_PLACEHOLDER/slurm_logs/job_%A_%a.err

# Activate environment
source VENV_DIR_PLACEHOLDER/bin/activate

# Create per-study output directory  
STUDY_OUTPUT_DIR="STUDIES_DIR_PLACEHOLDER/study_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$STUDY_OUTPUT_DIR"

# Run single study processor
cd "WORK_DIR_PLACEHOLDER"
python3 arrayized_run/process_study.py \
    --study-index "$SLURM_ARRAY_TASK_ID" \
    --config "JOBS_DIR_PLACEHOLDER/config.json" \
    --output-dir "$STUDY_OUTPUT_DIR"

exit $?
SLURMSCRIPT

# Substitute placeholders
sed -i "s|PARTITION_PLACEHOLDER|${PARTITION}|g" "$SLURM_SCRIPT"
sed -i "s|CPUS_PLACEHOLDER|${CPUS_PER_TASK}|g" "$SLURM_SCRIPT"
sed -i "s|MEM_PLACEHOLDER|${MEM_PER_TASK}|g" "$SLURM_SCRIPT"
sed -i "s|TIME_PLACEHOLDER|${TIME_LIMIT}|g" "$SLURM_SCRIPT"
sed -i "s|NSTUDIES_PLACEHOLDER|$((N_STUDIES - 1))|g" "$SLURM_SCRIPT"
sed -i "s|LIMIT_PLACEHOLDER|${SLURM_ARRAY_LIMIT}|g" "$SLURM_SCRIPT"
sed -i "s|JOBS_DIR_PLACEHOLDER|${JOBS_DIR}|g" "$SLURM_SCRIPT"
sed -i "s|STUDIES_DIR_PLACEHOLDER|${STUDIES_DIR}|g" "$SLURM_SCRIPT"
sed -i "s|VENV_DIR_PLACEHOLDER|${VENV_DIR}|g" "$SLURM_SCRIPT"
sed -i "s|WORK_DIR_PLACEHOLDER|${WORK_DIR}|g" "$SLURM_SCRIPT"

chmod +x "$SLURM_SCRIPT"
mkdir -p "${JOBS_DIR}/slurm_logs"
echo "  Created: $SLURM_SCRIPT"

# Step 4: Submit job
echo "[4] Submitting array job..."
ARRAY_JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $NF}')

echo ""
echo "✓ Array job submitted: $ARRAY_JOB_ID"
echo ""

# Save job info
cat > "${OUTPUT_DIR}/job_info.txt" << EOF_INFO
Array Job ID       : $ARRAY_JOB_ID
Number of studies  : $N_STUDIES
Output directory   : $OUTPUT_DIR
Studies directory  : $STUDIES_DIR
Submission time    : $(date)
Config file        : $CONFIG_FILE

Studies to process:
EOF_INFO

echo "$STUDY_NAMES" >> "${OUTPUT_DIR}/job_info.txt"

cat >> "${OUTPUT_DIR}/job_info.txt" << EOF_INFO

Next steps:
  1. Monitor:     watch squeue -j $ARRAY_JOB_ID
  2. Consolidate: python3 arrayized_run/process_study.py --consolidate \\
                    --studies-dir $STUDIES_DIR \\
                    --output-dir $OUTPUT_DIR
EOF_INFO

cat << EOF

╔════════════════════════════════════════════════════════════════════════════╗
║                              NEXT STEPS                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

1️⃣  Monitor job progress:
    watch squeue -j $ARRAY_JOB_ID

2️⃣  When all tasks complete, consolidate results:
    python3 arrayized_run/process_study.py --consolidate \\
        --studies-dir $STUDIES_DIR \\
        --output-dir $OUTPUT_DIR

3️⃣  Results will be in:
    $OUTPUT_DIR/master_network_*.tsv

════════════════════════════════════════════════════════════════════════════

Job information saved to: $OUTPUT_DIR/job_info.txt

EOF


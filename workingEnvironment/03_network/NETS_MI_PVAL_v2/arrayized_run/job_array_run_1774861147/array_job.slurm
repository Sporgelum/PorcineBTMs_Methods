#!/bin/bash
#SBATCH --job-name=MI_arrays
#SBATCH --partition=pibu_el8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=99
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --array=0-16%50
#SBATCH --output=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/NETS_MI_PVAL_v2/arrayized_run/job_array_run_1774861147/slurm_logs/job_%A_%a.log
#SBATCH --error=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/NETS_MI_PVAL_v2/arrayized_run/job_array_run_1774861147/slurm_logs/job_%A_%a.err

# Activate environment
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate

# Create per-study output directory  
STUDY_OUTPUT_DIR="/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/NETS_MI_PVAL_v2/arrayized_run/output/array_run_1774861147/studies/study_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$STUDY_OUTPUT_DIR"

# Run single study processor
cd "/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/NETS_MI_PVAL_v2"
python3 arrayized_run/process_study.py \
    --study-index "$SLURM_ARRAY_TASK_ID" \
    --config "/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/NETS_MI_PVAL_v2/arrayized_run/job_array_run_1774861147/config.json" \
    --output-dir "$STUDY_OUTPUT_DIR"

exit $?

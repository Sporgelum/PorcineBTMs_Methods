# SLURM Arrayized MI Pipeline - MINE per Study

**Parallelizes MINE-based MI computation across studies using SLURM array jobs.**

This system processes each BioProject (study) independently in parallel with the `mine_network` pipeline, then consolidates per-study adjacency matrices into a master consensus network.

---

## Quick Start

### 1. Submit array job (1 minute)
```bash
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/NETS_MI_PVAL_v2
bash arrayized_run/submit_array_jobs.sh 300000000 30000 "my_run"
```
**Prints Job ID → use in step 3**

### 2. Monitor (4 hours typical)
```bash
watch squeue -j <JOB_ID>  # Replace with actual ID
```

### 3. Consolidate results (30 minutes)
```bash
python3 arrayized_run/process_study.py --consolidate \
    --studies-dir arrayized_run/output/my_run/studies \
    --output-dir arrayized_run/output/my_run
```

**Results:** `arrayized_run/output/my_run/master_network_edgelist.tsv`

---

## What It Does

### MINE Study Workflow
- **Per-study**: Each BioProject runs through the same `mine_network` pipeline
- **MI estimation**: MINE-based scoring with configured pre-screening and permutation testing
- **Parallelization**: One SLURM task per study → studies run simultaneously on different CPUs
- **Consolidation**: Merges per-study networks into master consensus network

### Workflow

```
Input data (logCPM matrix + metadata)
    ↓
Discover studies (BioProject in metadata)
    ↓
[SLURM Array Job: Process each in parallel]
  Study 0 MI  →  Study 1 MI  →  Study N MI  (simultaneous)
    ↓
Consolidate: Merge all studies
    ↓
Master network results
```

---

## Performance

| Studies | Sequential | Array Jobs | Speedup |
|---------|-----------|-----------|---------|
| 3 | 12 hours | 4.5 hours | 2.6× |
| 5 | 20 hours | 4.5 hours | 4.4× |
| 10 | 40 hours | 4.5 hours | 8.8× |

Each study takes ~4 hours, but all run in parallel in array mode.

---

## Files (3 total)

### `README.md`
This documentation file.

### `submit_array_jobs.sh`
Main submission script. Handles everything: discovery, job submission, and monitoring.

**Usage:**
```bash
bash arrayized_run/submit_array_jobs.sh [max_pairs] [perms] [tag]
```

**Examples:**
```bash
bash arrayized_run/submit_array_jobs.sh                        # Defaults
bash arrayized_run/submit_array_jobs.sh 50000000 10000 "test"  # Custom params
```

**Environment overrides:**
```bash
export MAX_PAIRS=100000000 PERMS=50000 PARTITION=cpu
bash arrayized_run/submit_array_jobs.sh
```

**Output:** Creates `arrayized_run/output/<tag>/` with per-study results in `studies/` subfolder.

### `process_study.py`
Worker script (called by SLURM per study) AND consolidation script.

**Single-study mode** (SLURM task):
```bash
python3 process_study.py --study-index 0 --config config.json --output-dir /path/study_0
```

**Consolidation mode** (after jobs complete):
```bash
python3 process_study.py --consolidate \
    --studies-dir /path/studies \
    --output-dir /path/output
```

---

## Configuration

### MINE Parameters

```bash
export HIDDEN_DIM=64          # MINE hidden layer width
export EPOCHS=200             # Training epochs
export BATCH_PAIRS=512        # Pairs processed per batch (or auto)
export MAX_PAIRS=300000000   # Max candidate pairs per study
```

### Pre-screening

```bash
export PRESCREEN_THRESHOLD=0.3    # Spearman |r| cutoff
export PRESCREEN_METHOD=spearman
```

### Permutation Testing

```bash
export PERMS=30000   # Permutation count
export MODE=global   # or per_pair
export PVAL=0.001    # Significance threshold
```

### Network & Clustering

```bash
export MIN_STUDIES=2         # Min studies for master edge
export MODULE_METHOD=leiden  # leiden or mcode
```

### SLURM Resources

```bash
export PARTITION=cpu           # Queue name
export SLURM_ARRAY_LIMIT=50    # Max concurrent tasks
# Edit submit_array_jobs.sh for more: cpus-per-task, mem, time
```

**Example:**
```bash
export MAX_PAIRS=100000000 PERMS=10000
bash arrayized_run/submit_array_jobs.sh
```

---

## Output Structure

```
arrayized_run/output/my_run/
├── master_network_edgelist.tsv     ← Main result
├── master_network_adjacency.mtx
├── master_network_genes.txt
└── studies/
    ├── study_0/
    │   ├── edges_mine_0.tsv
    │   ├── adj_mine_0.mtx
    │   └── study_0.log
    └── study_1/
        └── [similar]
```

---

## Monitoring & Troubleshooting

### Check job status

```bash
squeue -j <JOB_ID>
watch squeue -j <JOB_ID>
```

### Tail logs

```bash
tail -f arrayized_run/output/*/studies/study_0/study_0.log
```

### Find errors

```bash
cat arrayized_run/output/*/studies/study_*/study_*.log | grep -i error
cat arrayized_run/output/*/job_*/slurm_logs/*.err | grep -i error
```

### Common issues

| Problem | Fix |
|---------|-----|
| `sbatch: command not found` | `module load slurm` |
| Job pending | Check queue: `sinfo` |
| Task fails | Review: `cat arrayized_run/output/*/studies/study_*/study_*.log` |
| Out of memory | Edit `submit_array_jobs.sh`, increase `--mem` |
| No consolidation results | Verify: `find studies/ -name "adj_mine.mtx" \| wc -l` |

---

## Advanced

### Rerun failed studies
```bash
FAILED=$(squeue -j <JOB_ID> -o "%a %T" | grep FAILED | awk '{print $1}')
sbatch --array=$FAILED arrayized_run/job_*/array_job.slurm
```

### Auto-consolidate on completion
```bash
export AUTO_CONSOLIDATE=1
bash arrayized_run/submit_array_jobs.sh
```

### Manual consolidation with options
```bash
python3 arrayized_run/process_study.py --consolidate \
    --studies-dir arrayized_run/output/*/studies \
    --output-dir arrayized_run/output/* \
    --min-studies 3
```

---

## Notes
- This arrayized runner uses the current `mine_network` implementation.
- Per-study artifacts are normalized to index-based names (`adj_mine_<idx>.mtx`, etc.) for deterministic consolidation.



# Run, and crashed after 48h running, some oom and some time limit...
bash arrayized_run/submit_array_jobs.sh

# Run minitests all seem fine, re run whole pipeline now with 128 GB RAM, 94 Threads per noode, and 72h
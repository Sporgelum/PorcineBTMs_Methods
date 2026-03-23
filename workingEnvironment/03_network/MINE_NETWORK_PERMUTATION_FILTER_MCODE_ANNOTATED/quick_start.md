# Quick Start

This project builds multi-study gene co-expression networks from logCPM counts and metadata using a MINE-based mutual information workflow. In each study, it computes candidate gene-pair dependence, applies permutation significance testing, keeps significant edges, then aggregates replicated edges across studies into a master network. The master network is clustered with Leiden and optionally refined with MCODE, then modules are biologically annotated.

## Current Workflow (Wrapper-Based)

You are currently running the full workflow with the wrapper script so studies can run across two GPUs with resume and caching behavior.

- Wrapper: run_whole_dataset_in_gpu_2.sh
- Typical dual-GPU setup: study workers on cuda:0 and cuda:1
- Current max pairs target: 300000000
- Output folder pattern: output/wholde_dataset_in_gpu_2_300000000

## Pipeline Diagram

```mermaid
flowchart TD
    A[Load logCPM matrix + metadata] --> B[Discover BioProject studies]
    B --> C[Per-study processing]

    C --> D[Z-score expression]
    D --> E[Pre-screen candidate pairs by correlation]
    E --> F[MINE MI estimation on candidate pairs]
    F --> G[Permutation null distribution]
    G --> H[Compute empirical p-values]
    H --> I[Keep significant edges]
    I --> J[Save per-study outputs]

    J --> K[Count replicated gene pairs across studies]
    K --> L[Build master network with min-studies filter]
    L --> M[Leiden clustering]
    M --> N[Optional MCODE refinement]
    N --> O[Module annotation]
    O --> P[Final reports, tables, and network files]

    subgraph Runtime controls used now
      R1[run_whole_dataset_in_gpu_2.sh]
      R2[study_gpu_workers=2]
      R3[study_gpu_devices=cuda:0 cuda:1]
      R4[max_pairs=300000000]
      R5[pval=0.001, perms=30000]
      R6[resume + cache enabled by default]
    end

    R1 -. controls .-> C
    R2 -. controls .-> C
    R3 -. controls .-> C
    R4 -. controls .-> E
    R5 -. controls .-> G
    R6 -. controls .-> J
```

## Run With The Wrapper (Current Recommended Call)

    cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED
    MAX_PAIRS=300000000 STUDY_GPU_WORKERS=2 STUDY_GPU_DEVICES="cuda:0 cuda:1" BATCH_PAIRS=auto PVAL=0.001 PERMS=30000 NO_RESUME_STUDIES=0 NO_REUSE_MINE_SCORES=0 ./run_whole_dataset_in_gpu_2.sh

## Continue An Existing Run (Important)

Use the same OUTPUT_DIR and keep resume enabled:

    OUTPUT_DIR=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED/output/wholde_dataset_in_gpu_2_300000000 MAX_PAIRS=300000000 STUDY_GPU_WORKERS=2 STUDY_GPU_DEVICES="cuda:0 cuda:1" NO_RESUME_STUDIES=0 NO_REUSE_MINE_SCORES=0 ./run_whole_dataset_in_gpu_2.sh

## Monitor

    nvidia-smi

    ls -t output/wholde_dataset_in_gpu_2_300000000/mine_network_*.log | head -1

    tail -f output/wholde_dataset_in_gpu_2_300000000/mine_network_YYYYMMDD_HHMMSS.log

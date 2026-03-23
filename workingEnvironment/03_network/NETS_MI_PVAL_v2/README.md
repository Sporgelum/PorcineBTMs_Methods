# MINE Network — Permutation Filter — MCODE — Annotated

## What This Project Is (And What You Are Running Now)

This repository runs a full multi-study network pipeline:

1. Load expression counts + metadata
2. Process each BioProject study independently
3. Estimate gene-pair dependence with MINE
4. Apply permutation-based significance filtering
5. Build a replicated master network across studies
6. Cluster with Leiden and refine with MCODE
7. Annotate modules biologically

Your current production workflow uses the wrapper script with study-level
parallelism on two GPUs, a max candidate cap of 300000000 per study,
and resume/cache enabled so reruns continue completed work.

See quick start and flow diagram in [quick_start.md](quick_start.md).

**MINE-based gene co-expression network inference with permutation
significance testing, multi-study consensus, MCODE module detection,
and biological annotation.**

---

## Table of Contents

1. [Quick Start](quick_start.md)
2. [Similarities & Differences vs. Prior Implementations](#similarities--differences)
3. [Pipeline Overview](#pipeline-overview)
4. [Package Structure](#package-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Configuration](#configuration)
8. [Output Files](#output-files)
9. [Improvements Over Initial Release](#improvements-over-initial-release)
10. [References](#references)

---

## Similarities & Differences

### Three implementations compared

| Aspect | `generate_net_python_pval.py` (histogram MI) | `Project_MINE_network/` (first MINE draft) | **This package** (final design) |
|--------|----------------------------------------------|---------------------------------------------|----------------------------------|
| **MI estimator** | Histogram 2D binning (KBinsDiscretizer, 5 quantile bins) | Neural MINE (batched DV bound) | Neural MINE (batched DV bound) |
| **Data format** | Discretised integers (0–4) | Continuous Z-scored floats | Continuous Z-scored floats |
| **Pre-screening** | None — full N×N matrix | Pearson \|r\| > 0.3 (always on) | **Configurable** — on/off, threshold tuneable |
| **Null distribution** | 30 000 shuffled random pairs, histogram MI | 10 000 shuffled random pairs, MINE | **10 000 permutations** (default), supports both **global** and **per-pair** modes |
| **Global null justification** | Quantile binning → uniform marginals | Z-scoring → N(0,1) marginals | Z-scoring → N(0,1) marginals |
| **p-value computation** | searchsorted on null | searchsorted on null | searchsorted (global) or **(1+k)/(1+B)** exact (per-pair) |
| **p-value threshold** | 0.001 | 0.001 | 0.001 (configurable) |
| **Master network** | ≥ 3 studies or ≥ 30% | ≥ 3 studies or ≥ 30% | ≥ 3 studies or ≥ 30% (configurable) |
| **Module detection** | MCODE | MCODE | MCODE |
| **Biological annotation** | None | None | **Hypergeometric enrichment against GMT gene-set files** |
| **Package structure** | Single 560-line script | Flat folder, 6 files | **Proper Python package** (pip-installable, `__init__.py`, `pyproject.toml`) |
| **CLI** | Hardcoded constants | Hardcoded constants | **`argparse` CLI** with all parameters exposed |
| **Permutation target** | 30 000 per study | 10 000 per study | 10 000 per study/pair (default; adjustable via `--perms`) |
| **Min samples** | 3 (configurable) | 3 (configurable) | 3 |
| **BH-FDR** | Optional | Optional | Optional |

### What is the SAME across all three

1. **Per-study design**: each BioProject is processed independently
2. **Study discovery**: auto-detect from `BioProject` column in metadata
3. **Gene-pair MI**: pairwise MI for all candidate gene pairs
4. **Permutation significance**: empirical p-value via null distribution
5. **Edge threshold**: p < 0.001
6. **Multi-study consensus**: edges must appear in ≥ k studies
7. **MCODE**: dense-subgraph module detection (Bader & Hogue 2003)
8. **Saving**: edge lists (TSV), adjacency (MTX), GraphML, module tables

### What is DIFFERENT in this package

| Feature | What changed | Why |
|---------|-------------|-----|
| **MI estimator** | Histogram → MINE | Operates on continuous data; captures nonlinear dependencies; no binning information loss |
| **Data transform** | KBinsDiscretizer → Z-scoring | No discretisation needed; Z-scoring standardises marginals for valid global null |
| **Pre-screening** | None → optional Pearson filter | 537M pairs infeasible for MINE; Pearson pre-filter reduces to ~500K candidates |
| **Permutation modes** | Only global | Global (fast) + per-pair (rigorous); user chooses |
| **Permutation count** | 30 000 | 10 000 default; configurable via `--perms` |
| **Module annotation** | Manual / external | Built-in hypergeometric enrichment against GMT gene-set files |
| **Architecture** | Single script | Modular package with `__init__.py`, `pyproject.toml`, CLI |
| **Configuration** | Magic constants | Dataclass hierarchy; JSON-serialisable; CLI overrides |

### Visual pipeline comparison

```
HISTOGRAM MI (generate_net_python_pval.py)
──────────────────────────────────────────
logCPM → KBinsDiscretizer (5 bins) → histogram MI (all N² pairs)
       → 30K shuffled null → searchsorted p-val → threshold 0.001
       → per-study adj → master (≥3 studies) → MCODE → done

MINE (this package)
───────────────────
logCPM → Z-score → [optional Pearson pre-screen]
       → batched MINE on candidates (GPU)
       → 10K permutation null (global or per-pair)
       → searchsorted or exact p-val → threshold 0.001
       → per-study adj → master (≥3 studies) → MCODE
       → hypergeometric enrichment (GMT) → done
```

---

## Pipeline Overview

The pipeline follows **Sections 1–8** of the user's conceptual design:

### Per study *s*:

1. **Input**: expression matrix X^(s) ∈ ℝ^{n_s × G}
   - n_s = samples (3–170), G = genes
2. **Z-score** each gene across samples (mean=0, std=1)
3. **(Optional)** Pre-screen: keep pairs with |Pearson r| > threshold
4. **MINE MI estimation**: for each candidate pair (g_i, g_j):
   - Feed (x_k, z_k) for k=1..n_s into T_θ
   - Optimise DV bound: I(X;Z) ≥ E_P[T] − log E_Q[e^T]
   - B=512 networks trained simultaneously via `torch.bmm`
5. **Permutation null**: build study-level null distribution
   - Global: random gene pairs + permute → MINE
   - Per-pair: permute z for each real pair → MINE
6. **Significance**: empirical p < 0.001 → adjacency matrix A^(s)

### Across studies:

7. **Master network**: c_ij = #{s: A_ij^(s) = 1}, keep c_ij ≥ 3
8. **MCODE** on master network → dense modules
9. **Annotation**: hypergeometric enrichment vs GMT gene sets

---

## Package Structure

```
MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED/
├── pyproject.toml              # Package metadata, dependencies
├── run_pipeline.py             # CLI entry point (argparse)
├── README.md                   # This file
└── mine_network/               # The Python package
    ├── __init__.py             # Package init, exports PipelineConfig + run_pipeline
    ├── config.py               # All configuration in typed dataclasses
    ├── data_loader.py          # Load expression, metadata, discover studies, Z-score
    ├── mine_estimator.py       # BatchedMINE network + estimate_mi_for_pairs
    ├── permutation.py          # Global & per-pair null builders + p-value computation
    ├── prescreen.py            # Pearson/Spearman pre-screening (optional)
    ├── network.py              # Edge filtering, master network, BH-FDR
    ├── mcode.py                # MCODE module detection (Bader & Hogue 2003)
    ├── annotation.py           # GMT loading + hypergeometric enrichment
    └── io_utils.py             # TeeLogger, Timer, all saving functions
```

Each module is documented with:
- Module-level docstring explaining responsibility and design rationale
- Cross-references to the user's conceptual pipeline sections
- Full parameter documentation
- Implementation notes (e.g. EMA bias correction, batched weights)

---

## Installation

### From the project folder (editable):

```bash
pip install -e .
```

### Or just run directly:

```bash
python run_pipeline.py
```

### Requirements:

- Python ≥ 3.9
- PyTorch ≥ 1.10 (CUDA recommended but not required)
- numpy, pandas, scipy, joblib, python-igraph, matplotlib

---

## Usage

### Basic (auto-detect data):

```bash
python run_pipeline.py
```

### Full control:

```bash
python run_pipeline.py \
    --counts  /path/to/logCPM_matrix.csv \
    --meta    /path/to/metadata.csv \
    --output  ./results \
    --device  cuda \
    --perms   1000 \
    --pval    0.001 \
    --mode    global \
    --gmt     hallmark.gmt reactome.gmt
```

### As a Python library:

```python
from mine_network import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    counts_path="logCPM_matrix.csv",
    metadata_path="metadata.csv",
    output_dir="./results",
)
cfg.permutation.n_permutations = 1000
cfg.permutation.mode = "global"
cfg.annotation.gmt_paths = ["hallmark.gmt"]

results = run_pipeline(cfg)
```

### CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--counts` | auto-detect | Expression matrix path |
| `--meta` | auto-detect | Metadata path |
| `--output` | `./output` | Output directory |
| `--device` | `auto` | `auto`, `cuda`, `cpu` |
| `--hidden` | 64 | MINE hidden-layer width |
| `--epochs` | 200 | MINE training epochs |
| `--batch-pairs` | 512 | Gene pairs per batch |
| `--no-prescreen` | (off) | Disable Pearson pre-screening |
| `--prescreen-threshold` | 0.3 | Pre-screen \|r\| cutoff |
| `--perms` | 10000 | Permutation count |
| `--pval` | 0.001 | P-value threshold |
| `--mode` | `global` | `global` or `per_pair` |
| `--min-studies` | 3 | Min studies for master edge |
| `--min-samples` | 3 | Min samples per study |
| `--module-method` | `mcode` | First-pass detector: `mcode` or `leiden` |
| `--module-min-size` | 3 | Minimum first-pass module size |
| `--module-mcode-score-threshold` | 0.2 | First-pass MCODE seed threshold |
| `--module-mcode-min-density` | 0.3 | First-pass MCODE density filter |
| `--module-leiden-resolution` | 1.0 | First-pass Leiden resolution |
| `--module-leiden-iterations` | -1 | First-pass Leiden iterations |
| `--submodule-method` | `none` | Refinement detector: `none`, `mcode`, `leiden` |
| `--submodule-size-threshold` | (none) | Refine parent modules larger than this size |
| `--submodule-min-size` | 3 | Minimum submodule size in refinement |
| `--submodule-mcode-score-threshold` | 0.2 | Refinement MCODE seed threshold |
| `--submodule-mcode-min-density` | 0.3 | Refinement MCODE density filter |
| `--submodule-leiden-resolution` | 1.0 | Refinement Leiden resolution |
| `--submodule-leiden-iterations` | -1 | Refinement Leiden iterations |
| `--gmt` | (none) | GMT files for annotation |
| `--download-gmt` | (off) | Auto-download GMT files from Enrichr API |
| `--enrichr-libs` | (all 5) | Enrichr library names to download |

### Clean module/submodule workflow

Use this pattern to avoid redundant flags:

1. Choose one first-pass method with `--module-method`.
2. Set only arguments relevant to that method.
3. Choose one refinement method with `--submodule-method` (or `none`).
4. Set `--submodule-size-threshold` only if refinement is enabled.

Examples:

- First pass Leiden + refinement Leiden:
   - `--module-method leiden --module-leiden-resolution 1.2 --module-min-size 3 --submodule-method leiden --submodule-size-threshold 100 --submodule-leiden-resolution 1.1 --submodule-min-size 10`
- First pass Leiden + refinement MCODE:
   - `--module-method leiden --module-leiden-resolution 1.2 --module-min-size 3 --submodule-method mcode --submodule-size-threshold 100 --submodule-mcode-score-threshold 0.2 --submodule-mcode-min-density 0.05 --submodule-min-size 10`
- First pass MCODE + no refinement:
   - `--module-method mcode --module-mcode-score-threshold 0.2 --module-mcode-min-density 0.05 --module-min-size 3 --submodule-method none`

---

## Configuration

All parameters are structured in typed dataclasses (see `mine_network/config.py`):

```
PipelineConfig
├── MINEConfig          hidden_dim, n_epochs, lr, ema_alpha, batch_pairs, ...
├── PrescreenConfig     enabled, method, threshold, max_pairs
├── PermutationConfig   n_permutations, seed, p_value_threshold, mode
├── NetworkConfig       min_study_count, min_study_fraction, min_samples_per_study
├── ModuleConfig        module/submodule method selection + method-specific settings
├── MCODEConfig         legacy compatibility defaults
└── AnnotationConfig    gmt_paths, fdr_threshold, min_overlap, background_genes, download_enrichr, enrichr_libraries
```

---

## Output Files

### Per study:

| File | Description |
|------|-------------|
| `edges_mine_{study}.tsv` | Significant edges: gene_A, gene_B, MI_MINE, p_value |
| `mine_diagnostics/mine_batch_summary_{study}.tsv` | Per-batch training diagnostics |
| `mine_diagnostics/mine_loss_curve_{study}.tsv` | Epoch-level loss curves |
| `mine_diagnostics/mine_raw_diagnostics_{study}.json` | Full raw training data |
| `adj_mine_{study}.mtx` | Sparse adjacency (Matrix Market) |
| `network_mine_{study}.graphml` | GraphML for Cytoscape/Gephi |
| `null_distribution_{study}.txt` | Null MI statistics (QC) |
| `edges_bh_fdr_{study}.tsv` | (Optional) BH-corrected edges |

### Master network:

| File | Description |
|------|-------------|
| `master_network_edgelist.tsv` | Consensus edges with study count |
| `master_network_adjacency.mtx` | Sparse binary adjacency |
| `master_edge_study_counts.mtx` | Study-count matrix |
| `master_network.graphml` | GraphML with module attribute |
| `master_BTM_modules.tsv` | Module membership table |
| `master_node_modules.tsv` | Gene → module mapping |
| `master_submodule_M{id}.graphml` | Per-module subgraph |
| `module_annotations.tsv` | Enrichment results (if GMT provided) |
| `module_annotation_summary.tsv` | Top 5 gene sets per module |
| `analysis_report_{ts}.txt` | Full summary + timing breakdown |

---

## Improvements Over Initial Release

The following enhancements were made after the first working version:

### 1. Vectorised Pearson pre-screen (performance)
- Original: Python for-loop over ~32K genes → hours to compute.
- Now: **chunked matrix multiply** (`block @ rest.T`, chunk size 1000). Centre + L2-normalise the full matrix once, then compute Pearson correlations via dot products.
- **537M pair pre-screen completes in ~3 seconds** (was effectively unusable before).
- Efficient capping via `np.argpartition` (O(n) top-K selection) when candidate pairs exceed `max_pairs`.

### 2. MINE training diagnostics
- `estimate_mi_batch` now returns per-batch diagnostics: loss curve (per epoch), final MI mean/std/max.
- Saved automatically under `output/mine_diagnostics/`:
  - `mine_batch_summary_{study}.tsv` — per-batch summary (initial/final loss, MI stats)
  - `mine_loss_curve_{study}.tsv` — epoch-level aggregated loss and MI
  - `mine_raw_diagnostics_{study}.json` — full raw data for downstream analysis

### 3. Automatic GMT download from Enrichr
- `--download-gmt` flag triggers automatic download of gene-set libraries from the Enrichr REST API.
- Default libraries: GO_Biological_Process_2023, KEGG_2021_Human, Reactome_2022, WikiPathway_2023_Human, MSigDB_Hallmark_2020.
- Downloaded GMTs are cached locally to avoid re-downloading.
- Custom libraries can be specified with `--enrichr-libs`.

### 4. Default permutations increased to 10 000
- Raised from 1 000 to 10 000 for better null resolution at the p < 0.001 threshold.

### 5. Max pairs cap raised to 5 000 000
- Raised from 500K to 5M to prevent the capping mechanism from pushing the effective |r| threshold too close to 1.0 (which would keep only trivially redundant gene pairs).

### 6. Bugfix: permutation.py tuple unpacking
- `estimate_mi_batch` was updated to return `(mi_result, diagnostics)`, but the two call sites in `permutation.py` (global null and per-pair null) were not updated. Fixed by unpacking with `mi_batch, _ = estimate_mi_batch(...)`.

---

## References

1. Belghazi, M.I. et al. (2018). *Mutual Information Neural Estimation.*
   Proceedings of the 35th ICML. [arXiv:1801.04062](https://arxiv.org/abs/1801.04062)

2. Bader, G.D. & Hogue, C.W.V. (2003). *An automated method for finding
   molecular complexes in large protein interaction networks.*
   BMC Bioinformatics, 4:2.

3. Li, S. et al. (2014). *Molecular signatures of antibody responses derived
   from a systems biology study of five human vaccines.*
   Nature Immunology, 15:195–204.


# First tests:

& "C:\Users\emari\OneDrive - Universitaet Bern (1)\Documents\Environments\scimilarity_2024_local\Scripts\python.exe" run_pipeline.py --output ./output --device cuda --perms 10000 --mode global --pval 0.001 --epochs 200 --batch-pairs 512 --prescreen-threshold 0.9  

-2.7 hours run using GPU!
#  first result
38,342 edges across 5,014 genes — edge appears in ≥ 5 of 17 studies
Extreme hub structure:
ENSSSCG00000027172: miRNA degree 3,977 (connected to 79% of all nodes!)
ENSSSCG00000036894: miRNA degree 3,791
ENSSSCG00000045186: not-annotated degree 3,663

##### next run expected: Key insight: the runtime scales with max-pairs, not with threshold. Whether 5M pairs come from |r|>0.3 or |r|>0.7 doesn't matter — MINE trains on the same number of batches either way. So:

5M pairs → 9,766 batches × 17 studies → roughly 27 hours on your GPU
2M pairs → ~3,906 batches × 17 studies → roughly 11 hours
1M pairs → ~1,953 batches × 17 studies → roughly 5.5 hours

### Second test:
(scimilarity_2024_local) PS C:\Users\emari\OneDrive - Universitaet Bern\GCB\GRANTS\DSL 2026 MURREN\Course\MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED> & "C:\Users\emari\OneDrive - Universitaet Bern (1)\Documents\Environments\scimilarity_2024_local\Scripts\python.exe" run_pipeline.py --output ./output --device cuda --perms 10000 --mode global --pval 0.001 --epochs 200 --batch-pairs 512 --prescreen-threshold 0.3 --max-pairs 50000000 --prescreen-method "spearman"
nvidia-smi --> ps -fp PID || true --> check if the process belongs to you or the script.
## to sloow on my machine, deploy on the cluster.

# Maybe helps to remove first ribosomal genes?

# IMPLEMENTED add filtering ability and visualization from DSL and furthermore leiden on weighted net and submodule if number of genes larger than 200 per module du mcode in submodules, read DSL repo readme.


### smoke test boot
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED && source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate && module load PyTorch && export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages && export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages && export PYTHONPATH=${VENV_SITE}:${TORCH_SITE} && export NUMEXPR_DISABLED=1 && python run_pipeline.py --output ./output/smoke_test_boot --device cuda --perms 10 --mode global --pval 0.05 --epochs 1 --batch-pairs 64 --prescreen-threshold 0.5 --max-pairs 1000 --prescreen-method spearman --mad-top-genes 500 --qc-preplot --qc-postplot --module-method leiden --master-edge-weight mean_neglog10p --normalize-weights --submodule-size-threshold 50

### smoke test leiden quick
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate
module load PyTorch
export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages
export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages
export PYTHONPATH=${VENV_SITE}:${TORCH_SITE}
export NUMEXPR_DISABLED=1

python run_pipeline.py \
  --output ./output/smoke_test_leiden_quick \
  --device cuda \
  --perms 10 \
  --mode global \
  --pval 0.05 \
  --epochs 1 \
  --batch-pairs 64 \
  --prescreen-threshold 0.5 \
  --max-pairs 1000 \
  --prescreen-method spearman \
  --mad-top-genes 500 \
  --qc-preplot \
  --qc-postplot \
  --module-method leiden \
  --master-edge-weight mean_neglog10p \
  --normalize-weights \
  --submodule-size-threshold 50


#### Test with all the integrations, fitlering, max pairs, leiden and pre and post filtering visu.
##### smoke test leiden


cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate
module load PyTorch
export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages
export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages
export PYTHONPATH=${VENV_SITE}:${TORCH_SITE}
export NUMEXPR_DISABLED=1

python run_pipeline.py --output ./output/smoke_test_leiden --device cuda --perms 10000 --mode global --pval 0.01 --epochs 100 --batch-pairs 512 --prescreen-threshold 0.3 --max-pairs 1000000 --prescreen-method "spearman" --mad-top-genes 10000 --qc-preplot --qc-postplot --module-method leiden --master-edge-weight mean_neglog10p --normalize-weights --submodule-size-threshold 200 --download-gmt

#### test adding new mcode density and size, lower the density to get a module but need modules with at least 10 genes, and convert pig to human before enrichment of data.
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate && module load PyTorch && export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages && export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages && export PYTHONPATH=${VENV_SITE}:${TORCH_SITE} && export NUMEXPR_DISABLED=1 && python run_pipeline.py --output ./output/smoke_test_leiden_v2 --device cuda --perms 1000 --mode global --pval 0.05 --epochs 50 --batch-pairs 256 --prescreen-threshold 0.5 --max-pairs 50000 --prescreen-method spearman --mad-top-genes 2000 --qc-preplot --qc-postplot --module-method leiden --master-edge-weight mean_neglog10p --normalize-weights --submodule-size-threshold 100 --mcode-min-size 10 --mcode-min-density 0.05   --ortholog-map ../gene_id_mapping.tsv --ortholog-source-col ensembl_gene_id --ortholog-target-col external_gene_name --download-gmt

# New Improvement now it allows to output more features as annotated modules "translated" the ensembl gene ids in anntoated modules.
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate && module load PyTorch && export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages && export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages && export PYTHONPATH=${VENV_SITE}:${TORCH_SITE} && export NUMEXPR_DISABLED=1 && python run_pipeline.py --output ./output/smoke_test_leiden_v2 --device cuda --perms 1000 --mode global --pval 0.05 --epochs 50 --batch-pairs 256 --prescreen-threshold 0.5 --max-pairs 50000 --prescreen-method spearman --mad-top-genes 2000 --qc-preplot --qc-postplot --module-method leiden --master-edge-weight mean_neglog10p --normalize-weights --submodule-size-threshold 100 --mcode-min-size 10 --mcode-min-density 0.05   --ortholog-map ../gene_id_mapping.tsv --ortholog-source-col ensembl_gene_id --ortholog-target-col external_gene_name --download-gmt --module-export-map ../gene_id_mapping.tsv --module-export-key-col ensembl_gene_id --module-export-cols external_gene_name entrezgene_id

## keeps now per gmt enrichment of modules, so we can explore those of interest, and also outputs a mini network per study and main network colored by main modules or (leiden) submodules (mcode)
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate && module load PyTorch && export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages && export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages && export PYTHONPATH=${VENV_SITE}:${TORCH_SITE} && export NUMEXPR_DISABLED=1 && python run_pipeline.py --output ./output/smoke_test_leiden_v2 --device cuda --perms 1000 --mode global --pval 0.05 --epochs 50 --batch-pairs 256 --prescreen-threshold 0.5 --max-pairs 50000 --prescreen-method spearman --mad-top-genes 2000 --qc-preplot --qc-postplot --module-method leiden --master-edge-weight mean_neglog10p --normalize-weights --submodule-size-threshold 100 --mcode-min-size 10 --mcode-min-density 0.05   --ortholog-map ../gene_id_mapping.tsv --ortholog-source-col ensembl_gene_id --ortholog-target-col external_gene_name --download-gmt --module-export-map ../gene_id_mapping.tsv --module-export-key-col ensembl_gene_id --module-export-cols external_gene_name entrezgene_id --save-per-gmt-enrichments --include-network-visualization

cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED && source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate && module load PyTorch && export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages && export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages && export PYTHONPATH=${VENV_SITE}:${TORCH_SITE} && export NUMEXPR_DISABLED=1 && python run_pipeline.py --output ./output/smoke_test_leiden_v2_res1p2 --device auto --perms 1000 --mode global --pval 0.05 --epochs 50 --batch-pairs 256 --prescreen-threshold 0.5 --max-pairs 50000 --prescreen-method spearman --mad-top-genes 2000 --qc-preplot --qc-postplot --module-method leiden --module-leiden-resolution 1.2 --module-min-size 3 --master-edge-weight mean_neglog10p --normalize-weights --submodule-method leiden --submodule-size-threshold 100 --submodule-leiden-resolution 1.0 --submodule-min-size 10 --ortholog-map ../gene_id_mapping.tsv --ortholog-source-col ensembl_gene_id --ortholog-target-col external_gene_name --download-gmt --module-export-map ../gene_id_mapping.tsv --module-export-key-col ensembl_gene_id --module-export-cols external_gene_name entrezgene_id --save-per-gmt-enrichments --include-network-visualization

# smoke test leiden v2 improved... with arguments controlling for firs modules and second submodule algorithm.
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED && source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate && module load PyTorch && export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages && export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages && export PYTHONPATH=${VENV_SITE}:${TORCH_SITE} && export NUMEXPR_DISABLED=1 && python run_pipeline.py --output ./output/compare_5M_3size_mod_ledien_res1p2_10size_submod_mcode_005 --device auto --perms 10000 --mode global --pval 0.05 --epochs 100 --batch-pairs 512 --prescreen-threshold 0.1 --max-pairs 5000000 --prescreen-method spearman --mad-top-genes 25000 --qc-preplot --qc-postplot --module-method leiden --module-leiden-resolution 1.2 --module-min-size 10 --master-edge-weight mean_neglog10p --normalize-weights --submodule-method mcode --submodule-size-threshold 200 --submodule-min-size 10 --submodule-mcode-score-threshold 0.2 --submodule-mcode-min-density 0.3 --ortholog-map ../gene_id_mapping.tsv --ortholog-source-col ensembl_gene_id --ortholog-target-col external_gene_name --download-gmt --module-export-map ../gene_id_mapping.tsv --module-export-key-col ensembl_gene_id --module-export-cols external_gene_name entrezgene_id --save-per-gmt-enrichments --include-network-visualization

# above run used the same space in GPU, wanted to make it a little faster and bigger the preocess by processing the double batch size and using less epochs..
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED && source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate && module load PyTorch && export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages && export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages && export PYTHONPATH=${VENV_SITE}:${TORCH_SITE} && export NUMEXPR_DISABLED=1 && python run_pipeline.py --output ./output/compare_1024_001_10000_60_50M_3size_mod_ledien_res1p2_10size_submod_mcode_005 --device auto --perms 10000 --mode global --pval 0.01 --epochs 60 --batch-pairs 1024 --prescreen-threshold 0.1 --max-pairs 50000000 --prescreen-method spearman --mad-top-genes 25000 --qc-preplot --qc-postplot --module-method leiden --module-leiden-resolution 1.2 --module-min-size 10 --master-edge-weight mean_neglog10p --normalize-weights --submodule-method mcode --submodule-size-threshold 200 --submodule-min-size 10 --submodule-mcode-score-threshold 0.2 --submodule-mcode-min-density 0.3 --ortholog-map ../gene_id_mapping.tsv --ortholog-source-col ensembl_gene_id --ortholog-target-col external_gene_name --download-gmt --module-export-map ../gene_id_mapping.tsv --module-export-key-col ensembl_gene_id --module-export-cols external_gene_name entrezgene_id --save-per-gmt-enrichments --include-network-visualization
# this increased a little the load on the GPU memory, but the workload is massive while memory not such a big issue.

## Smoke Tests: Single GPU vs Parallel 2-GPU

Use the same lightweight settings in both tests so runtime and outputs are comparable.

### A) Baseline smoke test (1 GPU worker)

```bash
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate
python /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED/run_pipeline.py --counts data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/logCPM_matrix_filtered_samples.csv --meta /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/metadata_with_sample_annotations.csv --output /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED/output_smoke_1gpu --device cuda --study-gpu-workers 1 --epochs 15 --batch-pairs 256 --perms 50 --prescreen-threshold 0.4 --max-pairs 120000 --min-studies 1 --module-method leiden --module-leiden-resolution 1.1 --module-min-size 3 --submodule-method none
```

### B) Parallel smoke test (2 GPU workers, one study per worker)

cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED && source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate && module load PyTorch && export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages && export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages && export PYTHONPATH=${VENV_SITE}:${TORCH_SITE} && export NUMEXPR_DISABLED=1 && python run_pipeline.py --output ./output/smoke_test_gpu_2 --device cuda --study-gpu-workers 2 --perms 10000 --mode global --pval 0.05 --epochs 100 --batch-pairs 512 --prescreen-threshold 0.1 --max-pairs 5000000 --prescreen-method spearman --mad-top-genes 25000 --qc-preplot --qc-postplot --module-method leiden --module-leiden-resolution 1.2 --module-min-size 10 --master-edge-weight mean_neglog10p --normalize-weights --submodule-method mcode --submodule-size-threshold 200 --submodule-min-size 10 --submodule-mcode-score-threshold 0.2 --submodule-mcode-min-density 0.3 --ortholog-map ../gene_id_mapping.tsv --ortholog-source-col ensembl_gene_id --ortholog-target-col external_gene_name --download-gmt --module-export-map ../gene_id_mapping.tsv --module-export-key-col ensembl_gene_id --module-export-cols external_gene_name entrezgene_id --save-per-gmt-enrichments --include-network-visualization


Expected parallel log indicator:

- Study-level GPU parallel mode enabled: 2 workers on cuda:0, cuda:1

If only one usable GPU is visible, the pipeline will warn and run sequentially.
### -- > working on double GPU!!!

# Re-Run whole datasets with no capping on 2 gpus save outputs.

cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED && source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate && module load PyTorch && export VENV_SITE=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages && export TORCH_SITE=${EBROOTPYTORCH}/lib/python3.9/site-packages && export PYTHONPATH=${VENV_SITE}:${TORCH_SITE} && export NUMEXPR_DISABLED=1 && python run_pipeline.py --output ./output/wholde_dataset_in_gpu_2 --device cuda --study-gpu-workers 2 --perms 30000 --mode global --pval 0.05 --epochs 150 --batch-pairs 512 --prescreen-threshold 0.3 --max-pairs 500000000 --prescreen-method spearman --mad-top-genes 32763 --qc-preplot --qc-postplot --min-studies 2 --module-method leiden --module-leiden-resolution 1.2 --module-min-size 10 --master-edge-weight mean_neglog10p --normalize-weights --submodule-method mcode --submodule-size-threshold 200 --submodule-min-size 10 --submodule-mcode-score-threshold 0.2 --submodule-mcode-min-density 0.3 --ortholog-map ../gene_id_mapping.tsv --ortholog-source-col ensembl_gene_id --ortholog-target-col external_gene_name --download-gmt --module-export-map ../gene_id_mapping.tsv --module-export-key-col ensembl_gene_id --module-export-cols external_gene_name entrezgene_id --save-per-gmt-enrichments --include-network-visualization

Results in output of whole_dataset_in_gpu_2 --> made two networks ready for the studies: adj_mine_PRJNA1107598.mtx and adj_mine_PRJNA1163897.mtx

# runs on two studies the gene pairs.
# Use now after improvement to check the time in UBELIX is not giving slots easy, but the pipeline with the bash wrap for running the whole dataset in the gpus of IBU.
Wraper is here: /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED/run_whole_dataset_in_gpu_2.sh

## Continue or start a production run on both GPUs
MAX_PAIRS=300000000 STUDY_GPU_WORKERS=2 STUDY_GPU_DEVICES="cuda:0 cuda:1" BATCH_PAIRS=auto PVAL=0.001 PERMS=30000 NO_RESUME_STUDIES=0 NO_REUSE_MINE_SCORES=0 ./run_whole_dataset_in_gpu_2.sh

## If you want to continue an existing interrupted run, force same output folder

OUTPUT_DIR=/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED/output/wholde_dataset_in_gpu_2_300000000 MAX_PAIRS=300000000 STUDY_GPU_WORKERS=2 STUDY_GPU_DEVICES="cuda:0 cuda:1" NO_RESUME_STUDIES=0 NO_REUSE_MINE_SCORES=0 ./run_whole_dataset_in_gpu_2.sh




















# TODO Apply pruning plus TF-prior filtering before claiming regulatory biology.

graph TD
    A["32,763 genes × 32,763<br/>536M possible pairs"] --> B["Spearman |r| &gt; 0.3<br/>Prescreen"]
    B -->|"~245M pairs<br/>~46% of all pairs"| C["MINE MI Estimation<br/>480k-712k batches"]
    C -->|"MI scores per pair"| D["Permutation Null<br/>30k permutations<br/>Global null"]
    D -->|"p-values computed"| E["Significance Filter<br/>p &lt; 0.05"]
    E -->|"~245M edges<br/>~100% of screened pairs"| F["❌ PERMISSIVE<br/>Nearly all pre-screened<br/>pairs pass significance"]
    
    style E fill:#ff9999
    style F fill:#ff6666
    
    B1["⚡ PRUNE Here<br/>threshold: 0.3→0.5<br/>Reduces to 45-50M pairs"] -.-> B
    E1["⚡ PRUNE Here<br/>threshold: 0.05→0.001<br/>Instant, no recompute"] -.-> E
    G["Master Network<br/>min_studies ≥ 3"] --> H["245M undirected edges"]
    F -->|"Study-level results"| G
    
    style B1 fill:#99ff99
    style E1 fill:#99ff99


graph TD
    A["Undirected Edge<br/>A — B<br/>MI symmetric"] -->|"Current State"| B["Network is co-expression<br/>Not regulatory"]
    
    A -->|"Option 1:<br/>Add TF Motifs"| C["TF Database<br/>+ Promoter Motifs"]
    C -->|"A→B if<br/>A=TF &amp;<br/>motif at B"| C1["245M → 5-50M<br/>Directed edges<br/>Regulatory"]
    
    A -->|"Option 2:<br/>Causal Heuristic"| D["MI Asymmetry<br/>or Time-series"]
    D -->|"Keep asymmetric<br/>pairs only"| D1["245M → 80M<br/>Directed edges<br/>Partial info"]
    
    A -->|"Option 3:<br/>PPIs + Partial Corr"| E["PPI Database<br/>+ Conditional"]
    E -->|"Keep direct<br/>interactions"| E1["245M → 2-5M<br/>Physical<br/>High confidence"]
    
    style B fill:#ffcccc
    style C1 fill:#ccffcc
    style D1 fill:#ffffcc
    style E1 fill:#ccffff

graph TD
    START["Start: 17 Studies<br/>245M edges each<br/>Too permissive"] 
    
    START -->|"Phase 1<br/>Quick wins"| S1["Change p-value<br/>0.05 → 0.001<br/>(No recompute)"]
    S1 -->|"Instant filter<br/>per each study"| S1R["Study-level:<br/>245M → 10-30M edges"]
    
    S1R -->|"If good?"| DECIDE1{"Selective<br/>enough?"}
    DECIDE1 -->|"Yes"| COMPLETE["✓ Done<br/>~30M master edges<br/>Small modules"]
    DECIDE1 -->|"No, need more pruning"| S2["Phase 2: Rerun pipeline<br/>Change prescreen<br/>0.3 → 0.5"]
    
    S2 -->|"Recompute MINE<br/>on 45-50M pairs<br/>~2x faster"| S2R["Study-level:<br/>50M pairs → 10-30M edges"]
    S2R --> DECIDE2{"Better?"}
    DECIDE2 -->|"Yes"| COMPLETE
    DECIDE2 -->|"Need regulatory interpretation"| S3["Phase 3: Directionality<br/>Acquire TF database<br/>Add TF motifs"]
    
    S3 --> S3R["Master-level:<br/>30M → 5-50M<br/>Directed edges<br/>A→B if A=TF"]
    S3R --> S4["Phase 4: PPI Validation<br/>Filter by PPIs<br/>Code optional module quality"]
    S4 --> FINAL["✓ Final Network<br/>Directed, TF-backed<br/>PPI-validated<br/>2-10M edges"]
    
    COMPLETE --> END["Decision Point:<br/>Publish or Iterate?"]
    FINAL --> END
    
    style S1 fill:#e1f5ff
    style S2 fill:#fff9c4
    style S3 fill:#f3e5f5
    style S4 fill:#e8f5e9
    style START fill:#ffebee
    style END fill:#f3e5f5
    style COMPLETE fill:#e8f5e9
    style FINAL fill:#c8e6c9

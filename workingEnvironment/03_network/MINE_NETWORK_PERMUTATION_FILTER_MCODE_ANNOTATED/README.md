# MINE Network — Permutation Filter — MCODE — Annotated

**MINE-based gene co-expression network inference with permutation
significance testing, multi-study consensus, MCODE module detection,
and biological annotation.**

---

## Table of Contents

1. [Similarities & Differences vs. Prior Implementations](#similarities--differences)
2. [Pipeline Overview](#pipeline-overview)
3. [Package Structure](#package-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Output Files](#output-files)
8. [Improvements Over Initial Release](#improvements-over-initial-release)
9. [References](#references)

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
| `--gmt` | (none) | GMT files for annotation |
| `--download-gmt` | (off) | Auto-download GMT files from Enrichr API |
| `--enrichr-libs` | (all 5) | Enrichr library names to download |

---

## Configuration

All parameters are structured in typed dataclasses (see `mine_network/config.py`):

```
PipelineConfig
├── MINEConfig          hidden_dim, n_epochs, lr, ema_alpha, batch_pairs, ...
├── PrescreenConfig     enabled, method, threshold, max_pairs
├── PermutationConfig   n_permutations, seed, p_value_threshold, mode
├── NetworkConfig       min_study_count, min_study_fraction, min_samples_per_study
├── MCODEConfig         score_threshold, min_size, min_density
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

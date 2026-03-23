"""
mine_network — MINE-based gene co-expression network inference.
===============================================================

A Python package that replaces histogram-based mutual information (MI) with
**MINE** (Mutual Information Neural Estimation, Belghazi et al. ICML 2018)
for inferring statistically significant gene co-expression networks from
bulk RNA-seq data.

Pipeline overview
-----------------
1. **Data loading** — Load logCPM expression matrix + sample metadata.
   Studies are auto-discovered from the ``BioProject`` column.

2. **Per-study inference** —
   a. Z-score expression across samples (continuous, no binning).
   b. *(Optional)* Pre-screen gene pairs by Pearson |r| to reduce candidates.
   c. Estimate MI for each candidate pair using a small neural MINE network
      trained on the Donsker–Varadhan (DV) representation.
   d. Build an empirical null MI distribution via permutation
      (global-null or per-pair, configurable).
   e. Retain edges with empirical p-value < threshold (default 0.001).

3. **Master consensus network** — Keep edges significant in ≥ k studies.

4. **MCODE module detection** — Bader & Hogue (2003) algorithm for finding
   dense subgraph complexes.

5. **Biological annotation** — Hypergeometric enrichment of modules against
   user-provided gene-set collections (GMT format).

Quick start
-----------
>>> from mine_network.config import PipelineConfig
>>> from mine_network.pipeline import run_pipeline
>>>
>>> cfg = PipelineConfig(
...     counts_path="logCPM_matrix.csv",
...     metadata_path="metadata.csv",
...     output_dir="./results",
... )
>>> run_pipeline(cfg)

Entry point
-----------
From the command line::

    python run_pipeline.py                        # default paths
    python run_pipeline.py --output ./my_output   # override output dir

Modules
-------
config          Configuration dataclasses
data_loader     Expression + metadata loading, study discovery
mine_estimator  Batched MINE network and MI estimation
permutation     Permutation-based null distributions
network         Edge filtering, master network construction
mcode           MCODE dense-subgraph module detection
annotation      Gene-set enrichment annotation of modules
io_utils        Logging, timing, result saving
pipeline        End-to-end pipeline orchestrator

References
----------
Belghazi, M.I. et al. (2018). Mutual Information Neural Estimation.
    Proceedings of the 35th ICML. arXiv:1801.04062.

Bader, G.D. & Hogue, C.W.V. (2003). An automated method for finding
    molecular complexes in large protein interaction networks.
    BMC Bioinformatics, 4:2.

Li, S. et al. (2014). Molecular signatures of antibody responses derived
    from a systems biology study of five human vaccines.
    Nature Immunology, 15:195–204.
"""

__version__ = "0.1.0"

from .config import PipelineConfig
from .pipeline import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline", "__version__"]

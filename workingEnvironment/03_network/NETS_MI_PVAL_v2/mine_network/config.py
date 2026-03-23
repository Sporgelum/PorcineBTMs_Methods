"""
Configuration — all pipeline parameters in structured dataclasses.
==================================================================

This module centralises every tuneable parameter so that:
  - The main pipeline script has no magic numbers.
  - Experiments are reproducible: serialise a ``PipelineConfig`` to JSON.
  - Defaults match the user's conceptual design (Section 4):
      • 1 000 permutations  (tractable even per-pair; Section 3)
      • p < 0.001           (empirical threshold; Section 3)
      • edges in ≥ 3 studies (master consensus; Section 5)

Dataclass hierarchy
-------------------
PipelineConfig
├── MINEConfig          MINE statistics network + training
├── PrescreenConfig     Optional Pearson/Spearman pre-filtering
├── PermutationConfig   Null distribution construction
├── NetworkConfig       Master network assembly rules
├── MCODEConfig         MCODE module detection thresholds
└── AnnotationConfig    Gene-set enrichment settings

Notes on the MINE architecture (Section 6)
-------------------------------------------
The statistics network T_θ(x, z) is a small MLP:

    Input  : (x_k, z_k) ∈ ℝ²  — one sample's expression for the gene pair
    Hidden : 2 → H → H → 1    with ELU activations
    Output : scalar T_θ for the Donsker–Varadhan bound

We train B such networks **simultaneously** using batched weight tensors
(Section 4b) so that a single ``torch.bmm`` call evaluates all B networks
in one GPU kernel.

Regarding sample sizes (Section 7)
-----------------------------------
Studies with as few as 3 samples are kept (``min_samples_per_study = 3``).
Their MI estimates will be noisy, but the multi-study ≥ 3 filter acts as a
strong meta-analytic safeguard — edges must replicate across independent
cohorts that often have larger sample sizes.

Why 10 000 permutations?
------------------------
With B = 10 000 the minimum resolvable p-value is ~1e-4, giving
substantially finer resolution than the 0.001 threshold.  This avoids
tied p-values at the boundary and provides a more reliable null.
Using a global null (Section 4c) makes this feasible even for large
gene counts.
"""

from dataclasses import dataclass, field
from typing import Union


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-configurations
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MINEConfig:
    """
    MINE statistics network architecture and training hyper-parameters.

    The network T_θ : ℝ² → ℝ is a 3-layer MLP with ELU activations.
    It is trained to maximise the Donsker–Varadhan (DV) lower bound on MI:

        I(X;Z) ≥ E_P[T] − log E_Q[e^T]

    where P = joint, Q = product of marginals.

    Attributes
    ----------
    hidden_dim : int
        Width of both hidden layers (Section 4a: 32–64).
    n_epochs : int
        SGD training epochs per batch of gene pairs.
    learning_rate : float
        Adam optimiser learning rate.
    ema_alpha : float
        Exponential moving average decay for the bias-correction term
        in the DV gradient estimator (Paper §3.2, Eq. 10).
    batch_pairs : int or str
        Number of gene pairs processed simultaneously (Section 4b).
        - int (e.g., 512): fixed batch size
        - "auto": dynamically compute based on device VRAM (recommended)
    gradient_clip : float
        Maximum gradient norm (stabilises training with small n).
    n_eval_shuffles : int
        Number of marginal shuffles averaged for the final MI estimate
        after training.  Reduces variance of the point estimate.
    mixed_precision : bool
        Enable float16 forward pass via torch.autocast for additional
        memory savings. Reduces memory ~40%, enables larger batches.
        Default False (conservative); set True to enable.
    """
    hidden_dim: int = 64
    n_epochs: int = 200
    learning_rate: float = 1e-3
    ema_alpha: float = 0.01
    batch_pairs: Union[int, str] = 512
    gradient_clip: float = 1.0
    n_eval_shuffles: int = 5
    mixed_precision: bool = False


@dataclass
class PrescreenConfig:
    """
    Optional fast correlation pre-screening.

    With G genes there are G(G-1)/2 unique pairs — potentially hundreds of
    millions.  Training a MINE network per pair is costly, so we optionally
    pre-filter using |Pearson r| > threshold (Section 4c).

    This is sound because pairs with near-zero linear *and* nonlinear MI
    dominate.  MINE's advantage is for pairs with moderate |r| masking
    stronger nonlinear dependencies; very low |r| pairs almost always have
    low MI too.

    Set ``enabled = False`` to skip pre-screening and estimate MI for
    *all* unique pairs (only feasible with small gene sets).

    Attributes
    ----------
    enabled : bool
        Whether to apply pre-screening.
    method : str
        "pearson" or "spearman".
    threshold : float
        Minimum |r| to retain a pair.  0.3 is conservative.
    max_pairs : int
        Hard cap on candidate pairs per study.  If the |r| filter
        returns more, the threshold is dynamically raised.
    """
    enabled: bool = True
    method: str = "pearson"
    threshold: float = 0.3
    max_pairs: int = 5_000_000


@dataclass
class PermutationConfig:
    """
    Permutation-based null distribution for significance testing.

    The user's design (Section 3) specifies:
      - 1 000 permutations (default; increase for finer p-value resolution)
      - p < 0.001 threshold
      - Global null per study (Section 4c: "probably the sweet spot")

    Two modes are supported:
      ``mode = "global"``  — One null distribution per study.  Random gene
        pairs are picked, one gene is permuted, MI is estimated.  Fast.
      ``mode = "per_pair"`` — Full per-pair permutation (Section 3).  More
        rigorous but O(n_pairs × n_permutations).  Only feasible with
        aggressive pre-screening.

    Attributes
    ----------
    n_permutations : int
        Null samples per study (global mode) or per pair (per-pair mode).
    seed : int
        Random seed for reproducibility.
    p_value_threshold : float
        Edges with p < this are retained.
    mode : str
        "global" (one distribution per study) or "per_pair" (per gene pair).
    """
    n_permutations: int = 10_000
    seed: int = 42
    p_value_threshold: float = 0.001
    mode: str = "global"


@dataclass
class NetworkConfig:
    """
    Master consensus network construction rules.

    From the user's design (Section 5):
      - Edge presence indicator: c_ij = #{studies where A_ij = 1}
      - Master network: keep edges with c_ij ≥ 3

    Attributes
    ----------
    min_study_count : int
        Static fallback: edge must appear in ≥ this many studies.
    min_study_fraction : float or None
        Dynamic: edge in ≥ fraction × n_studies studies.
        Overrides min_study_count when set.  None = use static.
    min_samples_per_study : int
        Minimum samples a BioProject must have to be included.
        Set to 3 per user request (Section 7 acknowledges noise but
        the multi-study filter compensates).
    weight_by : str or None
        Optional edge weighting in master network:
        "count" = number of supporting studies,
        "mean_mi" = mean MI across supporting studies,
        None = unweighted.
    """
    min_study_count: int = 3
    min_study_fraction: float = None
    min_samples_per_study: int = 3
    weight_by: str = "count"


@dataclass
class ModuleConfig:
    """
    Module detection and weighted-edge aggregation options.

    Attributes
    ----------
    method : str
        First-pass module detector on the master network:
        ``"mcode"`` or ``"leiden"``.
    submodule_method : str
        Refinement detector for oversized modules:
        ``"none"``, ``"mcode"``, or ``"leiden"``.
    master_edge_weight : str
        Master edge weighting mode:
        - ``"n_studies"``      : support count across studies
        - ``"mean_mi"``        : mean MI among significant study edges
        - ``"mean_neglog10p"`` : mean -log10(p + eps) among significant edges
    normalize_weights : bool
        If True, min-max normalize study-level weights before aggregation.
    weight_clip_min : float or None
        Optional lower clip for study-level weights before aggregation.
    weight_clip_max : float or None
        Optional upper clip for study-level weights before aggregation.
    weight_eps : float
        Numerical stabilizer for significance weights.
    module_min_size : int
        Minimum size for first-pass modules.
    module_leiden_resolution : float
        Leiden resolution for first-pass modules.
    module_leiden_iterations : int
        Leiden iterations for first-pass modules.
    module_mcode_score_threshold : float
        MCODE score-threshold for first-pass modules.
    module_mcode_min_density : float
        MCODE minimum density for first-pass modules.
    submodule_size_threshold : int or None
        If set, modules larger than this are refined by ``submodule_method``.
    submodule_min_size : int
        Minimum size for submodules produced during refinement.
    submodule_leiden_resolution : float
        Leiden resolution for refinement when ``submodule_method=leiden``.
    submodule_leiden_iterations : int
        Leiden iterations for refinement when ``submodule_method=leiden``.
    submodule_mcode_score_threshold : float
        MCODE score-threshold for refinement when ``submodule_method=mcode``.
    submodule_mcode_min_density : float
        MCODE minimum density for refinement when ``submodule_method=mcode``.
    """
    method: str = "mcode"
    submodule_method: str = "none"
    master_edge_weight: str = "n_studies"
    normalize_weights: bool = False
    weight_clip_min: float = None
    weight_clip_max: float = None
    weight_eps: float = 1e-12
    module_min_size: int = 3
    module_leiden_resolution: float = 1.0
    module_leiden_iterations: int = -1
    module_mcode_score_threshold: float = 0.2
    module_mcode_min_density: float = 0.3
    submodule_size_threshold: int = None
    submodule_min_size: int = 3
    submodule_leiden_resolution: float = 1.0
    submodule_leiden_iterations: int = -1
    submodule_mcode_score_threshold: float = 0.2
    submodule_mcode_min_density: float = 0.3


@dataclass
class MCODEConfig:
    """
    MCODE dense-subgraph module detection (Bader & Hogue 2003).

    Matching Li et al. (Nat Immunol 2014) / Cytoscape defaults.

    Algorithm stages:
      1. Vertex weighting: w(v) = k-core(v) × local_density(v)
      2. Seed-and-extend: BFS from highest-weight seeds
      3. Post-processing: filter by min_size and min_density

    Attributes
    ----------
    score_threshold : float
        Fraction of max node weight; neighbours below this × max_weight
        are excluded during seed extension.
    min_size : int
        Discard modules with fewer genes than this.
    min_density : float
        Discard modules with edge density below this.  Li et al. use 0.3.
    """
    score_threshold: float = 0.2
    min_size: int = 3
    min_density: float = 0.3


@dataclass
class AnnotationConfig:
    """
    Gene-set enrichment / biological annotation of MCODE modules.

    Modules are tested for enrichment against user-provided gene-set
    collections using hypergeometric (Fisher's exact) tests with
    Benjamini–Hochberg FDR correction.

    Gene-set files should be in GMT format (tab-separated):
        <set_name>\\t<description>\\t<gene1>\\t<gene2>\\t...

    Standard sources: MSigDB (Hallmark, C2-CP, C5-GO), KEGG, Reactome.

    Attributes
    ----------
    gmt_paths : list[str]
        Paths to GMT gene-set files.  Multiple collections supported.
    download_enrichr : bool
        If True, automatically download gene-set libraries from the
        Enrichr API (Ma'ayan Lab) before annotation.
    enrichr_libraries : list[str] or None
        Specific Enrichr library names to download.  If None and
        ``download_enrichr`` is True, downloads the default set:
        GO_Biological_Process_2023, KEGG_2021_Human, Reactome_2022,
        WikiPathway_2023_Human, MSigDB_Hallmark_2020.
    fdr_threshold : float
        Maximum BH-adjusted p-value to report an enrichment.
    min_overlap : int
        Minimum number of module genes in a gene set to test it.
    background_genes : str or None
        Path to a text file with one gene per line defining the universe.
        If None, all genes in the expression matrix are used.
    ortholog_map_path : str or None
        Optional TSV mapping file to translate module genes before
        enrichment (for example pig symbols -> human symbols).
    ortholog_source_col : str
        Source-species column name in mapping file.
    ortholog_target_col : str
        Target-species column name in mapping file.
    module_export_map_path : str or None
        Optional TSV/CSV file used to append extra identifier columns to
        module membership output tables.
    module_export_key_col : str
        Column in ``module_export_map_path`` matching the module gene IDs.
    module_export_cols : list[str]
        Extra columns to append to module tables. If empty, all columns in
        the mapping file except ``module_export_key_col`` are appended.
    save_per_gmt_results : bool
        If True, split enrichment output into per-library folders under
        ``enrichments_gmt/`` while still saving global combined tables.
    """
    gmt_paths: list = field(default_factory=list)
    download_enrichr: bool = False
    enrichr_libraries: list = field(default_factory=list)
    fdr_threshold: float = 0.05
    min_overlap: int = 2
    background_genes: str = None
    ortholog_map_path: str = None
    ortholog_source_col: str = "pig_gene"
    ortholog_target_col: str = "human_gene"
    module_export_map_path: str = None
    module_export_key_col: str = "ensembl_gene_id"
    module_export_cols: list = field(default_factory=list)
    save_per_gmt_results: bool = False


@dataclass
class VisualizationConfig:
    """
    Optional network minimap rendering settings.

    Attributes
    ----------
    enabled : bool
        If True, save small PNG minimaps for study and master networks.
    max_nodes : int
        Max nodes drawn per minimap (highest-degree nodes retained if larger).
    dpi : int
        PNG resolution.
    edge_alpha : float
        Transparency for drawn edges.
    """
    enabled: bool = False
    max_nodes: int = 1200
    dpi: int = 180
    edge_alpha: float = 0.08


@dataclass
class GeneFilterConfig:
    """
    Optional gene-level prefiltering before candidate-pair generation.

    Useful for removing high-degree hubs that are often biologically
    uninformative for co-expression structure (for example ribosomal or
    miRNA features), which can dominate downstream network topology.

    Attributes
    ----------
    enabled : bool
        Master switch for applying any gene filtering.
    remove_ribosomal : bool
        Exclude gene names matching ribosomal prefixes:
        RPL, RPS, MRPL, MRPS.
    remove_mirna : bool
        Exclude gene names matching miRNA-like patterns:
        MIR, MIRLET, LET-7, miR-.
    custom_regex : str or None
        Optional additional regex (case-insensitive) to exclude genes.
    exclude_genes_file : str or None
        Optional text file with one gene name per line to exclude.
    """
    enabled: bool = False
    remove_ribosomal: bool = False
    remove_mirna: bool = False
    custom_regex: str = None
    exclude_genes_file: str = None


@dataclass
class QCConfig:
    """
    Optional exploratory QC visualisation and MAD gene filtering.

    Attributes
    ----------
    plot_pre_filter : bool
        Save a three-panel sample QC figure before MAD filtering.
    plot_post_filter : bool
        Save the same QC figure after MAD filtering.
    mad_top_genes : int or None
        If set, keep only the top-N genes ranked by MAD across all samples.
    line_quantiles : int
        Number of quantile points for the sample distribution line panel.
    """
    plot_pre_filter: bool = False
    plot_post_filter: bool = False
    mad_top_genes: int = None
    line_quantiles: int = 200


# ═══════════════════════════════════════════════════════════════════════════════
# Master configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """
    Master configuration — glues all sub-configs and file paths.

    Attributes
    ----------
    mine : MINEConfig
    prescreen : PrescreenConfig
    permutation : PermutationConfig
    network : NetworkConfig
    module : ModuleConfig
    mcode : MCODEConfig
    annotation : AnnotationConfig
    visualization : VisualizationConfig
    gene_filter : GeneFilterConfig
    qc : QCConfig
    counts_path : str
        Path to the logCPM expression matrix (genes × samples, tab-separated).
    metadata_path : str
        Path to the sample metadata (must contain 'Run' and 'BioProject').
    output_dir : str
        Directory for all output files.
    device : str
        PyTorch device: "auto", "cuda", "cpu".
    n_jobs : int
        CPU cores for pre-screening parallelism (-1 = all cores).
    study_gpu_workers : int
        Number of concurrent study workers for GPU execution.
        Use 1 for sequential behavior (default), 2 to run one study per GPU
        on two visible devices.
    study_gpu_devices : list[str]
        Optional explicit CUDA device list for study workers, for example
        ["cuda:0", "cuda:1"]. If empty and ``study_gpu_workers > 1``,
        usable visible CUDA devices are auto-detected.
    apply_bh_fdr : bool
        Also save BH-corrected edge lists per study.
    bh_fdr_alpha : float
        FDR level for BH correction.
    resume_completed_studies : bool
        If True, skip studies whose per-study outputs already exist and
        load their saved results for master-network aggregation.
    reuse_mine_scores : bool
        If True, reuse cached per-study MINE scores when available
        (``mine_diagnostics/mine_scores_{study}.npz``) to avoid retraining.
    save_mine_score_cache : bool
        If True, save per-study MINE score cache for later resume runs.
    """
    mine: MINEConfig = field(default_factory=MINEConfig)
    prescreen: PrescreenConfig = field(default_factory=PrescreenConfig)
    permutation: PermutationConfig = field(default_factory=PermutationConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    module: ModuleConfig = field(default_factory=ModuleConfig)
    mcode: MCODEConfig = field(default_factory=MCODEConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    gene_filter: GeneFilterConfig = field(default_factory=GeneFilterConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    # Paths — set at runtime or via CLI
    counts_path: str = ""
    metadata_path: str = ""
    output_dir: str = ""

    # Hardware
    device: str = "auto"
    n_jobs: int = -1
    study_gpu_workers: int = 1
    study_gpu_devices: list = field(default_factory=list)

    # Optional BH-FDR per study
    apply_bh_fdr: bool = False
    bh_fdr_alpha: float = 0.05

    # Resume / checkpoint behavior
    resume_completed_studies: bool = True
    reuse_mine_scores: bool = True
    save_mine_score_cache: bool = True

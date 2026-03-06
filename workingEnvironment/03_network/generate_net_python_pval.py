##!/usr/bin/env python3
"""
Gene Network Inference: Permutation-based MI Edge Significance + Multi-Study Consensus

Pipeline overview
-----------------
For each study / dataset:
  1.  Load log-CPM expression matrix (genes × samples).
  2.  Discretize expression with equal-frequency (quantile) binning – 5 bins by default.
  3.  Compute the full pairwise Mutual Information (MI) matrix in parallel using a fast
      histogram estimator (same approach as generate_net_python.py).
  4.  Build an empirical null MI distribution via 30,000 random shuffled gene pairs:
        - For each trial k, pick random genes i and j, shuffle gene j's sample vector,
          compute MI(gene_i, shuffle(gene_j)).
        - Under equal-frequency discretization all genes share the same uniform marginal,
          so the null distribution is gene-pair-agnostic (one distribution serves all pairs).
  5.  For each observed MI(i, j), the empirical p-value is
          p(i,j) = #{null_k >= MI(i,j)} / N_PERMUTATIONS
      An edge is retained when p(i,j) < P_VALUE_THRESHOLD (default 0.001),
      equivalent to MI(i,j) > 99.9th percentile of the null distribution.
  6.  Save per-study significant edge list (with MI value and p-value).

Master reference network:
  7.  Count how many studies each edge was significant in.
  8.  Retain edges appearing in >= MIN_STUDY_COUNT studies (default 3).
  9.  Detect modules in master network with MCODE (Bader & Hogue 2003),

Statistical notes
-----------------
* Global null is valid under quantile (equal-frequency) discretization because
  all discretized gene marginals are approximately uniform, making the null
  distribution gene-pair-agnostic.
* With N_PERMUTATIONS = 30,000 the minimum resolvable p-value is ~3.3e-5,
  comfortably below the 0.001 threshold.
* Per-study type-I error rate: ~0.001 × N_pairs false positive edges expected.
  After the multi-study filter (k >= 3), the expected FP rate drops to ~0.001^k ≈ 1e-9.
  The two-stage design provides strong FDR control without Bonferroni.
* Optional Benjamini-Hochberg FDR correction is also implemented and can be
  enabled by setting APPLY_BH_FDR = True.

Improvements over generate_net_python.py
-----------------------------------------
- Principled significance test replaces arbitrary percentile threshold.
- 30,000 permutations computed ONCE per study (not per gene pair) – feasible for
  large gene sets (O(P × n_samples) instead of O(N² × P)).
- Exact empirical p-value stored per edge via vectorised np.searchsorted.
- Optional optional Benjamini-Hochberg FDR correction.
- Multi-study framework for meta-analysis / reproducibility filtering.
- Null distribution histogram saved for QC inspection.
- Study-count matrix saved separately (how many studies validated each edge).
- Gene-universe intersection handles cohorts with partially overlapping gene sets.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from joblib import Parallel, delayed
import igraph as ig
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = "/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment"
OUTPUT_DIR = os.path.join(BASE_DIR, "03_network/NETS_MI_PVAL")

# Ensure output directory exists before anything else writes to it
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Discretisation parameters (must match across studies for a valid common null)
N_BINS = 5
DISC_STRATEGY = "quantile"  # equal-frequency – gives uniform marginals

# Parallelism
N_JOBS = -1  # -1 = all available CPU cores

# Permutation test parameters
N_PERMUTATIONS = 30_000       # Number of null-distribution samples per study
P_VALUE_THRESHOLD = 0.001     # Empirical p-value cutoff for an edge
PERM_SEED = 42                # Reproducibility seed for the null distribution

# FDR correction (Benjamini-Hochberg) – applied *in addition* to raw p-value filter
APPLY_BH_FDR = False          # Set True to also save BH-corrected edge list
BH_FDR_ALPHA = 0.05           # FDR level (only used when APPLY_BH_FDR = True)

# Master network parameters
MIN_STUDY_COUNT = 3           # Edge must appear in >= this many studies

# ---- Shared input files (studies are auto-discovered from the metadata) ------
# The metadata CSV/TSV must contain at minimum two columns:
#   Run        – matches the column names of the logCPM matrix
#   BioProject – one unique BioProject ID per study / cohort
#
# The logCPM matrix is loaded once; each BioProject's Run columns are extracted
# automatically to form per-study expression sub-matrices.
COUNTS_PATH   = os.path.join(BASE_DIR, "02_counts/logCPM_matrix_filtered_samples.csv")
METADATA_PATH = os.path.join(BASE_DIR, "02_counts/metadata_with_sample_annotations.csv")

# Minimum number of samples a BioProject must have to be included as a study.
# Projects with too few samples produce unreliable MI estimates.
MIN_SAMPLES_PER_STUDY = 3 # change accordingly to sample size.. 

# MCODE parameters
MCODE_SCORE_THRESHOLD = 1.2
MCODE_MIN_SIZE = 3
MCODE_MIN_DENSITY = 0.1

# Log / report paths are assigned inside main() (not at module scope) so that
# dask or joblib worker subprocesses that import this module do not create
# spurious log files.
LOG_FILE = None
REPORT_FILE = None


# ============================================================================
# Logging / Timing utilities  (identical to generate_net_python.py)
# ============================================================================

class TeeLogger:
    """Writes to both stdout and a log file simultaneously."""
    def __init__(self, log_file):
        self.terminal = sys.__stdout__
        self.log = open(log_file, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class Timer:
    """Context manager that records wall-clock time for a named step."""
    def __init__(self, name, report_dict):
        self.name = name
        self.report_dict = report_dict

    def __enter__(self):
        self.start = time.time()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Starting: {self.name}")
        print("-" * 80)
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("-" * 80)
        print(f"[{ts}] Completed: {self.name}")
        print(f"[TIMING] Duration: {format_time(self.elapsed)}")
        self.report_dict[self.name] = self.elapsed


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes ({seconds:.1f}s)"
    else:
        h = seconds / 3600
        m = (seconds % 3600) / 60
        return f"{h:.2f} hours ({m:.1f}m)"


# ============================================================================
# Data Loading
# ============================================================================

def load_expression(counts_path):
    """
    Load the full logCPM expression matrix.

    Supports tab- and comma-separated files; auto-detects from extension.
    Returns a DataFrame with genes as rows and samples (Run IDs) as columns.
    """
    #sep = "\t" if counts_path.endswith((".tsv", ".txt")) else ","
    sep="\\t"
    expr = pd.read_csv(counts_path, sep=sep, index_col=0)
    print(f"[INFO] Loaded expression matrix: {counts_path}")
    print(f"[INFO] Shape: {expr.shape[0]} genes × {expr.shape[1]} samples")
    return expr


def load_metadata(metadata_path):
    """
    Load sample metadata.

    Expected columns (at minimum): Run, BioProject.
    Returns a DataFrame indexed by Run.
    """
    #sep = "\t" if metadata_path.endswith((".tsv", ".txt")) else ","
    sep="\\t"
    md = pd.read_csv(metadata_path, sep=sep)
    print(f"[INFO] Loaded metadata: {metadata_path}")
    print(f"[INFO] Metadata shape: {md.shape}")
    for col in ("Run", "BioProject"):
        if col not in md.columns:
            raise ValueError(
                f"Metadata file is missing required column '{col}'. "
                f"Found columns: {md.columns.tolist()}"
            )
    return md


def discover_studies(expr_full, metadata, min_samples=6):
    """
    Auto-detect studies from the BioProject column of the metadata.

    Each unique BioProject ID becomes one study.  Only Run IDs that are
    present as columns in the logCPM matrix are used (extra metadata rows
    are ignored silently).

    Parameters
    ----------
    expr_full   : pd.DataFrame (genes × all_samples)
    metadata    : pd.DataFrame  – must contain 'Run' and 'BioProject' columns
    min_samples : int  – skip BioProjects with fewer samples than this

    Returns
    -------
    studies : list of dict with keys
        name      – BioProject ID (safe for use in filenames)
        expr      – pd.DataFrame sub-matrix for that study (genes × study_samples)
        mi_cache  – path for caching the MI matrix
    """
    available_runs = set(expr_full.columns)
    # Keep only metadata rows whose Run appears in the expression matrix
    md_matched = metadata[metadata["Run"].isin(available_runs)].copy()

    n_unmatched = len(metadata) - len(md_matched)
    if n_unmatched:
        print(f"[WARN] {n_unmatched} metadata rows have no matching column in the "
              f"expression matrix and will be ignored.")

    studies = []
    for bioproj, group in md_matched.groupby("BioProject"):
        runs = group["Run"].tolist()
        if len(runs) < min_samples:
            print(f"[WARN] BioProject {bioproj}: only {len(runs)} samples "
                  f"(< MIN_SAMPLES_PER_STUDY={min_samples}) – skipping.")
            continue
        sub_expr = expr_full[runs]  # genes × study_samples
        # Sanitise BioProject ID for use in filenames
        safe_name = str(bioproj).replace(" ", "_").replace("/", "-")
        studies.append({
            "name": safe_name,
            "expr": sub_expr,
            "mi_cache": os.path.join(OUTPUT_DIR, f"mi_cache_{safe_name}.npy"),
        })
        print(f"[INFO] Study detected: {safe_name}  ({len(runs)} samples)")

    print(f"[INFO] Total studies discovered: {len(studies)}")
    return studies


# ============================================================================
# Discretisation
# ============================================================================

def discretize_expression(expr_data, n_bins=5, strategy="quantile"):
    """
    Quantile-discretise expression into n_bins equal-frequency bins.

    Matches the R code: disc(..., disc='equalfreq', nbins=5).
    Returns an integer array of shape (genes, samples).
    """
    print(f"[INFO] Discretising: {n_bins} bins, strategy={strategy}")
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    # sklearn expects (samples, features); transpose in / transpose out
    return disc.fit_transform(expr_data.values.T).T.astype(np.int32)


# ============================================================================
# MI computation  (identical fast histogram approach from generate_net_python.py)
# ============================================================================

def compute_mi_histogram(x, y, n_bins):
    """
    Compute MI(X; Y) from discrete vectors using a 2-D histogram.

    MI = sum_{x,y} P(x,y) * log( P(x,y) / (P(x)*P(y)) )
    """
    hist_2d, _, _ = np.histogram2d(
        x, y, bins=n_bins, range=[[0, n_bins], [0, n_bins]]
    )
    pxy = hist_2d / float(hist_2d.sum())
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz])))


def _mi_row(i, expr_discrete, n_bins):
    """Compute MI between gene i and all other genes (used by joblib)."""
    n_genes = expr_discrete.shape[0]
    target = expr_discrete[i, :]
    scores = np.zeros(n_genes)
    for j in range(n_genes):
        if i != j:
            scores[j] = compute_mi_histogram(target, expr_discrete[j, :], n_bins)
    if (i + 1) % 500 == 0:
        print(f"[INFO] MI progress: {i + 1}/{n_genes} genes")
    return scores


def compute_mi_matrix(expr_data, n_bins=5, strategy="quantile", n_jobs=-1,
                      cache_path=None):
    """
    Compute the full N×N mutual information matrix in parallel.

    If cache_path is provided and the file already exists, the cached matrix
    is loaded instead of recomputing.

    Returns (mi_matrix: np.ndarray float32, expr_discrete: np.ndarray int32).
    """
    expr_discrete = discretize_expression(expr_data, n_bins, strategy)
    n_genes = expr_data.shape[0]

    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached MI matrix from: {cache_path}")
        mi_matrix = np.load(cache_path).astype(np.float64)
        print(f"[INFO] Loaded: shape={mi_matrix.shape}")
        return mi_matrix, expr_discrete

    print(f"[INFO] Computing MI matrix ({n_genes} genes, {n_jobs} cores)...")
    t0 = time.time()
    mi_matrix = np.array(
        Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_mi_row)(i, expr_discrete, n_bins) for i in range(n_genes)
        )
    )
    # Symmetrise (floating-point noise may break exact symmetry)
    mi_matrix = (mi_matrix + mi_matrix.T) / 2.0
    print(f"[INFO] MI matrix computed in {format_time(time.time() - t0)}")
    print(f"[INFO] MI range: [{mi_matrix[mi_matrix > 0].min():.4f}, {mi_matrix.max():.4f}]")

    if cache_path:
        np.save(cache_path, mi_matrix.astype(np.float32))
        size_gb = os.path.getsize(cache_path) / 1e9
        print(f"[INFO] MI matrix cached to: {cache_path} ({size_gb:.2f} GB)")

    return mi_matrix, expr_discrete


# ============================================================================
# Permutation-based null distribution
# ============================================================================

def build_null_distribution(expr_discrete, n_bins, n_permutations=30_000, seed=42):
    """
    Build an empirical null distribution of MI under the null hypothesis of
    gene-pair independence.

    Algorithm
    ---------
    For each of the n_permutations trials:
      1. Randomly pick two gene indices i and j.
      2. Shuffle gene j's sample vector (breaks any dependency with gene i).
      3. Compute MI(gene_i, shuffle(gene_j)).

    Why a global null is valid
    --------------------------
    Under equal-frequency (quantile) discretization with n_bins bins, every gene's
    marginal distribution is approximately Uniform({0,...,n_bins-1}).  The null
    distribution of MI(X, shuffle(Y)) therefore does not depend on which gene pair
    (X, Y) we choose, so one global null distribution serves all N² pairs.

    The 30,000 permutations give a minimum resolvable p-value of ~3.3e-5, well
    below the target threshold of 0.001.

    Parameters
    ----------
    expr_discrete : ndarray (n_genes, n_samples)  – discretised expression
    n_bins        : int
    n_permutations: int
    seed          : int

    Returns
    -------
    null_mi : ndarray of shape (n_permutations,)
    """
    rng = np.random.default_rng(seed)
    n_genes, n_samples = expr_discrete.shape

    print(f"[INFO] Building null MI distribution: {n_permutations} permutations...")
    t0 = time.time()

    null_mi = np.empty(n_permutations)
    gene_i_idx = rng.integers(0, n_genes, size=n_permutations)
    gene_j_idx = rng.integers(0, n_genes, size=n_permutations)

    for k in range(n_permutations):
        gene_i = expr_discrete[gene_i_idx[k], :]
        gene_j = expr_discrete[gene_j_idx[k], :]
        shuffled_j = rng.permutation(gene_j)
        null_mi[k] = compute_mi_histogram(gene_i, shuffled_j, n_bins)
        if (k + 1) % 5000 == 0:
            print(f"[INFO] Null permutations: {k + 1}/{n_permutations}")

    elapsed = time.time() - t0
    print(f"[INFO] Null distribution built in {format_time(elapsed)}")
    print(f"[INFO] Null MI: mean={null_mi.mean():.4f}, "
          f"std={null_mi.std():.4f}, "
          f"99.9th pct={np.percentile(null_mi, 99.9):.4f}")
    return null_mi


def save_null_distribution(null_mi, study_name, p_threshold=0.001):
    """
    Save null distribution statistics and a simple text histogram for QC.
    """
    out_path = os.path.join(OUTPUT_DIR, f"null_distribution_{study_name}.txt")
    mi_threshold = np.percentile(null_mi, (1.0 - p_threshold) * 100)

    with open(out_path, "w") as fh:
        fh.write(f"Null MI distribution QC – {study_name}\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"N permutations : {len(null_mi)}\n")
        fh.write(f"Mean           : {null_mi.mean():.6f}\n")
        fh.write(f"Std            : {null_mi.std():.6f}\n")
        fh.write(f"Min            : {null_mi.min():.6f}\n")
        fh.write(f"Max            : {null_mi.max():.6f}\n")
        fh.write(f"95th pct       : {np.percentile(null_mi, 95):.6f}\n")
        fh.write(f"99th pct       : {np.percentile(null_mi, 99):.6f}\n")
        fh.write(f"99.9th pct     : {np.percentile(null_mi, 99.9):.6f}\n")
        fh.write(f"MI threshold   : {mi_threshold:.6f}  (p < {p_threshold})\n")
        fh.write("=" * 60 + "\n\n")

        # ASCII histogram (20 bins)
        counts, edges = np.histogram(null_mi, bins=20)
        fh.write("ASCII histogram:\n")
        max_count = counts.max()
        bar_width = 40
        for i, (cnt, left) in enumerate(zip(counts, edges)):
            right = edges[i + 1]
            bar = "█" * int(cnt / max_count * bar_width)
            fh.write(f"  [{left:.4f}, {right:.4f}) | {bar} {cnt}\n")

    print(f"[SAVED] Null distribution QC: {out_path}")
    return mi_threshold


# ============================================================================
# Edge significance filtering
# ============================================================================

def filter_edges_by_pvalue(mi_matrix, null_mi, p_threshold=0.001):
    """
    Filter gene pair edges to those with empirical p < p_threshold.

    Method
    ------
    1. Sort the null distribution.
    2. For each MI value, compute p-value = #{null >= MI} / N_perm using
       np.searchsorted (O(N² log P) – vectorised over the entire matrix).
    3. Threshold.

    Parameters
    ----------
    mi_matrix  : ndarray (n_genes, n_genes)
    null_mi    : ndarray (n_permutations,)  – empirical null
    p_threshold: float

    Returns
    -------
    adj_significant : ndarray uint8 (n_genes, n_genes)  – binary adjacency
    mi_threshold    : float  – MI value corresponding to the p threshold
    p_matrix_upper  : ndarray float32 (n_genes, n_genes)  – empirical p-values
                       (upper triangle only; lower triangle is zero to save memory)
    """
    n_perm = len(null_mi)
    null_sorted = np.sort(null_mi)
    mi_threshold = np.percentile(null_sorted, (1.0 - p_threshold) * 100)
    print(f"[INFO] MI significance threshold (p < {p_threshold}): {mi_threshold:.4f}")

    # Vectorised empirical p-values for upper triangle only
    rows_ut, cols_ut = np.triu_indices(mi_matrix.shape[0], k=1)
    mi_vals_upper = mi_matrix[rows_ut, cols_ut]

    # searchsorted on sorted null: index of first element > mi_val
    # fraction of null >= mi_val = (n_perm - idx) / n_perm
    insert_idx = np.searchsorted(null_sorted, mi_vals_upper, side="left")
    p_vals_upper = (n_perm - insert_idx) / n_perm

    # Build full p-value matrix (upper triangle only to halve memory)
    n_genes = mi_matrix.shape[0]
    p_matrix_upper = np.zeros((n_genes, n_genes), dtype=np.float32)
    p_matrix_upper[rows_ut, cols_ut] = p_vals_upper.astype(np.float32)

    # Binary adjacency
    sig_mask = p_vals_upper < p_threshold
    adj_significant = np.zeros((n_genes, n_genes), dtype=np.uint8)
    adj_significant[rows_ut[sig_mask], cols_ut[sig_mask]] = 1
    adj_significant[cols_ut[sig_mask], rows_ut[sig_mask]] = 1  # symmetrise

    n_edges = sig_mask.sum()
    print(f"[INFO] Significant edges (p < {p_threshold}): {n_edges:,}")
    return adj_significant, mi_threshold, p_matrix_upper


def apply_bh_fdr(p_matrix_upper, gene_names, mi_matrix, fdr_alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction to the upper-triangle p-values.

    Returns a DataFrame of edges that survive FDR correction, sorted by
    adjusted p-value.
    """
    n_genes = len(gene_names)
    rows_ut, cols_ut = np.triu_indices(n_genes, k=1)
    p_vals = p_matrix_upper[rows_ut, cols_ut]

    # BH correction
    n_tests = len(p_vals)
    rank_order = np.argsort(p_vals)
    sorted_p = p_vals[rank_order]
    ranks = np.arange(1, n_tests + 1)
    bh_threshold = (ranks / n_tests) * fdr_alpha

    # Find the largest rank where p <= BH threshold
    below = sorted_p <= bh_threshold
    if not below.any():
        print(f"[INFO] BH-FDR: no edges survive at alpha={fdr_alpha}")
        return pd.DataFrame(columns=["gene_A", "gene_B", "MI", "p_value", "p_adjusted"])

    cutoff = sorted_p[below].max()
    sig_mask = p_vals <= cutoff

    gene_names_arr = np.array(gene_names)
    df = pd.DataFrame({
        "gene_A": gene_names_arr[rows_ut[sig_mask]],
        "gene_B": gene_names_arr[cols_ut[sig_mask]],
        "MI": mi_matrix[rows_ut[sig_mask], cols_ut[sig_mask]],
        "p_value": p_vals[sig_mask],
    })
    # Re-estimate adjusted p under BH
    r = np.argsort(p_vals[sig_mask])
    m = len(df)
    df["p_adjusted"] = np.minimum(
        1.0, (p_vals[sig_mask] * n_tests) / (np.argsort(np.argsort(p_vals[sig_mask])) + 1)
    )
    df.sort_values("p_adjusted", inplace=True)
    print(f"[INFO] BH-FDR edges surviving alpha={fdr_alpha}: {len(df):,}")
    return df


# ============================================================================
# Per-study edge list
# ============================================================================

def build_study_edgelist(adj_significant, p_matrix_upper, mi_matrix, gene_names):
    """
    Build a tidy DataFrame of significant edges for one study.

    Columns: gene_A, gene_B, MI, p_value
    """
    gene_names_arr = np.array(gene_names)
    rows, cols = np.where(np.triu(adj_significant, k=1) == 1)
    df = pd.DataFrame({
        "gene_A": gene_names_arr[rows],
        "gene_B": gene_names_arr[cols],
        "MI": mi_matrix[rows, cols].astype(np.float32),
        "p_value": p_matrix_upper[rows, cols],
    })
    df.sort_values("p_value", inplace=True)
    return df


# ============================================================================
# Master reference network
# ============================================================================

def build_master_network(study_results, common_gene_names, min_count=3):
    """
    Combine per-study significant adjacency matrices into a master network.

    Parameters
    ----------
    study_results    : list of dict, each with keys 'name' and 'adj_significant'
                       (ndarray uint8, n_common_genes × n_common_genes)
    common_gene_names: list of str
    min_count        : int  – minimum number of studies an edge must appear in

    Returns
    -------
    master_adj       : ndarray uint8 (n × n) – binary master adjacency
    edge_count_matrix: ndarray int16 (n × n) – study count per edge
    """
    n = len(common_gene_names)
    edge_count_matrix = np.zeros((n, n), dtype=np.int16)

    for res in study_results:
        edge_count_matrix += res["adj_significant"].astype(np.int16)

    # Enforce symmetry (floating-point or study-order artefacts)
    edge_count_matrix = np.maximum(edge_count_matrix, edge_count_matrix.T)

    master_adj = (edge_count_matrix >= min_count).astype(np.uint8)
    np.fill_diagonal(master_adj, 0)

    n_master_edges = int(np.triu(master_adj, k=1).sum())
    print(f"[INFO] Master network: {n_master_edges:,} edges "
          f"(in >= {min_count} of {len(study_results)} studies)")
    return master_adj, edge_count_matrix


# ============================================================================
# Louvain clustering  ── COMMENTED OUT: replaced by MCODE (see below)
# ============================================================================
#
# def cluster_louvain(adj_matrix, gene_names):
#     """Louvain clustering on a binary adjacency matrix. Returns (modules, membership)."""
#     print("[INFO] Performing Louvain clustering...")
#     adj_sym = np.maximum(adj_matrix, adj_matrix.T)
#     g = ig.Graph.Adjacency((adj_sym > 0).tolist(), mode="undirected")
#     g.vs["name"] = gene_names
#     clusters = g.community_multilevel()
#     membership = {gene: mid for gene, mid in zip(gene_names, clusters.membership)}
#     modules = {}
#     for gene, mid in zip(gene_names, clusters.membership):
#         modules.setdefault(mid, []).append(gene)
#     print(f"[INFO] Modules: {len(modules)}, sizes: {sorted([len(v) for v in modules.values()], reverse=True)[:10]}")
#     return modules, membership


# ============================================================================
# MCODE  (Bader & Hogue 2003, doi:10.1186/1471-2105-4-2)
# ============================================================================
# Used by Li et al. (Nat Immunol 2014) for de-novo module search in all
# blood transcription module (BTM) networks (Cytoscape plug-in, default params).
#
# Algorithm summary
# -----------------
# Stage 1 – Vertex weighting
#   For every node v:
#     a) Find the highest k-core that contains v  → core_level(v)
#     b) Compute local density: density(N[v]) = edges_within_N[v] / possible_edges
#        where N[v] is the closed neighbourhood (v plus its direct neighbours).
#     c) w(v) = core_level(v) × density(N[v])
#
# Stage 2 – Complex prediction (seed-and-extend)
#   For each seed (nodes sorted by weight, highest first):
#     - Skip if already assigned.
#     - Grow a candidate complex by BFS: add a neighbour u if
#       w(u) ≥ MCODE_SCORE_THRESHOLD × max_weight_in_graph
#     - Record the complex.
#
# Stage 3 – Post-processing (matching Li et al. defaults)
#   - Keep modules with >= MCODE_MIN_SIZE nodes.
#   - Keep modules with degree density >= MCODE_MIN_DENSITY (Li et al.: 0.3).
#   - Nodes can appear in multiple modules (overlapping output like MCODE).
#
# A membership dict is also returned for downstream compatibility with
# save_master_results().  When a gene appears in multiple modules it is
# assigned to the largest one (by size).

# ---- MCODE parameters (matching Li et al. / Cytoscape defaults) ------------
MCODE_SCORE_THRESHOLD = 0.2   # fraction of max node weight to include in complex
MCODE_MIN_SIZE        = 3     # minimum number of genes per module
MCODE_MIN_DENSITY     = 0.3   # minimum degree density (Li et al.: >0.3)
# ---------------------------------------------------------------------------


def _k_core_levels(adj_sym):
    """
    Compute the k-core decomposition of a graph represented by a symmetric
    binary adjacency matrix.  Returns an integer array `core[v]` = the
    maximum k such that v belongs to the k-core.

    Uses the iterative peeling algorithm (O(V + E)).
    """
    n = adj_sym.shape[0]
    degree = adj_sym.sum(axis=1).astype(int)     # current degree
    core   = np.zeros(n, dtype=int)
    removed = np.zeros(n, dtype=bool)

    for k in range(1, n):
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if not removed[v] and degree[v] < k:
                    removed[v] = True
                    # Update degrees of neighbours
                    for u in np.where(adj_sym[v] > 0)[0]:
                        if not removed[u]:
                            degree[u] -= 1
                    changed = True
        if removed.all():
            break
        # Assign core level k to all still-active nodes
        core[~removed] = k

    return core


def _k_core_levels_fast(adj_sym):
    """
    Fast k-core decomposition using igraph (O(V + E)).
    Preferred over the pure-numpy version for large graphs.
    """
    g = ig.Graph.Adjacency((adj_sym > 0).tolist(), mode="undirected")
    return np.array(g.coreness(), dtype=int)


def _local_density(v, neighbours_v, adj_sym):
    """
    Fraction of present edges within the closed neighbourhood of v
    (v plus its direct neighbours) over all possible edges.
    """
    members = [v] + list(neighbours_v)
    n = len(members)
    if n < 2:
        return 0.0
    # Count edges among members (upper triangle only)
    sub = adj_sym[np.ix_(members, members)]
    present = int(np.triu(sub, k=1).sum())
    possible = n * (n - 1) // 2
    return present / possible


def mcode(adj_matrix, gene_names,
          score_threshold=MCODE_SCORE_THRESHOLD,
          min_size=MCODE_MIN_SIZE,
          min_density=MCODE_MIN_DENSITY):
    """
    MCODE – Molecular Complex Detection (Bader & Hogue 2003).

    Parameters
    ----------
    adj_matrix     : ndarray (n, n) binary symmetric adjacency matrix
    gene_names     : list of str  – must have length n
    score_threshold: float  – fraction of max node weight; neighbours with
                              weight < threshold × max_weight are excluded
    min_size       : int    – discard modules with fewer genes than this
    min_density    : float  – discard modules with degree density < this
                              (Li et al. post-processing threshold: 0.3)

    Returns
    -------
    modules    : dict  {module_id (int): [gene_name, ...]}
    membership : dict  {gene_name: module_id}  – genes in multiple modules
                       are assigned to the largest module they appear in
    """
    print("[INFO] Running MCODE module detection...")
    t0 = time.time()

    adj_sym = np.maximum(adj_matrix, adj_matrix.T).astype(np.uint8)
    n = adj_sym.shape[0]
    gene_arr = np.array(gene_names)

    # ----------------------------------------------------------------
    # Stage 1: Vertex weighting
    # ----------------------------------------------------------------
    print("[INFO] MCODE stage 1: vertex weighting (k-core + local density)...")
    core = _k_core_levels_fast(adj_sym)

    # Precompute neighbour lists
    neighbours = [list(np.where(adj_sym[v] > 0)[0]) for v in range(n)]

    weights = np.zeros(n)
    for v in range(n):
        dens = _local_density(v, neighbours[v], adj_sym)
        weights[v] = core[v] * dens

    max_weight = weights.max()
    if max_weight == 0:
        print("[WARN] All node weights are zero – graph may be empty.")
        return {}, {}

    print(f"[INFO] Node weight range: [{weights.min():.4f}, {max_weight:.4f}]")

    # ----------------------------------------------------------------
    # Stage 2: Seed-and-extend complex prediction
    # ----------------------------------------------------------------
    print("[INFO] MCODE stage 2: seed-and-extend...")
    wt_threshold = score_threshold * max_weight
    seed_order   = np.argsort(-weights)   # highest weight first
    visited      = np.zeros(n, dtype=bool)
    raw_modules  = []

    for seed in seed_order:
        if visited[seed]:
            continue
        # BFS/DFS expansion from seed
        module_nodes = set()
        queue = [seed]
        while queue:
            v = queue.pop()
            if v in module_nodes:
                continue
            if weights[v] >= wt_threshold:
                module_nodes.add(v)
                for u in neighbours[v]:
                    if u not in module_nodes and weights[u] >= wt_threshold:
                        queue.append(u)
        if module_nodes:
            raw_modules.append(sorted(module_nodes))
            for v in module_nodes:
                visited[v] = True

    print(f"[INFO] MCODE stage 2: {len(raw_modules)} raw complexes found")

    # ----------------------------------------------------------------
    # Stage 3: Post-processing  (Li et al. thresholds)
    # ----------------------------------------------------------------
    print(f"[INFO] MCODE stage 3: filtering (min_size={min_size}, "
          f"min_density={min_density})...")
    modules = {}
    mid = 0
    for node_list in raw_modules:
        if len(node_list) < min_size:
            continue
        # Compute degree density within the module
        sub = adj_sym[np.ix_(node_list, node_list)]
        m_edges = int(np.triu(sub, k=1).sum())
        m_nodes = len(node_list)
        m_possible = m_nodes * (m_nodes - 1) // 2
        density = m_edges / m_possible if m_possible > 0 else 0.0
        if density < min_density:
            continue
        modules[mid] = [gene_names[i] for i in node_list]
        mid += 1

    elapsed = time.time() - t0
    print(f"[INFO] MCODE: {len(modules)} modules after post-processing "
          f"(in {format_time(elapsed)})")
    if modules:
        sizes = sorted([len(v) for v in modules.values()], reverse=True)
        print(f"[INFO] Module sizes (top 10): {sizes[:10]}")

    # Build membership: assign each gene to the largest module it appears in
    # (genes not in any module are omitted)
    gene_to_modules = {}   # gene -> list of (module_size, module_id)
    for mid_id, genes in modules.items():
        for g in genes:
            gene_to_modules.setdefault(g, []).append((len(genes), mid_id))

    membership = {
        g: max(entries, key=lambda x: x[0])[1]
        for g, entries in gene_to_modules.items()
    }

    return modules, membership


# ============================================================================
# Saving results
# ============================================================================

def save_study_results(study_name, adj, edgelist_df, gene_names,
                       mi_matrix, bh_df=None):
    """Save per-study significant edges and network files."""
    print(f"\n[INFO] Saving results for study: {study_name}")

    # 1. Edge list with MI and p-value
    edges_file = os.path.join(OUTPUT_DIR, f"edges_pval_{study_name}.tsv")
    edgelist_df.to_csv(edges_file, sep="\t", index=False)
    print(f"[SAVED] {edges_file}")

    # 2. Binary adjacency (Matrix Market)
    mtx_file = os.path.join(OUTPUT_DIR, f"adj_pval_{study_name}.mtx")
    mmwrite(mtx_file, csr_matrix(adj))
    print(f"[SAVED] {mtx_file}")

    # 3. GraphML for Cytoscape
    adj_sym = np.maximum(adj, adj.T)
    g = ig.Graph.Adjacency((adj_sym > 0).tolist(), mode="undirected")
    g.vs["name"] = gene_names
    graphml_file = os.path.join(OUTPUT_DIR, f"network_pval_{study_name}.graphml")
    g.write_graphml(graphml_file)
    print(f"[SAVED] {graphml_file}")

    # 4. BH-FDR edges (optional)
    if bh_df is not None and len(bh_df) > 0:
        bh_file = os.path.join(OUTPUT_DIR, f"edges_bh_fdr_{study_name}.tsv")
        bh_df.to_csv(bh_file, sep="\t", index=False)
        print(f"[SAVED] {bh_file}")


def save_master_results(master_adj, edge_count_matrix, gene_names,
                        modules, membership, min_count, n_studies):
    """Save master reference network and associated files."""
    print("\n[INFO] Saving master reference network...")
    gene_names_arr = np.array(gene_names)
    rows, cols = np.where(np.triu(master_adj, k=1) == 1)

    # 1. Master edge list with study count
    master_df = pd.DataFrame({
        "gene_A": gene_names_arr[rows],
        "gene_B": gene_names_arr[cols],
        "n_studies": edge_count_matrix[rows, cols],
    })
    master_df.sort_values("n_studies", ascending=False, inplace=True)
    master_edges_file = os.path.join(OUTPUT_DIR, "master_network_edgelist.tsv")
    master_df.to_csv(master_edges_file, sep="\t", index=False)
    print(f"[SAVED] {master_edges_file}")

    # 2. Study count matrix (sparse)
    count_sparse_file = os.path.join(OUTPUT_DIR, "master_edge_study_counts.mtx")
    upper_count = np.triu(edge_count_matrix)
    mmwrite(count_sparse_file, csr_matrix(upper_count))
    print(f"[SAVED] {count_sparse_file}")

    # 3. Binary adjacency Matrix Market
    adj_mtx_file = os.path.join(OUTPUT_DIR, "master_network_adjacency.mtx")
    mmwrite(adj_mtx_file, csr_matrix(master_adj))
    print(f"[SAVED] {adj_mtx_file}")

    # 4. GraphML for Cytoscape
    master_sym = np.maximum(master_adj, master_adj.T)
    g = ig.Graph.Adjacency((master_sym > 0).tolist(), mode="undirected")
    g.vs["name"] = gene_names
    graphml_file = os.path.join(OUTPUT_DIR, "master_network.graphml")
    g.write_graphml(graphml_file)
    print(f"[SAVED] {graphml_file}")

    # 5. BTM module membership
    btm_df = pd.DataFrame([
        {"Gene": gene, "Module": f"M{mid}"}
        for mid, genes in modules.items()
        for gene in genes
    ])
    btm_file = os.path.join(OUTPUT_DIR, "master_BTM_modules.tsv")
    btm_df.to_csv(btm_file, sep="\t", index=False)
    print(f"[SAVED] {btm_file}")

    # 6. Node module membership
    node_file = os.path.join(OUTPUT_DIR, "master_node_modules.tsv")
    pd.DataFrame(
        [{"gene": g, "module": m} for g, m in membership.items()]
    ).to_csv(node_file, sep="\t", index=False)
    print(f"[SAVED] {node_file}")

    # 7. Submodule GraphML files (one per MCODE module)
    print("[INFO] Saving per-module subgraph files...")
    saved_submodules = 0
    for mid, mod_genes in modules.items():
        if len(mod_genes) < 3:
            continue  # skip singletons / pairs
        idx = [gene_names.index(gn) for gn in mod_genes]
        sub_adj = master_adj[np.ix_(idx, idx)]
        sub_adj_sym = np.maximum(sub_adj, sub_adj.T)
        sg = ig.Graph.Adjacency((sub_adj_sym > 0).tolist(), mode="undirected")
        sg.vs["name"] = mod_genes
        sub_file = os.path.join(OUTPUT_DIR, f"master_submodule_M{mid}.graphml")
        sg.write_graphml(sub_file)
        saved_submodules += 1
    print(f"[INFO] Saved {saved_submodules} submodule files")


def save_report(timings, info, report_path):
    """Write a comprehensive text report of the run."""
    with open(report_path, "w") as fh:
        fh.write("=" * 80 + "\n")
        fh.write("GENE NETWORK INFERENCE – PERMUTATION MI SIGNIFICANCE REPORT\n")
        fh.write("=" * 80 + "\n\n")
        fh.write(f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Script      : {os.path.basename(__file__)}\n")
        fh.write(f"Output dir  : {OUTPUT_DIR}\n\n")

        fh.write("CONFIGURATION\n" + "-" * 80 + "\n")
        for k in ("N_BINS", "DISC_STRATEGY", "N_PERMUTATIONS", "P_VALUE_THRESHOLD",
                  "MIN_STUDY_COUNT", "APPLY_BH_FDR", "BH_FDR_ALPHA", "N_JOBS",
                  "MIN_SAMPLES_PER_STUDY",
                  "MCODE_SCORE_THRESHOLD", "MCODE_MIN_SIZE", "MCODE_MIN_DENSITY"):
            fh.write(f"  {k:30s}: {globals()[k]}\n")
        fh.write(f"  {'COUNTS_PATH':30s}: {COUNTS_PATH}\n")
        fh.write(f"  {'METADATA_PATH':30s}: {METADATA_PATH}\n\n")

        fh.write("RESULTS\n" + "-" * 80 + "\n")
        for k, v in info.items():
            fh.write(f"  {k:50s}: {v}\n")
        fh.write("\n")

        fh.write("TIMING BREAKDOWN\n" + "-" * 80 + "\n")
        total = sum(timings.values())
        for step, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            pct = t / total * 100 if total > 0 else 0
            fh.write(f"  {step:55s}: {format_time(t):>20s} ({pct:5.1f}%)\n")
        fh.write("-" * 80 + "\n")
        fh.write(f"  {'TOTAL':55s}: {format_time(total):>20s}\n\n")
        fh.write("=" * 80 + "\n")

    print(f"\n[INFO] Report saved: {report_path}")


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    global LOG_FILE, REPORT_FILE

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = os.path.join(OUTPUT_DIR, f"network_pval_{ts}.log")
    REPORT_FILE = os.path.join(OUTPUT_DIR, f"analysis_report_pval_{ts}.txt")

    sys.stdout = TeeLogger(LOG_FILE)
    timings = {}
    info = {}

    # ------------------------------------------------------------------
    # Step 0: Load shared data and auto-discover studies from BioProject
    # ------------------------------------------------------------------
    with Timer("Loading expression matrix & metadata", timings):
        expr_full = load_expression(COUNTS_PATH)
        metadata  = load_metadata(METADATA_PATH)
        studies   = discover_studies(expr_full, metadata,
                                     min_samples=MIN_SAMPLES_PER_STUDY)

    if not studies:
        print("[ERROR] No studies discovered. Check COUNTS_PATH, METADATA_PATH "
              "and that the 'Run' column matches expression matrix column names.")
        sys.exit(1)

    print("=" * 80)
    print("GENE NETWORK INFERENCE: PERMUTATION-BASED MI SIGNIFICANCE")
    print("=" * 80)
    print(f"Start time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Studies found    : {len(studies)}")
    for s in studies:
        print(f"  {s['name']:30s}  {s['expr'].shape[1]} samples")
    print(f"N permutations   : {N_PERMUTATIONS}")
    print(f"p-value threshold: {P_VALUE_THRESHOLD}")
    print(f"Min study count  : {MIN_STUDY_COUNT}")
    print(f"Log file         : {LOG_FILE}")
    print("=" * 80)

    info["Studies discovered"] = len(studies)
    info["Expression matrix"] = COUNTS_PATH
    info["Metadata"] = METADATA_PATH

    # ------------------------------------------------------------------
    # Step 1: Process each study independently
    # ------------------------------------------------------------------
    import gc
    study_results = []         # Accumulate per-study adj matrices
    common_gene_names = None   # Will be the intersection across studies

    for study in studies:
        study_name = study["name"]
        expr_data  = study["expr"]          # pre-sliced genes × study_samples

        print(f"\n{'=' * 80}")
        print(f"PROCESSING STUDY: {study_name}  ({expr_data.shape[1]} samples)")
        print("=" * 80)

        gene_names = expr_data.index.tolist()
        n_genes    = len(gene_names)
        n_samples  = expr_data.shape[1]
        info[f"{study_name}: Genes"]   = n_genes
        info[f"{study_name}: Samples"] = n_samples

        # Compute (or load cached) MI matrix
        with Timer(f"{study_name}: MI Matrix", timings):
            mi_matrix, expr_discrete = compute_mi_matrix(
                expr_data,
                n_bins=N_BINS,
                strategy=DISC_STRATEGY,
                n_jobs=N_JOBS,
                cache_path=study.get("mi_cache"),
            )
            info[f"{study_name}: MI range"] = (
                f"{mi_matrix[mi_matrix > 0].min():.4f} – {mi_matrix.max():.4f}"
            )

        # Build null distribution
        with Timer(f"{study_name}: Null Distribution ({N_PERMUTATIONS} perms)", timings):
            null_mi = build_null_distribution(
                expr_discrete, N_BINS,
                n_permutations=N_PERMUTATIONS,
                seed=PERM_SEED,
            )
            mi_threshold = save_null_distribution(null_mi, study_name, P_VALUE_THRESHOLD)
            info[f"{study_name}: MI threshold (p<{P_VALUE_THRESHOLD})"] = f"{mi_threshold:.4f}"

        # Filter edges by p-value
        with Timer(f"{study_name}: Edge Significance Filtering", timings):
            adj_sig, mi_thr, p_matrix = filter_edges_by_pvalue(
                mi_matrix, null_mi, p_threshold=P_VALUE_THRESHOLD
            )
            n_edges = int(np.triu(adj_sig, k=1).sum())
            info[f"{study_name}: Significant edges (p<{P_VALUE_THRESHOLD})"] = f"{n_edges:,}"

        # Build edge list
        edgelist_df = build_study_edgelist(adj_sig, p_matrix, mi_matrix, gene_names)

        # Optional BH-FDR correction
        bh_df = None
        if APPLY_BH_FDR:
            with Timer(f"{study_name}: BH FDR Correction", timings):
                bh_df = apply_bh_fdr(p_matrix, gene_names, mi_matrix, fdr_alpha=BH_FDR_ALPHA)
                info[f"{study_name}: BH-FDR edges (alpha={BH_FDR_ALPHA})"] = f"{len(bh_df):,}"

        # Save per-study files
        with Timer(f"{study_name}: Saving Results", timings):
            save_study_results(
                study_name, adj_sig, edgelist_df, gene_names,
                mi_matrix, bh_df=bh_df,
            )

        # Track gene names for intersection and accumulate adj for master network.
        # All studies share the same full gene panel (drawn from the same logCPM
        # matrix), so the intersection path below is a safety net for edge cases
        # (e.g. after variance-filtering within a study).
        if common_gene_names is None:
            common_gene_names = gene_names
            study_results.append({"name": study_name, "adj_significant": adj_sig})
        else:
            common_set = set(common_gene_names) & set(gene_names)
            if len(common_set) < len(common_gene_names):
                print(f"[WARN] Gene universe intersection: "
                      f"{len(common_gene_names)} → {len(common_set)} genes")
                # Reindex all previously stored adjacency matrices
                old_idx = [common_gene_names.index(g) for g in common_gene_names
                           if g in common_set]
                common_gene_names = [g for g in common_gene_names if g in common_set]
                for prev in study_results:
                    prev["adj_significant"] = prev["adj_significant"][
                        np.ix_(old_idx, old_idx)]
                # Reindex current study adj to common genes
                idx_curr = [gene_names.index(g) for g in common_gene_names]
                adj_sig = adj_sig[np.ix_(idx_curr, idx_curr)]
            study_results.append({"name": study_name, "adj_significant": adj_sig})

        # Release large arrays to free memory before next study
        del mi_matrix, expr_discrete, p_matrix
        gc.collect()

    # ------------------------------------------------------------------
    # Step 2: Build master reference network
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("BUILDING MASTER REFERENCE NETWORK")
    print("=" * 80)

    if len(study_results) < MIN_STUDY_COUNT:
        print(f"[WARN] Only {len(study_results)} studies processed, but MIN_STUDY_COUNT={MIN_STUDY_COUNT}.")
        print(f"[WARN] Setting MIN_STUDY_COUNT to 1 for this run.")
        effective_min = 1
    else:
        effective_min = MIN_STUDY_COUNT

    with Timer("Master Network Construction", timings):
        master_adj, edge_count_matrix = build_master_network(
            study_results, common_gene_names, min_count=effective_min
        )
        n_master_edges = int(np.triu(master_adj, k=1).sum())
        info["Master network: genes"] = len(common_gene_names)
        info["Master network: edges"] = f"{n_master_edges:,}"

    # MCODE module detection on master network
    with Timer("Master Network: MCODE Module Detection", timings):
        modules, membership = mcode(master_adj, common_gene_names)
        info["Master network: MCODE modules"] = len(modules)

    # Save master network
    with Timer("Master Network: Saving", timings):
        save_master_results(
            master_adj, edge_count_matrix, common_gene_names,
            modules, membership,
            min_count=effective_min,
            n_studies=len(study_results),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = sum(timings.values())

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"End time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {format_time(total_time)}")

    print("\nTIMING SUMMARY")
    print("-" * 80)
    for step, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = t / total_time * 100 if total_time > 0 else 0
        print(f"  {step:55s}: {format_time(t):>20s} ({pct:5.1f}%)")
    print("-" * 80)
    print(f"  {'TOTAL':55s}: {format_time(total_time):>20s}")

    print(f"\nOutputs in    : {OUTPUT_DIR}")
    print(f"Log file      : {LOG_FILE}")
    print(f"Report file   : {REPORT_FILE}")

    save_report(timings, info, REPORT_FILE)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

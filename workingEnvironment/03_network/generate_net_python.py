#!/usr/bin/env python3
"""
Gene Network Inference: MI+CLR vs GRNBoost2 Comparison
Equivalent to generate_net.r but with parallel processing and GRNBoost2 comparison
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import KBinsDiscretizer
from scipy.sparse import csr_matrix, save_npz
from scipy.io import mmwrite
from joblib import Parallel, delayed
import igraph as ig
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = "/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment"
OUTPUT_DIR = os.path.join(BASE_DIR, "03_network/NETS_MI_CLR_GRNBoost2")
COUNTS_PATH = os.path.join(BASE_DIR, "02_counts/logCPM_matrix_filtered_samples.csv")
METADATA_PATH = os.path.join(BASE_DIR, "02_counts/metadata_with_sample_annotations.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Network parameters
N_BINS = 5  # Number of bins for discretization (matching R code)
DISC_STRATEGY = 'quantile'  # 'quantile' = 'equalfreq' in R
THRESHOLD_PERCENTILE = 95  # Top 5% of edges
N_JOBS = -1  # Use all available CPU cores (-1 = all cores)

# MI matrix output path
MI_MATRIX_PATH = os.path.join(OUTPUT_DIR, "mi_matrix_cache.npy")

# Logging configuration
# NOTE: LOG_FILE and REPORT_FILE are intentionally NOT set here at module scope.
# Setting them here would cause every dask worker subprocess (which imports this
# module) to call datetime.now() and create different timestamped paths.
# They are assigned inside main() instead.
LOG_FILE = None
REPORT_FILE = None


# ============================================================================
# Logging Setup
# ============================================================================

class TeeLogger:
    """Logger that writes to both file and console"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', buffering=1)  # Line buffered
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()


class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name, report_dict):
        self.name = name
        self.report_dict = report_dict
        
    def __enter__(self):
        self.start = time.time()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] Starting: {self.name}")
        print("-" * 80)
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("-" * 80)
        print(f"[{timestamp}] Completed: {self.name}")
        print(f"[TIMING] Duration: {format_time(self.elapsed)}")
        self.report_dict[self.name] = self.elapsed


def format_time(seconds):
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.2f} minutes ({seconds:.1f}s)"
    else:
        hours = seconds / 3600
        mins = (seconds % 3600) / 60
        return f"{hours:.2f} hours ({mins:.1f}m)"


def save_analysis_report(timings, results_info, report_file):
    """Save comprehensive analysis report"""
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENE NETWORK ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Header info
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script: {os.path.basename(__file__)}\n")
        f.write(f"Working Directory: {OUTPUT_DIR}\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Input Data: {COUNTS_PATH}\n")
        f.write(f"Number of Bins: {N_BINS}\n")
        f.write(f"Discretization Strategy: {DISC_STRATEGY}\n")
        f.write(f"Threshold Percentile: {THRESHOLD_PERCENTILE}%\n")
        f.write(f"CPU Cores: {N_JOBS if N_JOBS > 0 else 'All available'}\n\n")
        
        # Results summary
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 80 + "\n")
        for key, value in results_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Timing breakdown
        f.write("TIMING BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        total_time = sum(timings.values())
        
        for step, elapsed in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            f.write(f"{step:50s}: {format_time(elapsed):>20s} ({percentage:5.1f}%)\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL TIME':50s}: {format_time(total_time):>20s}\n\n")
        
        # GPU information if used
        if 'GPU_INFO' in results_info:
            f.write("GPU INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"{results_info['GPU_INFO']}\n\n")
        
        # Output files
        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Log file: {LOG_FILE}\n")
        f.write(f"Report file: {report_file}\n")
        for file_key in [k for k in results_info.keys() if k.startswith('OUTPUT_')]:
            f.write(f"{results_info[file_key]}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\n[INFO] Analysis report saved to: {report_file}")


# ============================================================================
# Load Data
# ============================================================================

def load_data():
    """Load gene expression data and metadata"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load logCPM counts (genes x samples)
    print(f"[INFO] Loading expression data from: {COUNTS_PATH}")
    expr_data = pd.read_csv(COUNTS_PATH, sep='\t', index_col=0)
    print(f"[INFO] Expression matrix shape: {expr_data.shape} (genes x samples)")
    
    # Load metadata
    print(f"[INFO] Loading metadata from: {METADATA_PATH}")
    metadata = pd.read_csv(METADATA_PATH, sep="\t")
    print(f"[INFO] Metadata shape: {metadata.shape}")
    
    return expr_data, metadata


# ============================================================================
# Method 1: Mutual Information + CLR (matching R code)
# ============================================================================

def discretize_expression(expr_data, n_bins=5, strategy='quantile'):
    """
    Discretize gene expression data using equal frequency binning
    
    Parameters:
    -----------
    expr_data : pd.DataFrame (genes x samples)
    n_bins : int
        Number of bins (default: 5, matching R code with nbins=5)
    strategy : str
        'quantile' for equal frequency (matching R disc="equalfreq")
        'uniform' for equal width
    
    Returns:
    --------
    expr_discrete : np.ndarray (genes x samples)
        Discretized expression values
    """
    print(f"[INFO] Discretizing expression data: {n_bins} bins, strategy={strategy}")
    
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode='ordinal',
        strategy=strategy
    )
    
    # Discretize each gene separately (transpose for sklearn)
    expr_discrete = discretizer.fit_transform(expr_data.T).T
    
    return expr_discrete.astype(int)


def compute_mi_histogram(x, y, n_bins):
    """
    Fast MI computation using 2D histogram (50-100x faster than mutual_info_classif)
    
    For discrete data, this is exact and much faster than sklearn's implementation.
    MI(X,Y) = sum P(x,y) * log(P(x,y) / (P(x)*P(y)))
    
    Parameters:
    -----------
    x, y : np.ndarray
        Discretized gene expression vectors
    n_bins : int
        Number of bins used for discretization
    
    Returns:
    --------
    mi : float
        Mutual information score
    """
    # Compute 2D histogram (joint distribution)
    # For discrete data with values 0 to n_bins-1, specify explicit range
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins, range=[[0, n_bins], [0, n_bins]])
    
    # Normalize to get probabilities
    pxy = hist_2d / float(np.sum(hist_2d))
    
    # Marginal probabilities
    px = np.sum(pxy, axis=1)  # sum over y
    py = np.sum(pxy, axis=0)  # sum over x
    
    # Compute MI (avoid log(0) with masking)
    px_py = px[:, None] * py[None, :]
    
    # Only compute for non-zero entries
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    return mi


def compute_mi_row(i, expr_discrete, n_bins):
    """
    Compute mutual information for one gene against all others using fast histogram method
    
    Parameters:
    -----------
    i : int
        Index of target gene
    expr_discrete : np.ndarray (genes x samples)
        Discretized expression matrix
    n_bins : int
        Number of bins used for discretization
    
    Returns:
    --------
    mi_scores : np.ndarray
        MI scores for gene i with all other genes
    """
    n_genes = expr_discrete.shape[0]
    target = expr_discrete[i, :]
    mi_scores = np.zeros(n_genes)
    
    for j in range(n_genes):
        if i != j:
            predictor = expr_discrete[j, :]
            mi_scores[j] = compute_mi_histogram(target, predictor, n_bins)
        else:
            mi_scores[j] = 0
    
    if (i + 1) % 500 == 0:
        print(f"[INFO] Processed {i+1}/{n_genes} genes for MI calculation")
    
    return mi_scores


def compute_mi_matrix_parallel(expr_data, n_bins=5, strategy='quantile', n_jobs=-1, save_path=None):
    """
    Compute mutual information matrix with parallel processing and automatic saving
    
    Parameters:
    -----------
    expr_data : pd.DataFrame (genes x samples)
    n_bins : int
    strategy : str
    n_jobs : int
        Number of parallel jobs (-1 = all cores)
    save_path : str, optional
        Path to save MI matrix immediately after computation
    
    Returns:
    --------
    mi_matrix : np.ndarray (genes x genes)
        Mutual information matrix
    """
    print("\n" + "-"*80)
    print("METHOD 1: MUTUAL INFORMATION + CLR (Fast histogram-based)")
    print("-"*80)
    
    start_time = time.time()
    
    # Step 1: Discretize
    print(f"[INFO] Discretizing {expr_data.shape[0]} genes with {n_bins} bins ({strategy})...")
    expr_discrete = discretize_expression(expr_data.values, n_bins, strategy)
    n_genes = expr_data.shape[0]
    
    print(f"[INFO] Computing MI matrix for {n_genes} genes using {n_jobs} cores...")
    print(f"[INFO] Using fast histogram-based MI (50-100x faster than sklearn)...")
    
    # Step 2: Parallel MI computation with fast histogram method
    mi_matrix = np.array(
        Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(compute_mi_row)(i, expr_discrete, n_bins)
            for i in range(n_genes)
        )
    )
    
    # Make symmetric
    mi_matrix = (mi_matrix + mi_matrix.T) / 2.0
    
    elapsed = time.time() - start_time
    print(f"[INFO] MI matrix computed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"[INFO] MI matrix shape: {mi_matrix.shape}")
    print(f"[INFO] MI range: [{mi_matrix[mi_matrix > 0].min():.4f}, {mi_matrix.max():.4f}]")
    
    # Save immediately if path provided
    if save_path:
        print(f"[INFO] Saving MI matrix to: {save_path}")
        np.save(save_path, mi_matrix.astype(np.float32))
        file_size_gb = os.path.getsize(save_path) / 1e9
        print(f"[INFO] ✓ MI matrix saved ({file_size_gb:.2f} GB, float32)")
    
    return mi_matrix


# def clr_transform(mi_matrix):
#     """
#     Apply CLR (Context Likelihood of Relatedness) transformation
    
#     This matches the R code: clr_net <- clr(mi)
    
#     CLR compares each gene pair's MI against the background MI for both genes
    
#     Parameters:
#     -----------
#     mi_matrix : np.ndarray (genes x genes)
#         Mutual information matrix
    
#     Returns:
#     --------
#     clr_matrix : np.ndarray (genes x genes)
#         CLR-transformed network
#     """
#     print("[INFO] Applying CLR transformation...")
    
#     n_genes = mi_matrix.shape[0]
#     clr_matrix = np.zeros_like(mi_matrix)
    
#     for i in range(n_genes):
#         for j in range(n_genes):
#             if i != j and mi_matrix[i, j] > 0:
#                 # Get MI values for gene i (excluding zeros and diagonal)
#                 mi_i = mi_matrix[i, :]
#                 mi_i_nonzero = mi_i[mi_i > 0]
                
#                 # Get MI values for gene j
#                 mi_j = mi_matrix[:, j]
#                 mi_j_nonzero = mi_j[mi_j > 0]
                
#                 if len(mi_i_nonzero) > 1 and len(mi_j_nonzero) > 1:
#                     # Z-score for gene i
#                     mean_i = np.mean(mi_i_nonzero)
#                     std_i = np.std(mi_i_nonzero)
#                     z_i = (mi_matrix[i, j] - mean_i) / std_i if std_i > 0 else 0
                    
#                     # Z-score for gene j
#                     mean_j = np.mean(mi_j_nonzero)
#                     std_j = np.std(mi_j_nonzero)
#                     z_j = (mi_matrix[i, j] - mean_j) / std_j if std_j > 0 else 0
                    
#                     # CLR score: combine z-scores
#                     clr_score = np.sqrt(z_i**2 + z_j**2) / np.sqrt(2)
#                     clr_matrix[i, j] = max(0, clr_score)
        
#         if (i + 1) % 100 == 0:
#             print(f"[INFO] CLR transformation: {i+1}/{n_genes} genes processed")
    
#     print(f"[INFO] CLR matrix computed")
#     print(f"[INFO] CLR range: [{clr_matrix[clr_matrix > 0].min():.4f}, {clr_matrix.max():.4f}]")
    
#     return clr_matrix
# def clr_transform_hybrid(mi_matrix, n_jobs=-1):
#     """
#     Hybrid vectorized + parallel CLR transformation
#     Pre-computes statistics, then parallelizes CLR calculation
    
#     This is the fastest approach for very large matrices
#     """
#     print(f"[INFO] Applying CLR transformation (hybrid mode)...")
#     start_time = time.time()
    
#     n_genes = mi_matrix.shape[0]
    
#     # Step 1: Pre-compute all gene statistics (vectorized)
#     print("[INFO] Pre-computing gene statistics...")
#     gene_stats = np.zeros((n_genes, 2))  # [mean, std] for each gene
    
#     for i in range(n_genes):
#         mi_i = mi_matrix[i, :]
#         mi_i_nonzero = mi_i[mi_i > 0]
#         if len(mi_i_nonzero) > 1:
#             gene_stats[i, 0] = np.mean(mi_i_nonzero)
#             gene_stats[i, 1] = np.std(mi_i_nonzero)
    
#     # Step 2: Parallel CLR computation using pre-computed stats
#     def compute_clr_row_fast(i, mi_matrix, gene_stats):
#         """Fast CLR row computation using pre-computed stats"""
#         n = mi_matrix.shape[0]
#         clr_row = np.zeros(n)
        
#         mean_i, std_i = gene_stats[i]
        
#         if std_i > 0:
#             for j in range(n):
#                 if i != j and mi_matrix[i, j] > 0:
#                     mean_j, std_j = gene_stats[j]
                    
#                     if std_j > 0:
#                         z_i = (mi_matrix[i, j] - mean_i) / std_i
#                         z_j = (mi_matrix[i, j] - mean_j) / std_j
#                         clr_row[j] = max(0, np.sqrt(z_i**2 + z_j**2) / np.sqrt(2))
        
#         return clr_row
    
#     print(f"[INFO] Computing CLR scores using {n_jobs} cores...")
#     clr_matrix = np.array(
#         Parallel(n_jobs=n_jobs, verbose=10)(
#             delayed(compute_clr_row_fast)(i, mi_matrix, gene_stats)
#             for i in range(n_genes)
#         )
#     )
    
#     elapsed = time.time() - start_time
#     print(f"[INFO] CLR transformation completed in {elapsed:.2f} seconds")
#     print(f"[INFO] CLR range: [{clr_matrix[clr_matrix > 0].min():.4f}, {clr_matrix.max():.4f}]")
    
#     return clr_matrix


# CPU fallback version of CLR (vectorized)
def clr_transform(mi_matrix):
    """
    CPU-based CLR transformation (vectorized, used as fallback if GPU unavailable)
    
    Parameters:
    -----------
    mi_matrix : np.ndarray
        Mutual information matrix
    
    Returns:
    --------
    clr_matrix : np.ndarray
        CLR-transformed matrix
    """
    print("[INFO] Applying CLR transformation (CPU vectorized)...")
    
    n_genes = mi_matrix.shape[0]
    
    # Mask diagonal
    mask = np.eye(n_genes, dtype=bool)
    mi_masked = np.where(mask, np.nan, mi_matrix)
    
    # Compute row and column statistics (excluding diagonal)
    row_mean = np.nanmean(mi_masked, axis=1)
    row_std = np.nanstd(mi_masked, axis=1)
    col_mean = np.nanmean(mi_masked, axis=0)
    col_std = np.nanstd(mi_masked, axis=0)
    
    # Handle zero standard deviations
    row_std = np.where(row_std == 0, 1.0, row_std)
    col_std = np.where(col_std == 0, 1.0, col_std)
    
    # Compute z-scores using broadcasting
    z_rows = (mi_matrix - row_mean[:, np.newaxis]) / row_std[:, np.newaxis]
    z_cols = (mi_matrix - col_mean[np.newaxis, :]) / col_std[np.newaxis, :]
    
    # CLR formula
    clr_matrix = np.sqrt(z_rows**2 + z_cols**2) / np.sqrt(2)
    
    # Set diagonal to zero and handle NaN/Inf
    clr_matrix[mask] = 0.0
    clr_matrix = np.nan_to_num(clr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"[INFO] CLR range: [{clr_matrix[clr_matrix > 0].min():.4f}, {clr_matrix.max():.4f}]")
    
    return clr_matrix


def threshold_network(clr_matrix, percentile=95):
    """
    Threshold network to keep top edges
    
    Matches R code: thr <- quantile(clr_net[upper.tri(clr_net)], 0.95)
    
    Parameters:
    -----------
    clr_matrix : np.ndarray
    percentile : float
        Percentile threshold (default: 95 = top 5%)
    
    Returns:
    --------
    adj_matrix : np.ndarray (0/1 binary adjacency)
    threshold : float
    """
    # Get upper triangle values (excluding diagonal)
    upper_tri_values = clr_matrix[np.triu_indices_from(clr_matrix, k=1)]
    threshold = np.percentile(upper_tri_values, percentile)
    
    print(f"[INFO] Threshold (top {100-percentile}%): {threshold:.4f}")
    
    # Create binary adjacency matrix
    adj_matrix = (clr_matrix >= threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    
    n_edges = np.sum(adj_matrix) // 2  # Divide by 2 for undirected
    print(f"[INFO] Number of edges after thresholding: {n_edges}")
    
    return adj_matrix, threshold


# ============================================================================
# Method 2: GRNBoost2
# ============================================================================

def run_grnboost2(expr_data, n_jobs=-1):
    """
    Run GRNBoost2 for network inference
    
    Parameters:
    -----------
    expr_data : pd.DataFrame (genes x samples)
    n_jobs : int
    
    Returns:
    --------
    network_df : pd.DataFrame
        Edge list with importance scores
    """
    print("\n" + "-"*80)
    print("METHOD 2: GRNBoost2")
    print("-"*80)
    
    try:
        from arboreto.algo import grnboost2
    except ImportError as e:
        print(f"[ERROR] Could not import arboreto: {e}")
        print("[ERROR] If arboreto is installed, a dependency (e.g. dask/distributed) may have a version conflict.")
        print("[ERROR] Try: pip install --upgrade arboreto dask[complete] distributed")
        print("[INFO] Skipping GRNBoost2 analysis")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error importing arboreto: {type(e).__name__}: {e}")
        print("[INFO] Skipping GRNBoost2 analysis")
        return None
    
    start_time = time.time()
    
    # Cap workers: 128 workers × ~160 MB data each ≈ 20 GB RAM just for data copies.
    # 32 workers is a practical upper bound that balances parallelism and memory.
    max_workers = 32
    n_workers = min(os.cpu_count() if n_jobs == -1 else max(1, n_jobs), max_workers)
    print(f"[INFO] Running GRNBoost2 with {n_workers} workers (capped at {max_workers} to limit RAM)...")
    print(f"[INFO] This may take several minutes for large datasets...")
    
    # Build a clean worker environment:
    # GRNBoost2 uses dask LocalCluster which spawns new Python sub-processes.
    # Those workers must NOT inherit the module-system PYTHONPATH (which contains
    # a NumPy-1.x-compiled numexpr that crashes with the venv's NumPy 2.x).
    # We patch os.environ BEFORE creating the cluster so all spawned workers
    # inherit the clean environment (dask does not have an 'env' kwarg in 2024.x).
    venv_site = "/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/lib/python3.9/site-packages"
    original_pythonpath = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = venv_site
    os.environ['NUMEXPR_DISABLED'] = '1'  # prevent pandas loading the system numexpr
    print(f"[INFO] Worker PYTHONPATH set to venv only (avoids NumPy 1.x/2.x conflicts)")
    print(f"[INFO] NUMEXPR_DISABLED=1 (prevents system numexpr from crashing workers)")
    
    client = None
    cluster = None
    try:
        from distributed import Client, LocalCluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            silence_logs=False,
        )
        client = Client(cluster)
        print(f"[INFO] Dask dashboard: {client.dashboard_link}")
        
        # GRNBoost2 expects samples x genes (transpose)
        network_df = grnboost2(
            expression_data=expr_data.T,
            gene_names=expr_data.index.tolist(),
            tf_names='all',  # All genes can be regulators
            client_or_address=client,
            seed=42,
            verbose=True
        )
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
        # Restore original PYTHONPATH so nothing else in the pipeline is affected
        if original_pythonpath:
            os.environ['PYTHONPATH'] = original_pythonpath
        else:
            os.environ.pop('PYTHONPATH', None)
        os.environ.pop('NUMEXPR_DISABLED', None)
    
    elapsed = time.time() - start_time
    print(f"[INFO] GRNBoost2 completed in {elapsed:.2f} seconds")
    print(f"[INFO] Network edges: {len(network_df)}")
    print(f"[INFO] Importance range: [{network_df['importance'].min():.4f}, {network_df['importance'].max():.4f}]")
    
    return network_df


def grnboost2_to_adjacency(network_df, gene_names, percentile=95):
    """
    Convert GRNBoost2 edge list to adjacency matrix
    
    Parameters:
    -----------
    network_df : pd.DataFrame
        GRNBoost2 output with columns: TF, target, importance
    gene_names : list
        List of gene names
    percentile : float
    
    Returns:
    --------
    adj_matrix : np.ndarray
    threshold : float
    """
    # Threshold by importance
    threshold = np.percentile(network_df['importance'], percentile)
    network_filtered = network_df[network_df['importance'] >= threshold]
    
    print(f"[INFO] GRNBoost2 threshold (top {100-percentile}%): {threshold:.4f}")
    print(f"[INFO] Edges after thresholding: {len(network_filtered)}")
    
    # Create adjacency matrix
    n_genes = len(gene_names)
    adj_matrix = np.zeros((n_genes, n_genes), dtype=int)
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    
    for _, row in network_filtered.iterrows():
        tf = row['TF']
        target = row['target']
        if tf in gene_to_idx and target in gene_to_idx:
            i = gene_to_idx[tf]
            j = gene_to_idx[target]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Make symmetric for comparison
    
    return adj_matrix, threshold


# ============================================================================
# Network Analysis and Clustering
# ============================================================================

def cluster_louvain(adj_matrix, gene_names):
    """
    Perform Louvain clustering on adjacency matrix
    
    Matches R code: cl <- cluster_louvain(g)
    
    Parameters:
    -----------
    adj_matrix : np.ndarray
    gene_names : list
    
    Returns:
    --------
    modules : dict
        {module_id: [gene_list]}
    membership : dict
        {gene: module_id}
    """
    print("[INFO] Performing Louvain clustering...")

    # Enforce strict symmetry — CLR float operations can introduce tiny asymmetries
    # np.maximum takes element-wise max of adj[i,j] and adj[j,i], guaranteeing symmetry
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

    # Create igraph object
    g = ig.Graph.Adjacency((adj_matrix > 0).tolist(), mode='undirected')
    g.vs['name'] = gene_names
    
    # Louvain clustering
    clusters = g.community_multilevel()
    
    # Extract modules
    membership_dict = {gene: module for gene, module in zip(gene_names, clusters.membership)}
    
    modules = {}
    for i, module_id in enumerate(clusters.membership):
        if module_id not in modules:
            modules[module_id] = []
        modules[module_id].append(gene_names[i])
    
    print(f"[INFO] Number of modules: {len(modules)}")
    print(f"[INFO] Module sizes: {[len(genes) for genes in modules.values()]}")
    
    return modules, membership_dict


# ============================================================================
# Save Results
# ============================================================================

def save_results(mi_matrix, clr_matrix, adj_clr, modules_clr, membership_clr, gene_names, method_suffix):
    """
    Save network results in multiple formats
    
    Parameters:
    -----------
    mi_matrix : np.ndarray or None
        Mutual information matrix (saved if not None)
    clr_matrix : np.ndarray
        Full CLR or importance matrix
    adj_clr : np.ndarray
        Binary adjacency matrix
    modules_clr : dict
    membership_clr : dict
    gene_names : list
    method_suffix : str
        e.g., "mi_clr" or "grnboost2"
    """
    print(f"\n[INFO] Saving {method_suffix} results...")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    # Note: MI matrix already saved during computation
    # (no need to save again here to avoid duplication)

    # Get upper-triangle edge indices (vectorized - avoids O(n²) Python loops)
    gene_names_arr = np.array(gene_names)
    rows, cols = np.where(np.triu(adj_clr, k=1) == 1)
    sources = gene_names_arr[rows]
    targets = gene_names_arr[cols]
    n_edges = len(rows)
    print(f"[INFO] Vectorized edge extraction: {n_edges:,} edges")

    # 1. Save adjacency matrix (Matrix Market format)
    adj_sparse = csr_matrix(adj_clr)
    mtx_file = os.path.join(OUTPUT_DIR, f"CLR_adjacency_matrix_{method_suffix}.mtx")
    mmwrite(mtx_file, adj_sparse)
    print(f"[SAVED] {mtx_file}")

    # 2. Save edgelist (binary)
    edgelist_df = pd.DataFrame({'source': sources, 'target': targets})
    edgelist_file = os.path.join(OUTPUT_DIR, f"CLR_network_edgelist_{method_suffix}.txt")
    edgelist_df.to_csv(edgelist_file, sep="\t", index=False)
    print(f"[SAVED] {edgelist_file}")

    # 3. Save weighted edgelist
    weights = clr_matrix[rows, cols]
    weighted_df = pd.DataFrame({'source': sources, 'target': targets, 'weight': weights})
    weighted_file = os.path.join(OUTPUT_DIR, f"CLR_network_weighted_{method_suffix}.txt")
    weighted_df.to_csv(weighted_file, sep="\t", index=False)
    print(f"[SAVED] {weighted_file}")
    
    # 4. Save GraphML for Cytoscape
    # Enforce symmetry before igraph (same CLR float asymmetry issue as in cluster_louvain)
    adj_clr_sym = np.maximum(adj_clr, adj_clr.T)
    g = ig.Graph.Adjacency((adj_clr_sym > 0).tolist(), mode='undirected')
    g.vs['name'] = gene_names
    graphml_file = os.path.join(OUTPUT_DIR, f"CLR_network_{method_suffix}.graphml")
    g.write_graphml(graphml_file)
    print(f"[SAVED] {graphml_file}")
    
    # 5. Save module membership
    module_df = pd.DataFrame([
        {'gene': gene, 'module': module_id}
        for gene, module_id in membership_clr.items()
    ])
    module_file = os.path.join(OUTPUT_DIR, f"node_modules_{method_suffix}.txt")
    module_df.to_csv(module_file, sep="\t", index=False)
    print(f"[SAVED] {module_file}")
    
    # 6. Save BTM modules (same format as R)
    btm_df = pd.DataFrame([
        {'Gene': gene, 'Module': f'M{module_id}'}
        for module_id, genes in modules_clr.items()
        for gene in genes
    ])
    btm_file = os.path.join(OUTPUT_DIR, f"BTM_modules_{method_suffix}.tsv")
    btm_df.to_csv(btm_file, sep="\t", index=False)
    print(f"[SAVED] {btm_file}")


# ============================================================================
# Network Comparison
# ============================================================================

def compare_networks(adj1, adj2, gene_names, method1="MI+CLR", method2="GRNBoost2"):
    """
    Compare two networks and report overlap statistics
    
    Parameters:
    -----------
    adj1, adj2 : np.ndarray
        Binary adjacency matrices
    gene_names : list
    method1, method2 : str
        Method names for reporting
    """
    print("\n" + "="*80)
    print("NETWORK COMPARISON")
    print("="*80)
    
    # Get edges
    edges1 = set()
    edges2 = set()
    
    for i in range(len(gene_names)):
        for j in range(i+1, len(gene_names)):
            if adj1[i, j] == 1:
                edges1.add((gene_names[i], gene_names[j]))
            if adj2[i, j] == 1:
                edges2.add((gene_names[i], gene_names[j]))
    
    # Calculate overlap
    overlap = edges1 & edges2
    only_method1 = edges1 - edges2
    only_method2 = edges2 - edges1
    
    print(f"\n{method1}:")
    print(f"  Total edges: {len(edges1)}")
    
    print(f"\n{method2}:")
    print(f"  Total edges: {len(edges2)}")
    
    print(f"\nOverlap:")
    print(f"  Shared edges: {len(overlap)}")
    print(f"  Jaccard similarity: {len(overlap) / len(edges1 | edges2):.4f}")
    print(f"  {method1} unique: {len(only_method1)}")
    print(f"  {method2} unique: {len(only_method2)}")
    
    # Save comparison
    comparison_file = os.path.join(OUTPUT_DIR, "network_comparison.txt")
    with open(comparison_file, 'w') as f:
        f.write(f"Network Comparison: {method1} vs {method2}\n")
        f.write("="*80 + "\n\n")
        f.write(f"{method1} edges: {len(edges1)}\n")
        f.write(f"{method2} edges: {len(edges2)}\n")
        f.write(f"Shared edges: {len(overlap)}\n")
        f.write(f"Jaccard similarity: {len(overlap) / len(edges1 | edges2):.4f}\n")
        f.write(f"{method1} unique: {len(only_method1)}\n")
        f.write(f"{method2} unique: {len(only_method2)}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("Sample of shared edges (first 20):\n")
        for edge in list(overlap)[:20]:
            f.write(f"  {edge[0]} -- {edge[1]}\n")
    
    print(f"\n[SAVED] {comparison_file}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main pipeline"""
    
    # Assign log/report file paths here (NOT at module scope) so that dask worker
    # sub-processes that import this module do NOT create spurious log files or
    # execute heavyweight I/O at import time.
    global LOG_FILE, REPORT_FILE
    LOG_FILE = os.path.join(OUTPUT_DIR, f"network_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    REPORT_FILE = os.path.join(OUTPUT_DIR, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Initialize logging and timing
    sys.stdout = TeeLogger(LOG_FILE)
    timings = {}

    print(f"[INFO] Using all available CPU cores for parallel processing")
    print(f"[INFO] Configuration: nbins={N_BINS}, disc_strategy={DISC_STRATEGY}")
    print(f"[INFO] Log file: {LOG_FILE}")
    print(f"[INFO] Report file: {REPORT_FILE}")
    results_info = {}
    
    print("\n" + "="*80)
    print("GENE NETWORK INFERENCE: MI+CLR vs GRNBoost2")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # NOTE: GPU CLR is disabled — the CLR matrix for 32,763 genes requires ~21 GB of VRAM
    # (5 full float32 copies of a 4.3 GB matrix held simultaneously), exceeding the RTX 2080 Ti
    # (11.5 GB). CPU vectorized CLR runs in ~40s and is fast enough. Re-enable below if a
    # larger GPU (≥24 GB VRAM) is available in a future run.
    #
    # try:
    #     from clr_gpu import clr_transform_pytorch, print_gpu_info
    #     gpu_info = print_gpu_info()
    #     if gpu_info and gpu_info['cuda_available']:
    #         results_info['GPU_INFO'] = f"Using GPU: {gpu_info['gpu_devices'][0]['name'] if gpu_info['gpu_devices'] else 'Unknown'}\nGPU Memory: {gpu_info['gpu_devices'][0]['memory_gb']:.1f} GB" if gpu_info['gpu_devices'] else "GPU available"
    #         use_gpu = True
    #     else:
    #         use_gpu = False
    # except ImportError:
    #     print("[INFO] GPU acceleration not available (clr_gpu.py not found)")
    #     use_gpu = False
    
    # Load data
    with Timer("Data Loading", timings):
        expr_data, metadata = load_data()
        gene_names = expr_data.index.tolist()
        results_info['Number of Genes'] = len(gene_names)
        results_info['Number of Samples'] = expr_data.shape[1]
    
    # ========================================================================
    # Method 1: MI + CLR
    # ========================================================================
    
    print("\n" + "="*80)
    print("METHOD 1: MUTUAL INFORMATION + CLR")
    print("="*80)
    
    # Load MI matrix from cache if available, otherwise compute and save it
    with Timer("MI Matrix Computation (Fast histogram-based)", timings):
        if os.path.exists(MI_MATRIX_PATH):
            print(f"[INFO] Cache found — loading MI matrix from: {MI_MATRIX_PATH}")
            mi_matrix = np.load(MI_MATRIX_PATH)
            file_size_gb = os.path.getsize(MI_MATRIX_PATH) / 1e9
            print(f"[INFO] ✓ Loaded MI matrix: shape={mi_matrix.shape}, size={file_size_gb:.2f} GB")
        else:
            print(f"[INFO] No cache found — computing MI matrix...")
            mi_matrix = compute_mi_matrix_parallel(
                expr_data,
                n_bins=N_BINS,
                strategy=DISC_STRATEGY,
                n_jobs=N_JOBS,
                save_path=MI_MATRIX_PATH
            )
        results_info['MI Matrix Size'] = f"{mi_matrix.shape[0]} × {mi_matrix.shape[1]}"
        results_info['MI Range'] = f"{mi_matrix.min():.4f} to {mi_matrix.max():.4f}"
        results_info['MI Matrix Path'] = MI_MATRIX_PATH
    
    # Apply CLR transformation (CPU vectorized - fast enough at ~40s, GPU needs >25 GB VRAM)
    with Timer("CLR Transformation (CPU vectorized)", timings):
        clr_matrix = clr_transform(mi_matrix)
        results_info['CLR Matrix Range'] = f"{clr_matrix.min():.4f} to {clr_matrix.max():.4f}"
    
    # Threshold network
    with Timer("Network Thresholding", timings):
        adj_clr, threshold_clr = threshold_network(clr_matrix, percentile=THRESHOLD_PERCENTILE)
        results_info['CLR Threshold Value'] = f"{threshold_clr:.4f}"
        results_info['CLR Number of Edges'] = adj_clr.nnz if hasattr(adj_clr, 'nnz') else np.sum(adj_clr > 0)
    
    # Cluster network
    with Timer("Louvain Clustering (MI+CLR)", timings):
        modules_clr, membership_clr = cluster_louvain(adj_clr, gene_names)
        results_info['CLR Number of Modules'] = len(modules_clr)
    
    # Save MI+CLR results (mi_matrix=None since already saved during computation)
    with Timer("Saving MI+CLR Results", timings):
        save_results(None, clr_matrix, adj_clr, modules_clr, membership_clr, gene_names, "mi_clr_python")
        results_info['OUTPUT_CLR'] = "  - MI+CLR results saved with prefix: mi_clr_python"
    
    # ========================================================================
    # Method 2: GRNBoost2
    # ========================================================================
    
    print("\n" + "="*80)
    print("METHOD 2: GRNBoost2")
    print("="*80)
    
    with Timer("GRNBoost2 Inference", timings):
        network_grnboost = run_grnboost2(expr_data, n_jobs=N_JOBS)
    
    if network_grnboost is not None:
        # Convert to adjacency matrix
        with Timer("GRNBoost2 to Adjacency Matrix", timings):
            adj_grnboost, threshold_grnboost = grnboost2_to_adjacency(
                network_grnboost, 
                gene_names, 
                percentile=THRESHOLD_PERCENTILE
            )
            results_info['GRNBoost2 Threshold'] = f"{threshold_grnboost:.4f}"
            results_info['GRNBoost2 Number of Edges'] = adj_grnboost.nnz if hasattr(adj_grnboost, 'nnz') else np.sum(adj_grnboost > 0)
        
        # Cluster network
        with Timer("Louvain Clustering (GRNBoost2)", timings):
            modules_grnboost, membership_grnboost = cluster_louvain(adj_grnboost, gene_names)
            results_info['GRNBoost2 Number of Modules'] = len(modules_grnboost)
        
        # Create importance matrix for saving
        importance_matrix = np.zeros((len(gene_names), len(gene_names)))
        gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
        for _, row in network_grnboost.iterrows():
            if row['TF'] in gene_to_idx and row['target'] in gene_to_idx:
                i = gene_to_idx[row['TF']]
                j = gene_to_idx[row['target']]
                importance_matrix[i, j] = row['importance']
                importance_matrix[j, i] = row['importance']
        
        # Save GRNBoost2 results (no MI matrix for this method)
        with Timer("Saving GRNBoost2 Results", timings):
            save_results(
                None,              # mi_matrix: not applicable for GRNBoost2
                importance_matrix, 
                adj_grnboost, 
                modules_grnboost, 
                membership_grnboost, 
                gene_names, 
                "grnboost2_python"
            )
            results_info['OUTPUT_GRNBoost2'] = "  - GRNBoost2 results saved with prefix: grnboost2_python"
        
        # ====================================================================
        # Compare Networks
        # ====================================================================
        
        with Timer("Network Comparison", timings):
            compare_networks(adj_clr, adj_grnboost, gene_names, "MI+CLR", "GRNBoost2")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    # Calculate total time
    total_time = sum(timings.values())
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {format_time(total_time)}")
    print("="*80)
    
    # Print timing summary
    print("\nTIMING SUMMARY:")
    print("-" * 80)
    for step, elapsed in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        percentage = (elapsed / total_time * 100) if total_time > 0 else 0
        print(f"{step:50s}: {format_time(elapsed):>20s} ({percentage:5.1f}%)")
    print("-" * 80)
    print(f"{'TOTAL':50s}: {format_time(total_time):>20s}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nOutput files:")
    print("  - CLR_adjacency_matrix_mi_clr_python.mtx")
    print("  - CLR_network_edgelist_mi_clr_python.txt")
    print("  - CLR_network_weighted_mi_clr_python.txt")
    print("  - CLR_network_mi_clr_python.graphml")
    print("  - BTM_modules_mi_clr_python.tsv")
    
    if network_grnboost is not None:
        print("  - CLR_adjacency_matrix_grnboost2_python.mtx")
        print("  - CLR_network_edgelist_grnboost2_python.txt")
        print("  - CLR_network_weighted_grnboost2_python.txt")
        print("  - CLR_network_grnboost2_python.graphml")
        print("  - BTM_modules_grnboost2_python.tsv")
        print("  - network_comparison.txt")
    
    # Save comprehensive report
    save_analysis_report(timings, results_info, REPORT_FILE)
    
    print("\n" + "="*80)
    print(f"Log file: {LOG_FILE}")
    print(f"Report file: {REPORT_FILE}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

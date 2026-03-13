"""
Fast correlation pre-screening to reduce candidate gene pairs.
==============================================================

With G genes there are G(G-1)/2 unique pairs.  For 32 000 genes that is
~537 million pairs.  Training a MINE network per pair would take weeks,
so we optionally pre-filter using fast linear correlation.

Justification (Section 4c)
---------------------------
Most gene pairs have near-zero linear correlation *and* near-zero MI.
MINE's advantage is for pairs that have moderate Pearson |r| but harbour
stronger nonlinear dependencies.  Filtering at |r| > 0.3 retains these
interesting pairs while eliminating the vast majority of truly independent
ones.

The pre-screen is **optional** (``PrescreenConfig.enabled``).  When disabled
(small gene sets or targeted analyses), all unique pairs are returned.

Implementation
--------------
Row-wise Pearson correlation is computed in parallel via ``joblib``.
For each gene i, |r(i, j)| is computed for all j > i (upper triangle only).
Pairs exceeding the threshold are collected.  If the total exceeds
``max_pairs``, the threshold is dynamically raised to cap the count.
"""

import numpy as np
import time


def _pearson_row(i: int, X: np.ndarray, n_genes: int) -> np.ndarray:
    """
    Compute |Pearson r| between gene i and genes i+1 … n-1.

    Uses vectorised NumPy dot products for speed.

    Parameters
    ----------
    i : int
        Row index of the query gene.
    X : np.ndarray, shape (n_genes, n_samples)
        Expression matrix (Z-scored or raw).
    n_genes : int
        Total number of genes.

    Returns
    -------
    np.ndarray, shape (n_genes - i - 1,)
        Absolute Pearson correlation with each subsequent gene.
    """
    xi = X[i]
    xi_centered = xi - xi.mean()
    xi_norm = np.sqrt(np.dot(xi_centered, xi_centered))
    if xi_norm == 0:
        return np.zeros(n_genes - i - 1, dtype=np.float32)

    # Vectorised: correlate gene i with all genes j > i at once
    Xj = X[i + 1:]
    Xj_centered = Xj - Xj.mean(axis=1, keepdims=True)
    Xj_norms = np.sqrt((Xj_centered ** 2).sum(axis=1))
    # Avoid division by zero
    safe_norms = np.where(Xj_norms == 0, 1.0, Xj_norms)
    r = np.abs(Xj_centered @ xi_centered / (xi_norm * safe_norms))
    r[Xj_norms == 0] = 0.0
    return r.astype(np.float32)


def prescreen_pairs(
    expr_matrix: np.ndarray,
    method: str = "pearson",
    threshold: float = 0.3,
    max_pairs: int = 500_000,
    n_jobs: int = -1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Return gene-index pairs (i, j) with |correlation| > threshold.

    Parameters
    ----------
    expr_matrix : np.ndarray, shape (n_genes, n_samples)
        Expression matrix (typically Z-scored).
    method : str
        ``"pearson"`` or ``"spearman"`` (rank-transform then Pearson).
    threshold : float
        Minimum |r| to keep a pair.
    max_pairs : int
        Hard cap.  If exceeded, the threshold is raised dynamically.
    n_jobs : int
        CPU cores for parallel computation (-1 = all).
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray, shape (n_pairs, 2), dtype int32
        Each row is (gene_i_index, gene_j_index) with i < j.
    """
    n_genes, n_samples = expr_matrix.shape

    if method == "spearman":
        from scipy.stats import rankdata
        expr_matrix = np.array(
            [rankdata(expr_matrix[g]) for g in range(n_genes)]
        )

    if verbose:
        total = n_genes * (n_genes - 1) // 2
        print(f"  Pre-screening {n_genes:,} genes ({method}, |r| > {threshold}, "
              f"{total:,} total pairs)...")

    t0 = time.time()

    # Centre and normalise once for all genes
    X_c = expr_matrix - expr_matrix.mean(axis=1, keepdims=True)
    norms = np.sqrt((X_c ** 2).sum(axis=1))
    safe_norms = np.where(norms == 0, 1.0, norms)
    X_normed = X_c / safe_norms[:, None]  # (G, S) — unit-norm rows

    # Chunked matrix-multiply approach: compute |r| block by block
    CHUNK = 1000
    pair_i_list = []
    pair_j_list = []
    pair_r_list = []

    for i_start in range(0, n_genes - 1, CHUNK):
        i_end = min(i_start + CHUNK, n_genes - 1)
        block = X_normed[i_start:i_end]   # (chunk, S)
        rest = X_normed[i_start + 1:]     # (G - i_start - 1, S)
        R = np.abs(block @ rest.T)        # (chunk, G - i_start - 1)

        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            offset = global_i - i_start
            row = R[local_i, offset:]
            above = np.where(row > threshold)[0]
            if len(above) > 0:
                js = above + global_i + 1
                pair_i_list.append(np.full(len(above), global_i, dtype=np.int32))
                pair_j_list.append(js.astype(np.int32))
                pair_r_list.append(row[above].astype(np.float32))

        if verbose and (i_start // CHUNK) % 10 == 0:
            n_so_far = sum(len(x) for x in pair_i_list)
            pct = i_start / n_genes * 100
            print(f"    chunk {i_start:,}/{n_genes:,} ({pct:.0f}%) — "
                  f"{n_so_far:,} pairs so far")

    # Concatenate
    if pair_i_list:
        all_i = np.concatenate(pair_i_list)
        all_j = np.concatenate(pair_j_list)
        all_r = np.concatenate(pair_r_list)
    else:
        all_i = np.array([], dtype=np.int32)
        all_j = np.array([], dtype=np.int32)
        all_r = np.array([], dtype=np.float32)

    elapsed = time.time() - t0
    n_found = len(all_i)
    if verbose:
        print(f"  Pre-screen: {n_found:,} pairs above |r| > {threshold} "
              f"in {elapsed:.1f}s")

    # Dynamic threshold raise if too many pairs
    if n_found > max_pairs:
        # Use partial sort (O(n)) instead of full sort
        cutoff_idx = n_found - max_pairs
        r_partition = np.argpartition(all_r, cutoff_idx)
        keep = r_partition[cutoff_idx:]
        all_i = all_i[keep]
        all_j = all_j[keep]
        all_r = all_r[keep]
        new_threshold = all_r.min()
        if verbose:
            print(f"  Capped to {len(all_i):,} pairs "
                  f"(raised threshold to |r| > {new_threshold:.3f})")

    pair_indices = np.column_stack([all_i, all_j]) if len(all_i) > 0 else np.empty((0, 2), dtype=np.int32)

    return pair_indices


def all_pairs(n_genes: int) -> np.ndarray:
    """
    Generate all unique (i, j) pairs with i < j.

    Used when pre-screening is disabled.

    Parameters
    ----------
    n_genes : int
        Number of genes.

    Returns
    -------
    np.ndarray, shape (n_genes*(n_genes-1)//2, 2), dtype int32
    """
    rows, cols = np.triu_indices(n_genes, k=1)
    return np.column_stack([rows, cols]).astype(np.int32)

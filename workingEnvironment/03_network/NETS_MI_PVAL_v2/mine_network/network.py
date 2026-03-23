"""
Network construction — edge filtering, master consensus, BH-FDR.
=================================================================

This module handles:
  1. **Edge filtering** — given observed MI and a null distribution,
     retain edges with empirical p < threshold.
  2. **Edge list construction** — tidy DataFrames of significant edges.
  3. **BH-FDR correction** — optional Benjamini–Hochberg adjustment.
  4. **Master network** — multi-study consensus (Section 5 of user design).

Master network logic (Section 5)
---------------------------------
For each gene pair (i, j):
    c_ij = #{studies where A_ij^(s) = 1}

Keep edges with c_ij ≥ min_study_count (default 3).

Optionally weight edges by:
    - ``"count"``   — number of supporting studies (c_ij)
    - ``"mean_mi"`` — mean MI across supporting studies

This gives modules that are:
    - Statistically supported within individual studies
    - Replicated across independent cohorts
    - Structurally dense (after MCODE)
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Edge significance filtering
# ═══════════════════════════════════════════════════════════════════════════════

def filter_edges(
    mi_values: np.ndarray,
    p_values: np.ndarray,
    pair_indices: np.ndarray,
    n_genes: int,
    p_threshold: float = 0.001,
) -> np.ndarray:
    """
    Build a binary adjacency matrix from significant gene pairs.

    Parameters
    ----------
    mi_values : np.ndarray, shape (n_pairs,)
        Observed MI values.
    p_values : np.ndarray, shape (n_pairs,)
        Empirical p-values from permutation test.
    pair_indices : np.ndarray, shape (n_pairs, 2)
        Gene index pairs.
    n_genes : int
        Total number of genes (determines adjacency matrix size).
    p_threshold : float
        Significance threshold (default 0.001).

    Returns
    -------
    np.ndarray, shape (n_genes, n_genes), dtype uint8
        Symmetric binary adjacency matrix.
    """
    sig_mask = p_values < p_threshold
    adj = np.zeros((n_genes, n_genes), dtype=np.uint8)
    sig_pairs = pair_indices[sig_mask]
    if len(sig_pairs) > 0:
        adj[sig_pairs[:, 0], sig_pairs[:, 1]] = 1
        adj[sig_pairs[:, 1], sig_pairs[:, 0]] = 1  # symmetrise

    n_sig = sig_mask.sum()
    print(f"[INFO] Significant edges (p < {p_threshold}): {n_sig:,}")
    return adj


# ═══════════════════════════════════════════════════════════════════════════════
# Edge list construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_edgelist(
    adj: np.ndarray,
    pair_indices: np.ndarray,
    mi_values: np.ndarray,
    p_values: np.ndarray,
    gene_names: list,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of significant edges for one study.

    Columns: ``gene_A``, ``gene_B``, ``MI_MINE``, ``p_value``.

    Parameters
    ----------
    adj : np.ndarray
        Binary adjacency from ``filter_edges``.
    pair_indices : np.ndarray, shape (n_pairs, 2)
    mi_values : np.ndarray, shape (n_pairs,)
    p_values : np.ndarray, shape (n_pairs,)
    gene_names : list[str]

    Returns
    -------
    pd.DataFrame
        Sorted by p-value (ascending).
    """
    gene_arr = np.array(gene_names)
    rows, cols = np.where(np.triu(adj, k=1) == 1)

    # Map back to MI and p-value via pair_indices lookup
    pair_set = {(int(i), int(j)): k for k, (i, j) in enumerate(pair_indices)}
    mi_list, p_list = [], []
    for r, c in zip(rows, cols):
        key = (min(r, c), max(r, c))
        k = pair_set.get(key)
        mi_list.append(float(mi_values[k]) if k is not None else 0.0)
        p_list.append(float(p_values[k]) if k is not None else 1.0)

    df = pd.DataFrame({
        "gene_A": gene_arr[rows],
        "gene_B": gene_arr[cols],
        "MI_MINE": mi_list,
        "p_value": p_list,
    })
    df.sort_values("p_value", inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BH-FDR correction
# ═══════════════════════════════════════════════════════════════════════════════

def apply_bh_fdr(
    pair_indices: np.ndarray,
    mi_values: np.ndarray,
    p_values: np.ndarray,
    gene_names: list,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Benjamini–Hochberg FDR correction on candidate-pair p-values.

    Parameters
    ----------
    pair_indices : np.ndarray, shape (n_pairs, 2)
    mi_values : np.ndarray, shape (n_pairs,)
    p_values : np.ndarray, shape (n_pairs,)
    gene_names : list[str]
    fdr_alpha : float
        FDR level.

    Returns
    -------
    pd.DataFrame
        Edges surviving BH correction, with columns:
        ``gene_A``, ``gene_B``, ``MI_MINE``, ``p_value``, ``p_adjusted``.
    """
    empty = pd.DataFrame(
        columns=["gene_A", "gene_B", "MI_MINE", "p_value", "p_adjusted"]
    )
    n_tests = len(p_values)
    if n_tests == 0:
        return empty

    rank_order = np.argsort(p_values)
    sorted_p = p_values[rank_order]
    ranks = np.arange(1, n_tests + 1)
    bh_threshold = (ranks / n_tests) * fdr_alpha

    below = sorted_p <= bh_threshold
    if not below.any():
        print(f"[INFO] BH-FDR: no edges survive at alpha={fdr_alpha}")
        return empty

    cutoff = sorted_p[below].max()
    sig_mask = p_values <= cutoff
    gene_arr = np.array(gene_names)

    sig_idx = pair_indices[sig_mask]
    df = pd.DataFrame({
        "gene_A": gene_arr[sig_idx[:, 0]],
        "gene_B": gene_arr[sig_idx[:, 1]],
        "MI_MINE": mi_values[sig_mask],
        "p_value": p_values[sig_mask],
    })
    rank_in_sig = np.argsort(np.argsort(df["p_value"].values)) + 1
    df["p_adjusted"] = np.minimum(
        1.0, df["p_value"].values * n_tests / rank_in_sig
    )
    df.sort_values("p_adjusted", inplace=True)
    print(f"[INFO] BH-FDR edges (alpha={fdr_alpha}): {len(df):,}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Master consensus network (Section 5)
# ═══════════════════════════════════════════════════════════════════════════════

def build_master_network(
    study_results: list,
    gene_names: list,
    min_count: int = 3,
) -> tuple:
    """
    Combine per-study adjacency matrices into a master consensus network.

    Parameters
    ----------
    study_results : list[dict]
        Each dict must have keys ``"name"`` (str) and ``"adj"`` (ndarray).
    gene_names : list[str]
        Common gene names across all studies (after intersection).
    min_count : int
        Minimum number of studies an edge must appear in.

    Returns
    -------
    master_adj : np.ndarray, shape (n, n), dtype uint8
        Binary master adjacency matrix.
    edge_count : np.ndarray, shape (n, n), dtype int16
        Number of studies each edge appeared in.
    """
    n = len(gene_names)
    edge_count = np.zeros((n, n), dtype=np.int16)
    for res in study_results:
        edge_count += res["adj"].astype(np.int16)

    # Enforce symmetry
    edge_count = np.maximum(edge_count, edge_count.T)

    master_adj = (edge_count >= min_count).astype(np.uint8)
    np.fill_diagonal(master_adj, 0)

    n_edges = int(np.triu(master_adj, k=1).sum())
    print(f"[INFO] Master network: {n_edges:,} edges "
          f"(in >= {min_count} of {len(study_results)} studies)")
    return master_adj, edge_count


def aggregate_master_weights(
    n_genes: int,
    study_weight_records: list,
    master_adj: np.ndarray,
    mode: str = "n_studies",
    edge_count: np.ndarray = None,
) -> np.ndarray:
    """
    Aggregate study-level edge weights into a master weighted matrix.

    Parameters
    ----------
    n_genes : int
        Number of genes in master network.
    study_weight_records : list[dict]
        Each record has arrays: ``pairs`` (n,2 int), ``weights`` (n float).
    master_adj : np.ndarray
        Binary master adjacency used as support mask.
    mode : str
        ``n_studies`` | ``mean_mi`` | ``mean_neglog10p``.
    edge_count : np.ndarray, optional
        Required for ``mode='n_studies'``.
    """
    if mode == "n_studies":
        if edge_count is None:
            raise ValueError("edge_count is required for mode='n_studies'")
        w = edge_count.astype(np.float32)
        w[master_adj == 0] = 0.0
        return w

    w_sum = np.zeros((n_genes, n_genes), dtype=np.float32)
    w_n = np.zeros((n_genes, n_genes), dtype=np.float32)

    for rec in study_weight_records:
        pairs = rec["pairs"]
        vals = rec["weights"].astype(np.float32)
        if len(pairs) == 0:
            continue
        i = pairs[:, 0]
        j = pairs[:, 1]
        np.add.at(w_sum, (i, j), vals)
        np.add.at(w_sum, (j, i), vals)
        np.add.at(w_n, (i, j), 1.0)
        np.add.at(w_n, (j, i), 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.divide(w_sum, w_n, out=np.zeros_like(w_sum), where=w_n > 0)
    w[master_adj == 0] = 0.0
    np.fill_diagonal(w, 0.0)
    return w

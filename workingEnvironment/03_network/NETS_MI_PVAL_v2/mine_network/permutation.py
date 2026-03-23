"""
Permutation-based null distributions for MINE MI significance testing.
======================================================================

MINE gives a continuous MI estimate for a gene pair — it does **not** give
a p-value (Section 2, key point).  This module wraps MINE in a permutation
layer to produce empirical p-values.

Two modes (Section 3 + Section 4c)
-----------------------------------

**Global null** (``mode = "global"``, default — "probably the sweet spot"):
    Build *one* null distribution per study.
    1. Pick ``n_permutations`` random gene pairs.
    2. For each, permute one gene's samples to break dependence.
    3. Estimate MI via MINE.
    This works because, after Z-scoring, all genes ≈ N(0, 1) so the null
    distribution is approximately gene-pair-agnostic.
    Then, for each real gene pair, compare its observed MI to this global
    null to get an empirical p-value.
    Cost: O(n_permutations × epochs).

**Per-pair null** (``mode = "per_pair"``):
    For each real gene pair (g_i, g_j):
    1. Compute observed MI(g_i, g_j).
    2. For b = 1, …, B: permute z, re-estimate MI.
    3. p = (1 + #{null ≥ observed}) / (1 + B).
    More rigorous but O(n_pairs × B × epochs).  Only feasible with
    aggressive pre-screening.

The global null is the default because:
    - It reduces O(n_pairs × B) MINE trainings to O(B) trainings.
    - Results are nearly identical when marginals are standardised.
    - It matches the strategy in ``generate_net_python_pval.py`` which
      builds one null per study from 30 000 shuffled random pairs.

Statistical notes
-----------------
With n_permutations = 1 000 the minimum resolvable p-value is ~1e-3,
matching the default p_value_threshold = 0.001.  For stricter thresholds,
increase n_permutations accordingly (e.g. 10 000 for p < 1e-4).
"""

import numpy as np
from typing import Optional

from .mine_estimator import estimate_mi_for_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Global null distribution
# ═══════════════════════════════════════════════════════════════════════════════

def build_global_null(
    expr_matrix: np.ndarray,
    mine_cfg,
    n_permutations: int = 1_000,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """
    Build an empirical null distribution of MI under independence.

    Algorithm (Section 4c):
        For each of ``n_permutations`` trials:
          1. Randomly pick two gene indices i and j.
          2. Permute gene j's samples → breaks any dependence with gene i.
          3. Estimate MI(gene_i, permuted(gene_j)) via classical histogram MI.

    Why a global null is valid:
        After Z-scoring, every gene's marginal ≈ N(0, 1).  The null
        distribution of MI(X, permuted(Y)) therefore does not depend
        on which specific (X, Y) pair we choose.  One distribution
        serves all pairs.

    Parameters
    ----------
    expr_matrix : np.ndarray, shape (n_genes, n_samples)
        Z-scored expression.
    mine_cfg : PipelineConfig
        Pipeline configuration with MI parameters.
    n_permutations : int
        Number of null MI samples to generate.
    seed : int
        Random seed for reproducibility.
    device : str
        Compute device (unused, kept for interface compatibility).
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray, shape (n_permutations,)
        Null MI values (in nats).
    """
    rng = np.random.default_rng(seed)
    n_genes, n_samples = expr_matrix.shape

    # Sample random gene pairs for the null (exclude self-pairs).
    gi_idx = rng.integers(0, n_genes, size=n_permutations)
    gj_idx = rng.integers(0, n_genes - 1, size=n_permutations)
    gj_idx = np.where(gj_idx >= gi_idx, gj_idx + 1, gj_idx)

    # Create pair indices array
    pair_indices = np.column_stack([gi_idx, gj_idx])

    # Build null MI distribution by estimating MI for permuted pairs
    null_mi = np.zeros(n_permutations, dtype=np.float32)
    batch_size = int(getattr(mine_cfg, "batch_pairs", 8192))
    n_batches = (n_permutations + batch_size - 1) // batch_size

    if verbose:
        print(f"  Building global MI null: {n_permutations} permutations in {n_batches} batches...")

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_permutations)
        B = e - s

        batch_pairs = pair_indices[s:e]
        
        # Create permuted expression matrix for this batch
        expr_permuted = expr_matrix.copy()
        
        # For each pair in batch, permute the second gene independently
        for pair_idx, (gi, gj) in enumerate(batch_pairs):
            perm_idx = rng.permutation(n_samples)
            expr_permuted[gj] = expr_matrix[gj, perm_idx]
        
        # Estimate MI for these permuted pairs
        mi_batch, _ = estimate_mi_for_pairs(
            expr_permuted,
            batch_pairs,
            mine_cfg,
            device=device,
            verbose=False,
            seed=seed,
            n_jobs=1,
        )
        null_mi[s:e] = mi_batch[:B]

        if verbose and (b + 1) % max(1, n_batches // 5) == 0:
            print(f"  Null progress: {b+1}/{n_batches} batches")

    if verbose:
        print(f"  Null MI: mean={null_mi.mean():.4f}, std={null_mi.std():.4f}, "
              f"99.9th={np.percentile(null_mi, 99.9):.4f}")

    return null_mi


# ═══════════════════════════════════════════════════════════════════════════════
# Per-pair null distribution
# ═══════════════════════════════════════════════════════════════════════════════

def build_per_pair_null(
    expr_matrix: np.ndarray,
    pair_indices: np.ndarray,
    mine_cfg,
    n_permutations: int = 100,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """
    Build null MI distributions *per gene pair* via permutation.

    For each pair (g_i, g_j) and each permutation b:
      1. Permute g_j's samples.
      2. Estimate MI(g_i, permuted(g_j)) using classical histogram MI.

    This is the more rigorous approach (Section 3) but much more expensive.
    Use with aggressive pre-screening (few thousand pairs max).

    Parameters
    ----------
    expr_matrix : np.ndarray, shape (n_genes, n_samples)
        Z-scored expression.
    pair_indices : np.ndarray, shape (n_pairs, 2)
        Gene pair indices.
    mine_cfg : PipelineConfig
        Pipeline configuration with MI parameters.
    n_permutations : int
        Permutations *per pair*.  100–500 is typical.
    seed : int
        Random seed.
    device : str
        Compute device (unused, kept for interface compatibility).
    verbose : bool

    Returns
    -------
    np.ndarray, shape (n_pairs, n_permutations)
        Null MI values for each pair.
    """
    rng = np.random.default_rng(seed)
    n_pairs = len(pair_indices)
    n_samples = expr_matrix.shape[1]
    null_mi = np.zeros((n_pairs, n_permutations), dtype=np.float32)
    batch_size = int(getattr(mine_cfg, "batch_pairs", 8192))

    if verbose:
        print(f"  Per-pair null: {n_pairs} pairs × {n_permutations} perms...")

    # For each permutation round, process all pairs
    for p_idx in range(n_permutations):
        # Create permuted expression matrix for this round
        expr_permuted = expr_matrix.copy()
        
        # Permute each gene j in the pair list independently
        for gi, gj in pair_indices:
            perm_idx = rng.permutation(n_samples)
            expr_permuted[gj] = expr_matrix[gj, perm_idx]
        
        # Estimate MI for all pairs with this permutation
        mi_batch, _ = estimate_mi_for_pairs(
            expr_permuted,
            pair_indices,
            mine_cfg,
            device=device,
            verbose=False,
            seed=seed,
            n_jobs=-1,
        )
        null_mi[:, p_idx] = mi_batch

        if verbose and (p_idx + 1) % max(1, n_permutations // 10) == 0:
            print(f"  Perm progress: {p_idx + 1}/{n_permutations}")

    return null_mi


# ═══════════════════════════════════════════════════════════════════════════════
# P-value computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pvalues_global(
    mi_observed: np.ndarray,
    null_mi: np.ndarray,
) -> np.ndarray:
    """
    Compute empirical p-values against a global null distribution.

    p(i,j) = #{null_k ≥ MI_obs(i,j)} / N_perm

    Uses ``np.searchsorted`` on the sorted null for O(n_pairs × log n_perm).

    Parameters
    ----------
    mi_observed : np.ndarray, shape (n_pairs,)
        Observed MI from MINE.
    null_mi : np.ndarray, shape (n_permutations,)
        Global null distribution.

    Returns
    -------
    np.ndarray, shape (n_pairs,)
        Empirical p-values.
    """
    null_sorted = np.sort(null_mi)
    n_perm = len(null_sorted)
    insert_idx = np.searchsorted(null_sorted, mi_observed, side="left")
    return (n_perm - insert_idx) / n_perm


def compute_pvalues_per_pair(
    mi_observed: np.ndarray,
    null_mi_per_pair: np.ndarray,
) -> np.ndarray:
    """
    Compute empirical p-values from per-pair null distributions.

    p(i,j) = (1 + #{null ≥ MI_obs}) / (1 + B)

    The +1 in numerator and denominator is the standard correction that
    prevents p = 0 and makes the test exact.

    Parameters
    ----------
    mi_observed : np.ndarray, shape (n_pairs,)
        Observed MI values.
    null_mi_per_pair : np.ndarray, shape (n_pairs, n_permutations)
        Null MI for each pair.

    Returns
    -------
    np.ndarray, shape (n_pairs,)
        Empirical p-values.
    """
    n_perms = null_mi_per_pair.shape[1]
    exceedances = (null_mi_per_pair >= mi_observed[:, None]).sum(axis=1)
    return (1 + exceedances) / (1 + n_perms)

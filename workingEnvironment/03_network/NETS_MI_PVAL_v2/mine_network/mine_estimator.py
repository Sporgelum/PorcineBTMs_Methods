"""
Classical MI estimator for candidate gene pairs.
================================================

This module provides a fast histogram-based mutual information (MI)
estimator for discrete expression vectors. It preserves the MINE-era
pipeline interface so resume/caching/per-study diagnostics continue to work.

Notes
-----
- Expression is discretized once per study (equal-frequency quantile bins by default).
- MI is computed per candidate pair with a 2D histogram.
- Returns MI values and batch diagnostics compatible with existing I/O helpers.
"""

from __future__ import annotations

import sys
import time
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from joblib import Parallel, delayed


def discretize_expression(expr_matrix: np.ndarray, n_bins: int = 5, strategy: str = "quantile") -> np.ndarray:
    """Discretize genes x samples matrix into integer bins."""
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    return disc.fit_transform(expr_matrix.T).T.astype(np.int16, copy=False)


def compute_mi_histogram_discrete(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Compute MI(X;Y) from two integer-binned vectors using a joint histogram."""
    joint = np.bincount(x * n_bins + y, minlength=n_bins * n_bins).reshape(n_bins, n_bins)
    total = float(joint.sum())
    if total <= 0:
        return 0.0

    pxy = joint / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz])))


def _mi_for_pair(expr_discrete: np.ndarray, i: int, j: int, n_bins: int) -> float:
    return compute_mi_histogram_discrete(expr_discrete[i], expr_discrete[j], n_bins)


def estimate_mi_for_pairs(
    expr_matrix: np.ndarray,
    pair_indices: np.ndarray,
    mine_cfg,
    device=None,
    verbose: bool = True,
    seed=None,
    n_jobs: int = -1,
):
    """
    Estimate MI for candidate pairs using histogram MI on discretized expression.

    Parameters mirror the old MINE interface so pipeline integration remains stable.
    """
    del device, seed

    n_pairs = len(pair_indices)
    if n_pairs == 0:
        return np.zeros((0,), dtype=np.float32), []

    n_bins = int(getattr(mine_cfg, "mi_n_bins", 5))
    strategy = str(getattr(mine_cfg, "mi_strategy", "quantile"))
    batch_size = int(getattr(mine_cfg, "batch_pairs", 8192))
    n_batches = (n_pairs + batch_size - 1) // batch_size

    t_disc0 = time.time()
    expr_discrete = discretize_expression(expr_matrix, n_bins=n_bins, strategy=strategy)
    t_disc = time.time() - t_disc0

    mi_all = np.zeros(n_pairs, dtype=np.float32)
    diagnostics = []

    def _run_batch(batch_id: int, batch_pairs: np.ndarray, run_parallel=None):
        t0 = time.time()
        if run_parallel is None:
            mi_vals_local = [
                _mi_for_pair(expr_discrete, int(i), int(j), n_bins)
                for i, j in batch_pairs
            ]
        else:
            mi_vals_local = run_parallel(
                delayed(_mi_for_pair)(expr_discrete, int(i), int(j), n_bins)
                for i, j in batch_pairs
            )

        mi_vals_local = np.asarray(mi_vals_local, dtype=np.float32)
        elapsed_local = time.time() - t0

        diagnostics.append({
            "batch_id": batch_id,
            "n_pairs": int(len(batch_pairs)),
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "loss_reduction": 0.0,
            "initial_train_mi": float(mi_vals_local.mean()) if len(mi_vals_local) else 0.0,
            "final_train_mi": float(mi_vals_local.mean()) if len(mi_vals_local) else 0.0,
            "final_mi_mean": float(mi_vals_local.mean()) if len(mi_vals_local) else 0.0,
            "final_mi_std": float(mi_vals_local.std()) if len(mi_vals_local) else 0.0,
            "final_mi_max": float(mi_vals_local.max()) if len(mi_vals_local) else 0.0,
            "loss_curve": [0.0],
            "mi_curve_train": [float(mi_vals_local.mean()) if len(mi_vals_local) else 0.0],
            "runtime_seconds": float(elapsed_local),
        })
        return mi_vals_local

    if n_jobs == 1:
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_pairs)
            batch_pairs = pair_indices[start:end]
            mi_vals = _run_batch(b, batch_pairs, run_parallel=None)
            mi_all[start:end] = mi_vals

            if verbose and (b + 1) % max(1, n_batches // 20) == 0:
                pct = (b + 1) / n_batches * 100.0
                print(
                    f"  MI progress: {b + 1}/{n_batches} batches ({pct:.0f}%) "
                    f"| MI_mean={float(mi_vals.mean()):.4f}"
                )
                sys.stdout.flush()
    else:
        # Reuse a single worker pool for all batches and keep computation in
        # shared memory to avoid per-batch loky temp-folder churn.
        with Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem") as parallel:
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_pairs)
                batch_pairs = pair_indices[start:end]
                mi_vals = _run_batch(b, batch_pairs, run_parallel=parallel)
                mi_all[start:end] = mi_vals

                if verbose and (b + 1) % max(1, n_batches // 20) == 0:
                    pct = (b + 1) / n_batches * 100.0
                    print(
                        f"  MI progress: {b + 1}/{n_batches} batches ({pct:.0f}%) "
                        f"| MI_mean={float(mi_vals.mean()):.4f}"
                    )
                    sys.stdout.flush()

    if verbose:
        print(
            f"  MI discretization complete in {t_disc:.2f}s "
            f"| bins={n_bins} strategy={strategy}"
        )
        sys.stdout.flush()

    return mi_all, diagnostics

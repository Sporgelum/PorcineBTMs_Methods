"""
QC visualisation utilities for sample-level inspection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def _sample_spearman_corr(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Return sample-sample Spearman correlation matrix."""
    return expr_df.corr(method="spearman")


def _sample_quantile_curves(expr_df: pd.DataFrame, n_quantiles: int = 200):
    """Build per-sample quantile curves over genes for line-plot display."""
    q = np.linspace(0.0, 1.0, n_quantiles)
    curves = {}
    for col in expr_df.columns:
        curves[col] = np.quantile(expr_df[col].values, q)
    return q, curves


def save_sample_qc_figure(
    expr_df: pd.DataFrame,
    out_path: str,
    title: str,
    corr_threshold: float = None,
    n_quantiles: int = 200,
) -> None:
    """
    Save one figure containing:
      1) sample hierarchical clustering (Spearman, distance = 1-r)
      2) line plot of per-sample expression distributions
      3) Spearman correlation heatmap (optionally thresholded by |r|)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    corr = _sample_spearman_corr(expr_df)
    dist = 1.0 - corr.values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 2.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.2, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    dend = dendrogram(
        Z,
        labels=list(expr_df.columns),
        leaf_rotation=90,
        leaf_font_size=6,
        ax=ax1,
        color_threshold=None,
    )
    ax1.set_title("Sample Hierarchical Clustering (Spearman)")
    ax1.set_ylabel("Distance (1 - Spearman r)")

    ax2 = fig.add_subplot(gs[0, 1])
    q, curves = _sample_quantile_curves(expr_df, n_quantiles=n_quantiles)
    for sample in expr_df.columns:
        ax2.plot(q, curves[sample], alpha=0.25, linewidth=0.8)
    ax2.set_title("Gene Expression Distribution per Sample")
    ax2.set_xlabel("Quantile")
    ax2.set_ylabel("Expression")

    ax3 = fig.add_subplot(gs[0, 2])
    order = dend["leaves"]
    corr_ord = corr.values[np.ix_(order, order)]

    if corr_threshold is not None:
        corr_plot = np.where(np.abs(corr_ord) >= corr_threshold, corr_ord, 0.0)
        hm_title = f"Spearman Correlation Heatmap (|r| >= {corr_threshold})"
    else:
        corr_plot = corr_ord
        hm_title = "Spearman Correlation Heatmap"

    im = ax3.imshow(corr_plot, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax3.set_title(hm_title)
    ax3.set_xticks([])
    ax3.set_yticks([])
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman r")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[SAVED] {out_path}")
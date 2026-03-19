"""
Data loading — expression matrices, metadata, and study discovery.
==================================================================

Responsibilities
----------------
1. **Load the logCPM expression matrix** (genes × samples, tab-separated).
   - Auto-detects TSV/CSV from extension (defaults to tab).
   - Returns a ``pandas.DataFrame`` with gene names as row index, sample
     (Run / SRR) IDs as column names.

2. **Load sample metadata** from a tab-separated table.
   - Must contain at least two columns:
       ``Run``        — matches expression‐matrix column names (SRR IDs).
       ``BioProject`` — study / cohort identifier (PRJ IDs).
   - Additional columns (tissue, condition, etc.) are preserved for
     downstream annotation but not required by the pipeline.

3. **Discover studies** automatically from the ``BioProject`` column.
   - Each unique ``BioProject`` ID becomes one independent study.
   - Only Run IDs present in *both* the expression matrix and the metadata
     are used; extra rows are silently ignored.
   - Studies with fewer than ``min_samples`` samples are dropped (default 3).
   - Returns a list of study dicts, each with the expression sub-matrix
     and gene names, ready for per-study processing.

4. **Z-score expression** — standardise each gene to mean 0, std 1.
   - This is the continuous alternative to KBinsDiscretizer (quantile
     binning) used in the histogram-MI pipeline.
   - After Z-scoring every gene ≈ N(0, 1), so the permutation null
     is approximately gene-pair-agnostic (Section 4c of the user design).

Design notes
------------
- Separator detection: if the file extension ends with ``.tsv`` or ``.txt``
  the separator is ``\\t``; otherwise ``\\t`` is still the default because
  the user's data is tab-separated even with ``.csv`` extension.
- Gene names are always taken from the first column (row index).
- No genes are filtered at this stage — that is the responsibility of
  upstream QC or the pre-screening step.
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_expression(counts_path: str) -> pd.DataFrame:
    """
    Load the full logCPM expression matrix.

    Parameters
    ----------
    counts_path : str
        Path to the expression file (genes × samples, tab-separated).
        The first column is treated as the gene name index.

    Returns
    -------
    pd.DataFrame
        Rows = genes, columns = sample Run IDs.
    """
    expr = pd.read_csv(counts_path, sep="\t", index_col=0)
    print(f"[INFO] Loaded expression matrix: {counts_path}")
    print(f"[INFO] Shape: {expr.shape[0]:,} genes × {expr.shape[1]:,} samples")
    return expr


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load sample metadata.

    Parameters
    ----------
    metadata_path : str
        Path to the metadata file (tab-separated).
        Must contain columns ``Run`` (SRR IDs) and ``BioProject`` (PRJ IDs).

    Returns
    -------
    pd.DataFrame
        Full metadata table.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    md = pd.read_csv(metadata_path, sep="\t")
    print(f"[INFO] Loaded metadata: {metadata_path}")
    print(f"[INFO] Metadata shape: {md.shape}")
    for col in ("Run", "BioProject"):
        if col not in md.columns:
            raise ValueError(
                f"Metadata is missing required column '{col}'. "
                f"Available columns: {list(md.columns)}"
            )
    return md


def filter_genes(
    expr_full: pd.DataFrame,
    *,
    remove_ribosomal: bool = False,
    remove_mirna: bool = False,
    custom_regex: str = None,
    exclude_genes_file: str = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Filter genes by name-based patterns or explicit exclusion list.

    Parameters
    ----------
    expr_full : pd.DataFrame
        Full expression matrix (genes x samples).
    remove_ribosomal : bool
        Remove genes with ribosomal-like prefixes (RPL, RPS, MRPL, MRPS).
    remove_mirna : bool
        Remove genes with miRNA-like prefixes (MIR, MIRLET, LET-7, miR-).
    custom_regex : str or None
        Additional regex for exclusion (case-insensitive).
    exclude_genes_file : str or None
        Path to text file with one gene name per line to exclude.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Filtered matrix and summary dictionary.
    """
    gene_names = expr_full.index.astype(str)
    mask_remove = np.zeros(len(gene_names), dtype=bool)
    summary = {
        "input_genes": int(len(gene_names)),
        "removed_ribosomal": 0,
        "removed_mirna": 0,
        "removed_custom_regex": 0,
        "removed_from_file": 0,
        "removed_total": 0,
        "output_genes": int(len(gene_names)),
    }

    if remove_ribosomal:
        ribo_pat = r"^(RPL|RPS|MRPL|MRPS)\d*[A-Z]*$"
        ribo_mask = gene_names.str.match(ribo_pat, case=False, na=False)
        summary["removed_ribosomal"] = int(ribo_mask.sum())
        mask_remove |= ribo_mask.to_numpy()

    if remove_mirna:
        mirna_pat = r"^(MIR|MIRLET|LET[-_]?7|MIR[-_]?|MIR\d|miR[-_])"
        mirna_mask = gene_names.str.match(mirna_pat, case=False, na=False)
        summary["removed_mirna"] = int(mirna_mask.sum())
        mask_remove |= mirna_mask.to_numpy()

    if custom_regex:
        regex_mask = gene_names.str.contains(custom_regex, case=False, na=False, regex=True)
        summary["removed_custom_regex"] = int(regex_mask.sum())
        mask_remove |= regex_mask.to_numpy()

    if exclude_genes_file:
        excluded = set()
        with open(exclude_genes_file, "r", encoding="utf-8") as f:
            for line in f:
                g = line.strip()
                if g:
                    excluded.add(g)
        file_mask = gene_names.isin(excluded)
        summary["removed_from_file"] = int(file_mask.sum())
        mask_remove |= file_mask.to_numpy()

    summary["removed_total"] = int(mask_remove.sum())
    filtered = expr_full.loc[~mask_remove].copy()
    summary["output_genes"] = int(filtered.shape[0])

    return filtered, summary


# ═══════════════════════════════════════════════════════════════════════════════
# Study discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_studies(
    expr_full: pd.DataFrame,
    metadata: pd.DataFrame,
    min_samples: int = 3,
) -> list:
    """
    Auto-discover studies from the BioProject column of the metadata.

    Each unique ``BioProject`` value becomes one independent study.
    Only ``Run`` IDs present in the expression matrix are included.
    Studies with fewer than ``min_samples`` samples are skipped.

    Parameters
    ----------
    expr_full : pd.DataFrame
        Full expression matrix (genes × all samples).
    metadata : pd.DataFrame
        Must contain ``Run`` and ``BioProject`` columns.
    min_samples : int
        Minimum number of samples to include a study.

    Returns
    -------
    list[dict]
        Each dict has keys:

        - ``name`` : str — BioProject ID (filesystem-safe).
        - ``expr`` : pd.DataFrame — sub-matrix (genes × study_samples).
        - ``gene_names`` : list[str] — gene name list.
    """
    available_runs = set(expr_full.columns)
    md_matched = metadata[metadata["Run"].isin(available_runs)].copy()

    n_unmatched = len(metadata) - len(md_matched)
    if n_unmatched:
        print(f"[WARN] {n_unmatched} metadata rows without matching "
              f"expression columns (ignored).")

    studies = []
    for bioproj, group in md_matched.groupby("BioProject"):
        runs = group["Run"].tolist()
        if len(runs) < min_samples:
            print(f"[WARN] {bioproj}: {len(runs)} samples < "
                  f"{min_samples} — skipping")
            continue
        sub = expr_full[runs]
        safe_name = str(bioproj).replace(" ", "_").replace("/", "-")
        studies.append({
            "name": safe_name,
            "expr": sub,
            "gene_names": sub.index.tolist(),
        })
        print(f"[INFO] Study: {safe_name} ({len(runs)} samples)")

    print(f"[INFO] Total studies discovered: {len(studies)}")
    return studies


# ═══════════════════════════════════════════════════════════════════════════════
# Z-scoring
# ═══════════════════════════════════════════════════════════════════════════════

def zscore_expression(expr_data: pd.DataFrame) -> np.ndarray:
    """
    Z-score each gene across samples: (x - μ) / σ.

    After this transform every gene's marginal ≈ N(0, 1), which makes the
    permutation null distribution approximately gene-pair-agnostic.  This
    is the continuous-data analogue of quantile binning.

    Parameters
    ----------
    expr_data : pd.DataFrame
        Expression sub-matrix (genes × study_samples), raw logCPM.

    Returns
    -------
    np.ndarray, shape (n_genes, n_samples), dtype float32
        Z-scored expression values.
    """
    X = expr_data.values.astype(np.float32)
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero for constant genes
    return (X - mu) / std


def select_top_genes_by_mad(expr_full: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Keep the top-N most variable genes by MAD across samples.

    Parameters
    ----------
    expr_full : pd.DataFrame
        Full expression matrix (genes × samples).
    top_n : int
        Number of genes to retain. If >= number of genes, matrix is returned
        unchanged.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix with top-N MAD genes.
    """
    n_genes = expr_full.shape[0]
    if top_n is None or top_n <= 0 or top_n >= n_genes:
        return expr_full

    X = expr_full.values.astype(np.float32)
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1)

    keep_idx = np.argpartition(mad, n_genes - top_n)[-top_n:]
    keep_idx = keep_idx[np.argsort(mad[keep_idx])[::-1]]

    filtered = expr_full.iloc[keep_idx].copy()
    print(f"[INFO] MAD filtering: kept top {top_n:,} / {n_genes:,} genes")
    print(f"[INFO] MAD range kept: {mad[keep_idx].min():.6f} – "
          f"{mad[keep_idx].max():.6f}")
    return filtered

"""
I/O utilities — logging, timing, saving all result artefacts.
=============================================================

This module handles all file I/O for the pipeline:

Logging
-------
``TeeLogger`` duplicates stdout to both the terminal and a log file,
ensuring every print statement is captured for reproducibility.

Timing
------
``Timer`` is a context manager that records wall-clock time for named
pipeline steps.  The accumulated dict is used in the final report.

Saving — per study
------------------
- ``edges_mine_{study}.tsv``       — significant edge list (gene_A, gene_B, MI, p)
- ``adj_mine_{study}.mtx``         — sparse adjacency matrix (Matrix Market)
- ``network_mine_{study}.graphml`` — igraph GraphML for Cytoscape/Gephi
- ``null_distribution_{study}.txt``— null MI statistics for QC
- ``edges_bh_fdr_{study}.tsv``     — (optional) BH-corrected edges

Saving — master network
------------------------
- ``master_network_edgelist.tsv``  — edges with study-count column
- ``master_network_adjacency.mtx`` — sparse binary adjacency
- ``master_edge_study_counts.mtx`` — sparse study-count matrix
- ``master_network.graphml``       — GraphML with module annotations
- ``master_BTM_modules.tsv``       — module membership (Gene, Module)
- ``master_node_modules.tsv``      — gene → module mapping
- ``master_submodule_M{id}.graphml``— per-module subgraph files
- ``module_annotations.tsv``       — enrichment results (if GMT provided)
- ``analysis_report_{timestamp}.txt`` — full pipeline summary
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import igraph as ig


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    """
    Duplicate stdout to both the terminal and a log file.

    Usage::

        sys.stdout = TeeLogger("pipeline.log")
        print("This goes to both screen and file")

    Parameters
    ----------
    log_file : str
        Path to the log file (created/overwritten).
    """

    def __init__(self, log_file: str):
        self.terminal = sys.__stdout__
        self.log = open(log_file, "w", buffering=1, encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# Timing
# ═══════════════════════════════════════════════════════════════════════════════

class Timer:
    """
    Context manager that records wall-clock time for a named step.

    Usage::

        timings = {}
        with Timer("Load data", timings):
            data = load_expression(path)
        # timings["Load data"] == elapsed seconds

    Parameters
    ----------
    name : str
        Human-readable name of the step.
    report_dict : dict
        Dictionary to store ``{name: elapsed_seconds}``.
    """

    def __init__(self, name: str, report_dict: dict):
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


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes ({seconds:.1f}s)"
    else:
        h = seconds / 3600
        m = (seconds % 3600) / 60
        return f"{h:.2f} hours ({m:.1f}m)"


# ═══════════════════════════════════════════════════════════════════════════════
# MINE training diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def save_mine_diagnostics(
    diagnostics: list,
    study_name: str,
    output_dir: str,
) -> None:
    """
    Save MINE training diagnostics (loss curves, MI statistics per batch).

    Produces two files:
      - ``mine_diagnostics_{study}.tsv``  — per-batch summary
      - ``mine_loss_curve_{study}.tsv``   — epoch-level loss (averaged)

    Parameters
    ----------
    diagnostics : list[dict]
        Each dict has: batch_id, n_pairs, loss_curve, final_mi_mean,
        final_mi_std, final_mi_max.
    study_name : str
        Study identifier.
    output_dir : str
        Output directory.
    """
    import json

    diag_dir = os.path.join(output_dir, "mine_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    # Per-batch summary table
    rows = []
    for d in diagnostics:
        lc = d["loss_curve"]
        rows.append({
            "batch_id": d["batch_id"],
            "n_pairs": d["n_pairs"],
            "initial_loss": lc[0] if lc else float("nan"),
            "final_loss": lc[-1] if lc else float("nan"),
            "loss_reduction": (lc[0] - lc[-1]) if lc else float("nan"),
            "final_mi_mean": d["final_mi_mean"],
            "final_mi_std": d["final_mi_std"],
            "final_mi_max": d["final_mi_max"],
        })
    df = pd.DataFrame(rows)
    batch_path = os.path.join(diag_dir, f"mine_batch_summary_{study_name}.tsv")
    df.to_csv(batch_path, sep="\t", index=False)
    print(f"[SAVED] {batch_path}")

    # Aggregate loss curve (average across batches, per epoch)
    max_epochs = max(len(d["loss_curve"]) for d in diagnostics) if diagnostics else 0
    if max_epochs > 0:
        epoch_losses = np.full((len(diagnostics), max_epochs), np.nan)
        for i, d in enumerate(diagnostics):
            lc = d["loss_curve"]
            epoch_losses[i, :len(lc)] = lc
        mean_loss = np.nanmean(epoch_losses, axis=0)
        std_loss = np.nanstd(epoch_losses, axis=0)

        lc_df = pd.DataFrame({
            "epoch": np.arange(max_epochs),
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "mean_mi": -mean_loss,  # loss = -MI, so MI ≈ -loss
        })
        lc_path = os.path.join(diag_dir, f"mine_loss_curve_{study_name}.tsv")
        lc_df.to_csv(lc_path, sep="\t", index=False)
        print(f"[SAVED] {lc_path}")

    # Also save full raw diagnostics as JSON for detailed analysis
    raw_path = os.path.join(diag_dir, f"mine_raw_diagnostics_{study_name}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"[SAVED] {raw_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Per-study saving
# ═══════════════════════════════════════════════════════════════════════════════

def save_null_qc(
    null_mi: np.ndarray,
    study_name: str,
    p_threshold: float,
    output_dir: str,
) -> float:
    """
    Save null distribution summary for QC inspection.

    Parameters
    ----------
    null_mi : np.ndarray
        Null MI values.
    study_name : str
        Study identifier.
    p_threshold : float
        P-value threshold (for computing the MI cutoff).
    output_dir : str
        Output directory.

    Returns
    -------
    float
        MI threshold corresponding to the p-value cutoff.
    """
    mi_thr = np.percentile(null_mi, (1.0 - p_threshold) * 100)
    out = os.path.join(output_dir, f"null_distribution_{study_name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Null MI distribution (MINE) — {study_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"N permutations : {len(null_mi)}\n")
        f.write(f"Mean           : {null_mi.mean():.6f}\n")
        f.write(f"Std            : {null_mi.std():.6f}\n")
        f.write(f"99.9th pct     : {np.percentile(null_mi, 99.9):.6f}\n")
        f.write(f"MI threshold   : {mi_thr:.6f}  (p < {p_threshold})\n")
    print(f"[SAVED] {out}")
    return mi_thr


def save_study_results(
    study_name: str,
    adj: np.ndarray,
    edgelist_df: pd.DataFrame,
    gene_names: list,
    output_dir: str,
    bh_df: pd.DataFrame = None,
) -> None:
    """
    Save per-study results: edge list, adjacency, GraphML.

    Parameters
    ----------
    study_name : str
    adj : np.ndarray
        Binary adjacency matrix.
    edgelist_df : pd.DataFrame
        Significant edges.
    gene_names : list[str]
    output_dir : str
    bh_df : pd.DataFrame, optional
        BH-corrected edges.
    """
    print(f"\n[INFO] Saving results for: {study_name}")

    # Edge list
    edgelist_df.to_csv(
        os.path.join(output_dir, f"edges_mine_{study_name}.tsv"),
        sep="\t", index=False,
    )

    # Sparse adjacency
    mtx_file = os.path.join(output_dir, f"adj_mine_{study_name}.mtx")
    mmwrite(mtx_file, csr_matrix(adj))

    # GraphML
    adj_sym = np.maximum(adj, adj.T)
    g = ig.Graph.Adjacency((adj_sym > 0).tolist(), mode="undirected")
    g.vs["name"] = gene_names
    g.write_graphml(
        os.path.join(output_dir, f"network_mine_{study_name}.graphml")
    )

    # Optional BH-FDR
    if bh_df is not None and len(bh_df) > 0:
        bh_df.to_csv(
            os.path.join(output_dir, f"edges_bh_fdr_{study_name}.tsv"),
            sep="\t", index=False,
        )

    print(f"[SAVED] Study {study_name}: edges, adjacency, GraphML")


# ═══════════════════════════════════════════════════════════════════════════════
# Master network saving
# ═══════════════════════════════════════════════════════════════════════════════

def save_master_results(
    master_adj: np.ndarray,
    edge_count: np.ndarray,
    gene_names: list,
    modules: dict,
    membership: dict,
    min_count: int,
    n_studies: int,
    output_dir: str,
) -> None:
    """
    Save master network, modules, and subgraph files.

    Creates:
    - Edge list TSV with study-count column
    - Sparse adjacency and study-count matrices (MTX)
    - GraphML with MCODE module attribute
    - Module membership tables
    - Per-module subgraph GraphML files

    Parameters
    ----------
    master_adj : np.ndarray
        Binary master adjacency.
    edge_count : np.ndarray
        Study count per edge.
    gene_names : list[str]
    modules : dict
        MCODE modules.
    membership : dict
        Gene → module mapping.
    min_count : int
        Minimum study count used.
    n_studies : int
        Total number of studies.
    output_dir : str
    """
    print("\n[INFO] Saving master network...")
    gene_arr = np.array(gene_names)
    rows, cols = np.where(np.triu(master_adj, k=1) == 1)

    # Edge list with study count
    pd.DataFrame({
        "gene_A": gene_arr[rows],
        "gene_B": gene_arr[cols],
        "n_studies": edge_count[rows, cols],
    }).sort_values("n_studies", ascending=False).to_csv(
        os.path.join(output_dir, "master_network_edgelist.tsv"),
        sep="\t", index=False,
    )

    # Sparse matrices
    mmwrite(os.path.join(output_dir, "master_network_adjacency.mtx"),
            csr_matrix(master_adj))
    mmwrite(os.path.join(output_dir, "master_edge_study_counts.mtx"),
            csr_matrix(np.triu(edge_count)))

    # GraphML with module annotation
    adj_sym = np.maximum(master_adj, master_adj.T)
    g = ig.Graph.Adjacency((adj_sym > 0).tolist(), mode="undirected")
    g.vs["name"] = gene_names
    g.vs["module"] = [
        f"M{membership[gn]}" if gn in membership else "unassigned"
        for gn in gene_names
    ]
    g.write_graphml(os.path.join(output_dir, "master_network.graphml"))

    # Module membership tables
    btm_rows = [
        {"Gene": gene, "Module": f"M{mid}"}
        for mid, genes in modules.items()
        for gene in genes
    ]
    pd.DataFrame(btm_rows).to_csv(
        os.path.join(output_dir, "master_BTM_modules.tsv"),
        sep="\t", index=False,
    )
    pd.DataFrame([
        {"gene": g, "module": f"M{m}"} for g, m in membership.items()
    ]).to_csv(
        os.path.join(output_dir, "master_node_modules.tsv"),
        sep="\t", index=False,
    )

    # Per-module subgraphs
    gene_name_list = list(gene_names)
    saved = 0
    for mid, mod_genes in modules.items():
        if len(mod_genes) < 3:
            continue
        idx = [gene_name_list.index(gn) for gn in mod_genes]
        sub = np.maximum(
            master_adj[np.ix_(idx, idx)],
            master_adj[np.ix_(idx, idx)].T,
        )
        sg = ig.Graph.Adjacency((sub > 0).tolist(), mode="undirected")
        sg.vs["name"] = mod_genes
        sg.write_graphml(
            os.path.join(output_dir, f"master_submodule_M{mid}.graphml")
        )
        saved += 1

    print(f"[SAVED] Master: edgelist, adjacency, GraphML, "
          f"{len(modules)} modules, {saved} subgraphs")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline report
# ═══════════════════════════════════════════════════════════════════════════════

def save_report(
    timings: dict,
    info: dict,
    report_path: str,
) -> None:
    """
    Write a summary report with timing breakdown and key statistics.

    Parameters
    ----------
    timings : dict[str, float]
        Step names → elapsed seconds.
    info : dict[str, str]
        Key → value pairs to include in the report.
    report_path : str
        Output file path.
    """
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MINE-BASED GENE NETWORK INFERENCE — REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("RESULTS\n" + "-" * 80 + "\n")
        for k, v in info.items():
            f.write(f"  {k:50s}: {v}\n")

        f.write("\nTIMING\n" + "-" * 80 + "\n")
        total = sum(timings.values()) if timings else 0
        for step, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            pct = t / total * 100 if total > 0 else 0
            f.write(f"  {step:55s}: {format_time(t):>20s} ({pct:5.1f}%)\n")
        f.write(f"\n  {'TOTAL':55s}: {format_time(total):>20s}\n")

    print(f"[SAVED] Report: {report_path}")

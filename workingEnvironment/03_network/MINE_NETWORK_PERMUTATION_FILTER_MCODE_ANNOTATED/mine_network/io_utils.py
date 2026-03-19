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
from .network_viz import save_study_minimap, save_master_minimaps


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
    make_minimap: bool = False,
    minimap_base_dir: str = None,
    minimap_max_nodes: int = 1200,
    minimap_dpi: int = 180,
    minimap_edge_alpha: float = 0.08,
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
    make_minimap : bool
        If True, save a small per-study network PNG.
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

    if make_minimap and minimap_base_dir:
        try:
            save_study_minimap(
                study_name=study_name,
                adj=adj_sym,
                gene_names=gene_names,
                base_dir=minimap_base_dir,
                max_nodes=minimap_max_nodes,
                dpi=minimap_dpi,
                edge_alpha=minimap_edge_alpha,
            )
        except Exception as e:
            print(f"[WARN] Could not save study minimap for {study_name}: {e}")

    print(f"[SAVED] Study {study_name}: edges, adjacency, GraphML")


def summarize_network_topology(study_name: str, adj: np.ndarray) -> dict:
    """
    Compute connectivity summary for one study network.

    Returns fields used by study_net_stats.tsv.
    """
    adj_sym = np.maximum(adj, adj.T).astype(np.uint8)
    np.fill_diagonal(adj_sym, 0)

    n = int(adj_sym.shape[0])
    m = int(np.triu(adj_sym, k=1).sum())
    max_edges = n * (n - 1) // 2
    density = (m / max_edges) if max_edges > 0 else 0.0

    rows, cols = np.where(np.triu(adj_sym, k=1) > 0)
    g = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())), directed=False)
    comp_sizes = sorted(g.components().sizes(), reverse=True)

    n_components = len(comp_sizes)
    giant_component_size = int(comp_sizes[0]) if comp_sizes else 0
    giant_component_fraction = (giant_component_size / n) if n > 0 else 0.0
    singleton_components = int(sum(1 for s in comp_sizes if s == 1))

    return {
        "study": study_name,
        "nodes": n,
        "edges": m,
        "density": density,
        "components": n_components,
        "giant_component": giant_component_size,
        "giant_component_fraction": giant_component_fraction,
        "singleton_components": singleton_components,
    }


def save_study_network_stats(stats_rows: list, minimap_base_dir: str) -> None:
    """Save per-study network connectivity stats to minimap_networks/study_net_stats.tsv."""
    if not stats_rows:
        return

    os.makedirs(minimap_base_dir, exist_ok=True)
    stats_df = pd.DataFrame(stats_rows).sort_values("study")
    out_path = os.path.join(minimap_base_dir, "study_net_stats.tsv")
    stats_df.to_csv(out_path, sep="\t", index=False)
    print(f"[SAVED] {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Master network saving
# ═══════════════════════════════════════════════════════════════════════════════

def _append_gene_metadata(
    module_df: pd.DataFrame,
    gene_col: str,
    map_path: str,
    key_col: str,
    extra_cols: list,
) -> pd.DataFrame:
    """
    Append gene metadata columns to a module table using a mapping file.

    The mapping file can be TSV or CSV. If duplicate keys exist, values are
    collapsed per key using semicolon-separated unique entries.
    """
    if module_df.empty:
        return module_df.copy()

    map_df = pd.read_csv(map_path, sep=None, engine="python", dtype=str)
    map_df.columns = [c.strip() for c in map_df.columns]

    if key_col not in map_df.columns:
        raise ValueError(
            f"Column '{key_col}' was not found in mapping file: {map_path}"
        )

    selected_cols = [c for c in (extra_cols or []) if c in map_df.columns]
    if not selected_cols:
        selected_cols = [c for c in map_df.columns if c != key_col]

    if not selected_cols:
        return module_df.copy()

    compact = map_df[[key_col] + selected_cols].copy()
    compact[key_col] = compact[key_col].astype(str).str.strip()

    def _collapse(series: pd.Series) -> str:
        vals = []
        for value in series.fillna("").astype(str):
            value = value.strip()
            if not value or value.lower() == "nan":
                continue
            vals.append(value)
        if not vals:
            return ""
        return ";".join(sorted(set(vals)))

    compact = (
        compact
        .groupby(key_col, as_index=False)
        .agg({col: _collapse for col in selected_cols})
    )

    out_df = module_df.copy()
    out_df[gene_col] = out_df[gene_col].astype(str).str.strip()
    out_df = out_df.merge(compact, left_on=gene_col, right_on=key_col, how="left")
    if key_col != gene_col:
        out_df = out_df.drop(columns=[key_col])
    return out_df

def save_master_results(
    master_adj: np.ndarray,
    edge_count: np.ndarray,
    master_edge_weight: np.ndarray,
    gene_names: list,
    modules: dict,
    membership: dict,
    parent_child_rows: list,
    min_count: int,
    n_studies: int,
    output_dir: str,
    module_export_map_path: str = None,
    module_export_key_col: str = "ensembl_gene_id",
    module_export_cols: list = None,
    make_minimap: bool = False,
    minimap_base_dir: str = None,
    minimap_max_nodes: int = 1200,
    minimap_dpi: int = 180,
    minimap_edge_alpha: float = 0.08,
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
    master_edge_weight : np.ndarray
        Weighted master edge matrix.
    gene_names : list[str]
    modules : dict
        MCODE modules.
    membership : dict
        Gene → module mapping.
    parent_child_rows : list[dict]
        Parent-child mapping for refined modules.
    min_count : int
        Minimum study count used.
    n_studies : int
        Total number of studies.
    output_dir : str
    module_export_map_path : str or None
        Optional mapping table to append additional gene identifier columns
        to module membership outputs.
    module_export_key_col : str
        Gene ID key column in the mapping table.
    module_export_cols : list[str] or None
        Columns to append from the mapping table. If None/empty, all columns
        except ``module_export_key_col`` are appended.
    make_minimap : bool
        If True, save master network minimap PNG files.
    """
    print("\n[INFO] Saving master network...")
    master_edge_weight = np.asarray(master_edge_weight, dtype=float)
    gene_arr = np.array(gene_names)
    rows, cols = np.where(np.triu(master_adj, k=1) == 1)

    # Edge list with study count + optional weight
    edge_df = pd.DataFrame({
        "gene_A": gene_arr[rows],
        "gene_B": gene_arr[cols],
        "n_studies": edge_count[rows, cols],
        "edge_weight": master_edge_weight[rows, cols],
    })
    edge_df.sort_values(["n_studies", "edge_weight"], ascending=False).to_csv(
        os.path.join(output_dir, "master_network_edgelist.tsv"),
        sep="\t", index=False,
    )

    edge_df[["gene_A", "gene_B", "edge_weight"]].sort_values(
        "edge_weight", ascending=False
    ).to_csv(
        os.path.join(output_dir, "master_network_weighted_edgelist.tsv"),
        sep="\t", index=False,
    )

    # Sparse matrices
    mmwrite(os.path.join(output_dir, "master_network_adjacency.mtx"),
            csr_matrix(master_adj))
    mmwrite(os.path.join(output_dir, "master_edge_study_counts.mtx"),
            csr_matrix(np.triu(edge_count)))
    mmwrite(os.path.join(output_dir, "master_edge_weights.mtx"),
            csr_matrix(np.triu(master_edge_weight)))

    # GraphML with module annotation + edge attributes
    g = ig.Graph(
        n=len(gene_names),
        edges=list(zip(rows.tolist(), cols.tolist())),
        directed=False,
    )
    g.vs["name"] = gene_names
    g.vs["module"] = [
        f"M{membership[gn]}" if gn in membership else "unassigned"
        for gn in gene_names
    ]
    g.es["n_studies"] = edge_count[rows, cols].astype(int).tolist()
    g.es["edge_weight"] = master_edge_weight[rows, cols].astype(float).tolist()
    g.write_graphml(os.path.join(output_dir, "master_network.graphml"))

    # Module membership tables
    btm_rows = [
        {"Gene": gene, "Module": f"M{mid}"}
        for mid, genes in modules.items()
        for gene in genes
    ]
    btm_df = pd.DataFrame(btm_rows)
    btm_df.to_csv(
        os.path.join(output_dir, "master_BTM_modules.tsv"),
        sep="\t", index=False,
    )
    node_df = pd.DataFrame([
        {"gene": g, "module": f"M{m}"} for g, m in membership.items()
    ])
    node_df.to_csv(
        os.path.join(output_dir, "master_node_modules.tsv"),
        sep="\t", index=False,
    )

    if module_export_map_path:
        try:
            btm_annot = _append_gene_metadata(
                btm_df,
                gene_col="Gene",
                map_path=module_export_map_path,
                key_col=module_export_key_col,
                extra_cols=module_export_cols or [],
            )
            node_annot = _append_gene_metadata(
                node_df,
                gene_col="gene",
                map_path=module_export_map_path,
                key_col=module_export_key_col,
                extra_cols=module_export_cols or [],
            )
            btm_annot.to_csv(
                os.path.join(output_dir, "master_BTM_modules_annotated.tsv"),
                sep="\t", index=False,
            )
            node_annot.to_csv(
                os.path.join(output_dir, "master_node_modules_annotated.tsv"),
                sep="\t", index=False,
            )
            mapped = int((btm_annot.drop(columns=["Gene", "Module"], errors="ignore")
                          .notna()
                          .any(axis=1)
                          .sum()))
            print(f"[SAVED] Annotated module tables: {mapped:,}/{len(btm_annot):,} rows with metadata")
        except Exception as e:
            print(f"[WARN] Could not append module metadata columns: {e}")

    if parent_child_rows:
        pd.DataFrame(parent_child_rows).to_csv(
            os.path.join(output_dir, "module_parent_child_mapping.tsv"),
            sep="\t", index=False,
        )

    if make_minimap and minimap_base_dir:
        try:
            save_master_minimaps(
                master_adj=master_adj,
                gene_names=gene_names,
                membership=membership,
                parent_child_rows=parent_child_rows,
                base_dir=minimap_base_dir,
                max_nodes=minimap_max_nodes,
                dpi=minimap_dpi,
                edge_alpha=minimap_edge_alpha,
            )
        except Exception as e:
            print(f"[WARN] Could not save master minimaps: {e}")

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

    print(f"[SAVED] Master: edgelist, weighted edges, adjacency, GraphML, "
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

"""
Pipeline orchestrator — end-to-end MINE gene network inference.
================================================================

This is the core logic that ties all modules together.  It implements the
complete workflow described in the user's conceptual pipeline (Sections 1–8):

    1. Load logCPM expression + metadata
    2. Discover studies from BioProject column
    3. Per study:
       a. Z-score expression (continuous, no binning)
       b. (Optional) Pre-screen gene pairs by Pearson |r|
       c. Estimate MI for candidates via batched MINE (GPU)
       d. Build permutation null (global or per-pair)
       e. Filter edges by empirical p-value < 0.001
       f. Save per-study network
    4. Build master consensus network (edges in ≥ 3 studies)
    5. Run MCODE on master network
    6. Annotate modules (hypergeometric enrichment against GMT)
    7. Save everything

Separation of concerns
-----------------------
This module only *orchestrates*.  All real work is delegated to:
  - ``data_loader`` for I/O
  - ``mine_estimator`` for MI computation
  - ``permutation`` for null distributions and p-values
  - ``prescreen`` for correlation filtering
  - ``network`` for edge filtering and master construction
  - ``mcode`` for module detection
  - ``annotation`` for enrichment
  - ``io_utils`` for saving

This design means you can import and use individual components in notebooks
or custom scripts without running the full pipeline.
"""

import os
import sys
import gc
import time
import json
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.io import mmread

from .config import PipelineConfig
from .data_loader import (
    load_expression,
    load_metadata,
    discover_studies,
    zscore_expression,
    filter_genes,
    select_top_genes_by_mad,
)
from .mine_estimator import estimate_mi_for_pairs
from .permutation import (
    build_global_null, build_per_pair_null,
    compute_pvalues_global, compute_pvalues_per_pair,
)
from .prescreen import prescreen_pairs, all_pairs
from .network import (
    filter_edges,
    build_edgelist,
    apply_bh_fdr,
    build_master_network,
    aggregate_master_weights,
)
from .mcode import mcode, leiden_modules, refine_large_modules
from .annotation import (
    load_multiple_gmt_with_sources,
    annotate_modules,
    save_annotations,
    save_annotations_by_source,
    download_enrichr_libraries,
)
from .ortholog import load_ortholog_map, map_modules, map_gene_set
from .io_utils import (
    TeeLogger, Timer, format_time,
    save_null_qc, save_study_results, save_master_results, save_report,
    summarize_network_topology, save_study_network_stats,
    save_mine_diagnostics,
    ensure_mine_diagnostics_plot,
)
from .qc_plots import save_sample_qc_figure


def _reindex_pairs_to_subset(
    pairs: np.ndarray,
    weights: np.ndarray,
    keep_idx: list,
    original_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map edge pairs from an original gene index space into a kept subset."""
    if len(pairs) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.asarray(weights, dtype=np.float32),
        )

    mapping = np.full(original_size, -1, dtype=np.int32)
    mapping[np.asarray(keep_idx, dtype=np.int32)] = np.arange(len(keep_idx), dtype=np.int32)
    new_pairs = mapping[pairs]
    keep_mask = (new_pairs[:, 0] >= 0) & (new_pairs[:, 1] >= 0)
    return new_pairs[keep_mask], np.asarray(weights, dtype=np.float32)[keep_mask]


def _study_artifact_paths(output_dir: str, study_name: str) -> dict:
    """Return per-study artifact paths used for resume/checkpoint logic."""
    diag_dir = os.path.join(output_dir, "mine_diagnostics")
    return {
        "edges": os.path.join(output_dir, f"edges_mine_{study_name}.tsv"),
        "adj": os.path.join(output_dir, f"adj_mine_{study_name}.mtx"),
        "graphml": os.path.join(output_dir, f"network_mine_{study_name}.graphml"),
        "null": os.path.join(output_dir, f"null_distribution_{study_name}.txt"),
        "batch_summary": os.path.join(diag_dir, f"mine_batch_summary_{study_name}.tsv"),
        "loss_curve": os.path.join(diag_dir, f"mine_loss_curve_{study_name}.tsv"),
        "raw_diag": os.path.join(diag_dir, f"mine_raw_diagnostics_{study_name}.json"),
        "diag_plot": os.path.join(diag_dir, f"mine_diagnostics_plot_{study_name}.png"),
        "mi_cache": os.path.join(diag_dir, f"mine_scores_{study_name}.npz"),
    }


def _mine_cache_fingerprint(cfg: PipelineConfig, study_name: str, n_genes: int) -> dict:
    """Build a compact config fingerprint to prevent stale MI cache reuse."""
    return {
        "schema": 1,
        "study": str(study_name),
        "n_genes": int(n_genes),
        "mine": {
            "hidden_dim": int(cfg.mine.hidden_dim),
            "n_epochs": int(cfg.mine.n_epochs),
            "learning_rate": float(cfg.mine.learning_rate),
            "ema_alpha": float(cfg.mine.ema_alpha),
            "batch_pairs": str(cfg.mine.batch_pairs),
            "gradient_clip": float(cfg.mine.gradient_clip),
            "n_eval_shuffles": int(cfg.mine.n_eval_shuffles),
            "mixed_precision": bool(cfg.mine.mixed_precision),
        },
        "prescreen": {
            "enabled": bool(cfg.prescreen.enabled),
            "method": str(cfg.prescreen.method),
            "threshold": float(cfg.prescreen.threshold),
            "max_pairs": int(cfg.prescreen.max_pairs),
        },
        "gene_filter": {
            "enabled": bool(cfg.gene_filter.enabled),
            "remove_ribosomal": bool(cfg.gene_filter.remove_ribosomal),
            "remove_mirna": bool(cfg.gene_filter.remove_mirna),
            "custom_regex": str(cfg.gene_filter.custom_regex),
            "exclude_genes_file": str(cfg.gene_filter.exclude_genes_file),
        },
        "qc": {
            "mad_top_genes": str(cfg.qc.mad_top_genes),
        },
    }


def _study_is_completed(paths: dict) -> bool:
    """Return True if core per-study outputs exist and study can be skipped."""
    required = [
        paths["edges"],
        paths["adj"],
        paths["graphml"],
        paths["null"],
        paths["batch_summary"],
        paths["loss_curve"],
        paths["raw_diag"],
    ]
    return all(os.path.exists(p) for p in required)


def _load_completed_study_payload(
    study_idx: int,
    study: dict,
    cfg: PipelineConfig,
    paths: dict,
) -> dict:
    """Load saved study results and convert to payload used by master merge."""
    study_name = study["name"]
    gene_names = list(study["gene_names"])
    n_genes = len(gene_names)
    n_samples = study["expr"].shape[1]
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    adj_sparse = mmread(paths["adj"]).tocsr()
    adj_sig = (adj_sparse.toarray() > 0).astype(np.int8)

    edgelist_df = pd.read_csv(paths["edges"], sep="\t")
    if len(edgelist_df) > 0:
        valid = (
            edgelist_df["gene_A"].isin(gene_to_idx)
            & edgelist_df["gene_B"].isin(gene_to_idx)
        )
        edgelist_df = edgelist_df.loc[valid].copy()

    if len(edgelist_df) > 0:
        sig_pairs = np.column_stack([
            edgelist_df["gene_A"].map(gene_to_idx).to_numpy(np.int32),
            edgelist_df["gene_B"].map(gene_to_idx).to_numpy(np.int32),
        ])
        if "edge_weight" in edgelist_df.columns:
            edge_weights = edgelist_df["edge_weight"].to_numpy(np.float32)
        elif cfg.module.master_edge_weight == "mean_neglog10p" and "p_value" in edgelist_df.columns:
            edge_weights = -np.log10(edgelist_df["p_value"].to_numpy(np.float32) + cfg.module.weight_eps)
        elif cfg.module.master_edge_weight == "mean_mi" and "MI_MINE" in edgelist_df.columns:
            edge_weights = edgelist_df["MI_MINE"].to_numpy(np.float32)
        else:
            edge_weights = np.ones(len(edgelist_df), dtype=np.float32)
    else:
        sig_pairs = np.empty((0, 2), dtype=np.int32)
        edge_weights = np.empty((0,), dtype=np.float32)

    if len(edge_weights) > 0:
        if cfg.module.weight_clip_min is not None:
            edge_weights = np.maximum(edge_weights, cfg.module.weight_clip_min)
        if cfg.module.weight_clip_max is not None:
            edge_weights = np.minimum(edge_weights, cfg.module.weight_clip_max)
        if cfg.module.normalize_weights:
            w_min = edge_weights.min()
            w_max = edge_weights.max()
            if w_max > w_min:
                edge_weights = (edge_weights - w_min) / (w_max - w_min)

    topology = summarize_network_topology(study_name, adj_sig)
    local_info = {
        f"{study_name}: genes": str(n_genes),
        f"{study_name}: samples": str(n_samples),
        f"{study_name}: resume": "loaded completed outputs",
        f"{study_name}: candidate pairs": f"{len(sig_pairs):,} (from saved edges)",
        f"{study_name}: significant edges": f"{int(np.triu(adj_sig, k=1).sum()):,}",
    }

    return {
        "idx": study_idx,
        "name": study_name,
        "skipped": False,
        "gene_names": gene_names,
        "adj": adj_sig,
        "pairs": sig_pairs,
        "weights": edge_weights,
        "topology": topology,
        "info": local_info,
        "timings": {f"{study_name}: resume load completed outputs": 0.0},
    }


def _process_single_study(study_idx: int, study: dict, cfg: PipelineConfig, device: str) -> dict:
    """Run one full study pipeline and return data required for master merge."""
    study_name = study["name"]
    expr_data = study["expr"]
    gene_names = study["gene_names"]
    n_genes = len(gene_names)
    n_samples = expr_data.shape[1]

    local_info = {
        f"{study_name}: genes": str(n_genes),
        f"{study_name}: samples": str(n_samples),
    }
    local_seed = int(cfg.permutation.seed) + int(study_idx)
    np.random.seed(local_seed)
    local_timings = {}
    paths = _study_artifact_paths(cfg.output_dir, study_name)

    if cfg.resume_completed_studies and _study_is_completed(paths):
        print(f"[RERUN] SKIP completed study {study_name}")
        print(f"\n[INFO] {study_name}: completed outputs found; skipping recompute and loading saved artifacts")
        ensure_mine_diagnostics_plot(study_name, cfg.output_dir)
        return _load_completed_study_payload(study_idx, study, cfg, paths)

    print(f"\n{'=' * 80}")
    print(f"STUDY: {study_name}  ({n_genes:,} genes, {n_samples} samples)")
    print(f"GPU worker device: {device}")
    print("=" * 80)

    t0 = time.time()
    X = zscore_expression(expr_data)
    if cfg.prescreen.enabled:
        pair_indices = prescreen_pairs(
            X,
            method=cfg.prescreen.method,
            threshold=cfg.prescreen.threshold,
            max_pairs=cfg.prescreen.max_pairs,
            n_jobs=cfg.n_jobs,
        )
    else:
        pair_indices = all_pairs(n_genes)
    local_timings[f"{study_name}: Z-score + candidate selection"] = time.time() - t0

    n_cand = len(pair_indices)
    local_info[f"{study_name}: candidate pairs"] = f"{n_cand:,}"
    if n_cand == 0:
        print(f"[WARN] No candidate pairs for {study_name} — skipping")
        return {
            "idx": study_idx,
            "name": study_name,
            "skipped": True,
            "info": local_info,
            "timings": local_timings,
        }

    t0 = time.time()
    mine_diag = []
    using_cache = False
    expected_cache_fp = _mine_cache_fingerprint(cfg, study_name, n_genes)
    if cfg.reuse_mine_scores and os.path.exists(paths["mi_cache"]):
        try:
            cached = np.load(paths["mi_cache"])
            if "n_genes" in cached.files:
                cache_n_genes = int(cached["n_genes"])
            else:
                cache_n_genes = n_genes
            if cache_n_genes != n_genes:
                raise ValueError(
                    f"cached n_genes={cache_n_genes} differs from current n_genes={n_genes}"
                )

            if "cache_fingerprint_json" not in cached.files:
                raise ValueError("cache fingerprint missing (legacy cache)")
            cached_fp_raw = cached["cache_fingerprint_json"]
            if hasattr(cached_fp_raw, "item"):
                cached_fp_raw = cached_fp_raw.item()
            cached_fp = json.loads(str(cached_fp_raw))
            if cached_fp != expected_cache_fp:
                raise ValueError("cache fingerprint mismatch")

            pair_indices = cached["pair_indices"].astype(np.int32, copy=False)
            mi_values = cached["mi_values"].astype(np.float32, copy=False)
            n_cand = len(pair_indices)
            local_info[f"{study_name}: candidate pairs"] = f"{n_cand:,} (from cache)"
            using_cache = True
            print(f"[RERUN] RESUME cached MI for study {study_name}")
            print(f"[INFO] {study_name}: loaded cached MINE scores from {paths['mi_cache']}")
        except Exception as e:
            print(f"[WARN] {study_name}: failed to load cached MINE scores ({e}); recomputing MINE")
            using_cache = False

    if not using_cache:
        print(f"[RERUN] RECOMPUTE study {study_name}")
        mi_values, mine_diag = estimate_mi_for_pairs(
            X,
            pair_indices,
            cfg.mine,
            device,
            verbose=True,
            seed=local_seed,
        )
        save_mine_diagnostics(mine_diag, study_name, cfg.output_dir)
        if cfg.save_mine_score_cache:
            os.makedirs(os.path.dirname(paths["mi_cache"]), exist_ok=True)
            np.savez_compressed(
                paths["mi_cache"],
                pair_indices=pair_indices.astype(np.int32, copy=False),
                mi_values=mi_values.astype(np.float32, copy=False),
                n_genes=np.int32(n_genes),
                cache_fingerprint_json=json.dumps(expected_cache_fp, sort_keys=True),
            )
            print(f"[SAVED] {paths['mi_cache']}")
    pos = mi_values[mi_values > 0]
    local_info[f"{study_name}: MI range"] = (
        f"{pos.min():.4f} – {pos.max():.4f}" if len(pos) > 0 else "all zero"
    )
    if using_cache:
        local_info[f"{study_name}: MINE source"] = "cached scores"
        local_timings[f"{study_name}: MINE MI estimation ({n_cand:,} pairs)"] = 0.0
    else:
        local_info[f"{study_name}: MINE source"] = "fresh training"
        local_timings[f"{study_name}: MINE MI estimation ({n_cand:,} pairs)"] = time.time() - t0

    t0 = time.time()
    if cfg.permutation.mode == "global":
        null_mi = build_global_null(
            X, cfg.mine,
            n_permutations=cfg.permutation.n_permutations,
            seed=local_seed,
            device=device,
        )
        mi_thr = save_null_qc(
            null_mi, study_name,
            cfg.permutation.p_value_threshold, cfg.output_dir,
        )
        p_values = compute_pvalues_global(mi_values, null_mi)
        local_info[f"{study_name}: MI threshold (p<{cfg.permutation.p_value_threshold})"] = (
            f"{mi_thr:.4f}"
        )
    elif cfg.permutation.mode == "per_pair":
        null_mi_pp = build_per_pair_null(
            X, pair_indices, cfg.mine,
            n_permutations=cfg.permutation.n_permutations,
            seed=local_seed,
            device=device,
        )
        p_values = compute_pvalues_per_pair(mi_values, null_mi_pp)
        del null_mi_pp
    else:
        raise ValueError(f"Unknown permutation mode: {cfg.permutation.mode}")
    local_timings[
        f"{study_name}: permutation null ({cfg.permutation.mode}, {cfg.permutation.n_permutations} perms)"
    ] = time.time() - t0

    t0 = time.time()
    sig_mask = p_values < cfg.permutation.p_value_threshold
    adj_sig = filter_edges(
        mi_values, p_values, pair_indices, n_genes,
        p_threshold=cfg.permutation.p_value_threshold,
    )
    n_edges = int(np.triu(adj_sig, k=1).sum())
    local_info[f"{study_name}: significant edges"] = f"{n_edges:,}"

    sig_pairs = pair_indices[sig_mask]
    if cfg.module.master_edge_weight == "mean_neglog10p":
        edge_weights = -np.log10(p_values[sig_mask] + cfg.module.weight_eps)
    elif cfg.module.master_edge_weight == "mean_mi":
        edge_weights = mi_values[sig_mask]
    else:
        edge_weights = np.ones(sig_mask.sum(), dtype=np.float32)

    if len(edge_weights) > 0:
        if cfg.module.weight_clip_min is not None:
            edge_weights = np.maximum(edge_weights, cfg.module.weight_clip_min)
        if cfg.module.weight_clip_max is not None:
            edge_weights = np.minimum(edge_weights, cfg.module.weight_clip_max)
        if cfg.module.normalize_weights:
            w_min = edge_weights.min()
            w_max = edge_weights.max()
            if w_max > w_min:
                edge_weights = (edge_weights - w_min) / (w_max - w_min)
    local_timings[f"{study_name}: edge filtering"] = time.time() - t0

    edgelist_df = build_edgelist(
        adj_sig, pair_indices, mi_values, p_values, gene_names,
    )
    if len(edgelist_df) > 0:
        if cfg.module.master_edge_weight == "mean_neglog10p":
            ew = -np.log10(edgelist_df["p_value"].values + cfg.module.weight_eps)
        elif cfg.module.master_edge_weight == "mean_mi":
            ew = edgelist_df["MI_MINE"].values
        else:
            ew = np.ones(len(edgelist_df), dtype=np.float32)
        edgelist_df["edge_weight"] = ew

    bh_df = None
    if cfg.apply_bh_fdr:
        bh_df = apply_bh_fdr(
            pair_indices, mi_values, p_values, gene_names,
            fdr_alpha=cfg.bh_fdr_alpha,
        )

    t0 = time.time()
    save_study_results(
        study_name, adj_sig, edgelist_df, gene_names,
        cfg.output_dir, bh_df=bh_df,
        make_minimap=cfg.visualization.enabled,
        minimap_base_dir=os.path.join(cfg.output_dir, "minimap_networks"),
        minimap_max_nodes=cfg.visualization.max_nodes,
        minimap_dpi=cfg.visualization.dpi,
        minimap_edge_alpha=cfg.visualization.edge_alpha,
    )
    local_timings[f"{study_name}: saving"] = time.time() - t0

    topology = summarize_network_topology(study_name, adj_sig)

    del mi_values, p_values, X
    gc.collect()

    return {
        "idx": study_idx,
        "name": study_name,
        "skipped": False,
        "gene_names": list(gene_names),
        "adj": adj_sig,
        "pairs": sig_pairs.astype(np.int32, copy=False),
        "weights": np.asarray(edge_weights, dtype=np.float32),
        "topology": topology,
        "info": local_info,
        "timings": local_timings,
    }


def run_pipeline(cfg: PipelineConfig) -> dict:
    """
    Execute the full MINE network inference pipeline.

    Parameters
    ----------
    cfg : PipelineConfig
        Complete configuration (paths, hyper-parameters, etc.).

    Returns
    -------
    dict
        Summary information including timings, study results, and
        master network statistics.
    """
    # ── Setup ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg.output_dir, f"mine_network_{ts}.log")
    report_file = os.path.join(cfg.output_dir, f"analysis_report_{ts}.txt")
    sys.stdout = TeeLogger(log_file)

    timings = {}
    info = {}
    device = "cpu"  # Classical MI runs on CPU only

    # For classical MI, batch_pairs is handled in config defaults (no auto-sizing needed)

    print("=" * 80)
    print("MINE-BASED GENE NETWORK INFERENCE")
    print("Neural MI estimation · Permutation significance · Multi-study consensus")
    print("=" * 80)
    print(f"Timestamp        : {ts}")
    print(f"Device           : {device}")
    print(f"MINE hidden_dim  : {cfg.mine.hidden_dim}")
    print(f"MINE epochs      : {cfg.mine.n_epochs}")
    print(f"Batch pairs      : {cfg.mine.batch_pairs}")
    print(f"Mixed precision  : {'ON (float16)' if cfg.mine.mixed_precision else 'OFF (float32)'}")
    print(f"Resume studies   : {'ON' if cfg.resume_completed_studies else 'OFF'}")
    print(f"Reuse MINE cache : {'ON' if cfg.reuse_mine_scores else 'OFF'}")
    print(f"Pre-screen       : {'ON (' + cfg.prescreen.method + ' |r| > ' + str(cfg.prescreen.threshold) + ')' if cfg.prescreen.enabled else 'OFF (all pairs)'}")
    print(f"Null mode        : {cfg.permutation.mode}")
    print(f"Null permutations: {cfg.permutation.n_permutations}")
    print(f"P-value threshold: {cfg.permutation.p_value_threshold}")
    if cfg.network.min_study_fraction is not None:
        print(f"Min studies mode : fraction ({cfg.network.min_study_fraction:.3f})")
    else:
        print(f"Min studies mode : count ({cfg.network.min_study_count})")
    print(f"Study GPU workers: {cfg.study_gpu_workers}")
    if cfg.study_gpu_devices:
        print(f"Study GPU devices: {', '.join(cfg.study_gpu_devices)}")
    print(f"Module method    : {cfg.module.method}")
    print(f"Submodule method : {cfg.module.submodule_method}")
    print(f"Master edge wt   : {cfg.module.master_edge_weight}")
    print(f"Network minimaps : {'ON' if cfg.visualization.enabled else 'OFF'}")
    if cfg.gene_filter.enabled:
        print("Gene filter      : ON")
        print(f"  ribosomal      : {cfg.gene_filter.remove_ribosomal}")
        print(f"  miRNA          : {cfg.gene_filter.remove_mirna}")
        print(f"  custom regex   : {cfg.gene_filter.custom_regex or 'none'}")
        print(f"  list file      : {cfg.gene_filter.exclude_genes_file or 'none'}")
    else:
        print("Gene filter      : OFF")
    if cfg.qc.mad_top_genes is not None or cfg.qc.plot_pre_filter or cfg.qc.plot_post_filter:
        print("QC/MAD           : ON")
        print(f"  mad top genes  : {cfg.qc.mad_top_genes or 'none'}")
        print(f"  pre plot       : {cfg.qc.plot_pre_filter}")
        print(f"  post plot      : {cfg.qc.plot_post_filter}")
    else:
        print("QC/MAD           : OFF")
    print("=" * 80)

    # ── Step 0: Load data ──
    with Timer("Load expression + metadata", timings):
        expr_full = load_expression(cfg.counts_path)
        metadata = load_metadata(cfg.metadata_path)

        if cfg.gene_filter.enabled:
            expr_full, filt_summary = filter_genes(
                expr_full,
                remove_ribosomal=cfg.gene_filter.remove_ribosomal,
                remove_mirna=cfg.gene_filter.remove_mirna,
                custom_regex=cfg.gene_filter.custom_regex,
                exclude_genes_file=cfg.gene_filter.exclude_genes_file,
            )
            print("[INFO] Gene filtering summary:")
            print(f"  input genes         : {filt_summary['input_genes']:,}")
            print(f"  removed ribosomal   : {filt_summary['removed_ribosomal']:,}")
            print(f"  removed miRNA       : {filt_summary['removed_mirna']:,}")
            print(f"  removed custom regex: {filt_summary['removed_custom_regex']:,}")
            print(f"  removed from file   : {filt_summary['removed_from_file']:,}")
            print(f"  removed total       : {filt_summary['removed_total']:,}")
            print(f"  output genes        : {filt_summary['output_genes']:,}")

        if cfg.qc.plot_pre_filter:
            qc_dir = os.path.join(cfg.output_dir, "qc")
            pre_qc_file = os.path.join(qc_dir, "qc_pre_filter.png")
            save_sample_qc_figure(
                expr_full,
                pre_qc_file,
                title=(f"Pre-filter QC ({expr_full.shape[0]:,} genes × "
                       f"{expr_full.shape[1]:,} samples)"),
                corr_threshold=None,
                n_quantiles=cfg.qc.line_quantiles,
            )

        if cfg.qc.mad_top_genes is not None:
            before_n = expr_full.shape[0]
            expr_full = select_top_genes_by_mad(expr_full, cfg.qc.mad_top_genes)
            info["Genes before MAD filter"] = f"{before_n:,}"
            info["Genes after MAD filter"] = f"{expr_full.shape[0]:,}"

        if cfg.qc.plot_post_filter:
            qc_dir = os.path.join(cfg.output_dir, "qc")
            post_qc_file = os.path.join(qc_dir, "qc_post_filter.png")
            save_sample_qc_figure(
                expr_full,
                post_qc_file,
                title=(f"Post-filter QC ({expr_full.shape[0]:,} genes × "
                       f"{expr_full.shape[1]:,} samples)"),
                corr_threshold=cfg.prescreen.threshold,
                n_quantiles=cfg.qc.line_quantiles,
            )

        studies = discover_studies(
            expr_full, metadata,
            min_samples=cfg.network.min_samples_per_study,
        )

    if not studies:
        print("[ERROR] No studies discovered. Check paths and metadata.")
        sys.exit(1)

    info["Studies"] = str(len(studies))
    info["Expression matrix"] = cfg.counts_path

    # ── Step 1: Process each study ──
    study_payloads = []
    worker_devices = ["cpu"]  # Classical MI always uses CPU
    
    # Sequential study processing (CPU-only, no GPU parallelism)
    for i, study in enumerate(studies):
        study_payloads.append(_process_single_study(i, study, cfg, device))

    # Merge per-study info and timings into global summary dictionaries.
    study_payloads.sort(key=lambda x: x["idx"])
    for payload in study_payloads:
        info.update(payload.get("info", {}))
        timings.update(payload.get("timings", {}))

    # Keep only studies that produced candidate pairs.
    study_payloads = [p for p in study_payloads if not p.get("skipped", False)]

    study_results = []
    study_weight_records = []
    study_network_stats_rows = [p["topology"] for p in study_payloads]

    common_gene_names = list(study_payloads[0]["gene_names"]) if study_payloads else []
    if study_payloads:
        for payload in study_payloads[1:]:
            gene_set = set(payload["gene_names"])
            common_gene_names = [g for g in common_gene_names if g in gene_set]

    for payload in study_payloads:
        gene_names = payload["gene_names"]
        current_gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        idx_curr = [current_gene_to_idx[g] for g in common_gene_names]
        adj_sig = payload["adj"][np.ix_(idx_curr, idx_curr)]
        sig_pairs, edge_weights = _reindex_pairs_to_subset(
            payload["pairs"],
            payload["weights"],
            idx_curr,
            len(gene_names),
        )
        study_results.append({"name": payload["name"], "adj": adj_sig})
        study_weight_records.append({
            "name": payload["name"],
            "pairs": sig_pairs,
            "weights": edge_weights,
        })

    # ── Step 2: Master network ──
    if not study_results:
        print("[ERROR] No study results to combine.")
        sys.exit(1)

    # Save study network connectivity stats by default.
    save_study_network_stats(
        study_network_stats_rows,
        os.path.join(cfg.output_dir, "minimap_networks"),
    )

    n_studies = len(study_results)

    if cfg.network.min_study_fraction is not None:
        effective_min = max(1, round(cfg.network.min_study_fraction * n_studies))
        print(f"\n[INFO] Dynamic min_study_count: "
              f"{cfg.network.min_study_fraction*100:.0f}% of "
              f"{n_studies} = {effective_min}")
    elif n_studies < cfg.network.min_study_count:
        effective_min = 1
        print(f"[WARN] Only {n_studies} studies, setting min_study_count=1")
    else:
        effective_min = cfg.network.min_study_count

    with Timer("Master network construction", timings):
        master_adj, edge_count = build_master_network(
            study_results, common_gene_names, min_count=effective_min,
        )
        master_edge_weight = aggregate_master_weights(
            n_genes=len(common_gene_names),
            study_weight_records=study_weight_records,
            master_adj=master_adj,
            mode=cfg.module.master_edge_weight,
            edge_count=edge_count,
        )
        n_master = int(np.triu(master_adj, k=1).sum())
        info["Master: genes"] = str(len(common_gene_names))
        info["Master: edges"] = f"{n_master:,}"
        if n_master > 0:
            w_vals = master_edge_weight[np.triu(master_adj, k=1) == 1]
            info["Master: edge weight range"] = (
                f"{float(np.min(w_vals)):.4f} – {float(np.max(w_vals)):.4f}"
            )

    # ── Step 3: Module detection ──
    with Timer("Module detection", timings):
        module_min_size = cfg.module.module_min_size

        if cfg.module.method == "leiden":
            modules, membership = leiden_modules(
                master_adj,
                common_gene_names,
                edge_weights=master_edge_weight,
                resolution=cfg.module.module_leiden_resolution,
                n_iterations=cfg.module.module_leiden_iterations,
                min_size=module_min_size,
            )
        else:
            modules, membership = mcode(
                master_adj, common_gene_names,
                score_threshold=cfg.module.module_mcode_score_threshold,
                min_size=module_min_size,
                min_density=cfg.module.module_mcode_min_density,
            )

        if (
            cfg.module.submodule_size_threshold is not None
            and cfg.module.submodule_method != "none"
        ):
            modules, membership, parent_child_rows = refine_large_modules(
                modules,
                master_adj,
                common_gene_names,
                size_threshold=cfg.module.submodule_size_threshold,
                method=cfg.module.submodule_method,
                leiden_resolution=cfg.module.submodule_leiden_resolution,
                leiden_iterations=cfg.module.submodule_leiden_iterations,
                score_threshold=cfg.module.submodule_mcode_score_threshold,
                min_size=cfg.module.submodule_min_size,
                min_density=cfg.module.submodule_mcode_min_density,
            )
        else:
            if (
                cfg.module.submodule_size_threshold is not None
                and cfg.module.submodule_method == "none"
            ):
                print(
                    "[INFO] Submodule refinement skipped "
                    "(--submodule-method none)."
                )
            parent_child_rows = []
        info["Master: modules"] = str(len(modules))

    # ── Step 4: Annotation ──
    # Download GMT files from Enrichr if requested
    if cfg.annotation.download_enrichr:
        with Timer("Download Enrichr gene-set libraries", timings):
            gmt_dir = os.path.join(cfg.output_dir, "gmt_cache")
            downloaded = download_enrichr_libraries(
                cfg.annotation.enrichr_libraries or None,
                cache_dir=gmt_dir,
            )
            cfg.annotation.gmt_paths = list(set(
                cfg.annotation.gmt_paths + downloaded
            ))

    if cfg.annotation.gmt_paths:
        with Timer("Module annotation (enrichment)", timings):
            gene_sets, geneset_to_library, library_manifest = load_multiple_gmt_with_sources(
                cfg.annotation.gmt_paths
            )

            modules_for_annotation = modules
            if cfg.annotation.background_genes:
                with open(cfg.annotation.background_genes, "r") as f:
                    bg_for_annotation = set(line.strip() for line in f if line.strip())
            else:
                bg_for_annotation = set(common_gene_names)

            if cfg.annotation.ortholog_map_path:
                print("[INFO] Applying ortholog mapping before enrichment:")
                print(f"  map file : {cfg.annotation.ortholog_map_path}")
                print(f"  source   : {cfg.annotation.ortholog_source_col}")
                print(f"  target   : {cfg.annotation.ortholog_target_col}")

                orth_map = load_ortholog_map(
                    cfg.annotation.ortholog_map_path,
                    cfg.annotation.ortholog_source_col,
                    cfg.annotation.ortholog_target_col,
                )
                modules_for_annotation, map_stats = map_modules(modules, orth_map)
                bg_for_annotation, bg_mapped = map_gene_set(bg_for_annotation, orth_map)

                print("[INFO] Ortholog mapping summary:")
                print(f"  module source genes  : {map_stats['source_unique_genes']:,}")
                print(f"  module genes mapped  : {map_stats['source_genes_mapped']:,}")
                print(f"  module target genes  : {map_stats['mapped_unique_genes']:,}")
                print(f"  background mapped    : {bg_mapped:,} -> {len(bg_for_annotation):,}")

            annotation_df = annotate_modules(
                modules_for_annotation,
                gene_sets,
                bg_for_annotation,
                fdr_threshold=cfg.annotation.fdr_threshold,
                min_overlap=cfg.annotation.min_overlap,
            )
            if not annotation_df.empty:
                annotation_df["GeneSetLibrary"] = annotation_df["GeneSet"].map(
                    geneset_to_library
                ).fillna("unknown")
            save_annotations(annotation_df, cfg.output_dir)
            if cfg.annotation.save_per_gmt_results:
                save_annotations_by_source(
                    annotation_df,
                    cfg.output_dir,
                    library_manifest=library_manifest,
                )
            info["Annotations"] = str(len(annotation_df))
    else:
        print("\n[INFO] No GMT gene-set files provided — skipping annotation. "
              "Set cfg.annotation.gmt_paths to enable enrichment analysis.")

    # ── Step 5: Save master ──
    with Timer("Save master network", timings):
        save_master_results(
            master_adj, edge_count, master_edge_weight,
            common_gene_names,
            modules, membership, parent_child_rows,
            min_count=effective_min, n_studies=n_studies,
            output_dir=cfg.output_dir,
            module_export_map_path=cfg.annotation.module_export_map_path,
            module_export_key_col=cfg.annotation.module_export_key_col,
            module_export_cols=cfg.annotation.module_export_cols,
            make_minimap=cfg.visualization.enabled,
            minimap_base_dir=os.path.join(cfg.output_dir, "minimap_networks"),
            minimap_max_nodes=cfg.visualization.max_nodes,
            minimap_dpi=cfg.visualization.dpi,
            minimap_edge_alpha=cfg.visualization.edge_alpha,
        )

    # ── Summary ──
    total = sum(timings.values())
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 80}")
    print(f"Runtime: {format_time(total)}")
    print(f"Output:  {cfg.output_dir}")

    print("\nTIMING SUMMARY")
    print("-" * 80)
    for step, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = t / total * 100 if total > 0 else 0
        print(f"  {step:55s}: {format_time(t):>20s} ({pct:5.1f}%)")

    save_report(timings, info, report_file)
    print("=" * 80)

    return {"timings": timings, "info": info, "modules": modules}

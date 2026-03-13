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
import numpy as np
import torch
from datetime import datetime

from .config import PipelineConfig
from .data_loader import load_expression, load_metadata, discover_studies, zscore_expression, filter_genes
from .mine_estimator import estimate_mi_for_pairs
from .permutation import (
    build_global_null, build_per_pair_null,
    compute_pvalues_global, compute_pvalues_per_pair,
)
from .prescreen import prescreen_pairs, all_pairs
from .network import filter_edges, build_edgelist, apply_bh_fdr, build_master_network
from .mcode import mcode
from .annotation import load_multiple_gmt, annotate_modules, save_annotations, download_enrichr_libraries
from .io_utils import (
    TeeLogger, Timer, format_time,
    save_null_qc, save_study_results, save_master_results, save_report,
    save_mine_diagnostics,
)


def _resolve_device(cfg: PipelineConfig) -> torch.device:
    """Resolve 'auto' to the best available device."""
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


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
    device = _resolve_device(cfg)

    print("=" * 80)
    print("MINE-BASED GENE NETWORK INFERENCE")
    print("Neural MI estimation · Permutation significance · Multi-study consensus")
    print("=" * 80)
    print(f"Timestamp        : {ts}")
    print(f"Device           : {device}")
    print(f"MINE hidden_dim  : {cfg.mine.hidden_dim}")
    print(f"MINE epochs      : {cfg.mine.n_epochs}")
    print(f"Batch pairs      : {cfg.mine.batch_pairs}")
    print(f"Pre-screen       : {'ON (' + cfg.prescreen.method + ' |r| > ' + str(cfg.prescreen.threshold) + ')' if cfg.prescreen.enabled else 'OFF (all pairs)'}")
    print(f"Null mode        : {cfg.permutation.mode}")
    print(f"Null permutations: {cfg.permutation.n_permutations}")
    print(f"P-value threshold: {cfg.permutation.p_value_threshold}")
    print(f"Min studies      : {cfg.network.min_study_count}")
    if cfg.gene_filter.enabled:
        print("Gene filter      : ON")
        print(f"  ribosomal      : {cfg.gene_filter.remove_ribosomal}")
        print(f"  miRNA          : {cfg.gene_filter.remove_mirna}")
        print(f"  custom regex   : {cfg.gene_filter.custom_regex or 'none'}")
        print(f"  list file      : {cfg.gene_filter.exclude_genes_file or 'none'}")
    else:
        print("Gene filter      : OFF")
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
    study_results = []
    study_mi_values = {}  # for optional mean-MI edge weighting
    common_gene_names = None

    for study in studies:
        study_name = study["name"]
        expr_data = study["expr"]
        gene_names = study["gene_names"]
        n_genes = len(gene_names)
        n_samples = expr_data.shape[1]

        print(f"\n{'=' * 80}")
        print(f"STUDY: {study_name}  ({n_genes:,} genes, {n_samples} samples)")
        print("=" * 80)

        info[f"{study_name}: genes"] = str(n_genes)
        info[f"{study_name}: samples"] = str(n_samples)

        # ── Z-score ──
        with Timer(f"{study_name}: Z-score + candidate selection", timings):
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

            n_cand = len(pair_indices)
            info[f"{study_name}: candidate pairs"] = f"{n_cand:,}"

        if n_cand == 0:
            print(f"[WARN] No candidate pairs for {study_name} — skipping")
            continue

        # ── MINE MI estimation ──
        with Timer(f"{study_name}: MINE MI estimation ({n_cand:,} pairs)", timings):
            mi_values, mine_diag = estimate_mi_for_pairs(
                X, pair_indices, cfg.mine, device, verbose=True,
            )
            pos = mi_values[mi_values > 0]
            info[f"{study_name}: MI range"] = (
                f"{pos.min():.4f} – {pos.max():.4f}" if len(pos) > 0
                else "all zero"
            )
            # Save MINE training diagnostics
            save_mine_diagnostics(mine_diag, study_name, cfg.output_dir)

        # ── Permutation null + p-values ──
        with Timer(f"{study_name}: permutation null ({cfg.permutation.mode}, "
                   f"{cfg.permutation.n_permutations} perms)", timings):

            if cfg.permutation.mode == "global":
                null_mi = build_global_null(
                    X, cfg.mine,
                    n_permutations=cfg.permutation.n_permutations,
                    seed=cfg.permutation.seed,
                    device=device,
                )
                mi_thr = save_null_qc(
                    null_mi, study_name,
                    cfg.permutation.p_value_threshold, cfg.output_dir,
                )
                p_values = compute_pvalues_global(mi_values, null_mi)
                info[f"{study_name}: MI threshold (p<{cfg.permutation.p_value_threshold})"] = (
                    f"{mi_thr:.4f}"
                )

            elif cfg.permutation.mode == "per_pair":
                null_mi_pp = build_per_pair_null(
                    X, pair_indices, cfg.mine,
                    n_permutations=cfg.permutation.n_permutations,
                    seed=cfg.permutation.seed,
                    device=device,
                )
                p_values = compute_pvalues_per_pair(mi_values, null_mi_pp)
                del null_mi_pp

            else:
                raise ValueError(f"Unknown permutation mode: {cfg.permutation.mode}")

        # ── Filter edges ──
        with Timer(f"{study_name}: edge filtering", timings):
            adj_sig = filter_edges(
                mi_values, p_values, pair_indices, n_genes,
                p_threshold=cfg.permutation.p_value_threshold,
            )
            n_edges = int(np.triu(adj_sig, k=1).sum())
            info[f"{study_name}: significant edges"] = f"{n_edges:,}"

        # Edge list
        edgelist_df = build_edgelist(
            adj_sig, pair_indices, mi_values, p_values, gene_names,
        )

        # Optional BH-FDR
        bh_df = None
        if cfg.apply_bh_fdr:
            bh_df = apply_bh_fdr(
                pair_indices, mi_values, p_values, gene_names,
                fdr_alpha=cfg.bh_fdr_alpha,
            )

        # Save per-study
        with Timer(f"{study_name}: saving", timings):
            save_study_results(
                study_name, adj_sig, edgelist_df, gene_names,
                cfg.output_dir, bh_df=bh_df,
            )

        # ── Track for master network ──
        if common_gene_names is None:
            common_gene_names = list(gene_names)
            study_results.append({"name": study_name, "adj": adj_sig})
            study_mi_values[study_name] = (pair_indices, mi_values)
        else:
            # Intersect gene names
            common_set = set(common_gene_names) & set(gene_names)
            if len(common_set) < len(common_gene_names):
                print(f"[WARN] Gene intersection: {len(common_gene_names)} → "
                      f"{len(common_set)}")
                old_idx = [
                    common_gene_names.index(g) for g in common_gene_names
                    if g in common_set
                ]
                common_gene_names = [
                    g for g in common_gene_names if g in common_set
                ]
                for prev in study_results:
                    prev["adj"] = prev["adj"][np.ix_(old_idx, old_idx)]
                idx_curr = [gene_names.index(g) for g in common_gene_names]
                adj_sig = adj_sig[np.ix_(idx_curr, idx_curr)]
            study_results.append({"name": study_name, "adj": adj_sig})
            study_mi_values[study_name] = (pair_indices, mi_values)

        # Free memory
        del mi_values, p_values, X
        gc.collect()

    # ── Step 2: Master network ──
    if not study_results:
        print("[ERROR] No study results to combine.")
        sys.exit(1)

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
        n_master = int(np.triu(master_adj, k=1).sum())
        info["Master: genes"] = str(len(common_gene_names))
        info["Master: edges"] = f"{n_master:,}"

    # ── Step 3: MCODE ──
    with Timer("MCODE module detection", timings):
        modules, membership = mcode(
            master_adj, common_gene_names,
            score_threshold=cfg.mcode.score_threshold,
            min_size=cfg.mcode.min_size,
            min_density=cfg.mcode.min_density,
        )
        info["Master: MCODE modules"] = str(len(modules))

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
            gene_sets = load_multiple_gmt(cfg.annotation.gmt_paths)

            if cfg.annotation.background_genes:
                with open(cfg.annotation.background_genes, "r") as f:
                    bg = set(line.strip() for line in f if line.strip())
            else:
                bg = set(common_gene_names)

            annotation_df = annotate_modules(
                modules, gene_sets, bg,
                fdr_threshold=cfg.annotation.fdr_threshold,
                min_overlap=cfg.annotation.min_overlap,
            )
            save_annotations(annotation_df, cfg.output_dir)
            info["Annotations"] = str(len(annotation_df))
    else:
        print("\n[INFO] No GMT gene-set files provided — skipping annotation. "
              "Set cfg.annotation.gmt_paths to enable enrichment analysis.")

    # ── Step 5: Save master ──
    with Timer("Save master network", timings):
        save_master_results(
            master_adj, edge_count, common_gene_names,
            modules, membership,
            min_count=effective_min, n_studies=n_studies,
            output_dir=cfg.output_dir,
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

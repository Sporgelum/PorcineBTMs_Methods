#!/usr/bin/env python3
"""
Process individual studies in parallel (via SLURM) + consolidate results.

Runs the MINE pipeline per BioProject, then merges per-study adjacency
matrices into a master consensus network.
"""

import sys
import os
import json
import argparse
import copy
from pathlib import Path
from datetime import datetime
from dataclasses import is_dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread, mmwrite

from mine_network.data_loader import (
    load_expression, load_metadata, discover_studies
)
from mine_network.pipeline import run_pipeline as run_full_pipeline
from mine_network.config import PipelineConfig
from mine_network.network import build_master_network, aggregate_master_weights
from mine_network.mcode import mcode, leiden_modules, refine_large_modules
from mine_network.annotation import (
    load_multiple_gmt_with_sources,
    annotate_modules,
    save_annotations,
    save_annotations_by_source,
    download_enrichr_libraries,
)
from mine_network.ortholog import load_ortholog_map, map_modules, map_gene_set
from mine_network.io_utils import save_master_results


def _update_dataclass_from_dict(obj, values):
    """Recursively update a dataclass-like config object from a dict."""
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass_from_dict(current, value)
        else:
            setattr(obj, key, value)


def process_single_study(study_idx, study_info, cfg, output_dir, full_metadata=None):
    """
    Process one study (BioProject) with the MINE pipeline.
    
    Parameters:
    -----------
    study_idx : int
        Study index (0-based)
    study_info : dict
        Study info dict with 'name', 'expr' (DataFrame samples × genes), 'gene_names'
    cfg : PipelineConfig
        Full pipeline configuration
    output_dir : str
        Where to save results
    full_metadata : pd.DataFrame, optional
        Full metadata to filter for this study's samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"study_{study_idx}.log")
    log_fh = open(log_file, 'w')
    
    def log_msg(msg=""):
        print(msg)
        log_fh.write(msg + '\n')
        log_fh.flush()
    
    try:
        # Extract data from study_info
        expr_df = study_info['expr']  # DataFrame with genes × samples
        sample_ids = expr_df.columns.tolist()  # Column names are sample IDs
        
        # Filter metadata for this study's samples
        if full_metadata is not None:
            meta = full_metadata[full_metadata['Run'].isin(sample_ids)].copy()
        else:
            meta = pd.DataFrame({'Run': sample_ids})
        
        log_msg("=" * 80)
        log_msg(f"STUDY {study_idx}: {study_info['name']}")
        log_msg("=" * 80)
        log_msg(f"Samples: {expr_df.shape[1]}, Genes: {expr_df.shape[0]}")
        log_msg()
        
        # Create temporary study-specific data files
        study_counts_file = os.path.join(output_dir, f"study_{study_idx}_counts.csv")
        study_meta_file = os.path.join(output_dir, f"study_{study_idx}_meta.csv")
        study_genes_file = os.path.join(output_dir, f"genes_{study_idx}.txt")
        study_output_dir = os.path.join(output_dir, f"study_{study_idx}_results")
        
        # Save study-specific count matrix (genes × samples, tab-separated)
        expr_df.to_csv(study_counts_file, sep="\t")
        log_msg(f"Saved counts: {study_counts_file}")
        
        # Save study-specific metadata
        meta.to_csv(study_meta_file, sep="\t", index=False)
        log_msg(f"Saved metadata: {study_meta_file}")

        # Save ordered gene list for consolidation alignment
        with open(study_genes_file, "w") as f:
            for gene in expr_df.index:
                f.write(f"{gene}\n")
        log_msg(f"Saved genes: {study_genes_file}")
        log_msg()
        
        # Create config for this study
        cfg_study = copy.deepcopy(cfg)
        cfg_study.counts_path = study_counts_file
        cfg_study.metadata_path = study_meta_file
        cfg_study.output_dir = study_output_dir

        # If n_jobs is not explicitly set, honor the scheduler allocation.
        # This avoids accidentally underusing large CPU allocations.
        if int(getattr(cfg_study, "n_jobs", -1)) <= 0:
            slurm_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "0") or 0)
            host_cpus = int(os.cpu_count() or 1)
            cfg_study.n_jobs = max(1, min(host_cpus, slurm_cpus if slurm_cpus > 0 else host_cpus))
        os.environ["OMP_NUM_THREADS"] = str(max(1, int(cfg_study.n_jobs)))
        os.environ["MKL_NUM_THREADS"] = str(max(1, int(cfg_study.n_jobs)))
        os.environ["OPENBLAS_NUM_THREADS"] = str(max(1, int(cfg_study.n_jobs)))

        # Keep annotation/visualization settings from config so study-level
        # pathway enrichment and minimaps are produced when requested.
        
        # Run pipeline for this study
        log_msg(
            f"Running MI pipeline (n_jobs={cfg_study.n_jobs}, "
            f"download_gmt={cfg_study.annotation.download_enrichr}, "
            f"save_per_gmt={cfg_study.annotation.save_per_gmt_results}, "
            f"network_viz={cfg_study.visualization.enabled})..."
        )
        results = run_full_pipeline(cfg_study)
        
        # Copy canonical per-study artifacts to stable index-based names.
        # run_pipeline writes names with study IDs (e.g., PRJNA...), so we glob.
        import shutil
        artifact_patterns = [
            ("edges_mine_*.tsv", f"edges_mine_{study_idx}.tsv"),
            ("adj_mine_*.mtx", f"adj_mine_{study_idx}.mtx"),
            ("network_mine_*.graphml", f"network_mine_{study_idx}.graphml"),
        ]
        for pattern, out_name in artifact_patterns:
            matches = sorted(Path(study_output_dir).glob(pattern))
            if not matches:
                log_msg(f"[WARN] No artifact matched: {pattern}")
                continue
            src = str(matches[0])
            dst = os.path.join(output_dir, out_name)
            shutil.copy(src, dst)
            log_msg(f"Copied: {dst}")
        
        log_msg()
        log_msg("✓ Study complete")
        log_msg()
        
        return True
        
    except Exception as e:
        log_msg(f"\n✗ ERROR: {e}")
        import traceback
        log_msg(traceback.format_exc())
        return False
    finally:
        log_fh.close()


def consolidate_studies(studies_dir, output_dir, counts_file, meta_file, config_file):
    """
    Merge all per-study results into master network.
    
    Parameters:
    -----------
    studies_dir : str
        Directory containing study_* subdirectories
    output_dir : str
        Where to save master network
    counts_file : str
        Original full expression matrix
    meta_file : str
        Original full metadata
    config_file : str
        Config JSON with min_studies, etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"consolidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_fh = open(log_file, 'w')
    
    def log_msg(msg=""):
        print(msg)
        log_fh.write(msg + '\n')
        log_fh.flush()
    
    try:
        log_msg("=" * 80)
        log_msg("CONSOLIDATION: Merging per-study results")
        log_msg("=" * 80)
        log_msg(f"Studies dir: {studies_dir}")
        log_msg(f"Output dir:  {output_dir}")
        log_msg()
        
        # Load configuration
        with open(config_file, 'r') as f:
            cfg_dict = json.load(f)
        cfg = PipelineConfig()
        _update_dataclass_from_dict(cfg, cfg_dict)
        
        # Discover per-study results
        studies_path = Path(studies_dir)
        study_dirs = sorted([d for d in studies_path.iterdir() if d.is_dir() and d.name.startswith('study_')])
        
        if not study_dirs:
            log_msg("[ERROR] No study directories found")
            return False
        
        log_msg(f"Found {len(study_dirs)} study directories")
        log_msg()
        
        # Load all adjacency matrices
        log_msg("Loading per-study adjacency matrices...")
        study_results = []
        gene_names_list = []
        
        for study_dir in study_dirs:
            study_num = study_dir.name.replace('study_', '')
            adj_file = study_dir / f"adj_mine_{study_num}.mtx"
            genes_file = study_dir / f"genes_{study_num}.txt"
            edges_file = study_dir / f"edges_mine_{study_num}.tsv"
            
            if not adj_file.exists():
                log_msg(f"  Skipping {study_num} (no adj matrix)")
                continue
            
            adj = mmread(str(adj_file)).toarray().astype(np.uint8)
            
            # Load gene names (or use generic)
            if genes_file.exists():
                with open(genes_file) as f:
                    genes = [line.strip() for line in f]
            else:
                genes = [f"Gene_{i}" for i in range(adj.shape[0])]
            gene_names_list.append(genes)

            edge_df = None
            if edges_file.exists():
                try:
                    edge_df = pd.read_csv(edges_file, sep="\t")
                except Exception as e:
                    log_msg(f"  [WARN] Could not read {edges_file.name}: {e}")

            study_results.append({
                "name": f"study_{study_num}",
                "adj": adj,
                "genes": genes,
                "edge_df": edge_df,
            })
            
            n_edges = int(np.triu(adj, k=1).sum())
            log_msg(f"  Study {study_num}: {adj.shape[0]} genes, {n_edges} edges")
        
        if not study_results:
            log_msg("[ERROR] No adjacency matrices loaded")
            return False
        
        log_msg()
        
        # Find common genes
        log_msg("Finding common genes...")
        common_genes = set(gene_names_list[0])
        for genes in gene_names_list[1:]:
            common_genes &= set(genes)
        common_genes = sorted(list(common_genes))
        if not common_genes:
            log_msg("[ERROR] No common genes across loaded studies")
            return False
        log_msg(f"Common genes: {len(common_genes)}")
        log_msg()
        
        # Reindex and build master
        log_msg("Building master network...")
        adj_common = []
        common_gene_to_idx = {g: i for i, g in enumerate(common_genes)}
        common_gene_set = set(common_genes)
        study_weight_records = []

        for i, res in enumerate(study_results):
            adj = res["adj"]
            genes = gene_names_list[i]
            gene_to_idx = {g: j for j, g in enumerate(genes)}
            idx = [gene_to_idx[g] for g in common_genes]
            adj_sub = adj[np.ix_(idx, idx)]
            adj_common.append({"name": res["name"], "adj": adj_sub})

            rec_df = res.get("edge_df")
            if rec_df is None or rec_df.empty:
                study_weight_records.append({
                    "pairs": np.empty((0, 2), dtype=np.int32),
                    "weights": np.empty((0,), dtype=np.float32),
                })
                continue

            if "gene_A" not in rec_df.columns or "gene_B" not in rec_df.columns:
                study_weight_records.append({
                    "pairs": np.empty((0, 2), dtype=np.int32),
                    "weights": np.empty((0,), dtype=np.float32),
                })
                continue

            weight_mode = str(cfg.module.master_edge_weight)
            if weight_mode == "mean_neglog10p" and "p_value" in rec_df.columns:
                weights = -np.log10(rec_df["p_value"].to_numpy(np.float32) + float(cfg.module.weight_eps))
            elif weight_mode == "mean_mi" and "MI_MINE" in rec_df.columns:
                weights = rec_df["MI_MINE"].to_numpy(np.float32)
            elif "edge_weight" in rec_df.columns:
                weights = rec_df["edge_weight"].to_numpy(np.float32)
            else:
                weights = np.ones(len(rec_df), dtype=np.float32)

            if len(weights) > 0:
                if cfg.module.weight_clip_min is not None:
                    weights = np.maximum(weights, float(cfg.module.weight_clip_min))
                if cfg.module.weight_clip_max is not None:
                    weights = np.minimum(weights, float(cfg.module.weight_clip_max))
                if cfg.module.normalize_weights:
                    w_min = float(weights.min())
                    w_max = float(weights.max())
                    if w_max > w_min:
                        weights = (weights - w_min) / (w_max - w_min)

            keep = rec_df["gene_A"].isin(common_gene_set) & rec_df["gene_B"].isin(common_gene_set)
            if not keep.any():
                study_weight_records.append({
                    "pairs": np.empty((0, 2), dtype=np.int32),
                    "weights": np.empty((0,), dtype=np.float32),
                })
                continue

            rec_f = rec_df.loc[keep, ["gene_A", "gene_B"]].copy()
            w_f = np.asarray(weights, dtype=np.float32)[keep.to_numpy()]

            pair_arr = np.column_stack([
                rec_f["gene_A"].map(common_gene_to_idx).to_numpy(np.int32),
                rec_f["gene_B"].map(common_gene_to_idx).to_numpy(np.int32),
            ])
            pair_arr = np.sort(pair_arr, axis=1)

            study_weight_records.append({
                "pairs": pair_arr,
                "weights": w_f,
            })

        n_loaded_studies = len(adj_common)
        if n_loaded_studies == 0:
            log_msg("[ERROR] No valid studies loaded for consolidation")
            return False

        if cfg.network.min_study_fraction is not None:
            min_studies = max(1, round(float(cfg.network.min_study_fraction) * n_loaded_studies))
            log_msg(
                f"Using min_study_fraction={cfg.network.min_study_fraction} -> "
                f"effective min_studies={min_studies}"
            )
        elif n_loaded_studies < int(cfg.network.min_study_count):
            min_studies = 1
            log_msg(
                f"[WARN] Only {n_loaded_studies} studies loaded; using min_studies=1 "
                f"(config requested {cfg.network.min_study_count})"
            )
        else:
            min_studies = int(cfg.network.min_study_count)

        adj_master, study_counts = build_master_network(
            adj_common,
            common_genes,
            min_count=min_studies,
        )

        master_edge_weight = aggregate_master_weights(
            n_genes=len(common_genes),
            study_weight_records=study_weight_records,
            master_adj=adj_master,
            mode=cfg.module.master_edge_weight,
            edge_count=study_counts,
        )
        
        n_master_edges = int(np.triu(adj_master, k=1).sum())
        log_msg(f"Master network: {adj_master.shape[0]} nodes, {n_master_edges} edges")
        log_msg()
        
        # Master module detection
        log_msg("Running master module detection...")
        module_min_size = int(cfg.module.module_min_size)
        if cfg.module.method == "leiden":
            modules, membership = leiden_modules(
                adj_master,
                common_genes,
                edge_weights=master_edge_weight,
                resolution=float(cfg.module.module_leiden_resolution),
                n_iterations=int(cfg.module.module_leiden_iterations),
                min_size=module_min_size,
            )
        else:
            modules, membership = mcode(
                adj_master,
                common_genes,
                score_threshold=float(cfg.module.module_mcode_score_threshold),
                min_size=module_min_size,
                min_density=float(cfg.module.module_mcode_min_density),
            )

        if cfg.module.submodule_size_threshold is not None and cfg.module.submodule_method != "none":
            modules, membership, parent_child_rows = refine_large_modules(
                modules,
                adj_master,
                common_genes,
                size_threshold=int(cfg.module.submodule_size_threshold),
                method=str(cfg.module.submodule_method),
                leiden_resolution=float(cfg.module.submodule_leiden_resolution),
                leiden_iterations=int(cfg.module.submodule_leiden_iterations),
                score_threshold=float(cfg.module.submodule_mcode_score_threshold),
                min_size=int(cfg.module.submodule_min_size),
                min_density=float(cfg.module.submodule_mcode_min_density),
            )
        else:
            parent_child_rows = []

        # Master annotation
        if cfg.annotation.download_enrichr:
            log_msg("Downloading GMT libraries for master annotation...")
            gmt_dir = os.path.join(output_dir, "gmt_cache")
            downloaded = download_enrichr_libraries(
                cfg.annotation.enrichr_libraries or None,
                cache_dir=gmt_dir,
            )
            cfg.annotation.gmt_paths = list(set((cfg.annotation.gmt_paths or []) + downloaded))

        if cfg.annotation.gmt_paths:
            log_msg("Annotating master modules...")
            gene_sets, geneset_to_library, library_manifest = load_multiple_gmt_with_sources(
                cfg.annotation.gmt_paths
            )

            modules_for_annotation = modules
            if cfg.annotation.background_genes:
                with open(cfg.annotation.background_genes, "r") as f:
                    bg_for_annotation = set(line.strip() for line in f if line.strip())
            else:
                bg_for_annotation = set(common_genes)

            if cfg.annotation.ortholog_map_path:
                log_msg("Applying ortholog mapping for master annotation...")
                orth_map = load_ortholog_map(
                    cfg.annotation.ortholog_map_path,
                    cfg.annotation.ortholog_source_col,
                    cfg.annotation.ortholog_target_col,
                )
                modules_for_annotation, map_stats = map_modules(modules, orth_map)
                bg_for_annotation, bg_mapped = map_gene_set(bg_for_annotation, orth_map)
                log_msg(
                    "  Ortholog map: "
                    f"source={map_stats['source_unique_genes']:,}, "
                    f"mapped={map_stats['source_genes_mapped']:,}, "
                    f"target={map_stats['mapped_unique_genes']:,}, "
                    f"background={bg_mapped:,}->{len(bg_for_annotation):,}"
                )

            annotation_df = annotate_modules(
                modules_for_annotation,
                gene_sets,
                bg_for_annotation,
                fdr_threshold=float(cfg.annotation.fdr_threshold),
                min_overlap=int(cfg.annotation.min_overlap),
            )
            if not annotation_df.empty:
                annotation_df["GeneSetLibrary"] = annotation_df["GeneSet"].map(
                    geneset_to_library
                ).fillna("unknown")
            save_annotations(annotation_df, output_dir)
            if cfg.annotation.save_per_gmt_results:
                save_annotations_by_source(
                    annotation_df,
                    output_dir,
                    library_manifest=library_manifest,
                )
        else:
            log_msg(
                "No GMT gene-set files provided for master annotation; "
                "skipping enrichment."
            )

        # Save master network + module outputs
        log_msg("Saving master network outputs...")
        save_master_results(
            master_adj=adj_master,
            edge_count=study_counts,
            master_edge_weight=master_edge_weight,
            gene_names=common_genes,
            modules=modules,
            membership=membership,
            parent_child_rows=parent_child_rows,
            min_count=min_studies,
            n_studies=n_loaded_studies,
            output_dir=output_dir,
            module_export_map_path=cfg.annotation.module_export_map_path,
            module_export_key_col=cfg.annotation.module_export_key_col,
            module_export_cols=cfg.annotation.module_export_cols,
            make_minimap=bool(cfg.visualization.enabled),
            minimap_base_dir=os.path.join(output_dir, "minimap_networks"),
            minimap_max_nodes=int(cfg.visualization.max_nodes),
            minimap_dpi=int(cfg.visualization.dpi),
            minimap_edge_alpha=float(cfg.visualization.edge_alpha),
        )

        # Save gene names for traceability
        genes_out = os.path.join(output_dir, "master_network_genes.txt")
        with open(genes_out, 'w') as f:
            for gene in common_genes:
                f.write(f"{gene}\n")
        log_msg(f"  ✓ {genes_out}")
        
        log_msg()
        log_msg("=" * 80)
        log_msg("✓ CONSOLIDATION COMPLETE")
        log_msg("=" * 80)
        
        return True
        
    except Exception as e:
        log_msg(f"\n✗ ERROR: {e}")
        import traceback
        log_msg(traceback.format_exc())
        return False
    finally:
        log_fh.close()


def main():
    parser = argparse.ArgumentParser(description="Process studies per-SLURM + consolidate")
    
    # Single-study mode
    parser.add_argument("--study-index", type=int, default=None,
                        help="Study index to process (for SLURM tasks)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for this study")
    
    # Consolidation mode
    parser.add_argument("--consolidate", action="store_true",
                        help="Consolidate per-study results")
    parser.add_argument("--studies-dir", type=str, default=None,
                        help="Directory with study_* subdirectories")
    parser.add_argument("--counts", type=str, default=None,
                        help="Full counts file (for reference)")
    parser.add_argument("--meta", type=str, default=None,
                        help="Full metadata file (for reference)")
    
    args = parser.parse_args()
    
    if args.consolidate:
        # Consolidation mode
        if not args.studies_dir or not args.output_dir:
            print("[ERROR] --consolidate requires --studies-dir and --output-dir")
            sys.exit(1)
        
        config_file = os.path.join(args.output_dir, "config.json") if os.path.exists(
            os.path.join(args.output_dir, "config.json")
        ) else os.path.join(str(Path(__file__).parent), "config.json")
        
        success = consolidate_studies(
            args.studies_dir, args.output_dir,
            args.counts, args.meta, config_file
        )
        sys.exit(0 if success else 1)
    
    else:
        # Single-study mode
        if args.study_index is None or not args.config or not args.output_dir:
            print("[ERROR] Single-study mode requires --study-index, --config, --output-dir")
            sys.exit(1)
        
        # Load config
        with open(args.config) as f:
            cfg_dict = json.load(f)
        
        cfg = PipelineConfig()
        _update_dataclass_from_dict(cfg, cfg_dict)
        
        # Load data
        print(f"Loading data...")
        expr = load_expression(cfg.counts_path)
        meta = load_metadata(cfg.metadata_path)
        
        # Discover studies
        studies = discover_studies(
            expr,
            meta,
            min_samples=cfg.network.min_samples_per_study,
        )
        
        if args.study_index >= len(studies):
            print(f"[ERROR] Study index {args.study_index} out of range")
            sys.exit(1)
        
        study = studies[args.study_index]
        print(f"Processing {study['name']} (study {args.study_index})")
        
        # Process
        success = process_single_study(args.study_index, study, cfg, args.output_dir, full_metadata=meta)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

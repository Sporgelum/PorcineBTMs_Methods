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
from mine_network.network import build_master_network


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

        # Keep per-study workers bounded to avoid OOM from oversubscription.
        if int(getattr(cfg_study, "n_jobs", -1)) <= 0:
            cfg_study.n_jobs = min(8, os.cpu_count() or 8)
        os.environ["OMP_NUM_THREADS"] = str(max(1, int(cfg_study.n_jobs)))
        os.environ["MKL_NUM_THREADS"] = str(max(1, int(cfg_study.n_jobs)))
        os.environ["OPENBLAS_NUM_THREADS"] = str(max(1, int(cfg_study.n_jobs)))

        # Per-study enrichment/visualization is expensive and not required for consolidation.
        cfg_study.annotation.download_enrichr = False
        cfg_study.annotation.save_per_gmt_results = False
        cfg_study.visualization.enabled = False
        
        # Run pipeline for this study
        log_msg(f"Running MI pipeline (n_jobs={cfg_study.n_jobs})...")
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
        min_studies = cfg_dict.get('network', {}).get('min_study_count', 3)
        
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

            study_results.append({"name": f"study_{study_num}", "adj": adj})
            
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
        for i, res in enumerate(study_results):
            adj = res["adj"]
            genes = gene_names_list[i]
            idx = [genes.index(g) for g in common_genes]
            adj_sub = adj[np.ix_(idx, idx)]
            adj_common.append({"name": res["name"], "adj": adj_sub})
        
        adj_master, study_counts = build_master_network(
            adj_common,
            common_genes,
            min_count=min_studies,
        )
        
        n_master_edges = int(np.triu(adj_master, k=1).sum())
        log_msg(f"Master network: {adj_master.shape[0]} nodes, {n_master_edges} edges")
        log_msg()
        
        # Save master network files
        log_msg("Saving master network...")
        
        # Adjacency matrix
        adj_sparse = csr_matrix(adj_master)
        adj_out = os.path.join(output_dir, "master_network_adjacency.mtx")
        mmwrite(adj_out, adj_sparse)
        log_msg(f"  ✓ {adj_out}")
        
        # Edge list
        rows, cols = np.where(np.triu(adj_master, k=1) == 1)
        edges_df = pd.DataFrame({
            "gene_A": np.array(common_genes)[rows],
            "gene_B": np.array(common_genes)[cols],
            "n_studies": study_counts[rows, cols],
        }).sort_values("n_studies", ascending=False)
        edges_out = os.path.join(output_dir, "master_network_edgelist.tsv")
        edges_df.to_csv(edges_out, sep="\t", index=False)
        log_msg(f"  ✓ {edges_out} ({len(edges_df)} edges)")
        
        # Gene names
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

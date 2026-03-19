#!/usr/bin/env python3
"""Run module annotation only on an existing master network output folder."""

import argparse
import importlib.util
from pathlib import Path


def _load_annotation_module():
    here = Path(__file__).resolve().parent
    ann_path = here / "mine_network" / "annotation.py"
    spec = importlib.util.spec_from_file_location("annotation_standalone", ann_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load annotation module from {ann_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_modules(modules_tsv: Path) -> dict:
    modules = {}
    with modules_tsv.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if header != ["Gene", "Module"]:
            raise ValueError(
                f"Unexpected columns in {modules_tsv}. "
                f"Expected ['Gene', 'Module'], got {header}"
            )
        for line in f:
            gene, module = line.rstrip("\n").split("\t")
            mid = module[1:] if module.startswith("M") else module
            mid = int(mid)
            modules.setdefault(mid, []).append(gene)
    return modules


def _load_background(background_file: Path) -> set:
    genes = set()
    with background_file.open("r", encoding="utf-8") as f:
        for line in f:
            gene = line.strip()
            if gene:
                genes.add(gene)
    return genes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate existing master modules with GMT gene sets."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Existing pipeline output folder (contains master_BTM_modules.tsv).",
    )
    parser.add_argument(
        "--gmt",
        nargs="+",
        required=True,
        help="One or more GMT files.",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="FDR cutoff (default: 0.05).",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Minimum overlap to test (default: 2).",
    )
    parser.add_argument(
        "--background-file",
        default=None,
        help="Optional text file (one gene per line) for background universe.",
    )

    args = parser.parse_args()
    ann_mod = _load_annotation_module()

    out_dir = Path(args.output_dir)
    modules_tsv = out_dir / "master_BTM_modules.tsv"

    if not modules_tsv.exists():
        raise FileNotFoundError(
            f"Missing {modules_tsv}. Run full pipeline first to create master modules."
        )

    modules = _load_modules(modules_tsv)
    print(f"[INFO] Loaded modules: {len(modules)}")

    if args.background_file:
        bg = _load_background(Path(args.background_file))
        print(f"[INFO] Background genes from file: {len(bg):,}")
    else:
        bg = {g for genes in modules.values() for g in genes}
        print(f"[INFO] Background genes from master modules: {len(bg):,}")

    gene_sets = ann_mod.load_multiple_gmt(args.gmt)
    ann = ann_mod.annotate_modules(
        modules,
        gene_sets,
        bg,
        fdr_threshold=args.fdr,
        min_overlap=args.min_overlap,
    )
    ann_mod.save_annotations(ann, str(out_dir))


if __name__ == "__main__":
    main()

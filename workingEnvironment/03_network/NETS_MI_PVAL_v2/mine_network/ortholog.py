"""
Ortholog mapping helpers for cross-species enrichment workflows.
"""

from __future__ import annotations

import pandas as pd


def load_ortholog_map(
    map_path: str,
    source_col: str,
    target_col: str,
) -> dict[str, set[str]]:
    """Load many-to-many ortholog mappings from a TSV file."""
    df = pd.read_csv(map_path, sep="\t", dtype=str)
    for col in (source_col, target_col):
        if col not in df.columns:
            raise ValueError(
                f"Ortholog map missing required column '{col}'. "
                f"Available: {list(df.columns)}"
            )

    df = df[[source_col, target_col]].dropna()
    df[source_col] = df[source_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df[(df[source_col] != "") & (df[target_col] != "")]

    mapping: dict[str, set[str]] = {}
    for src, tgt in df.itertuples(index=False, name=None):
        mapping.setdefault(src, set()).add(tgt)
    return mapping


def map_gene_set(genes: set[str], mapping: dict[str, set[str]]) -> tuple[set[str], int]:
    """Map source-species genes to target-species genes."""
    mapped = set()
    mapped_source = 0
    for gene in genes:
        tgts = mapping.get(gene)
        if tgts:
            mapped_source += 1
            mapped.update(tgts)
    return mapped, mapped_source


def map_modules(
    modules: dict[int, list[str]],
    mapping: dict[str, set[str]],
) -> tuple[dict[int, list[str]], dict[str, int]]:
    """Map module gene lists into target-species gene symbols."""
    mapped_modules: dict[int, list[str]] = {}
    source_total = 0
    source_mapped = 0

    for mid, genes in modules.items():
        src_set = set(genes)
        source_total += len(src_set)
        mapped_set, mapped_n = map_gene_set(src_set, mapping)
        source_mapped += mapped_n
        mapped_modules[mid] = sorted(mapped_set)

    stats = {
        "source_unique_genes": source_total,
        "source_genes_mapped": source_mapped,
        "mapped_unique_genes": len({g for genes in mapped_modules.values() for g in genes}),
    }
    return mapped_modules, stats

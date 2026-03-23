"""
Lightweight network minimap rendering.
"""

import os
import re
import numpy as np
import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def _downsample_by_degree(adj_sym: np.ndarray, gene_names: list, max_nodes: int):
    n = adj_sym.shape[0]
    if n <= max_nodes:
        return adj_sym, list(gene_names), np.arange(n, dtype=int)

    deg = adj_sym.sum(axis=1)
    keep = np.argsort(-deg)[:max_nodes]
    keep = np.sort(keep)
    sub_adj = adj_sym[np.ix_(keep, keep)]
    sub_names = [gene_names[i] for i in keep]
    return sub_adj, sub_names, keep


def _category_colors(categories: list):
    uniq = []
    seen = set()
    for c in categories:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    cmap = plt.get_cmap("tab20")
    color_map = {}
    for i, c in enumerate(uniq):
        color_map[c] = cmap(i % 20)
    return [color_map[c] for c in categories]


def _render_graph(adj_sym: np.ndarray,
                  gene_names: list,
                  out_path: str,
                  title: str,
                  node_categories: dict = None,
                  max_nodes: int = 1200,
                  dpi: int = 180,
                  edge_alpha: float = 0.08):
    adj_sym = np.maximum(adj_sym, adj_sym.T)
    np.fill_diagonal(adj_sym, 0)

    sub_adj, sub_names, keep_idx = _downsample_by_degree(adj_sym, gene_names, max_nodes)
    rows, cols = np.where(np.triu(sub_adj, k=1) > 0)

    g = ig.Graph(
        n=sub_adj.shape[0],
        edges=list(zip(rows.tolist(), cols.tolist())),
        directed=False,
    )
    layout = np.array(g.layout_fruchterman_reingold())

    if layout.shape[0] == 0:
        return

    x = layout[:, 0]
    y = layout[:, 1]

    if node_categories is None:
        node_colors = ["#2b8cbe"] * len(sub_names)
    else:
        cats = [node_categories.get(name, "unassigned") for name in sub_names]
        node_colors = _category_colors(cats)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for r, c in zip(rows, cols):
        ax.plot([x[r], x[c]], [y[r], y[c]], color="#8f8f8f", alpha=edge_alpha, linewidth=0.25)

    node_size = 8 if len(sub_names) <= 1000 else 5
    ax.scatter(x, y, c=node_colors, s=node_size, linewidths=0.0, alpha=0.95)

    ax.set_title(f"{title}\n(nodes shown: {len(sub_names)}/{len(gene_names)})", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_study_minimap(study_name: str,
                       adj: np.ndarray,
                       gene_names: list,
                       base_dir: str,
                       max_nodes: int = 1200,
                       dpi: int = 180,
                       edge_alpha: float = 0.08):
    """Save one small per-study network PNG."""
    study_dir = os.path.join(base_dir, "studies", _sanitize_name(study_name))
    out_path = os.path.join(study_dir, "network_minimap.png")
    _render_graph(
        adj,
        gene_names,
        out_path=out_path,
        title=f"Study {study_name}",
        node_categories=None,
        max_nodes=max_nodes,
        dpi=dpi,
        edge_alpha=edge_alpha,
    )
    print(f"[SAVED] {out_path}")


def save_master_minimaps(master_adj: np.ndarray,
                         gene_names: list,
                         membership: dict,
                         parent_child_rows: list,
                         base_dir: str,
                         max_nodes: int = 1200,
                         dpi: int = 180,
                         edge_alpha: float = 0.08):
    """Save master minimaps: plain, child-module colored, and parent-module colored."""
    master_dir = os.path.join(base_dir, "master_network")

    # Plain master map
    plain_path = os.path.join(master_dir, "master_network_minimap.png")
    _render_graph(
        master_adj,
        gene_names,
        out_path=plain_path,
        title="Master Network",
        node_categories=None,
        max_nodes=max_nodes,
        dpi=dpi,
        edge_alpha=edge_alpha,
    )
    print(f"[SAVED] {plain_path}")

    # Child module coloring (final modules after refinement)
    child_categories = {g: f"M{mid}" for g, mid in membership.items()}
    child_path = os.path.join(master_dir, "master_network_by_child_module.png")
    _render_graph(
        master_adj,
        gene_names,
        out_path=child_path,
        title="Master Network (Child Modules)",
        node_categories=child_categories,
        max_nodes=max_nodes,
        dpi=dpi,
        edge_alpha=edge_alpha,
    )
    print(f"[SAVED] {child_path}")

    # Parent module coloring (if Leiden module was split into MCODE submodules)
    if parent_child_rows:
        child_to_parent = {
            row.get("child_module"): row.get("parent_module")
            for row in parent_child_rows
            if row.get("child_module") and row.get("parent_module")
        }
        parent_categories = {}
        for gene, mid in membership.items():
            child_id = f"M{mid}"
            parent_categories[gene] = child_to_parent.get(child_id, child_id)

        parent_path = os.path.join(master_dir, "master_network_by_parent_module.png")
        _render_graph(
            master_adj,
            gene_names,
            out_path=parent_path,
            title="Master Network (Parent Modules)",
            node_categories=parent_categories,
            max_nodes=max_nodes,
            dpi=dpi,
            edge_alpha=edge_alpha,
        )
        print(f"[SAVED] {parent_path}")

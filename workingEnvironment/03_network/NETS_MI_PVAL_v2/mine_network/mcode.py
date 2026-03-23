"""
MCODE — Molecular Complex Detection (Bader & Hogue, BMC Bioinformatics 2003).
==============================================================================

Used by Li et al. (Nature Immunology, 2014) for de-novo module discovery in
blood transcription module (BTM) networks via the Cytoscape MCODE plug-in.

Algorithm (three stages)
-------------------------

**Stage 1 — Vertex weighting**

For every node v in the graph:

1. Find the highest k-core containing v → ``core_level(v)``
   (via igraph's O(V+E) peeling algorithm).
2. Compute local density of v's closed neighbourhood N[v]:
   ``density(N[v]) = edges_within_N[v] / possible_edges``
3. Node weight: ``w(v) = core_level(v) × density(N[v])``

**Stage 2 — Seed-and-extend**

Sort nodes by weight (descending).  For each unvisited seed:
- Grow a candidate complex by BFS: add neighbour u if
  ``w(u) ≥ score_threshold × max_weight_in_graph``
- Record the complex and mark nodes visited.

**Stage 3 — Post-processing** (Li et al. defaults)

Filter complexes:
- Drop modules with fewer than ``min_size`` nodes.
- Drop modules with edge density < ``min_density``.
- Remaining complexes become the final modules.

Overlapping membership
-----------------------
Nodes can appear in multiple modules (overlapping, like real MCODE output).
A ``membership`` dict is also returned for convenience where each gene is
assigned to the largest module it belongs to.

Parameters (matching Li et al. / Cytoscape defaults)
-----------------------------------------------------
- ``score_threshold = 0.2``  — fraction of max node weight
- ``min_size = 3``           — minimum genes per module
- ``min_density = 0.3``      — minimum edge density (Li et al.)
"""

import numpy as np
import igraph as ig
import time


def _membership_from_modules(modules: dict) -> dict:
    """Assign each gene to the largest module it belongs to."""
    gene_to_modules = {}
    for mid_id, genes in modules.items():
        for g in genes:
            gene_to_modules.setdefault(g, []).append((len(genes), mid_id))
    return {
        g: max(entries, key=lambda x: x[0])[1]
        for g, entries in gene_to_modules.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _k_core_levels(adj_sym: np.ndarray) -> np.ndarray:
    """
    K-core decomposition using igraph (fast, O(V + E)).

    Parameters
    ----------
    adj_sym : np.ndarray
        Symmetric binary adjacency matrix.

    Returns
    -------
    np.ndarray, shape (n,)
        Core level for each node.
    """
    g = ig.Graph.Adjacency((adj_sym > 0).tolist(), mode="undirected")
    return np.array(g.coreness(), dtype=int)


def _local_density(v: int, neighbours_v: list, adj_sym: np.ndarray) -> float:
    """
    Fraction of present edges within the closed neighbourhood of v.

    Parameters
    ----------
    v : int
        Node index.
    neighbours_v : list[int]
        Indices of v's direct neighbours.
    adj_sym : np.ndarray
        Symmetric binary adjacency matrix.

    Returns
    -------
    float
        Edge density in [0, 1].
    """
    members = [v] + list(neighbours_v)
    n = len(members)
    if n < 2:
        return 0.0
    sub = adj_sym[np.ix_(members, members)]
    present = int(np.triu(sub, k=1).sum())
    possible = n * (n - 1) // 2
    return present / possible


# ═══════════════════════════════════════════════════════════════════════════════
# MCODE algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def mcode(
    adj_matrix: np.ndarray,
    gene_names: list,
    score_threshold: float = 0.2,
    min_size: int = 3,
    min_density: float = 0.3,
) -> tuple:
    """
    Run MCODE module detection on a binary adjacency matrix.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape (n, n)
        Binary symmetric adjacency matrix.
    gene_names : list[str]
        Gene names corresponding to rows/columns.
    score_threshold : float
        Fraction of max node weight; neighbours below
        ``threshold × max_weight`` excluded during seed extension.
    min_size : int
        Discard modules with fewer genes.
    min_density : float
        Discard modules with edge density below this.

    Returns
    -------
    modules : dict[int, list[str]]
        ``{module_id: [gene_name, ...]}``.
    membership : dict[str, int]
        ``{gene_name: module_id}``.  Genes in multiple modules are
        assigned to the largest one (by gene count).
    """
    print("[INFO] Running MCODE module detection...")
    t0 = time.time()

    adj_sym = np.maximum(adj_matrix, adj_matrix.T).astype(np.uint8)
    n = adj_sym.shape[0]

    # ── Stage 1: Vertex weighting ──
    print("[INFO] MCODE stage 1: vertex weighting (k-core + local density)...")
    core = _k_core_levels(adj_sym)

    # Precompute neighbour lists
    neighbours = [list(np.where(adj_sym[v] > 0)[0]) for v in range(n)]

    weights = np.zeros(n)
    for v in range(n):
        weights[v] = core[v] * _local_density(v, neighbours[v], adj_sym)

    max_weight = weights.max()
    if max_weight == 0:
        print("[WARN] All node weights zero — graph may be empty or "
              "disconnected.")
        return {}, {}

    print(f"[INFO] Node weight range: [{weights.min():.4f}, {max_weight:.4f}]")

    # ── Stage 2: Seed-and-extend ──
    print("[INFO] MCODE stage 2: seed-and-extend...")
    wt_threshold = score_threshold * max_weight
    seed_order = np.argsort(-weights)   # highest weight first
    visited = np.zeros(n, dtype=bool)
    raw_modules = []

    for seed in seed_order:
        if visited[seed]:
            continue
        module_nodes = set()
        queue = [seed]
        while queue:
            v = queue.pop()
            if v in module_nodes:
                continue
            if weights[v] >= wt_threshold:
                module_nodes.add(v)
                for u in neighbours[v]:
                    if u not in module_nodes and weights[u] >= wt_threshold:
                        queue.append(u)
        if module_nodes:
            raw_modules.append(sorted(module_nodes))
            for v in module_nodes:
                visited[v] = True

    print(f"[INFO] MCODE stage 2: {len(raw_modules)} raw complexes found")

    # ── Stage 3: Post-processing ──
    print(f"[INFO] MCODE stage 3: filtering (min_size={min_size}, "
          f"min_density={min_density})...")
    modules = {}
    mid = 0
    for node_list in raw_modules:
        if len(node_list) < min_size:
            continue
        sub = adj_sym[np.ix_(node_list, node_list)]
        m_edges = int(np.triu(sub, k=1).sum())
        m_nodes = len(node_list)
        m_possible = m_nodes * (m_nodes - 1) // 2
        density = m_edges / m_possible if m_possible > 0 else 0.0
        if density < min_density:
            continue
        modules[mid] = [gene_names[i] for i in node_list]
        mid += 1

    elapsed = time.time() - t0
    print(f"[INFO] MCODE: {len(modules)} modules after post-processing "
          f"({elapsed:.1f}s)")
    if modules:
        sizes = sorted([len(v) for v in modules.values()], reverse=True)
        print(f"[INFO] Module sizes (top 10): {sizes[:10]}")

    membership = _membership_from_modules(modules)

    return modules, membership


def leiden_modules(
    adj_matrix: np.ndarray,
    gene_names: list,
    edge_weights: np.ndarray = None,
    resolution: float = 1.0,
    n_iterations: int = -1,
    min_size: int = 3,
) -> tuple:
    """
    Detect graph communities using Leiden and return MCODE-compatible output.
    """
    print("[INFO] Running Leiden community detection...")
    t0 = time.time()

    adj_sym = np.maximum(adj_matrix, adj_matrix.T)
    rows, cols = np.where(np.triu(adj_sym, k=1) > 0)

    g = ig.Graph(
        n=adj_sym.shape[0],
        edges=list(zip(rows.tolist(), cols.tolist())),
        directed=False,
    )
    g.vs["name"] = gene_names

    weights = None
    if edge_weights is not None:
        ew_sym = np.maximum(edge_weights, edge_weights.T)
        weights = ew_sym[rows, cols].astype(float)
        g.es["weight"] = weights.tolist()

    try:
        part = g.community_leiden(
            objective_function="modularity",
            weights=weights,
            resolution=resolution,
            n_iterations=n_iterations,
        )
    except TypeError:
        part = g.community_leiden(
            objective_function="modularity",
            weights=weights,
            resolution_parameter=resolution,
            n_iterations=n_iterations,
        )

    modules = {}
    mid = 0
    for comm in part:
        if len(comm) < min_size:
            continue
        modules[mid] = [gene_names[i] for i in comm]
        mid += 1

    elapsed = time.time() - t0
    print(f"[INFO] Leiden: {len(modules)} modules after size filtering "
          f"(min_size={min_size}, {elapsed:.1f}s)")
    if modules:
        sizes = sorted([len(v) for v in modules.values()], reverse=True)
        print(f"[INFO] Leiden module sizes (top 10): {sizes[:10]}")

    membership = _membership_from_modules(modules)
    return modules, membership


def refine_large_modules(
    modules: dict,
    master_adj: np.ndarray,
    gene_names: list,
    size_threshold: int,
    method: str = "mcode",
    leiden_resolution: float = 1.0,
    leiden_iterations: int = -1,
    score_threshold: float = 0.2,
    min_size: int = 3,
    min_density: float = 0.3,
) -> tuple:
    """
    Refine oversized modules and replace them with detected submodules.

    Parameters
    ----------
    method : str
        Submodule detector used within oversized parent modules:
        ``"mcode"`` or ``"leiden"``.
    """
    if size_threshold is None:
        membership = _membership_from_modules(modules)
        return modules, membership, []

    if method not in {"mcode", "leiden"}:
        raise ValueError(f"Unsupported submodule method: {method}")

    print(
        f"[INFO] Refining modules larger than {size_threshold} genes "
        f"with {method.upper()}..."
    )

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    refined = {}
    parent_child = []
    new_mid = 0

    for parent_id, genes in modules.items():
        if len(genes) <= size_threshold:
            refined[new_mid] = genes
            parent_child.append({
                "parent_module": f"M{parent_id}",
                "child_module": f"M{new_mid}",
                "parent_size": len(genes),
                "child_size": len(genes),
                "refined": False,
            })
            new_mid += 1
            continue

        idx = [gene_to_idx[g] for g in genes]
        sub_adj = master_adj[np.ix_(idx, idx)]

        if method == "leiden":
            submods, _ = leiden_modules(
                sub_adj,
                genes,
                edge_weights=None,
                resolution=leiden_resolution,
                n_iterations=leiden_iterations,
                min_size=min_size,
            )
        else:
            submods, _ = mcode(
                sub_adj,
                genes,
                score_threshold=score_threshold,
                min_size=min_size,
                min_density=min_density,
            )

        if not submods:
            refined[new_mid] = genes
            parent_child.append({
                "parent_module": f"M{parent_id}",
                "child_module": f"M{new_mid}",
                "parent_size": len(genes),
                "child_size": len(genes),
                "refined": False,
            })
            new_mid += 1
            continue

        for _, subgenes in submods.items():
            refined[new_mid] = subgenes
            parent_child.append({
                "parent_module": f"M{parent_id}",
                "child_module": f"M{new_mid}",
                "parent_size": len(genes),
                "child_size": len(subgenes),
                "refined": True,
            })
            new_mid += 1

    membership = _membership_from_modules(refined)
    print(f"[INFO] Refinement complete: {len(modules)} -> {len(refined)} modules")
    return refined, membership, parent_child


def refine_large_modules_with_mcode(
    modules: dict,
    master_adj: np.ndarray,
    gene_names: list,
    size_threshold: int,
    score_threshold: float = 0.2,
    min_size: int = 3,
    min_density: float = 0.3,
) -> tuple:
    """Backward-compatible wrapper for historical call sites."""
    return refine_large_modules(
        modules=modules,
        master_adj=master_adj,
        gene_names=gene_names,
        size_threshold=size_threshold,
        method="mcode",
        score_threshold=score_threshold,
        min_size=min_size,
        min_density=min_density,
    )

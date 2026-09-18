"""Principled subgraph sampling of the FlyWire connectome (PLAN.md §7, §9).

The 135k-neuron graph is too large to train on the 6 GB GPU, so experiments run on
subgraphs of increasing size (1k, 5k, 10k, 20k). **Sampling must be principled**:
a uniform-random pick of neuron ids yields a nearly edgeless, disconnected subgraph
(dead islands) because the graph is sparse. The methods here instead grow connected,
functionally-grounded subgraphs.

Methods
-------
* ``top_degree``   — the N highest-degree neurons (hubs; densely interconnected).
* ``bfs``          — breadth-first growth from seed neurons (a connected local circuit).
* ``bfs_sensory``  — BFS seeded from sensory / afferent neurons (input-driven circuit).
* ``random``       — a uniform-random control, provided ONLY as an ablation baseline
  (PLAN.md §17-18); never the default choice.

Each returns index arrays into the parent connectome; :func:`subgraph` slices the
adjacency accordingly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from connectome.adjacency import Connectome

_SENSORY_CLASSES = {"sensory", "sensory_ascending"}


def _degrees(adj: sp.csr_matrix) -> np.ndarray:
    """Total (in + out) degree per node, ignoring weights."""
    binary = adj.copy()
    binary.data[:] = 1.0
    out_deg = np.asarray(binary.sum(axis=1)).ravel()
    in_deg = np.asarray(binary.sum(axis=0)).ravel()
    return out_deg + in_deg


def largest_wcc(adj: sp.csr_matrix) -> np.ndarray:
    """Return node indices of the largest weakly-connected component."""
    n_comp, labels = csg.connected_components(adj, directed=True, connection="weak")
    if n_comp == 1:
        return np.arange(adj.shape[0])
    counts = np.bincount(labels)
    biggest = int(counts.argmax())
    return np.flatnonzero(labels == biggest)


def top_degree(adj: sp.csr_matrix, n: int) -> np.ndarray:
    """Indices of the ``n`` highest-degree nodes (descending degree)."""
    deg = _degrees(adj)
    n = min(n, adj.shape[0])
    return np.argsort(deg)[::-1][:n]


def bfs_from(adj: sp.csr_matrix, seeds: np.ndarray, n: int) -> np.ndarray:
    """Grow a set of ``n`` nodes by BFS from ``seeds`` over the undirected graph.

    Treats edges as undirected for reachability so the frontier can expand along both
    pre- and post-synaptic links. Falls back to filling with high-degree nodes if the
    reachable component is smaller than ``n``.
    """
    n = min(n, adj.shape[0])
    undirected = adj + adj.T
    visited: list[int] = []
    seen = np.zeros(adj.shape[0], dtype=bool)
    frontier: list[int] = []
    for s in seeds:  # de-duplicate seeds, mark as seen
        s = int(s)
        if not seen[s]:
            seen[s] = True
            frontier.append(s)
    while frontier and len(visited) < n:
        nxt: list[int] = []
        for node in frontier:
            if len(visited) >= n:
                break
            visited.append(node)
            start, end = undirected.indptr[node], undirected.indptr[node + 1]
            for nb in undirected.indices[start:end]:
                if not seen[nb]:
                    seen[nb] = True
                    nxt.append(int(nb))
        frontier = nxt
    if len(visited) < n:  # pad with highest-degree unseen nodes
        deg = _degrees(adj)
        for idx in np.argsort(deg)[::-1]:
            if len(visited) >= n:
                break
            if not seen[idx]:
                seen[idx] = True
                visited.append(int(idx))
    return np.array(visited[:n], dtype=np.int64)


def sensory_seed_indices(
    meta: pl.DataFrame,
    node_ids: list[str],
    *,
    adj: sp.csr_matrix | None = None,
    top_k: int | None = None,
) -> np.ndarray:
    """Indices (into ``node_ids``) of sensory / afferent neurons, for BFS seeding.

    Sensory neurons largely project *forward* into the central brain rather than to
    each other, so seeding BFS with *all* of them just collects a disconnected sensory
    layer. Pass ``adj`` and ``top_k`` to instead return the ``top_k`` highest-degree
    sensory neurons: seeding from a small hub set lets BFS grow one connected,
    input-driven circuit.
    """
    is_sensory = pl.col("super_class").is_in(list(_SENSORY_CLASSES)) | (
        pl.col("flow") == "afferent"
    )
    sensory_ids = set(meta.filter(is_sensory)["fafb_783_id"].to_list())
    idx = np.array([i for i, nid in enumerate(node_ids) if nid in sensory_ids], dtype=np.int64)
    if top_k is not None and adj is not None and idx.size > top_k:
        deg = _degrees(adj)
        idx = idx[np.argsort(deg[idx])[::-1][:top_k]]
    return idx


def subgraph(conn: Connectome, idx: np.ndarray) -> Connectome:
    """Slice a connectome down to the nodes in ``idx`` (order preserved)."""
    idx = np.asarray(idx)
    sub_adj = conn.adj[idx][:, idx].tocsr()
    sub_ids = [conn.node_ids[i] for i in idx]
    return Connectome(adj=sub_adj, node_ids=sub_ids)


def sample(
    conn: Connectome,
    n: int,
    *,
    method: str = "top_degree",
    meta: pl.DataFrame | None = None,
    seed: int = 0,
) -> Connectome:
    """Return an ``n``-node subgraph chosen by ``method`` (see module docstring).

    ``bfs_sensory`` requires ``meta`` for seed selection. ``random`` is a control
    baseline only.
    """
    if method == "top_degree":
        idx = top_degree(conn.adj, n)
    elif method == "bfs":
        idx = bfs_from(conn.adj, top_degree(conn.adj, 1), n)  # seed at the top hub
    elif method == "bfs_sensory":
        if meta is None:
            raise ValueError("bfs_sensory requires meta for sensory seed selection")
        # A small set of high-degree sensory hubs seeds a connected input-driven circuit.
        seeds = sensory_seed_indices(meta, conn.node_ids, adj=conn.adj, top_k=8)
        if seeds.size == 0:
            raise ValueError("no sensory/afferent seeds found in this node set")
        idx = bfs_from(conn.adj, seeds, n)
    elif method == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(conn.num_nodes, size=min(n, conn.num_nodes), replace=False)
    else:
        raise ValueError(f"unknown method {method!r}; choose top_degree/bfs/bfs_sensory/random")
    return subgraph(conn, idx)

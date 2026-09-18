"""Topology controls for the ablation study (PLAN.md §17-18).

The scientific question is whether the *biological wiring* of FlyWire helps learning,
or whether any graph with the same coarse statistics would do as well. To answer it we
compare the real connectome mask against matched controls, all at the same neuron count
N and (where noted) the same edge count M:

* ``degree_preserving_shuffle`` — rewires edges while preserving every neuron's in- and
  out-degree exactly. This is the **key null**: same degree sequence, biological
  structure destroyed. If FlyWire beats this, structure beyond degree matters.
* ``erdos_renyi_like`` — a uniform-random directed graph with the same M (degree
  sequence not preserved).
* ``dense`` — every possible directed edge (all-to-all), the unconstrained upper bound.
* ``feedforward`` — a layered DAG with the same M (no recurrence).

Each returns a :class:`Connectome` reusing the input's ``node_ids`` so downstream code
(the FlyWire SNN) treats every condition identically apart from the mask.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from connectome.adjacency import Connectome


def _edges_of(conn: Connectome) -> tuple[np.ndarray, np.ndarray]:
    coo = conn.adj.tocoo()
    return coo.row.copy(), coo.col.copy()


def _from_edges(node_ids: list[str], src: np.ndarray, dst: np.ndarray) -> Connectome:
    """Build a binary Connectome from edge endpoints (weights are the mask; set to 1)."""
    n = len(node_ids)
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return Connectome(adj=adj, node_ids=list(node_ids))


def degree_preserving_shuffle(
    conn: Connectome, *, swaps_per_edge: int = 10, seed: int = 0
) -> Connectome:
    """Rewire edges via directed double-edge swaps, preserving in/out degree exactly.

    Each accepted swap takes edges ``a->b`` and ``c->d`` to ``a->d`` and ``c->b``,
    which leaves out-degree(a), out-degree(c), in-degree(b) and in-degree(d) unchanged.
    Self-loops and duplicate edges are rejected. ``swaps_per_edge`` controls mixing.
    """
    src, dst = _edges_of(conn)
    m = len(src)
    rng = np.random.default_rng(seed)
    existing = set(zip(src.tolist(), dst.tolist()))
    target = swaps_per_edge * m
    done = 0
    tries = 0
    max_tries = target * 10
    while done < target and tries < max_tries:
        tries += 1
        i, j = int(rng.integers(m)), int(rng.integers(m))
        if i == j:
            continue
        a, b = src[i], dst[i]
        c, d = src[j], dst[j]
        if a == d or c == b:  # would create a self-loop
            continue
        if (a, d) in existing or (c, b) in existing:  # would duplicate an edge
            continue
        existing.discard((a, b))
        existing.discard((c, d))
        existing.add((a, d))
        existing.add((c, b))
        dst[i], dst[j] = d, b
        done += 1
    return _from_edges(conn.node_ids, src, dst)


def erdos_renyi_like(conn: Connectome, *, seed: int = 0) -> Connectome:
    """A uniform-random directed graph with the same node count and edge count."""
    n = conn.num_nodes
    m = conn.num_edges
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    # Sample distinct off-diagonal cells encoded as i * n + j.
    while len(chosen) < m:
        need = m - len(chosen)
        cand = rng.integers(0, n * n, size=need * 2)
        for c in cand:
            if c % (n + 1) != 0:  # skip diagonal (i == j)
                chosen.add(int(c))
            if len(chosen) >= m:
                break
    arr = np.fromiter(chosen, dtype=np.int64, count=m)
    return _from_edges(conn.node_ids, arr // n, arr % n)


def dense(conn: Connectome) -> Connectome:
    """All-to-all directed graph (no self-loops) on the same nodes."""
    n = conn.num_nodes
    idx = np.arange(n)
    src = np.repeat(idx, n)
    dst = np.tile(idx, n)
    keep = src != dst
    return _from_edges(conn.node_ids, src[keep], dst[keep])


def feedforward(conn: Connectome, *, layers: int = 4, seed: int = 0) -> Connectome:
    """A layered feed-forward DAG with the same node and edge counts (no recurrence).

    Nodes are partitioned into ``layers`` roughly-equal groups; edges go only from one
    layer to the next, sampled uniformly until the original edge count is matched.
    """
    n = conn.num_nodes
    m = conn.num_edges
    rng = np.random.default_rng(seed)
    layer_of = np.array_split(rng.permutation(n), layers)
    # Candidate directed edges between consecutive layers.
    src_list: list[int] = []
    dst_list: list[int] = []
    pairs = [(layer_of[k], layer_of[k + 1]) for k in range(layers - 1)]
    chosen: set[tuple[int, int]] = set()
    guard = 0
    while len(chosen) < m and guard < m * 50:
        guard += 1
        a_layer, b_layer = pairs[int(rng.integers(len(pairs)))]
        a = int(a_layer[rng.integers(len(a_layer))])
        b = int(b_layer[rng.integers(len(b_layer))])
        if (a, b) not in chosen:
            chosen.add((a, b))
    for a, b in chosen:
        src_list.append(a)
        dst_list.append(b)
    return _from_edges(conn.node_ids, np.array(src_list), np.array(dst_list))


def apply_topology(conn: Connectome, topology: str, *, seed: int = 0) -> Connectome:
    """Return ``conn`` transformed to the named topology (``flywire`` = unchanged)."""
    if topology == "flywire":
        return conn
    if topology == "shuffled":
        return degree_preserving_shuffle(conn, seed=seed)
    if topology == "random":
        return erdos_renyi_like(conn, seed=seed)
    if topology == "dense":
        return dense(conn)
    if topology == "feedforward":
        return feedforward(conn, seed=seed)
    raise ValueError(
        f"unknown topology {topology!r}; choose flywire/shuffled/random/dense/feedforward"
    )

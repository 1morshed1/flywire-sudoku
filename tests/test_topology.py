"""Phase 6 tests: topology controls for the ablation study.

Verifies the properties that make each control a valid comparison: the shuffle
preserves every neuron's in/out degree, the random and feed-forward controls match the
original edge count, feed-forward is acyclic, and dispatch/validation behave.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from connectome import topology
from connectome.adjacency import Connectome


def _toy(n: int, density: float = 0.08, seed: int = 0) -> Connectome:
    rng = np.random.default_rng(seed)
    m = int(n * n * density)
    r = rng.integers(0, n, m)
    c = rng.integers(0, n, m)
    keep = r != c
    adj = sp.coo_matrix((np.ones(keep.sum(), np.float32), (r[keep], c[keep])), shape=(n, n)).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return Connectome(adj=adj, node_ids=[str(i) for i in range(n)])


def _in_out_degree(conn: Connectome) -> tuple[np.ndarray, np.ndarray]:
    a = conn.adj.copy()
    a.data[:] = 1.0
    return np.asarray(a.sum(0)).ravel(), np.asarray(a.sum(1)).ravel()


def test_shuffle_preserves_degree_exactly():
    conn = _toy(120, seed=1)
    sh = topology.degree_preserving_shuffle(conn, swaps_per_edge=10, seed=0)
    oin, oout = _in_out_degree(conn)
    sin, sout = _in_out_degree(sh)
    assert np.array_equal(oin, sin)  # per-node in-degree preserved
    assert np.array_equal(oout, sout)  # per-node out-degree preserved
    assert sh.num_edges == conn.num_edges


def test_shuffle_actually_rewires():
    conn = _toy(120, seed=2)
    sh = topology.degree_preserving_shuffle(conn, swaps_per_edge=10, seed=0)
    # The edge set should differ from the original (structure changed).
    orig = set(zip(*conn.adj.nonzero()))
    new = set(zip(*sh.adj.nonzero()))
    assert orig != new


def test_random_and_feedforward_match_edge_count():
    conn = _toy(100, seed=3)
    for fn in (topology.erdos_renyi_like, topology.feedforward):
        ctrl = fn(conn, seed=0)
        assert ctrl.num_edges == conn.num_edges
        assert ctrl.num_nodes == conn.num_nodes


def test_no_self_loops_in_controls():
    conn = _toy(80, seed=4)
    for name in ("shuffled", "random", "dense", "feedforward"):
        ctrl = topology.apply_topology(conn, name, seed=0)
        assert ctrl.adj.diagonal().sum() == 0


def test_feedforward_is_acyclic():
    conn = _toy(90, seed=5)
    ff = topology.feedforward(conn, layers=4, seed=0)
    # A DAG has as many strongly-connected components as nodes.
    n_scc, _ = csg.connected_components(ff.adj, directed=True, connection="strong")
    assert n_scc == ff.num_nodes


def test_dense_is_all_to_all():
    conn = _toy(30, seed=6)
    d = topology.dense(conn)
    assert d.num_edges == conn.num_nodes * (conn.num_nodes - 1)


def test_apply_topology_dispatch_and_error():
    conn = _toy(20)
    assert topology.apply_topology(conn, "flywire") is conn  # unchanged
    with pytest.raises(ValueError, match="unknown topology"):
        topology.apply_topology(conn, "smallworld")

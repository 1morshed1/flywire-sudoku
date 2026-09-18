"""Phase 4 tests: connectome filtering, adjacency, sampling, statistics.

Logic is tested on small synthetic graphs (fast, deterministic). A guarded
integration test runs the full pipeline on the real FAFB dump when it is present
(downloaded under data/flywire/), and is skipped otherwise so CI stays offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from connectome import sampling, statistics
from connectome.adjacency import Connectome, build_adjacency, load, save, signs_from_meta
from connectome.filter import filter_edges, neuron_ids_from_edges, select_neurons

_REAL_EDGES = Path("data/flywire/fafb_783_simple_edgelist.feather")
_REAL_META = Path("data/flywire/fafb_783_meta.feather")


def _toy_edges() -> pl.DataFrame:
    # A small directed graph: a->b (7), b->c (5), a->c (3, below threshold), c->a (10).
    return pl.DataFrame(
        {
            "pre": ["a", "b", "a", "c"],
            "post": ["b", "c", "c", "a"],
            "count": [7, 5, 3, 10],
            "norm": [0.7, 0.5, 0.3, 1.0],
        }
    )


def test_filter_edges_threshold():
    e = _toy_edges()
    kept = filter_edges(e, min_count=5)
    assert kept.height == 3  # the count=3 edge is removed
    assert kept["count"].min() >= 5


def test_neuron_ids_sorted_unique():
    ids = neuron_ids_from_edges(filter_edges(_toy_edges()))
    assert ids == ["a", "b", "c"]  # sorted, unique, count=3 edge's nodes still present


def test_build_adjacency_values():
    kept = filter_edges(_toy_edges())
    conn = build_adjacency(kept, ["a", "b", "c"], weight="count")
    assert conn.num_nodes == 3
    dense = conn.adj.toarray()
    idx = conn.id_to_index()
    assert dense[idx["a"], idx["b"]] == 7
    assert dense[idx["b"], idx["c"]] == 5
    assert dense[idx["c"], idx["a"]] == 10
    assert dense[idx["a"], idx["c"]] == 0  # filtered out


def test_build_adjacency_drops_out_of_set_edges():
    kept = filter_edges(_toy_edges())
    conn = build_adjacency(kept, ["a", "b"], weight="binary")  # 'c' excluded
    assert conn.num_nodes == 2
    # Only a->b survives (all other edges touch 'c').
    assert conn.num_edges == 1
    assert conn.adj.toarray()[conn.id_to_index()["a"], conn.id_to_index()["b"]] == 1.0


def test_signs_from_meta():
    meta = pl.DataFrame(
        {
            "fafb_783_id": ["a", "b", "c", "d"],
            "neurotransmitter_predicted": ["acetylcholine", "gaba", "glutamate", None],
        }
    )
    signs = signs_from_meta(meta, ["a", "b", "c", "d"])
    assert list(signs) == [1, -1, -1, 1]  # None defaults to +1


def test_select_neurons_excludes_non_neuron():
    meta = pl.DataFrame(
        {
            "fafb_783_id": ["a", "b", "g"],
            "super_class": ["sensory", "central_brain_intrinsic", "glia"],
        }
    )
    ids = select_neurons(meta, exclude_non_neuron=True)
    assert ids == ["a", "b"]
    ids2 = select_neurons(meta, super_classes=["sensory"], exclude_non_neuron=True)
    assert ids2 == ["a"]


def _line_graph(n: int) -> Connectome:
    """A connected directed chain 0->1->...->(n-1) for sampling/stat tests."""
    rows = np.arange(n - 1)
    cols = np.arange(1, n)
    adj = sp.coo_matrix((np.ones(n - 1), (rows, cols)), shape=(n, n)).tocsr()
    return Connectome(adj=adj, node_ids=[str(i) for i in range(n)])


def test_top_degree_and_subgraph():
    conn = _line_graph(10)
    idx = sampling.top_degree(conn.adj, 4)
    assert len(idx) == 4
    sub = sampling.subgraph(conn, idx)
    assert sub.num_nodes == 4


def test_bfs_connected():
    conn = _line_graph(20)
    idx = sampling.bfs_from(conn.adj, np.array([0]), 10)
    assert len(idx) == 10
    sub = sampling.subgraph(conn, idx)
    # BFS from node 0 along the chain gives nodes 0..9 -> one weak component.
    s = statistics.summary(sub)
    assert s["largest_component_frac"] == 1.0


def test_statistics_summary_line_graph():
    s = statistics.summary(_line_graph(5))
    assert s["num_nodes"] == 5
    assert s["num_edges"] == 4
    assert s["largest_component_frac"] == 1.0
    assert s["reciprocity"] == 0.0  # a pure chain has no reciprocal edges


def test_save_load_roundtrip(tmp_path):
    conn = build_adjacency(filter_edges(_toy_edges()), ["a", "b", "c"])
    save(conn, tmp_path)
    back = load(tmp_path)
    assert back.node_ids == conn.node_ids
    assert (back.adj != conn.adj).nnz == 0


@pytest.mark.skipif(
    not (_REAL_EDGES.exists() and _REAL_META.exists()),
    reason="FlyWire FAFB dump not downloaded (data/flywire/)",
)
def test_real_pipeline_integration():
    from connectome.loader import ensure_and_load

    _meta, edges = ensure_and_load(download=False)
    kept = filter_edges(edges)
    ids = neuron_ids_from_edges(kept)
    assert len(ids) > 100_000  # ~135k neurons at count>=5
    conn = build_adjacency(kept, ids, weight="count")
    assert conn.num_edges > 3_000_000
    # Principled sampling must yield a connected subgraph; random must not.
    good = sampling.sample(conn, 1000, method="top_degree")
    assert statistics.summary(good)["largest_component_frac"] > 0.95
    bad = sampling.sample(conn, 1000, method="random", seed=0)
    assert statistics.summary(bad)["largest_component_frac"] < 0.5

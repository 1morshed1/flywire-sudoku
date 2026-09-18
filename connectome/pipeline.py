"""High-level convenience API: connectome dumps -> ready-to-train subgraph.

Wraps the download -> load -> filter -> adjacency -> sample chain behind two calls,
and caches the (expensive) full-connectome build in memory so repeated experiments in
one process do not rebuild the 135k-node graph each time.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import polars as pl

from connectome import sampling
from connectome.adjacency import Connectome, build_adjacency, signs_from_meta
from connectome.filter import filter_edges, neuron_ids_from_edges
from connectome.loader import ensure_and_load


@lru_cache(maxsize=2)
def full_connectome(
    min_count: int = 5, weight: str = "count", *, download: bool = True
) -> tuple[pl.DataFrame, Connectome]:
    """Return ``(meta, connectome)`` for the whole thresholded graph (cached).

    Downloads the FAFB dumps on first use unless ``download=False``.
    """
    meta, edges = ensure_and_load(download=download)
    kept = filter_edges(edges, min_count)
    ids = neuron_ids_from_edges(kept)
    conn = build_adjacency(kept, ids, weight=weight)
    return meta, conn


def subgraph(
    n: int,
    *,
    method: str = "top_degree",
    weight: str = "count",
    min_count: int = 5,
    seed: int = 0,
    download: bool = True,
) -> tuple[Connectome, np.ndarray]:
    """Return an ``n``-neuron subgraph plus its per-neuron sign vector.

    Parameters mirror :func:`connectome.sampling.sample`. The sign vector aligns to the
    subgraph's ``node_ids`` and is used for the Dale's-law (sign-constrained) regime.
    """
    meta, conn = full_connectome(min_count, weight, download=download)
    sub = sampling.sample(conn, n, method=method, meta=meta, seed=seed)
    signs = signs_from_meta(meta, sub.node_ids)
    return sub, signs

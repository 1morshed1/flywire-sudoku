"""Connectivity summary statistics for a connectome or subgraph (PLAN.md §22).

Reported per experiment so topology conditions (FlyWire vs random vs degree-matched)
can be compared on equal footing, and so subgraph selection can be sanity-checked
(e.g. a good sample should be one large component, not scattered islands).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from connectome.adjacency import Connectome


def summary(conn: Connectome) -> dict[str, float]:
    """Return node/edge counts, density, degree stats, reciprocity, and WCC info."""
    adj: sp.csr_matrix = conn.adj
    n = adj.shape[0]
    m = int(adj.nnz)

    binary = adj.copy()
    binary.data[:] = 1.0
    out_deg = np.asarray(binary.sum(axis=1)).ravel()
    in_deg = np.asarray(binary.sum(axis=0)).ravel()

    # Reciprocity: fraction of directed edges whose reverse edge also exists.
    recip_edges = int(binary.multiply(binary.T).nnz)
    reciprocity = recip_edges / m if m else 0.0

    n_comp, labels = csg.connected_components(adj, directed=True, connection="weak")
    largest = int(np.bincount(labels).max()) if n else 0

    max_possible = n * (n - 1)
    return {
        "num_nodes": float(n),
        "num_edges": float(m),
        "density": m / max_possible if max_possible else 0.0,
        "mean_out_degree": float(out_deg.mean()) if n else 0.0,
        "mean_in_degree": float(in_deg.mean()) if n else 0.0,
        "max_out_degree": float(out_deg.max()) if n else 0.0,
        "max_in_degree": float(in_deg.max()) if n else 0.0,
        "reciprocity": reciprocity,
        "num_weak_components": float(n_comp),
        "largest_component_frac": largest / n if n else 0.0,
    }

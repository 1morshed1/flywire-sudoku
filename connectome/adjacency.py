"""Build a sparse adjacency matrix from FlyWire edges (PLAN.md §8, §20).

The connectome is stored as a directed sparse matrix ``A`` where ``A[i, j]`` is the
weight of the synaptic connection from neuron ``i`` (presynaptic) to neuron ``j``
(postsynaptic). Neuron ids are mapped to contiguous integer indices ``0..N-1``.

Sparsity is mandatory (PLAN.md §20): a dense 135k x 135k matrix would be ~18 billion
entries. We keep everything in :mod:`scipy.sparse` CSR form (~3.5M nonzeros).

The adjacency is the fixed mask ``A`` in the FlyWire-SNN's ``W_effective = W * A``
(Phase 5). Separately, :func:`signs_from_meta` derives a per-neuron excitatory/
inhibitory sign (Dale's law) for the sign-constrained ablation (PLAN.md §8, Regime B).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

# Transmitter -> sign (Dale's law modeling choice; PLAN.md §8).
# Acetylcholine is excitatory; GABA and (in Drosophila, via GluCl) glutamate are
# inhibitory; the aminergic transmitters are treated as excitatory-ish by default.
# This is a documented modeling assumption, revisited in the Phase 6 ablations.
_TRANSMITTER_SIGN: dict[str, int] = {
    "acetylcholine": +1,
    "glutamate": -1,
    "gaba": -1,
    "dopamine": +1,
    "serotonin": +1,
    "octopamine": +1,
}


@dataclass
class Connectome:
    """A sparse directed connectome with its neuron-id <-> index mapping.

    Attributes
    ----------
    adj:
        CSR matrix, shape ``(N, N)``; ``adj[i, j]`` = weight of edge i->j.
    node_ids:
        List of neuron id strings; ``node_ids[i]`` is the id of index ``i``.
    """

    adj: sp.csr_matrix
    node_ids: list[str]

    @property
    def num_nodes(self) -> int:
        return self.adj.shape[0]

    @property
    def num_edges(self) -> int:
        return int(self.adj.nnz)

    def id_to_index(self) -> dict[str, int]:
        """Map neuron id -> row/col index (built on demand)."""
        return {nid: i for i, nid in enumerate(self.node_ids)}


def build_adjacency(
    edges: pl.DataFrame,
    node_ids: list[str] | None = None,
    *,
    weight: str = "count",
) -> Connectome:
    """Assemble a sparse directed adjacency matrix from an edge frame.

    Parameters
    ----------
    edges:
        Frame with ``pre``, ``post`` and the chosen weight column.
    node_ids:
        Node set / ordering. If None, uses the sorted unique endpoints of ``edges``.
        Edges touching a neuron outside this set are dropped.
    weight:
        ``"count"`` (synapse count), ``"norm"`` (normalized), or ``"binary"`` (1.0).
    """
    if node_ids is None:
        node_ids = pl.concat([edges["pre"], edges["post"]]).unique().sort().to_list()
    n = len(node_ids)

    # Vectorized id -> index mapping via a join (a Python dict loop over millions of
    # edges would be far slower). Edges touching a neuron outside the set become null
    # on join and are dropped.
    id_map = pl.DataFrame({"id": node_ids, "idx": np.arange(n, dtype=np.int32)})
    cols_needed = ["pre", "post"] + ([weight] if weight != "binary" else [])
    kept = (
        edges.select(cols_needed)
        .join(id_map.rename({"id": "pre", "idx": "row"}), on="pre", how="inner")
        .join(id_map.rename({"id": "post", "idx": "col"}), on="post", how="inner")
    )

    rows = kept["row"].to_numpy()
    cols = kept["col"].to_numpy()
    if weight == "binary":
        data = np.ones(kept.height, dtype=np.float32)
    elif weight in ("count", "norm"):
        data = kept[weight].to_numpy().astype(np.float32)
    else:
        raise ValueError(f"weight must be 'count', 'norm', or 'binary', got {weight!r}")

    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adj.sum_duplicates()
    return Connectome(adj=adj, node_ids=node_ids)


def signs_from_meta(meta: pl.DataFrame, node_ids: list[str]) -> np.ndarray:
    """Per-neuron sign vector (+1 excitatory / -1 inhibitory) by predicted transmitter.

    Neurons with an unknown/unmapped transmitter default to +1. Returned shape ``(N,)``
    aligned to ``node_ids``. Used to build the sign-constrained mask in Phase 5.
    """
    sign_map = (
        meta.select(["fafb_783_id", "neurotransmitter_predicted"])
        .with_columns(
            pl.col("neurotransmitter_predicted")
            .replace_strict(_TRANSMITTER_SIGN, default=1)
            .alias("sign")
        )
        .select(["fafb_783_id", "sign"])
    )
    lookup = dict(zip(sign_map["fafb_783_id"].to_list(), sign_map["sign"].to_list()))
    return np.array([lookup.get(nid, 1) for nid in node_ids], dtype=np.int8)


def save(conn: Connectome, out_dir: Path | str) -> None:
    """Persist a connectome: adjacency ``.npz`` + node ids ``.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / "adjacency.npz", conn.adj)
    (out / "node_ids.json").write_text(json.dumps(conn.node_ids))


def load(out_dir: Path | str) -> Connectome:
    """Load a connectome saved by :func:`save`."""
    out = Path(out_dir)
    adj = sp.load_npz(out / "adjacency.npz").tocsr()
    node_ids = json.loads((out / "node_ids.json").read_text())
    return Connectome(adj=adj, node_ids=node_ids)

"""Load FlyWire FAFB v783 metadata and edge list into Polars frames.

The feather (Arrow IPC) files are read with Polars. The edge list is large
(~15M rows) but has only five columns, so it fits comfortably in RAM on a 16 GB
machine; there is no need for chunked streaming here.

Edge-list schema (``simple_edgelist``):
    pre           str    presynaptic neuron id
    post          str    postsynaptic neuron id
    count         int    number of synapses pre->post (the threshold column)
    norm          float  count / total_input(post)  (a natural weight in [0, 1])
    total_input   int    total input synapses onto post

Meta schema (subset we use): ``fafb_783_id``, ``super_class``, ``cell_class``,
``cell_type``, ``side``, ``flow``, ``body_part_sensory``, ``neurotransmitter_predicted``,
``neurotransmitter_score``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from connectome.download import DEFAULT_DEST, FILES, ensure_data


def load_edges(path: Path | str) -> pl.DataFrame:
    """Read the simple edge list. Columns: pre, post, count, norm, total_input."""
    return pl.read_ipc(Path(path))


def load_meta(path: Path | str) -> pl.DataFrame:
    """Read the per-neuron metadata frame (keyed by ``fafb_783_id``)."""
    return pl.read_ipc(Path(path))


def ensure_and_load(
    dest: Path | str = DEFAULT_DEST, *, download: bool = True
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(meta, edges)``, downloading the dumps first if requested.

    Parameters
    ----------
    download:
        If True, fetch missing files via :func:`connectome.download.ensure_data`.
        If False, read whatever is already on disk (raises if absent).
    """
    dest = Path(dest)
    if download:
        paths = ensure_data(dest)
        meta_path, edge_path = paths["meta"], paths["simple_edgelist"]
    else:
        meta_path = dest / FILES["meta"].filename
        edge_path = dest / FILES["simple_edgelist"].filename
    return load_meta(meta_path), load_edges(edge_path)

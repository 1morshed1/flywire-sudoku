"""Filter FlyWire edges and select the neuron node set.

Design decisions (PLAN.md §2, §8):

* **Synapse-count threshold = 5.** Edges with fewer than 5 synapses are dropped as
  likely-spurious. On FAFB v783 this cuts 15.0M edges -> 3.53M, over 135,453 neurons.
* The node set is, by default, every neuron that participates in a surviving edge.
  A biologically-selected subset can instead be chosen with :func:`select_neurons`
  (e.g. exclude glia, or keep only sensory + central-brain classes).
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

MIN_SYNAPSE_COUNT = 5

# Non-neuronal / support classes we usually exclude from a "neurons only" node set.
_NON_NEURON_CLASSES = {"glia", "trachea", "visceral_circulatory"}


def filter_edges(edges: pl.DataFrame, min_count: int = MIN_SYNAPSE_COUNT) -> pl.DataFrame:
    """Return only edges with ``count >= min_count``."""
    return edges.filter(pl.col("count") >= min_count)


def neuron_ids_from_edges(edges: pl.DataFrame) -> list[str]:
    """Sorted unique neuron ids appearing as either endpoint of ``edges``.

    Sorting makes the id->index mapping deterministic across runs.
    """
    ids = pl.concat([edges["pre"], edges["post"]]).unique().sort()
    return ids.to_list()


def select_neurons(
    meta: pl.DataFrame,
    *,
    super_classes: Sequence[str] | None = None,
    exclude_non_neuron: bool = True,
) -> list[str]:
    """Choose a neuron id set from metadata (a biologically-scoped node set).

    Parameters
    ----------
    super_classes:
        If given, keep only these ``super_class`` values (e.g. ``("sensory",
        "central_brain_intrinsic")``). If None, keep all classes.
    exclude_non_neuron:
        Drop glia / trachea / circulatory support cells.

    Returns sorted ids for deterministic indexing.
    """
    frame = meta
    if exclude_non_neuron:
        frame = frame.filter(~pl.col("super_class").is_in(list(_NON_NEURON_CLASSES)))
    if super_classes is not None:
        frame = frame.filter(pl.col("super_class").is_in(list(super_classes)))
    return frame["fafb_783_id"].unique().sort().to_list()

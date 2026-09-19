"""Per-neuron 3D coordinates for the Tier-2 video (docs/VIDEO_PLAN.md).

Loads the FlyWire annotations TSV and aligns a coordinate to each neuron id. Prefers
the detected soma position (``soma_x/y/z``); falls back to a representative point on the
neuron (``pos_x/y/z``) when the soma is missing. Coordinates are in FlyWire voxel space
— fine for visualization, where only relative geometry matters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from connectome.download import COORDS_FILENAME, DEFAULT_DEST, ensure_coordinates


def load_coordinates(path: Path | str) -> pl.DataFrame:
    """Read the annotations TSV; return ``root_id`` + soma/pos xyz columns as strings→floats."""
    cols = ["root_id", "soma_x", "soma_y", "soma_z", "pos_x", "pos_y", "pos_z"]
    return pl.read_csv(
        Path(path), separator="\t", columns=cols, schema_overrides={"root_id": pl.String}
    )


def all_coordinates(
    *, dest: Path | str = DEFAULT_DEST, download: bool = True
) -> np.ndarray:
    """Return an ``(M, 3)`` array of every neuron's coordinate (soma, else position).

    This is the whole ~139k-soma point cloud used as the dim anatomical backdrop in the
    Tier-2 video (so the recognizable fly-brain shape and 3D depth read clearly).
    """
    path = ensure_coordinates(dest) if download else Path(dest) / COORDS_FILENAME
    df = load_coordinates(path).with_columns(
        [
            pl.coalesce(["soma_x", "pos_x"]).alias("x"),
            pl.coalesce(["soma_y", "pos_y"]).alias("y"),
            pl.coalesce(["soma_z", "pos_z"]).alias("z"),
        ]
    )
    xyz = df.select(["x", "y", "z"]).to_numpy().astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def neuron_coordinates(
    node_ids: list[str], *, dest: Path | str = DEFAULT_DEST, download: bool = True
) -> np.ndarray:
    """Return an ``(N, 3)`` coordinate array aligned to ``node_ids``.

    Soma coordinate where available, else the representative position; rows with no
    coordinate at all are left as NaN (callers can drop or place them). Downloads the
    annotations file first if requested.
    """
    path = ensure_coordinates(dest) if download else Path(dest) / COORDS_FILENAME
    df = load_coordinates(path)

    # Prefer soma_*, fall back to pos_* per axis.
    df = df.with_columns(
        [
            pl.coalesce(["soma_x", "pos_x"]).alias("x"),
            pl.coalesce(["soma_y", "pos_y"]).alias("y"),
            pl.coalesce(["soma_z", "pos_z"]).alias("z"),
        ]
    )
    lookup = {
        rid: (x, y, z)
        for rid, x, y, z in zip(
            df["root_id"].to_list(), df["x"].to_list(), df["y"].to_list(), df["z"].to_list()
        )
    }
    coords = np.full((len(node_ids), 3), np.nan, dtype=np.float64)
    for i, nid in enumerate(node_ids):
        xyz = lookup.get(nid)
        if xyz is not None and all(v is not None for v in xyz):
            coords[i] = xyz
    return coords

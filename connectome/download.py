"""Download FlyWire FAFB v783 connectome dumps from the public GCS mirror.

No authentication or API token is required: the files live in a public Google Cloud
Storage bucket and are fetched over plain HTTPS.

We deliberately pull only the two files the pipeline needs:

* ``meta``          — per-neuron metadata (id, class, side, predicted transmitter).
* ``simple_edgelist`` — pre->post edges already aggregated with a synapse count, which
  is all the adjacency builder needs. The ~2-4 GB raw ``synapses`` table is skipped.

Files land in ``data/flywire/`` (gitignored) and downloads are skipped when a file of
the expected size already exists, so re-running is cheap and idempotent.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Public GCS mirror of the FlyWire FAFB v783 compiled data (Lee Lab).
VERSION = "783"
_BUCKET = "lee-lab_brain-and-nerve-cord-fly-connectome"
BASE_URL = f"https://storage.googleapis.com/{_BUCKET}/compiled_data/fafb_{VERSION}"

CITATION = (
    "FlyWire FAFB connectome v783. Dorkenwald et al., Nature (2024); "
    "Schlegel et al., Nature (2024). Data via the public Lee Lab GCS mirror "
    f"(gs://{_BUCKET}/compiled_data/fafb_{VERSION})."
)


@dataclass(frozen=True)
class DataFile:
    """One downloadable dump: its logical key, remote filename, and expected size."""

    key: str
    filename: str
    size_bytes: int  # from the bucket listing; used to skip complete downloads


# Only the files the Phase 4 pipeline consumes.
FILES: dict[str, DataFile] = {
    "meta": DataFile("meta", f"fafb_{VERSION}_meta.feather", 13_539_866),
    "simple_edgelist": DataFile(
        "simple_edgelist", f"fafb_{VERSION}_simple_edgelist.feather", 302_625_658
    ),
}

DEFAULT_DEST = Path("data/flywire")


def _download_one(df: DataFile, dest_dir: Path, *, force: bool = False) -> Path:
    """Fetch a single file (streaming), skipping when already present and complete."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / df.filename
    if out.exists() and not force and out.stat().st_size == df.size_bytes:
        return out
    url = f"{BASE_URL}/{df.filename}"
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        chunk = 1 << 20
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
    tmp.replace(out)
    return out


def ensure_data(
    dest_dir: Path | str = DEFAULT_DEST,
    keys: tuple[str, ...] = ("meta", "simple_edgelist"),
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Download the requested dumps and write a provenance file; return their paths.

    Parameters
    ----------
    keys:
        Which files to fetch (see :data:`FILES`).
    force:
        Re-download even if a complete file already exists.
    """
    dest = Path(dest_dir)
    paths: dict[str, Path] = {}
    for key in keys:
        if key not in FILES:
            raise KeyError(f"unknown data file {key!r}; known: {sorted(FILES)}")
        paths[key] = _download_one(FILES[key], dest, force=force)
    _write_provenance(dest, paths)
    return paths


def _write_provenance(dest: Path, paths: dict[str, Path]) -> None:
    """Record version, source, citation, and file sizes for reproducibility."""
    prov = {
        "version": VERSION,
        "source": BASE_URL,
        "citation": CITATION,
        "files": {k: {"path": str(p), "size_bytes": p.stat().st_size} for k, p in paths.items()},
    }
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Download FlyWire FAFB v783 dumps.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    got = ensure_data(args.dest, force=args.force)
    for k, p in got.items():
        print(f"{k}: {p} ({p.stat().st_size:,} bytes)")
    print(CITATION)

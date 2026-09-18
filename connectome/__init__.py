"""FlyWire connectome data pipeline (PLAN.md Phase 4).

Turns the public FlyWire FAFB (v783) connectome dumps into a sparse, trainable
adjacency matrix for the FlyWire-constrained SNN (Phase 5).

Stages:

* :mod:`connectome.download`   — fetch dumps from the public GCS mirror (no auth).
* :mod:`connectome.loader`     — read neuron metadata + edge list (Polars/Arrow).
* :mod:`connectome.filter`     — threshold edges (synapse count >= 5), drop non-neurons.
* :mod:`connectome.adjacency`  — build a sparse COO/CSR adjacency matrix.
* :mod:`connectome.sampling`   — PRINCIPLED subgraph selection (never uniform-random).
* :mod:`connectome.statistics` — degree / connectivity summaries for reports.

Data source: FlyWire FAFB v783 (Dorkenwald et al., Nature 2024; Schlegel et al.,
Nature 2024). See :data:`connectome.download.CITATION`.
"""

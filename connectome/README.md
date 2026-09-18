# `connectome/` — Phase 4: FlyWire data pipeline

Turns the public **FlyWire FAFB v783** connectome into a sparse, trainable adjacency
matrix for the FlyWire-constrained SNN (Phase 5).

## Modules

| File | Role |
|------|------|
| `download.py` | Fetch dumps from the public GCS mirror (no auth); write `provenance.json` |
| `loader.py` | Read `meta` + `simple_edgelist` feather files (Polars/Arrow) |
| `filter.py` | Threshold edges (synapse count ≥ 5); select neuron node set |
| `adjacency.py` | Build sparse directed COO/CSR adjacency; transmitter signs; save/load |
| `sampling.py` | **Principled** subgraph selection (top-degree / BFS / BFS-from-sensory) |
| `statistics.py` | Degree / density / reciprocity / component summaries |

## Data source & provenance

Public GCS mirror (no token):
`gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/fafb_783/`

We pull only two files (~316 MB total), skipping the 2–4 GB raw synapse table:
- `fafb_783_meta.feather` (13 MB) — per-neuron metadata
- `fafb_783_simple_edgelist.feather` (302 MB) — pre→post edges with synapse `count`

Cite: **Dorkenwald et al., Nature (2024)** and **Schlegel et al., Nature (2024)**.
`data/flywire/provenance.json` records version, source, sizes, citation.

```bash
uv run --no-sync python -m connectome.download        # downloads to data/flywire/
```

## Real-data facts (FAFB v783, synapse count ≥ 5)

| Quantity | Value |
|----------|-------|
| Neurons (nodes) | 135,453 |
| Edges | 3,532,517 |
| Density | 2.0e-4 |
| Mean degree (in=out) | 26.1 |
| Largest weak component | 98.8% |
| Reciprocity | 0.14 |
| Transmitter signs | +1: 92,552  −1: 42,901 |

## Why sampling must be principled (PLAN.md §7, §17)

A uniform-**random** 1,000-node pick gives ~200 edges and a largest component of ~2%
(dead islands). Principled methods on the same budget:

| Method (N=1000) | Edges | Largest WCC | Mean degree |
|-----------------|------:|------------:|------------:|
| `top_degree` | 22,181 | 100% | 22.2 |
| `bfs_sensory` | 16,202 | 100% | 16.2 |
| `random` (control only) | 206 | 1.9% | 0.2 |

`random` exists solely as an ablation control (Phase 6), never as the default.

## Transmitter → sign (Dale's law modeling choice)

acetylcholine → **+1**; GABA, glutamate → **−1**; dopamine/serotonin/octopamine → +1;
unknown → +1. Used for the sign-constrained ablation (PLAN.md §8, Regime B). This is a
documented assumption, revisited in Phase 6.

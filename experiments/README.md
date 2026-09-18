# `experiments/` — research runs

## `topology_ablation.py` (Phase 6)

The central experiment: **does FlyWire's biological wiring help learning, or would any
graph with the same statistics do as well?**

Trains `FlyWireSNN` under several topology conditions at a **fixed neuron count and
matched edge budget**, over multiple seeds, and reports move accuracy / solve rate plus
graph statistics.

### Conditions (`connectome/topology.py`)

| Topology | What it controls for |
|----------|----------------------|
| `flywire` | the real connectome mask |
| `shuffled` | **key null** — degree-preserving shuffle (same in/out degree, structure destroyed) |
| `random` | Erdős–Rényi, matched edge count only |
| `dense` | all-to-all upper bound |
| `feedforward` | layered DAG, no recurrence |

The load-bearing comparison is **flywire vs shuffled**: if FlyWire wins at equal degree
sequence, structure beyond degree matters.

### Structural signature (real 1k top-degree subgraph)

All conditions share N=1000, ~22k edges, mean degree 22.2. What differs:

| Topology | reciprocity |
|----------|------------:|
| flywire | 0.405 |
| shuffled | 0.038 |
| random | 0.022 |
| feedforward | 0.000 |

FlyWire's high reciprocity is a biological signature the controls lack by construction.

### Run

```bash
uv run --no-sync python -m experiments.topology_ablation --quick            # fast 4x4 smoke
uv run --no-sync python -m experiments.topology_ablation --config <cfg>.yaml
```

Results are written to `results/topology_ablation.json` (gitignored) and printed as a
per-condition summary (mean over seeds). Headline numbers are logged in
`memory-bank/progress.md`.

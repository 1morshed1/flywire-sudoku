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

### First result (4×4 easy, N=1000, 30 epochs, 3 seeds)

| topology | move_acc | solve_rate | reciprocity |
|----------|---------:|-----------:|------------:|
| flywire | 0.936 | 0.667 | 0.405 |
| shuffled | 0.934 | 0.654 | 0.036 |
| random | 0.934 | 0.651 | 0.023 |

**Null result:** the topologies perform equally within seed noise (±0.01) despite a
20× spread in reciprocity. On 4×4-easy the task **saturates** — any connected graph of
matched degree suffices, so biological structure confers no measurable advantage here.
Finding topology separation (if it exists) needs a harder or capacity-pressured regime:
9×9, a smaller N, or fewer clues. Reported as-is — a negative result at saturation.

### Capacity-pressured regime (N=200, 4×4 easy, 3 seeds)

| topology | move_acc | solve_rate | reciprocity |
|----------|---------:|-----------:|------------:|
| flywire | 0.921 | 0.605 | 0.605 |
| shuffled | 0.920 | 0.595 | 0.060 |
| random | 0.919 | 0.580 | 0.044 |

**Faint ordered hint (3 seeds):** solve_rate followed the reciprocity ordering
(flywire ≥ shuffled ≥ random), but the gap was inside seed noise.

### Power test (N=200, 4×4 easy, 8 seeds)

| topology | solve_rate (mean ± std) | reciprocity |
|----------|------------------------:|------------:|
| flywire | 0.590 ± 0.027 | 0.605 |
| shuffled | 0.587 ± 0.029 | 0.061 |
| random | 0.575 ± 0.030 | 0.046 |

flywire − random = **0.015** (se 0.014, z ≈ 1.05) → **not significant**. The 3-seed
hint shrank with more seeds.

### Verdict: no topology separation at accessible scale

Across 4×4-easy (saturated) and N=200 (capacity-pressured, 8 seeds), the FlyWire mask
performs equal to a degree-preserving shuffle and to a random graph, within seed noise,
despite 10–13× differences in reciprocity. **Biological wiring confers no measurable
advantage on Sudoku here.** FlyWire is consistently numerically highest but it is a
whisper, not a signal. This is a legitimate negative result. Pushing to 9×9, larger N,
or harder difficulty could still surface an effect, but the current pattern suggests
the same null.


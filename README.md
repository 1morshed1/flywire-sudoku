# FlyWire-Sudoku

A **FlyWire-connectome-constrained spiking neural network (SNN)** trained to solve
Sudoku. The Sudoku task is conventional supervised learning (with optional RL
polish); the scientific interest is that the network's recurrent connectivity is
**constrained by the real FlyWire *Drosophila* connectome**.

> **Research question:** How does biological connectivity constrain learning on an
> abstract symbolic-reasoning task?

This is *not* an attempt to rebuild a fly brain. Biological wiring is a fixed
architectural constraint; experiments measure whether that wiring helps, hurts, or
is neutral versus matched controls (random / degree-matched / dense / feed-forward).

Full design: [`PLAN.md`](PLAN.md). Live state: [`memory-bank/`](memory-bank/).
Agent guide: [`CLAUDE.md`](CLAUDE.md).

---

## Status

Phases **0–7 are complete**, including Tier 1 / Tier 2 solve videos. Test suite:
~65 tests. Topology ablation at accessible scale is a **null result**; the
constraint-propagation solver loop lifts all models to near-perfect 4×4 completion.

| Phase | What | Status |
|-------|------|--------|
| 0 | Environment (uv, torch+Norse, CUDA on RTX 2060) | **done** |
| 1 | Sudoku generator / solver / env / renderer (4×4 → 9×9) | **done** |
| 2 | MLP baseline | **done** |
| 3 | Dense SNN (Norse LIF) | **done** |
| 4 | FlyWire data → sparse adjacency + principled sampling | **done** |
| 5 | FlyWire SNN scaling (1k→20k trainable ceiling) | **done** |
| 6 | Topology ablations | **done** (null at 4×4) |
| 7 | Autonomous solver (constraint loop) + videos | **done** |

**Open follow-ups:** 9×9 generalization (current 1k/6k run overfits unseen seeds),
research writeup, efficiency metrics, sparse-gradient autograd past ~20k neurons,
optional RL.

---

## Key results (honest)

| Finding | Detail |
|---------|--------|
| 4×4 FlyWire-SNN learns | ~0.93 move_acc / ~0.65 single-shot solve_rate @ 1k neurons |
| Constraint loop >> single-shot | 4×4 loop completion: MLP 0.97, dense 0.96, FlyWire **1.00** |
| Topology ablation (null) | FlyWire ≈ degree-shuffled ≈ random on 4×4 easy despite ~10× reciprocity gap |
| Trainable VRAM ceiling | ~**20k** neurons on 6 GB; 30k+ OOM (`torch.sparse.mm` backward → dense N×N grad) |
| 9×9 caveat | Earlier “solves ~65%” was train-seed leakage; real unseen completion ~**0%** until more data / regularization |

---

## Hardware target

Built and tuned for a **consumer GPU** — reproducible on accessible hardware is a
design goal.

| Resource | Value |
|----------|-------|
| GPU | NVIDIA RTX 2060, **6 GB** VRAM, compute 7.5 (Turing) |
| CPU | 12 cores |
| RAM | 15 GB |

**Trainable ceiling on 6 GB:** ~10k–20k neurons, batch 32–64, T=10. Full ~139k
connectome is **inference / visualization only**.

---

## Approach (one screen)

```text
Sudoku board (N×N int matrix)
  -> per-cell one-hot (0=empty, 1..N)        # 9×9: 810 features; 4×4: 80
  -> Poisson rate coding -> spike trains (T timesteps)
  -> input projection (dense)
  -> FlyWire recurrent LIF:  W_eff = W_trainable * A   # A = fixed sparse adjacency
  -> output projection (dense)
  -> logits (cells × digits), legal-move mask (illegal -> -inf)
  -> place digit
```

**Autonomous solve** = constraint-propagation loop: score every empty cell, place
the globally most-confident *legal* digit, repeat until solved (or stuck). RL is
optional fine-tuning, not the spine.

**The science** = topology ablations at matched neuron/edge budget:
FlyWire vs random vs degree-preserving shuffle vs dense vs feed-forward.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with CUDA 12.x drivers.

```bash
uv sync

# verify torch sees the GPU
uv run --no-sync python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Note:** `norse` is the fragile pin against torch/Python. If `uv sync` fails to
> resolve, see the risk notes in [`pyproject.toml`](pyproject.toml).

### Common commands

```bash
uv run --no-sync pytest tests/ -q

# download FlyWire FAFB v783 -> data/flywire/
uv run --no-sync python -m connectome.download

# train
uv run --no-sync python -m training.supervised --config configs/mlp_baseline.yaml
uv run --no-sync python -m training.supervised --config configs/dense_snn.yaml
uv run --no-sync python -m training.supervised --config configs/flywire_snn_1k.yaml
uv run --no-sync python -m training.supervised --config configs/flywire_snn_9x9.yaml

# topology ablation (fast 4×4 sweep)
uv run --no-sync python -m experiments.topology_ablation --quick

# solve videos (needs a saved state + connectome data)
uv run --no-sync python -m evaluation.visualization --tier 1 --side 4 \
  --state results/flywire4_state.pt --device cuda --out results/flywire_solve.mp4
uv run --no-sync python -m evaluation.visualization --tier 2 --side 9 \
  --state results/flywire9_state.pt --device cuda --out results/flywire_fx_9x9.mp4
```

Large feathers / results under `data/` and `results/` are gitignored.

---

## Repository layout

```text
flywire-sudoku/
  sudoku/         SudokuSpec, solver, generator, encoder, environment, renderer
  models/         mlp, dense_snn, flywire_snn  (shared I/O: board → logits)
  connectome/     download, loader, filter, adjacency, sampling, topology, pipeline
  training/       dataset (disk cache), supervised (TrainConfig + YAML dispatch)
  experiments/    topology_ablation
  evaluation/     solver_loop, metrics, visualization (Tier 1 + Tier 2 video)
  configs/        mlp_baseline, dense_snn, flywire_snn_1k, flywire_snn_9x9
  tests/          one test_*.py per package
  memory-bank/    live project state for agents / resume
  PLAN.md         locked design decisions
```

---

## Data & provenance

FlyWire connectome from the **public FlyWire Codex parquet dumps** (FAFB v783, no
auth). Edges kept at synapse count ≥ 5 (~135k nodes, ~3.5M edges after filter).
Subgraphs are built with **principled** methods only (largest WCC / BFS-from-sensory /
top-degree) — never random. Dump version, threshold, and sampling method are logged
with each run. Cite and license per FlyWire / Codex terms.

---

## Documentation

Design decisions live in [`PLAN.md`](PLAN.md). Current focus and known issues live in
[`memory-bank/`](memory-bank/) (`activeContext.md`, `progress.md`). Package-level
READMEs sit under each top-level package.

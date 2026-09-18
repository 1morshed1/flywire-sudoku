# FlyWire-Sudoku

A **FlyWire-connectome-constrained spiking neural network (SNN)** trained to solve
Sudoku. The Sudoku task is conventional supervised learning (with optional RL
polish); the scientific interest is that the network's recurrent connectivity is
**constrained by the real FlyWire *Drosophila* connectome**.

> **Research question:** How does biological connectivity constrain learning on an
> abstract symbolic-reasoning task?

This is *not* an attempt to rebuild a fly brain. It uses biological wiring as a
fixed architectural constraint and measures whether that wiring helps, hurts, or is
neutral versus matched controls (random / degree-matched / dense topologies).

---

## Status

Early scaffolding. Design is locked; no model code yet.

| Phase | What | Status |
|-------|------|--------|
| 0 | Environment (uv venv, torch+Norse, verify CUDA) | in progress |
| 1 | Sudoku generator / solver / env / renderer (4×4 first) | todo |
| 2 | MLP baseline | todo |
| 3 | Dense SNN (Norse LIF) | todo |
| 4 | FlyWire data → sparse adjacency | todo |
| 5 | FlyWire SNN scaling (1k→10k→20k) | todo |
| 6 | Topology ablations | todo |
| 7 | Autonomous solver (constraint loop) + optional RL | todo |

Full design: [`PLAN.md`](PLAN.md). Live project state: [`memory-bank/`](memory-bank/).
Agent working guide: [`CLAUDE.md`](CLAUDE.md).

---

## Hardware target

Built and tuned for a **consumer GPU** — reproducible on accessible hardware is a
design goal, not an afterthought.

| Resource | Value |
|----------|-------|
| GPU | NVIDIA RTX 2060, **6 GB** VRAM, compute 7.5 (Turing) |
| CPU | 12 cores |
| RAM | 15 GB |

**Trainable ceiling on 6 GB:** ~10k–20k neurons, batch 32–64, T=10 timesteps, with
AMP (mixed precision) + gradient checkpointing. The full ~139k-neuron connectome is
**inference/eval only** on this box.

---

## Approach (one screen)

```text
Sudoku board (N×N int matrix)
  -> per-cell one-hot (0=empty, 1..N)        # 9×9: 810 features; 4×4: 80
  -> Poisson rate coding -> spike trains (T timesteps)
  -> input projection (dense)
  -> FlyWire recurrent LIF neurons:  W_eff = W_trainable * A   # A = fixed sparse adjacency
  -> output projection (dense)
  -> 729 logits (81 cells × 9 digits)  [or 9 for next-cell task]
  -> legal-move mask (illegal -> -inf)
  -> place digit
```

**Autonomous solve** = constraint-propagation loop: score every empty cell, place
the globally most-confident *legal* digit, repeat until solved. No RL required for
the spine; RL is optional fine-tuning.

**The science** = topology ablations at matched budget:
FlyWire vs random vs degree-matched-random vs dense vs feed-forward.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with CUDA 12.x drivers.

```bash
# create venv + install all deps (torch pulled from the cu124 wheel index)
uv sync

# verify torch sees the GPU
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Note:** `norse` (the spiking-NN layer) is the fragile pin against torch/Python
> versions. If `uv sync` fails to resolve, see the risk notes in
> [`pyproject.toml`](pyproject.toml) for fallbacks.

FlyWire data dependencies are added in **Phase 4** (not part of the Phase 0 env).

---

## Repository layout

Directories are created **per phase**, not all up front. Target layout:

```text
flywire-sudoku/
  data/          flywire/ (connectome parquet)  sudoku/ (train/val/test)
  connectome/    loader, filter, adjacency, sampling, statistics
  sudoku/        generator, solver, environment, encoder, renderer
  models/        mlp, dense_snn, flywire_snn
  training/      supervised, reinforcement, curriculum
  experiments/   baseline, topology_ablation, scaling, rl
  evaluation/    accuracy, sudoku_metrics, visualization
  configs/       *.yaml (config-driven runs)
```

---

## Data & provenance

FlyWire connectome data is obtained from the **public FlyWire Codex parquet dumps**
(no auth). Edges are thresholded at synapse count ≥ 5. The exact dump version,
threshold, and subsampling method are logged with each run. FlyWire data carries its
own citation and license requirements — see Phase 4 docs when added.

---

## Documentation

This project is documented thoroughly by intent: every module carries docstrings,
each phase directory gets its own README, and design decisions + data provenance are
recorded in `PLAN.md` and `memory-bank/`.

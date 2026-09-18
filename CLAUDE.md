# CLAUDE.md — Agent Guide (FlyWire-Sudoku)

## Project

FlyWire-connectome-constrained SNN for Sudoku symbolic reasoning. Scientific goal:
measure how biological connectivity constrains learning — not rebuild a fly brain.

**Source of truth:** `PLAN.md`. **Live state:** `memory-bank/`. Read memory bank
before non-trivial work; update `activeContext.md` + `progress.md` after significant
changes.

## Non-negotiables

1. Hardware: RTX 2060 **6 GB**. Measured FlyWire-SNN trainable ceiling ≈ **20k neurons**
   (20k = 2.4 GB @ batch 64, T=10). 30k+ OOMs even at batch 16 because
   `torch.sparse.mm` backward materializes a **dense N×N** gradient for the sparse
   weights (30k² × 4B ≈ 3.35 GB) — so the limit is O(N²) backward, not batch. Going
   past ~20k needs a custom sparse-gradient autograd (future work). Full 139k =
   inference/eval only.
2. Package manager: **uv**. Eyeball deps before install.
3. Framework: **Norse** + torch CUDA 12.x. Pin versions at install.
4. Connectome: public Codex parquet; synapse count **≥ 5**; signs-free first.
5. Subsampling: **principled only** (largest WCC / BFS-from-sensory / top-degree).
   Never random.
6. Solve spine: **constraint-propagation loop**, not per-step RL. RL optional later.
7. Curriculum: **4×4 first**, then 9×9. Size-generic code.
8. Repo dirs grow **per phase** — do not scaffold entire tree up front.
9. Phase gate: if MLP (Phase 2) fails, stop — do not touch FlyWire.
10. No claims of “biological intelligence”; connectivity = architectural constraint.

## Explicitly do not (initially)

- Train full 139k brain
- Image-based Sudoku
- RL from scratch as primary method
- Transformer + FlyWire together
- Report accuracy only (include efficiency metrics)

## Architecture reminder

`W_eff = W_trainable * A` with `A` fixed sparse FlyWire adjacency. LIF + surrogate
BPTT. Legal-action mask on logits. Autonomous solve: score all empty cells → place
globally most-confident legal digit → repeat.

## Phases

0 env → 1 sudoku → 2 MLP → 3 dense SNN → 4 FlyWire data → 5 scaling →
6 topology ablations → 7 autonomous solve

Phases 0–6 DONE; Phase 7 pending. Current status/decisions live in `memory-bank/`.

## Commands

All commands use `uv run --no-sync` (env already synced).

```bash
uv sync                                                   # install deps (first time)
uv run --no-sync pytest tests/ -q                         # full test suite
uv run --no-sync ruff check . --fix && uv run --no-sync ruff format .   # lint + format
uv run --no-sync python -m connectome.download            # fetch FAFB v783 -> data/flywire/
uv run --no-sync python -m training.supervised --config configs/mlp_baseline.yaml
uv run --no-sync python -m training.supervised --config configs/dense_snn.yaml
uv run --no-sync python -m training.supervised --config configs/flywire_snn_1k.yaml
uv run --no-sync python -m experiments.topology_ablation --quick   # fast 4x4 sweep
```

Notes: training is config-driven (`configs/*.yaml` mirror `TrainConfig`). GPU runs
(5k-puzzle 9×9 gen + train) take minutes — run in background. Big feathers + results are
gitignored under `data/`, `results/`.

## Repo map

```text
sudoku/       core (SudokuSpec), solver, generator, encoder, environment, renderer
models/       mlp, dense_snn, flywire_snn — shared I/O: one-hot board -> (B, cells, side)
connectome/   download, loader, filter, adjacency, sampling, topology, pipeline, statistics
training/     dataset, supervised (train_supervised + build_model dispatch, TrainConfig)
experiments/  topology_ablation
tests/        one test_*.py per package
configs/      *.yaml per model / experiment
```

## Memory bank

| File | Role |
|------|------|
| `projectbrief.md` | Scope, goals, out-of-scope |
| `productContext.md` | Why / how / UX for researcher |
| `systemPatterns.md` | Architecture, ablations, phases |
| `techContext.md` | Hardware, stack, VRAM |
| `activeContext.md` | Current focus, next steps |
| `progress.md` | Status, risks |

When user says **update memory bank**, review all files; prioritize active + progress.

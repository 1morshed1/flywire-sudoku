# Progress

## What works

- Finalized plan (`PLAN.md`) with locked design decisions
- Memory bank + `CLAUDE.md` scaffolding
- **Phase 0 env verified**: uv venv, torch 2.5.1+cu124, norse 1.1.0, CUDA sees
  RTX 2060, GPU matmul OK. `pyproject.toml` + `uv.lock` committed.
- **Phase 1 Sudoku stack**: `sudoku/` package (core/solver/generator/encoder/
  environment/renderer), size-generic (4×4 + 9×9). 25 pytest passing in ~5.5s.
  Solver MRV backtracking; generator guarantees unique solution; env has
  place/next_digit tasks + legal mask. ruff clean.
- **Phase 2 MLP baseline**: `models/mlp.py` (SudokuMLP), `training/dataset.py` +
  `training/supervised.py` (masked CE over empty cells, move_acc + solve_rate,
  TrainConfig + YAML). 29 tests pass. Gate met: 4×4 easy move_acc ~0.94 /
  solve_rate ~0.68 (2k puzzles, 40 epochs, CPU).
  9×9 easy baseline (5k puzzles, 30 epochs, RTX 2060, 257s): move_acc 0.29,
  solve_rate 0.0, ~734k params — UNDERFIT, still climbing (loss 1.27→0.96).
  Single-shot full-grid MLP is inherently weak on 9×9 (chance ~0.11, so 0.29 is
  real learning); one-shot solve is not the goal — motivates the Phase 7 iterative
  constraint-loop. Baseline = reference point, not a solver.
- **Phase 3 dense SNN**: `models/dense_snn.py` (SudokuDenseSNN, Norse LIFCell layers
  + LICell readout, T-step sim inside forward, constant/poisson encoding). Training
  loop generalized: `train_supervised` dispatches via `build_model`; AMP added
  (torch.amp autocast + GradScaler). 33 tests pass. Gate met: 4×4 easy move_acc
  ~0.94 / solve_rate ~0.70 @50 epochs (CPU). 9×9 GPU+AMP: pending.

## What's left

| Phase | Status | Notes |
|-------|--------|-------|
| 0 env (uv, torch, Norse, CUDA check) | **DONE** | torch 2.5.1+cu124, norse 1.1.0, RTX 2060 ✓ |
| 1 Sudoku 4×4 then 9×9 | **DONE** | 25 tests pass; solver/gen/env/encoder/renderer |
| 2 MLP baseline | **DONE** | gate met (4×4 acc ~0.94); 9×9 logged |
| 3 dense SNN (Norse LIF, T=10, AMP) | **DONE** | 4×4 acc ~0.94 @50ep; 9×9 GPU number pending |
| 4 FlyWire Codex → sparse COO + sampling | pending | no GPU |
| 5 FlyWire-SNN scale 1k→20k | pending | VRAM bench each rung |
| 6 topology ablations | pending | science core |
| 7 autonomous constraint loop (+ opt RL) | pending | GPU |

## Current status

Phases 0-3 complete. MLP + dense SNN both learn the task (gates passed). Next:
Phase 4 — FlyWire connectome data (Codex public parquet dumps → filter edges
count≥5 → sparse COO adjacency → principled subsampling). No GPU; data engineering.

## Known issues / risks

- 6 GB VRAM hard ceiling; 50k+ train won't fit
- Norse resolved fine vs torch 2.5 on py3.12 (no fallback needed)
- Random connectome subsample → dead islands (must use principled methods)
- Multi-GB parquet needs Polars streaming, not eager Pandas
- Solver/count_solutions now reject invalid boards upfront (fixed an exponential
  backtracking hang on contradictory grids — see solver.py guards)

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
  ~0.94 / solve_rate ~0.70 @50 epochs (CPU). 9×9 GPU+AMP (5k, 30ep, 274s):
  move_acc 0.209, underfit — AMP verified working on RTX 2060, no crash.
- **Phase 4 FlyWire data**: `connectome/` package (download/loader/filter/adjacency/
  sampling/statistics). Public GCS mirror (FAFB v783, no auth) → meta (13MB) +
  simple_edgelist (302MB). count≥5 → 135,453 nodes, 3.53M edges, largest WCC 98.8%.
  Sparse scipy COO/CSR adjacency; transmitter signs (+92552/−42901). PRINCIPLED
  sampling (top_degree/bfs/bfs_sensory all give connected subgraphs; random control
  = dead islands, confirmed). 11 tests pass (incl. real-data integration).
- **Phase 5 FlyWire SNN**: `models/flywire_snn.py` (FlyWireSNN), `connectome/
  pipeline.py` (cached full graph + subgraph API). Recurrent LIF, W_eff = w_edge ⊙ A
  via torch.sparse.mm (autograd on edge weights, orientation test passes). 3 regimes
  (trainable/frozen/signs). Wired into train_supervised + build_model + config.
  50 tests pass. WORKS: 4×4 real 1k subgraph move_acc ~0.93 / solve_rate ~0.65.
  VRAM scaling (RTX 2060, batch 64, T=10): 1k=59MB, 5k=306MB, 10k=795MB, 20k=2.4GB
  — sparse recurrence → ~40k feasible (ceiling revised up from dense estimate).

## What's left

| Phase | Status | Notes |
|-------|--------|-------|
| 0 env (uv, torch, Norse, CUDA check) | **DONE** | torch 2.5.1+cu124, norse 1.1.0, RTX 2060 ✓ |
| 1 Sudoku 4×4 then 9×9 | **DONE** | 25 tests pass; solver/gen/env/encoder/renderer |
| 2 MLP baseline | **DONE** | gate met (4×4 acc ~0.94); 9×9 logged |
| 3 dense SNN (Norse LIF, T=10, AMP) | **DONE** | 4×4 acc ~0.94; 9×9 AMP verified |
| 4 FlyWire Codex → sparse COO + sampling | **DONE** | FAFB v783; 135k nodes/3.5M edges |
| 5 FlyWire-SNN scale 1k→20k | **DONE** | works (4×4 ~0.93); VRAM benchmarked, ~40k feasible |
| 5 FlyWire-SNN scale 1k→20k | pending | VRAM bench each rung |
| 6 topology ablations | pending | science core |
| 7 autonomous constraint loop (+ opt RL) | pending | GPU |

## Current status

Phases 0-5 complete. The core experiment runs: FlyWire-constrained SNN learns Sudoku
(4×4 real 1k subgraph ~0.93). Model ladder done (MLP / dense SNN / FlyWire SNN). Next:
Phase 6 — topology ablations (the science): FlyWire vs degree-matched-random vs random
vs dense/feed-forward at matched budget; the `random` sampler + a degree-preserving
shuffle are the key controls. Then Phase 7 autonomous constraint-loop solver.

## Known issues / risks

- 6 GB VRAM hard ceiling; 50k+ train won't fit
- Norse resolved fine vs torch 2.5 on py3.12 (no fallback needed)
- Random connectome subsample → dead islands (must use principled methods)
- Multi-GB parquet needs Polars streaming, not eager Pandas
- Solver/count_solutions now reject invalid boards upfront (fixed an exponential
  backtracking hang on contradictory grids — see solver.py guards)

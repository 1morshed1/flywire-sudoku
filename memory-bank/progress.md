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
  VRAM scaling (RTX 2060, batch 64, T=10): 1k=59MB, 5k=306MB, 10k=795MB, 20k=2.4GB.
  MEASURED trainable ceiling ≈ 20k. 30k+ OOMs even at batch 16 — torch.sparse.mm
  BACKWARD makes a dense N×N gradient (30k²×4B≈3.35GB); cap is O(N²) backward, not
  batch. Past ~20k needs a custom sparse-gradient autograd (future work). [Earlier
  "~40k feasible" was a bad extrapolation — corrected.]
- **Phase 6 topology ablations**: `connectome/topology.py` (degree_preserving_shuffle
  [key null, per-node in+out degree preserved exactly], erdos_renyi_like, dense,
  feedforward; all matched N + edge count). `topology`/`topology_seed` in TrainConfig +
  build_model. `experiments/topology_ablation.py` runner (conditions × seeds → JSON +
  table). 7 topology tests pass (57 total). Structural signature: flywire reciprocity
  0.405 vs shuffled 0.038 vs random 0.022 vs feedforward 0.000 (matched degree).
  ABLATION RESULT (4×4 easy, N=1000, 30ep, 3 seeds): flywire move_acc 0.936 /
  solve_rate 0.667; shuffled 0.934/0.654; random 0.934/0.651. NULL RESULT —
  topologies equal within seed noise (±0.01) despite 20× reciprocity gap. 4×4 easy
  SATURATES; any connected matched-degree graph learns it. Topology separation needs
  a harder / capacity-pressured regime (9×9, or smaller N, or fewer clues). This is a
  legit negative finding, not a bug.

## What's left

| Phase | Status | Notes |
|-------|--------|-------|
| 0 env (uv, torch, Norse, CUDA check) | **DONE** | torch 2.5.1+cu124, norse 1.1.0, RTX 2060 ✓ |
| 1 Sudoku 4×4 then 9×9 | **DONE** | 25 tests pass; solver/gen/env/encoder/renderer |
| 2 MLP baseline | **DONE** | gate met (4×4 acc ~0.94); 9×9 logged |
| 3 dense SNN (Norse LIF, T=10, AMP) | **DONE** | 4×4 acc ~0.94; 9×9 AMP verified |
| 4 FlyWire Codex → sparse COO + sampling | **DONE** | FAFB v783; 135k nodes/3.5M edges |
| 5 FlyWire-SNN scale 1k→20k | **DONE** | works (4×4 ~0.93); VRAM benchmarked, ~40k feasible |
| 6 topology ablations | **DONE** | topology controls + ablation runner; degree-shuffle null verified |
| 7 autonomous constraint loop (+ opt RL) | pending | GPU |

## Current status

Phases 0-6 complete. Model ladder + topology-ablation machinery done. Next: Phase 7 —
autonomous constraint-propagation solve loop (score empty cells, place most-confident
legal digit, repeat). Uses sudoku env + a trained model; add evaluation/ metrics
(puzzle completion, steps-to-solve, invalid-rate) per PLAN.md §22.

## Known issues / risks

- 6 GB VRAM ceiling: FlyWire-SNN measured ~20k neurons trainable (20k=2.4GB); 30k+ OOM
  because torch.sparse.mm backward allocates a dense N×N gradient. Scaling study caps at
  ~20k unless a custom sparse-gradient autograd is written (future work).
- 4×4-easy ablation saturates (topology null); need harder regime for separation.
- Norse resolved fine vs torch 2.5 on py3.12 (no fallback needed)
- Random connectome subsample → dead islands (must use principled methods)
- Multi-GB parquet needs Polars streaming, not eager Pandas
- Solver/count_solutions now reject invalid boards upfront (fixed an exponential
  backtracking hang on contradictory grids — see solver.py guards)

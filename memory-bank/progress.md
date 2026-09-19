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
- **Phase 7 autonomous solver**: `evaluation/solver_loop.py` (solve_with_model +
  SolveResult; constraint-propagation loop, legal-masked greedy placement, records
  trajectory for the video) + `evaluation/metrics.py` (evaluate_solver: completion /
  stuck / steps / placement-accuracy). `train_supervised(..., return_model=True)` added.
  4 tests pass (oracle 100%, legal-only, clues untouched, metrics). RESULT: iterative
  loop >> single-shot — MLP 4×4 completion_rate 0.98 vs single-shot solve_rate 0.68.
  CROSS-MODEL (4×4 easy, 100 puzzles, loop completion): MLP 0.970, dense_snn 0.960,
  flywire_snn 1.000. Loop lifts all models to near-perfect; FlyWire SNN matches/edges
  baselines — biological wiring is a viable solver substrate. (Near ceiling at 4×4.)
- **Path A (topology separation hunt) — CONCLUDED, NULL**: two regimes tested.
  4×4-easy (saturated): flywire/shuffled/random all ~0.935 move_acc, equal within noise.
  N=200 4×4 capacity bottleneck, 8 seeds: flywire solve_rate 0.590±0.027 / shuffled
  0.587±0.029 / random 0.575±0.030. flywire−random = 0.015, se 0.014, z≈1.05 → NOT
  significant. The 3-seed hint (0.025) shrank to 0.015 with more seeds. VERDICT: no
  topology separation at accessible scale — biological wiring gives no measurable
  advantage on Sudoku despite 10–13× reciprocity differences. flywire is consistently
  numerically highest but it's a whisper, not a signal. Legit negative result. Could
  push 9×9 / larger N / harder difficulty, but pattern suggests same null.

- **Video Tier 1 + Tier 2 DONE**: `models/flywire_snn.py` forward gained
  `return_activity` (spikes (T,B,N)); `solve_with_model(capture_activity=True)` logs
  per-step (T,N) spikes; `evaluation/visualization.py` has render_solve_video (Tier 1:
  board ‖ spike raster) and render_solve_video_3d (Tier 2: 3D neurons in real soma
  coords lighting up ‖ board) + make_solve_video CLI (--tier). Soma coords:
  `connectome/coordinates.py` + `download.ensure_coordinates` (annotations TSV ~31MB,
  soma_x/y/z; all 1000 subgraph neurons covered). imageio-ffmpeg dep. 4 video tests
  pass. Verified: results/flywire_solve.mp4 (Tier1) + flywire_solve_3d.mp4 (Tier2).
  Tier2 render fixed after user feedback ("don't see neurons firing, make it slower/
  longer"): was averaging spikes over T (flat dark cloud, 4s). Now animates PER
  TIMESTEP — dim grey brain + bright cyan spiking-neuron overlay that blinks; frames =
  steps×T×frames_per_timestep + hold. Result: 25s, visible firing (~150 frames @ 6fps).

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
| 7 autonomous constraint loop | **DONE** | solver loop + metrics; MLP completion 0.98 vs 0.68 single-shot |

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

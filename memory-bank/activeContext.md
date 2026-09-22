# Active Context

## RESUME HERE (B+C+D — ALL DONE)

Honest UNSEEN(seed=1234) scoreboard for 9×9 autonomous solve:

| Run | completion | placement | notes |
|-----|------------|-----------|-------|
| FlyWire 1k drop0.3 (earlier) | **0.01** | **0.74** | still best FlyWire |
| A loop-train 1k | 0.01 | 0.74 | no gain |
| C MLP 40k | **0.02** | **0.77** | best overall; still ~0% solves |
| C dense SNN 40k | 0.00 | 0.69 | did not learn (move_acc~0.11) |
| D 1k drop0+cosine 60ep | 0.00 | 0.73 | worse than drop0.3 |
| B curriculum 4→6→9 | 0.00 | 0.73 | 4×4 0.97 / 6×6 0.83; 9×9 resets |

**Verdict:** No recipe yields real 9×9 completion. MLP slightly edges FlyWire on
placement; capacity/dropout/curriculum/loop-train all fail to unlock solves.
Natural next: research writeup (E) or supervise greedy (cell,digit) actions.

## Current focus

ALL 8 phases done + video (both tiers, 4×4 & 9×9) done and user-approved. Project is a
complete tested whole. AWAITING USER PICK on what's next (menu below) — do not start
until they choose.

## What's next (menu presented to user; recommend 1 then 3)

1. **Fix 9×9 generalization (recommended).** 9×9 flywire OVERFITS (solves training-seed
   puzzles, ~0% unseen). Fix: n_train 6k→40k + weight_decay/dropout + maybe larger N.
   Makes the 9×9 solve/video real, not in-distribution-only. ~1 train run.
2. **Deeper topology ablation** — 9×9 / larger N / harder difficulty; add dense +
   feedforward conditions. Likely still null but closes the science.
3. **Research writeup** — no summary doc yet; user wanted this as a research project.
   Report: capability (connectome SNN learns + solves Sudoku) + honest null (topology ≈
   controls) + methods + limitations.
4. **Efficiency metrics** (PLAN §22, unreported): spikes/inference, latency, energy proxy.
5. **Break 20k VRAM ceiling** — custom sparse-gradient autograd (kills O(N²) backward).
6. **RL fine-tuning** (Phase 8, deferred).

## Video (Tier 1 + Tier 2) — DONE, approved

`evaluation/visualization.py`: render_solve_video (Tier1 board‖raster) +
render_solve_video_3d (Tier2 viral-style 3D fly-brain) + make_solve_video CLI
(--tier 1|2, --state, --side, --device). Tier2 look: full ~139k soma mist backdrop
(`connectome/coordinates.py::all_coordinates`, z-scaled ×10 for anisotropy), neurons
region-colored, firing glow + white core, active synapse edges, per-region rate panel,
FIXED frontal view (elev80/azim-90, zoom 1.1, no rotation), require solved + 3s hold.
Build: `uv run --no-sync python -m evaluation.visualization --tier 2 --side 9 --state
results/flywire9_state.pt --device cuda --out results/flywire_fx_9x9.mp4`. Verified 9×9
start→end full solve, full brain in view. Outputs gitignored.

## 9×9 caveat (IMPORTANT, honest)

9×9 flywire (1k neurons, 6k train, 60ep) OVERFITS. Earlier "solves 65%" was DATA
LEAKAGE (evaluate_solver default seed == train seed). Real: ~70% on training-seed
puzzles, ~0% unseen. The 9×9 video draws its puzzle from the training seed stream to
guarantee a solve to show. 4×4 generalizes fully (loop completion ~1.0 on unseen).

## Paths A + B outcomes (done)

- **A (topology hunt): NULL at accessible scale.** 4×4-easy saturates; N=200 8-seed
  power test → flywire 0.590 vs random 0.575, z≈1.05, not significant. Biological
  wiring gives no measurable Sudoku advantage despite big reciprocity gaps. Legit
  negative result. (9×9/larger-N could still be tried but pattern suggests same null.)
- **B (Phase 7 solver): WORKS.** Constraint loop makes every model a near-perfect
  solver (4×4 loop completion: MLP 0.970, dense 0.960, flywire 1.000).

## Recent changes

- Phase 5: `models/flywire_snn.py` (FlyWireSNN), sparse recurrent, 4×4 ~0.93.
- Phase 6: `connectome/topology.py` (degree_preserving_shuffle key null +
  erdos_renyi_like/dense/feedforward, matched N+edges). `topology` knob in
  build_model/TrainConfig. `experiments/topology_ablation.py` runner.
  57 tests pass. Structural signature: flywire recip 0.405 vs shuffled 0.038.
  VRAM: measured ceiling ~20k neurons (2.4GB @ batch64); 30k+ OOM — torch.sparse.mm
  backward makes a dense N×N grad. Scaling study caps ~20k without a custom autograd.

## Data facts (memorize)

- meta id column = `fafb_783_id`; classes in `super_class` (sensory 16.9k,
  optic_lobe_intrinsic 77.8k, central_brain_intrinsic 32.5k, motor 106…).
- sensory seeds = super_class in {sensory, sensory_ascending} OR flow=='afferent'.
- transmitters: acetylcholine +1; gaba/glutamate −1; amines +1; unknown +1.

## Reusable how-to (if resuming option 1 — 9×9 generalization)

- Train: `TrainConfig(model='flywire_snn', side=9, n_neurons=1000, n_train=40000,
  epochs=?, weight_decay=1e-4, ...)`; watch VAL (seed+1) move_acc, not train seed.
- Verify generalization: `evaluate_solver(model, spec, seed=1234)` (NOT seed 0 — that
  overlaps train data). Completion on an unseen seed is the real number.
- Save with `--state`; video reuses it. 9×9 gen is slow (~60ms/puzzle) → 40k ≈ 40min gen.

## Notes

- AMP kept OFF for flywire_snn (sparse.mm + autocast not verified); dense models use it.
- flywire_snn + ablation learnability tests guarded on data/flywire/ presence.
- 4×4 easy may SATURATE (~0.93 all topologies) → topology separation likely needs 9×9
  or harder/lower-capacity regime. Interpret ablation numbers with that in mind.

## Task-framing note

Chose full-grid prediction (input 810 one-hot -> logits 81×9, CE over empty cells)
over single-cell Task A. Cleaner, gives move_acc + solve_rate directly, matches the
729 head in PLAN.md §6. The env still supports next_digit (Task A) if needed later.

## Active decisions (locked)

- Connectome: Codex public parquet, synapse count ≥ 5
- Norse + torch cu12; uv package manager
- Constraint-propagation solve loop (not RL spine)
- Principled subsampling only
- Signs-free first
- Curriculum 4×4 → 9×9
- Repo dirs grow per phase

## Open / pending

- 9×9 generalization (overfit) — option 1 above; the main real gap.
- Research writeup doc — option 3; user framed this as a research project.
- Efficiency metrics, 20k VRAM ceiling autograd, RL fine-tuning — options 4-6.
- CAVE account only if live queries needed later.

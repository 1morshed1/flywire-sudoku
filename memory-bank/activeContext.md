# Active Context

## Current focus

Phases 0-7 done + video Tier 1 done. Remaining optional: video Tier 2 (3D real soma
coords), deeper Path-A regimes (9×9 / larger N), RL polish. Await user steer.

## Video (Tier 1) — done

`evaluation/visualization.py`: render_solve_video + make_solve_video CLI. FlyWireSNN
forward has return_activity; solve_with_model has capture_activity. Renders board ‖
spike raster to mp4 (results/, gitignored). Build:
`uv run --no-sync python -m evaluation.visualization`. Tier 2 = 3D soma coords (needs
fetching a coordinate file; meta.feather has no xyz).

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

## Next steps — Path A (topology separation hunt)

1. Re-run `experiments/topology_ablation.py` in capacity-pressured regimes (pick one
   or more): (a) side=9 with more epochs/data; (b) n_neurons=200; (c) difficulty
   hard/expert on 4×4 or 9×9. Add `dense`/`feedforward` conditions for full §18 set.
2. Multiple seeds (≥3), report mean±spread; compare flywire vs shuffled (key null).
3. If still null across regimes → honest finding: topology doesn't help this task.
   If separation appears → the headline result.

## Next steps — Path B (Phase 7 autonomous solver)

1. `sudoku`/`evaluation`: constraint-propagation solve loop. Given a puzzle + trained
   model: encode board → model logits (B=1) → mask illegal (encoder.legal_action_mask)
   → pick globally most-confident (cell,digit) → place via env → repeat until solved or
   stuck. Reuse SudokuEnv(task='place').
2. `evaluation/`: puzzle-completion rate, steps-to-solve, invalid-move rate, per-model
   comparison (MLP/dense/flywire). PLAN.md §22.
3. Optional Phase 8 polish + RL fine-tuning (deferred; constraint-loop is the spine).

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

- **Video simulation** of fly brain solving Sudoku — user wants this. Plan written in
  `docs/VIDEO_PLAN.md`. Needs Phase 7 solver first + a `return_activity` flag on
  FlyWireSNN.forward + `evaluation/visualization.py` renderer → mp4 (imageio-ffmpeg).
  Tier 1 = board + spike raster; Tier 2 = 3D real soma coords. Deferred, pick up later.
- CAVE account only if live queries needed later
- RL fine-tuning optional after autonomous loop works

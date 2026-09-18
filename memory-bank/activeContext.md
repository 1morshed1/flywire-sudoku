# Active Context

## Current focus

Phases 0-6 done. Next: Phase 7 autonomous constraint-loop solver.

## Recent changes

- Phase 5: `models/flywire_snn.py` (FlyWireSNN), sparse recurrent, 4×4 ~0.93.
- Phase 6: `connectome/topology.py` (degree_preserving_shuffle key null +
  erdos_renyi_like/dense/feedforward, matched N+edges). `topology` knob in
  build_model/TrainConfig. `experiments/topology_ablation.py` runner.
  57 tests pass. Structural signature: flywire recip 0.405 vs shuffled 0.038.

## Data facts (memorize)

- meta id column = `fafb_783_id`; classes in `super_class` (sensory 16.9k,
  optic_lobe_intrinsic 77.8k, central_brain_intrinsic 32.5k, motor 106…).
- sensory seeds = super_class in {sensory, sensory_ascending} OR flow=='afferent'.
- transmitters: acetylcholine +1; gaba/glutamate −1; amines +1; unknown +1.

## Next steps (Phase 7 — autonomous solver)

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

- Exact Norse/torch pin versions (verify at install)
- CAVE account only if live queries needed later
- RL fine-tuning optional after autonomous loop works

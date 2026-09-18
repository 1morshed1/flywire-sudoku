# Active Context

## Current focus

Phases 0-5 done. Next: Phase 6 topology ablations (the science).

## Recent changes

- Phase 4: `connectome/` pipeline, FAFB v783, 135k nodes/3.5M edges (count≥5).
- Phase 5: `models/flywire_snn.py` (FlyWireSNN) + `connectome/pipeline.py`
  (subgraph API, lru_cached full graph). Recurrent LIF, W_eff = w_edge ⊙ A via
  torch.sparse.mm. Wired into build_model ('flywire_snn') + TrainConfig
  (n_neurons/sample_method/train_recurrent/use_signs). 50 tests pass. 4×4 real
  1k subgraph ~0.93. VRAM: 20k=2.4GB → ~40k feasible on 6GB.

## Data facts (memorize)

- meta id column = `fafb_783_id`; classes in `super_class` (sensory 16.9k,
  optic_lobe_intrinsic 77.8k, central_brain_intrinsic 32.5k, motor 106…).
- sensory seeds = super_class in {sensory, sensory_ascending} OR flow=='afferent'.
- transmitters: acetylcholine +1; gaba/glutamate −1; amines +1; unknown +1.

## Next steps (Phase 6 — topology ablations)

1. Topology controls at matched budget (PLAN.md §17-18): FlyWire vs
   degree-preserving-shuffled FlyWire (the key null) vs uniform-random vs dense vs
   feed-forward. Add a `degree_preserving_shuffle(adj)` (configuration model) — put
   in connectome/ (e.g. topology.py) so build_model can swap the mask.
2. `experiments/topology_ablation.py`: run each condition (multiple seeds), same
   n_neurons/params, log move_acc/solve_rate + graph stats; compare FlyWire vs shuffle.
3. Then Phase 7: autonomous constraint-propagation solve loop (sudoku env + model
   scoring empty cells, place most-confident legal, repeat).

## Notes

- AMP kept OFF for flywire_snn (sparse.mm + autocast not verified); dense models use it.
- flywire_snn learnability test is guarded on data/flywire/ presence (skips offline).

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

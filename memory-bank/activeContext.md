# Active Context

## Current focus

Phases 0-2 done. Next: Phase 3 dense SNN.

## Recent changes

- Phase 1: `sudoku/` package, size-generic 4×4+9×9, 25 tests, ruff clean
- Phase 2: MLP baseline (`models/mlp.py`), supervised pipeline
  (`training/dataset.py`, `training/supervised.py`), configs/mlp_baseline.yaml.
  Task = full-grid move prediction, masked CE over empty cells. 29 tests pass.
  Gate met: 4×4 move_acc ~0.94.

## Next steps

1. Phase 3: `models/dense_snn.py` — Norse LIF spiking net, same I/O contract.
   Poisson-encode board (encoder.poisson_encode), T=10, surrogate gradients,
   AMP + gradient checkpointing for VRAM. Reuse training/supervised loop.
2. Add spike-rate readout -> logits; keep move_acc/solve_rate metrics.
3. Get dense SNN working BEFORE any FlyWire connectome (Phase 4).

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

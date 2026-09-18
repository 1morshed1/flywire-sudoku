# Active Context

## Current focus

Phases 0-1 done. Next: Phase 2 MLP baseline.

## Recent changes

- Phase 0: `uv sync` OK — torch 2.5.1+cu124, norse 1.1.0, CUDA on RTX 2060 verified
- Phase 1: `sudoku/` package (core/solver/generator/encoder/environment/renderer),
  size-generic 4×4+9×9, 25 pytest passing, ruff clean
- Fixed solver hang: guard solve/count_solutions against invalid boards

## Next steps

1. Phase 2: MLP baseline — `models/mlp.py`, supervised training on next-move
   labels from the solver. Gate: must work before touching FlyWire.
2. Build the supervised data pipeline (board -> one-hot -> solver label).
3. Decide Task A (9-way next-digit) vs Task B (729 place) for first baseline —
   recommend Task A first (simpler, faster to validate pipeline).

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

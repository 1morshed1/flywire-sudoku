# Active Context

## Current focus

Phase 0 done (env + GPU verified). Next: Phase 1 Sudoku env, 4×4 first.

## Recent changes

- Memory bank created from finalized plan
- `CLAUDE.md` added for agent guidance
- `PLAN.md` + docs committed & pushed to origin/main
- Phase 0: `uv sync` OK — torch 2.5.1+cu124, norse 1.1.0, CUDA on RTX 2060 verified

## Next steps

1. Phase 1: Sudoku generator/solver/env/renderer — **4×4 first**, size-generic
2. `sudoku/` dir + unit tests (solver must solve thousands correctly)
3. Then Phase 2 MLP baseline (gate before FlyWire)

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

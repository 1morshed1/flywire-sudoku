# Active Context

## Current focus

Bootstrap project docs + Phase 0 prep. Plan locked in `PLAN.md`. No code/env yet.

## Recent changes

- Memory bank created from finalized plan
- `CLAUDE.md` added for agent guidance

## Next steps

1. Draft `pyproject.toml` / dependency list for user eyeball (Phase 0)
2. On approval: `uv sync` + verify CUDA on RTX 2060
3. Then Phase 1: Sudoku generator/solver/env/renderer — **4×4 first**

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

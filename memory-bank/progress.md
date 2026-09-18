# Progress

## What works

- Finalized plan (`PLAN.md`) with locked design decisions
- Memory bank + `CLAUDE.md` scaffolding
- **Phase 0 env verified**: uv venv, torch 2.5.1+cu124, norse 1.1.0, CUDA sees
  RTX 2060, GPU matmul OK. `pyproject.toml` + `uv.lock` committed.

## What's left

| Phase | Status | Notes |
|-------|--------|-------|
| 0 env (uv, torch, Norse, CUDA check) | **DONE** | torch 2.5.1+cu124, norse 1.1.0, RTX 2060 ✓ |
| 1 Sudoku 4×4 then 9×9 | pending | no GPU |
| 2 MLP baseline | pending | gate: must work before FlyWire |
| 3 dense SNN (AMP, T=10, checkpoint) | pending | GPU |
| 4 FlyWire Codex → sparse COO + sampling | pending | no GPU |
| 5 FlyWire-SNN scale 1k→20k | pending | VRAM bench each rung |
| 6 topology ablations | pending | science core |
| 7 autonomous constraint loop (+ opt RL) | pending | GPU |

## Current status

Phase 0 complete. Env installed + GPU-verified. Next: Phase 1 Sudoku
generator/solver/env/renderer, 4×4 first.

## Known issues / risks

- 6 GB VRAM hard ceiling; 50k+ train won't fit
- Norse may lag latest torch — pin carefully
- Random connectome subsample → dead islands (must use principled methods)
- Multi-GB parquet needs Polars streaming, not eager Pandas

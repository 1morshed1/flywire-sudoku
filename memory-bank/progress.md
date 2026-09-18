# Progress

## What works

- Finalized plan (`PLAN.md`) with locked design decisions
- Memory bank + `CLAUDE.md` scaffolding

## What's left

| Phase | Status | Notes |
|-------|--------|-------|
| 0 env (uv, torch, Norse, CUDA check) | **next** | pyproject first — eyeball before install |
| 1 Sudoku 4×4 then 9×9 | pending | no GPU |
| 2 MLP baseline | pending | gate: must work before FlyWire |
| 3 dense SNN (AMP, T=10, checkpoint) | pending | GPU |
| 4 FlyWire Codex → sparse COO + sampling | pending | no GPU |
| 5 FlyWire-SNN scale 1k→20k | pending | VRAM bench each rung |
| 6 topology ablations | pending | science core |
| 7 autonomous constraint loop (+ opt RL) | pending | GPU |

## Current status

Pre-implementation. Repo has plan + docs only. No packages installed.

## Known issues / risks

- 6 GB VRAM hard ceiling; 50k+ train won't fit
- Norse may lag latest torch — pin carefully
- Random connectome subsample → dead islands (must use principled methods)
- Multi-GB parquet needs Polars streaming, not eager Pandas

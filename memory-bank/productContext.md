# Product Context

## Why this exists

Abstract symbolic tasks (Sudoku) usually use free architectures (MLP, dense SNN,
transformers). This project asks whether *real biological wiring* helps, hurts, or
is neutral when it is the hard constraint on learning — not whether we can rebuild a
fly brain.

## Problem it solves

Need controlled experiment matrix:

1. Same task, same training recipe
2. Vary only connectivity (FlyWire vs random vs degree-shuffled vs dense vs FF)
3. Scale neuron count under real VRAM limits
4. Measure solve behavior + efficiency, not just loss

## How it should work (user / researcher flow)

1. Generate Sudoku puzzles (4×4 then 9×9)
2. Encode board → Poisson spike trains over T timesteps
3. Run FlyWire-masked (or ablation) SNN → logits + legal-action mask
4. Train supervised next-move / cell-digit prediction
5. Autonomous solve: score all empty cells → place most-confident legal digit → repeat
6. Compare models on accuracy, completion, invalid-action rate, steps, VRAM, spikes

## Experience goals

- Phases grow repo dirs only when needed (no big-bang scaffold)
- Deps eyeballed before install (`uv`)
- Clear gate: Phase 2 MLP must pass before FlyWire work
- Signs-free first; neurotransmitter signs later as Regime B ablation

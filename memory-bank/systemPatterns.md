# System Patterns

## High-level architecture

```text
Sudoku board (NxN int)
  -> per-cell one-hot (0=empty, 1..N)     # 9x9: 810 feats; 4x4: 80
  -> Poisson rate -> spike trains (T)
  -> input projection (dense)
  -> FlyWire recurrent LIF: W_eff = W_trainable * A   # A fixed sparse
  -> output projection (dense)
  -> spike-rate readout -> logits
  -> legal-action mask (illegal -> -inf)
  -> argmax place digit
```

## Key design patterns

| Pattern | Detail |
|---------|--------|
| Fixed mask, trainable weights | `A` from FlyWire; `W` learned; later optional `W * A * sign` |
| Constraint-propagation spine | Global most-confident legal (cell, digit); not per-step RL |
| Principled subsampling | Largest WCC / BFS-from-sensory / top-degree — never random |
| Curriculum | 4×4 (action space 64) then 9×9; size-generic generator/env |
| Action head growth | Task A: next-cell 9-way → flat 729 (81×9) with masking |
| Synapse filter | Keep edges with synapse count ≥ 5 |
| Memory discipline | T=10, AMP, gradient checkpointing over time from day 1 |

## Experiment matrix (connectivity)

- A FlyWire
- B Random
- C Degree-preserving shuffled FlyWire
- D Dense
- E Feed-forward
- F FlyWire + learned signs
- G FlyWire + fixed biological signs

## Trainable regimes

- A All weights trainable; connectivity mask fixed
- B Adjacency + signs fixed; magnitude trainable
- C FlyWire fully fixed; only in/out projections train

## Repo growth (per phase)

```text
data/flywire|sudoku  connectome/  sudoku/  models/  training/
experiments/  evaluation/  configs/
```

Do not create all dirs up front — grow with phases.

## Phase gates

0 env → 1 sudoku 4×4/9×9 → 2 MLP → 3 dense SNN → 4 FlyWire data →
5 FlyWire scaling → 6 topology ablations → 7 autonomous solve (+ optional RL)

Stop if Phase 2 MLP fails.

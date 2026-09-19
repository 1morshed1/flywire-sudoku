# `evaluation/` — Phase 7: autonomous solving

Turns a move-scoring model into a full-puzzle solver and measures it.

## Modules

| File | Role |
|------|------|
| `solver_loop.py` | `solve_with_model` + `SolveResult` — constraint-propagation solve loop |
| `metrics.py` | `evaluate_solver` — completion / stuck / steps / placement-accuracy over a puzzle set |

## The solve loop (PLAN.md §5)

```text
board -> model -> per-cell digit logits
      -> mask to legal (cell,digit)     # row/col/box + empty
      -> place the most-confident legal digit
      -> repeat until solved or stuck
```

Only legal placements are ever made, so the model can never write a constraint-
violating digit; it decides *which* legal move to commit. Greedy most-confident policy.
The full trajectory (board after each step + placements) is recorded for the Phase-8
video.

## Metrics (PLAN.md §22)

- `completion_rate` — fraction of puzzles fully & correctly solved
- `stuck_rate` — fraction that dead-ended (no legal move) before completion
- `mean_steps` — average placements per attempt
- `placement_accuracy` — fraction of placements matching the reference solution

## Why this matters

Iterative solving **far exceeds single-shot**. On 4×4-easy a trained MLP scores
single-shot `solve_rate ≈ 0.68` but `completion_rate ≈ 0.98` through the loop — each
placement changes the board and constraints, so later moves get easier. This is why the
project's solve spine is the constraint loop, not one-shot prediction or RL.

## Cross-model comparison (4×4 easy, 100 puzzles)

| model | single-shot solve_rate | loop completion | placement_acc | mean steps |
|-------|-----------------------:|----------------:|--------------:|-----------:|
| MLP | 0.640 | 0.970 | 0.996 | 6.9 |
| dense SNN | 0.510 | 0.960 | 0.993 | 6.9 |
| FlyWire SNN (1k) | 0.705 | **1.000** | **1.000** | 7.0 |

The constraint loop lifts every model to near-perfect (0.96–1.0), far above single-shot.
The FlyWire-constrained SNN matches or edges out the unconstrained baselines — biological
wiring is a fully viable substrate for the solver. (4×4-easy is near ceiling; ranking
differences here are small.)

## Usage

```python
from evaluation.metrics import evaluate_solver
from sudoku import SudokuSpec
# model = a trained SudokuMLP / SudokuDenseSNN / FlyWireSNN
evaluate_solver(model, SudokuSpec.from_side(4), n_puzzles=50, difficulty="easy")
```

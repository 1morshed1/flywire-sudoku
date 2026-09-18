# `sudoku/` — Phase 1: Sudoku environment

Size-generic Sudoku engine. Everything is parameterized by `SudokuSpec`, so 4×4
(fast debug) and 9×9 share one implementation.

## Modules

| File | Role |
|------|------|
| `core.py` | `SudokuSpec` geometry; validity, candidates, peers, solved-check |
| `solver.py` | MRV backtracking `solve`, `count_solutions`, `has_unique_solution` |
| `generator.py` | `generate_full`, `make_puzzle` (unique-solution guarantee), `Puzzle` |
| `encoder.py` | `one_hot`, `poisson_encode` (spike trains), `legal_action_mask` |
| `environment.py` | `SudokuEnv` (Gymnasium-style), tasks `place` / `next_digit` |
| `renderer.py` | `render_ascii` box-gridded board printout |

## Task framings (PLAN.md §1)

- **`place`** (Task B / autonomous): action = `cell * side + (digit-1)`, space
  `Discrete(num_cells * side)` (9×9 → 729, 4×4 → 64). Mask illegal via
  `info["legal_mask"]`.
- **`next_digit`** (Task A): fixed target cell, action = digit, space `Discrete(side)`.

## Encoding (PLAN.md §4-5)

board → per-cell one-hot `(side+1)` channels (0=empty) → flat `num_cells*(side+1)`
features (9×9 → 810, 4×4 → 80) → optional Poisson spike train `(T, features)`.

## Quick start

```python
import numpy as np
from sudoku import SudokuSpec, make_puzzle, solve, render_ascii

spec = SudokuSpec.from_side(4)  # 4x4 for fast debug; 9 for standard
rng = np.random.default_rng(0)
puz = make_puzzle(spec, rng, "easy")
print(render_ascii(spec, puz.puzzle))
assert (solve(spec, puz.puzzle) == puz.solution).all()
```

## Tests

Gate: solver must solve thousands of puzzles correctly (PLAN.md Phase 1).

```bash
uv run pytest tests/test_sudoku.py -q
```

Covers both 4×4 and 9×9: solver correctness, uniqueness, generation, encoding
round-trips, action masking, and the env contract (valid/invalid/solved).

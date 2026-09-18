"""Sudoku puzzle generation with unique-solution guarantee.

Pipeline
--------
1. :func:`generate_full` — build a random *complete* valid grid (a solution).
2. :func:`make_puzzle` — remove clues from that grid while the puzzle keeps a
   **unique** solution, down to a target clue count set by difficulty.

Difficulty is expressed as a target number of remaining clues. Fewer clues => harder.
The mapping is coarse by design (see plan.txt / PLAN.md §3): human difficulty ratings
are out of scope; we only need a controllable, reproducible clue count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import SudokuSpec
from .solver import has_unique_solution, solve


@dataclass(frozen=True)
class Puzzle:
    """A generated puzzle bundled with its unique solution and metadata."""

    spec: SudokuSpec
    puzzle: np.ndarray  # (side, side) with zeros for blanks
    solution: np.ndarray  # (side, side) fully filled
    difficulty: str
    num_clues: int


# Difficulty -> fraction of cells kept as clues. Tuned per board size at call time.
_DIFFICULTY_CLUE_FRACTION: dict[str, float] = {
    "easy": 0.55,
    "medium": 0.45,
    "hard": 0.35,
    "expert": 0.28,
}


def generate_full(spec: SudokuSpec, rng: np.random.Generator) -> np.ndarray:
    """Return a random complete, valid Sudoku grid.

    Implemented by solving an empty board with randomized candidate order, which is
    both simple and correct: the MRV backtracking solver fills every cell and the
    shuffle makes the result uniformly varied enough for training data.
    """
    board = spec.new_board()
    full = solve(spec, board, rng=rng)
    if full is None:  # pragma: no cover - an empty board is always solvable
        raise RuntimeError("failed to generate a full grid (should never happen)")
    return full


def make_puzzle(
    spec: SudokuSpec,
    rng: np.random.Generator,
    difficulty: str = "medium",
    *,
    num_clues: int | None = None,
) -> Puzzle:
    """Generate a puzzle with a unique solution.

    Parameters
    ----------
    difficulty:
        One of ``easy``/``medium``/``hard``/``expert``. Ignored if ``num_clues`` is given.
    num_clues:
        Explicit target clue count. Overrides ``difficulty``. Clamped to a lower bound
        so a unique solution stays feasible.

    Notes
    -----
    Cells are removed in random order, each removal accepted only if the puzzle still
    has exactly one solution. The final clue count may sit slightly above the target
    when further removals would break uniqueness — that is expected and reported in
    :attr:`Puzzle.num_clues`.
    """
    if difficulty not in _DIFFICULTY_CLUE_FRACTION and num_clues is None:
        raise ValueError(
            f"unknown difficulty {difficulty!r}; "
            f"choose from {sorted(_DIFFICULTY_CLUE_FRACTION)} or pass num_clues"
        )

    solution = generate_full(spec, rng)
    puzzle = solution.copy()

    if num_clues is None:
        frac = _DIFFICULTY_CLUE_FRACTION[difficulty]
        num_clues = round(frac * spec.num_cells)
    # A minimum-clue floor keeps generation tractable; 17 is the theoretical min for
    # 9x9, but we stay conservative and scale with board size.
    min_clues = max(spec.side, spec.num_cells // 5)
    target_clues = max(num_clues, min_clues)

    cells = list(range(spec.num_cells))
    rng.shuffle(cells)
    remaining = spec.num_cells
    for idx in cells:
        if remaining <= target_clues:
            break
        r, c = divmod(idx, spec.side)
        saved = int(puzzle[r, c])
        if saved == 0:
            continue
        puzzle[r, c] = 0
        if has_unique_solution(spec, puzzle):
            remaining -= 1
        else:
            puzzle[r, c] = saved  # revert: removal broke uniqueness

    return Puzzle(
        spec=spec,
        puzzle=puzzle,
        solution=solution,
        difficulty=difficulty if num_clues is None else "custom",
        num_clues=int((puzzle != 0).sum()),
    )

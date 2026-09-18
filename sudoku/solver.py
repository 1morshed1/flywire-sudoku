"""Backtracking Sudoku solver and uniqueness checker.

Two public entry points:

* :func:`solve` — return one solution (or ``None`` if unsolvable).
* :func:`count_solutions` — count solutions up to a cap, used to guarantee a
  generated puzzle has a **unique** solution.

The search uses the Minimum-Remaining-Values (MRV) heuristic: always branch on the
empty cell with the fewest candidates. That makes even "hard" 9x9 puzzles solve in
microseconds-to-milliseconds, which matters because the generator (Phase 1) and the
supervised label pipeline (Phase 2) call the solver millions of times.
"""

from __future__ import annotations

import numpy as np

from .core import SudokuSpec, candidates, is_valid


def _select_mrv_cell(spec: SudokuSpec, board: np.ndarray) -> tuple[int, int, list[int]] | None:
    """Return the empty cell with the fewest candidates, plus that candidate list.

    Returns ``None`` if the board is already complete. If any empty cell has zero
    candidates, returns it (with an empty list) so the caller can prune immediately.
    """
    best: tuple[int, int, list[int]] | None = None
    side = spec.side
    for r in range(side):
        for c in range(side):
            if board[r, c] != 0:
                continue
            cands = candidates(spec, board, r, c)
            if len(cands) == 0:
                return (r, c, cands)  # dead end — prune now
            if best is None or len(cands) < len(best[2]):
                best = (r, c, cands)
                if len(cands) == 1:
                    return best  # can't beat a forced cell
    return best


def solve(
    spec: SudokuSpec, board: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray | None:
    """Solve ``board`` in place-safe fashion; return a solved copy or ``None``.

    Parameters
    ----------
    board:
        Partially filled board; not mutated (a copy is solved and returned).
    rng:
        Optional RNG. If given, candidate order is shuffled so repeated calls can
        yield different solutions / random full grids (used by the generator).

    Notes
    -----
    An already-invalid board (a pre-existing duplicate in some row/col/box) has no
    solution and is rejected immediately. Without this guard the backtracker would
    waste exponential time trying to complete a contradictory grid.
    """
    if not is_valid(spec, board):
        return None
    work = board.copy()

    def backtrack() -> bool:
        sel = _select_mrv_cell(spec, work)
        if sel is None:
            return True  # complete
        r, c, cands = sel
        if not cands:
            return False  # dead end
        if rng is not None:
            rng.shuffle(cands)
        for d in cands:
            work[r, c] = d
            if backtrack():
                return True
            work[r, c] = 0
        return False

    return work if backtrack() else None


def count_solutions(spec: SudokuSpec, board: np.ndarray, limit: int = 2) -> int:
    """Count solutions of ``board``, stopping once ``limit`` is reached.

    For puzzle generation we only care whether the count is exactly 1, so the
    default ``limit=2`` short-circuits as soon as a second solution appears.

    An already-invalid board has zero solutions and returns 0 immediately (same
    rationale as :func:`solve`).
    """
    if not is_valid(spec, board):
        return 0
    work = board.copy()
    count = 0

    def backtrack() -> None:
        nonlocal count
        if count >= limit:
            return
        sel = _select_mrv_cell(spec, work)
        if sel is None:
            count += 1
            return
        r, c, cands = sel
        for d in cands:
            work[r, c] = d
            backtrack()
            work[r, c] = 0
            if count >= limit:
                return

    backtrack()
    return count


def has_unique_solution(spec: SudokuSpec, board: np.ndarray) -> bool:
    """True iff the board has exactly one solution."""
    return count_solutions(spec, board, limit=2) == 1

"""Core Sudoku representation and constraint logic.

This module is the size-generic foundation for the whole Sudoku stack. Every other
module (solver, generator, environment, encoder, renderer) depends on :class:`SudokuSpec`
and the pure functions here — none of them hard-code the 9x9 board.

Conventions
-----------
* A board is a ``numpy.ndarray`` of shape ``(side, side)``, dtype int, where ``0``
  means an empty cell and ``1..side`` are filled digits.
* ``side`` is the edge length. It factors into boxes of ``box_rows x box_cols``
  (e.g. 9 -> 3x3 boxes, 4 -> 2x2 boxes, 6 -> 2x3 boxes).
* A "cell index" is the flattened position ``r * side + c`` in ``[0, side*side)``.
* "Peers" of a cell are the other cells sharing its row, column, or box — the cells
  its value is constrained against.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

# Default box shapes for common board sizes. side -> (box_rows, box_cols).
# Extend this map to support more sizes; box_rows * box_cols must equal side.
_DEFAULT_BOXES: dict[int, tuple[int, int]] = {
    4: (2, 2),
    6: (2, 3),
    9: (3, 3),
    12: (3, 4),
    16: (4, 4),
}


@dataclass(frozen=True)
class SudokuSpec:
    """Immutable description of a Sudoku board's geometry.

    Parameters
    ----------
    box_rows, box_cols:
        Dimensions of a single box. The board edge length is ``box_rows * box_cols``.

    Notes
    -----
    Prefer :meth:`from_side` to build a spec from just the edge length using the
    default box shapes.
    """

    box_rows: int
    box_cols: int

    @classmethod
    def from_side(cls, side: int) -> SudokuSpec:
        """Build a spec from the edge length using :data:`_DEFAULT_BOXES`.

        Raises
        ------
        ValueError
            If ``side`` has no registered default box shape.
        """
        if side not in _DEFAULT_BOXES:
            raise ValueError(
                f"no default box shape for side={side}; "
                f"known sizes: {sorted(_DEFAULT_BOXES)} "
                f"(or construct SudokuSpec(box_rows, box_cols) directly)"
            )
        br, bc = _DEFAULT_BOXES[side]
        return cls(br, bc)

    @property
    def side(self) -> int:
        """Edge length of the board (number of rows == number of cols == num digits)."""
        return self.box_rows * self.box_cols

    @property
    def num_cells(self) -> int:
        """Total cells on the board (``side * side``)."""
        return self.side * self.side

    @property
    def num_digits(self) -> int:
        """Number of distinct fill digits (``== side``); valid digits are ``1..side``."""
        return self.side

    @cached_property
    def peers(self) -> tuple[tuple[int, ...], ...]:
        """For each cell index, the tuple of peer cell indices (row+col+box, no self).

        Precomputed once and cached; the solver and validity checks hit this hot.
        """
        side = self.side
        peers: list[tuple[int, ...]] = []
        for idx in range(self.num_cells):
            r, c = divmod(idx, side)
            box_r0 = (r // self.box_rows) * self.box_rows
            box_c0 = (c // self.box_cols) * self.box_cols
            s: set[int] = set()
            for cc in range(side):  # row
                s.add(r * side + cc)
            for rr in range(side):  # col
                s.add(rr * side + c)
            for rr in range(box_r0, box_r0 + self.box_rows):  # box
                for cc in range(box_c0, box_c0 + self.box_cols):
                    s.add(rr * side + cc)
            s.discard(idx)
            peers.append(tuple(sorted(s)))
        return tuple(peers)

    def new_board(self) -> np.ndarray:
        """Return a fresh empty board (all zeros) of shape ``(side, side)``."""
        return np.zeros((self.side, self.side), dtype=np.int8)


def is_valid_placement(spec: SudokuSpec, board: np.ndarray, r: int, c: int, d: int) -> bool:
    """True if writing digit ``d`` into empty cell ``(r, c)`` breaks no constraint.

    Checks the row, column, and box for an existing ``d``. Does **not** check that
    ``(r, c)`` is currently empty — callers that need that must check separately.
    """
    if (board[r, :] == d).any():
        return False
    if (board[:, c] == d).any():
        return False
    box_r0 = (r // spec.box_rows) * spec.box_rows
    box_c0 = (c // spec.box_cols) * spec.box_cols
    box = board[box_r0 : box_r0 + spec.box_rows, box_c0 : box_c0 + spec.box_cols]
    return not (box == d).any()


def candidates(spec: SudokuSpec, board: np.ndarray, r: int, c: int) -> list[int]:
    """Legal digits for empty cell ``(r, c)`` given the current board.

    Returns an empty list for a filled cell.
    """
    if board[r, c] != 0:
        return []
    side = spec.side
    used: set[int] = set()
    used.update(int(x) for x in board[r, :])
    used.update(int(x) for x in board[:, c])
    box_r0 = (r // spec.box_rows) * spec.box_rows
    box_c0 = (c // spec.box_cols) * spec.box_cols
    used.update(
        int(x)
        for x in board[box_r0 : box_r0 + spec.box_rows, box_c0 : box_c0 + spec.box_cols].ravel()
    )
    return [d for d in range(1, side + 1) if d not in used]


def is_complete(board: np.ndarray) -> bool:
    """True if the board has no empty (zero) cells. Does not check validity."""
    return not (board == 0).any()


def is_valid(spec: SudokuSpec, board: np.ndarray) -> bool:
    """True if no row, column, or box contains a duplicate non-zero digit.

    A partially filled board can be valid; use :func:`is_solved` for completeness.
    """
    side = spec.side
    # Rows and columns.
    for line in list(board) + list(board.T):
        vals = line[line != 0]
        if len(vals) != len(set(vals.tolist())):
            return False
    # Boxes.
    for br in range(0, side, spec.box_rows):
        for bc in range(0, side, spec.box_cols):
            box = board[br : br + spec.box_rows, bc : bc + spec.box_cols].ravel()
            vals = box[box != 0]
            if len(vals) != len(set(vals.tolist())):
                return False
    return True


def is_solved(spec: SudokuSpec, board: np.ndarray) -> bool:
    """True if the board is completely filled and valid (a correct solution)."""
    return is_complete(board) and is_valid(spec, board)


def empty_cells(board: np.ndarray) -> list[tuple[int, int]]:
    """List of ``(row, col)`` for every empty cell, row-major order."""
    rs, cs = np.where(board == 0)
    return list(zip(rs.tolist(), cs.tolist()))

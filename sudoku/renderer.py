"""ASCII rendering of Sudoku boards for debugging and logs.

Size-generic: box separators are drawn from :class:`SudokuSpec`, so a 4x4 and a 9x9
board both render with correct box gridlines. Digits above 9 (e.g. 16x16 boards) are
right-padded to a common width.
"""

from __future__ import annotations

import numpy as np

from .core import SudokuSpec


def render_ascii(spec: SudokuSpec, board: np.ndarray, *, empty: str = ".") -> str:
    """Return a human-readable string for ``board``.

    Empty cells show ``empty`` (default ``.``). Vertical bars separate box columns and
    dashed lines separate box rows, matching the classic Sudoku layout.
    """
    side = spec.side
    width = len(str(side))  # cell text width (2 for boards > 9)

    def cell_text(v: int) -> str:
        return (empty if v == 0 else str(v)).rjust(width)

    # Horizontal separator spanning the rendered width of one full row.
    row_cells = side + (side // spec.box_cols - 1)  # cells + column separators
    sep_len = row_cells * (width + 1) - 1
    hsep = "-" * sep_len

    lines: list[str] = []
    for r in range(side):
        if r > 0 and r % spec.box_rows == 0:
            lines.append(hsep)
        parts: list[str] = []
        for c in range(side):
            if c > 0 and c % spec.box_cols == 0:
                parts.append("|")
            parts.append(cell_text(int(board[r, c])))
        lines.append(" ".join(parts))
    return "\n".join(lines)

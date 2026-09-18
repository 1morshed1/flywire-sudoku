"""Size-generic Sudoku stack: geometry, solver, generator, encoding, env, rendering.

Public API (see module docstrings for detail):

* :class:`SudokuSpec` — board geometry (4x4, 9x9, ...).
* :func:`solve`, :func:`count_solutions`, :func:`has_unique_solution` — solver.
* :func:`make_puzzle`, :func:`generate_full`, :class:`Puzzle` — generation.
* :func:`one_hot`, :func:`poisson_encode`, :func:`legal_action_mask` — encodings.
* :func:`render_ascii` — debugging output.
* :class:`SudokuEnv`, :class:`RewardConfig` — Gymnasium-style environment.

The whole stack is parameterized by :class:`SudokuSpec`, so 4x4 (fast debug) and 9x9
share one implementation — switching size is a config change, not a rewrite.
"""

from .core import (
    SudokuSpec,
    candidates,
    empty_cells,
    is_complete,
    is_solved,
    is_valid,
    is_valid_placement,
)
from .encoder import legal_action_mask, one_hot, poisson_encode
from .environment import RewardConfig, SudokuEnv
from .generator import Puzzle, generate_full, make_puzzle
from .renderer import render_ascii
from .solver import count_solutions, has_unique_solution, solve

__all__ = [
    "Puzzle",
    "RewardConfig",
    "SudokuEnv",
    "SudokuSpec",
    "candidates",
    "count_solutions",
    "empty_cells",
    "generate_full",
    "has_unique_solution",
    "is_complete",
    "is_solved",
    "is_valid",
    "is_valid_placement",
    "legal_action_mask",
    "make_puzzle",
    "one_hot",
    "poisson_encode",
    "render_ascii",
    "solve",
]

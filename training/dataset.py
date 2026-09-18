"""Supervised dataset for Sudoku move prediction.

Each example is a (puzzle, solution) pair produced by the generator. The training
target is the *full solved grid*; the loss (in :mod:`training.supervised`) is applied
only to cells that are empty in the puzzle, so the model learns "what digit belongs
in each blank" (PLAN.md §12).

Labels come directly from the generator's unique solution — no separate solver call
is needed because :func:`sudoku.make_puzzle` returns the solution alongside the puzzle,
and uniqueness guarantees the label is the only valid completion (PLAN.md §13 Option B).

Tensors produced per example
----------------------------
* ``x``     : float32 one-hot board, shape ``(num_cells * (side + 1),)``
* ``target``: int64 digit indices ``0..side-1``, shape ``(num_cells,)``
* ``empty`` : bool mask of originally-empty cells, shape ``(num_cells,)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from sudoku import SudokuSpec, make_puzzle, one_hot


@dataclass
class SudokuTensors:
    """A materialized dataset held entirely in memory as stacked tensors."""

    x: torch.Tensor  # (N, in_features) float32
    target: torch.Tensor  # (N, num_cells) int64, values 0..side-1
    empty: torch.Tensor  # (N, num_cells) bool


def build_dataset(
    spec: SudokuSpec,
    n: int,
    *,
    difficulty: str = "easy",
    seed: int = 0,
) -> SudokuTensors:
    """Generate ``n`` puzzles and stack them into tensors.

    Generation is deterministic given ``seed``. For 9x9 this is the slow step
    (tens of ms per puzzle), so callers typically build once and reuse / cache.
    """
    rng = np.random.default_rng(seed)
    xs = np.empty((n, spec.num_cells * (spec.side + 1)), dtype=np.float32)
    targets = np.empty((n, spec.num_cells), dtype=np.int64)
    empties = np.empty((n, spec.num_cells), dtype=bool)

    for i in range(n):
        puz = make_puzzle(spec, rng, difficulty)
        xs[i] = one_hot(spec, puz.puzzle, flatten=True)
        # Digit indices 0..side-1 (solution digits are 1..side).
        targets[i] = puz.solution.reshape(-1) - 1
        empties[i] = puz.puzzle.reshape(-1) == 0

    return SudokuTensors(
        x=torch.from_numpy(xs),
        target=torch.from_numpy(targets),
        empty=torch.from_numpy(empties),
    )


class SudokuMoveDataset(Dataset):
    """torch ``Dataset`` wrapper over :class:`SudokuTensors`.

    Yields ``(x, target, empty)`` triples for a standard DataLoader.
    """

    def __init__(self, tensors: SudokuTensors) -> None:
        self.t = tensors

    def __len__(self) -> int:
        return self.t.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.t.x[idx], self.t.target[idx], self.t.empty[idx]

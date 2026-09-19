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
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sudoku import SudokuSpec, make_puzzle, one_hot

# On-disk cache for generated datasets. 9x9 generation is ~44 ms/puzzle, so a 40k
# training set costs ~30 min; caching makes every re-run (retrain, video, eval)
# instant. Keyed by the full generation signature so a mismatch never loads stale data.
CACHE_DIR = Path("data/datasets")


@dataclass
class SudokuTensors:
    """A materialized dataset held entirely in memory as stacked tensors."""

    x: torch.Tensor  # (N, in_features) float32
    target: torch.Tensor  # (N, num_cells) int64, values 0..side-1
    empty: torch.Tensor  # (N, num_cells) bool


def _cache_path(spec: SudokuSpec, n: int, difficulty: str, seed: int) -> Path:
    """Deterministic cache filename for a generation signature."""
    return CACHE_DIR / f"{spec.side}_{difficulty}_{n}_{seed}.pt"


def build_dataset(
    spec: SudokuSpec,
    n: int,
    *,
    difficulty: str = "easy",
    seed: int = 0,
    cache: bool = True,
) -> SudokuTensors:
    """Generate ``n`` puzzles and stack them into tensors.

    Generation is deterministic given ``seed``. For 9x9 this is the slow step
    (~44 ms per puzzle), so results are cached to ``data/datasets/`` keyed by the
    generation signature ``(side, difficulty, n, seed)``. A later call with the same
    signature loads the cache instead of regenerating. Set ``cache=False`` to bypass.
    """
    path = _cache_path(spec, n, difficulty, seed)
    if cache and path.exists():
        blob = torch.load(path, weights_only=True)
        return SudokuTensors(x=blob["x"], target=blob["target"], empty=blob["empty"])

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

    tensors = SudokuTensors(
        x=torch.from_numpy(xs),
        target=torch.from_numpy(targets),
        empty=torch.from_numpy(empties),
    )
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"x": tensors.x, "target": tensors.target, "empty": tensors.empty}, path)
    return tensors


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

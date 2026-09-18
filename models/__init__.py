"""Neural models for FlyWire-Sudoku.

Model ladder (PLAN.md §6): MLP baseline -> dense SNN -> FlyWire-constrained SNN.
This package currently holds:

* :class:`SudokuMLP` — the Phase 2 non-spiking baseline. Establishes that the data
  pipeline and task framing are learnable before any spiking / connectome complexity.

All models share the same I/O contract so training/eval code is model-agnostic:

    input:  flat one-hot board, shape (B, num_cells * (side + 1))
    output: move logits,        shape (B, num_cells, side)

where output ``[b, cell, d]`` scores placing digit ``d+1`` in ``cell``.
"""

from .dense_snn import SudokuDenseSNN
from .flywire_snn import FlyWireSNN
from .mlp import SudokuMLP

__all__ = ["FlyWireSNN", "SudokuDenseSNN", "SudokuMLP"]

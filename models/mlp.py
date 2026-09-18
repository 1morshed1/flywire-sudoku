"""MLP baseline for Sudoku move prediction (PLAN.md Phase 2 / §6, Model 0).

A plain feed-forward network:

    810 -> 512 -> 256 -> 729        (9x9; sizes derived from SudokuSpec)

It maps a one-hot board to per-cell digit logits in a single forward pass. This is
the baseline every later model (dense SNN, FlyWire SNN) is compared against — if the
MLP cannot learn the task, nothing downstream will, so this is the Phase 2 gate.

Size-generic: input/output widths come from :class:`~sudoku.core.SudokuSpec`, so the
same class handles 4x4 (80 -> ... -> 64) and 9x9 (810 -> ... -> 729).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from sudoku.core import SudokuSpec


class SudokuMLP(nn.Module):
    """Feed-forward baseline mapping a one-hot board to per-cell digit logits.

    Parameters
    ----------
    spec:
        Board geometry; determines input and output widths.
    hidden:
        Hidden layer widths. Default ``(512, 256)`` per PLAN.md §6.
    dropout:
        Dropout probability applied after each hidden activation (0 disables).

    Shapes
    ------
    forward input:  ``(B, num_cells * (side + 1))`` float one-hot.
    forward output: ``(B, num_cells, side)`` logits; ``[b, cell, d]`` scores digit ``d+1``.
    """

    def __init__(
        self,
        spec: SudokuSpec,
        hidden: Sequence[int] = (512, 256),
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.in_features = spec.num_cells * (spec.side + 1)
        self.out_features = spec.num_cells * spec.side

        layers: list[nn.Module] = []
        prev = self.in_features
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, self.out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network; returns logits reshaped to ``(B, num_cells, side)``."""
        logits = self.net(x)
        return logits.view(-1, self.spec.num_cells, self.spec.num_digits)

    def num_parameters(self) -> int:
        """Total trainable parameter count (reported in the experiment matrix)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

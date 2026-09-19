"""Autonomous Sudoku solving by constraint propagation (PLAN.md §5, Phase 7).

This is the solve *spine*: it turns any move-scoring model (MLP / dense SNN / FlyWire
SNN — all share the same I/O contract) into a full-puzzle solver, without RL.

Loop, per step:
    1. encode the current board -> model -> per-cell digit logits.
    2. mask to *legal* (cell, digit) placements (row/col/box + empty).
    3. place the single globally most-confident legal digit.
    4. repeat until the board is solved, or no legal move remains (stuck).

Because only legal placements are ever made, the model never writes a
constraint-violating digit; the open question a model answers is *which* legal move to
commit to. A greedy most-confident policy is used here; the trajectory is recorded so
the Phase-8 video can replay it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from sudoku import SudokuSpec, is_solved, legal_action_mask, one_hot
from sudoku.core import is_complete


@dataclass
class SolveResult:
    """Outcome of one autonomous solve."""

    solved: bool
    steps: int
    stuck: bool  # no legal move remained before completion
    final_board: np.ndarray
    trajectory: list[np.ndarray] = field(default_factory=list)  # board after each step
    placements: list[tuple[int, int, int]] = field(default_factory=list)  # (r, c, digit)
    correct_placements: int = 0  # matched the reference solution (if provided)


@torch.no_grad()
def solve_with_model(
    model: torch.nn.Module,
    spec: SudokuSpec,
    board: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    solution: np.ndarray | None = None,
    max_steps: int | None = None,
    record_trajectory: bool = True,
) -> SolveResult:
    """Solve ``board`` autonomously with ``model`` via the constraint-propagation loop.

    Parameters
    ----------
    solution:
        Optional reference solution; when given, each placement is checked against it
        to count ``correct_placements`` (a per-move accuracy under the greedy policy).
    max_steps:
        Safety cap; defaults to the number of empty cells.
    """
    device = torch.device(device)
    model.eval()
    work = board.copy()
    side = spec.side
    max_steps = max_steps or int((work == 0).sum())

    result = SolveResult(solved=False, steps=0, stuck=False, final_board=work)
    if record_trajectory:
        result.trajectory.append(work.copy())

    for _ in range(max_steps):
        if is_complete(work):
            break
        mask = legal_action_mask(spec, work)  # (num_cells * side,) bool
        if not mask.any():
            result.stuck = True
            break

        x = torch.from_numpy(one_hot(spec, work, flatten=True)).unsqueeze(0).to(device)
        logits = model(x)[0].reshape(-1).cpu()  # (num_cells * side,)
        logits[~torch.from_numpy(mask)] = float("-inf")

        action = int(torch.argmax(logits))
        cell, digit_idx = divmod(action, side)
        r, c = divmod(cell, side)
        digit = digit_idx + 1

        work[r, c] = digit
        result.steps += 1
        result.placements.append((r, c, digit))
        if solution is not None and solution[r, c] == digit:
            result.correct_placements += 1
        if record_trajectory:
            result.trajectory.append(work.copy())

    result.final_board = work
    result.solved = is_solved(spec, work)
    return result

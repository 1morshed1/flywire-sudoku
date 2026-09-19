"""Solver evaluation metrics over a puzzle set (PLAN.md §22).

Aggregates :func:`evaluation.solver_loop.solve_with_model` across many puzzles into the
headline solver metrics, for comparing models (MLP / dense SNN / FlyWire SNN):

* ``completion_rate`` — fraction of puzzles fully and correctly solved.
* ``mean_steps`` — average placements made (over all attempts).
* ``stuck_rate`` — fraction that dead-ended with no legal move before completion.
* ``placement_accuracy`` — fraction of placements matching the reference solution.

A puzzle set is generated on demand from a :class:`SudokuSpec` and difficulty, so
evaluation is reproducible from a seed.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.solver_loop import solve_with_model
from sudoku import SudokuSpec, make_puzzle


def evaluate_solver(
    model: torch.nn.Module,
    spec: SudokuSpec,
    *,
    n_puzzles: int = 100,
    difficulty: str = "easy",
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, float]:
    """Solve ``n_puzzles`` generated puzzles with ``model``; return aggregate metrics."""
    rng = np.random.default_rng(seed)
    solved = 0
    stuck = 0
    total_steps = 0
    correct = 0
    placements = 0

    for _ in range(n_puzzles):
        puz = make_puzzle(spec, rng, difficulty)
        res = solve_with_model(
            model,
            spec,
            puz.puzzle,
            device=device,
            solution=puz.solution,
            record_trajectory=False,
        )
        solved += int(res.solved)
        stuck += int(res.stuck)
        total_steps += res.steps
        correct += res.correct_placements
        placements += res.steps

    return {
        "completion_rate": solved / n_puzzles,
        "stuck_rate": stuck / n_puzzles,
        "mean_steps": total_steps / n_puzzles,
        "placement_accuracy": correct / max(placements, 1),
        "n_puzzles": float(n_puzzles),
    }

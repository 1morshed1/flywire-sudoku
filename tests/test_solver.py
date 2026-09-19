"""Phase 7 tests: autonomous constraint-propagation solver + metrics.

An oracle model (logits that argmax to the true solution) must solve every puzzle,
which validates the loop mechanics. Every placement must be legal, and the aggregate
metrics must have the documented shape. A small trained MLP checks the end-to-end path.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.metrics import evaluate_solver
from evaluation.solver_loop import solve_with_model
from sudoku import SudokuSpec, is_valid, make_puzzle


class _Oracle(torch.nn.Module):
    """Returns logits whose argmax is the reference solution digit for each cell."""

    def __init__(self, spec: SudokuSpec) -> None:
        super().__init__()
        self.spec = spec
        self.solution: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.from_numpy((self.solution.reshape(-1) - 1).astype(np.int64))
        return torch.nn.functional.one_hot(idx, self.spec.side).float().unsqueeze(0) * 10.0


def test_oracle_solves_every_puzzle():
    spec = SudokuSpec.from_side(4)
    rng = np.random.default_rng(0)
    oracle = _Oracle(spec)
    for _ in range(25):
        puz = make_puzzle(spec, rng, "easy")
        oracle.solution = puz.solution
        res = solve_with_model(oracle, spec, puz.puzzle, solution=puz.solution)
        assert res.solved
        assert not res.stuck
        # Trajectory records the initial board plus one board per placement.
        assert len(res.trajectory) == res.steps + 1
        # Every placement is correct against the (unique) solution.
        assert res.correct_placements == res.steps


def test_solver_makes_only_legal_moves():
    spec = SudokuSpec.from_side(9)
    rng = np.random.default_rng(1)
    oracle = _Oracle(spec)
    puz = make_puzzle(spec, rng, "easy")
    oracle.solution = puz.solution
    res = solve_with_model(oracle, spec, puz.puzzle, solution=puz.solution)
    # The board stays valid at every recorded step (no constraint violations).
    for board in res.trajectory:
        assert is_valid(spec, board)
    assert res.solved


def test_solver_does_not_touch_clues():
    spec = SudokuSpec.from_side(4)
    rng = np.random.default_rng(2)
    oracle = _Oracle(spec)
    puz = make_puzzle(spec, rng, "easy")
    oracle.solution = puz.solution
    res = solve_with_model(oracle, spec, puz.puzzle, solution=puz.solution)
    clue_mask = puz.puzzle != 0
    # Original clues are unchanged in the final board.
    assert np.array_equal(res.final_board[clue_mask], puz.puzzle[clue_mask])


def test_metrics_shape_and_range():
    """A briefly trained MLP should solve most 4x4 puzzles via the loop."""
    from training.supervised import TrainConfig, train_supervised

    spec = SudokuSpec.from_side(4)
    _, model = train_supervised(
        TrainConfig(
            model="mlp",
            side=4,
            hidden=(256, 128),
            n_train=2000,
            n_val=200,
            difficulty="easy",
            epochs=40,
            device="cpu",
        ),
        verbose=False,
        return_model=True,
    )
    m = evaluate_solver(model, spec, n_puzzles=40, difficulty="easy", device="cpu")
    assert set(m) == {
        "completion_rate",
        "stuck_rate",
        "mean_steps",
        "placement_accuracy",
        "n_puzzles",
    }
    assert 0.0 <= m["completion_rate"] <= 1.0
    # Iterative solving far exceeds single-shot solve_rate; expect a strong solver.
    assert m["completion_rate"] > 0.8

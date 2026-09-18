"""Phase 2 tests: MLP baseline shape contract, loss/metrics, and learnability.

The learnability test is the Phase 2 gate in miniature: a small MLP trained briefly
on 4x4 puzzles must reach high empty-cell accuracy. If this fails, the data pipeline
or task framing is broken and no downstream (SNN / FlyWire) model can succeed.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from models import SudokuMLP
from sudoku import SudokuSpec
from training.dataset import SudokuMoveDataset, build_dataset
from training.supervised import TrainConfig, evaluate, masked_ce_loss, train_mlp


def test_forward_shapes():
    for side in (4, 9):
        spec = SudokuSpec.from_side(side)
        model = SudokuMLP(spec, hidden=(64, 32))
        x = torch.zeros(3, spec.num_cells * (spec.side + 1))
        out = model(x)
        assert out.shape == (3, spec.num_cells, spec.side)
        assert model.num_parameters() > 0


def test_masked_ce_ignores_filled_cells():
    spec = SudokuSpec.from_side(4)
    b, n, side = 2, spec.num_cells, spec.side
    logits = torch.zeros(b, n, side, requires_grad=True)
    target = torch.zeros(b, n, dtype=torch.int64)
    empty = torch.zeros(b, n, dtype=torch.bool)
    # No empty cells -> zero loss, and it must be differentiable (no NaN).
    loss = masked_ce_loss(logits, target, empty)
    assert float(loss) == 0.0
    loss.backward()
    # One empty cell -> positive loss.
    empty[0, 0] = True
    loss2 = masked_ce_loss(logits.detach().requires_grad_(True), target, empty)
    assert float(loss2) > 0.0


def test_evaluate_perfect_predictions():
    """A model that outputs the solution should score move_acc=1, solve_rate=1."""
    spec = SudokuSpec.from_side(4)
    t = build_dataset(spec, 8, difficulty="easy", seed=1)
    # Full-batch loader keeps the oracle's precomputed logits aligned with targets.
    full = DataLoader(SudokuMoveDataset(t), batch_size=t.x.shape[0])
    oracle = _AlignedOracle(t.target, spec.side)
    metrics = evaluate(oracle, full, torch.device("cpu"))
    assert metrics["move_acc"] == 1.0
    assert metrics["solve_rate"] == 1.0


def _one_hot_logits(target: torch.Tensor, side: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(target, num_classes=side).float() * 10.0


class _AlignedOracle(torch.nn.Module):
    """Returns logits whose argmax equals the stored targets (batch-size agnostic)."""

    def __init__(self, target: torch.Tensor, side: int) -> None:
        super().__init__()
        self.logits = _one_hot_logits(target, side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits[: x.shape[0]]


def test_mlp_learns_4x4():
    """Train briefly on 4x4; empty-cell accuracy must climb well above chance."""
    cfg = TrainConfig(
        side=4,
        hidden=(256, 128),
        n_train=2000,
        n_val=256,
        difficulty="easy",
        epochs=40,
        batch_size=64,
        lr=1e-3,
        seed=0,
        device="cpu",  # deterministic + no GPU dependency in CI
    )
    metrics = train_mlp(cfg, verbose=False)
    # Chance on 4x4 is 0.25; a working pipeline reaches ~0.9 empty-cell accuracy.
    assert metrics["move_acc"] > 0.8

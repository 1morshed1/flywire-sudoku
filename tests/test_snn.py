"""Phase 3 tests: dense spiking network (Norse LIF).

Covers the shared I/O contract, gradient flow through the surrogate-gradient LIF
simulation, both input encodings, and learnability on 4x4 (the Phase 3 gate: a
*spiking* net must learn the task before the FlyWire recurrent mask is introduced).
"""

from __future__ import annotations

import torch

from models import SudokuDenseSNN
from sudoku import SudokuSpec
from training.supervised import TrainConfig, train_supervised


def test_snn_forward_shapes_and_grad():
    for side in (4, 9):
        spec = SudokuSpec.from_side(side)
        model = SudokuDenseSNN(spec, hidden=(64, 32), T=6)
        x = torch.zeros(4, spec.num_cells * (spec.side + 1), requires_grad=False)
        x[:, :: spec.side + 1] = 1.0  # mark some channels active
        out = model(x)
        assert out.shape == (4, spec.num_cells, spec.side)
        # Gradients must flow back through the T-step surrogate-gradient simulation.
        model.zero_grad()
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


def test_snn_encodings():
    spec = SudokuSpec.from_side(4)
    x = torch.rand(3, spec.num_cells * (spec.side + 1))
    for enc in ("constant", "poisson"):
        model = SudokuDenseSNN(spec, hidden=(16,), T=4, encoding=enc)
        assert model(x).shape == (3, spec.num_cells, spec.side)


def test_snn_rejects_bad_encoding():
    import pytest

    with pytest.raises(ValueError, match="encoding"):
        SudokuDenseSNN(SudokuSpec.from_side(4), encoding="rate")


def test_snn_learns_4x4():
    """Train the spiking net briefly on 4x4; accuracy must climb well above chance."""
    cfg = TrainConfig(
        model="dense_snn",
        side=4,
        hidden=(256, 128),
        T=10,
        encoding="constant",
        n_train=2000,
        n_val=256,
        difficulty="easy",
        epochs=40,
        batch_size=64,
        lr=1e-3,
        seed=0,
        device="cpu",
    )
    metrics = train_supervised(cfg, verbose=False)
    # Chance on 4x4 is 0.25; the spiking net reaches ~0.9 with enough epochs.
    assert metrics["move_acc"] > 0.7

"""Phase 5 tests: FlyWire-constrained recurrent SNN.

Fast tests use small synthetic connectomes: I/O contract, gradient flow through the
sparse recurrent step, the correctness of that step's orientation (the main
implementation risk), and the three trainable regimes. A guarded test trains on a real
FlyWire subgraph when the FAFB dump is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from connectome.adjacency import Connectome
from models import FlyWireSNN
from sudoku.core import SudokuSpec

_REAL_EDGES = Path("data/flywire/fafb_783_simple_edgelist.feather")


def _toy_connectome(n: int, density: float = 0.05, seed: int = 0) -> Connectome:
    rng = np.random.default_rng(seed)
    m = int(n * n * density)
    r = rng.integers(0, n, m)
    c = rng.integers(0, n, m)
    adj = sp.coo_matrix((rng.random(m).astype(np.float32), (r, c)), shape=(n, n)).tocsr()
    adj.sum_duplicates()
    return Connectome(adj=adj, node_ids=[str(i) for i in range(n)])


def _board_batch(spec: SudokuSpec, b: int) -> torch.Tensor:
    x = torch.zeros(b, spec.num_cells * (spec.side + 1))
    x[:, :: spec.side + 1] = 1.0
    return x


def test_forward_shape_and_grad():
    for side in (4, 9):
        spec = SudokuSpec.from_side(side)
        model = FlyWireSNN(spec, _toy_connectome(120), T=6)
        out = model(_board_batch(spec, 4))
        assert out.shape == (4, spec.num_cells, spec.side)
        model.zero_grad()
        out.sum().backward()
        assert model.rec_weight.grad is not None
        assert torch.isfinite(model.rec_weight.grad).all()


def test_recurrent_step_orientation():
    """The sparse recurrent step must equal spikes @ W with W[pre, post] = edge weight.

    This pins down the transpose used in the sparse matmul: current[b, j] must be
    sum_i W[i, j] * spikes[b, i].
    """
    spec = SudokuSpec.from_side(4)
    conn = _toy_connectome(30, density=0.1, seed=1)
    model = FlyWireSNN(spec, conn, T=1)

    # Reconstruct the dense equivalent from the model's own buffers/params.
    n = model.N
    dense = torch.zeros(n, n)
    post = model.rec_indices[0]
    pre = model.rec_indices[1]
    dense[pre, post] = model.rec_weight.detach()  # W[pre, post]

    spikes = torch.rand(5, n)
    expected = spikes @ dense
    got = model._recurrent_current(spikes)
    assert torch.allclose(got, expected, atol=1e-5)


def test_regime_c_freezes_recurrent():
    spec = SudokuSpec.from_side(4)
    conn = _toy_connectome(100)
    trainable = FlyWireSNN(spec, conn, T=4, train_recurrent=True)
    frozen = FlyWireSNN(spec, conn, T=4, train_recurrent=False)
    names_trainable = {n for n, _ in trainable.named_parameters()}
    names_frozen = {n for n, _ in frozen.named_parameters()}
    assert "rec_weight" in names_trainable
    assert "rec_weight" not in names_frozen  # frozen -> buffer, not a parameter
    assert frozen.num_parameters() < trainable.num_parameters()


def test_regime_b_signs_fix_polarity():
    spec = SudokuSpec.from_side(4)
    conn = _toy_connectome(60, seed=2)
    signs = np.where(np.random.default_rng(0).random(conn.num_nodes) < 0.7, 1, -1).astype(np.int8)
    model = FlyWireSNN(spec, conn, T=3, signs=signs)
    assert model.use_signs
    # Force all magnitudes negative; effective weights must still respect edge signs.
    with torch.no_grad():
        model.rec_weight.fill_(-0.5)
    pre = model.rec_indices[1].numpy()
    values = (model.rec_weight.abs() * model.edge_sign).detach().numpy()
    assert np.all(np.sign(values[values != 0]) == signs[pre][values != 0])


def test_rejects_bad_encoding():
    with pytest.raises(ValueError, match="encoding"):
        FlyWireSNN(SudokuSpec.from_side(4), _toy_connectome(20), encoding="rate")


@pytest.mark.skipif(not _REAL_EDGES.exists(), reason="FlyWire FAFB dump not downloaded")
def test_learns_on_real_subgraph():
    """Train briefly on a real 500-neuron FlyWire subgraph; accuracy beats chance."""
    from training.supervised import TrainConfig, train_supervised

    cfg = TrainConfig(
        model="flywire_snn",
        side=4,
        n_neurons=500,
        sample_method="top_degree",
        T=10,
        n_train=800,
        n_val=200,
        difficulty="easy",
        epochs=20,
        batch_size=64,
        lr=1e-3,
        seed=0,
        device="cpu",
    )
    metrics = train_supervised(cfg, verbose=False)
    assert metrics["move_acc"] > 0.6  # chance on 4x4 is 0.25

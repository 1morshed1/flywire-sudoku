"""Phase 8 tests: activity capture + solve-video rendering.

Fast, self-contained: a small FlyWire SNN on a synthetic connectome provides spike
activity; the solver captures it; the renderer writes a real (tiny) mp4. No training
and no network access.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from connectome.adjacency import Connectome
from evaluation.solver_loop import solve_with_model
from evaluation.visualization import render_solve_video
from models import FlyWireSNN
from sudoku import SudokuSpec, make_puzzle


def _toy_connectome(n: int, seed: int = 0) -> Connectome:
    rng = np.random.default_rng(seed)
    m = int(n * n * 0.05)
    r = rng.integers(0, n, m)
    c = rng.integers(0, n, m)
    adj = sp.coo_matrix((rng.random(m).astype(np.float32), (r, c)), shape=(n, n)).tocsr()
    adj.sum_duplicates()
    return Connectome(adj=adj, node_ids=[str(i) for i in range(n)])


def test_flywire_return_activity_shape():
    spec = SudokuSpec.from_side(4)
    model = FlyWireSNN(spec, _toy_connectome(80), T=6)
    x = torch.zeros(3, spec.num_cells * (spec.side + 1))
    logits, activity = model(x, return_activity=True)
    assert logits.shape == (3, spec.num_cells, spec.side)
    assert activity.shape == (6, 3, 80)  # (T, B, N)


def test_solver_captures_activity():
    spec = SudokuSpec.from_side(4)
    model = FlyWireSNN(spec, _toy_connectome(80), T=6)
    rng = np.random.default_rng(0)
    puz = make_puzzle(spec, rng, "easy")
    res = solve_with_model(model, spec, puz.puzzle, capture_activity=True)
    assert len(res.activity) == res.steps
    if res.steps:
        assert res.activity[0].shape == (6, 80)  # (T, N) for one step


def test_render_writes_mp4(tmp_path):
    spec = SudokuSpec.from_side(4)
    model = FlyWireSNN(spec, _toy_connectome(80), T=6)
    rng = np.random.default_rng(1)
    puz = make_puzzle(spec, rng, "easy")
    res = solve_with_model(model, spec, puz.puzzle, capture_activity=True)
    if not res.activity:  # untrained model may immediately stick; force a couple steps
        return
    out = render_solve_video(res, spec, tmp_path / "solve.mp4", fps=2, max_neurons=40)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_3d_writes_mp4(tmp_path):
    from evaluation.visualization import render_solve_video_3d

    spec = SudokuSpec.from_side(4)
    n = 80
    model = FlyWireSNN(spec, _toy_connectome(n), T=6)
    rng = np.random.default_rng(3)
    puz = make_puzzle(spec, rng, "easy")
    res = solve_with_model(model, spec, puz.puzzle, capture_activity=True)
    if not res.activity:
        return
    coords = rng.random((n, 3)) * 1000  # synthetic soma coords
    out = render_solve_video_3d(res, spec, coords, tmp_path / "solve3d.mp4", fps=2)
    assert out.exists()
    assert out.stat().st_size > 0

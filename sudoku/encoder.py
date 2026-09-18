"""Board -> feature encodings for the neural models.

Two stages, matching PLAN.md §4-5:

1. :func:`one_hot` — each cell becomes a ``(side + 1)``-dim one-hot vector
   (channel 0 = empty, channels ``1..side`` = digit). A 9x9 board -> ``81 x 10 = 810``
   features; a 4x4 board -> ``16 x 5 = 80``.
2. :func:`poisson_encode` — turn a rate vector in ``[0, 1]`` into a ``(T, features)``
   binary spike train for the spiking models. Kept in NumPy here; the torch path in
   ``models/`` will mirror this.

Also provides :func:`legal_action_mask` for the flat 729-style action head
(``cell * side + (digit - 1)``), used by the environment and decoders.
"""

from __future__ import annotations

import numpy as np

from .core import SudokuSpec, candidates


def one_hot(spec: SudokuSpec, board: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    """One-hot encode a board.

    Parameters
    ----------
    flatten:
        If True (default) return shape ``(num_cells * (side + 1),)``; otherwise
        ``(num_cells, side + 1)``.

    Channel 0 marks an empty cell; channel ``d`` marks digit ``d``.
    """
    side = spec.side
    channels = side + 1
    flat = board.reshape(-1).astype(np.int64)  # values in 0..side == channel index
    enc = np.zeros((spec.num_cells, channels), dtype=np.float32)
    enc[np.arange(spec.num_cells), flat] = 1.0
    return enc.reshape(-1) if flatten else enc


def poisson_encode(
    rates: np.ndarray,
    T: int,
    rng: np.random.Generator,
    *,
    max_rate: float = 1.0,
) -> np.ndarray:
    """Rate-code a feature vector into a ``(T, num_features)`` binary spike train.

    Each feature fires independently at each timestep with probability
    ``rate * max_rate``. With one-hot inputs (rates in {0, 1}) this yields clean
    per-timestep spikes on the active channels.

    Parameters
    ----------
    rates:
        1-D array of per-feature firing probabilities in ``[0, 1]``.
    T:
        Number of simulation timesteps (PLAN.md defaults to T=10 for VRAM reasons).
    max_rate:
        Scales probabilities; keep <= 1.0.
    """
    if rates.ndim != 1:
        raise ValueError(f"rates must be 1-D, got shape {rates.shape}")
    probs = np.clip(rates * max_rate, 0.0, 1.0)
    draws = rng.random((T, rates.shape[0]))
    return (draws < probs[None, :]).astype(np.float32)


def legal_action_mask(spec: SudokuSpec, board: np.ndarray) -> np.ndarray:
    """Boolean mask over the flat action space ``cell * side + (digit - 1)``.

    ``True`` at an index means that (cell, digit) placement is legal: the cell is
    empty and the digit breaks no row/col/box constraint. Shape ``(num_cells * side,)``.
    Illegal-action masking (set logits to -inf) uses this in the decoder and env.
    """
    side = spec.side
    mask = np.zeros(spec.num_cells * side, dtype=bool)
    for r in range(side):
        for c in range(side):
            if board[r, c] != 0:
                continue
            cell = r * side + c
            for d in candidates(spec, board, r, c):
                mask[cell * side + (d - 1)] = True
    return mask

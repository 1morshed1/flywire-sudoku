"""Dense spiking neural network baseline (PLAN.md Phase 3 / §6, Model 1).

A feed-forward LIF network with the same external contract as the MLP:

    input:  one-hot board   (B, in_features)
    output: move logits      (B, num_cells, side)

Internally it runs a ``T``-step spiking simulation. Each layer is
``Linear -> LIFCell`` (leaky integrate-and-fire, Norse, with surrogate gradients);
the readout is ``Linear -> LICell`` (a non-spiking leaky integrator) whose membrane
voltage, averaged over time, is the logit vector. Keeping the simulation inside
``forward`` means the Phase 2 training loop and metrics work unchanged.

Input encoding (see :meth:`_encode`):
* ``"constant"`` (default) — feed the real-valued one-hot as input current every
  timestep ("direct" encoding). Most stable for surrogate-gradient training, so it is
  the default for the baseline gate.
* ``"poisson"`` — Bernoulli-sample spikes from the one-hot each step (PLAN.md §5).
  Noisier; available for spiking-realism experiments.

This is the last non-connectome model: it establishes that a *spiking* net can learn
the task before the FlyWire recurrent mask is introduced (Phase 5).
"""

from __future__ import annotations

import torch
from norse.torch import LICell, LIFCell, LIFParameters
from torch import nn

from sudoku.core import SudokuSpec


class SudokuDenseSNN(nn.Module):
    """Feed-forward LIF spiking network for Sudoku move prediction.

    Parameters
    ----------
    spec:
        Board geometry (sets input/output widths).
    hidden:
        Hidden LIF layer widths. Default ``(512, 256)`` per PLAN.md §6.
    T:
        Number of simulation timesteps. Default 10 (VRAM-conscious; PLAN.md §R3).
    encoding:
        ``"constant"`` or ``"poisson"`` input encoding.
    lif_params:
        Optional Norse :class:`LIFParameters` (thresholds, time constants, surrogate).
        Defaults to Norse's standard LIF with a superspike surrogate gradient.
    """

    def __init__(
        self,
        spec: SudokuSpec,
        hidden: tuple[int, ...] = (512, 256),
        *,
        T: int = 10,
        encoding: str = "constant",
        lif_params: LIFParameters | None = None,
    ) -> None:
        super().__init__()
        if encoding not in ("constant", "poisson"):
            raise ValueError(f"encoding must be 'constant' or 'poisson', got {encoding!r}")
        self.spec = spec
        self.T = T
        self.encoding = encoding
        self.in_features = spec.num_cells * (spec.side + 1)
        self.out_features = spec.num_cells * spec.side
        p = lif_params or LIFParameters()

        # Linear projections and their spiking cells, layer by layer.
        self.linears = nn.ModuleList()
        self.cells = nn.ModuleList()
        prev = self.in_features
        for width in hidden:
            self.linears.append(nn.Linear(prev, width))
            self.cells.append(LIFCell(p))
            prev = width

        # Non-spiking leaky-integrator readout: its membrane voltage carries the logits.
        self.readout_linear = nn.Linear(prev, self.out_features)
        self.readout = LICell()

    def _encode(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return the per-timestep input current for step t (same shape as ``x``)."""
        if self.encoding == "constant":
            return x
        # poisson: Bernoulli-sample spikes from the one-hot probabilities.
        return torch.bernoulli(x, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the T-step simulation; return logits ``(B, num_cells, side)``.

        The readout voltage is averaged over time (rate readout), which gives a
        smooth, well-conditioned target for cross-entropy.
        """
        states: list = [None] * len(self.cells)
        read_state = None
        voltage_sum = torch.zeros(x.shape[0], self.out_features, device=x.device, dtype=x.dtype)

        for _ in range(self.T):
            cur = self._encode(x)
            for i, (lin, cell) in enumerate(zip(self.linears, self.cells)):
                cur = lin(cur)
                cur, states[i] = cell(cur, states[i])
            out = self.readout_linear(cur)
            v, read_state = self.readout(out, read_state)
            voltage_sum = voltage_sum + v

        logits = voltage_sum / self.T
        return logits.view(-1, self.spec.num_cells, self.spec.num_digits)

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

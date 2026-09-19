"""FlyWire-constrained recurrent spiking network (PLAN.md Phase 5 / §6, Model 2).

The core experiment. A recurrent LIF population whose recurrent connectivity is fixed
to a FlyWire subgraph: only the synapses that exist in the connectome carry a weight,
and that weight is what training adjusts. Formally the recurrent weight matrix is

    W_effective = w_edge  ⊙  A          (A = fixed sparse connectome mask)

so the *topology* is biological and immutable while the *strengths* are learned
(PLAN.md §8). Optional Dale's-law signs pin each synapse's excitatory/inhibitory
polarity by the presynaptic neuron's predicted transmitter.

Architecture (same external I/O contract as the other models):

    one-hot board (B, in_features)
      -> input projection  Linear(in_features, N)        [dense, trainable]
      -> recurrent FlyWire LIF population of N neurons     [sparse mask A, T steps]
      -> output projection Linear(N, out_features)         [dense, trainable]
      -> LICell readout, voltage averaged over time -> logits (B, num_cells, side)

The recurrent step computes, for neuron j, the current sum_i W[i,j] * spike_i using a
single sparse matmul. Weights live only on the ~E existing edges (E << N^2), so memory
scales with the connectome's edge count, not N^2 (PLAN.md §20).

Trainable regimes (PLAN.md §19):
* ``train_recurrent=True``  — edge weights are learned (mask fixed).            [A]
* ``use_signs=True``        — magnitudes learned, signs fixed by transmitter.   [B]
* ``train_recurrent=False`` — recurrent weights frozen; only I/O projections
  train, testing whether the untrained connectome itself provides useful structure. [C]
"""

from __future__ import annotations

import numpy as np
import torch
from norse.torch import LICell, LIFCell, LIFParameters
from torch import nn

from connectome.adjacency import Connectome
from sudoku.core import SudokuSpec


class FlyWireSNN(nn.Module):
    """Recurrent LIF network with connectivity constrained by a FlyWire subgraph.

    Parameters
    ----------
    spec:
        Board geometry (sets input/output widths).
    connectome:
        The FlyWire subgraph. ``connectome.adj`` (N x N sparse) fixes the recurrent
        topology; ``N = connectome.num_nodes`` is the recurrent population size.
    T:
        Simulation timesteps (default 10).
    encoding:
        ``"constant"`` or ``"poisson"`` input encoding (see :class:`SudokuDenseSNN`).
    train_recurrent:
        If False, freeze the recurrent edge weights (Regime C).
    signs:
        Optional per-neuron sign vector ``(N,)`` of +1/-1 (from
        :func:`connectome.adjacency.signs_from_meta`). When given, each edge's weight
        is ``|w| * sign[pre]`` so polarity is fixed by the presynaptic transmitter
        (Regime B). When None, weights are unconstrained in sign.
    recurrent_scale:
        Scale for the initial edge-weight magnitudes.
    lif_params:
        Optional Norse :class:`LIFParameters`.
    """

    def __init__(
        self,
        spec: SudokuSpec,
        connectome: Connectome,
        *,
        T: int = 10,
        encoding: str = "constant",
        train_recurrent: bool = True,
        signs: np.ndarray | None = None,
        recurrent_scale: float = 1.0,
        lif_params: LIFParameters | None = None,
    ) -> None:
        super().__init__()
        if encoding not in ("constant", "poisson"):
            raise ValueError(f"encoding must be 'constant' or 'poisson', got {encoding!r}")
        self.spec = spec
        self.T = T
        self.encoding = encoding
        self.N = connectome.num_nodes
        self.in_features = spec.num_cells * (spec.side + 1)
        self.out_features = spec.num_cells * spec.side
        self.use_signs = signs is not None
        p = lif_params or LIFParameters()

        # --- fixed recurrent topology (the connectome mask) ---
        coo = connectome.adj.tocoo()
        pre = torch.from_numpy(coo.row.astype(np.int64))  # presynaptic index i
        post = torch.from_numpy(coo.col.astype(np.int64))  # postsynaptic index j
        # Store transposed indices [j, i] so a single sparse.mm gives, for neuron j,
        # sum_i W[i,j] * spike_i. Registered as a buffer: fixed, not trained.
        self.register_buffer("rec_indices", torch.stack([post, pre], dim=0))
        n_edges = coo.nnz

        # Edge weight magnitudes: trainable parameter or frozen buffer.
        deg = max(float(coo.nnz) / max(self.N, 1), 1.0)  # ~mean degree, for init scale
        init = torch.randn(n_edges) * (recurrent_scale / np.sqrt(deg))
        if train_recurrent:
            self.rec_weight = nn.Parameter(init)
        else:
            self.register_buffer("rec_weight", init)

        if self.use_signs:
            sign_per_edge = torch.from_numpy(signs[coo.row].astype(np.float32))
            self.register_buffer("edge_sign", sign_per_edge)

        # --- dense trainable projections + spiking cells ---
        self.input_linear = nn.Linear(self.in_features, self.N)
        self.lif = LIFCell(p)
        self.output_linear = nn.Linear(self.N, self.out_features)
        self.readout = LICell()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Per-timestep input current (constant direct current or Poisson spikes)."""
        if self.encoding == "constant":
            return x
        return torch.bernoulli(x)

    def _recurrent_current(self, spikes: torch.Tensor) -> torch.Tensor:
        """Recurrent input current from the previous step's spikes via the sparse mask.

        Returns ``(B, N)`` where entry ``[b, j] = sum_i W[i, j] * spikes[b, i]``.
        """
        values = self.rec_weight
        if self.use_signs:
            values = values.abs() * self.edge_sign
        wt = torch.sparse_coo_tensor(
            self.rec_indices, values, (self.N, self.N), device=spikes.device
        )
        # sparse (N,N) @ dense (N,B) -> (N,B); transpose back to (B,N).
        return torch.sparse.mm(wt, spikes.t()).t()

    def forward(
        self, x: torch.Tensor, return_activity: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the T-step recurrent simulation; return logits ``(B, num_cells, side)``.

        When ``return_activity=True`` also returns the recurrent-population spike tensor
        ``(T, B, N)`` for visualization (PLAN §Phase 8 / docs/VIDEO_PLAN.md).
        """
        b = x.shape[0]
        lif_state = None
        read_state = None
        spikes = torch.zeros(b, self.N, device=x.device, dtype=x.dtype)
        voltage_sum = torch.zeros(b, self.out_features, device=x.device, dtype=x.dtype)
        activity: list[torch.Tensor] = []

        for _ in range(self.T):
            inp_cur = self.input_linear(self._encode(x))
            rec_cur = self._recurrent_current(spikes)
            spikes, lif_state = self.lif(inp_cur + rec_cur, lif_state)
            if return_activity:
                activity.append(spikes.detach())
            out = self.output_linear(spikes)
            v, read_state = self.readout(out, read_state)
            voltage_sum = voltage_sum + v

        logits = (voltage_sum / self.T).view(-1, self.spec.num_cells, self.spec.num_digits)
        if return_activity:
            return logits, torch.stack(activity)  # (T, B, N)
        return logits

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def recurrent_edges(self) -> int:
        """Number of fixed recurrent synapses (nonzeros in the connectome mask)."""
        return int(self.rec_indices.shape[1])

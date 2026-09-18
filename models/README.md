# `models/` — neural models

Model ladder (PLAN.md §6): **MLP → dense SNN → FlyWire-constrained SNN**. Every model
shares one I/O contract so training/eval code is model-agnostic:

```
input:  one-hot board   (B, num_cells * (side + 1))
output: move logits     (B, num_cells, side)     # [b, cell, d] scores digit d+1
```

## Present

| File | Model | Phase |
|------|-------|-------|
| `mlp.py` | `SudokuMLP` — feed-forward baseline (810→512→256→729 for 9×9) | 2 |
| `dense_snn.py` | `SudokuDenseSNN` — feed-forward LIF spiking net (Norse) | 3 |
| `flywire_snn.py` | `FlyWireSNN` — connectome-masked recurrent LIF SNN | 5 |

## FlyWire SNN notes

Recurrent LIF population whose connectivity is fixed to a FlyWire subgraph:
`W_effective = w_edge ⊙ A` (A = fixed sparse mask). Input proj (810→N) + recurrent
FlyWire layer + output proj (N→729) + LICell readout. The recurrent step is a single
`torch.sparse.mm` (autograd on edge weights); memory scales with edge count, not N².

Regimes (PLAN.md §19): `train_recurrent=True` (A), `use_signs=True` Dale's law (B),
`train_recurrent=False` frozen connectome (C).

### VRAM scaling on RTX 2060 (batch 64, T=10, top_degree subgraph)

| N | edges | params | fwd+bwd | peak VRAM |
|---|------:|-------:|--------:|----------:|
| 1k | 22k | 1.6M | 34 ms | 59 MB |
| 5k | 201k | 7.9M | 56 ms | 306 MB |
| 10k | 470k | 15.9M | 138 ms | 795 MB |
| 20k | 955k | 31.8M | 337 ms | 2.4 GB |

Measured trainable ceiling ≈ **20k neurons** (2.4 GB). 30k+ OOMs even at batch 16:
`torch.sparse.mm` backward materializes a dense N×N gradient for the sparse weights
(30k² × 4B ≈ 3.35 GB), so the cap is O(N²) in the backward pass, not batch. Past ~20k
needs a custom sparse-gradient autograd (future work). Learns the task — 4×4 real 1k
subgraph reaches move_acc ~0.93.

## Dense SNN notes

- Each layer `Linear → LIFCell` (surrogate gradients); readout `Linear → LICell`
  whose membrane voltage, averaged over `T` steps, is the logit vector.
- The T-step simulation lives inside `forward(x)`, so the Phase 2 training loop and
  metrics run unchanged.
- Input encoding: `constant` (direct current, default, stable) or `poisson`.
- Trains with AMP (`amp: true`) on the RTX 2060 to save VRAM (PLAN.md §R4).

## Notes

- Sizes derive from `SudokuSpec`, so the same class handles 4×4 and 9×9.
- `num_parameters()` feeds the experiment matrix (matched-budget comparisons).

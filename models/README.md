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

## Coming (per phase)

| File | Model | Phase |
|------|-------|-------|
| `flywire_snn.py` | connectome-masked recurrent SNN | 5 |

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

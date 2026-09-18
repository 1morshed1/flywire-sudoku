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

## Coming (per phase)

| File | Model | Phase |
|------|-------|-------|
| `dense_snn.py` | dense LIF spiking net (Norse) | 3 |
| `flywire_snn.py` | connectome-masked recurrent SNN | 5 |

## Notes

- Sizes derive from `SudokuSpec`, so the same class handles 4×4 and 9×9.
- `num_parameters()` feeds the experiment matrix (matched-budget comparisons).

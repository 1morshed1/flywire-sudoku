# `training/` — supervised pipeline

## Modules

| File | Role |
|------|------|
| `dataset.py` | `build_dataset`, `SudokuMoveDataset` — (puzzle, solution) → tensors |
| `supervised.py` | `train_mlp`, `evaluate`, `masked_ce_loss`, `TrainConfig` |

## Task & loss (PLAN.md §12)

board → one-hot → model → per-cell logits. Cross-entropy is applied **only to
originally-empty cells** (clues carry no loss). Labels are the generator's unique
solution, so each blank has exactly one correct digit (PLAN.md §13 Option B).

## Metrics (PLAN.md §22)

- `move_acc` — fraction of empty cells whose argmax digit is correct.
- `solve_rate` — fraction of boards fully correct in **one** forward pass (strict;
  iterative constraint-loop solving arrives in Phase 7).

## Run the MLP baseline

```bash
# default 9x9 config
uv run --no-sync python -m training.supervised --config configs/mlp_baseline.yaml

# quick 4x4 debug from Python
uv run --no-sync python -c "from training.supervised import TrainConfig, train_mlp; \
train_mlp(TrainConfig(side=4, n_train=2000, epochs=40, device='cpu'))"
```

## Reference numbers

- 4×4, easy, 2k puzzles, 40 epochs, CPU: move_acc ≈ 0.94, solve_rate ≈ 0.68.
- 9×9 headline numbers are logged in `memory-bank/progress.md` as runs complete.

Notes: `device: auto` uses the RTX 2060 when present. 9×9 dataset generation is the
slow step (~40–60 ms/puzzle); build once and reuse for larger sweeps.

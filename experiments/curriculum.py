"""Option B: curriculum 4→6→9 for FlyWire-SNN with recurrent-weight transfer.

Board size changes the dense I/O projections, but the connectome-masked recurrent
core (N neurons, fixed edges) can carry across stages. Each stage:

  1. build a fresh FlyWireSNN for the new side (same subgraph seed/method)
  2. copy recurrent edge weights from the previous stage (if any)
  3. train; I/O layers learn the new board encoding from scratch

Run:
  CUDA_VISIBLE_DEVICES=2 uv run --no-sync python -m experiments.curriculum \\
      --config configs/flywire_curriculum.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from training.supervised import TrainConfig, _load_yaml_config, train_supervised


# (side, n_train, n_val, epochs) — more data / epochs as the board grows.
_DEFAULT_STAGES: list[tuple[int, int, int, int]] = [
    (4, 10_000, 1_000, 20),
    (6, 20_000, 1_000, 20),
    (9, 40_000, 2_000, 40),
]


def run_curriculum(base: TrainConfig, *, verbose: bool = True) -> dict[str, float]:
    """Train FlyWire-SNN through 4→6→9, transferring recurrent weights each hop."""
    if base.model != "flywire_snn":
        raise ValueError(f"curriculum expects flywire_snn, got {base.model!r}")

    model = None
    last_metrics: dict[str, float] = {}
    for side, n_train, n_val, epochs in _DEFAULT_STAGES:
        # Only the final 9x9 stage runs the honest unseen solver eval + save.
        cfg = replace(
            base,
            side=side,
            n_train=n_train,
            n_val=n_val,
            epochs=epochs,
            eval_solver_seed=base.eval_solver_seed if side == 9 else -1,
            save_state=base.save_state if side == 9 else "",
        )
        if verbose:
            print(f"\n=== curriculum stage side={side} train={n_train} epochs={epochs} ===")
        last_metrics, model = train_supervised(
            cfg, verbose=verbose, return_model=True, warm_recurrent=model
        )
        # Keep model on CPU between stages to free VRAM; next stage rebuilds on device.
        model = model.cpu()
        if verbose:
            print(
                f"=== stage side={side} done: move_acc={last_metrics.get('move_acc', float('nan')):.3f} ==="
            )
    return last_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyWire curriculum 4→6→9")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    base = _load_yaml_config(args.config)
    metrics = run_curriculum(base)
    Path("results").mkdir(parents=True, exist_ok=True)
    print(
        f"[curriculum] final UNSEEN completion "
        f"{metrics.get('completion_rate_unseen', float('nan')):.3f}  "
        f"placement {metrics.get('placement_accuracy_unseen', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()

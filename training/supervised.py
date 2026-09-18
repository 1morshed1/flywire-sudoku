"""Supervised training loop and metrics for Sudoku move prediction.

Task (PLAN.md §12): map a one-hot board to per-cell digit logits and train with
cross-entropy applied **only to originally-empty cells**. Filled clues carry no loss.

Metrics (PLAN.md §22):
* ``move_acc``   — fraction of empty cells whose argmax digit is correct.
* ``solve_rate`` — fraction of boards where *every* empty cell is correct in one
  forward pass (a strict, single-shot solve; iterative filling comes in Phase 7).

This module is model-agnostic: it trains anything with the shared I/O contract
(input ``(B, in_features)`` -> logits ``(B, num_cells, side)``), so the same code
serves the MLP baseline and the later SNNs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from sudoku import SudokuSpec
from training.dataset import SudokuMoveDataset, build_dataset


def pick_device(prefer: str = "auto") -> torch.device:
    """Return the training device. ``auto`` uses CUDA when available."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def masked_ce_loss(logits: torch.Tensor, target: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over empty cells only.

    Parameters
    ----------
    logits: ``(B, num_cells, side)``
    target: ``(B, num_cells)`` int64 digit indices
    empty:  ``(B, num_cells)`` bool — True where a loss should be applied.

    Returns a scalar mean over all empty cells in the batch. Guards against an
    all-filled batch (returns 0) so the loop never divides by zero.
    """
    b, n, side = logits.shape
    flat_logits = logits.reshape(b * n, side)
    flat_target = target.reshape(b * n)
    flat_empty = empty.reshape(b * n)
    if not flat_empty.any():
        return logits.sum() * 0.0
    per_cell = F.cross_entropy(flat_logits, flat_target, reduction="none")
    return per_cell[flat_empty].mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compute move accuracy and single-shot solve rate over a loader."""
    model.eval()
    correct_cells = 0
    total_cells = 0
    solved_boards = 0
    total_boards = 0
    for x, target, empty in loader:
        x, target, empty = x.to(device), target.to(device), empty.to(device)
        pred = model(x).argmax(dim=-1)  # (B, num_cells)
        hit = (pred == target) & empty
        correct_cells += int(hit.sum())
        total_cells += int(empty.sum())
        # A board is solved iff all its empty cells are predicted correctly.
        per_board_empty = empty.sum(dim=1)
        per_board_hit = hit.sum(dim=1)
        solved_boards += int((per_board_hit == per_board_empty).sum())
        total_boards += x.shape[0]
    return {
        "move_acc": correct_cells / max(total_cells, 1),
        "solve_rate": solved_boards / max(total_boards, 1),
    }


@dataclass
class TrainConfig:
    """Configuration for a supervised run. Mirrors the YAML config files."""

    side: int = 9
    hidden: tuple[int, ...] = (512, 256)
    dropout: float = 0.0
    n_train: int = 5000
    n_val: int = 500
    difficulty: str = "easy"
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "auto"
    log_every: int = 1
    extra: dict = field(default_factory=dict)


def train_mlp(cfg: TrainConfig, *, verbose: bool = True) -> dict[str, float]:
    """Train the MLP baseline and return final validation metrics.

    Kept deliberately small and dependency-light (no W&B) so it runs anywhere; a
    richer experiment-tracking path can wrap this later.
    """
    from models import SudokuMLP  # local import keeps torch optional at module load

    torch.manual_seed(cfg.seed)
    spec = SudokuSpec.from_side(cfg.side)
    device = pick_device(cfg.device)

    train_t = build_dataset(spec, cfg.n_train, difficulty=cfg.difficulty, seed=cfg.seed)
    val_t = build_dataset(spec, cfg.n_val, difficulty=cfg.difficulty, seed=cfg.seed + 1)
    train_loader = DataLoader(SudokuMoveDataset(train_t), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(SudokuMoveDataset(val_t), batch_size=cfg.batch_size)

    model = SudokuMLP(spec, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if verbose:
        print(
            f"[mlp] side={cfg.side} params={model.num_parameters():,} "
            f"device={device} train={cfg.n_train} val={cfg.n_val}"
        )

    metrics: dict[str, float] = {}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for x, target, empty in train_loader:
            x, target, empty = x.to(device), target.to(device), empty.to(device)
            opt.zero_grad()
            loss = masked_ce_loss(model(x), target, empty)
            loss.backward()
            opt.step()
            running += float(loss) * x.shape[0]
        metrics = evaluate(model, val_loader, device)
        if verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs):
            print(
                f"[mlp] epoch {epoch:3d}  loss {running / cfg.n_train:.4f}  "
                f"move_acc {metrics['move_acc']:.3f}  solve_rate {metrics['solve_rate']:.3f}"
            )

    metrics["params"] = float(model.num_parameters())
    return metrics


def _load_yaml_config(path: str) -> TrainConfig:
    """Build a :class:`TrainConfig` from a YAML file (keys mirror the dataclass)."""
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if "hidden" in raw:
        raw["hidden"] = tuple(raw["hidden"])
    known = {k: raw[k] for k in raw if k in TrainConfig.__dataclass_fields__}
    return TrainConfig(**known)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Train the Sudoku MLP baseline.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    args = parser.parse_args()
    config = _load_yaml_config(args.config) if args.config else TrainConfig()
    train_mlp(config)

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

from dataclasses import dataclass, field, replace

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

    model: str = "mlp"  # "mlp" | "dense_snn"
    side: int = 9
    hidden: tuple[int, ...] = (512, 256)
    dropout: float = 0.0  # MLP only
    # Spiking-model options (ignored by the MLP):
    T: int = 10
    encoding: str = "constant"  # "constant" | "poisson"
    # FlyWire-SNN options (used only when model == "flywire_snn"):
    n_neurons: int = 1000
    sample_method: str = "top_degree"  # top_degree | bfs | bfs_sensory | random
    conn_weight: str = "count"  # edge weight source for the connectome mask
    topology: str = "flywire"  # flywire | shuffled | random | dense | feedforward
    topology_seed: int = 0
    train_recurrent: bool = True  # False = Regime C (frozen connectome)
    use_signs: bool = False  # True = Regime B (Dale's-law signs)
    recurrent_scale: float = 1.0
    snn_dropout: float = 0.0  # dropout on recurrent spikes (FlyWireSNN regularization)
    n_train: int = 5000
    n_val: int = 500
    difficulty: str = "easy"
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    amp: bool = False  # mixed precision (CUDA only); PLAN.md §R4
    seed: int = 0
    device: str = "auto"
    log_every: int = 1
    save_state: str = ""  # path to torch.save the trained model state_dict (for video/eval)
    eval_solver_seed: int = -1  # if >=0, run an honest unseen-seed solver eval after training
    extra: dict = field(default_factory=dict)


def build_model(spec: SudokuSpec, cfg: TrainConfig) -> nn.Module:
    """Instantiate the model named by ``cfg.model`` (shared I/O contract)."""
    if cfg.model == "mlp":
        from models import SudokuMLP

        return SudokuMLP(spec, hidden=cfg.hidden, dropout=cfg.dropout)
    if cfg.model == "dense_snn":
        from models import SudokuDenseSNN

        return SudokuDenseSNN(spec, hidden=cfg.hidden, T=cfg.T, encoding=cfg.encoding)
    if cfg.model == "flywire_snn":
        from connectome.pipeline import subgraph
        from connectome.topology import apply_topology
        from models import FlyWireSNN

        sub, signs = subgraph(
            cfg.n_neurons, method=cfg.sample_method, weight=cfg.conn_weight, seed=cfg.seed
        )
        # Topology ablation: swap the connectome mask for a matched control (§17-18).
        # node_ids are preserved, so the sign vector stays aligned.
        sub = apply_topology(sub, cfg.topology, seed=cfg.topology_seed)
        return FlyWireSNN(
            spec,
            sub,
            T=cfg.T,
            encoding=cfg.encoding,
            train_recurrent=cfg.train_recurrent,
            signs=signs if cfg.use_signs else None,
            recurrent_scale=cfg.recurrent_scale,
            dropout=cfg.snn_dropout,
        )
    raise ValueError(f"unknown model {cfg.model!r}")


def train_supervised(
    cfg: TrainConfig, *, verbose: bool = True, return_model: bool = False
) -> dict[str, float] | tuple[dict[str, float], nn.Module]:
    """Train any model with the shared contract; return final validation metrics.

    Model-agnostic: dispatches on ``cfg.model``. Optional AMP (mixed precision) is
    used only on CUDA and helps the spiking models fit the 6 GB budget (PLAN.md §R4).
    Kept dependency-light (no W&B) so it runs anywhere.

    Parameters
    ----------
    return_model:
        If True, return ``(metrics, model)`` so callers (e.g. the Phase 7 solver
        evaluation) can use the trained network directly. The model stays on its
        training device.
    """
    torch.manual_seed(cfg.seed)
    spec = SudokuSpec.from_side(cfg.side)
    device = pick_device(cfg.device)

    train_t = build_dataset(spec, cfg.n_train, difficulty=cfg.difficulty, seed=cfg.seed)
    val_t = build_dataset(spec, cfg.n_val, difficulty=cfg.difficulty, seed=cfg.seed + 1)
    train_loader = DataLoader(SudokuMoveDataset(train_t), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(SudokuMoveDataset(val_t), batch_size=cfg.batch_size)

    model = build_model(spec, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    # GradScaler is a no-op when disabled, so it is safe to construct on CPU too.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if verbose:
        print(
            f"[{cfg.model}] side={cfg.side} params={model.num_parameters():,} "
            f"device={device} amp={use_amp} train={cfg.n_train} val={cfg.n_val}"
        )

    metrics: dict[str, float] = {}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for x, target, empty in train_loader:
            x, target, empty = x.to(device), target.to(device), empty.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = masked_ce_loss(model(x), target, empty)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss) * x.shape[0]
        metrics = evaluate(model, val_loader, device)
        if verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs):
            print(
                f"[{cfg.model}] epoch {epoch:3d}  loss {running / cfg.n_train:.4f}  "
                f"move_acc {metrics['move_acc']:.3f}  solve_rate {metrics['solve_rate']:.3f}"
            )

    metrics["params"] = float(model.num_parameters())

    # Honest generalization check: run the autonomous solver on puzzles drawn from a
    # seed the model never trained or validated on (train=seed, val=seed+1). This is
    # the number that exposes 9x9 overfitting — evaluating on seed 0 leaks train data.
    if cfg.eval_solver_seed >= 0:
        from evaluation.metrics import evaluate_solver

        if cfg.eval_solver_seed in (cfg.seed, cfg.seed + 1):
            raise ValueError(
                f"eval_solver_seed {cfg.eval_solver_seed} overlaps train/val seeds "
                f"({cfg.seed}, {cfg.seed + 1}); pick a disjoint seed for an honest eval."
            )
        solver_metrics = evaluate_solver(
            model,
            spec,
            n_puzzles=100,
            difficulty=cfg.difficulty,
            device=device,
            seed=cfg.eval_solver_seed,
        )
        metrics["completion_rate_unseen"] = solver_metrics["completion_rate"]
        metrics["placement_accuracy_unseen"] = solver_metrics["placement_accuracy"]
        if verbose:
            print(
                f"[{cfg.model}] UNSEEN(seed={cfg.eval_solver_seed}) "
                f"completion {solver_metrics['completion_rate']:.3f}  "
                f"placement_acc {solver_metrics['placement_accuracy']:.3f}"
            )

    if cfg.save_state:
        from pathlib import Path

        Path(cfg.save_state).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), cfg.save_state)
        if verbose:
            print(f"[{cfg.model}] saved state -> {cfg.save_state}")

    if return_model:
        return metrics, model
    return metrics


def train_mlp(cfg: TrainConfig, *, verbose: bool = True) -> dict[str, float]:
    """Backwards-compatible wrapper: train with ``cfg.model`` forced to ``"mlp"``."""
    if cfg.model != "mlp":
        cfg = replace(cfg, model="mlp")
    return train_supervised(cfg, verbose=verbose)


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

    parser = argparse.ArgumentParser(description="Train a Sudoku model (MLP or SNN).")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    args = parser.parse_args()
    config = _load_yaml_config(args.config) if args.config else TrainConfig()
    train_supervised(config)

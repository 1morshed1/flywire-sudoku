"""Render a solve as an mp4: board filling in, synced with neuron firing (Tier 1).

Left panel  — the Sudoku board; the just-placed cell is highlighted each step.
Right panel — a spike raster of the most-active recurrent neurons over the T
simulation timesteps of that solve step.

One video frame per solve step. Requires a :class:`SolveResult` produced with
``capture_activity=True`` (so ``result.activity`` holds per-step ``(T, N)`` spikes),
which currently means the FlyWire SNN. See docs/VIDEO_PLAN.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from evaluation.solver_loop import SolveResult
from sudoku.core import SudokuSpec

# Point matplotlib at the ffmpeg binary bundled with imageio-ffmpeg.
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def _draw_board(ax, spec: SudokuSpec, board: np.ndarray, highlight: tuple[int, int] | None):
    """Draw the Sudoku grid with digits, box lines, and an optional highlighted cell."""
    side = spec.side
    ax.clear()
    ax.set_xlim(0, side)
    ax.set_ylim(0, side)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    if highlight is not None:
        r, c = highlight
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#ffe08a", zorder=0))

    for i in range(side + 1):
        lw = 2.2 if i % spec.box_rows == 0 else 0.6
        ax.axhline(i, color="black", lw=lw)
        lw = 2.2 if i % spec.box_cols == 0 else 0.6
        ax.axvline(i, color="black", lw=lw)

    for r in range(side):
        for c in range(side):
            v = int(board[r, c])
            if v != 0:
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                )


def _top_neurons(activity: list[np.ndarray], k: int) -> np.ndarray:
    """Indices of the ``k`` most-active neurons across the whole solve (stable rows)."""
    total = np.zeros(activity[0].shape[1])
    for a in activity:
        total += a.sum(axis=0)
    k = min(k, total.shape[0])
    return np.argsort(total)[::-1][:k]


def render_solve_video(
    result: SolveResult,
    spec: SudokuSpec,
    out_path: str | Path,
    *,
    fps: int = 2,
    max_neurons: int = 100,
    title: str = "FlyWire SNN solving Sudoku",
) -> Path:
    """Render ``result`` to an mp4 at ``out_path``; return the path.

    ``result`` must carry per-step activity (solve with ``capture_activity=True``).
    """
    if not result.activity:
        raise ValueError("result has no activity; solve with capture_activity=True")

    neurons = _top_neurons(result.activity, max_neurons)
    n_steps = len(result.placements)

    fig, (ax_board, ax_spk) = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    def draw(frame: int):
        # frame 0 = initial board; frames 1..n_steps = after each placement.
        board = result.trajectory[frame]
        highlight = result.placements[frame - 1][:2] if frame > 0 else None
        _draw_board(ax_board, spec, board, highlight)
        if frame > 0:
            place = result.placements[frame - 1]
            ax_board.set_title(f"step {frame}: place ({place[0]},{place[1]}) = {place[2]}")
        else:
            ax_board.set_title("initial puzzle")

        ax_spk.clear()
        ax_spk.set_title(f"recurrent spikes (top {len(neurons)} neurons)")
        ax_spk.set_xlabel("timestep")
        ax_spk.set_ylabel("neuron")
        if frame > 0:
            raster = result.activity[frame - 1][:, neurons].T  # (neurons, T)
            ax_spk.imshow(raster, aspect="auto", cmap="magma", interpolation="nearest")
        else:
            n_t = result.activity[0].shape[0]
            ax_spk.imshow(np.zeros((len(neurons), n_t)), aspect="auto", cmap="magma")

    anim = FuncAnimation(fig, draw, frames=n_steps + 1, interval=1000 / fps)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(str(out), writer=writer)
    plt.close(fig)
    return out


def make_solve_video(
    out_path: str | Path = "results/flywire_solve.mp4",
    *,
    side: int = 4,
    n_neurons: int = 1000,
    epochs: int = 40,
    difficulty: str = "easy",
    seed: int = 0,
    device: str = "cpu",
) -> Path:
    """Train a FlyWire SNN, solve one puzzle capturing activity, and render the video."""
    import numpy as np

    from evaluation.solver_loop import solve_with_model
    from sudoku import SudokuSpec, make_puzzle
    from training.supervised import TrainConfig, train_supervised

    spec = SudokuSpec.from_side(side)
    cfg = TrainConfig(
        model="flywire_snn",
        side=side,
        n_neurons=n_neurons,
        T=10,
        n_train=2000,
        n_val=200,
        difficulty=difficulty,
        epochs=epochs,
        batch_size=64,
        lr=1e-3,
        seed=seed,
        device=device,
    )
    _, model = train_supervised(cfg, verbose=False, return_model=True)

    rng = np.random.default_rng(seed + 100)
    puz = make_puzzle(spec, rng, difficulty)
    result = solve_with_model(
        model,
        spec,
        puz.puzzle,
        device=device,
        solution=puz.solution,
        capture_activity=True,
    )
    path = render_solve_video(result, spec, out_path)
    print(f"solved={result.solved} steps={result.steps} -> {path}")
    return path


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Render a FlyWire SNN solving Sudoku.")
    parser.add_argument("--out", default="results/flywire_solve.mp4")
    parser.add_argument("--side", type=int, default=4)
    parser.add_argument("--n-neurons", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    make_solve_video(
        args.out,
        side=args.side,
        n_neurons=args.n_neurons,
        epochs=args.epochs,
        device=args.device,
    )

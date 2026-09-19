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


def render_solve_video_3d(
    result: SolveResult,
    spec: SudokuSpec,
    coords: np.ndarray,
    out_path: str | Path,
    *,
    fps: int = 6,
    frames_per_timestep: int = 2,
    fire_color: str = "#00e5ff",
    title: str = "FlyWire brain solving Sudoku",
) -> Path:
    """Tier-2 render: 3D neurons in real soma coordinates blinking as they spike.

    One video frame per *simulation timestep* (not per solve step), so the recurrent
    population visibly fires: a dim grey point cloud shows the whole brain, and neurons
    that spike at the current timestep light up bright and large. The board (right)
    reveals each placement on the final timestep of that solve step.

    ``coords`` is ``(N, 3)`` aligned to the population; NaN rows are dropped.
    ``frames_per_timestep`` repeats each timestep to slow the video down; raise it (or
    lower ``fps``) for a slower, longer clip.
    """
    if not result.activity:
        raise ValueError("result has no activity; solve with capture_activity=True")

    finite = np.isfinite(coords).all(axis=1)
    xyz = coords[finite]
    acts = [a[:, finite] for a in result.activity]  # each (T, n_finite) binary spikes
    n_t = acts[0].shape[0]
    n_steps = len(result.placements)

    # One entry per (solve step, timestep), each repeated frames_per_timestep times,
    # plus a hold on the final solved board.
    plan = [(s, t) for s in range(n_steps) for t in range(n_t) for _ in range(frames_per_timestep)]
    plan += [(n_steps - 1, n_t - 1)] * (fps * 2)  # ~2 s hold at the end

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_board = fig.add_subplot(1, 2, 2)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("white")

    def draw(idx: int):
        s, t = plan[idx]
        fire = acts[s][t] > 0

        ax3d.clear()
        ax3d.set_axis_off()
        # Dim grey cloud = the whole recurrent population (the brain shape).
        ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="#c9c9c9", s=3, alpha=0.18, linewidths=0)
        # Bright overlay = neurons spiking right now.
        if fire.any():
            ax3d.scatter(
                xyz[fire, 0],
                xyz[fire, 1],
                xyz[fire, 2],
                c=fire_color,
                s=30,
                alpha=0.95,
                linewidths=0,
            )
        azim = -70 + 150 * (idx / max(len(plan) - 1, 1))  # slow rotate over the clip
        ax3d.view_init(elev=18, azim=azim)
        ax3d.set_title(f"step {s + 1}/{n_steps} · t={t + 1}/{n_t} · {int(fire.sum())} firing")

        # Board: pre-placement board while computing; reveal the placement on last t.
        last_t = t == n_t - 1
        board = result.trajectory[s + 1] if last_t else result.trajectory[s]
        highlight = result.placements[s][:2] if last_t else None
        _draw_board(ax_board, spec, board, highlight)
        p = result.placements[s]
        ax_board.set_title(f"placed ({p[0]},{p[1]}) = {p[2]}" if last_t else "computing…")

    anim = FuncAnimation(fig, draw, frames=len(plan), interval=1000 / fps)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=FFMpegWriter(fps=fps, bitrate=2800))
    plt.close(fig)
    return out


def make_solve_video(
    out_path: str | Path = "results/flywire_solve.mp4",
    *,
    tier: int = 1,
    side: int = 4,
    n_neurons: int = 1000,
    epochs: int = 40,
    difficulty: str = "easy",
    seed: int = 0,
    device: str = "cpu",
) -> Path:
    """Train a FlyWire SNN, solve one puzzle capturing activity, and render the video.

    ``tier=1`` renders board + spike raster (no extra data). ``tier=2`` renders a 3D
    fly-brain in real soma coordinates (fetches the annotations file).
    """
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

    if tier == 2:
        from connectome.coordinates import neuron_coordinates
        from connectome.pipeline import subgraph

        # Reproduce the exact node order the model was built with (same cfg.seed).
        sub, _ = subgraph(
            n_neurons, method=cfg.sample_method, weight=cfg.conn_weight, seed=cfg.seed
        )
        coords = neuron_coordinates(sub.node_ids)
        path = render_solve_video_3d(result, spec, coords, out_path)
    else:
        path = render_solve_video(result, spec, out_path)
    print(f"tier={tier} solved={result.solved} steps={result.steps} -> {path}")
    return path


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Render a FlyWire SNN solving Sudoku.")
    parser.add_argument("--out", default="results/flywire_solve.mp4")
    parser.add_argument("--tier", type=int, default=1, choices=(1, 2))
    parser.add_argument("--side", type=int, default=4)
    parser.add_argument("--n-neurons", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    make_solve_video(
        args.out,
        tier=args.tier,
        side=args.side,
        n_neurons=args.n_neurons,
        epochs=args.epochs,
        device=args.device,
    )

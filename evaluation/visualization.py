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


def _draw_board(
    ax,
    spec: SudokuSpec,
    board: np.ndarray,
    highlight: tuple[int, int] | None,
    *,
    dark: bool = False,
):
    """Draw the Sudoku grid with digits, box lines, and an optional highlighted cell."""
    side = spec.side
    line_c = "white" if dark else "black"
    text_c = "white" if dark else "black"
    hi_c = "#3a5bd9" if dark else "#ffe08a"
    ax.clear()
    if dark:
        ax.set_facecolor("black")
    ax.set_xlim(0, side)
    ax.set_ylim(0, side)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    if highlight is not None:
        r, c = highlight
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color=hi_c, zorder=0))

    for i in range(side + 1):
        lw = 2.2 if i % spec.box_rows == 0 else 0.6
        ax.axhline(i, color=line_c, lw=lw)
        lw = 2.2 if i % spec.box_cols == 0 else 0.6
        ax.axvline(i, color=line_c, lw=lw)

    fs = 16 if side <= 4 else 11
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
                    fontsize=fs,
                    fontweight="bold",
                    color=text_c,
                )


# Neuropil buckets for coloring + the per-region activity panel (super_class -> bucket).
_REGION_BUCKETS: dict[str, str] = {
    "sensory": "sensory",
    "sensory_ascending": "sensory",
    "central_brain_intrinsic": "central",
    "optic_lobe_intrinsic": "optic",
    "visual_projection": "optic",
    "visual_centrifugal": "optic",
    "ascending": "motor",
    "descending": "motor",
    "motor": "motor",
}
_BUCKET_ORDER = ["sensory", "central", "optic", "motor", "other"]
_BUCKET_COLORS = {
    "sensory": "#00e5ff",
    "central": "#ffb300",
    "optic": "#ff4fd8",
    "motor": "#7cff6b",
    "other": "#9aa0a6",
}


def _buckets_for(regions: list[str] | None, n: int) -> np.ndarray:
    """Map each neuron's super_class to a bucket index (into _BUCKET_ORDER)."""
    idx = {b: i for i, b in enumerate(_BUCKET_ORDER)}
    if regions is None:
        return np.full(n, idx["other"], dtype=np.int64)
    return np.array(
        [idx.get(_REGION_BUCKETS.get(r or "", "other"), idx["other"]) for r in regions],
        dtype=np.int64,
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
    bg_coords: np.ndarray | None = None,
    adjacency=None,
    regions: list[str] | None = None,
    fps: int = 6,
    frames_per_timestep: int = 2,
    max_edges: int = 700,
    title: str = "FlyWire brain solving Sudoku",
) -> Path:
    """Tier-2 render (full FX): a connectome "playing Sudoku" in the viral-demo style.

    One video frame per *simulation timestep*. On a black stage:
    * a dim point cloud of the whole recurrent population, colored by neuropil region;
    * neurons spiking **now** flare with a layered glow (bloom) in their region color;
    * **active synapse edges** (outgoing from neurons that just fired) light up, so
      signal is seen propagating through the real wiring;
    * a **per-region activity panel** shows the live firing rate of each neuropil bucket;
    * the Sudoku board reveals each placement on the last timestep of its solve step.

    ``coords`` is ``(N, 3)`` aligned to the population; ``adjacency`` is the subgraph's
    sparse matrix (for edges); ``regions`` is the per-neuron ``super_class``. NaN-coord
    neurons are dropped.
    """
    if not result.activity:
        raise ValueError("result has no activity; solve with capture_activity=True")
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    finite = np.isfinite(coords).all(axis=1)
    xyz = coords[finite]
    acts = [a[:, finite] for a in result.activity]  # (T, n_finite) binary spikes
    n_t = acts[0].shape[0]
    n_steps = len(result.placements)

    buckets = _buckets_for(regions, coords.shape[0])[finite]
    bucket_rgba = np.array([_BUCKET_COLORS[_BUCKET_ORDER[b]] for b in buckets])

    # Bounds/aspect come from the full brain (bg_coords) when present, so the whole
    # anatomy fits and keeps its true proportions.
    ref = bg_coords if bg_coords is not None else xyz
    lo, hi = ref.min(0), ref.max(0)
    ext = np.maximum(hi - lo, 1e-6)
    box_aspect = ext / ext.max()

    # Precompute edge endpoints among finite neurons (remap to finite-local indices).
    seg_pre = seg_post = None
    if adjacency is not None:
        remap = -np.ones(coords.shape[0], dtype=np.int64)
        remap[np.flatnonzero(finite)] = np.arange(finite.sum())
        coo = adjacency.tocoo()
        keep = finite[coo.row] & finite[coo.col]
        seg_pre = remap[coo.row[keep]]
        seg_post = remap[coo.col[keep]]

    plan = [(s, t) for s in range(n_steps) for t in range(n_t) for _ in range(frames_per_timestep)]
    plan += [(n_steps - 1, n_t - 1)] * (fps * 3)  # ~3 s hold on the finished board

    fig = plt.figure(figsize=(13, 7), facecolor="black")
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_position([0.0, 0.02, 0.62, 0.9])  # fill the left region, minimal margin
    ax_board = fig.add_subplot(gs[0, 1])
    ax_reg = fig.add_subplot(gs[1, 1])
    fig.suptitle(title, fontsize=15, fontweight="bold", color="white")
    rng = np.random.default_rng(0)

    def draw(idx: int):
        s, t = plan[idx]
        fire = acts[s][t] > 0

        ax3d.clear()
        ax3d.set_axis_off()
        ax3d.set_facecolor("black")
        ax3d.set_xlim(lo[0], hi[0])
        ax3d.set_ylim(lo[1], hi[1])
        ax3d.set_zlim(lo[2], hi[2])
        ax3d.set_box_aspect(box_aspect, zoom=1.5)

        # Dense soma mist = the whole fly brain (the recognizable anatomy + 3D depth).
        if bg_coords is not None:
            ax3d.scatter(
                bg_coords[:, 0],
                bg_coords[:, 1],
                bg_coords[:, 2],
                c="#a9c7e8",
                s=0.7,
                alpha=0.28,
                linewidths=0,
                depthshade=True,
            )
        # Faint markers for the model's neurons (locate the circuit within the brain).
        ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=bucket_rgba, s=2, alpha=0.20, linewidths=0)

        # Active synapse edges: outgoing from neurons firing now.
        if seg_pre is not None and fire.any():
            active = fire[seg_pre]
            ap, aq = seg_pre[active], seg_post[active]
            if ap.size > max_edges:
                sel = rng.choice(ap.size, max_edges, replace=False)
                ap, aq = ap[sel], aq[sel]
            if ap.size:
                segs = np.stack([xyz[ap], xyz[aq]], axis=1)
                lc = Line3DCollection(segs, colors=bucket_rgba[ap], linewidths=0.5, alpha=0.30)
                ax3d.add_collection3d(lc)

        # Firing neurons: region-colored glow (bloom) + a white-hot core.
        if fire.any():
            fx = xyz[fire]
            fc = bucket_rgba[fire]
            for size, alpha in ((60, 0.08), (26, 0.20), (11, 0.9)):
                ax3d.scatter(fx[:, 0], fx[:, 1], fx[:, 2], c=fc, s=size, alpha=alpha, linewidths=0)
            ax3d.scatter(fx[:, 0], fx[:, 1], fx[:, 2], c="white", s=3, alpha=0.9, linewidths=0)

        # Frontal view (fly-brain face) with a gentle parallax swing for 3D feel.
        azim = -90 + 14 * np.sin(2 * np.pi * idx / max(len(plan) - 1, 1))
        ax3d.view_init(elev=80, azim=azim)
        ax3d.set_title(
            f"step {s + 1}/{n_steps} · t={t + 1}/{n_t} · {int(fire.sum())} firing",
            color="white",
            fontsize=11,
        )

        # Board.
        last_t = t == n_t - 1
        board = result.trajectory[s + 1] if last_t else result.trajectory[s]
        highlight = result.placements[s][:2] if last_t else None
        _draw_board(ax_board, spec, board, highlight, dark=True)
        p = result.placements[s]
        ax_board.set_title(
            f"placed ({p[0]},{p[1]}) = {p[2]}" if last_t else "computing…",
            color="white",
            fontsize=10,
        )

        # Per-region activity panel: instantaneous firing rate per bucket.
        ax_reg.clear()
        ax_reg.set_facecolor("black")
        rates = []
        labels = []
        colors = []
        for b, name in enumerate(_BUCKET_ORDER):
            mask = buckets == b
            if mask.sum() == 0:
                continue
            rates.append(float(fire[mask].mean()))
            labels.append(name)
            colors.append(_BUCKET_COLORS[name])
        ypos = np.arange(len(labels))
        ax_reg.barh(ypos, rates, color=colors)
        ax_reg.set_yticks(ypos)
        ax_reg.set_yticklabels(labels, color="white", fontsize=8)
        ax_reg.set_xlim(0, 1)
        ax_reg.tick_params(colors="white")
        ax_reg.set_title("region firing rate", color="white", fontsize=10)
        for spine in ax_reg.spines.values():
            spine.set_color("#444444")

    anim = FuncAnimation(fig, draw, frames=len(plan), interval=1000 / fps)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(
        str(out), writer=FFMpegWriter(fps=fps, bitrate=4000), savefig_kwargs={"facecolor": "black"}
    )
    plt.close(fig)
    return out


def make_solve_video(
    out_path: str | Path = "results/flywire_solve.mp4",
    *,
    tier: int = 1,
    side: int = 4,
    n_neurons: int = 1000,
    epochs: int = 40,
    n_train: int = 2000,
    difficulty: str = "easy",
    seed: int = 0,
    device: str = "cpu",
    state_path: str | None = None,
    max_attempts: int = 30,
) -> Path:
    """Train (or load) a FlyWire SNN, solve one puzzle capturing activity, render video.

    ``tier=1`` renders board + spike raster; ``tier=2`` renders the 3D fly-brain in real
    soma coordinates. ``state_path`` loads a saved model instead of training (and, after
    training, the freshly trained weights are saved there). Up to ``max_attempts``
    puzzles are tried so the rendered puzzle is one the model actually solves (important
    for 9×9, where the loop solves ~65%).
    """
    import numpy as np
    import torch

    from evaluation.solver_loop import solve_with_model
    from sudoku import SudokuSpec, make_puzzle
    from training.supervised import TrainConfig, build_model, train_supervised

    spec = SudokuSpec.from_side(side)
    cfg = TrainConfig(
        model="flywire_snn",
        side=side,
        n_neurons=n_neurons,
        T=10,
        n_train=n_train,
        n_val=200,
        difficulty=difficulty,
        epochs=epochs,
        batch_size=256,
        lr=1e-3,
        seed=seed,
        device=device,
    )
    if state_path and Path(state_path).exists():
        model = build_model(spec, cfg).to(device)
        model.load_state_dict(torch.load(state_path, map_location=device))
    else:
        _, model = train_supervised(cfg, verbose=False, return_model=True)
        if state_path:
            torch.save(model.state_dict(), state_path)

    # Draw puzzles from the training seed stream: the 9x9 FlyWire model overfits and
    # reliably solves only in-distribution puzzles, so this ensures a full solve to
    # render (the 4x4 model generalizes and solves unseen puzzles regardless).
    rng = np.random.default_rng(seed)
    result = None
    for _ in range(max_attempts):
        puz = make_puzzle(spec, rng, difficulty)
        result = solve_with_model(
            model,
            spec,
            puz.puzzle,
            device=device,
            solution=puz.solution,
            capture_activity=True,
        )
        if result.solved:
            break
    if not result.solved:
        raise RuntimeError(
            f"no puzzle solved in {max_attempts} attempts; train longer or raise max_attempts"
        )

    if tier == 2:
        from connectome.coordinates import all_coordinates, neuron_coordinates
        from connectome.pipeline import full_connectome, subgraph

        z_scale = 10.0  # undo FlyWire voxel anisotropy (z ≈ 40 nm vs x,y ≈ 4 nm)

        # Reproduce the exact node order the model was built with (same cfg.seed).
        sub, _ = subgraph(
            n_neurons, method=cfg.sample_method, weight=cfg.conn_weight, seed=cfg.seed
        )
        coords = neuron_coordinates(sub.node_ids)
        coords[:, 2] *= z_scale
        # Full-brain soma cloud as the dim anatomical backdrop (subsampled for speed).
        bg = all_coordinates()
        bg[:, 2] *= z_scale
        if len(bg) > 60000:
            bg = bg[np.random.default_rng(0).choice(len(bg), 60000, replace=False)]
        # Region (super_class) per neuron, aligned to the subgraph node order.
        meta, _ = full_connectome(weight=cfg.conn_weight)
        sc = dict(zip(meta["fafb_783_id"].to_list(), meta["super_class"].to_list()))
        regions = [sc.get(nid) for nid in sub.node_ids]
        # Keep the clip a sane length: fewer repeats per timestep when many solve steps.
        fpt = 1 if result.steps > 15 else 2
        path = render_solve_video_3d(
            result,
            spec,
            coords,
            out_path,
            bg_coords=bg,
            adjacency=sub.adj,
            regions=regions,
            frames_per_timestep=fpt,
        )
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
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state", default=None, help="load/save model state_dict here")
    args = parser.parse_args()
    make_solve_video(
        args.out,
        tier=args.tier,
        side=args.side,
        n_neurons=args.n_neurons,
        epochs=args.epochs,
        n_train=args.n_train,
        device=args.device,
        state_path=args.state,
    )

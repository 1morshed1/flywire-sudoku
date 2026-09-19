# Video Simulation Plan — Fly Brain Solving Sudoku

**Status:** Tier 1 AND Tier 2 DONE. `evaluation/visualization.py` renders the FlyWire
SNN solving 4×4 as mp4.
- Tier 1 (board ‖ spike raster): `python -m evaluation.visualization --out results/flywire_solve.mp4`
- Tier 2 (3D fly-brain in real soma coords ‖ board): `python -m evaluation.visualization --tier 2 --out results/flywire_solve_3d.mp4`
Soma coordinates come from the FlyWire annotations TSV (soma_x/y/z, `connectome/
coordinates.py`), fetched by `connectome.download.ensure_coordinates` (~31 MB). All
1000 subgraph neurons have coordinates. Outputs gitignored (regenerable).

**Goal:** a video showing the FlyWire-constrained SNN solving a Sudoku puzzle —
the board filling in step by step, synchronized with the connectome's neurons firing.

## What it actually is

Not a real-time brain simulation. An **offline render of captured tensors**:

```
run solver  ->  capture per-step board + per-timestep spikes  ->  render frames  ->  encode mp4
```

## Prerequisites (2 missing pieces)

1. **Phase 7 solve loop** — not built yet. The video needs the sequence of placements
   the model makes. Build the constraint-propagation solver first (or a stub).
2. **Activity capture** — instrument `FlyWireSNN.forward` to optionally return the spike
   tensor `(T, N)` for each solve step (one `return_activity=True` flag).

## Setup steps

### 1. Dependencies

Add to `pyproject.toml`:

```
imageio-ffmpeg   # bundled ffmpeg binary for mp4 encoding (no system install needed)
# matplotlib already present
```

Optional: system `ffmpeg` instead of the bundled one.

### 2. Capture spikes in the model

```python
def forward(self, x, return_activity=False):
    spikes_log = []
    for t in range(self.T):
        # ... spikes, lif_state = self.lif(inp_cur + rec_cur, lif_state)
        if return_activity:
            spikes_log.append(spikes.detach())
    # ...
    if return_activity:
        return logits, torch.stack(spikes_log)   # (T, N)
    return logits
```

### 3. Solve and log

Run the constraint loop; per placement step record:
- `board_state` (the grid after the placement)
- `chosen` (cell, digit)
- `activity` — spikes `(T, N)` from that step's forward pass

### 4. Render (`evaluation/visualization.py`)

matplotlib `FuncAnimation`, two panels:
- **Left:** Sudoku grid; cells fill in over steps; newest placement highlighted.
- **Right:** neural activity — spike raster (neuron × time) OR a neuron scatter that
  lights up as neurons fire.

### 5. Encode

```python
anim.save("solve.mp4", fps=..., writer="ffmpeg")
```

## Two fidelity tiers

### Tier 1 — simple (no extra data)
Board + spike raster / activity heatmap side panel. Pure matplotlib. Works at
N = 1k–20k. Recommended first.

### Tier 2 — the "fly brain" look
3D scatter of neurons in **real FlyWire soma coordinates**, firing lights them up,
board overlaid. Needs one extra data file: soma xyz positions.
- `meta.feather` has **no** coordinates — must fetch a positions/coordinates export
  from the FlyWire GCS mirror (soma or skeleton coords) and join on `fafb_783_id`.
- Without coords: fall back to a 2D force-directed layout (still readable).

## Reality checks

- **N cap:** trainable subgraph is ~1k–20k (VRAM). 1k neurons = clean, readable video.
  Full 139k = static image only (can't train, but could render a frozen connectome).
- **Soma coords** make it recognizably fly-brain-shaped; needed only for Tier 2.
- **ffmpeg** is the only external dependency.

## Build order (when resumed)

1. Phase 7 constraint-propagation solver (produces the solve sequence).
2. Add `return_activity` to `FlyWireSNN.forward`.
3. `evaluation/visualization.py` — Tier 1 renderer (board + spike raster) → mp4.
4. (Optional) fetch soma coordinates → Tier 2 3D fly-brain render.

## Open questions for later

- Tier 1 only, or also Tier 2 (3D)?
- Fetch real neuron coordinates for the 3D look?
- Which trained model to visualize (flywire_snn 1k is the natural choice)?
- 4×4 (fast, clean) or 9×9 (impressive but needs a model that actually solves it —
  depends on Phase 7 results)?

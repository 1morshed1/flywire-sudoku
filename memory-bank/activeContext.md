# Active Context

## Current focus

Phases 0-3 done. Next: Phase 4 FlyWire connectome data.

## Recent changes

- Phase 2: MLP baseline + supervised pipeline. 4×4 move_acc ~0.94.
- Phase 3: `models/dense_snn.py` (SudokuDenseSNN, Norse LIF + LICell readout,
  T-step sim in forward, constant/poisson encoding). Generalized training:
  `train_supervised` + `build_model` dispatch, AMP support. 33 tests pass.
  Gate met: 4×4 SNN move_acc ~0.94 @50ep.

## Next steps

1. Phase 4 (no GPU): FlyWire connectome data engineering.
   - Fetch Codex public parquet dumps (neurons, synapses, neurotransmitters).
   - `connectome/loader.py` (Polars lazy read), `filter.py` (synapse count ≥5),
     `adjacency.py` (sparse COO/scipy .npz), `sampling.py` (PRINCIPLED subsampling:
     largest WCC / BFS-from-sensory / top-degree — never random), `statistics.py`.
   - Store under data/flywire/ (gitignored). Log dump version + thresholds.
2. NEED FROM USER: confirm Codex dump source/URL, or whether to write a small
   download helper vs manual download. FlyWire data has citation/license terms.

## Task-framing note

Chose full-grid prediction (input 810 one-hot -> logits 81×9, CE over empty cells)
over single-cell Task A. Cleaner, gives move_acc + solve_rate directly, matches the
729 head in PLAN.md §6. The env still supports next_digit (Task A) if needed later.

## Active decisions (locked)

- Connectome: Codex public parquet, synapse count ≥ 5
- Norse + torch cu12; uv package manager
- Constraint-propagation solve loop (not RL spine)
- Principled subsampling only
- Signs-free first
- Curriculum 4×4 → 9×9
- Repo dirs grow per phase

## Open / pending

- Exact Norse/torch pin versions (verify at install)
- CAVE account only if live queries needed later
- RL fine-tuning optional after autonomous loop works

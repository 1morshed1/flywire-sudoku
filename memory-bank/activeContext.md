# Active Context

## Current focus

Phases 0-4 done. Next: Phase 5 FlyWire-constrained SNN.

## Recent changes

- Phase 3: dense SNN works (4×4 ~0.94), AMP verified on 2060.
- Phase 4: `connectome/` pipeline. FAFB v783 via public GCS mirror
  (gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/fafb_783/),
  no auth. Files: meta.feather (13MB), simple_edgelist.feather (302MB, cols
  pre/post/count/norm/total_input). count≥5 → 135,453 nodes / 3.53M edges,
  WCC 98.8%. Sparse scipy adjacency, transmitter signs, principled sampling.
  11 tests pass. Data in data/flywire/ (gitignored) + provenance.json.

## Data facts (memorize)

- meta id column = `fafb_783_id`; classes in `super_class` (sensory 16.9k,
  optic_lobe_intrinsic 77.8k, central_brain_intrinsic 32.5k, motor 106…).
- sensory seeds = super_class in {sensory, sensory_ascending} OR flow=='afferent'.
- transmitters: acetylcholine +1; gaba/glutamate −1; amines +1; unknown +1.

## Next steps (Phase 5)

1. `models/flywire_snn.py`: recurrent LIF SNN. Fixed sparse mask A from a sampled
   subgraph (connectome.sampling); trainable recurrent weights W, W_eff = W ⊙ A.
   Input projection (810→N) + recurrent FlyWire layer + output projection (N→729).
   Use torch.sparse for the masked recurrent matmul. T=10, surrogate grads, AMP.
2. Scale N=1k→5k→10k→20k; VRAM-benchmark each rung (log neurons/edges/VRAM/time).
3. Reuse the supervised loop; extend build_model with 'flywire_snn' + connectome cfg.
4. Optional: cache a built subgraph (connectome.adjacency.save) for fast reload.

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

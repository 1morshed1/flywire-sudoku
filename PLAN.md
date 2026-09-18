# FlyWire-Sudoku — Finalized Plan

**Goal:** A FlyWire-connectome-constrained spiking neural network (SNN) trained to
perform symbolic reasoning on Sudoku. The Sudoku task is conventional
supervised/RL learning; the scientific interest is that the neural controller's
recurrent connectivity is **constrained by the real FlyWire Drosophila
connectome**.

Core research question:
> How does biological connectivity constrain learning on an abstract symbolic
> reasoning task?

---

## 1. Hardware Budget (fixed constraint)

| Resource | Value |
|----------|-------|
| GPU | NVIDIA RTX 2060, **6 GB** VRAM, compute 7.5 (Turing) |
| CPU | 12 cores |
| RAM | 15 GB (~8.7 GB free) |
| Python | 3.12.3 |
| Package mgr | **uv** 0.11.17 |
| torch | not yet installed |

**VRAM reality (float32, BPTT stores every timestep):**

| N neurons | batch | T | approx activation VRAM | verdict |
|-----------|-------|---|------------------------|---------|
| 10k | 64 | 20 | ~1.5 GB | fits |
| 50k | 64 | 20 | ~7.5 GB | **won't fit** |
| 139k | any | any | — | inference/eval only |

**Trainable ceiling on this box: ~10k–20k neurons, batch 32–64, T=10, AMP +
gradient checkpointing.** Full 139k connectome is inference/eval only. This
still yields a full scaling study + topology ablations = the real science.

---

## 2. Locked Design Decisions

1. **Connectome:** real FlyWire, via **public Codex parquet dumps** (no auth).
   CAVE account created later only if we need live/versioned queries.
2. **Task:** full autonomous 9×9 solve — achieved via **constraint-propagation
   loop**, not per-step RL (see §5). RL is optional polish, not the spine.
3. **Framework:** **Norse** (PyTorch-native LIF) + torch (CUDA 12.x).
4. **Data loader:** written by us (Codex dumps → sparse adjacency).
5. **Action head:** start Task A (next-cell, 9-way classify) → grow to flat
   729-action head (81 cell × 9 digit) with legal-move masking.
6. **Signs:** **signs-free first.** Neurotransmitter excito/inhibitory signs
   added later as an ablation (Regime B), not day 1.
7. **Curriculum:** **4×4 Sudoku first** (action space 64), then 9×9. Generator/
   solver/env are size-generic — 9×9 is a config change, not a rewrite.
8. **Synapse threshold:** keep FlyWire edges with synapse count **≥ 5**.
9. **Subsampling:** **principled, never random** — largest weakly-connected
   component / BFS-from-sensory / top-degree. Random picks give dead islands.
10. **Env/repo growth:** dependencies eyeballed before install; repo dirs grow
    **per phase**, not all up front. Package manager: **uv**.

---

## 3. Key Refinements vs Original plan.txt

- **R1 — Constraint-propagation solve loop** replaces per-step RL as the
  autonomous-solve spine: SNN scores all empty cells, place highest-confidence
  legal digit, repeat until solved. Human-like, no sample-hungry RL required.
- **R2 — Principled connectome subsampling** (§2.9) to avoid disconnected
  subgraphs.
- **R3 — T=10 not 20** to start. Timesteps are the hidden VRAM multiplier under
  BPTT. Gradient checkpointing over time from day 1.
- **R4 — Mixed precision (AMP)** on Turing fp16 → ~40% VRAM cut, ~1.5× N ceiling.
- **R5 — Norse/torch version pin** verified at install (Norse may lag latest
  torch).
- **R6 — Polars lazy/streaming** for multi-GB synapse parquet (not eager Pandas);
  adjacency persisted as `scipy.sparse` COO `.npz`.

---

## 4. Architecture

```text
                 Sudoku board (9x9 int matrix)
                            |
                   per-cell 10-way one-hot        (0=empty, 1..9)
                            |
                  81 x 10 = 810 input features     (4x4: 16 x 5 = 80)
                            |
                   Poisson rate -> spike trains, T timesteps
                            |
                            v
            +--------------------------------+
            |   FLYWIRE-CONSTRAINED SNN      |
            |                                |
            |   input projection (dense)     |
            |   FlyWire recurrent neurons    |
            |     W_eff = W_trainable * A    |   A = fixed sparse adjacency
            |     (LIF dynamics, surrogate)  |
            |   output projection (dense)    |
            +---------------+----------------+
                            |
                     spike-rate readout
                            |
                     729 logits  (or 9 for Task A)
                            |
                     legal-action mask (illegal -> -inf)
                            |
                     argmax -> place digit
```

- **Adjacency `A`:** fixed sparse mask from FlyWire. `W = trainable`.
  Optional later: `W_eff = W_magnitude * A * sign` (neurotransmitter).
- **LIF discrete:** `v = beta*v + I; spike = surrogate(v > thresh); v = reset`.
- **Optimizer:** Adam, surrogate-gradient BPTT, AMP, grad checkpointing.

---

## 5. Autonomous Solve (constraint-propagation loop)

```text
board -> SNN scores every empty cell's 9 digits
      -> mask illegal (row/col/box constraints)
      -> pick globally most-confident legal (cell, digit)
      -> place it
      -> repeat
      -> solved / stuck
```
No RL needed for the spine. RL fine-tuning optional after this works.

---

## 6. Experiment Matrix

| Model | Connectivity | Neurons | Training |
|-------|-------------|--------:|----------|
| MLP | Dense | 256 | Supervised |
| SNN | Dense | 256 | Supervised |
| SNN | Dense | 1k | Supervised |
| FlyWire-SNN | FlyWire | 1k | Supervised |
| FlyWire-SNN | FlyWire | 5k | Supervised |
| FlyWire-SNN | FlyWire | 10k | Supervised |
| FlyWire-SNN | FlyWire | 20k | Supervised (VRAM ceiling) |
| FlyWire-SNN | FlyWire | 139k | Inference/eval only |

**Topology ablations (the science):**
- A. FlyWire connectivity
- B. Random connectivity
- C. Degree-preserving shuffled FlyWire
- D. Dense connectivity
- E. Feed-forward
- F. FlyWire topology + learned signs
- G. FlyWire topology + fixed biological signs

**Trainable regimes:**
- A. All weights trainable, connectivity mask fixed.
- B. Adjacency + signs fixed, magnitude trainable.
- C. FlyWire fully fixed; only input/output projections train ("does an
  untrained connectome provide useful structure?").

---

## 7. Phases (repo grows per phase)

| Phase | What | GPU? |
|-------|------|------|
| **0** | env: uv venv, torch (cu12) + Norse + deps, **verify CUDA on 2060** | check |
| **1** | Sudoku generator + solver + env + renderer, **4x4 first** then 9x9 | no |
| **2** | MLP baseline (validates data pipeline) | light |
| **3** | dense SNN (Norse LIF, AMP, T=10, checkpointing) | yes |
| **4** | FlyWire data: Codex dumps -> edges (count>=5) -> sparse COO; principled sampling | no |
| **5** | FlyWire SNN scaling 1k -> 10k -> 20k; VRAM benchmark each rung | yes |
| **6** | topology ablations (FlyWire vs random vs degree-matched vs dense) | yes |
| **7** | autonomous solver (constraint loop); RL fine-tuning optional | yes |

Gate: each phase must work before next. If MLP (Phase 2) fails, stop and debug —
do not touch FlyWire.

---

## 8. Metrics (not just loss)

- Move accuracy = correct / total actions
- Puzzle completion = % fully solved
- Invalid-action rate
- Steps to solve (mean actions)
- Generalization (train on generator A, test on generator B)
- **Network efficiency:** neurons used, synapses used, spikes/inference,
  latency, VRAM, energy proxy

---

## 9. Target Repo Layout (grown per phase, not built up front)

```text
flywire-sudoku/
  data/
    flywire/   neurons.parquet synapses.parquet neurotransmitters.parquet metadata.json
    sudoku/    train/ val/ test/
  connectome/  loader.py filter.py adjacency.py sampling.py statistics.py
  sudoku/      generator.py solver.py environment.py encoder.py renderer.py
  models/      mlp.py dense_snn.py flywire_snn.py
  training/    supervised.py reinforcement.py curriculum.py
  experiments/ baseline.py topology_ablation.py scaling.py rl.py
  evaluation/  accuracy.py sudoku_metrics.py visualization.py
  configs/     baseline.yaml flywire_1k.yaml flywire_5k.yaml flywire_10k.yaml
```

---

## 10. Explicitly NOT doing (initially)

- Full 139k brain for training
- Image-based Sudoku (adds a needless computer-vision problem)
- RL from scratch (supervised/constraint-loop first)
- Transformer + FlyWire simultaneously
- Reporting accuracy only
- Claiming "biological intelligence" (this is biological connectivity as an
  architectural constraint, not a reproduced brain)

---

## 11. Next Action

Phase 0. First deliverable = a `pyproject.toml` / dependency list for the user to
**eyeball before install** (uv). Pending approval, then `uv sync` + CUDA-on-2060
verification.

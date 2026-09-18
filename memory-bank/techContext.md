# Tech Context

## Hardware (fixed)

| Resource | Value |
|----------|-------|
| GPU | NVIDIA RTX 2060, 6 GB VRAM, compute 7.5 (Turing) |
| CPU | 12 cores |
| RAM | 15 GB (~8.7 GB free typical) |
| Python | 3.12.3 |
| Package manager | **uv** 0.11.17 |
| torch | not yet installed |

## VRAM / scale limits (float32 BPTT)

| N neurons | batch | T | approx act VRAM | verdict |
|-----------|-------|---|-----------------|---------|
| 10k | 64 | 20 | ~1.5 GB | fits |
| 50k | 64 | 20 | ~7.5 GB | won't fit |
| 139k | any | any | — | inference/eval only |

**Trainable ceiling:** ~10k–20k neurons, batch 32–64, T=10, AMP + grad checkpointing.

## Stack (locked)

- **Framework:** Norse (PyTorch-native LIF) + torch (CUDA 12.x)
- **Connectome source:** public Codex parquet dumps (no auth); CAVE later if needed
- **Data:** Polars lazy/streaming for multi-GB synapse parquet; persist `scipy.sparse` COO `.npz`
- **Optimizer:** Adam, surrogate-gradient BPTT
- **Precision:** AMP (Turing fp16)

## Install discipline

- Eyeball deps before install; pin Norse/torch versions at install time (Norse may lag torch)
- Next deliverable: `pyproject.toml` for approval → then `uv sync` + CUDA verify on 2060

## Constraints that drive design

- T is hidden VRAM multiplier under BPTT → start T=10 not 20
- Full connectome too big to train → subsample + scale study is the science
- Signs-free day 1; neurotransmitter ablation later

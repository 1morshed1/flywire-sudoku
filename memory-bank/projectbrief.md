# Project Brief — FlyWire-Sudoku

## Goal

FlyWire-connectome-constrained spiking neural network (SNN) trained for symbolic
reasoning on Sudoku. Task itself is conventional supervised / optional RL learning.
Scientific interest: recurrent connectivity constrained by real FlyWire Drosophila
connectome.

## Core research question

> How does biological connectivity constrain learning on an abstract symbolic
> reasoning task?

## In scope

- Real FlyWire connectivity as fixed sparse adjacency mask on SNN recurrent weights
- Sudoku as size-generic env (4×4 curriculum → 9×9)
- Supervised training + constraint-propagation autonomous solve loop
- Scaling study (1k–20k trainable neurons) + topology ablations
- Network-efficiency metrics (not accuracy alone)

## Out of scope (initially)

- Full 139k brain for training (inference/eval only)
- Image-based Sudoku / vision pipeline
- RL from scratch as primary spine
- Transformer + FlyWire simultaneously
- Reporting accuracy only
- Claims of “biological intelligence” (connectivity as architectural constraint only)

## Success criteria

- Phases gated: each must work before next; if MLP baseline fails, stop — no FlyWire
- Trainable FlyWire-SNN on RTX 2060 (6 GB): ~10k–20k neurons, batch 32–64, T=10
- Ablations isolate topology effect vs random / degree-matched / dense / feed-forward
- Autonomous solve via constraint-propagation loop (RL optional polish)

## Source of truth

`PLAN.md` at repo root. Memory bank tracks live state; plan holds locked decisions.

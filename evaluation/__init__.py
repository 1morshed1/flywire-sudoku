"""Evaluation: autonomous solving and solver metrics (PLAN.md Phase 7 / §22).

* :mod:`evaluation.solver_loop` — the constraint-propagation solve loop that turns a
  move-scoring model into a full-puzzle solver.
* :mod:`evaluation.metrics` — puzzle-completion rate, steps-to-solve, stuck rate over a
  set of puzzles, for comparing models (MLP / dense SNN / FlyWire SNN).
"""

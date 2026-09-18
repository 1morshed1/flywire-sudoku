"""Gymnasium-style Sudoku environment.

Supports the two task framings from PLAN.md §1:

* ``task="place"``  (Task B / autonomous): the agent chooses BOTH cell and digit.
  Action space is ``Discrete(num_cells * side)`` decoded as ``cell * side + (digit-1)``.
  Illegal actions can be masked via ``info["legal_mask"]``.
* ``task="next_digit"`` (Task A): a target empty cell is fixed each episode and the
  agent only chooses a digit. Action space is ``Discrete(side)`` -> digit ``a + 1``.

Rewards follow PLAN.md §14 and are fully configurable via ``reward``. The env is the
substrate for the constraint-propagation solve loop (Phase 7) and optional RL.

gymnasium is imported lazily so the rest of the Sudoku stack (solver/generator/tests)
works even in a minimal environment without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from . import core, encoder
from .core import SudokuSpec
from .generator import make_puzzle

try:  # gymnasium is optional for non-RL use
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_BASE = gym.Env
except Exception:  # noqa: BLE001  # pragma: no cover - optional dep may be absent
    gym = None
    spaces = None
    _GYM_BASE = object


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping constants (see PLAN.md §14)."""

    valid_move: float = 1.0
    invalid_move: float = -1.0
    solved: float = 100.0
    step_cost: float = -0.05


class SudokuEnv(_GYM_BASE):
    """A single-puzzle-per-episode Sudoku environment.

    Parameters
    ----------
    spec:
        Board geometry. Defaults to standard 9x9.
    task:
        ``"place"`` or ``"next_digit"`` (see module docstring).
    difficulty:
        Passed to the generator on each :meth:`reset`.
    reward:
        Reward shaping config.
    end_on_invalid:
        If True, an illegal action terminates the episode (a strict regime useful for
        supervised-style evaluation). If False, the board is left unchanged and the
        agent may try again (useful for RL exploration).
    max_steps:
        Truncation limit; defaults to ``num_cells`` (generous).
    """

    metadata: ClassVar[dict] = {"render_modes": ["ansi"]}

    def __init__(
        self,
        spec: SudokuSpec | None = None,
        *,
        task: str = "place",
        difficulty: str = "medium",
        reward: RewardConfig | None = None,
        end_on_invalid: bool = False,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        if task not in ("place", "next_digit"):
            raise ValueError(f"task must be 'place' or 'next_digit', got {task!r}")
        self.spec = spec or SudokuSpec.from_side(9)
        self.task = task
        self.difficulty = difficulty
        self.reward_cfg = reward or RewardConfig()
        self.end_on_invalid = end_on_invalid
        self.max_steps = max_steps or self.spec.num_cells
        self._rng = np.random.default_rng(seed)

        self._num_features = self.spec.num_cells * (self.spec.side + 1)
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self._num_features,), dtype=np.float32
            )
            if task == "place":
                self.action_space = spaces.Discrete(self.spec.num_cells * self.spec.side)
            else:
                self.action_space = spaces.Discrete(self.spec.side)

        # Episode state (populated by reset()).
        self._board: np.ndarray = self.spec.new_board()
        self._solution: np.ndarray = self.spec.new_board()
        self._target_cell: int | None = None
        self._steps = 0

    # ------------------------------------------------------------------ helpers

    def _obs(self) -> np.ndarray:
        """Current observation: flat one-hot encoding of the board."""
        return encoder.one_hot(self.spec, self._board, flatten=True)

    def _info(self) -> dict:
        """Auxiliary info: legal-action mask, board/solution copies, target cell."""
        info: dict = {
            "legal_mask": encoder.legal_action_mask(self.spec, self._board),
            "board": self._board.copy(),
            "solution": self._solution.copy(),
            "num_empty": int((self._board == 0).sum()),
        }
        if self.task == "next_digit":
            info["target_cell"] = self._target_cell
        return info

    def _pick_target_cell(self) -> int:
        empties = np.flatnonzero(self._board.reshape(-1) == 0)
        return int(self._rng.choice(empties))

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Generate a fresh puzzle and return ``(obs, info)``."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        difficulty = (options or {}).get("difficulty", self.difficulty)
        puzzle = make_puzzle(self.spec, self._rng, difficulty)
        self._board = puzzle.puzzle.copy()
        self._solution = puzzle.solution
        self._steps = 0
        self._target_cell = self._pick_target_cell() if self.task == "next_digit" else None
        return self._obs(), self._info()

    def step(self, action: int):
        """Apply ``action``; return ``(obs, reward, terminated, truncated, info)``."""
        self._steps += 1
        side = self.spec.side
        rc = self.reward_cfg

        if self.task == "place":
            cell, digit_idx = divmod(int(action), side)
            r, c = divmod(cell, side)
            digit = digit_idx + 1
        else:  # next_digit
            assert self._target_cell is not None
            r, c = divmod(self._target_cell, side)
            digit = int(action) + 1

        legal = self._board[r, c] == 0 and core.is_valid_placement(
            self.spec, self._board, r, c, digit
        )

        if not legal:
            reward = rc.invalid_move + rc.step_cost
            terminated = self.end_on_invalid
            truncated = self._steps >= self.max_steps
            return self._obs(), reward, terminated, truncated, self._info()

        # Apply the legal placement.
        self._board[r, c] = digit
        solved = core.is_solved(self.spec, self._board)
        reward = rc.valid_move + rc.step_cost + (rc.solved if solved else 0.0)
        terminated = solved
        # next_digit episodes are single-shot: one placement ends the episode.
        if self.task == "next_digit":
            terminated = True
        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        """Return an ASCII rendering of the current board (render_mode 'ansi')."""
        from .renderer import render_ascii

        return render_ascii(self.spec, self._board)

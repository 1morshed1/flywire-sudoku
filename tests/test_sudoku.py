"""Phase 1 correctness tests for the Sudoku stack.

The gate from PLAN.md Phase 1: "the conventional solver must solve thousands of
puzzles correctly." These tests exercise solver correctness, uniqueness guarantees,
generation, encoding round-trips, and the environment contract, on both 4x4 and 9x9
boards so the size-generic code is covered at two sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from sudoku import (
    RewardConfig,
    SudokuEnv,
    SudokuSpec,
    count_solutions,
    has_unique_solution,
    is_solved,
    is_valid,
    legal_action_mask,
    make_puzzle,
    one_hot,
    render_ascii,
    solve,
)
from sudoku.core import candidates, is_valid_placement
from sudoku.encoder import poisson_encode
from sudoku.generator import generate_full


@pytest.fixture(params=[4, 9], ids=["4x4", "9x9"])
def spec(request) -> SudokuSpec:
    return SudokuSpec.from_side(request.param)


def test_spec_geometry():
    s9 = SudokuSpec.from_side(9)
    assert s9.side == 9 and s9.num_cells == 81 and s9.num_digits == 9
    s4 = SudokuSpec.from_side(4)
    assert s4.side == 4 and s4.num_cells == 16
    # Peers: each cell has the same peer count; no self-peer.
    assert all(idx not in s9.peers[idx] for idx in range(81))
    assert len(set(s9.peers[0])) == 20  # 8 row + 8 col + 4 box unique


def test_generate_full_is_valid_solution(spec):
    rng = np.random.default_rng(0)
    for _ in range(20):
        grid = generate_full(spec, rng)
        assert is_solved(spec, grid)


def test_solver_solves_generated_puzzles(spec):
    """Solve many puzzles and confirm each solution matches the generator's."""
    rng = np.random.default_rng(1)
    n = 200 if spec.side == 4 else 50
    for _ in range(n):
        puz = make_puzzle(spec, rng, "easy")
        sol = solve(spec, puz.puzzle)
        assert sol is not None
        assert is_solved(spec, sol)
        # Unique solution => must equal the generator's solution.
        assert np.array_equal(sol, puz.solution)


def test_generated_puzzles_are_unique(spec):
    rng = np.random.default_rng(2)
    n = 100 if spec.side == 4 else 25
    for _ in range(n):
        puz = make_puzzle(spec, rng, "medium")
        assert has_unique_solution(spec, puz.puzzle)
        assert puz.num_clues >= spec.side


def test_solver_reports_no_solution_for_contradiction(spec):
    board = spec.new_board()
    # Force a contradiction: same digit twice in row 0.
    board[0, 0] = 1
    board[0, 1] = 1
    assert not is_valid(spec, board)
    assert solve(spec, board) is None
    assert count_solutions(spec, board) == 0


def test_candidates_and_placement_agree(spec):
    rng = np.random.default_rng(3)
    puz = make_puzzle(spec, rng, "hard")
    board = puz.puzzle
    for r in range(spec.side):
        for c in range(spec.side):
            if board[r, c] != 0:
                continue
            cands = candidates(spec, board, r, c)
            for d in range(1, spec.side + 1):
                assert (d in cands) == is_valid_placement(spec, board, r, c, d)


def test_one_hot_shape_and_content(spec):
    rng = np.random.default_rng(4)
    puz = make_puzzle(spec, rng, "easy")
    enc = one_hot(spec, puz.puzzle, flatten=False)
    assert enc.shape == (spec.num_cells, spec.side + 1)
    # Exactly one hot channel per cell.
    assert np.array_equal(enc.sum(axis=1), np.ones(spec.num_cells))
    flat = one_hot(spec, puz.puzzle, flatten=True)
    assert flat.shape == (spec.num_cells * (spec.side + 1),)


def test_poisson_encode_shapes_and_binary(spec):
    rng = np.random.default_rng(5)
    rates = one_hot(spec, make_puzzle(spec, rng, "easy").puzzle, flatten=True)
    T = 10
    spikes = poisson_encode(rates, T, rng)
    assert spikes.shape == (T, rates.shape[0])
    assert set(np.unique(spikes).tolist()).issubset({0.0, 1.0})
    # Zero-rate features never spike; rate-1 features always spike.
    assert spikes[:, rates == 0].sum() == 0
    assert np.all(spikes[:, rates == 1] == 1.0)


def test_legal_action_mask_matches_candidates(spec):
    rng = np.random.default_rng(6)
    board = make_puzzle(spec, rng, "medium").puzzle
    mask = legal_action_mask(spec, board)
    side = spec.side
    assert mask.shape == (spec.num_cells * side,)
    for r in range(side):
        for c in range(side):
            cell = r * side + c
            cands = set(candidates(spec, board, r, c))
            for d in range(1, side + 1):
                assert mask[cell * side + (d - 1)] == (d in cands)


def test_env_place_valid_and_invalid(spec):
    env = SudokuEnv(spec, task="place", difficulty="easy", reward=RewardConfig())
    obs, info = env.reset(seed=7)
    assert obs.shape == (spec.num_cells * (spec.side + 1),)
    mask = info["legal_mask"]
    legal_actions = np.flatnonzero(mask)
    assert len(legal_actions) > 0
    # A legal action yields positive reward and no invalid termination.
    obs, reward, _term, _trunc, info = env.step(int(legal_actions[0]))
    assert reward > 0
    # An illegal action (masked False) is penalized.
    illegal = int(np.flatnonzero(~info["legal_mask"])[0])
    _, reward2, _, _, _ = env.step(illegal)
    assert reward2 < 0


def test_env_solves_puzzle_via_solver(spec):
    """Drive the env with solver moves; it should reach a solved, terminated state."""
    env = SudokuEnv(spec, task="place", difficulty="easy")
    _, info = env.reset(seed=8)
    solution = info["solution"]
    side = spec.side
    terminated = False
    steps = 0
    while not terminated and steps < spec.num_cells + 5:
        board = info["board"]
        empties = np.argwhere(board == 0)
        r, c = empties[0]
        digit = int(solution[r, c])
        action = (r * side + c) * side + (digit - 1)
        _, _reward, terminated, _, info = env.step(int(action))
        steps += 1
    assert terminated
    assert is_solved(spec, info["board"])


def test_next_digit_task(spec):
    env = SudokuEnv(spec, task="next_digit", difficulty="easy")
    _, info = env.reset(seed=9)
    target = info["target_cell"]
    r, c = divmod(target, spec.side)
    correct = int(info["solution"][r, c])
    _, reward, term, _, _ = env.step(correct - 1)
    assert term  # single-shot
    assert reward > 0


def test_render_runs(spec):
    rng = np.random.default_rng(10)
    text = render_ascii(spec, make_puzzle(spec, rng, "easy").puzzle)
    assert isinstance(text, str) and len(text) > 0

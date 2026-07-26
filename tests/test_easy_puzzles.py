"""Unit + smoke tests for pure-logic modules (no LLM required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constraints import extract_constraints, hard_constraints
from src.library import (
    crop_to_bounding_box,
    gravity_drop,
    recolor,
    reflect_horizontal,
    rotate_90,
    rotate_180,
    tile_grid,
)
from src.loader import load_puzzle, load_puzzles_from_dir
from src.preprocess import describe_grid, find_objects
from src.sandbox import safe_execute
from src.tiebreak import mdl_score

DATA = ROOT / "data" / "arc-agi-2" / "training"


@pytest.fixture
def sample_puzzles():
    puzzles = load_puzzles_from_dir(DATA)
    assert len(puzzles) >= 5, "Expected at least 5 sample puzzles in data/"
    return puzzles


def test_loader_reads_train_and_test(sample_puzzles):
    p = sample_puzzles[0]
    assert p.id
    assert len(p.train_pairs) >= 2
    assert len(p.test_inputs) >= 1
    inp, out = p.train_pairs[0]
    assert isinstance(inp[0][0], int)
    assert isinstance(out[0][0], int)


def test_preprocess_objects_on_samples(sample_puzzles):
    for p in sample_puzzles[:5]:
        inp, _ = p.train_pairs[0]
        objs = find_objects(inp, connectivity=4)
        text = describe_grid(inp, include_raw=True)
        assert "Grid size:" in text or "size:" in text
        assert "Background color:" in text
        assert isinstance(objs, list)
        for obj in objs:
            assert 0 <= obj.color <= 9
            assert obj.size >= 1


def test_constraints_on_rotate_puzzle():
    # 6150a2bd is rotate-180: size preserved, colors preserved
    path = DATA / "6150a2bd.json"
    if not path.exists():
        pytest.skip("sample puzzle missing")
    p = load_puzzle(path)
    c = extract_constraints(p.train_pairs)
    assert c["grid_size"] == "preserved"
    hard = hard_constraints(c)
    assert "grid_size" in hard
    assert "not_constant" not in hard.values()


def test_sandbox_working_function():
    code = """
def solve(grid):
    return [row[::-1] for row in grid]
"""
    grid = [[1, 2], [3, 4]]
    result = safe_execute(code, grid, timeout_seconds=3)
    assert result.success
    assert result.output_grid == [[2, 1], [4, 3]]


def test_sandbox_syntax_error():
    code = "def solve(grid)\n    return grid"
    result = safe_execute(code, [[1]], timeout_seconds=3)
    assert not result.success
    assert result.error_message


def test_sandbox_timeout():
    code = """
def solve(grid):
    while True:
        pass
"""
    result = safe_execute(code, [[1]], timeout_seconds=1)
    assert not result.success
    assert "Timeout" in (result.error_message or "")


def test_sandbox_blocks_os_import():
    code = """
import os
def solve(grid):
    return grid
"""
    result = safe_execute(code, [[1]], timeout_seconds=3)
    assert not result.success
    assert "Blocked" in (result.error_message or "") or "import" in (result.error_message or "").lower()


def test_library_primitives():
    g = [[1, 2], [3, 4]]
    assert rotate_180(g) == [[4, 3], [2, 1]]
    assert reflect_horizontal(g) == [[2, 1], [4, 3]]
    assert rotate_90(g) == [[3, 1], [4, 2]]
    assert recolor([[1, 0], [1, 2]], 1, 5) == [[5, 0], [5, 2]]
    assert tile_grid([[1]], 2, 2) == [[1, 1], [1, 1]]
    dropped = gravity_drop([[0, 1], [0, 0]], "down")
    assert dropped[1][1] == 1
    cropped = crop_to_bounding_box([[0, 0, 0], [0, 7, 0], [0, 0, 0]])
    assert cropped == [[7]]


def test_mdl_prefers_simpler_code():
    simple = """
def solve(grid):
    return [row[::-1] for row in grid]
"""
    overfit = """
def solve(grid):
    if grid[0][0] == 3 and grid[1][2] == 5 and len(grid) == 7:
        return [[9, 8, 7], [6, 5, 4]]
    return grid
"""
    assert mdl_score(simple) < mdl_score(overfit)


def test_mock_e2e_rotate180():
    """Full pipeline wiring check with mock backend (no LLM)."""
    from src.llm_client import LLMClient
    from src.pipeline import solve_puzzle

    path = DATA / "6150a2bd.json"
    if not path.exists():
        pytest.skip("sample puzzle missing")
    puzzle = load_puzzle(path)
    client = LLMClient(backend="mock", model_name="mock")
    guesses = solve_puzzle(
        puzzle,
        client,
        time_budget_seconds=60,
        n_abstractions=1,
        max_self_debug_retries=1,
        n_judges=1,
        early_stop_after_cycles=2,
    )
    assert len(guesses) == 2
    assert puzzle.test_outputs
    assert guesses[0] == puzzle.test_outputs[0]

"""End-to-end tests for Phase 2: Test-Time Augmentation in Pipeline."""

import pytest
from src.loader import Puzzle
from src.llm_client import LLMClient
from src.pipeline import solve_puzzle, solve_with_augmentation, solve_all
from src.library import rotate_90, rotate_270, reflect_horizontal


def test_solve_with_augmentation_brute_force():
    """Test that solve_with_augmentation solves simple transforms instantly."""
    # Puzzle: 90-degree rotation
    train_pairs = [
        ([[1, 2], [3, 4]], [[3, 1], [4, 2]]),
        ([[5, 6], [7, 8]], [[7, 5], [8, 6]]),
    ]
    test_inputs = [[[9, 0], [1, 2]]]
    test_outputs = [[[1, 9], [2, 0]]]
    puzzle = Puzzle(
        id="test_rot90",
        train_pairs=train_pairs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
    )

    client = LLMClient(backend="mock", model_name="mock")
    results = solve_with_augmentation(puzzle, client, n_augmentations=4)
    assert len(results) == 2
    assert results[0] == test_outputs[0]


def test_solve_with_augmentation_directional_primitive():
    """Test a directional rule (e.g. gravity drop) under rotation."""
    raw_in = [
        [0, 1, 0],
        [0, 0, 2],
        [0, 0, 0],
    ]
    raw_out = [
        [1, 0, 0],
        [2, 0, 0],
        [0, 0, 0],
    ]
    
    puzzle = Puzzle(
        id="test_gravity_left",
        train_pairs=[(raw_in, raw_out)],
        test_inputs=[raw_in],
        test_outputs=[raw_out],
    )
    
    client = LLMClient(backend="mock", model_name="mock")
    results = solve_with_augmentation(puzzle, client, n_augmentations=8)
    assert len(results) == 2
    assert results[0] == raw_out


def test_solve_all_with_augmentation():
    """Test solve_all with use_augmentation=True."""
    train_pairs = [
        ([[1, 2], [3, 4]], [[3, 1], [4, 2]]),
    ]
    puzzle = Puzzle(
        id="puzzle_1",
        train_pairs=train_pairs,
        test_inputs=[[[1, 1], [2, 2]]],
        test_outputs=[[[2, 1], [2, 1]]],
    )
    client = LLMClient(backend="mock", model_name="mock")
    all_res = solve_all([puzzle], client, total_time_budget_seconds=10.0, use_augmentation=True, n_augmentations=2)
    assert "puzzle_1" in all_res
    assert len(all_res["puzzle_1"]) == 2
    assert all_res["puzzle_1"][0] == [[2, 1], [2, 1]]

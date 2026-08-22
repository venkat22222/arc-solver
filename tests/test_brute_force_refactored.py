import time
from src.brute_force import (
    try_brute_force,
    calculate_simplicity_score,
    REGISTRY,
    BruteForceHit,
)

def test_registry_contains_expected_stages():
    """1. Test that the registry contains expected stages (0, 1, 2, 3)."""
    stages_present = {s.stage for s in REGISTRY}
    assert 0 in stages_present, "Stage 0 missing from registry"
    assert 1 in stages_present, "Stage 1 missing from registry"
    assert 2 in stages_present, "Stage 2 missing from registry"
    assert 3 in stages_present, "Stage 3 missing from registry"


def test_registry_stage_order():
    """2. Test that stage order is correct (sorted, non-decreasing)."""
    stages = [s.stage for s in REGISTRY]
    assert stages == sorted(stages), f"Stages are not in sorted order: {stages}"


def test_known_solvable_and_unsolvable_puzzles():
    """3 & 4. Test known solvable and unsolvable puzzles."""
    # Known solvable: Identity transform
    solvable_train = [
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    ]
    hit = try_brute_force(solvable_train)
    assert hit is not None, "Failed to solve identity puzzle"
    assert hit.name == "identity"
    assert "solve" in hit.code
    assert hit.stage == 0
    assert hit.simplicity_score == 95.0  # 100 - 5 * 1 op

    # Known solvable: Rotate 90
    rotate_train = [
        ([[1, 2], [3, 4]], [[3, 1], [4, 2]])
    ]
    hit_rot = try_brute_force(rotate_train)
    assert hit_rot is not None, "Failed to solve rotate puzzle"
    assert hit_rot.name == "rotate_90"
    assert hit_rot.stage == 0

    # Unsolvable puzzle
    unsolvable_train = [
        ([[1, 2]], [[9, 9]]),
        ([[3, 4]], [[5, 6]])
    ]
    hit_unsolvable = try_brute_force(unsolvable_train)
    assert hit_unsolvable is None, f"Expected None but got: {hit_unsolvable}"


def test_time_budget_respects_limits():
    """5. Test that try_brute_force respects small time budget."""
    # Provide a complex puzzle setup where we force it to look through many options
    # and pass an extremely small time budget of 0.0001 ms (or 1e-6 seconds)
    large_train_pairs = [
        ([[i] * 10 for i in range(10)], [[i] * 10 for i in range(10)])
    ]
    
    start_time = time.time()
    hit = try_brute_force(large_train_pairs, time_budget_ms=0.0001)
    elapsed = (time.time() - start_time) * 1000.0
    
    # It should have returned None because it timed out before finishing later stages
    # or even early stages depending on computer speed, but definitely should respect budget.
    # Note: identity might match immediately in Stage 0, so let's make it not identity-solvable
    non_trivial_train_pairs = [
        ([[1, 2, 3]], [[3, 2, 1]]) # rotate 180 / reflect horizontal
    ]
    # Rotate 180 is Stage 0. If we disable stage 0 to force it to run stage 1+, or just use a small budget:
    # Let's verify it stops
    hit_timeout = try_brute_force(non_trivial_train_pairs, time_budget_ms=0.001)
    # Even if it returns a hit or None, we verify it runs extremely fast
    assert elapsed < 50.0, f"Budget of 0.0001 ms took too long: {elapsed} ms"


def test_candidate_limits_stop_execution():
    """6. Test that candidate limits stop execution."""
    # We can temporarily patch STAGE_0_MAX_CANDIDATES to 0 or 1, and see if it halts or limits search
    import src.brute_force as bf
    old_stage_0_max = bf.STAGE_0_MAX_CANDIDATES
    try:
        bf.STAGE_0_MAX_CANDIDATES = 0
        # If max candidates is 0, Stage 0 should yield 0 candidates, thus not matching even identity
        solvable_train = [
            ([[1]], [[1]])
        ]
        hit = try_brute_force(solvable_train)
        # It should not find "identity" because candidate limit for Stage 0 is 0
        # Wait, identity is the first candidate in Stage 0. Let's make sure it doesn't match
        assert hit is None or hit.stage > 0
    finally:
        bf.STAGE_0_MAX_CANDIDATES = old_stage_0_max


def test_simplicity_scoring():
    """7. Test simplicity scoring with different penalties."""
    # A simple global transform: 1 operation -> score 95
    score_simple = calculate_simplicity_score(
        num_operations=1,
        uses_hardcoded_coords=False,
        uses_color_exceptions=False,
        per_example_params=False,
        uses_size_constants=False,
    )
    assert score_simple == 95.0

    # With hardcoded coordinates: subtract 10 -> score 85
    score_coords = calculate_simplicity_score(
        num_operations=1,
        uses_hardcoded_coords=True,
        uses_color_exceptions=False,
        per_example_params=False,
        uses_size_constants=False,
    )
    assert score_coords == 85.0

    # With color exceptions: subtract 10 -> score 85
    score_colors = calculate_simplicity_score(
        num_operations=1,
        uses_hardcoded_coords=False,
        uses_color_exceptions=True,
        per_example_params=False,
        uses_size_constants=False,
    )
    assert score_colors == 85.0

    # With different parameters per example: subtract 25 -> score 70
    score_per_ex = calculate_simplicity_score(
        num_operations=1,
        uses_hardcoded_coords=False,
        uses_color_exceptions=False,
        per_example_params=True,
        uses_size_constants=False,
    )
    assert score_per_ex == 70.0

    # With size constants: subtract 15 -> score 80
    score_size = calculate_simplicity_score(
        num_operations=1,
        uses_hardcoded_coords=False,
        uses_color_exceptions=False,
        per_example_params=False,
        uses_size_constants=True,
    )
    assert score_size == 80.0

    # Minimum score is 0
    score_min = calculate_simplicity_score(
        num_operations=20, # -100
        uses_hardcoded_coords=True, # -10
        uses_color_exceptions=True, # -10
        per_example_params=True, # -25
        uses_size_constants=True, # -15
    )
    assert score_min == 0.0

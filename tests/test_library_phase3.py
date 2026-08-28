"""Tests for Phase 3 DSL primitives in library and brute_force."""

from src.library import (
    scale_grid,
    downscale_grid,
    complete_symmetry,
    extract_color_mask,
    outline_objects,
    extract_interior,
)
from src.brute_force import try_brute_force


def test_scale_grid_2x2():
    grid = [[1, 2], [3, 4]]
    scaled = scale_grid(grid, 2, 2)
    assert scaled == [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]


def test_downscale_grid_2x2():
    grid = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]
    down = downscale_grid(grid, 2, 2)
    assert down == [[1, 2], [3, 4]]


def test_complete_symmetry_vertical():
    # Left half drawn, right half empty (0)
    grid = [
        [1, 0, 0, 0],
        [2, 3, 0, 0],
    ]
    sym = complete_symmetry(grid, "vertical")
    assert sym == [
        [1, 0, 0, 1],
        [2, 3, 3, 2],
    ]


def test_complete_symmetry_horizontal():
    # Top half drawn, bottom half empty (0)
    grid = [
        [1, 2],
        [0, 0],
    ]
    sym = complete_symmetry(grid, "horizontal")
    assert sym == [
        [1, 2],
        [1, 2],
    ]


def test_extract_color_mask():
    grid = [
        [1, 2, 1],
        [3, 1, 4],
    ]
    mask = extract_color_mask(grid, 1, crop=False)
    assert mask == [
        [1, 0, 1],
        [0, 1, 0],
    ]


def test_extract_interior():
    grid = [
        [9, 9, 9, 9],
        [9, 1, 2, 9],
        [9, 3, 4, 9],
        [9, 9, 9, 9],
    ]
    interior = extract_interior(grid)
    assert interior == [
        [1, 2],
        [3, 4],
    ]


def test_outline_objects():
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    outlined = outline_objects(grid, 2)
    # The center remains 1, surrounding non-diagonal neighbors get colored 2
    assert outlined[1][1] == 1
    assert outlined[0][1] == 2
    assert outlined[2][1] == 2
    assert outlined[1][0] == 2
    assert outlined[1][2] == 2


def test_brute_force_solves_scale_puzzle():
    train_pairs = [
        ([[1, 2], [3, 4]], [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),
    ]
    hit = try_brute_force(train_pairs)
    assert hit is not None
    assert "scale_grid" in hit.name
    assert "scale_grid(grid, 2, 2)" in hit.code


def test_brute_force_solves_interior_puzzle():
    train_pairs = [
        (
            [[8, 8, 8, 8], [8, 1, 8, 8], [8, 8, 8, 8], [8, 8, 8, 8]],
            [[1, 8], [8, 8]],
        )
    ]
    hit = try_brute_force(train_pairs)
    assert hit is not None
    assert "extract_interior" in hit.name

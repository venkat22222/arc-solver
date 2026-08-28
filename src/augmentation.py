"""Test-time augmentation: solve each puzzle under 8 geometric transforms.

For each of the 8 isometries of the square (identity, 3 rotations, 2 flips,
2 transposes), transform all train pairs and test inputs, solve the
transformed puzzle, then un-transform the output and vote.

This is a *free* accuracy boost used by virtually all top-100 ARC competitors.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

Grid = List[List[int]]
Transform = Callable[[Grid], Grid]
TrainPair = Tuple[Grid, Grid]


# ── 8 isometries and their inverses ──────────────────────────────────────

def _identity(g: Grid) -> Grid:
    return [row[:] for row in g]


def _rot90(g: Grid) -> Grid:
    """Rotate 90° clockwise."""
    h, w = len(g), len(g[0])
    return [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]


def _rot180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]


def _rot270(g: Grid) -> Grid:
    """Rotate 270° clockwise (= 90° counter-clockwise)."""
    h, w = len(g), len(g[0])
    return [[g[r][w - 1 - c] for r in range(h)] for c in range(w)]


def _flip_h(g: Grid) -> Grid:
    """Flip horizontally (left-right mirror)."""
    return [row[::-1] for row in g]


def _flip_v(g: Grid) -> Grid:
    """Flip vertically (top-bottom mirror)."""
    return g[::-1]


def _transpose(g: Grid) -> Grid:
    h, w = len(g), len(g[0])
    return [[g[r][c] for r in range(h)] for c in range(w)]


def _anti_transpose(g: Grid) -> Grid:
    """Transpose along the anti-diagonal."""
    h, w = len(g), len(g[0])
    return [[g[h - 1 - r][w - 1 - c] for r in range(h)] for c in range(w)]


# Each entry: (name, forward_transform, inverse_transform)
# The inverse undoes the forward so we can un-transform the output.
AUGMENTATIONS: List[Tuple[str, Transform, Transform]] = [
    ("identity",       _identity,       _identity),
    ("rot90",          _rot90,          _rot270),      # inv(rot90) = rot270
    ("rot180",         _rot180,         _rot180),      # inv(rot180) = rot180
    ("rot270",         _rot270,         _rot90),       # inv(rot270) = rot90
    ("flip_h",         _flip_h,         _flip_h),      # self-inverse
    ("flip_v",         _flip_v,         _flip_v),      # self-inverse
    ("transpose",      _transpose,      _transpose),   # self-inverse
    ("anti_transpose", _anti_transpose, _anti_transpose),  # self-inverse
]


def augment_train_pairs(
    train_pairs: Sequence[TrainPair], fwd: Transform
) -> List[TrainPair]:
    """Apply forward transform to both input and output of every train pair."""
    return [(fwd(inp), fwd(out)) for inp, out in train_pairs]


def augment_test_inputs(
    test_inputs: Sequence[Grid], fwd: Transform
) -> List[Grid]:
    """Apply forward transform to each test input."""
    return [fwd(t) for t in test_inputs]

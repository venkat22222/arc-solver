"""Pre-pipeline brute-force: try library primitives (and simple pairs) on train.

If any candidate matches every training pair, return a ready-to-submit solve()
string so the LLM path can be skipped entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .library import (
    crop_to_bounding_box,
    gravity_drop,
    reflect_horizontal,
    reflect_vertical,
    rotate_180,
    rotate_270,
    rotate_90,
    tile_grid,
    transpose,
    recolor,
    fill_enclosed_regions,
)
from .preprocess import grid_shape

Grid = List[List[int]]
TrainPair = Tuple[Grid, Grid]
Transform = Callable[[Grid], Grid]


@dataclass(frozen=True)
class BruteForceHit:
    name: str
    code: str
    n_ops: int
    stage: int = 0
    candidate_name: str = ""
    simplicity_score: float = 0.0
    params: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _grids_equal(a: Grid, b: Grid) -> bool:
    return a == b


def _matches_all(fn: Transform, train_pairs: Sequence[TrainPair]) -> bool:
    try:
        for inp, expected in train_pairs:
            if not _grids_equal(fn(inp), expected):
                return False
        return True
    except Exception:
        return False


def _colors_in_pairs(train_pairs: Sequence[TrainPair]) -> List[int]:
    colors: set[int] = set()
    for inp, out in train_pairs:
        for row in inp:
            colors.update(row)
        for row in out:
            colors.update(row)
    return sorted(colors)


def _infer_tile_factors(train_pairs: Sequence[TrainPair]) -> List[Tuple[int, int]]:
    factors: set[Tuple[int, int]] = set()
    for inp, out in train_pairs:
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)
        if ih > 0 and iw > 0 and oh % ih == 0 and ow % iw == 0:
            factors.add((oh // ih, ow // iw))
    # Common small tilings even if inference fails on a weird pair
    for nr, nc in ((2, 2), (3, 3), (2, 1), (1, 2), (3, 1), (1, 3)):
        factors.add((nr, nc))
    return sorted(factors)


def _mirror_tile_2x2(grid: Grid) -> Grid:
    """Common ARC pattern: [G | flipH(G)] stacked over its vertical flip."""
    top = [row + row[::-1] for row in grid]
    return top + top[::-1]


def _hstack_reflect_h(grid: Grid) -> Grid:
    return [row + row[::-1] for row in grid]


def _vstack_reflect_v(grid: Grid) -> Grid:
    return grid + grid[::-1]


def _unary_candidates() -> List[Tuple[str, Transform, str]]:
    """(name, fn, code_body_expr) — expr plugged into return <expr>."""
    return [
        ("identity", _identity, "grid"),
        ("rotate_90", rotate_90, "rotate_90(grid)"),
        ("rotate_180", rotate_180, "rotate_180(grid)"),
        ("rotate_270", rotate_270, "rotate_270(grid)"),
        ("reflect_horizontal", reflect_horizontal, "reflect_horizontal(grid)"),
        ("reflect_vertical", reflect_vertical, "reflect_vertical(grid)"),
        ("transpose", transpose, "transpose(grid)"),
        ("crop_to_bounding_box", crop_to_bounding_box, "crop_to_bounding_box(grid)"),
    ]


def _code_return(expr: str) -> str:
    return f"def solve(grid):\n    return {expr}"


def _code_mirror_tile_2x2() -> str:
    return (
        "def solve(grid):\n"
        "    top = [row + row[::-1] for row in grid]\n"
        "    return top + top[::-1]"
    )


def try_brute_force(train_pairs: Sequence[TrainPair]) -> Optional[BruteForceHit]:
    """Return the first primitive / simple-pair program that fits all train pairs."""
    if not train_pairs:
        return None

    # STAGE 0: Unary transforms
    unaries = _unary_candidates()
    for name, fn, expr in unaries:
        if _matches_all(fn, train_pairs):
            return BruteForceHit(name=name, code=_code_return(expr), n_ops=1, stage=0, candidate_name=name)

    # STAGE 1: Simple pairs of unaries
    for name_f, fn_f, expr_f in unaries:
        if name_f == "identity": continue
        for name_g, fn_g, expr_g in unaries:
            if name_g == "identity" or name_f == name_g: continue
            composed: Transform = lambda g, a=fn_f, b=fn_g: b(a(g))
            if _matches_all(composed, train_pairs):
                inner = expr_f
                outer = expr_g.replace("(grid)", f"({inner})", 1)
                name = f"{name_g}_of_{name_f}"
                return BruteForceHit(name=name, code=_code_return(outer), n_ops=2, stage=1, candidate_name=name)

    # STAGE 2: Parameter-sweep-style candidates
    for direction in ("up", "down", "left", "right"):
        fn = lambda g, d=direction: gravity_drop(g, d)  # noqa: E731
        if _matches_all(fn, train_pairs):
            name = f"gravity_drop_{direction}"
            return BruteForceHit(name=name, code=_code_return(f'gravity_drop(grid, "{direction}")'), n_ops=1, stage=2, candidate_name="gravity_drop")

    for nr, nc in _infer_tile_factors(train_pairs):
        fn = lambda g, a=nr, b=nc: tile_grid(g, a, b)  # noqa: E731
        if _matches_all(fn, train_pairs):
            name = f"tile_grid_{nr}x{nc}"
            return BruteForceHit(name=name, code=_code_return(f"tile_grid(grid, {nr}, {nc})"), n_ops=1, stage=2, candidate_name="tile_grid")

    colors = _colors_in_pairs(train_pairs)
    for src in colors:
        for dst in colors:
            if src == dst: continue
            fn = lambda g, a=src, b=dst: recolor(g, a, b)  # noqa: E731
            if _matches_all(fn, train_pairs):
                name = f"recolor_{src}_to_{dst}"
                return BruteForceHit(name=name, code=_code_return(f"recolor(grid, {src}, {dst})"), n_ops=1, stage=2, candidate_name="recolor")

    for fill in colors:
        fn = lambda g, c=fill: fill_enclosed_regions(g, c)  # noqa: E731
        if _matches_all(fn, train_pairs):
            name = f"fill_enclosed_{fill}"
            return BruteForceHit(name=name, code=_code_return(f"fill_enclosed_regions(grid, {fill})"), n_ops=1, stage=2, candidate_name="fill_enclosed")

    # STAGE 3: Complex / tiling candidates
    if _matches_all(_mirror_tile_2x2, train_pairs):
        return BruteForceHit(name="mirror_tile_2x2", code=_code_mirror_tile_2x2(), n_ops=2, stage=3, candidate_name="mirror_tile_2x2")

    specials: List[Tuple[str, Transform, str]] = [
        ("hstack_reflect_horizontal", _hstack_reflect_h, "[row + row[::-1] for row in grid]"),
        ("vstack_reflect_vertical", _vstack_reflect_v, "grid + grid[::-1]"),
        ("tile_after_reflect_horizontal_2x2", lambda g: tile_grid(reflect_horizontal(g), 2, 2), "tile_grid(reflect_horizontal(grid), 2, 2)"),
        ("tile_after_reflect_vertical_2x2", lambda g: tile_grid(reflect_vertical(g), 2, 2), "tile_grid(reflect_vertical(grid), 2, 2)"),
        ("tile_after_rotate_180_2x2", lambda g: tile_grid(rotate_180(g), 2, 2), "tile_grid(rotate_180(grid), 2, 2)"),
    ]
    for name, fn, expr in specials:
        if _matches_all(fn, train_pairs):
            return BruteForceHit(name=name, code=_code_return(expr), n_ops=2, stage=3, candidate_name=name)

    # tile(f(grid)) for remaining unaries
    for name_f, fn_f, expr_f in unaries:
        if name_f == "identity": continue
        for nr, nc in _infer_tile_factors(train_pairs):
            composed = lambda g, a=fn_f, x=nr, y=nc: tile_grid(a(g), x, y)
            if _matches_all(composed, train_pairs):
                name = f"tile_{nr}x{nc}_of_{name_f}"
                return BruteForceHit(name=name, code=_code_return(f"tile_grid({expr_f}, {nr}, {nc})"), n_ops=2, stage=3, candidate_name="tile_unaries")

    return None

__all__ = ["BruteForceHit", "try_brute_force"]

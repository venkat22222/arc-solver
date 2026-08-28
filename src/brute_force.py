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
    scale_grid,
    downscale_grid,
    complete_symmetry,
    extract_color_mask,
    outline_objects,
    extract_interior,
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
        ("extract_interior", extract_interior, "extract_interior(grid)"),
        ("complete_symmetry_vertical", lambda g: complete_symmetry(g, "vertical"), 'complete_symmetry(grid, "vertical")'),
        ("complete_symmetry_horizontal", lambda g: complete_symmetry(g, "horizontal"), 'complete_symmetry(grid, "horizontal")'),
        ("complete_symmetry_diagonal", lambda g: complete_symmetry(g, "diagonal"), 'complete_symmetry(grid, "diagonal")'),
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

    # STAGE 1: Cross-Primitive Compositions

    colors = _colors_in_pairs(train_pairs)
    directions = ("up", "down", "left", "right")

    # 10. Anti-transpose (special named case)
    comp_at = lambda g: reflect_horizontal(transpose(g))
    if _matches_all(comp_at, train_pairs):
        return BruteForceHit(name="anti_transpose", code=_code_return("reflect_horizontal(transpose(grid))"), n_ops=2, stage=1, candidate_name="anti_transpose")

    # 1. Crop + Geometric & 3. Recolor + Crop & 9. Gravity Drop + Crop
    # Note: crop_to_bounding_box is already in `unaries`, so loops over `unaries` naturally include crop!
    
    # 2. Recolor + Geometric (including Crop) & Geometric + Recolor
    for name_u, fn_u, expr_u in unaries:
        if name_u == "identity": continue
        for src in colors:
            for dst in colors:
                if src == dst: continue
                # Recolor + Geometric: fn_u(recolor(grid, src, dst))
                comp1 = lambda g, u=fn_u, s=src, d=dst: u(recolor(g, s, d))
                if _matches_all(comp1, train_pairs):
                    expr = expr_u.replace("(grid)", f"(recolor(grid, {src}, {dst}))", 1)
                    return BruteForceHit(name=f"{name_u}_of_recolor_{src}_{dst}", code=_code_return(expr), n_ops=2, stage=1, candidate_name=f"{name_u}_of_recolor")
                
                # Geometric + Recolor: recolor(fn_u(grid), src, dst)
                comp2 = lambda g, u=fn_u, s=src, d=dst: recolor(u(g), s, d)
                if _matches_all(comp2, train_pairs):
                    expr = f"recolor({expr_u}, {src}, {dst})"
                    return BruteForceHit(name=f"recolor_{src}_{dst}_of_{name_u}", code=_code_return(expr), n_ops=2, stage=1, candidate_name=f"recolor_of_{name_u}")

    # 4. Double Recolor
    for src1 in colors:
        for dst1 in colors:
            if src1 == dst1: continue
            for src2 in colors:
                for dst2 in colors:
                    if src2 == dst2: continue
                    if src1 == src2 and dst1 == dst2: continue
                    comp_rr = lambda g, s1=src1, d1=dst1, s2=src2, d2=dst2: recolor(recolor(g, s1, d1), s2, d2)
                    if _matches_all(comp_rr, train_pairs):
                        expr = f"recolor(recolor(grid, {src1}, {dst1}), {src2}, {dst2})"
                        return BruteForceHit(name=f"double_recolor_{src1}_{dst1}_{src2}_{dst2}", code=_code_return(expr), n_ops=2, stage=1, candidate_name="double_recolor")

    # 5. Gravity Drop + Geometric (including Crop) & Geometric + Gravity Drop
    for name_u, fn_u, expr_u in unaries:
        if name_u == "identity": continue
        for d in directions:
            # Gravity + Geometric: fn_u(gravity_drop(grid, d))
            comp1 = lambda g, u=fn_u, dr=d: u(gravity_drop(g, dr))
            if _matches_all(comp1, train_pairs):
                expr = expr_u.replace("(grid)", f'(gravity_drop(grid, "{d}"))', 1)
                return BruteForceHit(name=f"{name_u}_of_gravity_{d}", code=_code_return(expr), n_ops=2, stage=1, candidate_name=f"{name_u}_of_gravity")
            
            # Geometric + Gravity: gravity_drop(fn_u(grid), d)
            comp2 = lambda g, u=fn_u, dr=d: gravity_drop(u(g), dr)
            if _matches_all(comp2, train_pairs):
                expr = f'gravity_drop({expr_u}, "{d}")'
                return BruteForceHit(name=f"gravity_{d}_of_{name_u}", code=_code_return(expr), n_ops=2, stage=1, candidate_name=f"gravity_of_{name_u}")

    # 6. Gravity Drop + Recolor & Recolor + Gravity Drop
    for d in directions:
        for src in colors:
            for dst in colors:
                if src == dst: continue
                # Recolor + Gravity: gravity_drop(recolor(grid, src, dst), d)
                comp1 = lambda g, s=src, ds=dst, dr=d: gravity_drop(recolor(g, s, ds), dr)
                if _matches_all(comp1, train_pairs):
                    expr = f'gravity_drop(recolor(grid, {src}, {dst}), "{d}")'
                    return BruteForceHit(name=f"gravity_{d}_of_recolor_{src}_{dst}", code=_code_return(expr), n_ops=2, stage=1, candidate_name="gravity_of_recolor")
                
                # Gravity + Recolor: recolor(gravity_drop(grid, d), src, dst)
                comp2 = lambda g, s=src, ds=dst, dr=d: recolor(gravity_drop(g, dr), s, ds)
                if _matches_all(comp2, train_pairs):
                    expr = f'recolor(gravity_drop(grid, "{d}"), {src}, {dst})'
                    return BruteForceHit(name=f"recolor_{src}_{dst}_of_gravity_{d}", code=_code_return(expr), n_ops=2, stage=1, candidate_name="recolor_of_gravity")

    # 7. Fill Enclosed + Geometric (including Crop) & Geometric + Fill Enclosed
    for name_u, fn_u, expr_u in unaries:
        if name_u == "identity": continue
        for c in colors:
            # Fill + Geometric: fn_u(fill_enclosed_regions(grid, c))
            comp1 = lambda g, u=fn_u, fill=c: u(fill_enclosed_regions(g, fill))
            if _matches_all(comp1, train_pairs):
                expr = expr_u.replace("(grid)", f"(fill_enclosed_regions(grid, {c}))", 1)
                return BruteForceHit(name=f"{name_u}_of_fill_{c}", code=_code_return(expr), n_ops=2, stage=1, candidate_name=f"{name_u}_of_fill")
            
            # Geometric + Fill: fill_enclosed_regions(fn_u(grid), c)
            comp2 = lambda g, u=fn_u, fill=c: fill_enclosed_regions(u(g), fill)
            if _matches_all(comp2, train_pairs):
                expr = f"fill_enclosed_regions({expr_u}, {c})"
                return BruteForceHit(name=f"fill_{c}_of_{name_u}", code=_code_return(expr), n_ops=2, stage=1, candidate_name=f"fill_of_{name_u}")

    # 8. Fill Enclosed + Recolor & Recolor + Fill Enclosed
    for fill in colors:
        for src in colors:
            for dst in colors:
                if src == dst: continue
                # Recolor + Fill: fill_enclosed_regions(recolor(grid, src, dst), fill)
                comp1 = lambda g, s=src, d=dst, f=fill: fill_enclosed_regions(recolor(g, s, d), f)
                if _matches_all(comp1, train_pairs):
                    expr = f"fill_enclosed_regions(recolor(grid, {src}, {dst}), {fill})"
                    return BruteForceHit(name=f"fill_{fill}_of_recolor_{src}_{dst}", code=_code_return(expr), n_ops=2, stage=1, candidate_name="fill_of_recolor")
                
                # Fill + Recolor: recolor(fill_enclosed_regions(grid, fill), src, dst)
                comp2 = lambda g, s=src, d=dst, f=fill: recolor(fill_enclosed_regions(g, f), s, d)
                if _matches_all(comp2, train_pairs):
                    expr = f"recolor(fill_enclosed_regions(grid, {fill}), {src}, {dst})"
                    return BruteForceHit(name=f"recolor_{src}_{dst}_of_fill_{fill}", code=_code_return(expr), n_ops=2, stage=1, candidate_name="recolor_of_fill")

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

    # Scale grid sweeps
    for fy, fx in ((2, 2), (3, 3), (4, 4), (2, 1), (1, 2), (3, 1), (1, 3)):
        fn = lambda g, a=fy, b=fx: scale_grid(g, a, b)  # noqa: E731
        if _matches_all(fn, train_pairs):
            name = f"scale_grid_{fy}x{fx}"
            return BruteForceHit(name=name, code=_code_return(f"scale_grid(grid, {fy}, {fx})"), n_ops=1, stage=2, candidate_name="scale_grid")

    # Downscale grid sweeps
    for fy, fx in ((2, 2), (3, 3), (2, 1), (1, 2)):
        fn = lambda g, a=fy, b=fx: downscale_grid(g, a, b)  # noqa: E731
        if _matches_all(fn, train_pairs):
            name = f"downscale_grid_{fy}x{fx}"
            return BruteForceHit(name=name, code=_code_return(f"downscale_grid(grid, {fy}, {fx})"), n_ops=1, stage=2, candidate_name="downscale_grid")

    # Extract color mask sweeps
    for c in colors:
        fn_uncropped = lambda g, col=c: extract_color_mask(g, col, crop=False)  # noqa: E731
        if _matches_all(fn_uncropped, train_pairs):
            name = f"extract_color_mask_{c}"
            return BruteForceHit(name=name, code=_code_return(f"extract_color_mask(grid, {c})"), n_ops=1, stage=2, candidate_name="extract_color_mask")

        fn_cropped = lambda g, col=c: extract_color_mask(g, col, crop=True)  # noqa: E731
        if _matches_all(fn_cropped, train_pairs):
            name = f"extract_color_mask_{c}_cropped"
            return BruteForceHit(name=name, code=_code_return(f"extract_color_mask(grid, {c}, crop=True)"), n_ops=1, stage=2, candidate_name="extract_color_mask_cropped")

    # Outline objects sweeps
    for c in colors:
        fn = lambda g, col=c: outline_objects(g, col)  # noqa: E731
        if _matches_all(fn, train_pairs):
            name = f"outline_objects_{c}"
            return BruteForceHit(name=name, code=_code_return(f"outline_objects(grid, {c})"), n_ops=1, stage=2, candidate_name="outline_objects")

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

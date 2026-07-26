"""DreamCoder/STITCH-style primitive library for ARC grid transforms.

Generated code is encouraged to compose these helpers rather than
re-implementing common operations from scratch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

from .preprocess import ObjectInfo, background_color, find_objects, grid_shape

Grid = List[List[int]]
Direction = Literal["up", "down", "left", "right"]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def rotate_90(grid: Grid) -> Grid:
    """Rotate 90° clockwise."""
    h, w = grid_shape(grid)
    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]


def rotate_180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: Grid) -> Grid:
    """Rotate 90° counter-clockwise (== 270° clockwise)."""
    return rotate_90(rotate_90(rotate_90(grid)))


def reflect_horizontal(grid: Grid) -> Grid:
    """Mirror left↔right."""
    return [row[::-1] for row in grid]


def reflect_vertical(grid: Grid) -> Grid:
    """Mirror top↔bottom."""
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    h, w = grid_shape(grid)
    return [[grid[r][c] for r in range(h)] for c in range(w)]


# ---------------------------------------------------------------------------
# Color / objects
# ---------------------------------------------------------------------------

def recolor(grid: Grid, from_color: int, to_color: int) -> Grid:
    return [[to_color if c == from_color else c for c in row] for row in grid]


def count_color(grid: Grid, color: int) -> int:
    return sum(1 for row in grid for c in row if c == color)


def count_objects(grid: Grid, connectivity: int = 4) -> int:
    return len(find_objects(grid, connectivity=connectivity))  # type: ignore[arg-type]


def find_largest_object(grid: Grid, connectivity: int = 4) -> Optional[ObjectInfo]:
    objs = find_objects(grid, connectivity=connectivity)  # type: ignore[arg-type]
    if not objs:
        return None
    return max(objs, key=lambda o: (o.size, -o.id))


def find_smallest_object(grid: Grid, connectivity: int = 4) -> Optional[ObjectInfo]:
    objs = find_objects(grid, connectivity=connectivity)  # type: ignore[arg-type]
    if not objs:
        return None
    return min(objs, key=lambda o: (o.size, o.id))


def crop_to_bounding_box(grid: Grid, connectivity: int = 4) -> Grid:
    """Crop to the bounding box of all non-background cells."""
    bg = background_color(grid)
    h, w = grid_shape(grid)
    cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != bg]
    if not cells:
        return [row[:] for row in grid]
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return [row[c0 : c1 + 1] for row in grid[r0 : r1 + 1]]


# ---------------------------------------------------------------------------
# Composition / tiling
# ---------------------------------------------------------------------------

def tile_grid(grid: Grid, n_rows: int, n_cols: int) -> Grid:
    """Repeat the grid into an n_rows × n_cols block tiling."""
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be >= 1")
    return [row * n_cols for _ in range(n_rows) for row in grid]


def overlay(grid_a: Grid, grid_b: Grid, transparent: int = 0) -> Grid:
    """Overlay B onto A; transparent cells in B leave A's value."""
    ha, wa = grid_shape(grid_a)
    hb, wb = grid_shape(grid_b)
    h, w = max(ha, hb), max(wa, wb)
    out = [[transparent for _ in range(w)] for _ in range(h)]
    for r in range(ha):
        for c in range(wa):
            out[r][c] = grid_a[r][c]
    for r in range(hb):
        for c in range(wb):
            if grid_b[r][c] != transparent:
                out[r][c] = grid_b[r][c]
    return out


# ---------------------------------------------------------------------------
# Physics-ish
# ---------------------------------------------------------------------------

def gravity_drop(grid: Grid, direction: Direction = "down") -> Grid:
    """Slide non-background cells until blocked by another cell or the edge."""
    bg = background_color(grid)
    h, w = grid_shape(grid)
    out = [[bg for _ in range(w)] for _ in range(h)]

    if direction in ("down", "up"):
        for c in range(w):
            col = [grid[r][c] for r in range(h) if grid[r][c] != bg]
            if direction == "down":
                for i, val in enumerate(col):
                    out[h - len(col) + i][c] = val
            else:
                for i, val in enumerate(col):
                    out[i][c] = val
    else:
        for r in range(h):
            row = [grid[r][c] for c in range(w) if grid[r][c] != bg]
            if direction == "right":
                for i, val in enumerate(row):
                    out[r][w - len(row) + i] = val
            else:
                for i, val in enumerate(row):
                    out[r][i] = val
    return out


def fill_enclosed_regions(grid: Grid, fill_color: int) -> Grid:
    """Fill background regions not connected to the border."""
    bg = background_color(grid)
    h, w = grid_shape(grid)
    out = [row[:] for row in grid]
    reachable = [[False] * w for _ in range(h)]

    from collections import deque

    q: deque[Tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if out[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if out[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr][nc] and out[nr][nc] == bg:
                reachable[nr][nc] = True
                q.append((nr, nc))

    for r in range(h):
        for c in range(w):
            if out[r][c] == bg and not reachable[r][c]:
                out[r][c] = fill_color
    return out


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisInfo:
    kind: str  # "horizontal" | "vertical" | "none"
    index: Optional[int] = None  # row or col of reflection axis when applicable


def symmetry_axis(grid: Grid) -> AxisInfo:
    """Detect the simplest reflection symmetry axis, if any."""
    if grid == reflect_vertical(grid):
        h = len(grid)
        return AxisInfo(kind="horizontal", index=h // 2)
    if grid == reflect_horizontal(grid):
        w = len(grid[0]) if grid else 0
        return AxisInfo(kind="vertical", index=w // 2)
    return AxisInfo(kind="none")


# ---------------------------------------------------------------------------
# Library catalog (injected into code-gen prompts)
# ---------------------------------------------------------------------------

PRIMITIVES = {
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "rotate_270": rotate_270,
    "reflect_horizontal": reflect_horizontal,
    "reflect_vertical": reflect_vertical,
    "transpose": transpose,
    "recolor": recolor,
    "find_objects": find_objects,
    "find_largest_object": find_largest_object,
    "find_smallest_object": find_smallest_object,
    "gravity_drop": gravity_drop,
    "tile_grid": tile_grid,
    "fill_enclosed_regions": fill_enclosed_regions,
    "count_objects": count_objects,
    "count_color": count_color,
    "crop_to_bounding_box": crop_to_bounding_box,
    "overlay": overlay,
    "symmetry_axis": symmetry_axis,
    "background_color": background_color,
}


def primitive_signatures() -> List[str]:
    """Human-readable signatures for prompt injection."""
    return [
        "rotate_90(grid) -> grid",
        "rotate_180(grid) -> grid",
        "rotate_270(grid) -> grid",
        "reflect_horizontal(grid) -> grid",
        "reflect_vertical(grid) -> grid",
        "transpose(grid) -> grid",
        "recolor(grid, from_color, to_color) -> grid",
        "find_objects(grid, connectivity=4) -> List[Object]",
        "find_largest_object(grid) -> Object | None",
        "find_smallest_object(grid) -> Object | None",
        "gravity_drop(grid, direction) -> grid  # direction in up|down|left|right",
        "tile_grid(grid, n_rows, n_cols) -> grid",
        "fill_enclosed_regions(grid, fill_color) -> grid",
        "count_objects(grid) -> int",
        "count_color(grid, color) -> int",
        "crop_to_bounding_box(grid) -> grid",
        "overlay(grid_a, grid_b, transparent=0) -> grid",
        "symmetry_axis(grid) -> AxisInfo",
        "background_color(grid) -> int",
    ]


def library_source_for_sandbox() -> str:
    """Return source that defines primitives inside the sandbox namespace.

    Generated solve() code can call these by name when this preamble is
    prepended before exec. Kept intentionally self-contained (no imports
    of project modules) so the sandbox stays closed.
    """
    # For Week 1 we inject signatures into prompts and also provide a
    # restricted in-process helper set via get_sandbox_helpers().
    return ""


def get_sandbox_helpers() -> dict:
    """Callables safe to inject into the sandbox globals."""
    return dict(PRIMITIVES)


def format_library_for_prompt() -> str:
    lines = ["Available helper primitives:"]
    for sig in primitive_signatures():
        lines.append(f"- {sig}")
    return "\n".join(lines)

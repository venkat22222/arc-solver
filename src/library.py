"""DreamCoder/STITCH-style primitive library for ARC grid transforms.

Generated code is encouraged to compose these helpers rather than
re-implementing common operations from scratch.
"""

from __future__ import annotations

import inspect
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
    """Rotate a grid 90° clockwise."""
    h, w = grid_shape(grid)
    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]


def rotate_180(grid: Grid) -> Grid:
    """Rotate a grid 180°."""
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: Grid) -> Grid:
    """Rotate a grid 270° clockwise (90° counter-clockwise)."""
    return rotate_90(rotate_90(rotate_90(grid)))


def reflect_horizontal(grid: Grid) -> Grid:
    """Reflect a grid horizontally (left-right mirror flip)."""
    return [row[::-1] for row in grid]


def reflect_vertical(grid: Grid) -> Grid:
    """Reflect a grid vertically (top-bottom mirror flip)."""
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    """Transpose rows and columns of a grid."""
    h, w = grid_shape(grid)
    return [[grid[r][c] for r in range(h)] for c in range(w)]


# ---------------------------------------------------------------------------
# Color / objects
# ---------------------------------------------------------------------------

def recolor(grid: Grid, from_color: int, to_color: int) -> Grid:
    """Replace all cells of from_color with to_color in grid."""
    return [[to_color if c == from_color else c for c in row] for row in grid]


def count_color(grid: Grid, color: int) -> int:
    """Count number of cells matching color (int 0-9) in grid."""
    return sum(1 for row in grid for c in row if c == color)


def count_objects(grid: Grid, connectivity: int = 4) -> int:
    """Count connected components of non-background cells (connectivity 4 or 8)."""
    return len(find_objects(grid, connectivity=connectivity))  # type: ignore[arg-type]


def find_largest_object(grid: Grid, connectivity: int = 4) -> Optional[ObjectInfo]:
    """Return the largest ObjectInfo dataclass by size (cells count), or None if empty."""
    objs = find_objects(grid, connectivity=connectivity)  # type: ignore[arg-type]
    if not objs:
        return None
    return max(objs, key=lambda o: (o.size, -o.id))


def find_smallest_object(grid: Grid, connectivity: int = 4) -> Optional[ObjectInfo]:
    """Return the smallest ObjectInfo dataclass by size (cells count), or None if empty."""
    objs = find_objects(grid, connectivity=connectivity)  # type: ignore[arg-type]
    if not objs:
        return None
    return min(objs, key=lambda o: (o.size, o.id))


def crop_to_bounding_box(grid: Grid, connectivity: int = 4) -> Grid:
    """Crop grid to the bounding box enclosing all non-background cells."""
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
    """Repeat grid into an n_rows × n_cols block tiling."""
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be >= 1")
    return [row * n_cols for _ in range(n_rows) for row in grid]


def overlay(grid_a: Grid, grid_b: Grid, transparent: int = 0) -> Grid:
    """Overlay grid_b onto grid_a; cells in grid_b with transparent color do not overwrite grid_a."""
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
    """Slide non-background cells in direction ('up'|'down'|'left'|'right') until blocked."""
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
    """Fill background regions enclosed by non-background cells and not touching borders with fill_color."""
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
    """Detect reflection symmetry axis; returns AxisInfo(kind='horizontal'|'vertical'|'none', index=...)."""
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
        "rotate_90(grid: List[List[int]]) -> List[List[int]]",
        "rotate_180(grid: List[List[int]]) -> List[List[int]]",
        "rotate_270(grid: List[List[int]]) -> List[List[int]]",
        "reflect_horizontal(grid: List[List[int]]) -> List[List[int]]",
        "reflect_vertical(grid: List[List[int]]) -> List[List[int]]",
        "transpose(grid: List[List[int]]) -> List[List[int]]",
        "recolor(grid: List[List[int]], from_color: int, to_color: int) -> List[List[int]]",
        "find_objects(grid: List[List[int]], connectivity: int = 4) -> List[ObjectInfo]",
        "find_largest_object(grid: List[List[int]], connectivity: int = 4) -> Optional[ObjectInfo]",
        "find_smallest_object(grid: List[List[int]], connectivity: int = 4) -> Optional[ObjectInfo]",
        "gravity_drop(grid: List[List[int]], direction: str = 'down') -> List[List[int]]",
        "tile_grid(grid: List[List[int]], n_rows: int, n_cols: int) -> List[List[int]]",
        "fill_enclosed_regions(grid: List[List[int]], fill_color: int) -> List[List[int]]",
        "count_objects(grid: List[List[int]], connectivity: int = 4) -> int",
        "count_color(grid: List[List[int]], color: int) -> int",
        "crop_to_bounding_box(grid: List[List[int]], connectivity: int = 4) -> List[List[int]]",
        "overlay(grid_a: List[List[int]], grid_b: List[List[int]], transparent: int = 0) -> List[List[int]]",
        "symmetry_axis(grid: List[List[int]]) -> AxisInfo",
        "background_color(grid: List[List[int]]) -> int",
    ]


def list_primitive_names() -> List[str]:
    """Names of all 19 primitives available to generated code."""
    return list(PRIMITIVES)


def _format_type(annotation: object) -> str:
    if annotation is inspect.Signature.empty:
        return ""
    if annotation is None or annotation is type(None):
        return "None"
    s = str(annotation)
    s = s.replace("typing.", "").replace("src.preprocess.", "").replace("src.library.", "")
    if hasattr(annotation, "__name__"):
        s = getattr(annotation, "__name__")
    elif hasattr(annotation, "_name") and getattr(annotation, "_name"):
        name = getattr(annotation, "_name")
        args = getattr(annotation, "__args__", None)
        if args:
            args_str = ", ".join(_format_type(a) for a in args)
            s = f"{name}[{args_str}]"
        else:
            s = name
    return s


def generate_library_schema() -> str:
    """Build a compact, exact API schema for prompt injection.

    Uses ``inspect`` to derive the real parameter names, defaults, and
    type hints from each primitive, plus its one-line docstring. Generated
    code must call these functions with exactly these signatures.
    """
    lines = ["Available helper primitives (exact signatures & docstrings):"]
    for name in list_primitive_names():
        fn = PRIMITIVES[name]
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = inspect.Signature()
        params = []
        for pname, p in sig.parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            default = "" if p.default is inspect.Parameter.empty else f"={p.default!r}"
            ann = ""
            if p.annotation is not inspect.Parameter.empty:
                ann = f": {_format_type(p.annotation)}"
            params.append(f"{pname}{ann}{default}")
        ret = sig.return_annotation
        ret_str = ""
        if ret is not inspect.Signature.empty:
            ret_str = f" -> {_format_type(ret)}"
        doc = (fn.__doc__ or "").strip().splitlines()
        doc_str = f"  # {doc[0]}" if doc else ""
        lines.append(f"- {name}({', '.join(params)}){ret_str}{doc_str}")

    lines.append("")
    lines.append("ObjectInfo structure (returned by find_objects, find_largest_object, find_smallest_object):")
    lines.append("- obj.id: int (component ID)")
    lines.append("- obj.color: int (color 0-9)")
    lines.append("- obj.size: int (cell count)")
    lines.append("- obj.bbox: tuple[int, int, int, int] (r0, c0, r1, c1 inclusive)")
    lines.append("- obj.shape_pixels: tuple[tuple[int, int], ...] (coordinates relative to bbox top-left)")
    lines.append("- obj.shape_name: str (e.g. 'single-pixel', 'L-tromino', 'blob(...)')")
    lines.append("CRITICAL: ObjectInfo is a dataclass, NOT an iterable (do NOT write `for p in obj:`).")
    lines.append("To iterate cells in grid coordinates:")
    lines.append("  for (dr, dc) in obj.shape_pixels:")
    lines.append("      r, c = obj.bbox[0] + dr, obj.bbox[1] + dc")

    return "\n".join(lines)


def library_source_for_sandbox() -> str:
    """Return source that defines primitives inside the sandbox namespace."""
    return ""


def get_sandbox_helpers() -> dict:
    """Callables safe to inject into the sandbox globals."""
    return dict(PRIMITIVES)


def format_library_for_prompt() -> str:
    return generate_library_schema()

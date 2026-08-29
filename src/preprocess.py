"""Convert ARC grids into structured, LLM-readable text descriptions.

Detects connected components (objects) and extracts color, bbox, size,
and a normalized shape descriptor. Also provides a raw-grid text fallback.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

Grid = List[List[int]]
Connectivity = Literal[4, 8]

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = _NEIGHBORS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class ObjectInfo:
    """One connected component in a grid."""

    id: int
    color: int
    size: int
    bbox: Tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive
    shape_pixels: Tuple[Tuple[int, int], ...]  # coords relative to bbox top-left
    shape_name: str
    cells: Tuple[Tuple[int, int], ...] = ()  # Absolute (r, c) coordinates in grid

    @property
    def pixels(self) -> Tuple[Tuple[int, int], ...]:
        """Alias for absolute coordinates (r, c)."""
        return self.cells

    @property
    def coords(self) -> Tuple[Tuple[int, int], ...]:
        """Alias for absolute coordinates (r, c)."""
        return self.cells

    def __iter__(self):
        """Allows direct iteration: for r, c in obj: ..."""
        return iter(self.cells)


def grid_shape(grid: Grid) -> Tuple[int, int]:
    return len(grid), (len(grid[0]) if grid else 0)


def background_color(grid: Grid) -> int:
    """Most common color in the grid (ties broken by smaller color id)."""
    counts = Counter(c for row in grid for c in row)
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def find_objects(
    grid: Grid,
    connectivity: Connectivity = 4,
    ignore_background: bool = True,
    bg: Optional[int] = None,
) -> List[ObjectInfo]:
    """Flood-fill connected components of the same color.

    By default the background (most common color) is ignored so objects
    are the non-background blobs — the usual ARC convention.
    """
    h, w = grid_shape(grid)
    if h == 0 or w == 0:
        return []

    if bg is None:
        bg = background_color(grid)

    neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
    visited = [[False] * w for _ in range(h)]
    objects: List[ObjectInfo] = []

    def flood(sr: int, sc: int) -> List[Tuple[int, int]]:
        color = grid[sr][sc]
        q: deque[Tuple[int, int]] = deque([(sr, sc)])
        visited[sr][sc] = True
        cells: List[Tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                    if grid[nr][nc] == color:
                        visited[nr][nc] = True
                        q.append((nr, nc))
        return cells

    for r in range(h):
        for c in range(w):
            if visited[r][c]:
                continue
            if ignore_background and grid[r][c] == bg:
                visited[r][c] = True
                continue
            cells = flood(r, c)
            color = grid[r][c]
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            shape_pixels = tuple(sorted((pr - r0, pc - c0) for pr, pc in cells))
            objects.append(
                ObjectInfo(
                    id=len(objects) + 1,
                    color=color,
                    size=len(cells),
                    bbox=(r0, c0, r1, c1),
                    shape_pixels=shape_pixels,
                    shape_name=_name_shape(shape_pixels, r1 - r0 + 1, c1 - c0 + 1),
                    cells=tuple(sorted(cells)),
                )
            )
    return objects


def _name_shape(pixels: Sequence[Tuple[int, int]], bh: int, bw: int) -> str:
    n = len(pixels)
    if n == 1:
        return "single-pixel"
    if n == bh * bw:
        if bh == 1:
            return f"horizontal-line({bw})"
        if bw == 1:
            return f"vertical-line({bh})"
        if bh == bw:
            return f"filled-square({bh}x{bw})"
        return f"filled-rect({bh}x{bw})"

    pixel_set = set(pixels)
    # Classic L tromino / tetromino heuristics
    if n == 3 and bh == 2 and bw == 2:
        return "L-tromino"
    if n == 4 and bh == 2 and bw == 2 and len(pixel_set) == 4:
        return "filled-square(2x2)"
    if n == 4 and { (0, 0), (1, 0), (2, 0), (2, 1) }.issubset(pixel_set) or _is_l_tetromino(pixel_set, bh, bw):
        return "L-block"
    if bh == 1 or bw == 1:
        return f"line-fragment({n})"
    return f"blob({bh}x{bw}, n={n})"


def _is_l_tetromino(pixel_set: set, bh: int, bw: int) -> bool:
    if len(pixel_set) != 4:
        return False
    # Any 3-long arm + 1 stub at an end → L
    if bh == 3 and bw == 2:
        return True
    if bh == 2 and bw == 3:
        return True
    return False


def raw_grid_text(grid: Grid) -> str:
    """Plain numeric grid, one row per line."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def cell_changes(
    input_grid: Grid, output_grid: Grid
) -> List[Tuple[int, int, int, int]]:
    """Same-size grids only: list of (row, col, old_color, new_color) that differ."""
    ih, iw = grid_shape(input_grid)
    oh, ow = grid_shape(output_grid)
    if (ih, iw) != (oh, ow):
        return []
    return [
        (r, c, input_grid[r][c], output_grid[r][c])
        for r in range(ih)
        for c in range(iw)
        if input_grid[r][c] != output_grid[r][c]
    ]


def format_cell_diff(input_grid: Grid, output_grid: Grid) -> Optional[str]:
    """Text block of changed cells, or None when sizes differ / nothing changed."""
    ih, iw = grid_shape(input_grid)
    oh, ow = grid_shape(output_grid)
    if (ih, iw) != (oh, ow):
        return None
    changes = cell_changes(input_grid, output_grid)
    if not changes:
        return "Changed cells: (none — output identical to input)"
    lines = [f"Changed cells ({len(changes)} of {ih * iw}):"]
    # Compact mask: . = same, X = changed
    mask = []
    changed = {(r, c) for r, c, _, _ in changes}
    for r in range(ih):
        mask.append(" ".join("X" if (r, c) in changed else "." for c in range(iw)))
    lines.append("Diff mask (. = same, X = changed):")
    lines.extend(mask)
    lines.append("Sample changes (row, col): old -> new:")
    max_changes_to_show = 15
    for r, c, old, new in changes[:max_changes_to_show]:
        lines.append(f"  ({r},{c}): {old} -> {new}")
    if len(changes) > max_changes_to_show:
        lines.append(f"  (... and {len(changes) - max_changes_to_show} more pixel changes omitted for brevity)")
    return "\n".join(lines)


def describe_grid(
    grid: Grid,
    connectivity: Connectivity = 4,
    include_raw: bool = True,
    label: str = "Grid",
) -> str:
    """Build a structured text description of a grid."""
    h, w = grid_shape(grid)
    bg = background_color(grid)
    objects = find_objects(grid, connectivity=connectivity, bg=bg)

    lines = [
        f"{label} size: {h}x{w}",
        f"Background color: {bg}",
        f"Connectivity: {connectivity}",
        f"Objects ({len(objects)}):",
    ]
    if not objects:
        lines.append("  (none — empty or solid background)")
    else:
        # Prioritize largest / most prominent objects and cap at 10
        sorted_objs = sorted(objects, key=lambda o: -o.size)
        max_show = 10
        for obj in sorted_objs[:max_show]:
            r0, c0, r1, c1 = obj.bbox
            lines.append(
                f"- Object {obj.id}: color={obj.color}, size={obj.size} cells, "
                f"bbox=({r0},{c0})-({r1},{c1}), shape={obj.shape_name}"
            )
        if len(objects) > max_show:
            lines.append(f"  (... and {len(objects) - max_show} smaller background/fragment objects omitted)")

    if include_raw:
        lines.append("Raw grid:")
        lines.append(raw_grid_text(grid))
    return "\n".join(lines)


def describe_pair(
    input_grid: Grid,
    output_grid: Grid,
    connectivity: Connectivity = 4,
    include_raw: bool = True,
    example_index: int = 1,
) -> str:
    """Describe one train input→output example."""
    header = f"=== Example {example_index} ==="
    inp = describe_grid(
        input_grid, connectivity=connectivity, include_raw=include_raw, label="Input"
    )
    out = describe_grid(
        output_grid, connectivity=connectivity, include_raw=include_raw, label="Output"
    )
    parts = [header, "--- INPUT ---", inp, "--- OUTPUT ---", out]
    diff = format_cell_diff(input_grid, output_grid)
    if diff is not None:
        parts.extend(["--- CELL DIFF (same-size only) ---", diff])
    return "\n".join(parts)


def describe_train_pairs(
    train_pairs: Sequence[Tuple[Grid, Grid]],
    connectivity: Connectivity = 4,
    include_raw: bool = True,
) -> str:
    """Describe all training pairs for prompt injection."""
    blocks = [
        describe_pair(inp, out, connectivity=connectivity, include_raw=include_raw, example_index=i)
        for i, (inp, out) in enumerate(train_pairs, start=1)
    ]
    return "\n\n".join(blocks)


def should_include_raw(
    train_pairs: Sequence[Tuple[Grid, Grid]],
    connectivity: Connectivity = 4,
) -> bool:
    """Include raw grids only when object lists are ambiguous (Topic F).

    Skip raw by default when every train grid has a usable object list —
    saves prompt tokens on P100 / API. For large grids (>15x15), always skip
    raw to prevent quadratic SDPA attention memory explosion.
    """
    if not train_pairs:
        return True
    for inp, out in train_pairs:
        for g in (inp, out):
            h, w = grid_shape(g)
            if h * w > 200:  # Grids larger than ~14x14 should never dump full raw text
                return False
            bg = background_color(g)
            objs = find_objects(g, connectivity=connectivity, bg=bg)
            if not objs:
                return True
            # Too many tiny fragments → object list is noise; raw is clearer for small grids.
            if len(objs) > 18:
                return True
    return False

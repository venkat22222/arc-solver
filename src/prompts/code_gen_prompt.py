"""Stage 1b: prompt template for turning one hypothesis into Python code."""

from __future__ import annotations

import re

from ..library import format_library_for_prompt

_FEW_SHOT = """
Examples of correct solve() functions (call helpers directly — do not reimplement them):
```python
def solve(grid):
    return rotate_180(grid)
```
```python
def solve(grid):
    return reflect_horizontal(grid)
```
```python
def solve(grid):
    return recolor(grid, 1, 2)
```
```python
def solve(grid):
    # Hypothesis mentioned BOTH mirror/reflect AND 2x2 tiling/expansion — compose them.
    # tile_grid alone does NOT reflect.
    top = [row + row[::-1] for row in grid]
    return top + top[::-1]
```
```python
def solve(grid):
    return tile_grid(reflect_vertical(grid), 2, 1)
```
"""

_COMPOSE_RULE = """
COMPOSITION RULE: If the hypothesis mentions BOTH reflection/mirroring AND tiling/expansion
(2x2, scale-up, repeat), you MUST compose those operations. Never drop one of them.
`tile_grid(grid, n, m)` alone only repeats the block — it does NOT mirror or flip.
"""

_WHOLE_GRID_RULE = """
WHOLE-GRID PRIOR: Size/color/object-count are preserved for this puzzle. If the hypothesis
describes transforming the entire grid (rotate / mirror / transpose), implement it as a
SINGLE helper return — e.g. `return rotate_180(grid)`. Do NOT use find_objects or
per-object loops unless the hypothesis explicitly requires treating objects separately.
"""


def build_code_gen_prompt(
    hypothesis: str,
    library_text: str | None = None,
    prefer_whole_grid: bool = False,
) -> str:
    if library_text is None:
        library_text = format_library_for_prompt()

    whole = _WHOLE_GRID_RULE if prefer_whole_grid else ""

    return f"""Implement this hypothesized rule as a Python function:
"{hypothesis}"

Requirements:
- Function signature: def solve(grid: List[List[int]]) -> List[List[int]]
- Do NOT use any import statements. All needed helpers are already available in the execution namespace — call them directly by name.
- Use ONLY these helper primitives if possible:
{library_text}
{_COMPOSE_RULE}{whole}{_FEW_SHOT}
- Do not hardcode specific coordinates or grid dimensions — the rule must generalize
- Respond with ONLY a single ```python code fence containing the function. No explanation before or after.
"""


def extract_code(response: str) -> str:
    """Pull a Python code block out of a model response; repair common fence issues."""
    text = (response or "").strip()
    if not text:
        return ""

    # Prefer fenced blocks (greedy last-resort for truncated closing fence)
    fences = list(re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE))
    if fences:
        code = fences[-1].group(1).strip()
        return _strip_imports(_ensure_solve(code))

    # Truncated opening fence without closing ```
    open_fence = re.search(r"```(?:python)?\s*\n(.*)$", text, re.DOTALL | re.IGNORECASE)
    if open_fence:
        code = open_fence.group(1).strip().rstrip("`").strip()
        return _strip_imports(_ensure_solve(code))

    if "def solve" in text:
        start = text.index("def solve")
        return _strip_imports(_ensure_solve(text[start:].strip()))

    return _strip_imports(text)


def _ensure_solve(code: str) -> str:
    """If multiple defs exist, keep from first def solve onward when present."""
    if "def solve" in code:
        idx = code.index("def solve")
        return code[idx:].strip()
    return code.strip()


def _strip_imports(code: str) -> str:
    """Remove import lines so sandbox doesn't waste cycles on blocked imports."""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        lines.append(line)
    return "\n".join(lines).strip()

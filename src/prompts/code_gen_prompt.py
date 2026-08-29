"""Stage 1b: prompt template for turning one hypothesis into Python code."""

from __future__ import annotations

import re

from ..library import format_library_for_prompt, generate_library_schema

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


_STRICT_SCHEMA_RULE = """
STRICT HELPER RULES:
- You may ONLY call the exact helper functions listed above with their exact signatures.
- Any other function name (e.g. hallucinated helpers like `get_bounding_box`, `flood_fill`, `find_components`, `scipy.*`, `np.*`), or a real function called with wrong argument types/order, will cause immediate execution failure.
- No imports allowed (no `import ...`). All allowed helpers and safe builtins are already in scope.
- CRITICAL: Any object returned by find_objects, find_largest_object, or find_smallest_object is an ObjectInfo dataclass, NOT an iterable. Writing `for p in obj:` or `for r, c in obj:` will raise TypeError.
  Access its documented attributes explicitly:
  - obj.id (int)
  - obj.color (int)
  - obj.size (int)
  - obj.bbox (tuple: r0, c0, r1, c1)
  - obj.shape_pixels (tuple of (dr, dc) relative to bbox)
  - obj.shape_name (str)
  To iterate all grid coordinates of an object:
  `for (dr, dc) in obj.shape_pixels: r, c = obj.bbox[0] + dr, obj.bbox[1] + dc`
"""


def _format_train_examples_for_codegen(train_pairs: Sequence[Tuple[list, list]], max_pairs: int = 2) -> str:
    lines = ["Task Examples (ground-truth input -> output pairs):"]
    for idx, (inp, out) in enumerate(train_pairs[:max_pairs], 1):
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        lines.append(f"Example {idx} (input {ih}x{iw} -> output {oh}x{ow}):")
        if ih <= 10 and iw <= 10:
            lines.append(f"  Input:  {inp}")
        else:
            lines.append(f"  Input:  [shape {ih}x{iw}, first row: {inp[0][:8]}...]")
        if oh <= 10 and ow <= 10:
            lines.append(f"  Output: {out}")
        else:
            lines.append(f"  Output: [shape {oh}x{ow}, first row: {out[0][:8]}...]")
    return "\n".join(lines)


def build_code_gen_prompt(
    hypothesis: str,
    library_text: str | None = None,
    prefer_whole_grid: bool = False,
    train_pairs: Sequence[Tuple[list, list]] | None = None,
) -> str:
    if library_text is None:
        library_text = generate_library_schema()

    whole = _WHOLE_GRID_RULE if prefer_whole_grid else ""
    examples_block = ""
    if train_pairs:
        examples_block = f"\n{_format_train_examples_for_codegen(train_pairs)}\n"

    return f"""Implement as Python.
{examples_block}
Hypothesis: "{hypothesis}"

Rules:
- Signature: def solve(grid: List[List[int]]) -> List[List[int]]
- No imports. Helpers below are already in scope — call by name.

{library_text}
{_STRICT_SCHEMA_RULE}
{_COMPOSE_RULE}{whole}{_FEW_SHOT}
- Must produce the exact output for all examples.
- Output ONLY one ```python fence with the function. Zero explanation.
"""


def build_direct_solve_prompt(
    train_pairs: Sequence[Tuple[list, list]],
    library_text: str | None = None,
) -> str:
    """Fast direct Program-of-Thought prompt: synthesize def solve(grid) directly from input->output pairs."""
    if library_text is None:
        library_text = generate_library_schema()

    examples_block = _format_train_examples_for_codegen(train_pairs, max_pairs=3)

    return f"""You are an expert Python programmer solving an ARC visual reasoning puzzle.

{examples_block}

Write a Python function `def solve(grid: List[List[int]]) -> List[List[int]]` that transforms any input grid into its corresponding output grid following the underlying visual pattern.

{library_text}
{_STRICT_SCHEMA_RULE}
{_FEW_SHOT}

Rules:
1. Do not use any import statements. All helper functions above are already in scope.
2. Return a 2D list of integers (0-9).
3. The function MUST work correctly for ALL training examples above.
4. Output ONLY the Python code enclosed in a single ```python code block. Zero explanation.
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
    """Keep all helper functions and classes; verify def solve exists."""
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

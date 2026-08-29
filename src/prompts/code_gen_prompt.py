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


def format_grid_compact(grid: Sequence[Sequence[int]]) -> str:
    """Format a 2D grid into clean, spatial space-separated text rows for LLM vision."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _format_train_examples_for_codegen(train_pairs: Sequence[Tuple[list, list]], max_pairs: int = 3) -> str:
    blocks = []
    for idx, (inp, out) in enumerate(train_pairs[:max_pairs], 1):
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        if ih <= 20 and iw <= 20:
            inp_str = format_grid_compact(inp)
            out_str = format_grid_compact(out)
            blocks.append(f"=== Example {idx} ===\nInput ({ih}x{iw}):\n{inp_str}\n\nOutput ({oh}x{ow}):\n{out_str}")
        else:
            blocks.append(f"=== Example {idx} ===\nInput ({ih}x{iw})\nOutput ({oh}x{ow})")
    return "\n\n".join(blocks)


_COMPACT_HELPERS = """Available helper functions in scope:
- rotate_90(grid), rotate_180(grid), rotate_270(grid)
- reflect_horizontal(grid), reflect_vertical(grid), transpose(grid)
- recolor(grid, mapping_dict) -> e.g. recolor(grid, {8: 1, 2: 3})
- find_objects(grid, connectivity=4) -> list of ObjectInfo (accessible: obj.cells, obj.color, obj.bbox, obj.size)
- gravity_drop(grid, bg=0), fill_enclosed_regions(grid, color)
- crop_to_bounding_box(grid, bbox), scale_grid(grid, factor_y, factor_x)"""


_REFERENCE_EXAMPLE = """Example Pattern:
Input:
0 1
1 0
Output:
1 0
0 1
Solution:
```python
def solve(grid: List[List[int]]) -> List[List[int]]:
    return reflect_horizontal(grid)
```"""


def build_code_gen_prompt(
    hypothesis: str,
    library_text: str | None = None,
    prefer_whole_grid: bool = False,
    train_pairs: Sequence[Tuple[list, list]] | None = None,
) -> str:
    examples_block = ""
    if train_pairs:
        examples_block = f"\n{_format_train_examples_for_codegen(train_pairs)}\n"

    return f"""You are solving an ARC visual reasoning puzzle in Python.

{_REFERENCE_EXAMPLE}

Task to Solve:
{examples_block}
Hypothesis: "{hypothesis}"

{_COMPACT_HELPERS}

Write the Python function `def solve(grid: List[List[int]]) -> List[List[int]]` that implements this transformation.
Output ONLY the Python code in a ```python ``` code block:

```python
def solve(grid: List[List[int]]) -> List[List[int]]:
"""


def build_direct_solve_prompt(
    train_pairs: Sequence[Tuple[list, list]],
    library_text: str | None = None,
) -> str:
    """Fast direct Program-of-Thought prompt: synthesize def solve(grid) directly from input->output pairs."""
    examples_block = _format_train_examples_for_codegen(train_pairs, max_pairs=3)

    return f"""You are solving an ARC visual reasoning puzzle in Python.

{_REFERENCE_EXAMPLE}

Task to Solve:
{examples_block}

{_COMPACT_HELPERS}

Instructions:
Write the Python function `def solve(grid: List[List[int]]) -> List[List[int]]` that transforms any input grid into its corresponding output grid following the visual pattern demonstrated above.
- Must work for all example pairs above.
- Output ONLY the Python code in a ```python ``` block.

```python
def solve(grid: List[List[int]]) -> List[List[int]]:
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
        return _strip_imports(_sanitize_code_lines(_ensure_solve(code)))

    # Truncated opening fence without closing ```
    open_fence = re.search(r"```(?:python)?\s*\n(.*)$", text, re.DOTALL | re.IGNORECASE)
    if open_fence:
        code = open_fence.group(1).strip().rstrip("`").strip()
        return _strip_imports(_sanitize_code_lines(_ensure_solve(code)))

    if "def solve" in text:
        start = text.index("def solve")
        return _strip_imports(_sanitize_code_lines(_ensure_solve(text[start:].strip())))

    # If the response starts immediately with the function body or logic
    if any(k in text for k in ("return ", "out =", "grid", "for ", "if ", "    ")):
        full_code = "def solve(grid: List[List[int]]) -> List[List[int]]:\n"
        for line in text.splitlines():
            if not line.startswith("    ") and not line.startswith("\t"):
                full_code += "    " + line + "\n"
            else:
                full_code += line + "\n"
        return _strip_imports(_sanitize_code_lines(full_code.strip()))

    return _strip_imports(_sanitize_code_lines(text))


def _sanitize_code_lines(code: str) -> str:
    """First-pass: comment out bare space-separated digit rows.
    Then: compile-and-repair loop — if a SyntaxError remains, comment out the
    exact offending line (by line number) and retry, up to 10 times."""
    lines = code.splitlines()

    # Pass 1 – regex scrub for obvious bare-matrix lines
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Bare space-separated digit line (e.g. "0 0 3 0 0")
        if stripped and re.match(r"^[0-9\s,\[\]]+$", stripped) and not stripped.isdigit():
            cleaned.append("    # [scrubbed] " + stripped)
        else:
            cleaned.append(line)
    code = "\n".join(cleaned)

    # Pass 2 – compile-and-repair loop
    for _ in range(10):
        try:
            compile(code, "<generated>", "exec")
            break  # compiles cleanly
        except SyntaxError as exc:
            lineno = exc.lineno  # 1-based
            if lineno is None:
                break
            code_lines = code.splitlines()
            idx = lineno - 1
            if 0 <= idx < len(code_lines):
                bad = code_lines[idx]
                code_lines[idx] = "    # [repaired] " + bad.strip()
                code = "\n".join(code_lines)
            else:
                break
        except Exception:
            break

    return code


def _ensure_solve(code: str) -> str:
    """Ensure def solve exists in code."""
    if "def solve" not in code:
        return "def solve(grid: List[List[int]]) -> List[List[int]]:\n    " + code.replace("\n", "\n    ")
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

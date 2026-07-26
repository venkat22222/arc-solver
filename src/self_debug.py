"""Stage 2: self-debug loop — feed execution traces back on failure."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .prompts.code_gen_prompt import extract_code
from .sandbox import ExecutionResult, safe_execute

Grid = List[List[int]]
TrainPair = Tuple[Grid, Grid]


def _grid_to_text(grid: Optional[Grid]) -> str:
    if grid is None:
        return "(no output)"
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _visual_diff(expected: Grid, actual: Optional[Grid]) -> str:
    if actual is None:
        return "Actual output missing."
    eh, ew = len(expected), len(expected[0]) if expected else 0
    ah, aw = len(actual), len(actual[0]) if actual else 0
    lines = [f"Shape: expected {eh}x{ew}, got {ah}x{aw}"]
    if (eh, ew) != (ah, aw):
        lines.append("Expected:")
        lines.append(_grid_to_text(expected))
        lines.append("Got:")
        lines.append(_grid_to_text(actual))
        return "\n".join(lines)

    mismatch = 0
    mark_rows = []
    for r in range(eh):
        marks = []
        for c in range(ew):
            if expected[r][c] != actual[r][c]:
                marks.append("X")
                mismatch += 1
            else:
                marks.append(".")
        mark_rows.append(" ".join(marks))
    lines.append(f"Cell mismatches: {mismatch}")
    lines.append("Diff mask (. = match, X = mismatch):")
    lines.extend(mark_rows)
    lines.append("Expected:")
    lines.append(_grid_to_text(expected))
    lines.append("Got:")
    lines.append(_grid_to_text(actual))
    return "\n".join(lines)


def build_debug_feedback(
    results: Sequence[ExecutionResult],
    train_pairs: Sequence[TrainPair],
) -> str:
    """Per-example visual diffs + traces for the self-debug prompt."""
    parts: List[str] = []
    for i, (result, (inp, expected)) in enumerate(zip(results, train_pairs), start=1):
        ok = result.success and result.output_grid == expected
        header = f"--- Example {i}: {'PASS' if ok else 'FAIL'} ---"
        block = [header]
        if not result.success:
            block.append(f"Error: {result.error_message}")
        if result.execution_trace:
            block.append("Execution trace:")
            block.extend(f"  {line}" for line in result.execution_trace[:50])
        if not ok:
            block.append(_visual_diff(expected, result.output_grid))
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def feedback_prompt(code_str: str, feedback: str) -> str:
    return f"""The following Python function failed on one or more training examples.
Fix the code so it produces the expected outputs for ALL examples.
Return only the corrected code.

Current code:
```python
{code_str}
```

Failure report (X-ray diffs):
{feedback}
"""


def self_debug_loop(
    llm_client,
    code_str: str,
    train_pairs: Sequence[TrainPair],
    max_retries: int = 3,
    timeout_seconds: float = 5.0,
) -> Optional[str]:
    """Verify code on train pairs; on failure, ask the LLM to fix it."""
    for _attempt in range(max_retries):
        results = [
            safe_execute(code_str, inp, timeout_seconds=timeout_seconds)
            for inp, _ in train_pairs
        ]
        if all(
            r.success and r.output_grid == expected
            for r, (_, expected) in zip(results, train_pairs)
        ):
            return code_str

        feedback = build_debug_feedback(results, train_pairs)
        response = llm_client.generate(feedback_prompt(code_str, feedback))
        code_str = extract_code(response)
    return None

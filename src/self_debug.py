"""Stage 2: self-debug loop — feed execution traces back on failure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .prompts.code_gen_prompt import extract_code
from .sandbox import ExecutionResult, cell_match_fraction, safe_execute

Grid = List[List[int]]
TrainPair = Tuple[Grid, Grid]


@dataclass
class SelfDebugStats:
    attempts: int = 0
    noop_attempts: int = 0
    productive_attempts: int = 0
    details: List[dict] = field(default_factory=list)


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
    frac = cell_match_fraction(expected, actual)
    lines.append(f"Cell mismatches: {mismatch} (partial_credit={frac:.3f})")
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
    for i, (result, (_, expected)) in enumerate(zip(results, train_pairs), start=1):
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


def feedback_prompt(
    code_str: str,
    feedback: str,
    *,
    noop: bool = False,
    timeout_failure: bool = False,
    library_text: str | None = None,
) -> str:
    if library_text is None:
        from .library import generate_library_schema

        library_text = generate_library_schema()

    noop_line = (
        "\nIMPORTANT: Your previous fix produced the EXACT SAME (still wrong) outputs "
        "on every training example — it changed nothing. Try a genuinely different "
        "approach (different primitive, opposite rotation direction, compose ops, etc.).\n"
        if noop
        else ""
    )

    timeout_line = (
        "\nCRITICAL TIMEOUT GUIDANCE:\n"
        "Your previous code execution timed out. This is almost always caused by an unbounded "
        "`while` loop, infinite recursion, or unconstrained coordinate scanning.\n"
        "Required fixes:\n"
        "- Replace `while` loops with bounded `for` loops (e.g. `for _ in range(...)`) with explicit iteration limits.\n"
        "- Add strict boundary checks (e.g. `0 <= r < len(grid) and 0 <= c < len(grid[0])`) to prevent index runaway.\n"
        "- Simplify search logic and avoid repeated exhaustive scans inside nested loops.\n"
        if timeout_failure
        else ""
    )

    return f"""The following Python function failed on one or more training examples.
Fix the code so it produces the expected outputs for ALL examples.
Return only the corrected code.
{noop_line}{timeout_line}
Current code:
```python
{code_str}
```

Failure report (X-ray diffs & execution traces):
{feedback}

{library_text}

Rules:
- Output ONLY one ```python fence with the corrected def solve(grid) function. Zero explanation.
- No imports.
- Do not treat ObjectInfo as iterable. Access .bbox, .shape_pixels, .color, .size explicitly.
"""


def _outputs_fingerprint(results: Sequence[ExecutionResult]) -> Tuple:
    """Hashable fingerprint of produced grids (None on failure)."""
    return tuple(
        None
        if (not r.success or r.output_grid is None)
        else tuple(tuple(row) for row in r.output_grid)
        for r in results
    )


def self_debug_loop(
    llm_client,
    code_str: str,
    train_pairs: Sequence[TrainPair],
    max_retries: int = 3,
    timeout_seconds: float = 5.0,
    puzzle_id: str = "unknown",
    cumulative_timeout_ceiling_s: float = 90.0,
    puzzle_start_time: Optional[float] = None,
) -> Tuple[Optional[str], SelfDebugStats]:
    """Verify code on train pairs; on failure, ask the LLM to fix it.

    Uses graduated timeout tiers per retry attempt:
    - Attempt 0: tier 0 (5.0s)
    - Retry 1: tier 1 (8.0s)
    - Retry 2+: tier 2 (10.0s)

    Aborts immediately if cumulative time across Tier 2+ exceeds cumulative_timeout_ceiling_s.
    """
    import time
    from .sandbox import get_timeout_for_attempt

    stats = SelfDebugStats()
    retries_left = max_retries
    noop_budget = max_retries
    prev_fp: Optional[Tuple] = None
    last_was_noop = False
    p_start = puzzle_start_time if puzzle_start_time is not None else time.time()

    # Initial verification (attempt index 0)
    t0 = get_timeout_for_attempt(0)
    results = [
        safe_execute(code_str, inp, timeout_seconds=t0)
        for inp, _ in train_pairs
    ]
    for r, (_, expected) in zip(results, train_pairs):
        r.partial_credit = cell_match_fraction(expected, r.output_grid)

    for r in results:
        if r.is_timeout or (r.error_message and "Timeout after" in r.error_message):
            cum_time = time.time() - p_start
            ceiling_hit = cum_time >= cumulative_timeout_ceiling_s
            print(
                f"[sandbox_timeout] puzzle={puzzle_id} attempt=0 timeout_tier={t0:.1f}s "
                f"cumulative={cum_time:.1f}s ceiling_triggered={ceiling_hit}"
            )

    if (time.time() - p_start) >= cumulative_timeout_ceiling_s:
        stats.details.append({"attempt": 0, "ceiling_triggered": True})
        return None, stats

    if all(
        r.success and r.output_grid == expected
        for r, (_, expected) in zip(results, train_pairs)
    ):
        return code_str, stats

    while retries_left > 0:
        if (time.time() - p_start) >= cumulative_timeout_ceiling_s:
            stats.details.append({"attempt": stats.attempts, "ceiling_triggered": True})
            return None, stats

        has_timeout = any(
            r.is_timeout or (r.error_message and "Timeout after" in r.error_message)
            for r in results
        )
        fp = _outputs_fingerprint(results)
        feedback = build_debug_feedback(results, train_pairs)
        response = llm_client.generate(
            feedback_prompt(code_str, feedback, noop=last_was_noop, timeout_failure=has_timeout)
        )
        new_code = extract_code(response)
        stats.attempts += 1

        # Attempt timeout tier depends on retry attempt index
        t_attempt = get_timeout_for_attempt(stats.attempts)
        new_results = [
            safe_execute(new_code, inp, timeout_seconds=t_attempt)
            for inp, _ in train_pairs
        ]
        for r in new_results:
            if r.is_timeout or (r.error_message and "Timeout after" in r.error_message):
                cum_time = time.time() - p_start
                ceiling_hit = cum_time >= cumulative_timeout_ceiling_s
                print(
                    f"[sandbox_timeout] puzzle={puzzle_id} attempt={stats.attempts} timeout_tier={t_attempt:.1f}s "
                    f"cumulative={cum_time:.1f}s ceiling_triggered={ceiling_hit}"
                )

        if (time.time() - p_start) >= cumulative_timeout_ceiling_s:
            stats.details.append({"attempt": stats.attempts, "ceiling_triggered": True})
            return None, stats

        new_fp = _outputs_fingerprint(new_results)
        is_noop = new_fp == fp or (prev_fp is not None and new_fp == prev_fp)

        if is_noop:
            stats.noop_attempts += 1
            stats.details.append({"attempt": stats.attempts, "noop": True})
            last_was_noop = True
            if noop_budget > 0:
                noop_budget -= 1
            else:
                retries_left -= 1
            code_str = new_code
            results = new_results
            prev_fp = new_fp
            continue

        stats.productive_attempts += 1
        stats.details.append({"attempt": stats.attempts, "noop": False})
        last_was_noop = False
        retries_left -= 1
        code_str = new_code
        results = new_results
        prev_fp = new_fp

        if all(
            r.success and r.output_grid == expected
            for r, (_, expected) in zip(new_results, train_pairs)
        ):
            return code_str, stats

    return None, stats

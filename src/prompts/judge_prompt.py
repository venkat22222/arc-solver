"""Stage 3: prompt template for holistic trace judging."""

from __future__ import annotations

from typing import Any, Sequence


def build_judge_prompt(candidates: Sequence[Any]) -> str:
    """Build a long-context prompt comparing full reasoning traces.

    Each candidate should expose: .hypothesis, .code, .verification_summary
    (and optionally .candidate_id).
    """
    blocks = []
    for i, c in enumerate(candidates, start=1):
        hyp = getattr(c, "hypothesis", str(c))
        code = getattr(c, "code", "")
        summary = getattr(c, "verification_summary", "verified on all train pairs")
        blocks.append(
            f"=== CANDIDATE {i} ===\n"
            f"Hypothesis: {hyp}\n"
            f"Verification: {summary}\n"
            f"Code:\n{code}\n"
        )

    joined = "\n".join(blocks)
    return f"""You are judging candidate solutions to an abstract reasoning puzzle.
Compare the FULL reasoning traces below — not just the final answers.
Prefer hypotheses that are general, simple, and consistent with all examples.
Avoid candidates that appear to overfit specific coordinates or magic numbers.

{joined}

Rank the top 2 candidates by reasoning quality.
Format exactly:
FIRST: <candidate number>
SECOND: <candidate number>
BRIEF: <one sentence why>
"""

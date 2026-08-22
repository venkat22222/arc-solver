"""Stage 1a: prompt template for proposing multiple rule abstractions."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..constraints import ConstraintDict, constraints_to_text
from ..preprocess import describe_train_pairs, should_include_raw

Grid = list  # type alias hint only
TrainPair = Tuple[list, list]

WHOLE_GRID_OPS = (
    "rotate_90, rotate_180, rotate_270, "
    "reflect_horizontal, reflect_vertical, transpose"
)


def is_preserved_geometry(constraints: ConstraintDict) -> bool:
    """True when size, color set, and object count are all preserved."""
    return (
        constraints.get("grid_size") == "preserved"
        and constraints.get("color_set") == "preserved"
        and constraints.get("object_count") == "preserved"
    )


def _hypothesis_slot_instructions(n_hypotheses: int) -> str:
    """Vary framing across slots so hypotheses diversify without extra LLM calls."""
    lines = []
    for i in range(1, n_hypotheses + 1):
        if i <= max(1, n_hypotheses - 1):
            lines.append(
                f"- HYPOTHESIS {i}: Most likely incremental ARC transform that fits."
            )
        else:
            lines.append(
                f"- HYPOTHESIS {i}: Outside-the-box alternative — do NOT rephrase H1."
            )
    return "\n".join(lines)


def build_abstraction_prompt(
    train_pairs: Sequence[TrainPair],
    constraints: ConstraintDict,
    n_hypotheses: int = 3,
    connectivity: int = 4,
    include_raw: Optional[bool] = None,
) -> str:
    if include_raw is None:
        include_raw = should_include_raw(train_pairs, connectivity=connectivity)
    examples = describe_train_pairs(
        train_pairs, connectivity=connectivity, include_raw=include_raw  # type: ignore[arg-type]
    )
    constraint_block = constraints_to_text(constraints, hard_only=True)
    slot_block = _hypothesis_slot_instructions(n_hypotheses)

    geometry_bias = ""
    if is_preserved_geometry(constraints):
        geometry_bias = f"""
IMPORTANT — size, color set, and object count are all preserved:
- HYPOTHESIS 1 MUST name exactly one whole-grid op from: {WHOLE_GRID_OPS}
- Phrase as transforming the ENTIRE grid. No shift/gravity/per-object unless later slots.
"""

    return f"""Abstract reasoning puzzle. Example transformations:

{examples}

Constraints:
{constraint_block}
{geometry_bias}
Propose {n_hypotheses} DIFFERENT hypotheses (plain English, one sentence each).
{slot_block}
Format exactly:
HYPOTHESIS N: <description>
No preamble. No commentary after the list.
"""


def parse_hypotheses(response: str) -> list[str]:
    """Extract hypothesis texts from a model response."""
    import re

    hyps: list[str] = []
    pattern = re.compile(
        r"HYPOTHESIS\s*(\d+)\s*[:.\-]\s*(.+?)(?=HYPOTHESIS\s*\d+\s*[:.\-]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(response):
        text = " ".join(match.group(2).strip().split())
        if text:
            hyps.append(text)
    if not hyps:
        # Fallback: non-empty lines that look like hypotheses
        for line in response.splitlines():
            line = line.strip()
            if line and len(line) > 10:
                hyps.append(line)
    return hyps

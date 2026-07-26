"""Stage 1a: prompt template for proposing multiple rule abstractions."""

from __future__ import annotations

from typing import Sequence, Tuple

from ..constraints import ConstraintDict, constraints_to_text
from ..preprocess import describe_train_pairs

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


def build_abstraction_prompt(
    train_pairs: Sequence[TrainPair],
    constraints: ConstraintDict,
    n_hypotheses: int = 3,
    connectivity: int = 4,
    include_raw: bool = True,
) -> str:
    examples = describe_train_pairs(
        train_pairs, connectivity=connectivity, include_raw=include_raw  # type: ignore[arg-type]
    )
    constraint_block = constraints_to_text(constraints, hard_only=True)

    geometry_bias = ""
    if is_preserved_geometry(constraints):
        geometry_bias = f"""
IMPORTANT — size, color set, and object count are all preserved:
- HYPOTHESIS 1 MUST name exactly one whole-grid geometry op from this closed set:
  {WHOLE_GRID_OPS}
- Phrase it as transforming the ENTIRE grid (e.g. "The entire grid is rotated 180 degrees."
  or "The entire grid is mirrored left-right via reflect_horizontal.").
- Do NOT lead with shift/translate/wrap, per-object rotation, or gravity unless later
  hypotheses explore those after the named whole-grid options.
"""

    return f"""You are looking at an abstract reasoning puzzle. Here are the example transformations:

{examples}

Known constraints these examples satisfy:
{constraint_block}
{geometry_bias}
Propose {n_hypotheses} DIFFERENT possible hypotheses for the transformation rule, in plain English.
Be genuinely different from each other — don't just rephrase the same idea.
Format each as:
HYPOTHESIS N: <one or two sentence description>
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

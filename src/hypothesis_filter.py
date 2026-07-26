"""Filter Stage-1a hypotheses that contradict hard constraints (cheap heuristic)."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

ConstraintDict = Dict[str, str]

# Phrases that imply size change when grid_size is preserved
_SIZE_CHANGE = re.compile(
    r"\b(shrink|resize|resiz(?:e|ing)|downsampl|upsampl|scale\s*(?:up|down)|"
    r"half\s*size|double\s*size|tiled?|tiling|crop(?:ped|ping)?|"
    r"expand(?:ed|ing)?\s*(?:the\s*)?(?:grid|size)|"
    r"reduc(?:e|ing)\s*(?:the\s*)?(?:grid|size|dimensions?)|"
    r"smaller\s+grid|larger\s+grid|1\.5x|2x\s*grid)\b",
    re.IGNORECASE,
)

_COLOR_EXPAND = re.compile(
    r"\b(new\s+colou?r|introduce(?:s|d)?\s+(?:a\s+)?(?:new\s+)?colou?r|"
    r"random\s+colou?r|invent(?:s|ed)?\s+colou?rs?)\b",
    re.IGNORECASE,
)

_OBJECT_COUNT_CHANGE = re.compile(
    r"\b(more\s+objects|fewer\s+objects|add(?:s|ing)?\s+objects?|"
    r"remov(?:e|ing)\s+objects?|split(?:s|ting)?\s+(?:into\s+)?(?:more\s+)?objects|"
    r"merg(?:e|ing)\s+(?:all\s+)?objects)\b",
    re.IGNORECASE,
)


def hypothesis_violates_constraints(hypothesis: str, constraints: ConstraintDict) -> str | None:
    """Return a short reason if hypothesis contradicts a hard constraint, else None."""
    hard = {k: v for k, v in constraints.items() if v != "not_constant"}
    text = hypothesis

    if hard.get("grid_size") == "preserved" and _SIZE_CHANGE.search(text):
        # Allow "crop" only if also says same size? Still risky — reject.
        return "mentions size change while grid_size=preserved"

    if hard.get("color_set") in ("preserved", "subset_or_equal") and _COLOR_EXPAND.search(text):
        return "mentions new/random colors while color_set is preserved/subset"

    if hard.get("object_count") == "preserved" and _OBJECT_COUNT_CHANGE.search(text):
        return "mentions object-count change while object_count=preserved"

    return None


def filter_hypotheses(
    hypotheses: Sequence[str],
    constraints: ConstraintDict,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Keep hypotheses that do not contradict hard constraints.

    Returns (kept, rejected_with_reasons).
    If filtering would remove ALL hypotheses, keep the originals (avoid empty Stage 1a).
    """
    kept: List[str] = []
    rejected: List[Tuple[str, str]] = []
    for h in hypotheses:
        reason = hypothesis_violates_constraints(h, constraints)
        if reason:
            rejected.append((h, reason))
        else:
            kept.append(h)
    if not kept and hypotheses:
        return list(hypotheses), rejected  # fallback: don't starve the pipeline
    return kept, rejected

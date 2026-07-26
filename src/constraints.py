"""Stage 0: extract hard constraints from train pairs.

Only constraints that hold for EVERY train pair are marked as hard.
Inconsistent ones are marked ``not_constant`` and must not be injected
as MUST-SATISFY requirements.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .preprocess import background_color, find_objects, grid_shape

Grid = List[List[int]]
TrainPair = Tuple[Grid, Grid]
ConstraintDict = Dict[str, str]


def _size_relationship(inp: Grid, out: Grid) -> str:
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    if (ih, iw) == (oh, ow):
        return "preserved"
    if ih > 0 and iw > 0 and oh % ih == 0 and ow % iw == 0:
        return f"tiled({oh // ih}x{ow // iw})"
    if oh > 0 and ow > 0 and ih % oh == 0 and iw % ow == 0:
        return f"downsampled({ih // oh}x{iw // ow})"
    return f"resized({ih}x{iw}->{oh}x{ow})"


def grid_size_preserved(train_pairs: Sequence[TrainPair]) -> str:
    rels = {_size_relationship(inp, out) for inp, out in train_pairs}
    if len(rels) == 1:
        return next(iter(rels))
    return "not_constant"


def color_set_preserved(train_pairs: Sequence[TrainPair]) -> str:
    """Check whether output colors are always ⊆ / == input colors."""
    statuses = set()
    for inp, out in train_pairs:
        in_colors = {c for row in inp for c in row}
        out_colors = {c for row in out for c in row}
        if out_colors == in_colors:
            statuses.add("equal")
        elif out_colors <= in_colors:
            statuses.add("subset")
        else:
            statuses.add("expanded")
    if statuses == {"equal"}:
        return "preserved"
    if statuses <= {"equal", "subset"}:
        return "subset_or_equal"
    if len(statuses) == 1:
        return next(iter(statuses))
    return "not_constant"


def object_count_preserved(
    train_pairs: Sequence[TrainPair], connectivity: int = 4
) -> str:
    deltas = set()
    for inp, out in train_pairs:
        n_in = len(find_objects(inp, connectivity=connectivity))  # type: ignore[arg-type]
        n_out = len(find_objects(out, connectivity=connectivity))  # type: ignore[arg-type]
        if n_in == n_out:
            deltas.add("preserved")
        else:
            deltas.add(f"changed({n_in}->{n_out})")
    if deltas == {"preserved"}:
        return "preserved"
    if len(deltas) == 1:
        return next(iter(deltas))
    return "not_constant"


def background_color_unchanged(train_pairs: Sequence[TrainPair]) -> str:
    results = set()
    for inp, out in train_pairs:
        bg_in = background_color(inp)
        bg_out = background_color(out)
        if bg_in == bg_out:
            results.add(f"unchanged ({bg_in})")
        else:
            results.add(f"changed({bg_in}->{bg_out})")
    if len(results) == 1 and next(iter(results)).startswith("unchanged"):
        return next(iter(results))
    return "not_constant"


def extract_constraints(
    train_pairs: Sequence[TrainPair], connectivity: int = 4
) -> ConstraintDict:
    """Return the full constraint dict (including not_constant entries)."""
    return {
        "grid_size": grid_size_preserved(train_pairs),
        "color_set": color_set_preserved(train_pairs),
        "object_count": object_count_preserved(train_pairs, connectivity=connectivity),
        "background_color": background_color_unchanged(train_pairs),
    }


def hard_constraints(constraints: ConstraintDict) -> ConstraintDict:
    """Filter to only constraints that hold consistently."""
    return {k: v for k, v in constraints.items() if v != "not_constant"}


def constraints_to_text(constraints: ConstraintDict, hard_only: bool = True) -> str:
    """Render constraints for prompt injection as a MUST SATISFY list."""
    items = hard_constraints(constraints) if hard_only else constraints
    if not items:
        return "(no hard constraints detected)"
    lines = ["MUST SATISFY:"]
    for key, value in items.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)

"""Cheap structural triage — no LLM — to rank puzzle difficulty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from .brute_force import BruteForceHit, try_brute_force
from .constraints import extract_constraints
from .loader import Puzzle
from .preprocess import find_objects, grid_shape

TriageBucket = Literal["solved", "tractable", "hard", "hopeless"]


@dataclass
class TriageResult:
    puzzle_id: str
    bucket: TriageBucket
    hardness: float  # 0 = easy, 1 = hopeless
    brute_force: Optional[BruteForceHit] = None
    note: str = ""


def structural_hardness(puzzle: Puzzle, connectivity: int = 4) -> float:
    """0..1 score from Stage-0-ish signals (no LLM)."""
    if not puzzle.train_pairs:
        return 1.0
    cons = extract_constraints(puzzle.train_pairs, connectivity=connectivity)
    score = 0.0

    gs = cons.get("grid_size", "")
    if gs == "not_constant":
        score += 0.35
    elif gs.startswith("resized"):
        score += 0.25
    elif gs.startswith("tiled") or gs.startswith("downsampled"):
        score += 0.05

    if cons.get("color_set") in ("expanded", "not_constant"):
        score += 0.15
    if cons.get("object_count", "").startswith("changed") or cons.get("object_count") == "not_constant":
        score += 0.15

    # Size / object load
    max_cells = 0
    max_objs = 0
    for inp, out in puzzle.train_pairs:
        for g in (inp, out):
            h, w = grid_shape(g)
            max_cells = max(max_cells, h * w)
            max_objs = max(max_objs, len(find_objects(g, connectivity=connectivity)))
    if max_cells > 400:
        score += 0.25
    elif max_cells > 100:
        score += 0.1
    if max_objs > 12:
        score += 0.15
    elif max_objs > 6:
        score += 0.05

    return min(1.0, score)


def triage_puzzle(puzzle: Puzzle, connectivity: int = 4) -> TriageResult:
    bf = try_brute_force(puzzle.train_pairs)
    if bf is not None:
        return TriageResult(
            puzzle_id=puzzle.id,
            bucket="solved",
            hardness=0.0,
            brute_force=bf,
            note=f"brute_force:{bf.name}",
        )
    h = structural_hardness(puzzle, connectivity=connectivity)
    if h >= 0.75:
        bucket: TriageBucket = "hopeless"
    elif h >= 0.4:
        bucket = "hard"
    else:
        bucket = "tractable"
    return TriageResult(puzzle_id=puzzle.id, bucket=bucket, hardness=h, note=f"hardness={h:.2f}")


def triage_all(puzzles: Sequence[Puzzle], connectivity: int = 4) -> List[TriageResult]:
    return [triage_puzzle(p, connectivity=connectivity) for p in puzzles]


def order_for_solve(puzzles: Sequence[Puzzle], triage: Sequence[TriageResult]) -> List[Puzzle]:
    """Process solved first (instant), then tractable, hard, hopeless last."""
    by_id = {p.id: p for p in puzzles}
    order = {"solved": 0, "tractable": 1, "hard": 2, "hopeless": 3}
    ranked = sorted(triage, key=lambda t: (order[t.bucket], t.hardness, t.puzzle_id))
    return [by_id[t.puzzle_id] for t in ranked if t.puzzle_id in by_id]


def budget_weight(bucket: TriageBucket) -> float:
    """Relative time share after triage (solved gets near-zero)."""
    return {
        "solved": 0.02,
        "tractable": 1.5,
        "hard": 1.0,
        "hopeless": 0.25,
    }[bucket]

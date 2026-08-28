"""Majority voting over verified candidate output grids.

Given multiple candidate output grids that each independently passed
train-pair verification, pick the most popular one by exact-match voting.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

Grid = List[List[int]]


def _grid_key(grid: Grid) -> str:
    """Hashable string key for a grid (for grouping identical outputs)."""
    return "|".join(",".join(str(c) for c in row) for row in grid)


def majority_vote(grids: List[Grid]) -> Grid:
    """Return the grid that appears most frequently.

    Ties are broken by first-occurrence order (the candidate that was
    generated earliest wins), which favours lower-temperature / higher-
    confidence outputs.
    """
    if not grids:
        raise ValueError("majority_vote called with empty list")
    if len(grids) == 1:
        return grids[0]

    counts: Counter[str] = Counter()
    key_to_grid: dict[str, Grid] = {}
    for g in grids:
        k = _grid_key(g)
        counts[k] += 1
        if k not in key_to_grid:
            key_to_grid[k] = g

    best_key = counts.most_common(1)[0][0]
    return key_to_grid[best_key]


def top_k_by_votes(grids: List[Grid], k: int = 2) -> List[Grid]:
    """Return the top-k most-voted-for distinct grids.

    Used to produce the 2 submission guesses: guess1 = most popular,
    guess2 = second most popular (or same as guess1 if only one cluster).
    """
    if not grids:
        return []

    counts: Counter[str] = Counter()
    key_to_grid: dict[str, Grid] = {}
    # Track insertion order for stable tie-breaking
    order: dict[str, int] = {}
    for idx, g in enumerate(grids):
        gk = _grid_key(g)
        counts[gk] += 1
        if gk not in key_to_grid:
            key_to_grid[gk] = g
            order[gk] = idx

    # Sort by (count desc, first-seen asc) for stable ordering
    ranked = sorted(counts.keys(), key=lambda k: (-counts[k], order.get(k, 0)))
    return [key_to_grid[r] for r in ranked[:k]]


def vote_confidence(grids: List[Grid]) -> Tuple[Grid, float]:
    """Return (winning_grid, confidence) where confidence = votes/total."""
    if not grids:
        raise ValueError("vote_confidence called with empty list")
    winner = majority_vote(grids)
    wk = _grid_key(winner)
    n_votes = sum(1 for g in grids if _grid_key(g) == wk)
    return winner, n_votes / len(grids)

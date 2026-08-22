"""Cheap AST heuristic: flag code that hardcodes train-specific literals."""

from __future__ import annotations

import ast
from typing import List, Sequence, Set, Tuple

Grid = List[List[int]]
TrainPair = Tuple[Grid, Grid]


def _train_literal_pool(train_pairs: Sequence[TrainPair]) -> Set[int]:
    """Colors and dimensions that appear in train pairs."""
    pool: Set[int] = set()
    for inp, out in train_pairs:
        for grid in (inp, out):
            if not grid:
                continue
            pool.add(len(grid))
            pool.add(len(grid[0]))
            for row in grid:
                pool.update(row)
    return pool


def _literal_ints(code: str) -> Set[int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    found: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            found.add(node.value)
    return found


def transfer_risk(
    code: str,
    train_pairs: Sequence[TrainPair] | None = None,
) -> float:
    """0.0 = looks general; 1.0 = many suspicious train-tied literals.

    Heuristic only: counts int literals that are not 0/1 (common) and that
    appear in the train color/size pool (or any literal >= 2 if no train given).
    """
    lits = _literal_ints(code)
    # Ignore ubiquitous constants
    suspicious = {n for n in lits if n not in (0, 1, -1)}
    if not suspicious:
        return 0.0
    if train_pairs:
        pool = _train_literal_pool(train_pairs)
        tied = suspicious & pool
    else:
        tied = suspicious
    # Soft score: fraction of suspicious lits that look train-tied
    return min(1.0, len(tied) / max(1, len(suspicious)))


def transfer_confidence(
    code: str,
    train_pairs: Sequence[TrainPair] | None = None,
) -> float:
    """1.0 = high transfer confidence; inverse of transfer_risk."""
    return 1.0 - transfer_risk(code, train_pairs)

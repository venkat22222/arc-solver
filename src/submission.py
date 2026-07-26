"""Kaggle / ARC Prize submission helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from .loader import Puzzle, _as_grid

Grid = List[List[int]]


def load_challenges_json(path: Union[str, Path]) -> List[Puzzle]:
    """Load ARC Prize challenges file: {task_id: {train, test}, ...}."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    if not isinstance(blob, dict):
        raise ValueError(f"Expected dict of tasks in {path}")

    puzzles: List[Puzzle] = []
    for task_id, data in sorted(blob.items()):
        train_pairs = [
            (_as_grid(pair["input"]), _as_grid(pair["output"])) for pair in data["train"]
        ]
        test_inputs = [_as_grid(pair["input"]) for pair in data["test"]]
        test_outputs: List[Grid] = []
        for pair in data["test"]:
            if "output" in pair:
                test_outputs.append(_as_grid(pair["output"]))
        puzzles.append(
            Puzzle(
                id=str(task_id),
                train_pairs=train_pairs,
                test_inputs=test_inputs,
                test_outputs=test_outputs,
            )
        )
    return puzzles


def attempts_for_task(guesses_per_test: Sequence[Sequence[Grid]]) -> List[Dict[str, Grid]]:
    """Build the per-task list of {attempt_1, attempt_2} dicts."""
    out: List[Dict[str, Grid]] = []
    for guesses in guesses_per_test:
        g = list(guesses)
        while len(g) < 2:
            g.append(g[0] if g else [[0]])
        out.append({"attempt_1": g[0], "attempt_2": g[1]})
    return out


def write_submission(submission: Dict[str, Any], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(submission, f)
    return path


__all__ = [
    "load_challenges_json",
    "attempts_for_task",
    "write_submission",
]

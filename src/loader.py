"""Load ARC-AGI puzzle JSON files into Python objects.

Pure I/O — no puzzle-solving logic here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

Grid = List[List[int]]
TrainPair = Tuple[Grid, Grid]


@dataclass(frozen=True)
class Puzzle:
    """One ARC-AGI task."""

    id: str
    train_pairs: List[TrainPair]
    test_inputs: List[Grid]
    # Public train/eval sets include test outputs; private Kaggle eval does not.
    test_outputs: List[Grid]

    @property
    def n_train(self) -> int:
        return len(self.train_pairs)

    @property
    def n_test(self) -> int:
        return len(self.test_inputs)


def _as_grid(raw) -> Grid:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Grid must be a non-empty nested list")
    grid: Grid = []
    width = None
    for row in raw:
        if not isinstance(row, list) or not row:
            raise ValueError("Each grid row must be a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("Grid rows must have equal length")
        int_row = [int(c) for c in row]
        if any(c < 0 or c > 9 for c in int_row):
            raise ValueError("Grid cells must be integers in 0..9")
        grid.append(int_row)
    return grid


def load_puzzle(path: Union[str, Path]) -> Puzzle:
    """Read a puzzle JSON file and return a Puzzle object."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "train" not in data or "test" not in data:
        raise ValueError(f"Puzzle JSON missing 'train' or 'test': {path}")

    train_pairs: List[TrainPair] = []
    for pair in data["train"]:
        train_pairs.append((_as_grid(pair["input"]), _as_grid(pair["output"])))

    test_inputs: List[Grid] = []
    test_outputs: List[Grid] = []
    for pair in data["test"]:
        test_inputs.append(_as_grid(pair["input"]))
        if "output" in pair:
            test_outputs.append(_as_grid(pair["output"]))

    return Puzzle(
        id=path.stem,
        train_pairs=train_pairs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
    )


def load_puzzles_from_dir(directory: Union[str, Path], limit: int | None = None) -> List[Puzzle]:
    """Load all ``*.json`` puzzles from a directory, sorted by id."""
    directory = Path(directory)
    files = sorted(directory.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    return [load_puzzle(p) for p in files]


def list_puzzle_ids(directory: Union[str, Path]) -> List[str]:
    """Return puzzle ids (filenames without .json) in a directory."""
    directory = Path(directory)
    return sorted(p.stem for p in directory.glob("*.json"))

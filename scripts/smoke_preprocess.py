"""Smoke-print structured descriptions + constraints for sample puzzles."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constraints import constraints_to_text, extract_constraints
from src.loader import load_puzzles_from_dir
from src.preprocess import describe_pair

DATA = ROOT / "data" / "arc-agi-2" / "training"


def main() -> None:
    puzzles = load_puzzles_from_dir(DATA, limit=5)
    for p in puzzles:
        print("=" * 72)
        print(f"PUZZLE {p.id}  (train={p.n_train}, test={p.n_test})")
        print("=" * 72)
        for i, (inp, out) in enumerate(p.train_pairs[:2], start=1):
            print(describe_pair(inp, out, example_index=i, include_raw=True))
            print()
        cons = extract_constraints(p.train_pairs)
        print(constraints_to_text(cons, hard_only=False))
        print("HARD ONLY:")
        print(constraints_to_text(cons, hard_only=True))
        print()


if __name__ == "__main__":
    main()

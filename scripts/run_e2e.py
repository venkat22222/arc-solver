"""End-to-end solve for one (or a few) easy training puzzles.

Usage:
  python -m scripts.run_e2e                     # mock backend (no model)
  python -m scripts.run_e2e --backend ollama    # local Ollama
  python -m scripts.run_e2e --puzzle 6150a2bd --backend ollama
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_client import LLMClient
from src.loader import load_puzzle
from src.pipeline import solve_puzzle

EASY_DEFAULTS = [
    "6150a2bd",  # rotate 180
    "67a3c6ac",  # reflect horizontal
    "ed36ccf7",  # rotate 90
    "3c9b0459",  # rotate 180
    "74dd1130",  # transpose
]


def _grids_equal(a, b) -> bool:
    return a == b


def run_one(puzzle_id: str, client: LLMClient, **kwargs) -> dict:
    path = ROOT / "data" / "arc-agi-2" / "training" / f"{puzzle_id}.json"
    if not path.exists():
        return {"id": puzzle_id, "error": f"missing file {path}"}

    puzzle = load_puzzle(path)
    t0 = time.time()
    guesses = solve_puzzle(puzzle, client, **kwargs)
    elapsed = time.time() - t0

    expected = puzzle.test_outputs[0] if puzzle.test_outputs else None
    hit = False
    if expected is not None:
        hit = any(_grids_equal(g, expected) for g in guesses)

    return {
        "id": puzzle_id,
        "elapsed_s": round(elapsed, 2),
        "n_guesses": len(guesses),
        "hit": hit,
        "expected_available": expected is not None,
        "guess0_shape": (len(guesses[0]), len(guesses[0][0])) if guesses else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", default="mock", choices=["mock", "ollama", "api", "kaggle_local"]
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None, help="Optional config.yaml for kaggle_local extras")
    parser.add_argument("--puzzle", default=None, help="Single puzzle id")
    parser.add_argument("--all-easy", action="store_true", help="Run the easy default set")
    parser.add_argument("--budget", type=float, default=90.0, help="Per-puzzle seconds")
    parser.add_argument("--n-abstractions", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    model = args.model
    if model is None:
        if args.backend == "ollama":
            model = "qwen2.5:1.5b"
        elif args.backend == "kaggle_local":
            model = "Qwen/Qwen3-8B"
        else:
            model = "mock-rotate180"

    extra = {}
    if args.backend == "ollama":
        extra["base_url"] = "http://localhost:11434"
    elif args.backend == "kaggle_local":
        from src.llm_client import load_config

        cfg = load_config(args.config) if args.config else load_config()
        extra.update(dict(cfg.get("kaggle_local") or {}))

    print(f"Backend={args.backend} model={model}")
    client = LLMClient(backend=args.backend, model_name=model, **extra)

    if args.puzzle:
        ids = [args.puzzle]
    elif args.all_easy:
        ids = EASY_DEFAULTS
    else:
        ids = ["6150a2bd"]

    results = []
    for pid in ids:
        print(f"\n--- Solving {pid} ---")
        r = run_one(
            pid,
            client,
            time_budget_seconds=args.budget,
            n_abstractions=args.n_abstractions,
            max_self_debug_retries=args.max_retries,
            n_judges=1,
            early_stop_after_cycles=2,
        )
        results.append(r)
        print(r)

    hits = sum(1 for r in results if r.get("hit"))
    print(f"\nSummary: {hits}/{len(results)} hit expected test output")


if __name__ == "__main__":
    main()

"""Build ARC Prize submission.json from a challenges file."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_client import LLMClient, load_config
from src.pipeline import solve_puzzle
from src.submission import attempts_for_task, load_challenges_json, write_submission


def solve_task_attempts(puzzle, client, **kwargs):
    """Return [{attempt_1, attempt_2}, ...] for every test input."""
    results = []
    for i in range(puzzle.n_test):
        focused = replace(
            puzzle,
            test_inputs=[puzzle.test_inputs[i]],
            test_outputs=(
                [puzzle.test_outputs[i]] if i < len(puzzle.test_outputs) else []
            ),
        )
        results.append(solve_puzzle(focused, client, **kwargs))
    return attempts_for_task(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.kaggle.yaml"))
    ap.add_argument(
        "--challenges",
        default=None,
        help="Path to arc-agi_*_challenges.json (Kaggle or local)",
    )
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget-hours", type=float, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    client = LLMClient.from_config(config)

    challenges = args.challenges
    if challenges is None:
        candidates = [
            Path("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"),
            Path("/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"),
            ROOT / "data" / "arc-agi-2" / "evaluation_challenges.json",
        ]
        challenges = next((str(p) for p in candidates if p.exists()), None)
        if challenges is None:
            raise SystemExit(
                "No challenges JSON found. Pass --challenges or run "
                "scripts/pack_challenges.py"
            )

    puzzles = load_challenges_json(challenges)
    if args.limit:
        puzzles = puzzles[: args.limit]

    hours = (
        args.budget_hours
        if args.budget_hours is not None
        else float(config.get("total_time_budget_hours", 12))
    )
    time_left = hours * 3600.0
    submission = {}
    print(f"Solving {len(puzzles)} tasks | budget={hours}h | model={config.get('model_name')}")

    for i, puzzle in enumerate(puzzles):
        n_left = len(puzzles) - i
        per = time_left / n_left
        t0 = time.time()
        try:
            submission[puzzle.id] = solve_task_attempts(
                puzzle,
                client,
                time_budget_seconds=per,
                n_abstractions=config.get("n_abstractions_per_puzzle", 3),
                max_self_debug_retries=config.get("max_self_debug_retries", 3),
                n_judges=config.get("n_judges", 3),
                early_stop_after_cycles=config.get("early_stop_after_cycles", 3),
                sandbox_timeout=config.get("sandbox_timeout_seconds", 5),
                connectivity=config.get("connectivity", 4),
            )
        except Exception as e:
            print(f"ERROR {puzzle.id}: {e}")
            submission[puzzle.id] = [
                {"attempt_1": ti, "attempt_2": ti} for ti in puzzle.test_inputs
            ]
        elapsed = time.time() - t0
        time_left = max(0.0, time_left - elapsed)
        print(f"[{i+1}/{len(puzzles)}] {puzzle.id} {elapsed:.1f}s left={time_left/3600:.2f}h")

    out = Path(args.out)
    if not out.is_absolute() and Path("/kaggle/working").exists():
        out = Path("/kaggle/working") / out.name
    write_submission(submission, out)
    print(f"Wrote {out} ({len(submission)} tasks)")


if __name__ == "__main__":
    main()

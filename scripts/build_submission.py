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
from src.pipeline import (
    ESTIMATED_TIER2_TIME_SAVED_PER_PUZZLE_S,
    HOPELESS_HARD_GATE,
    _max_effort_for_bucket,
    fallback_guess,
    solve_puzzle,
)
from src.submission import attempts_for_task, load_challenges_json, write_submission
from src.triage import budget_weight, order_for_solve, triage_all


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
    ap.add_argument("--no-triage", action="store_true", help="Disable triage ordering/weights")
    args = ap.parse_args()

    config = load_config(args.config)
    # Topic H: one client for the entire run — never recreate inside the loop.
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
    use_triage = (not args.no_triage) and bool(config.get("triage_enabled", True))
    connectivity = int(config.get("connectivity", 4))

    print(
        f"Solving {len(puzzles)} tasks | budget={hours}h | model={config.get('model_name')} "
        f"| triage={use_triage} | client_once=True"
    )

    triage_map = {}
    if use_triage and puzzles:
        t0 = time.time()
        triage_results = triage_all(puzzles, connectivity=connectivity)
        triage_map = {t.puzzle_id: t for t in triage_results}
        puzzles = order_for_solve(puzzles, triage_results)
        print(
            f"[triage] {time.time()-t0:.1f}s | "
            + ", ".join(
                f"{b}={sum(1 for t in triage_results if t.bucket == b)}"
                for b in ("solved", "tractable", "hard", "hopeless")
            )
        )

    weights = [
        budget_weight(triage_map[p.id].bucket) if p.id in triage_map else 1.0 for p in puzzles
    ]

    solve_base = dict(
        n_abstractions=config.get("n_abstractions_per_puzzle", 3),
        max_self_debug_retries=config.get("max_self_debug_retries", 3),
        n_judges=config.get("n_judges", 3),
        early_stop_after_cycles=config.get("early_stop_after_cycles", 3),
        sandbox_timeout=config.get("sandbox_timeout_seconds", 5),
        connectivity=connectivity,
        partial_escalate_threshold=config.get("partial_escalate_threshold", 0.5),
        close_escalate_threshold=config.get("close_escalate_threshold", 0.8),
        hyp_dedupe_threshold=config.get("hyp_dedupe_threshold", 0.45),
    )

    hard_gated_count = 0
    tier2_plus_count = 0
    hopeless_hard_gate = bool(config.get("hopeless_hard_gate", HOPELESS_HARD_GATE))

    for i, puzzle in enumerate(puzzles):
        n_left = len(puzzles) - i
        if n_left <= 0 or time_left <= 0:
            submission[puzzle.id] = [
                {"attempt_1": ti, "attempt_2": ti} for ti in puzzle.test_inputs
            ]
            continue
        w_left = sum(weights[i:]) or float(n_left)
        per = time_left * (weights[i] / w_left)
        t_res = triage_map.get(puzzle.id)
        bucket = t_res.bucket if t_res else "tractable"
        h_score = t_res.hardness if t_res else 0.5
        effort = _max_effort_for_bucket(bucket, hopeless_hard_gate=hopeless_hard_gate)

        t0 = time.time()
        try:
            submission[puzzle.id] = solve_task_attempts(
                puzzle,
                client,
                time_budget_seconds=per,
                max_effort=effort,
                hopeless_hard_gate=hopeless_hard_gate,
                triage_bucket=bucket,
                structural_hardness_score=h_score,
                **solve_base,
            )
            if bucket == "hopeless" and hopeless_hard_gate:
                hard_gated_count += 1
            elif effort >= 2:
                tier2_plus_count += 1
        except Exception as e:
            print(f"ERROR {puzzle.id}: {e}")
            guesses = fallback_guess(puzzle)
            submission[puzzle.id] = [
                {"attempt_1": guesses[0], "attempt_2": guesses[1]} for _ in puzzle.test_inputs
            ]
        elapsed = time.time() - t0
        time_left = max(0.0, time_left - elapsed)
        print(
            f"[{i+1}/{len(puzzles)}] {puzzle.id} bucket={bucket} effort={effort} "
            f"{elapsed:.1f}s left={time_left/3600:.2f}h"
        )

    out = Path(args.out)
    if not out.is_absolute() and Path("/kaggle/working").exists():
        out = Path("/kaggle/working") / out.name
    write_submission(submission, out)
    print(f"Wrote {out} ({len(submission)} tasks)")

    time_saved_s = hard_gated_count * ESTIMATED_TIER2_TIME_SAVED_PER_PUZZLE_S
    print(
        f"\n=== Run Summary ===\n"
        f"Total tasks processed: {len(puzzles)}\n"
        f"Hard-gated (hopeless, 0 LLM calls): {hard_gated_count}\n"
        f"Sent to Tier 2+ (LLM pipeline): {tier2_plus_count}\n"
        f"Estimated time saved: {time_saved_s:.1f}s ({time_saved_s / 60.0:.2f} min based on ~{ESTIMATED_TIER2_TIME_SAVED_PER_PUZZLE_S:.0f}s/task avg)\n"
        f"===================\n"
    )


if __name__ == "__main__":
    main()

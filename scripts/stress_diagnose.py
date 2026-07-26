"""Stress-diagnose current Ollama model against easy puzzles.

Does not modify pipeline logic — only instruments stage outcomes.
Usage:
  python -m scripts.stress_diagnose --limit 5 --budget 180
  python -m scripts.stress_diagnose --model qwen2.5:3b --limit 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constraints import extract_constraints, constraints_to_text
from src.hypothesis_filter import filter_hypotheses
from src.llm_client import LLMClient
from src.loader import load_puzzle
from src.prompts.abstraction_prompt import build_abstraction_prompt, parse_hypotheses
from src.prompts.code_gen_prompt import build_code_gen_prompt, extract_code
from src.sandbox import safe_execute
from src.self_debug import self_debug_loop

EASY = [
    "6150a2bd",  # rotate 180
    "3c9b0459",  # rotate 180
    "67a3c6ac",  # reflect H
    "74dd1130",  # transpose
    "ed36ccf7",  # rotate 270
    "25ff71a9",  # move object down
    "c8f0f002",  # recolor
    "b1948b0a",  # recolor
    "d631b094",  # extract nonzero
    "5614dbcf",  # downsample 3x3
]


def diagnose_one(puzzle_id: str, client: LLMClient, budget: float, n_hyp: int, retries: int) -> dict:
    path = ROOT / "data" / "arc-agi-2" / "training" / f"{puzzle_id}.json"
    puzzle = load_puzzle(path)
    t0 = time.time()
    report = {
        "id": puzzle_id,
        "n_train": puzzle.n_train,
        "stages": {},
        "failure_point": None,
        "hit": False,
        "any_train_pass": False,
        "any_verified": False,
    }

    # Stage 0
    cons = extract_constraints(puzzle.train_pairs)
    report["stages"]["constraints"] = cons
    report["stages"]["constraints_text"] = constraints_to_text(cons, hard_only=True)

    # Stage 1a
    try:
        prompt = build_abstraction_prompt(puzzle.train_pairs, cons, n_hypotheses=n_hyp)
        raw = client.generate(prompt, max_tokens=512, temperature=0.8)
        hyps = parse_hypotheses(raw)
        hyps, rejected = filter_hypotheses(hyps, cons)
        report["stages"]["abstraction_raw_preview"] = raw[:600]
        report["stages"]["hypotheses"] = hyps
        report["stages"]["rejected_hypotheses"] = [
            {"text": h[:200], "reason": r} for h, r in rejected
        ]
        if not hyps:
            report["failure_point"] = "1a_parse_hypotheses_empty"
            report["elapsed_s"] = round(time.time() - t0, 2)
            return report
    except Exception as e:
        report["failure_point"] = f"1a_llm_error: {e}"
        report["traceback"] = traceback.format_exc()[-800:]
        report["elapsed_s"] = round(time.time() - t0, 2)
        return report

    code_attempts = []
    verified_codes = []

    for i, hyp in enumerate(hyps[:n_hyp]):
        if time.time() - t0 > budget:
            report["failure_point"] = report["failure_point"] or "time_budget"
            break
        attempt = {"hyp_index": i, "hypothesis": hyp, "code": None, "train_results": [], "verified": False}
        try:
            code_raw = client.generate(build_code_gen_prompt(hyp), max_tokens=512, temperature=0.2)
            code = extract_code(code_raw)
            attempt["code_preview"] = code[:400]
            attempt["code"] = code
        except Exception as e:
            attempt["error"] = f"codegen: {e}"
            code_attempts.append(attempt)
            continue

        # Execute on train
        train_ok = 0
        for ti, (inp, expected) in enumerate(puzzle.train_pairs):
            r = safe_execute(code, inp, timeout_seconds=5)
            entry = {
                "train_i": ti,
                "success": r.success,
                "match": bool(r.success and r.output_grid == expected),
                "error": r.error_message,
            }
            attempt["train_results"].append(entry)
            if entry["match"]:
                train_ok += 1
                report["any_train_pass"] = True
        attempt["n_train_match"] = train_ok

        if train_ok == len(puzzle.train_pairs):
            attempt["verified"] = True
            verified_codes.append(code)
            report["any_verified"] = True
        else:
            # Self-debug
            try:
                fixed = self_debug_loop(client, code, puzzle.train_pairs, max_retries=retries, timeout_seconds=5)
                attempt["self_debug_ok"] = fixed is not None
                if fixed:
                    verified_codes.append(fixed)
                    report["any_verified"] = True
                    attempt["verified"] = True
                    attempt["code"] = fixed
                    # re-check
                    train_ok = 0
                    for inp, expected in puzzle.train_pairs:
                        r = safe_execute(fixed, inp, timeout_seconds=5)
                        if r.success and r.output_grid == expected:
                            train_ok += 1
                    attempt["n_train_match_after_debug"] = train_ok
            except Exception as e:
                attempt["self_debug_error"] = str(e)

        code_attempts.append(attempt)

    report["stages"]["code_attempts"] = [
        {k: v for k, v in a.items() if k != "code"} for a in code_attempts
    ]

    if not verified_codes:
        if report["any_train_pass"]:
            report["failure_point"] = "2_partial_train_only_never_fully_verified"
        elif any(a.get("code") for a in code_attempts):
            # classify dominant error
            errs = []
            for a in code_attempts:
                for tr in a.get("train_results", []):
                    if tr.get("error"):
                        errs.append(tr["error"].split(":")[0])
            from collections import Counter
            report["dominant_exec_errors"] = Counter(errs).most_common(5)
            report["failure_point"] = "2_codegen_or_exec_never_matches_train"
        else:
            report["failure_point"] = "1b_no_code_extracted"
        report["elapsed_s"] = round(time.time() - t0, 2)
        return report

    # Apply best verified to test
    code = verified_codes[0]
    r = safe_execute(code, puzzle.test_inputs[0], timeout_seconds=5)
    expected = puzzle.test_outputs[0] if puzzle.test_outputs else None
    report["stages"]["test_exec_success"] = r.success
    report["stages"]["test_match"] = bool(expected is not None and r.output_grid == expected)
    report["hit"] = report["stages"]["test_match"]
    if not report["hit"]:
        report["failure_point"] = "3_verified_on_train_but_wrong_on_test"
    else:
        report["failure_point"] = None  # success
    report["elapsed_s"] = round(time.time() - t0, 2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--budget", type=float, default=180)
    ap.add_argument("--n-hyp", type=int, default=2)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "api", "mock"])
    ap.add_argument("--provider", default=None, help="api provider, e.g. gemini")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--out", default="stress_results.json")
    ap.add_argument("--no-early-stop", action="store_true")
    args = ap.parse_args()

    extra = {}
    if args.backend == "ollama":
        extra["base_url"] = "http://localhost:11434"
    elif args.backend == "api":
        extra["provider"] = args.provider or "gemini"
        extra["api_key_env"] = args.api_key_env

    client = LLMClient(backend=args.backend, model_name=args.model, **extra)
    ids = EASY[: args.limit]
    results = []
    print(
        f"Stress testing {len(ids)} puzzles with backend={args.backend} "
        f"model={args.model} (budget={args.budget}s each)"
    )
    for pid in ids:
        print(f"\n======== {pid} ========")
        r = diagnose_one(pid, client, args.budget, args.n_hyp, args.retries)
        results.append(r)
        print(json.dumps({
            "id": r["id"],
            "elapsed_s": r.get("elapsed_s"),
            "hit": r.get("hit"),
            "any_train_pass": r.get("any_train_pass"),
            "any_verified": r.get("any_verified"),
            "failure_point": r.get("failure_point"),
            "n_hyps": len(r.get("stages", {}).get("hypotheses") or []),
            "n_rejected": len(r.get("stages", {}).get("rejected_hypotheses") or []),
            "dominant_exec_errors": r.get("dominant_exec_errors"),
            "hyp_preview": (r.get("stages", {}).get("hypotheses") or [""])[0][:120],
        }, indent=2))
        # Stop early only if we already have a clear repeated failure pattern AND at least 3 puzzles
        if (not args.no_early_stop) and len(results) >= 3:
            fails = [x.get("failure_point") for x in results if not x.get("hit")]
            if len(fails) == len(results) and len(set(fails)) <= 2:
                print("\nRepeated failure pattern detected — continuing one more for confirmation..." if len(results) < 4 else "\nFailure pattern confirmed.")
                if len(results) >= 4 and all(not x.get("hit") for x in results):
                    break

    hits = sum(1 for r in results if r.get("hit"))
    verified = sum(1 for r in results if r.get("any_verified"))
    partial = sum(1 for r in results if r.get("any_train_pass"))
    summary = {
        "backend": args.backend,
        "model": args.model,
        "n": len(results),
        "hits": hits,
        "any_verified": verified,
        "any_partial_train": partial,
        "failure_points": [r.get("failure_point") for r in results],
    }
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    out = ROOT / args.out
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

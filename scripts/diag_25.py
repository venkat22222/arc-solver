"""25-puzzle diagnostic suite (easy/medium/hard) — no pipeline changes.

Usage:
  set GEMINI_API_KEY=...
  python -m scripts.diag_25 --model gemini-flash-lite-latest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brute_force import try_brute_force
from src.constraints import extract_constraints
from src.hypothesis_filter import filter_hypotheses
from src.llm_client import LLMClient
from src.loader import load_puzzle
from src.prompts.abstraction_prompt import (
    build_abstraction_prompt,
    is_preserved_geometry,
    parse_hypotheses,
)
from src.prompts.code_gen_prompt import build_code_gen_prompt, extract_code
from src.sandbox import safe_execute
from src.self_debug import self_debug_loop

# ---------------------------------------------------------------------------
# Tiered puzzle set (25)
# ---------------------------------------------------------------------------

PUZZLES = [
    # --- easy geometric (10) ---
    {"id": "6150a2bd", "tier": "easy", "note": "rotate 180", "keywords": ["rotate", "180", "point reflection"]},
    {"id": "3c9b0459", "tier": "easy", "note": "rotate 180", "keywords": ["rotate", "180"]},
    {"id": "67a3c6ac", "tier": "easy", "note": "reflect horizontal", "keywords": ["reflect", "mirror", "horizontal", "flip", "left"]},
    {"id": "74dd1130", "tier": "easy", "note": "transpose", "keywords": ["transpose", "swap row"]},
    {"id": "ed36ccf7", "tier": "easy", "note": "rotate 270 / CCW 90", "keywords": ["rotate", "270", "counter", "90"]},
    {"id": "62c24649", "tier": "easy", "note": "geometric symmetry-ish", "keywords": ["mirror", "reflect", "rotate", "symmetry", "tile"]},
    {"id": "6fa7a44f", "tier": "easy", "note": "reflect/flip style", "keywords": ["reflect", "flip", "mirror", "vertical", "horizontal"]},
    {"id": "68b16354", "tier": "easy", "note": "vertical flip style", "keywords": ["reflect", "flip", "vertical", "mirror"]},
    {"id": "9dfd6313", "tier": "easy", "note": "simple remap", "keywords": ["transpose", "rotate", "reflect", "flip"]},
    {"id": "a416b8f3", "tier": "easy", "note": "copy / identity-ish geometric", "keywords": ["copy", "same", "identity", "unchanged", "duplicate"]},
    # --- medium: recolor / objects / gravity (10) ---
    {"id": "c8f0f002", "tier": "medium", "note": "recolor", "keywords": ["recolor", "color", "replace", "swap color"]},
    {"id": "b1948b0a", "tier": "medium", "note": "recolor", "keywords": ["recolor", "color", "replace"]},
    {"id": "d631b094", "tier": "medium", "note": "extract nonzero / crop", "keywords": ["extract", "crop", "non-zero", "nonzero", "remove background"]},
    {"id": "25ff71a9", "tier": "medium", "note": "gravity / move down", "keywords": ["gravity", "fall", "drop", "down", "move"]},
    {"id": "5614dbcf", "tier": "medium", "note": "downsample 3x3 blocks", "keywords": ["downsample", "block", "3x3", "compress", "reduce"]},
    {"id": "5582e5ca", "tier": "medium", "note": "majority / most common color", "keywords": ["majority", "most common", "dominant", "frequent"]},
    {"id": "b94a9452", "tier": "medium", "note": "object crop/recolor style", "keywords": ["crop", "object", "bounding", "recolor", "extract"]},
    {"id": "67385a82", "tier": "medium", "note": "recolor by size/count", "keywords": ["recolor", "size", "count", "large", "small"]},
    {"id": "aedd82e4", "tier": "medium", "note": "object/color interaction", "keywords": ["object", "color", "adjacent", "neighbor"]},
    {"id": "00d62c1b", "tier": "medium", "note": "fill enclosed regions", "keywords": ["fill", "enclosed", "inside", "hollow", "closed"]},
    # --- harder / composite (5) ---
    {"id": "007bbfb7", "tier": "hard", "note": "fractal / scale-up composite", "keywords": ["scale", "tile", "fractal", "repeat", "pattern"]},
    {"id": "0520fde7", "tier": "hard", "note": "composite spatial rule", "keywords": ["quadrant", "split", "combine", "overlay"]},
    {"id": "05f2a901", "tier": "hard", "note": "move objects to markers", "keywords": ["move", "toward", "align", "gravity", "attract"]},
    {"id": "08ed6ac7", "tier": "hard", "note": "rank/recolor by size", "keywords": ["rank", "order", "size", "recolor", "largest"]},
    {"id": "09629e4f", "tier": "hard", "note": "multi-step object rule", "keywords": ["object", "pattern", "multiple", "compose"]},
]


class RetryingClient:
    """Wrap LLMClient with 429 backoff — diagnostic only."""

    def __init__(self, inner: LLMClient, min_interval_s: float = 13.0, max_retries: int = 6):
        self.inner = inner
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._last = 0.0
        self.n_calls = 0
        self.n_retries = 0

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        wait = self.min_interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self._last = time.time()
                self.n_calls += 1
                return self.inner.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    self.n_retries += 1
                    m = re.search(r"retry in ([0-9.]+)s", msg, re.I)
                    sleep_s = float(m.group(1)) + 1.0 if m else min(60.0, 10.0 * (attempt + 1))
                    print(f"  [rate-limit] sleep {sleep_s:.1f}s (attempt {attempt+1})")
                    time.sleep(sleep_s)
                    continue
                raise
        raise RuntimeError(f"API failed after retries: {last_err}")


def _hyp_looks_related(hyps: list[str], keywords: list[str]) -> bool:
    blob = " ".join(hyps).lower()
    return any(k.lower() in blob for k in keywords)


def categorize_miss(report: dict, meta: dict) -> str | None:
    if report.get("hit"):
        return None
    if report.get("any_train_pass") and not report.get("any_verified"):
        return "c_partial_train"
    if report.get("failure_point", "").startswith("1a_"):
        return "a_wrong_hypothesis"

    attempts = report.get("stages", {}).get("code_attempts") or []
    any_clean_wrong = False
    any_exec_success = False
    for a in attempts:
        for tr in a.get("train_results") or []:
            if tr.get("success"):
                any_exec_success = True
                if not tr.get("match"):
                    any_clean_wrong = True

    hyps = report.get("stages", {}).get("hypotheses") or []
    related = _hyp_looks_related(hyps, meta.get("keywords") or [])

    # (b) hyp mentions relevant concepts OR code ran cleanly but wrong
    if related or any_clean_wrong:
        return "b_hyp_okish_code_wrong"
    if any_exec_success:
        return "b_hyp_okish_code_wrong"
    return "a_wrong_hypothesis"


def diagnose_one(meta: dict, client: RetryingClient, budget: float, n_hyp: int, retries: int) -> dict:
    puzzle_id = meta["id"]
    path = ROOT / "data" / "arc-agi-2" / "training" / f"{puzzle_id}.json"
    puzzle = load_puzzle(path)
    t0 = time.time()
    report: dict[str, Any] = {
        "id": puzzle_id,
        "tier": meta["tier"],
        "note": meta["note"],
        "n_train": puzzle.n_train,
        "stages": {},
        "failure_point": None,
        "hit": False,
        "any_train_pass": False,
        "any_verified": False,
        "miss_category": None,
    }

    cons = extract_constraints(puzzle.train_pairs)
    report["stages"]["constraints"] = cons

    # Pre-pipeline brute-force: skip LLM if a library primitive / simple pair fits.
    bf = try_brute_force(puzzle.train_pairs)
    if bf is not None:
        report["stages"]["brute_force"] = {"hit": True, "name": bf.name, "n_ops": bf.n_ops}
        report["stages"]["hypotheses"] = [f"[brute_force] {bf.name}"]
        report["any_verified"] = True
        report["any_train_pass"] = True
        r = safe_execute(bf.code, puzzle.test_inputs[0], timeout_seconds=5)
        expected = puzzle.test_outputs[0] if puzzle.test_outputs else None
        report["stages"]["test_exec_success"] = r.success
        report["stages"]["test_match"] = bool(
            expected is not None and r.output_grid == expected
        )
        report["hit"] = report["stages"]["test_match"]
        report["stages"]["code_attempts"] = [
            {
                "hyp_index": -1,
                "hypothesis": f"[brute_force] {bf.name}",
                "code_preview": bf.code[:350],
                "verified": True,
                "n_train_match": puzzle.n_train,
            }
        ]
        if not report["hit"]:
            report["failure_point"] = "3_verified_on_train_but_wrong_on_test"
            report["miss_category"] = "b_hyp_okish_code_wrong"
        report["elapsed_s"] = round(time.time() - t0, 2)
        return report

    report["stages"]["brute_force"] = {"hit": False}
    prefer_whole_grid = is_preserved_geometry(cons)

    try:
        prompt = build_abstraction_prompt(puzzle.train_pairs, cons, n_hypotheses=n_hyp)
        raw = client.generate(prompt, max_tokens=512, temperature=0.7)
        hyps = parse_hypotheses(raw)
        hyps, rejected = filter_hypotheses(hyps, cons)
        report["stages"]["abstraction_raw_preview"] = raw[:500]
        report["stages"]["hypotheses"] = hyps
        report["stages"]["rejected_hypotheses"] = [{"text": h[:160], "reason": r} for h, r in rejected]
        if not hyps:
            report["failure_point"] = "1a_parse_hypotheses_empty"
            report["miss_category"] = categorize_miss(report, meta)
            report["elapsed_s"] = round(time.time() - t0, 2)
            return report
    except Exception as e:
        report["failure_point"] = f"1a_llm_error: {e}"
        report["traceback"] = traceback.format_exc()[-600:]
        report["miss_category"] = "a_wrong_hypothesis"
        report["elapsed_s"] = round(time.time() - t0, 2)
        return report

    code_attempts = []
    verified_codes = []

    for i, hyp in enumerate(hyps[:n_hyp]):
        if time.time() - t0 > budget:
            report["failure_point"] = report["failure_point"] or "time_budget"
            break
        attempt: dict[str, Any] = {
            "hyp_index": i,
            "hypothesis": hyp,
            "train_results": [],
            "verified": False,
        }
        try:
            code_raw = client.generate(
                build_code_gen_prompt(hyp, prefer_whole_grid=prefer_whole_grid),
                max_tokens=512,
                temperature=0.2,
            )
            code = extract_code(code_raw)
            attempt["code_preview"] = code[:350]
        except Exception as e:
            attempt["error"] = f"codegen: {e}"
            code_attempts.append(attempt)
            continue

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
            try:
                fixed = self_debug_loop(
                    client, code, puzzle.train_pairs, max_retries=retries, timeout_seconds=5
                )
                attempt["self_debug_ok"] = fixed is not None
                if fixed:
                    verified_codes.append(fixed)
                    report["any_verified"] = True
                    attempt["verified"] = True
                    train_ok = 0
                    for inp, expected in puzzle.train_pairs:
                        r = safe_execute(fixed, inp, timeout_seconds=5)
                        if r.success and r.output_grid == expected:
                            train_ok += 1
                            report["any_train_pass"] = True
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
        elif any(a.get("code_preview") for a in code_attempts):
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
        report["miss_category"] = categorize_miss(report, meta)
        report["elapsed_s"] = round(time.time() - t0, 2)
        return report

    code = verified_codes[0]
    r = safe_execute(code, puzzle.test_inputs[0], timeout_seconds=5)
    expected = puzzle.test_outputs[0] if puzzle.test_outputs else None
    report["stages"]["test_exec_success"] = r.success
    report["stages"]["test_match"] = bool(expected is not None and r.output_grid == expected)
    report["hit"] = report["stages"]["test_match"]
    if not report["hit"]:
        report["failure_point"] = "3_verified_on_train_but_wrong_on_test"
        # rare: train OK test fail — treat as (b)
        report["miss_category"] = "b_hyp_okish_code_wrong"
    report["elapsed_s"] = round(time.time() - t0, 2)
    return report


def summarize(results: list[dict]) -> dict:
    by_tier: dict[str, dict] = {}
    for tier in ("easy", "medium", "hard"):
        subset = [r for r in results if r.get("tier") == tier]
        n = len(subset)
        hits = sum(1 for r in subset if r.get("hit"))
        verified = sum(1 for r in subset if r.get("any_verified"))
        cats = {"a_wrong_hypothesis": 0, "b_hyp_okish_code_wrong": 0, "c_partial_train": 0}
        for r in subset:
            if r.get("hit"):
                continue
            c = r.get("miss_category")
            if c in cats:
                cats[c] += 1
            elif c:
                cats[c] = cats.get(c, 0) + 1
        by_tier[tier] = {
            "n": n,
            "hits": hits,
            "hit_rate": f"{hits}/{n}",
            "verified": verified,
            "miss_categories": cats,
        }

    overall_hits = sum(1 for r in results if r.get("hit"))
    overall_verified = sum(1 for r in results if r.get("any_verified"))
    brute_force_hits = sum(
        1 for r in results if ((r.get("stages") or {}).get("brute_force") or {}).get("hit")
    )
    overall_cats = {"a_wrong_hypothesis": 0, "b_hyp_okish_code_wrong": 0, "c_partial_train": 0}
    for r in results:
        if r.get("hit"):
            continue
        c = r.get("miss_category")
        if c in overall_cats:
            overall_cats[c] += 1

    return {
        "n": len(results),
        "hits": overall_hits,
        "hit_rate": f"{overall_hits}/{len(results)}",
        "verified_programs": overall_verified,
        "verified_rate": f"{overall_verified}/{len(results)}",
        "brute_force_solves": brute_force_hits,
        "miss_categories_overall": overall_cats,
        "by_tier": by_tier,
        "per_puzzle": [
            {
                "id": r["id"],
                "tier": r["tier"],
                "note": r.get("note"),
                "hit": r.get("hit"),
                "verified": r.get("any_verified"),
                "partial": r.get("any_train_pass"),
                "brute_force": ((r.get("stages") or {}).get("brute_force") or {}).get("hit"),
                "brute_force_name": ((r.get("stages") or {}).get("brute_force") or {}).get("name"),
                "miss_category": r.get("miss_category"),
                "elapsed_s": r.get("elapsed_s"),
                "hyp_preview": ((r.get("stages") or {}).get("hypotheses") or [""])[0][:120],
            }
            for r in results
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--budget", type=float, default=180)
    ap.add_argument("--n-hyp", type=int, default=2)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--min-interval", type=float, default=13.0, help="seconds between API calls")
    ap.add_argument("--out", default="diag_25_results.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    inner = LLMClient(
        backend="api",
        model_name=args.model,
        provider="gemini",
        api_key_env=args.api_key_env,
    )
    # smoke
    print(f"Smoke {args.model}...")
    smoke = RetryingClient(inner, min_interval_s=1.0, max_retries=4)
    print("SMOKE:", repr(smoke.generate("Reply with exactly: PONG", max_tokens=8, temperature=0)[:40]))

    client = RetryingClient(inner, min_interval_s=args.min_interval, max_retries=8)
    selected = PUZZLES[args.start : args.start + args.limit]
    results = []
    print(f"\nDiagnostic: {len(selected)} puzzles | model={args.model} | interval={args.min_interval}s")

    for i, meta in enumerate(selected, 1):
        print(f"\n======== [{i}/{len(selected)}] {meta['tier'].upper()} {meta['id']} ({meta['note']}) ========")
        r = diagnose_one(meta, client, args.budget, args.n_hyp, args.retries)
        results.append(r)
        print(
            json.dumps(
                {
                    "id": r["id"],
                    "tier": r["tier"],
                    "hit": r["hit"],
                    "verified": r["any_verified"],
                    "partial": r["any_train_pass"],
                    "brute_force": ((r.get("stages") or {}).get("brute_force") or {}).get("hit"),
                    "brute_force_name": ((r.get("stages") or {}).get("brute_force") or {}).get("name"),
                    "miss_category": r["miss_category"],
                    "elapsed_s": r.get("elapsed_s"),
                    "hyp": ((r.get("stages") or {}).get("hypotheses") or [""])[0][:100],
                },
                indent=2,
            )
        )
        # checkpoint after each puzzle
        summary = summarize(results)
        out = {
            "model": args.model,
            "api_calls": client.n_calls,
            "api_retries": client.n_retries,
            "summary": summary,
            "results": results,
        }
        (ROOT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n===== FINAL SUMMARY =====")
    print(json.dumps(summarize(results), indent=2))
    print(f"API calls={client.n_calls} retries={client.n_retries}")
    print(f"Wrote {ROOT / args.out}")


if __name__ == "__main__":
    main()

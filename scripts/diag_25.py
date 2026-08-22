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

from src.constraints import extract_constraints
from src.llm_client import LLMClient
from src.loader import load_puzzle
from src.pipeline import solve_puzzle_detailed
from src.triage import triage_puzzle

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

    def __getattr__(self, name):
        return getattr(self.inner, name)

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
    any_clean_wrong = any(
        (a.get("mean_partial_credit") or 0) > 0 and not a.get("verified") for a in attempts
    )

    hyps = report.get("stages", {}).get("hypotheses") or []
    related = _hyp_looks_related(hyps, meta.get("keywords") or [])

    if related or any_clean_wrong:
        return "b_hyp_okish_code_wrong"
    if attempts:
        return "b_hyp_okish_code_wrong"
    return "a_wrong_hypothesis"


def diagnose_one(meta: dict, client: RetryingClient, budget: float, n_hyp: int, retries: int) -> dict:
    puzzle_id = meta["id"]
    path = ROOT / "data" / "arc-agi-2" / "training" / f"{puzzle_id}.json"
    puzzle = load_puzzle(path)
    t0 = time.time()
    triage = triage_puzzle(puzzle)
    report: dict[str, Any] = {
        "id": puzzle_id,
        "tier": meta["tier"],
        "note": meta["note"],
        "n_train": puzzle.n_train,
        "stages": {
            "triage": {
                "bucket": triage.bucket,
                "hardness": round(triage.hardness, 3),
                "note": triage.note,
            }
        },
        "failure_point": None,
        "hit": False,
        "any_train_pass": False,
        "any_verified": False,
        "miss_category": None,
    }

    cons = extract_constraints(puzzle.train_pairs)
    report["stages"]["constraints"] = cons

    try:
        detail = solve_puzzle_detailed(
            puzzle,
            client,  # type: ignore[arg-type]
            time_budget_seconds=budget,
            n_abstractions=n_hyp,
            max_self_debug_retries=retries,
            n_judges=2,
            early_stop_after_cycles=2,
            sandbox_timeout=5.0,
            max_effort=4,
        )
    except Exception as e:
        report["failure_point"] = f"pipeline_error: {e}"
        report["traceback"] = traceback.format_exc()[-600:]
        report["miss_category"] = "a_wrong_hypothesis"
        report["elapsed_s"] = round(time.time() - t0, 2)
        return report

    report["stages"]["effort_tier"] = detail.tier_reached
    report["stages"]["include_raw"] = detail.include_raw
    report["stages"]["n_hyp_deduped"] = detail.n_hyp_deduped
    report["stages"]["hypotheses"] = detail.hypotheses
    report["stages"]["code_attempts"] = detail.code_attempts
    report["stages"]["best_mean_partial"] = detail.best_mean_partial
    report["any_verified"] = detail.any_verified
    report["any_train_pass"] = detail.any_train_pass
    report["early_stopped"] = detail.early_stopped

    if detail.brute_force_name:
        report["stages"]["brute_force"] = {
            "hit": True,
            "name": detail.brute_force_name,
        }
    else:
        report["stages"]["brute_force"] = {"hit": False}

    expected = puzzle.test_outputs[0] if puzzle.test_outputs else None
    pred = detail.outputs[0] if detail.outputs else None
    report["stages"]["test_match"] = bool(expected is not None and pred == expected)
    report["hit"] = report["stages"]["test_match"]

    if report["hit"]:
        report["elapsed_s"] = round(time.time() - t0, 2)
        return report

    if detail.brute_force_name:
        report["failure_point"] = "3_verified_on_train_but_wrong_on_test"
        report["miss_category"] = "b_hyp_okish_code_wrong"
    elif detail.any_verified:
        report["failure_point"] = "3_verified_on_train_but_wrong_on_test"
        report["miss_category"] = "b_hyp_okish_code_wrong"
    elif detail.any_train_pass:
        report["failure_point"] = "2_partial_train_only_never_fully_verified"
        report["miss_category"] = "c_partial_train"
    elif detail.hypotheses:
        report["failure_point"] = "2_codegen_or_exec_never_matches_train"
        report["miss_category"] = categorize_miss(report, meta)
    else:
        report["failure_point"] = "1a_parse_hypotheses_empty"
        report["miss_category"] = "a_wrong_hypothesis"

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
    close_misses = []
    effort_hist: dict[int, int] = {}
    triage_hist: dict[str, int] = {}
    for r in results:
        et = (r.get("stages") or {}).get("effort_tier")
        if et is not None:
            effort_hist[int(et)] = effort_hist.get(int(et), 0) + 1
        tb = ((r.get("stages") or {}).get("triage") or {}).get("bucket")
        if tb:
            triage_hist[tb] = triage_hist.get(tb, 0) + 1
        if r.get("hit"):
            continue
        c = r.get("miss_category")
        if c in overall_cats:
            overall_cats[c] += 1
        best_pc = (r.get("stages") or {}).get("best_mean_partial") or 0.0
        attempts = (r.get("stages") or {}).get("code_attempts") or []
        if attempts and not best_pc:
            best_pc = max((a.get("mean_partial_credit") or 0.0) for a in attempts)
        close_misses.append(
            {
                "id": r["id"],
                "tier": r.get("tier"),
                "miss_category": c,
                "best_mean_partial_credit": best_pc,
                "effort_tier": et,
                "hypotheses": ((r.get("stages") or {}).get("hypotheses") or [])[:3],
            }
        )
    close_misses.sort(key=lambda x: -x["best_mean_partial_credit"])

    return {
        "n": len(results),
        "hits": overall_hits,
        "hit_rate": f"{overall_hits}/{len(results)}",
        "verified_programs": overall_verified,
        "verified_rate": f"{overall_verified}/{len(results)}",
        "brute_force_solves": brute_force_hits,
        "miss_categories_overall": overall_cats,
        "effort_tier_histogram": effort_hist,
        "triage_histogram": triage_hist,
        "close_misses_by_partial_credit": close_misses[:15],
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
                "effort_tier": ((r.get("stages") or {}).get("effort_tier")),
                "triage_bucket": ((r.get("stages") or {}).get("triage") or {}).get("bucket"),
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
    ap.add_argument("--backend", default="api")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--budget", type=float, default=180)
    ap.add_argument("--n-hyp", type=int, default=2)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--min-interval", type=float, default=13.0, help="seconds between API calls")
    ap.add_argument("--out", default="diag_25_results_alloc.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    inner = LLMClient(
        backend=args.backend,
        model_name=args.model,
        provider="gemini" if args.backend == "api" else None,
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
                    "effort_tier": ((r.get("stages") or {}).get("effort_tier")),
                    "triage": ((r.get("stages") or {}).get("triage") or {}).get("bucket"),
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

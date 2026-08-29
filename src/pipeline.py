"""Main orchestrator — wires all stages into solve_puzzle / solve_all.

Architecture (Phase 1 upgrade):
  1. Brute-force DSL match → instant solve (0.05s)
  2. LLM hypothesis generation (1 call)
  3. Multi-candidate code sampling (N=8 at temp=0.7)
  4. Hard verification: keep only candidates that pass ALL train pairs
  5. Majority voting on test outputs → submit top 2
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .brute_force import try_brute_force
from .constraints import extract_constraints
from .early_stop import EarlyStopState, record_early_stop, should_early_stop
from .judge import Candidate, holistic_judge
from .library import format_library_for_prompt, get_sandbox_helpers
from .llm_client import LLMClient, load_config
from .loader import Puzzle, load_puzzle, load_puzzles_from_dir
from .prompts.abstraction_prompt import (
    build_abstraction_prompt,
    is_preserved_geometry,
    parse_hypotheses,
)
from .prompts.code_gen_prompt import (
    build_code_gen_prompt,
    build_direct_solve_prompt,
    extract_code,
)
from .hypothesis_filter import filter_hypotheses
from .sandbox import safe_execute
from .self_debug import self_debug_loop
from .tiebreak import mdl_score
from .voting import majority_vote, top_k_by_votes
from .augmentation import AUGMENTATIONS, augment_train_pairs, augment_test_inputs

Grid = List[List[int]]


def _identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def fallback_guess(puzzle: Puzzle) -> List[Grid]:
    """Cheap fallbacks when no candidate verifies: identity + first train output shape fill."""
    test_in = puzzle.test_inputs[0]
    guess1 = _identity(test_in)
    # Second guess: if all train outputs share size with input, identity again;
    # else return a blank grid of the first train output's shape filled with bg 0.
    if puzzle.train_pairs:
        _, out0 = puzzle.train_pairs[0]
        oh, ow = len(out0), len(out0[0])
        ih, iw = len(test_in), len(test_in[0])
        if (oh, ow) == (ih, iw):
            guess2 = _identity(test_in)
        else:
            guess2 = [[0 for _ in range(ow)] for _ in range(oh)]
    else:
        guess2 = _identity(test_in)
    return [guess1, guess2]


def apply_to_test(code_str: str, test_input: Grid, timeout_seconds: float = 5.0) -> Optional[Grid]:
    result = safe_execute(code_str, test_input, timeout_seconds=timeout_seconds)
    if result.success:
        return result.output_grid
    return None


def verify_on_all_train(
    code: str,
    train_pairs: Sequence,
    timeout_seconds: float = 5.0,
) -> Tuple[bool, str]:
    """Hard verification: code must produce correct output for EVERY train pair.

    Returns (True, 'Passed') or (False, reason_string).
    """
    if not code or not code.strip():
        return False, "Empty code from LLM"
    if "def solve" not in code:
        return False, "Missing def solve(grid)"
    for pair_i, (inp, expected) in enumerate(train_pairs, 1):
        result = safe_execute(code, inp, timeout_seconds=timeout_seconds)
        if not result.success:
            return False, f"Pair {pair_i} error: {result.error_message}"
        if result.output_grid != expected:
            eh, ew = len(expected), len(expected[0]) if expected else 0
            oh, ow = len(result.output_grid), len(result.output_grid[0]) if result.output_grid else 0
            return False, f"Pair {pair_i} mismatch (got {oh}x{ow} vs expected {eh}x{ew})"
    return True, "Passed all training pairs"


def tiebreak_fill(
    candidates: Sequence[Candidate],
    top: List[Candidate],
    n: int = 2,
) -> List[Candidate]:
    """Ensure exactly n candidates using MDL among leftovers."""
    chosen = list(top)
    seen_codes = {c.code for c in chosen}
    leftovers = sorted(
        (c for c in candidates if c.code not in seen_codes),
        key=lambda c: mdl_score(c.code),
    )
    for c in leftovers:
        chosen.append(c)
        if len(chosen) >= n:
            break
    return chosen[:n]


def solve_puzzle(
    puzzle: Puzzle,
    llm_client: LLMClient,
    time_budget_seconds: float = 120.0,
    n_abstractions: int = 2,
    max_self_debug_retries: int = 3,
    n_candidates: int = 8,
    n_judges: int = 3,
    early_stop_after_cycles: int = 3,
    sandbox_timeout: float = 5.0,
    connectivity: int = 4,
) -> List[Grid]:
    """Solve one puzzle; return exactly 2 candidate output grids for test[0].

    Phase 1 upgrade: multi-candidate sampling + majority voting.
    Instead of generating 1 code and self-debugging, we sample N=8 codes
    and keep all that pass hard verification on ALL train pairs, then
    majority-vote on the test output.
    """
    start = time.time()
    constraints = extract_constraints(puzzle.train_pairs, connectivity=connectivity)

    # ── Stage 0: brute-force DSL match → instant solve ──
    bf = try_brute_force(puzzle.train_pairs)
    if bf is not None:
        test_out = apply_to_test(bf.code, puzzle.test_inputs[0], sandbox_timeout)
        if test_out is None:
            test_out = _identity(puzzle.test_inputs[0])
        return [test_out, test_out]

    library_text = format_library_for_prompt()
    verified_outputs: List[Grid] = []     # test outputs from verified candidates
    verified_codes: List[str] = []        # corresponding code strings
    candidates: List[Candidate] = []      # for fallback to old judge path
    state = EarlyStopState()
    prefer_whole_grid = is_preserved_geometry(constraints)

    # ── Stage 1: Fast Direct Program-of-Thought (PoT) Synthesis ──
    # Top competitors generate direct Python candidates first before doing multi-turn abstractions
    print(f"  [Direct PoT] Sampling {min(n_candidates, 3)} direct candidates...")
    direct_prompt = build_direct_solve_prompt(puzzle.train_pairs, library_text=library_text)
    for sample_i in range(min(n_candidates, 3)):
        if time.time() - start > time_budget_seconds:
            print("  [Direct PoT] Time budget reached.")
            break
        sample_temp = 0.2 if sample_i == 0 else (0.5 + 0.2 * sample_i)
        code_raw = llm_client.generate(direct_prompt, temperature=sample_temp)
        code = extract_code(code_raw)
        ok, reason = verify_on_all_train(code, puzzle.train_pairs, sandbox_timeout)
        if ok:
            print(f"  [Direct Candidate #{sample_i+1}] PASSED all train pairs!")
            state.candidates_verified += 1
            test_out = apply_to_test(code, puzzle.test_inputs[0], sandbox_timeout)
            if test_out is not None:
                verified_outputs.append(test_out)
                verified_codes.append(code)
            candidates.append(
                Candidate(
                    hypothesis="direct program synthesis",
                    code=code,
                    verification_summary="verified on all train pairs",
                    output_grids=[test_out] if test_out else None,
                )
            )
            # If direct candidate verifies, we have a working solution!
            if len(verified_outputs) >= 2:
                break
        else:
            first_line = code.splitlines()[0] if code else "EMPTY"
            print(f"  [Direct Candidate #{sample_i+1}] FAILED: {reason} | head: {first_line[:50]}")

    # If direct synthesis already produced 2+ verified outputs, return immediately!
    if len(verified_outputs) >= 2:
        top_outputs = top_k_by_votes(verified_outputs, k=2)
        outputs: List[Grid] = list(top_outputs)
        while len(outputs) < 2:
            outputs.append(_identity(puzzle.test_inputs[0]))
        return outputs[:2]

    # ── Stage 2: LLM hypothesis generation + candidate sampling ──
    for cycle in range(n_abstractions):
        if time.time() - start > time_budget_seconds:
            break

        state.cycles_attempted += 1
        prompt = build_abstraction_prompt(
            puzzle.train_pairs,
            constraints,
            n_hypotheses=n_abstractions,
            connectivity=connectivity,
        )
        raw = llm_client.generate(prompt, temperature=0.8)
        hypotheses = parse_hypotheses(raw)
        hypotheses, _rejected = filter_hypotheses(hypotheses, constraints)
        if not hypotheses:
            print(f"  [Hypothesis Cycle {cycle+1}] No valid hypotheses generated.")
            continue

        print(f"  [Hypothesis Cycle {cycle+1}] Generated {len(hypotheses)} hypotheses.")
        for hyp_idx, hyp in enumerate(hypotheses, 1):
            if time.time() - start > time_budget_seconds:
                break

            code_prompt = build_code_gen_prompt(
                hyp,
                library_text=library_text,
                prefer_whole_grid=prefer_whole_grid,
                train_pairs=puzzle.train_pairs,
            )

            # ── Multi-candidate sampling: generate N codes, keep verified ──
            unverified_codes: List[str] = []
            for sample_i in range(n_candidates):
                if time.time() - start > time_budget_seconds:
                    break

                # Vary temperature across samples to maximize exploration diversity
                sample_temp = 0.2 if sample_i == 0 else (0.5 + 0.1 * (sample_i % 4))
                code_raw = llm_client.generate(code_prompt, temperature=sample_temp)
                code = extract_code(code_raw)

                ok, reason = verify_on_all_train(code, puzzle.train_pairs, sandbox_timeout)
                if ok:
                    print(f"  [Hyp #{hyp_idx} Sample #{sample_i+1}] PASSED all train pairs!")
                    state.candidates_verified += 1
                    test_out = apply_to_test(code, puzzle.test_inputs[0], sandbox_timeout)
                    if test_out is not None:
                        verified_outputs.append(test_out)
                        verified_codes.append(code)
                    candidates.append(
                        Candidate(
                            hypothesis=hyp,
                            code=code,
                            verification_summary="verified on all train pairs",
                            output_grids=[test_out] if test_out else None,
                        )
                    )
                else:
                    first_line = code.splitlines()[0] if code else "EMPTY"
                    print(f"  [Hyp #{hyp_idx} Sample #{sample_i+1}] FAILED: {reason} | head: {first_line[:50]}")
                    if code:
                        unverified_codes.append(code)

                # Stop early if we have 2+ verified outputs for voting
                if len(verified_outputs) >= 2:
                    break

            # If no candidates verified directly and time remains, try 1 self-debug
            # on the best unverified candidate
            if not verified_outputs and unverified_codes and max_self_debug_retries > 0:
                if (time_budget_seconds - (time.time() - start)) > 20.0:
                    verified_code, _stats = self_debug_loop(
                        llm_client,
                        unverified_codes[0],
                        puzzle.train_pairs,
                        max_retries=1,
                        timeout_seconds=sandbox_timeout,
                    )
                    if verified_code:
                        state.candidates_verified += 1
                        test_out = apply_to_test(
                            verified_code, puzzle.test_inputs[0], sandbox_timeout
                        )
                        if test_out is not None:
                            verified_outputs.append(test_out)
                            verified_codes.append(verified_code)
                        candidates.append(
                            Candidate(
                                hypothesis=hyp,
                                code=verified_code,
                                verification_summary="verified via self-debug",
                                output_grids=[test_out] if test_out else None,
                            )
                        )

            if len(verified_outputs) >= 2:
                break

        if not candidates and should_early_stop(state, early_stop_after_cycles):
            record_early_stop(puzzle.id, state)
            break

    # ── Stage 2: Majority voting on verified outputs ──
    if not verified_outputs:
        return fallback_guess(puzzle)

    # Use majority voting to pick the top 2 most-agreed-upon outputs
    top_outputs = top_k_by_votes(verified_outputs, k=2)
    outputs: List[Grid] = list(top_outputs)

    while len(outputs) < 2:
        outputs.append(_identity(puzzle.test_inputs[0]))
    return outputs[:2]


def solve_with_augmentation(
    puzzle: Puzzle,
    llm_client: LLMClient,
    time_budget_seconds: float = 120.0,
    n_augmentations: int = 4,
    **solve_kwargs,
) -> List[Grid]:
    """Solve a puzzle using test-time augmentation + voting.

    Phase 2: Try solving under multiple geometric transforms (rotations,
    flips, transposes). Un-transform each output and vote across all
    augmentations for the most consistent answer.

    Strategy:
      1. Always try identity (original puzzle) first with full budget.
      2. If identity finds verified candidates, still try a few more
         augmentations to improve voting confidence.
      3. Collect all un-transformed outputs and majority-vote.

    n_augmentations controls how many of the 8 isometries to try.
    Default 4 = identity + rot90 + flip_h + transpose (best diversity).
    """
    start = time.time()

    # Select which augmentations to use (identity is always first)
    augs_to_try = AUGMENTATIONS[:n_augmentations]

    all_outputs: List[Grid] = []

    for aug_name, fwd, inv in augs_to_try:
        remaining_time = time_budget_seconds - (time.time() - start)
        if remaining_time < 5.0:
            break

        # Budget per augmentation: divide remaining time among untried augs
        augs_remaining = n_augmentations - len(all_outputs) // max(1, len(all_outputs) or 1)
        per_aug_budget = remaining_time / max(1, n_augmentations - AUGMENTATIONS.index((aug_name, fwd, inv)))

        # Create augmented puzzle
        aug_train = augment_train_pairs(puzzle.train_pairs, fwd)
        aug_tests = augment_test_inputs(puzzle.test_inputs, fwd)
        aug_test_outputs = [fwd(to) for to in puzzle.test_outputs] if puzzle.test_outputs else []
        aug_puzzle = Puzzle(
            id=f"{puzzle.id}_{aug_name}",
            train_pairs=aug_train,
            test_inputs=aug_tests,
            test_outputs=aug_test_outputs,
        )

        # Solve the augmented puzzle
        aug_results = solve_puzzle(
            aug_puzzle,
            llm_client,
            time_budget_seconds=per_aug_budget,
            **solve_kwargs,
        )

        # Un-transform the outputs back to original orientation
        for grid in aug_results:
            untransformed = inv(grid)
            all_outputs.append(untransformed)

        # If we already have 4+ outputs from multiple augmentations,
        # we have enough diversity for a good vote
        if len(all_outputs) >= 6:
            break

    if not all_outputs:
        return fallback_guess(puzzle)

    # Majority vote across all un-transformed outputs
    top_outputs = top_k_by_votes(all_outputs, k=2)
    outputs: List[Grid] = list(top_outputs)
    while len(outputs) < 2:
        outputs.append(_identity(puzzle.test_inputs[0]))
    return outputs[:2]


def solve_all(
    puzzle_list: Sequence[Puzzle],
    llm_client: LLMClient,
    total_time_budget_seconds: float,
    use_augmentation: bool = True,
    n_augmentations: int = 4,
    **solve_kwargs,
) -> Dict[str, List[Grid]]:
    """Iterate evaluation set with a global time budget; redistribute unused time."""
    remaining = list(puzzle_list)
    results: Dict[str, List[Grid]] = {}
    time_left = total_time_budget_seconds

    for i, puzzle in enumerate(remaining):
        n_left = len(remaining) - i
        per_puzzle = time_left / n_left
        start = time.time()
        if use_augmentation:
            results[puzzle.id] = solve_with_augmentation(
                puzzle, llm_client,
                time_budget_seconds=per_puzzle,
                n_augmentations=n_augmentations,
                **solve_kwargs,
            )
        else:
            results[puzzle.id] = solve_puzzle(
                puzzle, llm_client,
                time_budget_seconds=per_puzzle,
                **solve_kwargs,
            )
        elapsed = time.time() - start
        time_left = max(0.0, time_left - elapsed)
    return results


def solve_from_config(puzzle_path: str | Path, config_path: str | Path | None = None) -> List[Grid]:
    config = load_config(config_path)
    client = LLMClient.from_config(config)
    puzzle = load_puzzle(puzzle_path)
    use_aug = config.get("use_augmentation", True)
    solve_kwargs = dict(
        n_abstractions=config.get("n_abstractions_per_puzzle", 2),
        max_self_debug_retries=config.get("max_self_debug_retries", 1),
        n_candidates=config.get("n_candidates_per_hypothesis", 8),
        n_judges=config.get("n_judges", 3),
        early_stop_after_cycles=config.get("early_stop_after_cycles", 3),
        sandbox_timeout=config.get("sandbox_timeout_seconds", 5),
        connectivity=config.get("connectivity", 4),
    )
    if use_aug:
        return solve_with_augmentation(
            puzzle, client,
            n_augmentations=config.get("n_augmentations", 4),
            **solve_kwargs,
        )
    return solve_puzzle(puzzle, client, **solve_kwargs)


__all__ = [
    "solve_puzzle",
    "solve_with_augmentation",
    "solve_all",
    "solve_from_config",
    "fallback_guess",
    "get_sandbox_helpers",
    "load_puzzles_from_dir",
]

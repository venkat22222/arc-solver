"""Stage 3: holistic trace judging — compare reasoning, not just answers."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .prompts.judge_prompt import build_judge_prompt
from .tiebreak import mdl_score
from .transfer import transfer_risk


@dataclass
class Candidate:
    hypothesis: str
    code: str
    verification_summary: str = "verified on all train pairs"
    candidate_id: Optional[int] = None
    output_grids: Optional[List[List[List[int]]]] = None  # applied to test inputs
    train_pairs: Optional[Sequence] = None  # optional, for transfer heuristic
    mean_partial_credit: Optional[float] = None
    transfer_confidence: Optional[float] = None


def parse_ranked_choice(response: str) -> Tuple[Optional[int], Optional[int]]:
    first = second = None
    m1 = re.search(r"FIRST\s*:\s*(\d+)", response, re.IGNORECASE)
    m2 = re.search(r"SECOND\s*:\s*(\d+)", response, re.IGNORECASE)
    if m1:
        first = int(m1.group(1))
    if m2:
        second = int(m2.group(1))
    return first, second


def weighted_score(votes: Sequence[Tuple[Optional[int], Optional[int]]]) -> dict[int, int]:
    scores: dict[int, int] = defaultdict(int)
    for first, second in votes:
        if first is not None:
            scores[first] += 2
        if second is not None and second != first:
            scores[second] += 1
    return dict(scores)


def _answer_key(candidate: Candidate) -> str:
    """Fingerprint of the predicted test output(s) for distinctness."""
    if candidate.output_grids is None:
        return candidate.code.strip()
    return repr(candidate.output_grids)


def _annotate_transfer(candidate: Candidate) -> float:
    if candidate.transfer_confidence is not None:
        return candidate.transfer_confidence
    risk = transfer_risk(candidate.code, candidate.train_pairs)
    conf = 1.0 - risk
    candidate.transfer_confidence = conf
    return conf


def get_top_distinct_answers(
    scores: dict[int, int],
    candidates: Sequence[Candidate],
    n: int = 2,
) -> List[Candidate]:
    """Pick top-N by judge score, then transfer confidence, then MDL."""
    ranked = sorted(
        scores.items(),
        key=lambda kv: (
            -kv[1],
            -_annotate_transfer(candidates[kv[0] - 1])
            if 0 <= kv[0] - 1 < len(candidates)
            else 0.0,
            kv[0],
        ),
    )
    chosen: List[Candidate] = []
    seen = set()
    for idx, _score in ranked:
        i = idx - 1
        if i < 0 or i >= len(candidates):
            continue
        key = _answer_key(candidates[i])
        if key in seen:
            continue
        seen.add(key)
        chosen.append(candidates[i])
        if len(chosen) >= n:
            break

    if len(chosen) < n:
        remaining = sorted(
            (c for c in candidates if _answer_key(c) not in seen),
            key=lambda c: (-_annotate_transfer(c), mdl_score(c.code)),
        )
        for c in remaining:
            chosen.append(c)
            seen.add(_answer_key(c))
            if len(chosen) >= n:
                break
    return chosen


def holistic_judge(llm_client, candidates: Sequence[Candidate], n_judges: int = 3) -> List[Candidate]:
    if len(candidates) <= 1:
        return list(candidates)
    if len(candidates) == 2:
        return sorted(candidates, key=lambda c: (-_annotate_transfer(c), mdl_score(c.code)))

    prompt = build_judge_prompt(candidates)
    votes = []
    for _ in range(n_judges):
        response = llm_client.generate(prompt, temperature=0.3)
        votes.append(parse_ranked_choice(response))
    scores = weighted_score(votes)
    if not scores:
        return sorted(
            candidates, key=lambda c: (-_annotate_transfer(c), mdl_score(c.code))
        )[:2]
    return get_top_distinct_answers(scores, candidates, n=2)

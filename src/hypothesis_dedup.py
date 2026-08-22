"""Cheap hypothesis deduplication before code-gen."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple


_STOP = frozenset(
    "the a an of to and or by is are be for with from into on in that this then via all one".split()
)

# Collapse near-synonyms so paraphrases collide (Topic B).
_SYNONYMS = {
    "shift": "move",
    "translate": "move",
    "slide": "move",
    "rightward": "right",
    "leftward": "left",
    "upward": "up",
    "downward": "down",
    "mirror": "reflect",
    "flip": "reflect",
    "mirrored": "reflect",
    "rotation": "rotate",
    "rotated": "rotate",
    "recolour": "recolor",
    "replace": "recolor",
    "swap": "recolor",
    "entire": "whole",
    "grid": "grid",
    "objects": "object",
    "cells": "cell",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        if w in _STOP or len(w) <= 1:
            continue
        out.add(_SYNONYMS.get(w, w))
    return out


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def dedupe_hypotheses(
    hypotheses: Sequence[str],
    threshold: float = 0.45,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """Keep first of near-duplicate pairs. Returns (kept, dropped_as_(kept,dup,score))."""
    kept: List[str] = []
    dropped: List[Tuple[str, str, float]] = []
    for hyp in hypotheses:
        twin = None
        score = 0.0
        for k in kept:
            score = jaccard(hyp, k)
            if score >= threshold:
                twin = k
                break
        if twin is not None:
            dropped.append((twin, hyp, score))
        else:
            kept.append(hyp)
    return kept, dropped

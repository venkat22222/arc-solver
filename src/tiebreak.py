"""Stage 4: MDL-style simplicity scoring for tiebreaks.

Lower score = simpler = preferred. Heavily penalizes hardcoded numeric
literals (strong overfitting signal).
"""

from __future__ import annotations

import ast
import re
from typing import Set


def _count_unique_names(tree: ast.AST) -> int:
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return len(names)


def _cyclomatic_complexity(tree: ast.AST) -> int:
    """Simple branch counter: decision points + 1."""
    decisions = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
            decisions += 1
        elif isinstance(node, ast.BoolOp):
            decisions += max(0, len(node.values) - 1)
        elif isinstance(node, ast.comprehension):
            decisions += 1 + len(node.ifs)
    return decisions + 1


def _count_numeric_literals(tree: ast.AST) -> int:
    """Count int/float literals, excluding 0/1/-1 which are often structural."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in (0, 1, -1, True, False):
                count += 1
    return count


def mdl_score(code_str: str) -> float:
    """Lower = simpler = preferred."""
    code_str = code_str.strip()
    # Strip markdown fences if present
    fence = re.match(r"^```(?:python)?\s*\n(.*?)```\s*$", code_str, re.DOTALL | re.IGNORECASE)
    if fence:
        code_str = fence.group(1).strip()

    line_count = len([ln for ln in code_str.splitlines() if ln.strip()])
    try:
        tree = ast.parse(code_str)
        n_vars = _count_unique_names(tree)
        complexity = _cyclomatic_complexity(tree)
        literals = _count_numeric_literals(tree)
    except SyntaxError:
        # Unparseable code is worst
        return 1e9

    # Weighted combination — literals dominate (overfitting signal)
    return (
        1.0 * line_count
        + 0.5 * n_vars
        + 2.0 * complexity
        + 5.0 * literals
    )


def pick_simpler(code_a: str, code_b: str) -> str:
    return code_a if mdl_score(code_a) <= mdl_score(code_b) else code_b

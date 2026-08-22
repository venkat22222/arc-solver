"""Hybrid routing between local LLM (kaggle_local / ollama) and API LLM fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LocalAttemptRecord:
    backend: str = "kaggle_local"
    attempt_index: int = 0
    code: Optional[str] = None
    syntax_valid: bool = True
    unknown_functions: List[str] = field(default_factory=list)
    train_passed: bool = False
    retries_exhausted: bool = False


def evaluate_escalation(
    attempt_history: List[Dict[str, Any] | LocalAttemptRecord],
) -> Tuple[bool, Optional[str]]:
    """Evaluate escalation condition and return (should_escalate, reason_key).
    
    Escalation triggers:
    a. All self-debug retries exhausted on local backend without full train pass.
    b. validate_function_calls (hallucinated-call check) failed on 2+ consecutive retries.
    c. Raw output failed to parse as valid Python on the first attempt.
    """
    if not attempt_history:
        return False, None

    # Normalize to dicts
    records: List[Dict[str, Any]] = []
    for a in attempt_history:
        if isinstance(a, LocalAttemptRecord):
            records.append({
                "backend": a.backend,
                "attempt_index": a.attempt_index,
                "code": a.code,
                "syntax_valid": a.syntax_valid,
                "unknown_functions": a.unknown_functions,
                "train_passed": a.train_passed,
                "retries_exhausted": a.retries_exhausted,
            })
        else:
            records.append(dict(a))

    # Condition c: First attempt failed to parse as valid Python code
    first = records[0]
    is_local_first = first.get("backend") in ("kaggle_local", "ollama", "local")
    if is_local_first:
        code = first.get("code")
        syntax_valid = first.get("syntax_valid", True)
        if not code or not str(code).strip() or not syntax_valid:
            return True, "c_invalid_python_on_first_attempt"

    # Condition b: validate_function_calls failed on 2+ consecutive retries on local backend
    consecutive_unknown = 0
    for r in records:
        if r.get("backend") in ("kaggle_local", "ollama", "local"):
            unknown = r.get("unknown_functions", [])
            if unknown:
                consecutive_unknown += 1
                if consecutive_unknown >= 2:
                    return True, "b_consecutive_hallucinated_calls"
            else:
                consecutive_unknown = 0
        else:
            consecutive_unknown = 0

    # Condition a: All self-debug retries exhausted on local backend without passing train pairs
    for r in records:
        if r.get("backend") in ("kaggle_local", "ollama", "local"):
            if r.get("retries_exhausted", False) and not r.get("train_passed", False):
                return True, "a_retries_exhausted_no_pass"

    return False, None


def should_escalate_to_api(
    attempt_history: List[Dict[str, Any] | LocalAttemptRecord],
) -> bool:
    """Determine whether to escalate code-generation to API fallback."""
    should_esc, _ = evaluate_escalation(attempt_history)
    return should_esc

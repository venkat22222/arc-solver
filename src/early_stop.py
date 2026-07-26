"""Compute-aware early stopping for structurally stuck puzzles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EarlyStopState:
    cycles_attempted: int = 0
    candidates_verified: int = 0
    any_partial_train_pass: bool = False
    marked_unsolvable: bool = False
    log_notes: List[str] = field(default_factory=list)


def should_early_stop(
    state: EarlyStopState,
    early_stop_after_cycles: int = 3,
) -> bool:
    """Stop if N full cycles produced zero verified candidates and no partial passes."""
    if state.candidates_verified > 0:
        return False
    if state.any_partial_train_pass:
        return False
    if state.cycles_attempted >= early_stop_after_cycles:
        state.marked_unsolvable = True
        state.log_notes.append(
            f"Early stop after {state.cycles_attempted} cycles with zero progress "
            "(likely 0-step collapse)."
        )
        return True
    return False


# Puzzles flagged as likely unsolvable by the current approach — review later.
EARLY_STOP_LOG: List[str] = []


def record_early_stop(puzzle_id: str, state: EarlyStopState) -> None:
    note = f"{puzzle_id}: cycles={state.cycles_attempted}; " + "; ".join(state.log_notes)
    EARLY_STOP_LOG.append(note)

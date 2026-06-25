"""
Canned (non-LLM) answers as typed blocks, plus the terminal-turn rule.

The refuse / out-of-scope / emergency short-circuits never call the LLM, but the
transport is uniform NDJSON — so they must emit Blocks too, not raw strings.
These builders are the single source for that canned content; both the async
orchestrator and the legacy sync pipeline render them onto the same wire.

`is_terminal_turn` centralises the (newly introduced) terminal definition. The
spec's `assessment_ready` / `closure_directive` analyzer signals do not exist in
this codebase, so "terminal" is derived from the diagnostic turn cap
(`settings.MAX_DIAGNOSTIC_TURNS`) and an explicit emergency redirect. A terminal
turn must not ask further follow-up questions (enforced again in the validator).
"""

from __future__ import annotations

from typing import Any

from graphrag.config.settings import settings
from graphrag.schemas.blocks import (
    Block,
    NextStepsBlock,
    NextStepsData,
    SummaryBlock,
    SummaryData,
    WarningBlock,
    WarningData,
)

_REFUSAL_TEXT = (
    "I'm designed to assist only with healthcare-related questions. Please ask a "
    "medical or health-related question so I can help."
)

_EMERGENCY_TEXT = (
    "Your symptoms may indicate a serious or life-threatening condition. Do not "
    "wait — seek emergency care now."
)


def refusal_blocks() -> list[Block]:
    """Non-medical request refused. -> [summary]"""
    return [SummaryBlock(type="summary", data=SummaryData(text=_REFUSAL_TEXT))]


def out_of_scope_blocks() -> list[Block]:
    """Out-of-scope (non-health) request. -> [summary]"""
    return [SummaryBlock(type="summary", data=SummaryData(text=_REFUSAL_TEXT))]


def emergency_blocks() -> list[Block]:
    """Emergency redirect. -> [warning(critical), next_steps]"""
    return [
        WarningBlock(
            type="warning",
            data=WarningData(text=_EMERGENCY_TEXT, severity="critical"),
        ),
        NextStepsBlock(
            type="next_steps",
            data=NextStepsData(
                steps=[
                    "Call your local emergency number now (112 / 911 / 108).",
                    "Go to the nearest emergency room or hospital immediately.",
                    "If symptoms worsen while waiting, call back and report the change.",
                ]
            ),
        ),
    ]


def canned_blocks_for(final_action: str) -> list[Block]:
    """Map a gatekeeper short-circuit action to its canned block list."""
    if final_action == "emergency_redirect":
        return emergency_blocks()
    # "refuse" (and any other non-LLM short-circuit) -> refusal/out-of-scope.
    return refusal_blocks()


def is_terminal_turn(*, turn_count: int, analysis: dict[str, Any] | None) -> bool:
    """
    Whether this is a closing turn that must not emit follow-up questions.

    Terminal when the conversation has reached the diagnostic turn cap, or the
    gatekeeper flagged an emergency redirect. (No `assessment_ready` /
    `closure_directive` signal exists in this codebase; revisit if the analyzer
    gains one.)
    """
    if turn_count >= settings.MAX_DIAGNOSTIC_TURNS:
        return True
    if (analysis or {}).get("final_action") == "emergency_redirect":
        return True
    return False

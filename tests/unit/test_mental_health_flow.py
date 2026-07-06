"""
Unit tests for the dedicated mental-health crisis flow (observation 3).

A psychological crisis must NOT reuse the generic physical-emergency response.
It gets its own canned blocks — empathy + validation, crisis helplines, safety
guidance, and imminent-danger escalation — and is treated as a terminal turn.
"""

from __future__ import annotations

from graphrag.domain.messages import (
    canned_blocks_for,
    emergency_blocks,
    is_terminal_turn,
    mental_health_crisis_blocks,
)
from graphrag.schemas.blocks import AnswerResponse


def test_crisis_blocks_lead_with_empathy_then_escalate():
    blocks = mental_health_crisis_blocks()
    types = [b.type for b in blocks]
    # Empathy first, then the imminent-danger warning, then resources.
    assert types == ["summary", "warning", "next_steps"]
    # Validation / non-judgemental tone up front.
    text = blocks[0].data.text.lower()
    assert "alone" in text or "glad you told me" in text
    # Critical severity on the danger warning.
    assert blocks[1].data.severity == "critical"


def test_crisis_blocks_include_helplines_and_safety_steps():
    steps = " ".join(mental_health_crisis_blocks()[2].data.steps).lower()
    # Crisis-specific support lines (not just "go to the ER").
    assert "tele-manas" in steps or "14416" in steps
    assert "1800-599-0019" in steps or "kiran" in steps
    # Imminent-danger escalation still present.
    assert "112" in steps or "emergency room" in steps
    # Safety guidance (reach out / remove means).
    assert "trust" in steps or "harm" in steps


def test_crisis_flow_differs_from_generic_emergency():
    mh = mental_health_crisis_blocks()
    er = emergency_blocks()
    assert [b.type for b in mh] != [b.type for b in er]
    # The generic emergency path has no empathy summary lead.
    assert er[0].type == "warning" and mh[0].type == "summary"


def test_canned_router_maps_mental_health_crisis():
    blocks = canned_blocks_for("mental_health_crisis")
    assert [b.type for b in blocks] == ["summary", "warning", "next_steps"]


def test_crisis_blocks_are_schema_valid():
    # Every canned block must pass the strict wire schema.
    AnswerResponse(blocks=mental_health_crisis_blocks())


def test_mental_health_crisis_is_terminal():
    assert is_terminal_turn(turn_count=0, analysis={"final_action": "mental_health_crisis"}) is True

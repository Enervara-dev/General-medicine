"""
Regression: the information-gathering block plan must tell the model the
`question` block (clickable options) exists.

Root cause of the reported bug: Nova asked about associated symptoms ("a
cough, sore throat, body aches, or changes in your bowel habits") as plain
prose inside a `follow_up_questions` block, so the frontend rendered one text
bubble instead of clickable cards. The `question` block already existed
(graphrag/schemas/blocks.py) and Core's renderer already supported it, but
the GATHERING-turn branch of layer_block_plan — the branch active for almost
every triage clarifying question, including exactly this one — hard-coded its
own "follow_up_questions only" instruction and never mentioned `question` at
all. Every OTHER block plan already used the shared `_FOLLOWUP_LINE` constant,
which does mention it; this branch was the one place that didn't.
"""

from app.services.orchestration.prompt_layers import (
    _FOLLOWUP_LINE,
    compose_system_prompt,
    layer_block_plan,
)


def test_gathering_turn_mentions_the_question_block():
    plan = layer_block_plan(
        query_type="symptom_query", risk_level="none",
        terminal=False, allow_followups=True, consolidate=False,
    )
    assert "- question:" in plan
    assert "PREFER this over follow_up_questions" in plan
    assert "small, known set" in plan


def test_gathering_turn_still_mentions_follow_up_questions_for_open_ended_asks():
    """The fix adds `question`; it must not remove the free-text fallback."""
    plan = layer_block_plan(
        query_type="symptom_query", terminal=False, allow_followups=True, consolidate=False,
    )
    assert "follow_up_questions:" in plan


def test_gathering_turn_reuses_the_shared_followup_line_verbatim():
    """
    Pins the actual fix: the branch now embeds the SAME constant every other
    block plan already uses, rather than a second hand-written copy that can
    drift out of sync with it again.
    """
    plan = layer_block_plan(
        query_type="symptom_query", terminal=False, allow_followups=True, consolidate=False,
    )
    assert _FOLLOWUP_LINE in plan


def test_gathering_turn_other_constraints_are_unchanged():
    """The fix only adds the question-block line; the rest of the gathering
    turn's constraints (no summary/narration, red-flag-only warning) hold."""
    plan = layer_block_plan(
        query_type="symptom_query", terminal=False, allow_followups=True, consolidate=False,
    ).lower()
    assert "information-gathering" in plan
    assert "do not emit a summary" in plan
    assert "warning: only if a red flag" in plan


def test_non_gathering_turns_are_unaffected():
    """Consolidation/terminal turns don't go through the gathering branch at
    all; confirm the fix didn't leak into them."""
    consolidated = layer_block_plan(query_type="symptom_query", consolidate=True)
    assert "information-gathering" not in consolidated.lower()

    terminal = layer_block_plan(query_type="symptom_query", terminal=True, allow_followups=True)
    assert "information-gathering" not in terminal.lower()


def test_composed_blocks_mode_prompt_carries_the_question_block_guidance():
    """End-to-end: the composed system prompt for a live gathering turn
    actually contains the fix, not just the isolated layer function."""
    prompt = compose_system_prompt(
        query_type="symptom_query", risk_level="none", terminal=False,
        allow_followups=True, consolidate=False, output_format="blocks",
    )
    assert "- question:" in prompt
    assert "PREFER this over follow_up_questions" in prompt


def test_critical_risk_gathering_turn_is_unaffected():
    """Critical-risk turns use a separate fixed escalation plan entirely —
    confirm the fix didn't bleed into that branch."""
    plan = layer_block_plan(query_type="symptom_query", risk_level="critical")
    assert "CRITICAL RISK" in plan
    assert "- question:" not in plan

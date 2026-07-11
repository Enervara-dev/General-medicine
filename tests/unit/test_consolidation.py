"""
Unit tests for the gathering→consolidation signal.

Summaries are consolidation checkpoints, not per-turn narration: a triage turn
only summarises once enough distinct clinical facts have accumulated
(``count_clinical_facts`` >= ``CONSOLIDATE_MIN_FACTS``) or the model is confident.
"""

from __future__ import annotations

from Memory_Layer.session_memory import count_clinical_facts
from Memory_Layer.session_memory.models import StructuredState

from app.services.orchestration.prompt_layers import layer_block_plan


# ---------------------------------------------------------------------------
# count_clinical_facts — distinct slot-types, not entries
# ---------------------------------------------------------------------------


def test_empty_state_has_no_facts():
    assert count_clinical_facts(StructuredState()) == 0


def test_multiple_symptoms_count_as_one_slot():
    s = StructuredState(symptoms=["fever", "cough", "chills"])
    assert count_clinical_facts(s) == 1


def test_distinct_slot_types_accumulate():
    s = StructuredState(symptoms=["fever"], duration=["5 days"], drugs=["paracetamol"])
    assert count_clinical_facts(s) == 3


# ---------------------------------------------------------------------------
# Block plan — gathering (no summary) vs consolidate (summary)
# ---------------------------------------------------------------------------


def test_triage_gathering_turn_has_no_summary():
    lo = layer_block_plan(query_type="symptom_query", consolidate=False).lower()
    assert "information-gathering" in lo
    assert "do not emit a summary" in lo
    assert "follow_up_questions" in lo


def test_triage_consolidate_turn_emits_summary():
    out = layer_block_plan(query_type="symptom_query", consolidate=True)
    assert "substantive clinical reply" in out.lower()
    assert "summary" in out


def test_educational_never_gathers_even_without_consolidate():
    # Non-triage intents answer directly; they are never question-only.
    lo = layer_block_plan(query_type="condition_explanation", consolidate=False).lower()
    assert "information-gathering" not in lo
    assert "summary" in lo

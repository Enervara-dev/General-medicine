"""
Complaint label: derivation, grounding, and response-shape wiring.

Feedback #4/#5 (Care Journey rail, Health Timeline): both showed the
patient's full raw first message instead of a short standardised label
("Fever", "Cough & Fever", "Knee Pain"). Root cause traced to Core, not GM —
see app/services/orchestration/pipeline.py's derive_complaint_label docstring
and the implementation report. This file pins the ONE new piece GM needed to
add: a label derived from the gatekeeper's own already-grounded extraction,
exposed identically across all three response shapes so Core can persist one
canonical value and both UI locations read the same field.
"""

import pytest

from app.services.orchestration.pipeline import ChatResult, derive_complaint_label
from graphrag.schemas.blocks import (
    BLOCK_TYPES,
    CONTROL_BLOCK_TYPES,
    MODEL_BLOCK_TYPES,
    ComplaintLabelBlock,
    ComplaintLabelData,
)


# ---------------------------------------------------------------------------
# Derivation — matches the exact examples from the Nova feedback screenshots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "analysis,expected",
    [
        ({"medical_entities": {"symptoms": ["fever"], "conditions": []}}, "Fever"),
        ({"medical_entities": {"symptoms": ["cough", "fever"], "conditions": []}}, "Cough & Fever"),
        ({"medical_entities": {"symptoms": ["knee pain"], "conditions": []}}, "Knee Pain"),
        ({"medical_entities": {"symptoms": [], "conditions": ["asthma"]}}, "Asthma"),
    ],
)
def test_derives_the_exact_labels_from_the_feedback_examples(analysis, expected):
    assert derive_complaint_label(analysis) == expected


def test_symptoms_preferred_over_conditions_when_both_present():
    analysis = {"medical_entities": {"symptoms": ["fever"], "conditions": ["asthma"]}}
    assert derive_complaint_label(analysis) == "Fever"


def test_at_most_two_items_composed():
    analysis = {"medical_entities": {"symptoms": ["fever", "cough", "sore throat", "body ache"]}}
    assert derive_complaint_label(analysis) == "Cough & Fever" or derive_complaint_label(analysis).count("&") == 1


# ---------------------------------------------------------------------------
# Grounding — never invents; degrades to None, never fabricates
# ---------------------------------------------------------------------------

def test_no_entities_yields_none_not_a_fabricated_label():
    assert derive_complaint_label({"medical_entities": {"symptoms": [], "conditions": []}}) is None


def test_missing_medical_entities_key_yields_none():
    assert derive_complaint_label({"intent": "greeting"}) is None


def test_none_analysis_yields_none():
    assert derive_complaint_label(None) is None


def test_empty_analysis_yields_none():
    assert derive_complaint_label({}) is None


def test_error_analysis_yields_none():
    analysis = {"error": "x", "medical_entities": {"symptoms": ["fever"]}}
    assert derive_complaint_label(analysis) is None


def test_non_dict_medical_entities_yields_none():
    assert derive_complaint_label({"medical_entities": "not a dict"}) is None


def test_non_list_symptoms_yields_none():
    assert derive_complaint_label({"medical_entities": {"symptoms": "fever"}}) is None


def test_non_string_items_in_list_are_ignored_not_fatal():
    analysis = {"medical_entities": {"symptoms": [None, 123, "fever", {}]}}
    assert derive_complaint_label(analysis) == "Fever"


# ---------------------------------------------------------------------------
# Graceful handling — long, conversational, empty, multilingual
# ---------------------------------------------------------------------------

def test_a_runaway_full_sentence_is_rejected_not_truncated_mid_word():
    """
    Defends against the gatekeeper ever putting a whole message in `symptoms`
    instead of a concise noun phrase: degrade to None (caller's own
    title-based fallback) rather than show a label cut off mid-word, which
    would read as broken rather than standardised.
    """
    runaway = "im feeling feverish what should i do and also my whole body aches"
    analysis = {"medical_entities": {"symptoms": [runaway]}}
    label = derive_complaint_label(analysis)
    assert label is None


def test_a_short_real_symptom_survives_alongside_a_runaway_candidate():
    analysis = {"medical_entities": {"symptoms": ["fever", "im feeling feverish what should i do"]}}
    assert derive_complaint_label(analysis) == "Fever"


def test_whitespace_only_items_are_ignored():
    analysis = {"medical_entities": {"symptoms": ["   ", "", "fever"]}}
    assert derive_complaint_label(analysis) == "Fever"


def test_multilingual_symptom_text_is_not_rejected_for_being_non_ascii():
    """
    The gatekeeper already classifies/extracts in whatever language the
    patient wrote; this function must not special-case or reject non-ASCII
    text, since "handle multilingual input gracefully" means passing a
    short, genuinely-extracted non-English phrase through just like English.
    """
    analysis = {"medical_entities": {"symptoms": ["fiebre"]}}
    assert derive_complaint_label(analysis) == "Fiebre"


def test_never_raises_on_malformed_input():
    for bad in [123, "a string", [], {"medical_entities": None}, {"medical_entities": {"symptoms": None}}]:
        derive_complaint_label(bad)  # must not raise


# ---------------------------------------------------------------------------
# Block schema — server-injected control block, never model-forgeable
# ---------------------------------------------------------------------------

def test_complaint_label_is_a_control_block_not_model_facing():
    assert "complaint_label" in BLOCK_TYPES
    assert "complaint_label" in CONTROL_BLOCK_TYPES
    assert "complaint_label" not in MODEL_BLOCK_TYPES


def test_block_requires_non_empty_label():
    with pytest.raises(Exception):
        ComplaintLabelData(label="")


def test_block_rejects_extra_fields():
    with pytest.raises(Exception):
        ComplaintLabelData(label="Fever", extra="nope")


def test_block_round_trips():
    block = ComplaintLabelBlock(type="complaint_label", data=ComplaintLabelData(label="Cough & Fever"))
    assert block.model_dump() == {"type": "complaint_label", "data": {"label": "Cough & Fever"}}


# ---------------------------------------------------------------------------
# Response-shape wiring
# ---------------------------------------------------------------------------

def test_chat_result_defaults_to_none():
    r = ChatResult(answer="x", session_id="s", request_id="r")
    assert r.complaint_label is None


def test_chat_response_carries_a_populated_label():
    from app.schemas.chat import ChatResponse

    r = ChatResponse(answer="x", session_id="s", request_id="r", complaint_label="Fever")
    assert r.model_dump()["complaint_label"] == "Fever"


def test_chat_response_defaults_to_none():
    from app.schemas.chat import ChatResponse

    r = ChatResponse(answer="x", session_id="s", request_id="r")
    assert r.complaint_label is None

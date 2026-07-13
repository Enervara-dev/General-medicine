"""
Unit tests for reliable conversation-state tracking.

The regex extractor under-captures free-text symptoms (e.g. "watery eyes"), so
the gatekeeper analyzer's LLM-extracted ``medical_entities`` are merged into the
session state. This keeps state accurate so the model never re-asks and can
consolidate on time.
"""

from __future__ import annotations

from Memory_Layer.session_memory import merge_analysis_entities
from Memory_Layer.session_memory.models import StructuredState


def _analysis(symptoms=None, drugs=None, conditions=None, allergies=None,
              duration=None, severity=None, negated=None):
    return {"medical_entities": {
        "symptoms": symptoms or [], "drugs": drugs or [], "conditions": conditions or [],
        "allergies": allergies or [], "duration": duration or [],
        "severity": severity or [], "negated": negated or [],
    }}


def test_folds_analyzer_symptoms_the_regex_would_miss():
    out = merge_analysis_entities(StructuredState(), _analysis(symptoms=["watery eyes", "sneezing"]))
    assert "watery eyes" in out.symptoms
    assert "sneezing" in out.symptoms


def test_merges_drugs_and_conditions_too():
    out = merge_analysis_entities(
        StructuredState(), _analysis(drugs=["paracetamol"], conditions=["allergic rhinitis"])
    )
    assert "paracetamol" in out.drugs
    assert "allergic rhinitis" in out.conditions


def test_is_idempotent_no_duplicates():
    s = StructuredState(symptoms=["watery eyes"])
    out = merge_analysis_entities(s, _analysis(symptoms=["watery eyes"]))
    assert out.symptoms.count("watery eyes") == 1


def test_noop_when_no_entities():
    s = StructuredState(symptoms=["fever"])
    assert merge_analysis_entities(s, {}).symptoms == ["fever"]
    assert merge_analysis_entities(s, None).symptoms == ["fever"]


def test_captures_duration_and_severity_the_regex_misses():
    # These distinct slots drive the consolidation fact-count.
    out = merge_analysis_entities(
        StructuredState(), _analysis(symptoms=["fever"], duration=["5 days"], severity=["102°f"])
    )
    assert "5 days" in out.duration
    assert "102°f" in out.severity


def test_negation_removes_denied_findings():
    # "no swelling" must drop swelling from state, not add it.
    s = StructuredState(symptoms=["fever", "swelling", "high fever"])
    out = merge_analysis_entities(s, _analysis(negated=["swelling", "fever"]))
    assert "swelling" not in out.symptoms
    # 'fever' and 'high fever' both contain the negated 'fever'.
    assert "fever" not in out.symptoms and "high fever" not in out.symptoms


def test_negation_wins_over_same_turn_affirmation():
    out = merge_analysis_entities(
        StructuredState(), _analysis(symptoms=["swelling"], negated=["swelling"])
    )
    assert "swelling" not in out.symptoms


def test_ignores_malformed_entities():
    # Non-list / junk values must not crash.
    out = merge_analysis_entities(StructuredState(), {"medical_entities": {"symptoms": "not-a-list"}})
    assert out.symptoms == []

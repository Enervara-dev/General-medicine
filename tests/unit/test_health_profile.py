"""
Backend-supplied clinical profile: contract, relevance, rendering, privacy.

The patient's whole active profile arrives on every request; almost none of it
belongs in any given prompt. These tests pin both halves of that: the contract
parses safely whatever the Backend sends, and the relevance rules keep the
profile out of prompts that have no use for it.

Privacy is asserted rather than assumed: profile content must not reach PMS,
episodic memory, the embedding path, or the logs.
"""


import pytest
from pydantic import ValidationError

from app.identity import IdentityContext
from app.schemas.chat import ChatRequest, HealthProfileEnvelope
from app.services.profile import render_health_profile_block, select_relevant_sections
from app.services.profile.relevance import (
    ALLERGIES,
    CONDITIONS,
    LIFESTYLE,
    MEDICATIONS,
    SURGERIES,
    WELLBEING,
)

FULL = {
    "lifestyle": {
        "diet": "vegetarian", "exercise": "sometimes", "alcohol": "no",
        "smoking": "no", "sleep_hours": 6.5, "water_cups": 8,
    },
    "wellbeing": {
        "overall_mood": 6, "stress_level": 8, "energy_level": 4,
        "social_connectedness": 5, "work_academic_pressure": 9,
        "relaxation_practices": "occasionally",
    },
    "allergies": {
        "items": [{
            "allergen_name": "Penicillin",
            "reaction_types": ["rash", "swelling"],
            "severity": "moderate",
            "last_reaction_on": "2024-03-02",
        }],
        "none_confirmed_at": None,
    },
    "conditions": {
        "items": [{
            "condition_code": "type_2_diabetes",
            "display_name": "Type 2 diabetes",
            "category": "metabolic_cardio",
            "since_bucket": "1_5_years",
            "since_exact_date": None,
            "currently_troubling": "sometimes",
            "on_medication": True,
        }],
        "none_confirmed_at": None,
    },
    "medications": [{
        "medication_name": "Metformin",
        "course_type": "ongoing",
        "dose_amount": 500.0,
        "dose_unit": "mg",
        "frequency_count": 2,
        "frequency_period": "per_day",
        "time_of_day": ["morning", "night"],
        "food_relation": "with_food",
        "reason_or_condition": "blood sugar control",
        "linked_condition_id": "11111111-1111-1111-1111-111111111111",
    }],
    "surgeries": [{
        "surgery_name": "Appendectomy",
        "performed_year": 2015,
        "reason": "acute appendicitis",
        "hospital": "General Hospital",
        "current_status": "fully_recovered",
    }],
}


def _p(**over):
    data = {**FULL, **over}
    return HealthProfileEnvelope.model_validate(data)


def _render(profile, intent, query):
    return render_health_profile_block(profile, {"intent": intent}, query)


# ---------------------------------------------------------------------------
# Contract: parsing, partial, empty, backward compatibility
# ---------------------------------------------------------------------------

def test_full_profile_parses():
    p = _p()
    assert p.lifestyle.diet == "vegetarian"
    assert p.wellbeing.relaxation_practices == "occasionally"
    assert p.allergies.items[0].allergen_name == "Penicillin"
    assert p.conditions.items[0].display_name == "Type 2 diabetes"
    assert p.medications[0].medication_name == "Metformin"
    assert p.surgeries[0].performed_year == 2015


def test_relaxation_practices_is_the_field_not_the_enum_type():
    """`relaxation_frequency` is the enum type; the column is relaxation_practices."""
    assert hasattr(_p().wellbeing, "relaxation_practices")
    assert not hasattr(_p().wellbeing, "relaxation_frequency")


def test_partial_profile_parses():
    p = HealthProfileEnvelope.model_validate({"medications": [{"medication_name": "Aspirin"}]})
    assert p.medications[0].medication_name == "Aspirin"
    assert p.lifestyle is None and p.wellbeing is None
    assert p.allergies is None and p.conditions is None


def test_empty_profile_parses_with_safe_collections():
    """Regression: nothing may read `.items`/list length off None."""
    p = HealthProfileEnvelope.model_validate({})
    assert p.medications == [] and p.surgeries == []
    assert len(p.medications) == 0 and len(p.surgeries) == 0
    assert p.allergies is None and p.conditions is None


def test_envelope_sections_default_to_empty_lists_not_none():
    e = HealthProfileEnvelope.model_validate({"allergies": {}, "conditions": {}})
    assert e.allergies.items == [] and e.conditions.items == []
    assert e.allergies.none_confirmed_at is None


def test_unknown_backend_fields_are_ignored_not_fatal():
    p = HealthProfileEnvelope.model_validate({"reproductive_health": {"x": 1}, "medications": []})
    assert not hasattr(p, "reproductive_health")


def test_demographics_only_request_still_works():
    """Backward compatibility: existing callers send no health_profile."""
    r = ChatRequest(query="hi", identity={"patient_id": "p1", "demographics": {"age": 31}})
    assert r.identity.demographics.age == 31
    assert r.identity.health_profile is None


def test_identity_carries_profile_and_omits_when_absent():
    ic = IdentityContext.resolve(
        request_id="r", legacy_session_id="S", legacy_user_id="u",
        envelope_health_profile=FULL, identity_v1_enabled=True,
    )
    assert ic.health_profile["medications"][0]["medication_name"] == "Metformin"
    assert IdentityContext.from_request(
        session_id="S", request_id="r", user_id="u"
    ).health_profile is None


def test_identity_v1_disabled_ignores_envelope_profile():
    ic = IdentityContext.resolve(
        request_id="r", legacy_session_id="S", legacy_user_id="u",
        envelope_health_profile=FULL, identity_v1_enabled=False,
    )
    assert ic.health_profile is None


# ---------------------------------------------------------------------------
# Confirmed absence is not the same as an empty list
# ---------------------------------------------------------------------------

def test_no_known_allergies_confirmed_is_distinct_from_empty():
    empty = _p(allergies={"items": [], "none_confirmed_at": None})
    confirmed = _p(allergies={"items": [], "none_confirmed_at": "2026-01-05T10:00:00Z"})
    assert empty.allergies.none_confirmed_at is None
    assert confirmed.allergies.none_confirmed_at == "2026-01-05T10:00:00Z"

    q = "Do I have any allergies?"
    assert "confirmed no known allergies" in _render(confirmed, "symptom_query", q)
    assert "confirmed no known allergies" not in _render(empty, "symptom_query", q)


def test_no_known_conditions_confirmed_is_distinct_from_empty():
    confirmed = _p(conditions={"items": [], "none_confirmed_at": "2026-01-05T10:00:00Z"})
    out = _render(confirmed, "symptom_query", "Why am I tired?")
    assert "confirmed no known conditions" in out


def test_active_allergy_and_confirmed_absence_cannot_both_be_rendered():
    """A list plus an assertion is contradictory; the list wins and is shown."""
    both = _p(allergies={
        "items": [{"allergen_name": "Penicillin", "reaction_types": [], "severity": "mild"}],
        "none_confirmed_at": "2026-01-05T10:00:00Z",
    })
    out = _render(both, "medication_query", "Can I take this?")
    assert "Penicillin" in out
    assert "confirmed no known allergies" not in out


# ---------------------------------------------------------------------------
# Relevance selection
# ---------------------------------------------------------------------------

def test_generic_educational_question_selects_nothing():
    assert select_relevant_sections(_p(), {"intent": "condition_explanation"}, "What is diabetes?") == set()
    assert _render(_p(), "condition_explanation", "What is diabetes?") == ""


def test_medication_question_pulls_the_interaction_triad():
    got = select_relevant_sections(_p(), {"intent": "medication_query"}, "Can I take ibuprofen?")
    assert {MEDICATIONS, CONDITIONS, ALLERGIES} <= got


def test_alcohol_with_medication_pulls_lifestyle_and_medications():
    got = select_relevant_sections(_p(), {"intent": "unknown"}, "Can I drink alcohol with my medicine?")
    assert LIFESTYLE in got and MEDICATIONS in got


def test_diet_question_pulls_lifestyle():
    assert LIFESTYLE in select_relevant_sections(_p(), {"intent": "unknown"}, "What should I eat?")


def test_tiredness_pulls_wellbeing_and_conditions():
    got = select_relevant_sections(_p(), {"intent": "symptom_query"}, "Why am I feeling tired?")
    assert WELLBEING in got and CONDITIONS in got


def test_direct_surgery_question_pulls_surgeries():
    assert SURGERIES in select_relevant_sections(_p(), {"intent": "unknown"}, "What surgeries have I had?")


def test_direct_allergy_question_pulls_allergies():
    assert ALLERGIES in select_relevant_sections(_p(), {"intent": "unknown"}, "Do I have any allergies?")


def test_symptom_intent_pulls_standing_clinical_picture():
    got = select_relevant_sections(_p(), {"intent": "symptom_query"}, "I have a headache")
    assert CONDITIONS in got and MEDICATIONS in got


def test_none_profile_selects_nothing():
    assert select_relevant_sections(None, {"intent": "medication_query"}, "x") == set()
    assert render_health_profile_block(None, {"intent": "medication_query"}, "x") == ""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_includes_only_selected_sections():
    out = _render(_p(), "unknown", "What surgeries have I had?")
    assert "Appendectomy" in out
    assert "Metformin" not in out and "Penicillin" not in out


def test_render_formats_medication_compactly():
    out = _render(_p(), "medication_query", "Can I take this?")
    assert "Metformin 500 mg 2x per day morning/night with food (ongoing)" in out
    assert "for blood sugar control" in out


def test_render_omits_unpopulated_fields():
    sparse = _p(medications=[{"medication_name": "Aspirin"}])
    out = _render(sparse, "medication_query", "Can I take this?")
    assert "- Aspirin" in out
    assert "None" not in out


def test_render_is_not_raw_json():
    out = _render(_p(), "medication_query", "Can I take this?")
    assert "{" not in out and "'medication_name'" not in out


def test_render_is_deterministic():
    a = _render(_p(), "medication_query", "Can I take this?")
    b = _render(_p(), "medication_query", "Can I take this?")
    assert a == b


def test_medication_condition_link_is_one_directional():
    """GM carries linked_condition_id and never derives the reverse."""
    p = _p()
    assert p.medications[0].linked_condition_id
    assert not hasattr(p.conditions.items[0], "linked_medication_id")


def test_reproductive_health_is_not_part_of_the_contract():
    assert not hasattr(HealthProfileEnvelope, "reproductive_health")
    assert "reproductive" not in HealthProfileEnvelope.model_fields


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"medications": "not-a-list"},
    {"lifestyle": {"sleep_hours": "abc"}},
    {"conditions": {"items": [{}]}},
])
def test_malformed_profile_raises_validation_not_crash(bad):
    with pytest.raises(ValidationError):
        HealthProfileEnvelope.model_validate(bad)


def test_pipeline_helper_fails_open_on_malformed_profile(caplog):
    """The turn must proceed with no block rather than raising."""
    import logging

    from app.services.orchestration.pipeline import AsyncOrchestrator

    orch = AsyncOrchestrator.__new__(AsyncOrchestrator)
    ident = IdentityContext.from_request(
        session_id="S", request_id="r", user_id="u",
        health_profile={"medications": "not-a-list"},
    )
    caplog.set_level(logging.WARNING)
    assert orch._health_profile_block(ident, {"intent": "medication_query"}, "x") == ""
    assert "Health profile unusable" in caplog.text


def test_pipeline_helper_returns_empty_when_no_profile():
    from app.services.orchestration.pipeline import AsyncOrchestrator

    orch = AsyncOrchestrator.__new__(AsyncOrchestrator)
    ident = IdentityContext.from_request(session_id="S", request_id="r", user_id="u")
    assert orch._health_profile_block(ident, {"intent": "medication_query"}, "x") == ""


# ---------------------------------------------------------------------------
# Privacy: the profile must not cross any persistence boundary
# ---------------------------------------------------------------------------

def test_profile_fields_are_not_on_the_pms_wire_contract():
    from app.services.pms.events import PmsMemoryEventV1

    fields = set(PmsMemoryEventV1.model_fields)
    for leaked in ("health_profile", "lifestyle", "wellbeing", "allergies",
                   "medications", "conditions", "surgeries", "demographics"):
        assert leaked not in fields, leaked


def test_profile_fields_are_not_on_the_episode_schema():
    from episodic.schemas.episode import Episode, EpisodeCandidate

    for model in (Episode, EpisodeCandidate):
        fields = set(model.model_fields)
        for leaked in ("health_profile", "lifestyle", "wellbeing", "allergies",
                       "medications", "surgeries", "demographics"):
            assert leaked not in fields, f"{model.__name__}.{leaked}"


def test_pms_producer_does_not_read_health_profile():
    """The producer maps named fields only; profile is not among them."""
    import inspect

    from app.services.pms import producer

    src = inspect.getsource(producer)
    assert "health_profile" not in src
    assert "medications" not in src or "entities" in src  # entity lists are clinical extraction, not profile


def test_episodic_extractor_does_not_read_health_profile():
    import inspect

    from episodic.services import extractor

    assert "health_profile" not in inspect.getsource(extractor)


def test_rendered_block_is_not_logged(caplog):
    """Rendering must not emit profile content into the log stream."""
    import logging

    caplog.set_level(logging.DEBUG)
    out = _render(_p(), "medication_query", "Can I take this?")
    assert "Metformin" in out          # it was rendered
    assert "Metformin" not in caplog.text
    assert "Penicillin" not in caplog.text
    assert "Appendectomy" not in caplog.text


def test_pipeline_helper_logs_no_profile_content(caplog):
    import logging

    from app.services.orchestration.pipeline import AsyncOrchestrator

    orch = AsyncOrchestrator.__new__(AsyncOrchestrator)
    ident = IdentityContext.from_request(
        session_id="S", request_id="r", user_id="u",
        health_profile={"medications": [{"medication_name": "SecretDrug"}]},
    )
    caplog.set_level(logging.DEBUG)
    orch._health_profile_block(ident, {"intent": "medication_query"}, "Can I take this?")
    assert "SecretDrug" not in caplog.text

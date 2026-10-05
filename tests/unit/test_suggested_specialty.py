"""
Cross-specialty suggestion: validation, gating, and response-shape wiring.

The gatekeeper's raw `suggested_specialty` claim is untrusted model output.
`_extract_suggested_specialty` is the one place that decides whether it is fit
to reach a client — these tests pin every rejection path as well as the one
acceptance path, and confirm the signal reaches all three response shapes
(JSON, SSE done event, blocks control block) without being model-forgeable.
"""

import pytest

from app.services.orchestration.pipeline import (
    SUGGESTED_SPECIALTY_MIN_CONFIDENCE,
    _extract_suggested_specialty,
)
from graphrag.schemas.blocks import (
    BLOCK_TYPES,
    CONTROL_BLOCK_TYPES,
    MODEL_BLOCK_TYPES,
    SuggestedSpecialtyBlock,
    SuggestedSpecialtyData,
)

VALID = {
    "suggested_specialty": {
        "slug": "dermatology",
        "confidence": 0.92,
        "reason_code": "skin_condition",
        "display_message": "This sounds like a skin-related concern.",
    }
}


def _analysis(**over):
    base = {"intent": "symptom_query", **VALID}
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

def test_valid_suggestion_from_a_different_specialty_is_accepted():
    out = _extract_suggested_specialty(_analysis(), current_specialty="ent")
    assert out is not None
    assert out.slug == "dermatology"
    assert out.confidence == 0.92
    assert out.reason_code == "skin_condition"
    assert out.display_message == "This sounds like a skin-related concern."


def test_slug_is_normalised():
    raw = _analysis(suggested_specialty={**VALID["suggested_specialty"], "slug": "  Dermatology "})
    out = _extract_suggested_specialty(raw, current_specialty="ent")
    assert out.slug == "dermatology"


def test_confidence_exactly_at_threshold_is_accepted():
    raw = _analysis(suggested_specialty={**VALID["suggested_specialty"], "confidence": SUGGESTED_SPECIALTY_MIN_CONFIDENCE})
    assert _extract_suggested_specialty(raw, current_specialty="ent") is not None


# ---------------------------------------------------------------------------
# Rejection — must degrade to None, never raise
# ---------------------------------------------------------------------------

def test_none_analysis_yields_none():
    assert _extract_suggested_specialty(None, current_specialty="ent") is None


def test_empty_analysis_yields_none():
    assert _extract_suggested_specialty({}, current_specialty="ent") is None


def test_error_analysis_yields_none():
    assert _extract_suggested_specialty({"error": "x", **VALID}, current_specialty="ent") is None


def test_missing_key_yields_none():
    assert _extract_suggested_specialty({"intent": "symptom_query"}, current_specialty="ent") is None


def test_non_dict_value_yields_none():
    for bad in ("dermatology", 123, None, ["dermatology"]):
        assert _extract_suggested_specialty({"suggested_specialty": bad}, current_specialty="ent") is None


def test_self_suggestion_is_rejected():
    """Suggesting the specialty that is already answering is meaningless."""
    out = _extract_suggested_specialty(_analysis(), current_specialty="dermatology")
    assert out is None


def test_unknown_specialty_slug_is_rejected():
    raw = _analysis(suggested_specialty={**VALID["suggested_specialty"], "slug": "neurology"})
    assert _extract_suggested_specialty(raw, current_specialty="ent") is None


def test_empty_slug_is_rejected():
    raw = _analysis(suggested_specialty={**VALID["suggested_specialty"], "slug": "  "})
    assert _extract_suggested_specialty(raw, current_specialty="ent") is None


@pytest.mark.parametrize("bad_confidence", [0.84, 0.0, -0.1, 1.5, "0.9", True, None])
def test_low_or_invalid_confidence_is_rejected(bad_confidence):
    raw = _analysis(suggested_specialty={**VALID["suggested_specialty"], "confidence": bad_confidence})
    assert _extract_suggested_specialty(raw, current_specialty="ent") is None


def test_missing_confidence_is_rejected():
    raw = _analysis(suggested_specialty={"slug": "dermatology"})
    assert _extract_suggested_specialty(raw, current_specialty="ent") is None


def test_reason_code_and_message_are_optional():
    raw = _analysis(suggested_specialty={"slug": "dermatology", "confidence": 0.9})
    out = _extract_suggested_specialty(raw, current_specialty="ent")
    assert out is not None
    assert out.reason_code == ""
    assert out.display_message == ""


def test_overlong_message_is_truncated_not_rejected():
    raw = _analysis(suggested_specialty={
        "slug": "dermatology", "confidence": 0.9,
        "reason_code": "x" * 500, "display_message": "y" * 500,
    })
    out = _extract_suggested_specialty(raw, current_specialty="ent")
    assert out is not None
    assert len(out.reason_code) <= 60
    assert len(out.display_message) <= 200


def test_malformed_field_types_degrade_to_none_not_raise():
    raw = _analysis(suggested_specialty={"slug": "dermatology", "confidence": 0.9,
                                          "reason_code": 12345, "display_message": {"x": 1}})
    out = _extract_suggested_specialty(raw, current_specialty="ent")
    # Non-string optional fields fall back to "" rather than raising.
    assert out is not None
    assert out.reason_code == "" and out.display_message == ""


# ---------------------------------------------------------------------------
# Block schema: server-injected control block, never model-forgeable
# ---------------------------------------------------------------------------

def test_suggested_specialty_is_a_control_block_not_model_facing():
    assert "suggested_specialty" in BLOCK_TYPES
    assert "suggested_specialty" in CONTROL_BLOCK_TYPES
    assert "suggested_specialty" not in MODEL_BLOCK_TYPES


def test_block_validates_confidence_range():
    with pytest.raises(Exception):
        SuggestedSpecialtyData(slug="dermatology", confidence=1.5)


def test_block_rejects_extra_fields():
    with pytest.raises(Exception):
        SuggestedSpecialtyData(slug="dermatology", confidence=0.9, extra_field="nope")


def test_block_round_trips():
    data = SuggestedSpecialtyData(slug="dermatology", confidence=0.92,
                                   reason_code="skin_condition", display_message="msg")
    block = SuggestedSpecialtyBlock(type="suggested_specialty", data=data)
    dumped = block.model_dump()
    assert dumped == {
        "type": "suggested_specialty",
        "data": {"slug": "dermatology", "confidence": 0.92,
                  "reason_code": "skin_condition", "display_message": "msg"},
    }


# ---------------------------------------------------------------------------
# Response-shape wiring
# ---------------------------------------------------------------------------

def test_chat_response_defaults_to_none():
    from app.schemas.chat import ChatResponse

    r = ChatResponse(answer="x", session_id="s", request_id="r")
    assert r.suggested_specialty is None


def test_chat_response_carries_a_populated_suggestion():
    from app.schemas.chat import ChatResponse

    data = SuggestedSpecialtyData(slug="dermatology", confidence=0.92)
    r = ChatResponse(answer="x", session_id="s", request_id="r", suggested_specialty=data)
    assert r.model_dump()["suggested_specialty"]["slug"] == "dermatology"


def test_chat_result_defaults_to_none():
    from app.services.orchestration.pipeline import ChatResult

    r = ChatResult(answer="x", session_id="s", request_id="r")
    assert r.suggested_specialty is None


# ---------------------------------------------------------------------------
# Prompt contract: the 6 non-GM specialties carry the field; GM does not
# ---------------------------------------------------------------------------

def test_six_specialties_advertise_the_field_general_medicine_does_not():
    from app.specialty.registry import build_default_registry

    reg = build_default_registry()
    for key in ("cardiology", "dermatology", "ent", "ophthalmology", "orthopaedics", "pulmonology"):
        assert "suggested_specialty" in reg.get(key).gatekeeper_system_prompt, key

    assert "suggested_specialty" not in reg.get("general_medicine").gatekeeper_system_prompt


def test_every_specialty_never_offers_itself_in_its_own_instruction():
    """Each prompt's instruction text names every OTHER specialty but warns
    against suggesting its own — spot-check the self-exclusion sentence."""
    from app.specialty.registry import build_default_registry

    reg = build_default_registry()
    for key in ("cardiology", "dermatology", "ent", "ophthalmology", "orthopaedics", "pulmonology"):
        prompt = reg.get(key).gatekeeper_system_prompt
        assert f"Never suggest your own specialty ({key})" in prompt

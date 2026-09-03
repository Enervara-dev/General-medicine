"""
Regression tests for the extractor → EpisodeCandidate entity contract.

The extraction LLM emits a bare entity as a string but upgrades the same slot to
an object ({"name": "fever", "value": "101F"}) once it carries a qualifier.
``EpisodeEntities`` declares ``list[str]``, so the object form failed validation
and the extractor dropped the ENTIRE candidate — costing the turn its episode and
the downstream PMS memory event.

These tests pin the normalisation, and carry the fix through to the canonical PMS
event and the PMS client so the whole chain is covered, not just the schema.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.identity import IdentityContext
from app.services.pms import ClinicalMemoryProducer, PmsMemoryEventV1
from episodic.schemas.entity_normalization import coerce_entity, normalize_entity_list
from episodic.schemas.episode import (
    ClinicalPriority,
    Episode,
    EpisodeCandidate,
    EpisodeCategory,
    Severity,
)

# The exact payload observed failing in the deployed /chat/blocks flow.
PRODUCTION_FAILURE = [{"name": "fever", "value": "101F"}, {"name": "weakness"}]


def _candidate(**entities) -> EpisodeCandidate:
    return EpisodeCandidate.model_validate(
        {
            "user_id": "u123",
            "summary": "Fever and weakness for 5 days",
            "category": "symptom",
            "embedding_text": "fever 101F weakness 5 days",
            "confidence": 0.9,
            "entities": entities,
        }
    )


class _RecordingPMSClient:
    def __init__(self):
        self.events: list[PmsMemoryEventV1] = []
        self.assertions: list[str | None] = []

    async def ingest_clinical_memory(self, event, *, user_assertion=None, request_id=None):
        self.events.append(event)
        self.assertions.append(user_assertion)


# ---------------------------------------------------------------------------
# 1. A symptom with an optional value/attribute
# ---------------------------------------------------------------------------

def test_symptom_object_with_value_is_normalized():
    c = _candidate(symptoms=[{"name": "fever", "value": "101F"}])
    assert c.entities.symptoms == ["fever 101F"]


def test_value_attribute_is_not_discarded():
    """The qualifier must survive — folding it in is the whole point."""
    c = _candidate(symptoms=[{"name": "fever", "value": "101F"}])
    assert "101F" in c.entities.symptoms[0]


# ---------------------------------------------------------------------------
# 2. A plain symptom
# ---------------------------------------------------------------------------

def test_plain_symptom_object_is_normalized():
    c = _candidate(symptoms=[{"name": "weakness"}])
    assert c.entities.symptoms == ["weakness"]


def test_plain_string_symptom_still_works():
    """Backward compatibility: the original shape must be untouched."""
    c = _candidate(symptoms=["weakness", "nausea"])
    assert c.entities.symptoms == ["weakness", "nausea"]


# ---------------------------------------------------------------------------
# 3. Multiple mixed representations in one extraction result
# ---------------------------------------------------------------------------

def test_production_failure_payload_now_validates():
    c = _candidate(symptoms=PRODUCTION_FAILURE)
    assert c.entities.symptoms == ["fever 101F", "weakness"]


def test_mixed_strings_and_objects_in_one_list():
    c = _candidate(
        symptoms=[
            "headache",
            {"name": "fever", "value": "101F"},
            {"name": "weakness"},
            {"name": "cough", "severity": "mild"},
        ]
    )
    assert c.entities.symptoms == ["headache", "fever 101F", "weakness", "cough mild"]


def test_every_entity_list_is_normalized_not_just_symptoms():
    """The fix is generic across the entity vocabulary, not symptom-specific."""
    c = _candidate(
        symptoms=[{"name": "fever", "value": "101F"}],
        conditions=[{"name": "diabetes", "status": "type 2"}],
        medications=[{"name": "paracetamol", "dose": "500mg"}],
        labs=[{"label": "BP", "reading": "140/90"}],
        body_parts=[{"name": "chest"}],
    )
    assert c.entities.symptoms == ["fever 101F"]
    assert c.entities.conditions == ["diabetes type 2"]
    assert c.entities.medications == ["paracetamol 500mg"]
    assert c.entities.labs == ["BP 140/90"]
    assert c.entities.body_parts == ["chest"]


def test_no_condition_specific_hardcoding():
    """An entity the code has never seen normalises the same way."""
    c = _candidate(symptoms=[{"name": "photophobia", "value": "severe"}])
    assert c.entities.symptoms == ["photophobia severe"]


# ---------------------------------------------------------------------------
# 4. Valid normalized EpisodeCandidate output
# ---------------------------------------------------------------------------

def test_candidate_is_valid_and_fully_typed():
    c = _candidate(symptoms=PRODUCTION_FAILURE)
    assert isinstance(c, EpisodeCandidate)
    assert c.category is EpisodeCategory.SYMPTOM
    assert all(isinstance(s, str) for s in c.entities.symptoms)
    assert c.store_memory is True


def test_structured_entities_are_preserved_in_metadata():
    """Nothing is silently discarded: the raw objects are kept internally."""
    c = _candidate(symptoms=PRODUCTION_FAILURE)
    assert c.metadata["entities_raw"]["symptoms"] == PRODUCTION_FAILURE


def test_metadata_untouched_when_entities_were_already_strings():
    c = _candidate(symptoms=["weakness"])
    assert "entities_raw" not in c.metadata


def test_existing_metadata_is_not_clobbered():
    c = EpisodeCandidate.model_validate(
        {
            "user_id": "u1",
            "summary": "s",
            "category": "symptom",
            "embedding_text": "x",
            "metadata": {"origin": "unit-test"},
            "entities": {"symptoms": [{"name": "fever", "value": "101F"}]},
        }
    )
    assert c.metadata["origin"] == "unit-test"
    assert "entities_raw" in c.metadata


def test_episode_roundtrip_revalidates_cleanly():
    """Episode.from_candidate re-validates; normalisation must be idempotent."""
    c = _candidate(symptoms=PRODUCTION_FAILURE)
    ep = Episode.from_candidate(c)
    assert ep.entities.symptoms == ["fever 101F", "weakness"]


# ---------------------------------------------------------------------------
# 5. The candidate still produces the canonical PMS memory event
# ---------------------------------------------------------------------------

def _episode_from(symptoms) -> Episode:
    c = _candidate(symptoms=symptoms)
    ep = Episode.from_candidate(c)
    return ep.model_copy(
        update={
            "severity": Severity.MODERATE,
            "clinical_priority": ClinicalPriority.MEDIUM,
            "timestamp": datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc),
        }
    )


async def test_normalized_candidate_produces_canonical_pms_event():
    from app.services.pms import SourceChannel

    client = _RecordingPMSClient()
    ic = IdentityContext.from_request(
        session_id="S1", request_id="R1", user_id="u123", user_assertion="assert.jwt"
    )
    await ClinicalMemoryProducer(client).emit_from_episode(
        identity=ic, episode=_episode_from(PRODUCTION_FAILURE)
    )

    assert len(client.events) == 1
    ev = client.events[0]
    assert isinstance(ev, PmsMemoryEventV1)
    assert ev.schema_version == "pms.memory_event/v1"
    # The qualifier reached the wire intact, as plain strings the contract allows.
    assert ev.entities.symptoms == ("fever 101F", "weakness")
    assert all(isinstance(s, str) for s in ev.entities.symptoms)
    assert ev.source.channel is SourceChannel.PATIENT_CONVERSATION
    # Contract still forbids a body-supplied patient.
    assert "patient_id" not in ev.model_dump()


async def test_event_reaches_the_pms_client_with_the_assertion():
    """The whole point: a validated extraction must arrive at the PMS client."""
    client = _RecordingPMSClient()
    ic = IdentityContext.from_request(
        session_id="S1", request_id="R1", user_id="u123", user_assertion="assert.jwt"
    )
    await ClinicalMemoryProducer(client).emit_from_episode(
        identity=ic, episode=_episode_from(PRODUCTION_FAILURE)
    )

    assert client.assertions == ["assert.jwt"]


def test_normalized_event_is_still_idempotency_stable():
    from app.services.pms import SourceChannel

    ic = IdentityContext.from_request(
        session_id="S1", request_id="R1", user_id="u123", user_assertion="a"
    )
    ep = _episode_from(PRODUCTION_FAILURE)
    a = ClinicalMemoryProducer._to_event(
        identity=ic, episode=ep, channel=SourceChannel.PATIENT_CONVERSATION
    )
    b = ClinicalMemoryProducer._to_event(
        identity=ic, episode=ep, channel=SourceChannel.PATIENT_CONVERSATION
    )
    assert a.idempotency_key() == b.idempotency_key()


# ---------------------------------------------------------------------------
# Normaliser robustness — it runs inside validation on the chat path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "item,expected",
    [
        ("fever", "fever"),
        ("  fever  ", "fever"),
        ({"name": "fever"}, "fever"),
        ({"name": "fever", "value": "101F"}, "fever 101F"),
        ({"name": "fever", "value": ""}, "fever"),
        ({"name": "fever", "value": None}, "fever"),
        ({"label": "BP", "reading": "140/90"}, "BP 140/90"),
        ({"text": "chills"}, "chills"),
        ({"value": "malaise"}, "malaise"),          # no name key at all
        ({"name": "fever", "id": "x1"}, "fever"),   # structural keys dropped
        ({"name": "fever", "value": "fever"}, "fever"),  # qualifier restates name
        ({}, ""),
        ({"name": ""}, ""),
        (None, ""),
        (42, "42"),
    ],
)
def test_coerce_entity_shapes(item, expected):
    assert coerce_entity(item) == expected


def test_normalize_drops_empties_and_dedupes_case_insensitively():
    assert normalize_entity_list(
        ["fever", "", None, {"name": ""}, "Fever", {"name": "fever"}]
    ) == ["fever"]


def test_normalize_handles_none_and_bare_entity():
    assert normalize_entity_list(None) == []
    assert normalize_entity_list("fever") == ["fever"]
    assert normalize_entity_list({"name": "fever"}) == ["fever"]


def test_normalize_flattens_one_level_of_nesting():
    assert normalize_entity_list([["fever"], [{"name": "cough"}]]) == ["fever", "cough"]


def test_malformed_input_is_left_for_pydantic_to_reject():
    """The normaliser must not mask a genuinely malformed payload."""
    with pytest.raises(ValidationError):
        _candidate(symptoms=123)


def test_deterministic_output_regardless_of_key_order():
    a = coerce_entity({"name": "cough", "severity": "mild", "site": "chest"})
    b = coerce_entity({"site": "chest", "severity": "mild", "name": "cough"})
    assert a == b


# ---------------------------------------------------------------------------
# The extractor boundary itself — where the production failure was logged
# ---------------------------------------------------------------------------

async def test_extractor_accepts_structured_symptoms(monkeypatch, caplog):
    """
    End-to-end at the failing boundary: the extractor previously logged
    "Extractor output failed validation" and returned None for this payload,
    which is why no episode — and therefore no PMS event — was ever produced.
    """
    import json
    import logging

    from episodic.services import extractor as extractor_mod

    payload = {
        "user_id": "ignored-by-design",
        "summary": "Fever and weakness for 5 days",
        "category": "symptom",
        "entities": {"symptoms": PRODUCTION_FAILURE, "body_parts": []},
        "temporal_data": {"duration": "5 days"},
        "severity": "moderate",
        "clinical_priority": "medium",
        "confidence": 0.9,
        "embedding_text": "fever 101F weakness 5 days",
        "store_memory": True,
    }

    async def fake_llm(*_a, **_kw):
        return json.dumps(payload)

    monkeypatch.setattr(extractor_mod, "generate_text_async", fake_llm)
    caplog.set_level(logging.WARNING)

    candidate = await extractor_mod.ExtractorService().extract(
        user_id="u123", utterance="I have had a fever of 101F and weakness for 5 days"
    )

    assert candidate is not None, "extraction must no longer be dropped"
    assert candidate.entities.symptoms == ["fever 101F", "weakness"]
    assert candidate.user_id == "u123"  # pinned from input, not model output
    assert "failed validation" not in caplog.text

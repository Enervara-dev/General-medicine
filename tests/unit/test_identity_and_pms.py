"""Tests for IdentityContext / PatientId and the PMS clinical-memory producer."""

from datetime import datetime, timezone

import pytest

from app.identity import IdentityContext, PatientId
from app.services.pms import (
    PmsMemoryEventV1,
    ClinicalMemoryProducer,
    NullPMSClient,
    PMSClient,
)


# ---------------------------------------------------------------------------
# PatientId — wraps the authenticated Mongo User._id, never mints one
# ---------------------------------------------------------------------------

# GM treats the assertion as opaque, so tests need no real JWT here.
ASSERTION = "backend.minted.assertion"


def test_patient_id_from_user_id():
    assert PatientId.from_user_id("507f1f77bcf86cd799439011").value == "507f1f77bcf86cd799439011"
    assert PatientId.from_user_id("  abc ").value == "abc"  # trimmed


def test_patient_id_none_and_empty_are_anonymous():
    assert PatientId.from_user_id(None) is None
    assert PatientId.from_user_id("") is None
    assert PatientId.from_user_id("   ") is None


def test_patient_id_rejects_empty_direct_construction():
    with pytest.raises(ValueError):
        PatientId(value="")


# ---------------------------------------------------------------------------
# IdentityContext
# ---------------------------------------------------------------------------

def test_identity_from_request_maps_user_id_to_patient_id():
    ic = IdentityContext.from_request(session_id="S1", request_id="R1", user_id="u123")
    assert ic.patient_id == PatientId(value="u123")
    assert ic.user_id == "u123"          # leaf-boundary string for existing stores
    assert ic.is_identified is True
    assert ic.session_id == "S1" and ic.request_id == "R1"


def test_identity_anonymous_preserved():
    ic = IdentityContext.from_request(session_id="S1", request_id="R1", user_id=None)
    assert ic.patient_id is None
    assert ic.user_id is None
    assert ic.is_identified is False


# ---------------------------------------------------------------------------
# Fakes + fixtures
# ---------------------------------------------------------------------------

class _RecordingPMSClient:
    def __init__(self):
        self.events: list[PmsMemoryEventV1] = []

    async def ingest_clinical_memory(
        self, event: PmsMemoryEventV1, *, user_assertion: str | None = None,
        request_id: str = "-"
    ) -> None:
        self.events.append(event)


class _FailingPMSClient:
    async def ingest_clinical_memory(
        self, event: PmsMemoryEventV1, *, user_assertion: str | None = None,
        request_id: str = "-"
    ) -> None:
        raise RuntimeError("PMS down")


def _episode():
    from episodic.schemas.episode import (
        Episode,
        EpisodeCategory,
        EpisodeEntities,
        Severity,
        TemporalData,
        ClinicalPriority,
    )
    return Episode(
        user_id="u123",
        summary="Fever for 5 days, 102F",
        category=EpisodeCategory.SYMPTOM,
        entities=EpisodeEntities(symptoms=["fever"], medications=["paracetamol"]),
        temporal_data=TemporalData(duration="5 days"),
        severity=Severity.MODERATE,
        clinical_priority=ClinicalPriority.MEDIUM,
        confidence=0.9,
        embedding_text="fever 5 days 102F",
        timestamp=datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Producer — the integration point after clinical extraction
# ---------------------------------------------------------------------------

async def test_producer_emits_event_from_episode():
    from app.services.pms import (
        ClinicalCategory,
        ClinicalPriority,
        ClinicalSeverity,
        SourceChannel,
    )

    client = _RecordingPMSClient()
    ic = IdentityContext.from_request(session_id="S1", request_id="R1", user_id="u123")
    await ClinicalMemoryProducer(client).emit_from_episode(identity=ic, episode=_episode())

    assert len(client.events) == 1
    ev = client.events[0]
    assert ev.schema_version == "pms.memory_event/v1"
    # Identity is NOT on the wire: it travels in the verified assertion.
    assert "patient_id" not in ev.model_dump()
    assert ev.conversation_id == "S1"
    assert ev.source.service == "general-medicine"
    # Contract vocabulary — NOT episodic values / subsystem names.
    assert ev.source.channel is SourceChannel.PATIENT_CONVERSATION
    assert ev.category is ClinicalCategory.SYMPTOM
    assert ev.severity is ClinicalSeverity.MODERATE
    assert ev.priority is ClinicalPriority.MEDIUM
    assert ev.summary.startswith("Fever")
    # Typed, contract-owned payload models (not dumped storage models).
    assert ev.entities.symptoms == ("fever",)
    assert ev.entities.medications == ("paracetamol",)
    assert ev.timing.duration == "5 days"
    assert ev.confidence == pytest.approx(0.9)
    # occurred_at is the episode's domain time; NO storage identifier is present.
    # `occurred_at` is a real datetime on the canonical contract, not an ISO string.
    assert ev.occurred_at is not None
    assert ev.occurred_at.isoformat().startswith("2026-07-25T10:00")
    assert not hasattr(ev, "source_episode_id")
    assert not hasattr(ev, "episode_id")


async def test_producer_noop_for_anonymous_identity():
    client = _RecordingPMSClient()
    ic = IdentityContext.from_request(session_id="S1", request_id="R1", user_id=None)
    await ClinicalMemoryProducer(client).emit_from_episode(identity=ic, episode=_episode())
    assert client.events == []  # no patient → nothing emitted


async def test_producer_is_fail_open_on_client_error():
    ic = IdentityContext.from_request(session_id="S1", request_id="R1", user_id="u123")
    # Must NOT raise even though the client throws.
    await ClinicalMemoryProducer(_FailingPMSClient()).emit_from_episode(
        identity=ic, episode=_episode()
    )


async def test_null_pms_client_is_a_noop_and_satisfies_protocol():
    assert isinstance(NullPMSClient(), PMSClient)
    # Returns without error and sends nothing.
    ic = IdentityContext.from_request(session_id="S1", request_id="R1", user_id="u123")
    await ClinicalMemoryProducer(NullPMSClient()).emit_from_episode(
        identity=ic, episode=_episode()
    )

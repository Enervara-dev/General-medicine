"""
Tests that PMS's SourceRef.service identity is resolved per-specialty rather
than permanently hardcoded to "general-medicine" (AUDIT_REPORT.md §5/§12).

The existing test_identity_and_pms.py::test_producer_emits_event_from_episode
already pins the DEFAULT ("general-medicine" when no specialty is resolved);
these tests cover the new, specialty-scoped behavior on top of it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.identity import IdentityContext
from app.services.pms import SPECIALTY_SERVICE_DEFAULT, ClinicalMemoryProducer
from app.specialty import build_default_registry


class _RecordingPMSClient:
    def __init__(self):
        self.events = []

    async def ingest_clinical_memory(self, event, *, user_assertion=None, request_id="-"):
        self.events.append(event)


def _episode():
    from episodic.schemas.episode import (
        ClinicalPriority,
        Episode,
        EpisodeCategory,
        EpisodeEntities,
        Severity,
        TemporalData,
    )

    return Episode(
        user_id="u123",
        summary="Chest pain for 2 days",
        category=EpisodeCategory.SYMPTOM,
        entities=EpisodeEntities(symptoms=["chest pain"]),
        temporal_data=TemporalData(duration="2 days"),
        severity=Severity.MODERATE,
        clinical_priority=ClinicalPriority.MEDIUM,
        confidence=0.8,
        embedding_text="chest pain 2 days",
        timestamp=datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc),
    )


async def test_default_source_service_is_general_medicine():
    assert SPECIALTY_SERVICE_DEFAULT == "general-medicine"
    client = _RecordingPMSClient()
    ic = IdentityContext.from_request(session_id="S1", request_id="R1", user_id="u123")
    await ClinicalMemoryProducer(client).emit_from_episode(identity=ic, episode=_episode())
    assert client.events[0].source.service == "general-medicine"


async def test_resolved_specialty_source_service_reaches_pms_event():
    registry = build_default_registry()
    client = _RecordingPMSClient()
    ic = IdentityContext.from_request(session_id="S1", request_id="R1", user_id="u123")
    cardiology = registry.get("cardiology")

    await ClinicalMemoryProducer(client).emit_from_episode(
        identity=ic, episode=_episode(), source_service=cardiology.source_service
    )

    assert client.events[0].source.service == "cardiology"


@pytest.mark.parametrize(
    "key,expected", [
        ("cardiology", "cardiology"),
        ("dermatology", "dermatology"),
        ("ent", "ent"),
        ("ophthalmology", "ophthalmology"),
        ("orthopaedics", "orthopaedics"),
        ("pulmonology", "pulmonology"),
        ("general_medicine", "general-medicine"),
    ],
)
async def test_each_specialty_maps_to_its_own_pms_identity(key: str, expected: str):
    registry = build_default_registry()
    client = _RecordingPMSClient()
    ic = IdentityContext.from_request(session_id="S1", request_id="R1", user_id="u123")
    cfg = registry.get(key)

    await ClinicalMemoryProducer(client).emit_from_episode(
        identity=ic, episode=_episode(), source_service=cfg.source_service
    )

    assert client.events[0].source.service == expected

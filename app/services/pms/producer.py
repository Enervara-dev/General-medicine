"""
ClinicalMemoryProducer — the ONLY translation seam between internal episodic
storage and the outbound PMS contract (an anti-corruption layer).

It is the single place allowed to know BOTH sides. The outbound
``ClinicalMemoryEventV1`` never inherits episodic enums, identifiers, models, or
vocabulary — this module maps them EXPLICITLY, with safe defaults, so that:

    * a new episodic enum value maps to a canonical fallback (OTHER/UNKNOWN/
      MEDIUM) rather than leaking onto the wire, and
    * the storage episode id + persistence models are dropped entirely.

Mapping tables are keyed by the episodic enum's STRING value, so this module
does not even import episodic enums at runtime. The completeness of the maps is
guarded by tests: if episodic adds a value, CI fails until a deliberate mapping
decision is made (the wire still stays valid via the default).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.services.pms.client import PMSClient
from app.services.pms.events import (
    ClinicalCategory,
    ClinicalEntities,
    ClinicalMemoryEventV1,
    ClinicalPriority,
    ClinicalSeverity,
    ClinicalTiming,
    EventSource,
)

if TYPE_CHECKING:
    from app.identity import IdentityContext
    from episodic.schemas.episode import Episode

logger = logging.getLogger(__name__)


# episodic storage value (string) -> contract value. Unmapped -> safe default.
_CATEGORY_MAP: dict[str, ClinicalCategory] = {
    "symptom": ClinicalCategory.SYMPTOM,
    "condition": ClinicalCategory.CONDITION,
    "medication": ClinicalCategory.MEDICATION,
    "allergy": ClinicalCategory.ALLERGY,
    "lab": ClinicalCategory.LAB_RESULT,
    "lifestyle": ClinicalCategory.LIFESTYLE,
    "consultation": ClinicalCategory.CONSULTATION,
    "followup": ClinicalCategory.FOLLOW_UP,
}
_SEVERITY_MAP: dict[str, ClinicalSeverity] = {
    "mild": ClinicalSeverity.MILD,
    "moderate": ClinicalSeverity.MODERATE,
    "severe": ClinicalSeverity.SEVERE,
    "critical": ClinicalSeverity.CRITICAL,
    "unknown": ClinicalSeverity.UNKNOWN,
}
_PRIORITY_MAP: dict[str, ClinicalPriority] = {
    "low": ClinicalPriority.LOW,
    "medium": ClinicalPriority.MEDIUM,
    "high": ClinicalPriority.HIGH,
    "critical": ClinicalPriority.CRITICAL,
}


class ClinicalMemoryProducer:
    """Builds + emits longitudinal clinical-memory events. Fail-open."""

    def __init__(self, client: PMSClient) -> None:
        self._client = client

    async def emit_from_episode(
        self,
        *,
        identity: "IdentityContext",
        episode: "Episode",
        source: EventSource = EventSource.PATIENT_CONVERSATION,
    ) -> None:
        """
        Emit ONE event for a freshly-extracted episode. No-op (and never raises)
        when there is no patient to attribute it to, or on any client error —
        producing PMS events must never affect the chat turn.
        """
        if identity.patient_id is None:
            return
        try:
            event = self._to_event(identity=identity, episode=episode, source=source)
            await self._client.ingest_clinical_memory(event)
        except Exception as exc:  # noqa: BLE001 — never break the turn
            logger.warning("PMS clinical-memory emit failed (ignored): %s", exc)

    @staticmethod
    def _to_event(
        *, identity: "IdentityContext", episode: "Episode", source: EventSource
    ) -> ClinicalMemoryEventV1:
        """Translate an internal Episode into the contract — field by field."""
        ent = episode.entities
        tmp = episode.temporal_data
        return ClinicalMemoryEventV1(
            # identity/correlation — domain identifiers only (no storage id)
            patient_id=identity.patient_id.value,
            session_id=identity.session_id,
            request_id=identity.request_id,
            source=source,
            occurred_at=episode.timestamp.isoformat(),
            # clinical payload — mapped to CONTRACT vocabulary, safe defaults
            category=_CATEGORY_MAP.get(episode.category.value, ClinicalCategory.OTHER),
            summary=episode.summary,
            entities=ClinicalEntities(
                symptoms=tuple(ent.symptoms),
                conditions=tuple(ent.conditions),
                medications=tuple(ent.medications),
                labs=tuple(ent.labs),
                body_parts=tuple(ent.body_parts),
            ),
            timing=ClinicalTiming(
                duration=tmp.duration,
                onset=tmp.onset,
                frequency=tmp.frequency,
                progression=tmp.progression,
            ),
            severity=_SEVERITY_MAP.get(episode.severity.value, ClinicalSeverity.UNKNOWN),
            priority=_PRIORITY_MAP.get(
                episode.clinical_priority.value, ClinicalPriority.MEDIUM
            ),
            confidence=float(episode.confidence),
        )


__all__ = ["ClinicalMemoryProducer"]

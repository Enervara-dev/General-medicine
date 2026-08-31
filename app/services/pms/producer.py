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

import hashlib
import logging
from typing import TYPE_CHECKING

from app.services.pms.client import PMSClient
from app.services.pms.events import (
    ClinicalCategory,
    ClinicalEntities,
    ClinicalPriority,
    ClinicalSeverity,
    ClinicalTiming,
    PmsMemoryEventV1,
    SourceChannel,
    SourceRef,
)

if TYPE_CHECKING:
    from app.identity import IdentityContext
    from episodic.schemas.episode import Episode

logger = logging.getLogger(__name__)

# Mirrors the assertion's `azp`. PMS stays specialty-agnostic: this is
# provenance, never authorization.
SPECIALTY_SERVICE = "general-medicine"


# episodic storage value (string) -> contract value. Unmapped -> safe default.
_CATEGORY_MAP: dict[str, ClinicalCategory] = {
    "symptom": ClinicalCategory.SYMPTOM,
    "condition": ClinicalCategory.CONDITION,
    "medication": ClinicalCategory.MEDICATION,
    "allergy": ClinicalCategory.ALLERGY,
    "lab": ClinicalCategory.LAB_RESULT,
    "lab_result": ClinicalCategory.LAB_RESULT,
    "procedure": ClinicalCategory.PROCEDURE,
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
        channel: SourceChannel = SourceChannel.PATIENT_CONVERSATION,
    ) -> None:
        """
        Emit ONE event for a freshly-extracted episode. Never raises: producing PMS
        events must not affect the chat turn.

        The user assertion is taken from the request-scoped identity and forwarded
        verbatim. When it is absent the client refuses to send — GM does not
        substitute the unauthenticated ``identity.patient_id``.
        """
        if identity.patient_id is None:
            return
        try:
            event = self._to_event(identity=identity, episode=episode, channel=channel)
            await self._client.ingest_clinical_memory(
                event,
                user_assertion=identity.user_assertion,
                request_id=identity.request_id,
            )
        except Exception as exc:  # noqa: BLE001 — never break the turn
            logger.warning("PMS clinical-memory emit failed (ignored): %s", type(exc).__name__)

    @staticmethod
    def _to_event(
        *, identity: "IdentityContext", episode: "Episode", channel: SourceChannel
    ) -> PmsMemoryEventV1:
        """Translate an internal Episode into the canonical wire contract.

        Note what does NOT cross: the storage episode id, persistence models, and the
        patient id. Patient identity is carried by the verified assertion, never the
        body — PMS rejects a body-supplied patient outright.
        """
        ent = episode.entities
        tmp = episode.temporal_data
        summary = episode.summary
        occurred = episode.timestamp

        # Deterministic per clinical fact, so a retry of the same fact reuses the same
        # event identity (and therefore the same idempotency key). A storage id is not
        # used: it must not cross this boundary.
        digest = hashlib.sha256(
            "|".join(
                [
                    identity.session_id,
                    str(occurred.isoformat()),
                    str(episode.category.value),
                    summary,
                ]
            ).encode("utf-8")
        ).hexdigest()[:32]

        return PmsMemoryEventV1(
            event_id=digest,
            conversation_id=identity.session_id,
            turn_ref=identity.request_id or digest,
            source=SourceRef(service=SPECIALTY_SERVICE, channel=channel),
            occurred_at=occurred,
            category=_CATEGORY_MAP.get(episode.category.value, ClinicalCategory.OTHER),
            severity=_SEVERITY_MAP.get(episode.severity.value, ClinicalSeverity.UNKNOWN),
            priority=_PRIORITY_MAP.get(
                episode.clinical_priority.value, ClinicalPriority.MEDIUM
            ),
            confidence=float(episode.confidence),
            summary=summary,
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
        )


__all__ = ["ClinicalMemoryProducer"]

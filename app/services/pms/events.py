"""``PmsMemoryEventV1`` — the canonical GM→PMS ingestion wire contract.

MIRRORED CONTRACT. This must stay compatible with
``Patient-Memory-Service/app/api/schemas/memory_event.py``. The repos share no
package, so the schema is duplicated deliberately; changing one side without the
other is a breaking change. A test pins the field set so drift fails CI.

ONE contract, for every specialty. This replaces the two incompatible shapes that
previously both went by ``ClinicalMemoryEventV1`` (a PMS design-doc envelope and a
General-Medicine implementation model) and the minimal ``IngestRequest`` stub the
route actually accepted. The name is deliberately new and unambiguous so neither of
the old ones can be mistaken for it.

WHO EXTRACTS
    Specialty services perform clinical extraction; PMS manages centralized
    longitudinal memory. So this carries *structured clinical content*, not a raw
    transcript for PMS to re-analyze. Nothing here is specialty-specific: the
    originating specialty is identified by ``source.service``, mirroring the
    assertion's ``azp``.

IDENTITY IS NOT ON THE WIRE
    There is deliberately no ``patient_id`` field. The authoritative patient is the
    verified ``sub`` of the user assertion. A body-supplied patient id would be a
    second, forgeable source of the one fact that governs every authorization
    decision — exactly the confused-deputy hole the boundary exists to close.
    ``extra="forbid"`` means smuggling one in is a validation error, not a silent
    override.

EXTENSIBILITY
    Within v1 only additive, backward-compatible changes are allowed: new OPTIONAL
    fields, and new enum members only where a documented safe fallback exists.
    Anything else requires ``pms.memory_event/v2``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "pms.memory_event/v1"

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class SourceChannel(StrEnum):
    """How the evidence reached the specialty service."""

    PATIENT_CONVERSATION = "patient_conversation"
    UPLOADED_DOCUMENT = "uploaded_document"
    CLINICIAN_ENTRY = "clinician_entry"
    DEVICE = "device"
    OTHER = "other"


class ClinicalCategory(StrEnum):
    SYMPTOM = "symptom"
    CONDITION = "condition"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    LAB_RESULT = "lab_result"
    PROCEDURE = "procedure"
    LIFESTYLE = "lifestyle"
    CONSULTATION = "consultation"
    FOLLOW_UP = "follow_up"
    OTHER = "other"  # safe sink for anything a specialty cannot map


class ClinicalStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    HISTORICAL = "historical"
    NEGATED = "negated"
    UNKNOWN = "unknown"


class ClinicalSeverity(StrEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ClinicalPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SourceRef(BaseModel):
    """Which service produced this event, and through what channel."""

    model_config = _FROZEN

    service: str = Field(min_length=1, max_length=64)  # mirrors the assertion's azp
    channel: SourceChannel = SourceChannel.PATIENT_CONVERSATION
    version: str | None = Field(default=None, max_length=32)


class ClinicalEntities(BaseModel):
    """Extracted entities. Every list is optional and defaults empty."""

    model_config = _FROZEN

    symptoms: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    medications: tuple[str, ...] = ()
    labs: tuple[str, ...] = ()
    procedures: tuple[str, ...] = ()
    body_parts: tuple[str, ...] = ()


class ClinicalTiming(BaseModel):
    """Free-text temporal expressions, as stated. PMS resolves them later."""

    model_config = _FROZEN

    duration: str | None = Field(default=None, max_length=200)
    onset: str | None = Field(default=None, max_length=200)
    frequency: str | None = Field(default=None, max_length=200)
    progression: str | None = Field(default=None, max_length=200)


class Evidence(BaseModel):
    """Provenance for the assertion being made — what it was derived from."""

    model_config = _FROZEN

    quote: str | None = Field(default=None, max_length=2_000)
    extractor: str | None = Field(default=None, max_length=64)
    extractor_version: str | None = Field(default=None, max_length=32)


class PmsMemoryEventV1(BaseModel):
    """One structured clinical fact contributed to a patient's shared memory."""

    model_config = _FROZEN

    schema_version: Literal["pms.memory_event/v1"] = "pms.memory_event/v1"

    # --- event identity + correlation ------------------------------------- #
    # `event_id` is the producer's identity for this fact. `conversation_id` and
    # `turn_ref` locate it in the interaction and form the idempotency key.
    event_id: str = Field(min_length=1, max_length=200)
    conversation_id: str = Field(min_length=1, max_length=200)
    turn_ref: str = Field(min_length=1, max_length=200)

    # --- provenance -------------------------------------------------------- #
    source: SourceRef
    occurred_at: datetime | None = None  # clinical time; server clock when absent
    recorded_at: datetime | None = None  # when the specialty observed it

    # --- clinical payload -------------------------------------------------- #
    category: ClinicalCategory = ClinicalCategory.OTHER
    status: ClinicalStatus = ClinicalStatus.UNKNOWN
    severity: ClinicalSeverity = ClinicalSeverity.UNKNOWN
    priority: ClinicalPriority = ClinicalPriority.MEDIUM
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    summary: str = Field(min_length=1, max_length=4_000)
    text: str | None = Field(default=None, max_length=20_000)
    entities: ClinicalEntities = ClinicalEntities()
    timing: ClinicalTiming = ClinicalTiming()
    evidence: Evidence = Evidence()

    def idempotency_key(self) -> str:
        """Stable content-derived key for de-duplicating this clinical fact.

        Deterministic: the same event yields the same key on every delivery attempt,
        so PMS collapses redeliveries. Nothing delivery-specific participates — a
        retry of the same fact must never mint a new key.
        """
        import hashlib
        import json

        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def content_payload(self) -> dict[str, object]:
        """The clinical body, as stored in the immutable event's opaque content.

        Identity and correlation stay out: they are carried by the domain event's
        own fields (patient, provenance, idempotency), not duplicated into content.
        """
        return self.model_dump(
            mode="json",
            exclude={"event_id", "conversation_id", "turn_ref", "occurred_at", "recorded_at"},
        )


__all__ = [
    "SCHEMA_VERSION",
    "ClinicalCategory",
    "ClinicalEntities",
    "ClinicalPriority",
    "ClinicalSeverity",
    "ClinicalStatus",
    "ClinicalTiming",
    "Evidence",
    "PmsMemoryEventV1",
    "SourceChannel",
    "SourceRef",
]

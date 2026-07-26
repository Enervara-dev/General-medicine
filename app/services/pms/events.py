"""
ClinicalMemoryEventV1 — the stable, independent OUTBOUND contract to PMS.

OWNERSHIP
    This schema is owned by the service↔PMS boundary. It is deliberately
    self-contained: it imports NOTHING from `episodic/` (or any storage layer)
    and carries its OWN vocabulary. Episodic storage is free to change its
    enums, models, and identifiers without ever changing this wire format — the
    translation lives solely in `producer.py` (the anti-corruption layer).

WHAT IS DELIBERATELY EXCLUDED (never crosses this boundary)
    * storage enums        — episodic EpisodeCategory / Severity / ClinicalPriority
    * storage identifiers  — the Pinecone episode UUID (episode_id)
    * persistence models   — EpisodeEntities / TemporalData (dumped shapes)
    * implementation names  — e.g. "episodic_extraction" as a source

VERSIONING
    `schema_version` pins the contract. Within v1 only ADDITIVE, backward-
    compatible changes are allowed (new optional fields; consumers ignore
    unknown fields). Any breaking change (removing/renaming/repurposing a field,
    or changing an enum's meaning) requires a NEW model `ClinicalMemoryEventV2`
    with `schema_version="clinical_memory_event.v2"` — v1 is never mutated.

IMMUTABILITY
    Every model here is `frozen=True` — once produced, an event cannot be
    mutated. `extra="forbid"` keeps producers from smuggling undocumented fields.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "clinical_memory_event.v1"

_FROZEN = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Contract-owned vocabulary (closed sets — NOT episodic's enums)
# ---------------------------------------------------------------------------

class EventSource(str, Enum):
    """Domain-level provenance — never the name of an internal subsystem."""

    PATIENT_CONVERSATION = "patient_conversation"
    UPLOADED_DOCUMENT = "uploaded_document"
    OTHER = "other"


class ClinicalCategory(str, Enum):
    SYMPTOM = "symptom"
    CONDITION = "condition"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    LAB_RESULT = "lab_result"
    LIFESTYLE = "lifestyle"
    CONSULTATION = "consultation"
    FOLLOW_UP = "follow_up"
    OTHER = "other"        # safe sink for any unmapped storage category


class ClinicalSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    UNKNOWN = "unknown"    # safe default


class ClinicalPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"      # safe default
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Contract-owned payload models (independent of any persistence model)
# ---------------------------------------------------------------------------

class ClinicalEntities(BaseModel):
    model_config = _FROZEN
    symptoms: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    medications: tuple[str, ...] = ()
    labs: tuple[str, ...] = ()
    body_parts: tuple[str, ...] = ()


class ClinicalTiming(BaseModel):
    model_config = _FROZEN
    duration: Optional[str] = None
    onset: Optional[str] = None
    frequency: Optional[str] = None
    progression: Optional[str] = None


class ClinicalMemoryEventV1(BaseModel):
    """One extracted clinical fact, attributed to a patient + conversation."""

    model_config = _FROZEN

    schema_version: Literal["clinical_memory_event.v1"] = SCHEMA_VERSION

    # Identity + correlation — canonical DOMAIN identifiers only.
    patient_id: str = Field(min_length=1)   # authenticated Mongo User._id
    session_id: str = Field(min_length=1)   # canonical conversation id
    request_id: str = ""                     # cross-cutting correlation (not storage)

    # Provenance — domain-level, never an internal subsystem name.
    source: EventSource = EventSource.PATIENT_CONVERSATION
    occurred_at: str = ""                    # ISO-8601 UTC (domain time)

    # Clinical payload — contract-owned vocabulary.
    category: ClinicalCategory
    summary: str
    entities: ClinicalEntities = Field(default_factory=ClinicalEntities)
    timing: ClinicalTiming = Field(default_factory=ClinicalTiming)
    severity: ClinicalSeverity = ClinicalSeverity.UNKNOWN
    priority: ClinicalPriority = ClinicalPriority.MEDIUM
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


__all__ = [
    "SCHEMA_VERSION",
    "EventSource",
    "ClinicalCategory",
    "ClinicalSeverity",
    "ClinicalPriority",
    "ClinicalEntities",
    "ClinicalTiming",
    "ClinicalMemoryEventV1",
]

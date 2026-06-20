"""DTOs for clinical facts and risk flags emitted during extraction."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memory.models.clinical_fact import FactSource, FactStatus, FactType


class ClinicalFactCandidate(BaseModel):
    """
    A fact proposed by the ExtractionService, not yet persisted.

    Produced from a single user turn. The ConsolidationService is responsible
    for merging this against existing rows (supersede / contradict / dedupe).
    """

    fact_type: FactType
    canonical_name: str = Field(min_length=1, max_length=256)
    normalized_code: str | None = Field(default=None, max_length=64)
    value: dict[str, Any] = Field(default_factory=dict)

    onset_at: datetime | None = None
    expires_at: datetime | None = None

    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    importance: float = Field(ge=0.0, le=1.0, default=0.5)

    # Negation: a "no fever" extraction is a candidate that will mark any
    # existing active "fever" fact as contradicted.
    negated: bool = False

    # Extracted free-text rationale for downstream audit / debugging.
    rationale: str | None = None


class ClinicalFactDTO(BaseModel):
    """Read-side projection of a persisted ClinicalFact."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    fact_type: FactType
    canonical_name: str
    normalized_code: str | None
    value: dict[str, Any]
    onset_at: datetime | None
    observed_at: datetime
    expires_at: datetime | None
    status: FactStatus
    confidence: float
    importance: float
    decay_score: float
    source: FactSource
    source_event_id: uuid.UUID | None
    supersedes_id: uuid.UUID | None
    contradicts_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


RiskSeverity = Literal["info", "warn", "block"]


class RiskFlag(BaseModel):
    """A safety risk identified by the SafetyService."""

    severity: RiskSeverity
    code: str                       # e.g. "allergy_collision", "drug_interaction"
    message: str                    # human-readable
    related_fact_ids: list[uuid.UUID] = Field(default_factory=list)

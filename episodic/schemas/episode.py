"""
Episode — the canonical episodic memory object.

An Episode is a structured, clinically-meaningful event extracted from a
patient utterance. It is the only thing the layer stores in Pinecone.
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from episodic.schemas.entity_normalization import (
    has_structured_entities,
    normalize_entity_list,
)


class EpisodeCategory(str, Enum):
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    CONSULTATION = "consultation"
    LAB = "lab"
    FOLLOWUP = "followup"
    CONDITION = "condition"   # chronic / diagnosed condition
    LIFESTYLE = "lifestyle"
    ALLERGY = "allergy"


class ClinicalPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EpisodeEntities(BaseModel):
    """
    Flat entity vocabulary. Every list is ``list[str]``.

    The extraction LLM emits a bare entity as a string but upgrades it to an
    object ({"name": "fever", "value": "101F"}) the moment it carries a
    qualifier. Both shapes are accepted and folded to one display string by the
    validator below, so a qualified entity no longer fails validation and takes
    the whole candidate down with it. See ``entity_normalization``.
    """

    symptoms: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    labs: list[str] = Field(default_factory=list)
    body_parts: list[str] = Field(default_factory=list)

    # "*" so any entity list added later is normalised without a code change.
    @field_validator("*", mode="before")
    @classmethod
    def _normalize_entities(cls, value: Any) -> Any:
        return normalize_entity_list(value)


class TemporalData(BaseModel):
    duration: str | None = None       # e.g. "2 weeks"
    onset: str | None = None          # e.g. "yesterday morning"
    frequency: str | None = None      # e.g. "every evening"
    progression: str | None = None    # e.g. "worsening"


class EpisodeCandidate(BaseModel):
    """
    Output of the extractor BEFORE the candidate is committed to storage.
    No episode_id yet; the storage layer mints one on persist.
    """

    model_config = ConfigDict(extra="ignore")

    user_id: str
    summary: str
    category: EpisodeCategory
    entities: EpisodeEntities = Field(default_factory=EpisodeEntities)
    temporal_data: TemporalData = Field(default_factory=TemporalData)
    severity: Severity = Severity.UNKNOWN
    clinical_priority: ClinicalPriority = ClinicalPriority.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    source: str = "user_self_report"
    embedding_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Extractor sets this to False for conversational noise / non-medical content
    # so the storage layer can drop the candidate cheaply.
    store_memory: bool = True

    @model_validator(mode="before")
    @classmethod
    def _preserve_structured_entities(cls, data: Any) -> Any:
        """
        Keep the extractor's structured entities verbatim on ``metadata`` before
        they are flattened to display strings, so no attribute is silently lost.

        ``metadata`` is internal: the PMS producer maps named fields only, so
        this never crosses the PMS boundary and changes no wire contract.
        """
        if not isinstance(data, dict):
            return data
        entities = data.get("entities")
        if not has_structured_entities(entities):
            return data
        metadata = data.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata.setdefault("entities_raw", copy.deepcopy(entities))
        data = dict(data)
        data["metadata"] = metadata
        return data

    def is_chronic(self) -> bool:
        """Chronic episodes don't decay. Heuristic: category=condition + no end date."""
        return (
            self.category == EpisodeCategory.CONDITION
            and not self.temporal_data.duration
        ) or self.category == EpisodeCategory.ALLERGY


class Episode(EpisodeCandidate):
    """
    A persisted episodic memory. Carries identity + timestamp.
    """

    episode_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    @classmethod
    def from_candidate(cls, c: EpisodeCandidate) -> "Episode":
        return cls(**c.model_dump())

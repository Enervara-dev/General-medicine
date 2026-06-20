"""DTOs returned by RetrievalService — what the pipeline gets back."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memory.schemas.fact import ClinicalFactDTO
from memory.schemas.state import PatientStateSnapshot


class EpisodicMemoryDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: str
    event_type: str
    occurred_at: datetime
    importance: float


class SemanticMemoryDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    content: str
    importance: float
    decay_score: float
    similarity: float | None = None      # populated only after vector search


RetrievalStrategy = Literal[
    "structured_only",
    "structured_plus_episodic",
    "structured_plus_semantic",
    "structured_plus_episodic_plus_semantic",
    "cache_hit",
]


class RetrievalDecision(BaseModel):
    """Diagnostic record of what the retrieval service decided and why."""

    strategy: RetrievalStrategy
    used_cache: bool
    vector_recall_invoked: bool
    fact_count: int
    episode_count: int
    semantic_count: int


class MemoryContext(BaseModel):
    """
    The complete memory bundle handed to GraphRAGPipeline for prompt assembly.

    This is what replaces the legacy `WorkingMemory` dataclass. It contains
    only what should be visible to the LLM — never raw conversation turns.
    """

    model_config = ConfigDict(from_attributes=True)

    patient_id: uuid.UUID
    request_id: uuid.UUID

    snapshot: PatientStateSnapshot
    active_facts: list[ClinicalFactDTO] = Field(default_factory=list)
    recent_episodes: list[EpisodicMemoryDTO] = Field(default_factory=list)
    semantic_recalls: list[SemanticMemoryDTO] = Field(default_factory=list)

    decision: RetrievalDecision
    estimated_tokens: int = 0

"""Retrieval DTOs — request, ranked results, compressed context blocks."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from episodic.schemas.episode import Episode


class RetrievalRequest(BaseModel):
    """Input to /episodic/retrieve and /episodic/context."""
    user_id: str
    query_text: str
    top_k: int | None = None              # before rerank
    return_k: int | None = None           # final size
    categories: list[str] | None = None   # metadata filter (episode categories)
    since: datetime | None = None         # metadata filter (timestamp >= since)
    until: datetime | None = None         # metadata filter (timestamp <= until)


class RankedEpisode(BaseModel):
    """An episode with its composite rank score and the contributing factors."""

    model_config = ConfigDict(extra="ignore")

    episode: Episode
    score: float
    factors: dict[str, float] = Field(default_factory=dict)


class CompressedEpisode(BaseModel):
    """A cluster of redundant episodes collapsed into one prose summary."""
    representative_id: uuid.UUID
    member_ids: list[uuid.UUID]
    category: str
    summary: str
    first_seen: datetime
    last_seen: datetime
    peak_severity: str | None = None
    score: float


class ContextBlock(BaseModel):
    """End-to-end output for the orchestrator: ranked + compressed + prompt-ready."""
    user_id: str
    query_text: str
    episodes: list[RankedEpisode] = Field(default_factory=list)
    compressed: list[CompressedEpisode] = Field(default_factory=list)
    rendered_prompt: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Raw retrieval (no compression)."""
    user_id: str
    query_text: str
    episodes: list[RankedEpisode]

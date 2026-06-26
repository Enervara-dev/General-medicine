"""Chat request/response/stream schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    # When provided, the orchestrator loads the user's episodic memory before
    # the LLM call and ingests the turn after the answer. When omitted, the
    # episodic stage is skipped (parity with the CLI's --user-id flag).
    user_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    request_id: str
    analysis: dict[str, Any] | None = None
    timing_ms: dict[str, int] = Field(default_factory=dict)
    routing: dict[str, Any] = Field(default_factory=dict)
    followup_questions: list[str] = Field(default_factory=list)


class ChatStreamEvent(BaseModel):
    type: Literal["chunk", "done", "error", "meta"]
    data: str | None = None
    timing_ms: dict[str, int] | None = None
    error: dict[str, str] | None = None


class MediaInfo(BaseModel):
    """Metadata-only view of a processed upload (never carries raw bytes)."""

    category: str
    route: str
    mime_type: str
    size_bytes: int
    filename: str | None = None
    storage_uri: str | None = None
    caption: str | None = None
    extracted_facts: list[str] = Field(default_factory=list)


class ImageChatResponse(ChatResponse):
    """A `/chat/image` answer: a normal chat response plus the upload metadata."""

    media: MediaInfo

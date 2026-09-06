"""Chat request/response/stream schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class DemographicsEnvelope(BaseModel):
    """
    AI-safe demographics supplied BY the Backend on the request.

    This replaces reading the Backend's database directly. GM used to open its
    own MongoDB connection to Enervara's `users` collection and look the patient
    up by ObjectId — a second service reaching into another service's database,
    which broke outright when the Backend moved to PostgreSQL and would have
    broken again on any schema change there.

    Only the seven fields that may reach the LLM travel here. Note that the
    Backend sends a derived ``age``, never a date of birth: GM has no use for
    the exact date, so it is not sent at all.
    """

    age: int | None = None
    sex: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    bmi: float | None = None
    state: str | None = None
    city: str | None = None


class IdentityEnvelope(BaseModel):
    """
    New identity contract the Backend MAY send alongside (or instead of) the
    legacy top-level ``user_id``/``session_id``. Fully optional and additive —
    absent → the legacy fields are used, so existing callers are unaffected.
    """

    # ``patient_id`` is the Backend's canonical patient id: a UUID (``patients.id``
    # in its PostgreSQL schema). It was a 24-char Mongo ``User._id`` before that
    # migration. GM never parses or validates the shape — the id is opaque here
    # and is only ever carried, so the Backend can change its scheme without a
    # coordinated release. ``user_id`` is accepted as an alias.
    patient_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    consumer_id: str | None = None   # which upstream consumer/service initiated
    # Supplied by the Backend so GM never needs its own view of patient data.
    demographics: DemographicsEnvelope | None = None


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    # When provided, the orchestrator loads the user's episodic memory before
    # the LLM call and ingests the turn after the answer. When omitted, the
    # episodic stage is skipped (parity with the CLI's --user-id flag).
    user_id: str | None = None
    # New identity contract (optional). Preferred over the legacy fields above
    # when ENABLE_IDENTITY_V1 is on; ignored/absent keeps legacy behaviour.
    identity: IdentityEnvelope | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    request_id: str
    analysis: dict[str, Any] | None = None
    timing_ms: dict[str, int] = Field(default_factory=dict)
    routing: dict[str, Any] = Field(default_factory=dict)
    followup_questions: list[str] = Field(default_factory=list)
    # True once the consultation has reached a concluded answer — the client may
    # then offer "Show this to your doctor" (the SOAP note at POST /chat/soap).
    show_doctor_summary: bool = False


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


class SoapRequest(BaseModel):
    """Trigger a fresh doctor-facing SOAP note for an existing session."""

    session_id: str = Field(min_length=1)
    user_id: str | None = None
    identity: IdentityEnvelope | None = None


class SoapNote(BaseModel):
    """
    Doctor-facing SOAP note, generated on demand from the latest conversation.

    Grounded strictly in the conversation — never fabricated. Each section is
    plain prose; `unavailable` explicitly names clinically relevant information
    the conversation did not provide (e.g. "no vital signs recorded").
    """

    subjective: str
    objective: str
    assessment: str
    plan: str
    unavailable: list[str] = Field(default_factory=list)
    session_id: str
    request_id: str
    generated_at: str  # ISO-8601 UTC, stamped by the route

"""Chat request/response/stream schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


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


class LifestyleEnvelope(BaseModel):
    """`patient_lifestyle`. Enum values travel as the Backend's own strings."""

    diet: str | None = None
    exercise: str | None = None
    alcohol: str | None = None
    smoking: str | None = None
    sleep_hours: float | None = None
    water_cups: int | None = None


class WellbeingEnvelope(BaseModel):
    """
    `patient_wellbeing`. The five rating dimensions are 1-10 in the Backend.

    ``relaxation_practices`` is the COLUMN name; ``relaxation_frequency`` is the
    enum type behind it. Naming the field after the type would not match the
    schema on either side.
    """

    overall_mood: int | None = None
    stress_level: int | None = None
    energy_level: int | None = None
    social_connectedness: int | None = None
    work_academic_pressure: int | None = None
    relaxation_practices: str | None = None


class AllergyItem(BaseModel):
    allergen_name: str
    reaction_types: list[str] = Field(default_factory=list)
    severity: str | None = None
    last_reaction_on: str | None = None


class AllergiesEnvelope(BaseModel):
    """
    An envelope, not a bare list, because two different facts must be told apart:

        items == []  and  none_confirmed_at is None   -> nothing recorded
        items == []  and  none_confirmed_at is set    -> patient confirmed none

    A bare array cannot express the second, and reading "unknown" as "none" is
    the kind of error that matters when the subject is allergies.
    """

    items: list[AllergyItem] = Field(default_factory=list)
    none_confirmed_at: str | None = None


class ConditionItem(BaseModel):
    condition_code: str
    display_name: str | None = None
    category: str | None = None
    since_bucket: str | None = None
    since_exact_date: str | None = None
    currently_troubling: str | None = None
    on_medication: bool = False


class ConditionsEnvelope(BaseModel):
    """Same asymmetry as allergies, for the same clinical reason."""

    items: list[ConditionItem] = Field(default_factory=list)
    none_confirmed_at: str | None = None


class MedicationItem(BaseModel):
    medication_name: str
    course_type: str | None = None
    dose_amount: float | None = None
    dose_unit: str | None = None
    frequency_count: int | None = None
    frequency_period: str | None = None
    time_of_day: list[str] = Field(default_factory=list)
    food_relation: str | None = None
    reason_or_condition: str | None = None
    # The single writable direction of the condition link. GM carries it and
    # never derives the reverse condition -> medication relationship.
    linked_condition_id: str | None = None


class SurgeryItem(BaseModel):
    surgery_name: str
    performed_year: int | None = None
    reason: str | None = None
    hospital: str | None = None
    current_status: str | None = None


class HealthProfileEnvelope(BaseModel):
    """
    The patient's ACTIVE clinical profile, supplied by the Backend per request.

    Every section is optional: a patient may have recorded none, some or all of
    them, and the Backend omits a section rather than sending it empty. The
    list-valued sections default to empty lists so a consumer can iterate
    unconditionally without a presence check.

    Archived rows never appear here; the Backend filters on ``archived_at is
    null`` in the aggregation query. Reproductive health is deliberately absent:
    there is no table, endpoint or data source behind it.
    """

    model_config = ConfigDict(extra="ignore")

    lifestyle: LifestyleEnvelope | None = None
    wellbeing: WellbeingEnvelope | None = None
    allergies: AllergiesEnvelope | None = None
    conditions: ConditionsEnvelope | None = None
    medications: list[MedicationItem] = Field(default_factory=list)
    surgeries: list[SurgeryItem] = Field(default_factory=list)


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
    # The wider clinical profile, also Backend-supplied. Optional and additive:
    # a caller sending only `demographics` behaves exactly as before.
    health_profile: HealthProfileEnvelope | None = None


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


class HealthContext(BaseModel):
    """
    Longitudinal Health Context, sourced from the Patient Memory Service.

    ``available`` is the load signal, not an error: PMS being unreachable, the
    patient being anonymous, or PMS being disabled all yield ``available=False``
    with a machine-readable ``reason``, and the client simply renders no panel.

    Lists are pass-through views of PMS's own shapes rather than a second
    modelling of clinical memory. PMS owns that vocabulary; restating it here
    would create a contract to keep in sync for no benefit.
    """

    available: bool
    request_id: str
    reason: str | None = None
    current_episodes: list[dict[str, Any]] = Field(default_factory=list)
    historical_episodes: list[dict[str, Any]] = Field(default_factory=list)
    assertions: list[dict[str, Any]] = Field(default_factory=list)
    open_clarifications: int = 0

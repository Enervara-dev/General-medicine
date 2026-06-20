"""
ClinicalFact — the atomic unit of memory.

Every piece of medically-relevant information about a patient is one row here.
Facts have temporal validity, provenance, confidence, importance, and supersede
chains so contradictions are detected rather than blended.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from memory.db.base import Base


class FactType(str, enum.Enum):
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    CONDITION = "condition"
    LAB_VALUE = "lab_value"
    VITAL = "vital"
    LIFESTYLE = "lifestyle"
    SOCIAL = "social"
    FAMILY_HISTORY = "family_history"
    ADHERENCE = "adherence"
    EMOTIONAL = "emotional"
    PREFERENCE = "preference"


class FactStatus(str, enum.Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    REFUTED = "refuted"


class FactSource(str, enum.Enum):
    PATIENT_REPORT = "patient_report"
    LLM_EXTRACTION = "llm_extraction"
    LAB_IMPORT = "lab_import"
    MANUAL_ENTRY = "manual_entry"


class ClinicalFact(Base):
    __tablename__ = "clinical_fact"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
    )

    # ── Identity of the fact ────────────────────────────────────────────────
    fact_type: Mapped[FactType] = mapped_column(
        Enum(FactType, name="fact_type"), nullable=False
    )
    canonical_name: Mapped[str] = mapped_column(String(256), nullable=False)
    normalized_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    value: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )

    # ── Temporal validity ───────────────────────────────────────────────────
    onset_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[FactStatus] = mapped_column(
        Enum(FactStatus, name="fact_status"),
        nullable=False,
        default=FactStatus.ACTIVE,
    )

    # ── Scoring ────────────────────────────────────────────────────────────
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    decay_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    # ── Provenance ─────────────────────────────────────────────────────────
    source: Mapped[FactSource] = mapped_column(
        Enum(FactSource, name="fact_source"), nullable=False
    )
    source_event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_event.id", ondelete="SET NULL"),
        nullable=True,
    )
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clinical_fact.id", ondelete="SET NULL"),
        nullable=True,
    )
    contradicts_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clinical_fact.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        # Hot path: active facts of a given type for a patient, ordered by importance.
        Index(
            "ix_clinical_fact_active",
            "patient_id",
            "fact_type",
            "status",
            "importance",
            postgresql_where=text("status = 'active'"),
        ),
        # Expiry sweep.
        Index(
            "ix_clinical_fact_expiry",
            "expires_at",
            postgresql_where=text("status = 'active' AND expires_at IS NOT NULL"),
        ),
        # JSONB attribute queries (e.g. value->>'severity' = 'severe').
        Index(
            "ix_clinical_fact_value_gin",
            "value",
            postgresql_using="gin",
        ),
        # Provenance back-lookup.
        Index("ix_clinical_fact_source_event", "source_event_id"),
    )

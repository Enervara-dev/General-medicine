"""EpisodicMemory — clinically-significant events worth preserving distinctly."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from memory.db.base import Base


class EpisodeType(str, enum.Enum):
    ER_VISIT = "er_visit"
    HOSPITALIZATION = "hospitalization"
    ADVERSE_DRUG_REACTION = "adverse_drug_reaction"
    NEW_DIAGNOSIS = "new_diagnosis"
    SYMPTOM_ONSET = "symptom_onset"
    TREATMENT_CHANGE = "treatment_change"
    MILESTONE = "milestone"


class EpisodicMemory(Base):
    __tablename__ = "episodic_memory"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    event_type: Mapped[EpisodeType] = mapped_column(
        Enum(EpisodeType, name="episode_type"), nullable=False
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    related_fact_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    source_event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_event.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        Index(
            "ix_episodic_memory_patient_time",
            "patient_id",
            "occurred_at",
        ),
    )

"""ConversationEvent — raw turn log, audit-only, NEVER injected into a prompt."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from memory.db.base import Base


class ConvRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationEvent(Base):
    __tablename__ = "conversation_event"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[ConvRole] = mapped_column(
        Enum(ConvRole, name="conv_role"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    request_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index(
            "ix_conversation_event_patient_time",
            "patient_id",
            "created_at",
            postgresql_using="btree",
        ),
        Index("ix_conversation_event_session", "session_id", "created_at"),
    )

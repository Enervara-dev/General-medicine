"""
RetrievalLog — audit trail for every LLM call.

Records exactly which clinical_fact / episodic_memory / semantic_memory rows
entered the prompt, the strategy used, and the token counts. Allows the
question "why did the assistant say X?" to be answered by a single query.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from memory.db.base import Base


class RetrievalLog(Base):
    __tablename__ = "retrieval_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
    )
    request_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    routing_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    retrieved_fact_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    retrieved_episode_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    retrieved_semantic_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    retrieval_strategy: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    prompt_tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens_out: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_retrieval_log_patient_time", "patient_id", "created_at"),
        Index("ix_retrieval_log_request", "request_id"),
    )

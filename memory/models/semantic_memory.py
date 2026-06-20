"""
SemanticMemory — pgvector-backed store for nuanced prior discussions.

Used as a NARROW retrieval channel, only when structured retrieval cannot
satisfy a query. Capped at top-2 results during retrieval. Decayed and
periodically pruned.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from memory.db.base import Base

# pgvector's SQLAlchemy integration. Install via: pip install pgvector
try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - kept so module imports without pgvector
    Vector = None  # type: ignore[assignment]


# Match the embedding dimension of llama-text-embed-v2 (1024).
EMBEDDING_DIM = 1024


class SemanticMemory(Base):
    __tablename__ = "semantic_memory"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    if Vector is not None:
        embedding: Mapped[list[float]] = mapped_column(
            Vector(EMBEDDING_DIM), nullable=False
        )
    else:  # pragma: no cover - fallback when pgvector not installed
        embedding = None  # type: ignore[assignment]

    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    decay_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_event.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # HNSW index is created in the Alembic migration via raw DDL; SQLAlchemy
    # doesn't generate HNSW syntax natively.
    __table_args__ = (
        Index("ix_semantic_memory_patient", "patient_id"),
    )

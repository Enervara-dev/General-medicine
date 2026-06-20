"""semantic_memory table + pgvector extension

Revision ID: 0002_semantic_memory_pgvector
Revises: 0001_initial
Create Date: 2026-05-17

Adds the pgvector extension and the semantic_memory table with an HNSW
index on the embedding column.

PREREQUISITES:
    The Postgres server must have the pgvector extension installed.
    See docs/MEMORY_PGVECTOR_SETUP.md for installation instructions per
    platform (Docker pgvector image, Windows binaries, Linux apt, etc.).

If pgvector is unavailable, this migration intentionally fails fast with
the upstream error so the operator knows to install the extension before
running it. The core schema (migration 0001) remains usable without
semantic_memory; only the vector-recall fallback path in
RetrievalService becomes a no-op.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# pgvector's SQLAlchemy integration — required at migration time so we can
# generate the proper "vector(1024)" DDL.
from pgvector.sqlalchemy import Vector


revision: str = "0002_semantic_memory_pgvector"
down_revision: Union[str, None] = "0001_initial"
branch_labels = None
depends_on = None


# Must match memory/models/semantic_memory.py::EMBEDDING_DIM and the
# embedding model used to populate it (llama-text-embed-v2 → 1024).
EMBEDDING_DIM = 1024


def upgrade() -> None:
    # 1) Install pgvector extension. Fails fast if the extension binaries
    #    are not present on the Postgres server.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2) semantic_memory table — uses the proper Vector(N) column type.
    op.create_table(
        "semantic_memory",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "patient_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patient.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("importance", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("decay_score", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True)),
        sa.Column("access_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "source_event_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversation_event.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_semantic_memory_patient", "semantic_memory", ["patient_id"])

    # 3) HNSW index for fast cosine-distance search.
    op.execute(
        "CREATE INDEX ix_semantic_memory_embedding_hnsw "
        "ON semantic_memory USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_semantic_memory_embedding_hnsw;")
    op.drop_table("semantic_memory")
    # Do NOT drop the extension — other tables / future migrations may need it.

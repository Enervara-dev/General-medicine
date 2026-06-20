"""initial longitudinal memory schema (core tables, no pgvector)

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-17

Creates the core memory schema:
  patient, conversation_event, clinical_fact, patient_state,
  episodic_memory, retrieval_log.

semantic_memory (which requires pgvector) is created by migration 0002.
This split lets the core schema land on stock Postgres installations that
do not have pgvector built/installed yet.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels = None
depends_on = None


def _create_enum_if_missing(name: str, values: list[str]) -> None:
    """Create a Postgres ENUM type only if it doesn't already exist.

    Postgres has no `CREATE TYPE ... IF NOT EXISTS`; a DO-block with an
    EXCEPTION handler is the idiomatic workaround. Idempotent — safe to run
    on a database where the type was created by a previous partial run.
    """
    literal_values = ", ".join(f"'{v}'" for v in values)
    op.execute(
        f"""
        DO $$ BEGIN
            CREATE TYPE {name} AS ENUM ({literal_values});
        EXCEPTION WHEN duplicate_object THEN null;
        END $$;
        """
    )


def upgrade() -> None:
    # ── Extensions (core only — pgcrypto for gen_random_uuid) ─────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    # ── Enums ──────────────────────────────────────────────────────────────
    # Postgres has no `CREATE TYPE ... IF NOT EXISTS`; use a DO-block guard
    # so this migration is idempotent against partially-applied prior runs.
    _create_enum_if_missing(
        "conv_role", ["user", "assistant", "system"],
    )
    _create_enum_if_missing(
        "fact_type",
        [
            "symptom", "medication", "allergy", "condition",
            "lab_value", "vital", "lifestyle", "social",
            "family_history", "adherence", "emotional", "preference",
        ],
    )
    _create_enum_if_missing(
        "fact_status",
        ["active", "resolved", "superseded", "contradicted", "refuted"],
    )
    _create_enum_if_missing(
        "fact_source",
        ["patient_report", "llm_extraction", "lab_import", "manual_entry"],
    )
    _create_enum_if_missing(
        "episode_type",
        [
            "er_visit", "hospitalization", "adverse_drug_reaction",
            "new_diagnosis", "symptom_onset", "treatment_change", "milestone",
        ],
    )

    # create_type=False prevents SQLAlchemy from re-emitting CREATE TYPE
    # when these enums appear on tables created below.
    conv_role = postgresql.ENUM(
        "user", "assistant", "system",
        name="conv_role", create_type=False,
    )
    fact_type = postgresql.ENUM(
        "symptom", "medication", "allergy", "condition",
        "lab_value", "vital", "lifestyle", "social",
        "family_history", "adherence", "emotional", "preference",
        name="fact_type", create_type=False,
    )
    fact_status = postgresql.ENUM(
        "active", "resolved", "superseded", "contradicted", "refuted",
        name="fact_status", create_type=False,
    )
    fact_source = postgresql.ENUM(
        "patient_report", "llm_extraction", "lab_import", "manual_entry",
        name="fact_source", create_type=False,
    )
    episode_type = postgresql.ENUM(
        "er_visit", "hospitalization", "adverse_drug_reaction",
        "new_diagnosis", "symptom_onset", "treatment_change", "milestone",
        name="episode_type", create_type=False,
    )

    # ── patient ───────────────────────────────────────────────────────────
    op.create_table(
        "patient",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("external_id", sa.String(128), unique=True),
        sa.Column("metadata", postgresql.JSONB, nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_patient_external_id", "patient", ["external_id"])

    # ── conversation_event ────────────────────────────────────────────────
    op.create_table(
        "conversation_event",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("patient.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("role", conv_role, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("analysis_payload", postgresql.JSONB),
        sa.Column("request_id", postgresql.UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_conversation_event_patient_time",
                    "conversation_event", ["patient_id", "created_at"])
    op.create_index("ix_conversation_event_session",
                    "conversation_event", ["session_id", "created_at"])

    # ── clinical_fact ─────────────────────────────────────────────────────
    op.create_table(
        "clinical_fact",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("patient.id", ondelete="CASCADE"), nullable=False),
        sa.Column("fact_type", fact_type, nullable=False),
        sa.Column("canonical_name", sa.String(256), nullable=False),
        sa.Column("normalized_code", sa.String(64)),
        sa.Column("value", postgresql.JSONB, nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("onset_at", sa.DateTime(timezone=True)),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("status", fact_status, nullable=False,
                  server_default=sa.text("'active'")),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.8"),
        sa.Column("importance", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("decay_score", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("source", fact_source, nullable=False),
        sa.Column("source_event_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("conversation_event.id", ondelete="SET NULL")),
        sa.Column("supersedes_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clinical_fact.id", ondelete="SET NULL")),
        sa.Column("contradicts_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clinical_fact.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.execute(
        "CREATE INDEX ix_clinical_fact_active "
        "ON clinical_fact (patient_id, fact_type, status, importance DESC) "
        "WHERE status = 'active';"
    )
    op.execute(
        "CREATE INDEX ix_clinical_fact_expiry "
        "ON clinical_fact (expires_at) "
        "WHERE status = 'active' AND expires_at IS NOT NULL;"
    )
    op.execute(
        "CREATE INDEX ix_clinical_fact_value_gin "
        "ON clinical_fact USING GIN (value);"
    )
    op.create_index("ix_clinical_fact_source_event",
                    "clinical_fact", ["source_event_id"])

    # ── patient_state ─────────────────────────────────────────────────────
    op.create_table(
        "patient_state",
        sa.Column("patient_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("patient.id", ondelete="CASCADE"),
                  primary_key=True),
        sa.Column("snapshot", postgresql.JSONB, nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("summary_text", sa.Text),
        sa.Column("risk_level", sa.String(16), nullable=False,
                  server_default="none"),
        sa.Column("last_consolidated_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text("now()")),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
    )
    op.execute(
        "CREATE INDEX ix_patient_state_snapshot_gin "
        "ON patient_state USING GIN (snapshot);"
    )

    # ── episodic_memory ───────────────────────────────────────────────────
    op.create_table(
        "episodic_memory",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("patient.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("event_type", episode_type, nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("importance", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("related_fact_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
                  nullable=False, server_default=sa.text("'{}'")),
        sa.Column("source_event_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("conversation_event.id", ondelete="SET NULL")),
    )
    op.create_index("ix_episodic_memory_patient_time",
                    "episodic_memory", ["patient_id", "occurred_at"])

    # ── retrieval_log ─────────────────────────────────────────────────────
    op.create_table(
        "retrieval_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("patient.id", ondelete="CASCADE"), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_text", sa.Text, nullable=False),
        sa.Column("routing_mode", sa.String(32), nullable=False),
        sa.Column("retrieved_fact_ids",
                  postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
                  nullable=False, server_default=sa.text("'{}'")),
        sa.Column("retrieved_episode_ids",
                  postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
                  nullable=False, server_default=sa.text("'{}'")),
        sa.Column("retrieved_semantic_ids",
                  postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
                  nullable=False, server_default=sa.text("'{}'")),
        sa.Column("retrieval_strategy", sa.String(128)),
        sa.Column("prompt_tokens_in", sa.Integer),
        sa.Column("completion_tokens_out", sa.Integer),
        sa.Column("answer_text", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_retrieval_log_patient_time",
                    "retrieval_log", ["patient_id", "created_at"])
    op.create_index("ix_retrieval_log_request", "retrieval_log", ["request_id"])


def downgrade() -> None:
    for table in (
        "retrieval_log", "episodic_memory", "patient_state",
        "clinical_fact", "conversation_event", "patient",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")

    for name in ("episode_type", "fact_source", "fact_status", "fact_type", "conv_role"):
        op.execute(f"DROP TYPE IF EXISTS {name};")

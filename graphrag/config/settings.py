"""
Typed configuration for the Enervera Medical GraphRAG system.

All environment variables are declared in `Settings` (a `pydantic-settings`
model). Required vars default to `None` so importing this module never fails;
callers must invoke `settings.validate_required(mode)` at startup to fail fast
with a clear error listing every missing variable.

Backward compatibility:
    The legacy `Config` class is preserved as a thin facade so existing code
    paths (`from graphrag.config.settings import Config`) keep working. New
    code should `from graphrag.config.settings import settings`.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load `.env` once at import so legacy modules that read `os.getenv` directly
# (e.g. clean_chunks.py, check_api.py) still see the values. pydantic-settings
# also reads the file; the double load is a no-op for existing env vars.
load_dotenv()


Mode = Literal["api", "cli", "ingest"]


class ConfigError(RuntimeError):
    """Raised when required configuration is missing for the requested mode."""


class Settings(BaseSettings):
    """Typed environment configuration loaded from `.env` or the process env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ----- HTTP service (FastAPI on Render) -----
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    # Optional API key for the /chat and /episodic routes. If unset, those
    # routes are open. If set, every request must send X-API-Key matching.
    API_KEY: Optional[str] = None
    # Comma-separated list of allowed CORS origins for the frontend. Use "*"
    # to allow any origin (fine while the service is gated by API_KEY, but
    # tighten this for production once the real frontend URL is known).
    # Example: "https://app.enervera.com,https://staging.enervera.com"
    CORS_ORIGINS: str = "*"

    # ----- Pinecone (vector index) -----
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "enervera"

    # ----- Neo4j (knowledge graph) -----
    NEO4J_URI: str = "bolt://127.0.0.1:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: Optional[str] = None

    # ----- Google Gemini (LLM provider: answer, classifier, analyzer, extraction, cleaning) -----
    GEMINI_API_KEY: Optional[str] = None

    # All LLM call sites use gemini-2.5-flash-lite. Override per-role via env
    # if a specific role needs the heavier gemini-2.5-flash model.
    ANSWER_MODEL: str = "gemini-2.5-flash-lite"
    EXTRACTION_MODEL: str = "gemini-2.5-flash-lite"
    CLEANING_MODEL: str = "gemini-2.5-flash-lite"
    CLASSIFIER_MODEL: str = "gemini-2.5-flash-lite"
    QUERY_ANALYZER_MODEL: str = "gemini-2.5-flash-lite"
    SUMMARIZATION_MODEL: str = "gemini-2.5-flash-lite"

    # Legacy — kept readable for backward compat but no longer required.
    OPEN_ROUTER_KEY: Optional[str] = None

    # ----- Redis (session memory; optional — falls back to in-memory) -----
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_TTL_SEC: int = 7200

    # ----- PostgreSQL (longitudinal memory; required for new memory/ subsystem) -----
    DATABASE_URL: Optional[str] = None
    MEMORY_CACHE_TTL_SEC: int = 300

    # ----- Episodic memory (Pinecone-backed, isolated from longitudinal) -----
    # Master switch for wiring the episodic layer into the main pipeline.
    # When false, the layer + its FastAPI app still work standalone, but the
    # CLI pipeline does not call it. Activated automatically when --user-id
    # is passed on the CLI.
    EPISODIC_MEMORY_ENABLED: bool = True
    PINECONE_EPISODIC_INDEX_NAME: str = "episodicmemory"
    EPISODIC_EXTRACTION_MODEL: str = "gemini-2.5-flash-lite"
    EPISODIC_CLARIFICATION_MODEL: str = "gemini-2.5-flash-lite"
    EPISODIC_CONTRADICTION_MODEL: str = "gemini-2.5-flash-lite"
    EPISODIC_COMPRESSION_MODEL: str = "gemini-2.5-flash-lite"
    EPISODIC_DEFAULT_TOP_K: int = 20      # how many to pull from Pinecone before rerank
    EPISODIC_DEFAULT_RETURN_K: int = 5    # how many to return after rerank + compression
    EPISODIC_DECAY_HALF_LIFE_DAYS: int = 14
    EPISODIC_MAX_CLARIFICATIONS_PER_TURN: int = 1  # contract — never exceed
    # Off-line ranking evaluation: when true, the retriever logs every query +
    # its ranked results to data/eval/retrieval_log.jsonl. Used by
    # `python -m episodic.eval.cli` for labeling and precision@k.
    EPISODIC_EVAL_LOGGING_ENABLED: bool = False
    EPISODIC_EVAL_LOG_PATH: str = "data/eval/retrieval_log.jsonl"
    EPISODIC_EVAL_LABELS_PATH: str = "data/eval/labels.jsonl"

    # Per-mode required-field sets. Add new modes here as needed.
    _REQUIRED_BY_MODE: ClassVar[dict[Mode, tuple[str, ...]]] = {
        "cli": ("PINECONE_API_KEY", "NEO4J_PASSWORD", "GEMINI_API_KEY"),
        "api": ("PINECONE_API_KEY", "NEO4J_PASSWORD", "GEMINI_API_KEY"),
        "ingest": ("PINECONE_API_KEY", "NEO4J_PASSWORD", "GEMINI_API_KEY"),
    }

    def validate_required(self, mode: Mode) -> None:
        """Raise ConfigError if any env var required for `mode` is unset."""
        required = self._REQUIRED_BY_MODE.get(mode)
        if required is None:
            raise ConfigError(
                f"Unknown mode '{mode}'. Expected one of: "
                f"{sorted(self._REQUIRED_BY_MODE)}"
            )
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ConfigError(
                f"Missing required environment variables for mode '{mode}': "
                + ", ".join(missing)
                + ". Set them in your .env (see .env.example) or process environment."
            )


# Module-level singleton — safe to import even when env is empty.
settings = Settings()


class Config:
    """
    Legacy facade preserved for backward compatibility.

    Prefer `from graphrag.config.settings import settings` in new code.
    Attribute names are intentionally kept to match the original Config class
    (note: NEO4J_USER aliases NEO4J_USERNAME, NEO4J_PWD aliases NEO4J_PASSWORD).
    """

    PINECONE_API_KEY = settings.PINECONE_API_KEY
    PINECONE_INDEX_NAME = settings.PINECONE_INDEX_NAME
    NEO4J_URI = settings.NEO4J_URI
    NEO4J_USER = settings.NEO4J_USERNAME
    NEO4J_PWD = settings.NEO4J_PASSWORD
    GEMINI_API_KEY = settings.GEMINI_API_KEY
    OPEN_ROUTER_KEY = settings.OPEN_ROUTER_KEY  # legacy


__all__ = ["Settings", "Config", "ConfigError", "Mode", "settings"]

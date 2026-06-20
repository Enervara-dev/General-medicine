"""
Structured-logging bootstrap.

Goal: every log line is one JSON object with at minimum
    {"event": "...", "level": "info", "timestamp": "...", "logger": "..."}
and (when set by middleware) `request_id`, `session_id`, `user_id`, `stage`.

Implementation: structlog renders the JSON; the stdlib logging module is
configured to also route through structlog so every `logging.getLogger(__name__)`
call in the existing codebase produces structured output without code changes.

If `structlog` is unavailable, falls back to plain stdlib formatting so the
service still starts (e.g. in tests without the api extra).
"""

from __future__ import annotations

import logging
import sys

from app.core.config import settings


_CONFIGURED = False


def setup_logging() -> None:
    """Idempotent. Safe to call multiple times (lifespan + tests)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    try:
        import structlog
    except ImportError:
        # Plain stdlib formatter when structlog isn't installed.
        logging.basicConfig(
            level=level,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        _CONFIGURED = True
        return

    # Shared processors for both structlog-native AND stdlib-bridged logs.
    shared_processors = [
        structlog.contextvars.merge_contextvars,  # request-scoped binding
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # 1) structlog itself — for code calling structlog.get_logger() directly.
    structlog.configure(
        processors=shared_processors + [structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # 2) stdlib `logging.getLogger(...)` — route through structlog's ProcessorFormatter
    #    so existing `logger.info(...)` calls across the codebase emit JSON too.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Replace any existing handlers (defensive against re-runs).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet third-party noise.
    for noisy in ("httpx", "httpcore", "urllib3", "pinecone", "neo4j"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True


def bind_request_context(**fields) -> None:
    """Bind request-scoped fields onto structlog's contextvars."""
    try:
        import structlog
        structlog.contextvars.bind_contextvars(**fields)
    except ImportError:
        pass


def clear_request_context() -> None:
    try:
        import structlog
        structlog.contextvars.clear_contextvars()
    except ImportError:
        pass

"""
FastAPI application factory.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT

Routes:
    GET  /health, /healthz/ready
    POST /chat, /chat/stream
    GET  /metrics
    POST /episodic/{extract,store,retrieve,clarify,context,contradictions}

The episodic router is imported from episodic.api.routes — it ships with
its own schemas + dependency injection, and is mounted under /episodic so
its routes share this app's lifespan, middleware, and auth.

This is an API-only service. There is no bundled UI. A separately hosted
frontend connects via the documented HTTP contract (see docs/FRONTEND.md).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import APIKeyMiddleware, RequestIDMiddleware, TimingMiddleware
from app.api.routes import chat, health, metrics
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.lifespan import lifespan
from episodic.api.routes import router as episodic_router


def _parse_cors_origins(raw: str) -> list[str]:
    """Split the CORS_ORIGINS setting into a clean list. "*" stays as a single wildcard."""
    if not raw or raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enervera Medical GraphRAG",
        description=(
            "Production HTTP service for the Enervera medical GraphRAG "
            "assistant. Streams answers, manages session + episodic memory, "
            "and exposes the episodic memory layer under /episodic. "
            "Frontend integration contract: see docs/FRONTEND.md."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS first (it's added LAST below — Starlette adds middlewares
    # outermost-last — so it wraps everything else and answers preflights
    # before auth runs).
    origins = _parse_cors_origins(settings.CORS_ORIGINS)
    # When using "*" the browser disallows credentials; we don't need cookie
    # auth (the frontend sends X-API-Key explicitly), so allow_credentials=False
    # keeps the wildcard usable.
    allow_credentials = origins != ["*"]

    # Middlewares are added outermost-last in Starlette. Order at runtime:
    #   CORS → RequestID → Timing → APIKey → app
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "Accept"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(chat.router)
    # Episodic memory layer mounts at /episodic/* — its router already
    # declares the prefix internally; we share lifespan + container.
    app.include_router(episodic_router)

    return app


app = create_app()

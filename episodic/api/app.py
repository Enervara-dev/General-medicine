"""
FastAPI app for the Episodic Memory Layer.

Run locally:
    uvicorn episodic.api.app:app --reload --port 8001

Or programmatically:
    from episodic.api.app import create_app
    app = create_app()
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from episodic.api.dependencies import build_container
from episodic.api.routes import router as episodic_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build singletons + ensure Pinecone index exists at startup."""
    container = build_container()
    try:
        await container.repository.ensure_index()
    except Exception as exc:
        logger.warning(
            "ensure_index failed at startup; will retry on first write: %s", exc
        )
    app.state.episodic_container = container
    yield
    # No teardown needed — Pinecone client has no async close.


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enervera — Episodic Memory Layer",
        description=(
            "Clinically-aware episodic memory for the Enervera medical AI. "
            "Isolated from longitudinal memory; backed by Pinecone."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(episodic_router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

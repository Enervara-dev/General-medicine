"""
FastAPI lifespan — startup builds the AppContainer (orchestrator + all clients),
shutdown closes the long-lived connections (Redis, Neo4j).

Phase 1: lifespan is a minimal placeholder that only sets up logging.
Phase 2 fills in container construction.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("FastAPI lifespan: starting up")
    try:
        # AppContainer construction is filled in by phase 2.
        from app.container import build_container

        container = await build_container()
        app.state.container = container
        # Existing episodic routes read app.state.episodic_container directly,
        # so expose the sub-container under that name for backward compat.
        app.state.episodic_container = container.episodic
        logger.info("AppContainer built")
        yield
    finally:
        container = getattr(app.state, "container", None)
        if container is not None:
            await container.aclose()
        logger.info("FastAPI lifespan: shut down")

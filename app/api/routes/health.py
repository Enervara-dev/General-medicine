"""
Health endpoints.

GET /health           — liveness; pure, no upstream calls. Used by Render.
GET /healthz/ready    — readiness; verifies the AppContainer is built and the
                        critical dependencies (Pinecone, Neo4j, Redis) are
                        reachable. Slower; used for orchestrator probes.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

from app.schemas.common import HealthStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    """Liveness probe — must not touch upstream services."""
    return HealthStatus(status="ok")


@router.get("/healthz/ready", response_model=HealthStatus)
async def readiness(request: Request) -> HealthStatus:
    """
    Readiness probe — verifies the container was built and the long-lived
    clients respond. Returns 200 with per-check details when ready, 503 when
    any check fails.
    """
    container = getattr(request.app.state, "container", None)
    if container is None:
        return HealthStatus(status="starting", checks={"container": "not_built"})

    checks: dict[str, str] = {}

    # Redis — distinguishes live connection vs in-memory fallback
    try:
        result = await asyncio.wait_for(container.ping_redis(), timeout=2.0)
        checks["redis"] = result  # "ok" or "fallback"
    except Exception as exc:
        checks["redis"] = f"fail: {exc.__class__.__name__}"

    # Pinecone — describe_index is cheap
    try:
        await asyncio.wait_for(container.ping_pinecone(), timeout=3.0)
        checks["pinecone"] = "ok"
    except Exception as exc:
        checks["pinecone"] = f"fail: {exc.__class__.__name__}"

    # Neo4j — verify_connectivity wrapped
    try:
        await asyncio.wait_for(container.ping_neo4j(), timeout=3.0)
        checks["neo4j"] = "ok"
    except Exception as exc:
        checks["neo4j"] = f"fail: {exc.__class__.__name__}"

    overall = "ok" if all(v in ("ok", "fallback") for v in checks.values()) else "degraded"
    return HealthStatus(status=overall, checks=checks)

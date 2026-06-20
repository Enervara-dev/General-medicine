"""
HTTP middlewares for the FastAPI service.

Mounted (outermost first):
    1. RequestIDMiddleware — assign + propagate X-Request-ID
    2. TimingMiddleware    — record request duration; update metrics counters
    3. APIKeyMiddleware    — optional X-API-Key gate on /chat + /episodic

Order matters: RequestID first so timing + auth log + 401 responses both
carry the request_id.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Awaitable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import settings
from app.core.logging import bind_request_context, clear_request_context
from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


_AUTH_EXEMPT_PREFIXES = (
    "/health",
    "/healthz",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
)
# Exact-match exemptions kept separate from prefixes (so we never accidentally
# whitelist every route by listing "/"). Empty by default.
_AUTH_EXEMPT_PATHS: frozenset[str] = frozenset()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Mint or echo X-Request-ID; expose on request.state and the response."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable["Response"]],
    ):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        # Bind request_id + path to structlog contextvars so every log line
        # emitted while this request is in flight carries them automatically.
        bind_request_context(request_id=rid, path=request.url.path, method=request.method)
        try:
            response = await call_next(request)
        finally:
            clear_request_context()
        response.headers["x-request-id"] = rid
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure request duration; update in-process metrics + log."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable["Response"]],
    ):
        from app.api.routes.metrics import METRICS  # avoid import cycle at module load

        METRICS.inflight_inc()
        t0 = time.monotonic()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = (time.monotonic() - t0) * 1000.0
            METRICS.record_request(duration_ms=duration_ms, status_code=status_code)
            METRICS.inflight_dec()
            rid = getattr(request.state, "request_id", "-")
            logger.info(
                "request_complete method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
                request.method, request.url.path, status_code, duration_ms, rid,
            )


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    When settings.API_KEY is set, require X-API-Key on /chat and /episodic
    routes. /health, /metrics, and the docs endpoints are always public.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable["Response"]],
    ):
        required = settings.API_KEY
        path = request.url.path

        exempt = path in _AUTH_EXEMPT_PATHS or any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES)
        if required and not exempt:
            supplied = request.headers.get("x-api-key")
            if supplied != required:
                rid = getattr(request.state, "request_id", None)
                body = ErrorResponse(
                    code="UNAUTHORIZED",
                    message="Missing or invalid X-API-Key header.",
                    request_id=rid,
                )
                return JSONResponse(status_code=401, content=body.model_dump())

        return await call_next(request)

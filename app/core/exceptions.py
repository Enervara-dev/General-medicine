"""
Application error types + JSON error handler registration.

Every error response goes out as:
    {"code": "STRING_CODE", "message": "human msg", "request_id": "..."}

Raising AppError or one of its subclasses from anywhere in the app produces
a consistent JSON envelope; the catch-all wraps unexpected exceptions so we
never leak a stack trace to the client (but they're still logged with the
request_id for correlation).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base for all application-level errors. Subclass to add specific codes."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class UpstreamUnavailable(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "UPSTREAM_UNAVAILABLE"


class RateLimited(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "RATE_LIMITED"


class InvalidInput(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "INVALID_INPUT"


class Unauthorized(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "UNAUTHORIZED"


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    body = ErrorResponse(
        code=exc.code,
        message=exc.message,
        request_id=_request_id(request),
        details=exc.details,
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    body = ErrorResponse(
        code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
        request_id=_request_id(request),
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = ErrorResponse(
        code="VALIDATION_ERROR",
        message="Request body failed validation",
        request_id=_request_id(request),
        details={"errors": exc.errors()},
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=body.model_dump())


async def _catch_all_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log full stack trace with the request_id; respond with a tame envelope.
    rid = _request_id(request)
    logger.exception("Unhandled exception (request_id=%s): %s", rid, exc)
    body = ErrorResponse(
        code="INTERNAL_ERROR",
        message="An unexpected error occurred. Check logs with this request_id.",
        request_id=rid,
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(Exception, _catch_all_handler)

"""Shared response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    checks: dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class MetricsSnapshot(BaseModel):
    requests_total: int
    requests_inflight: int
    errors_total: int
    latency_ms_p50: float
    latency_ms_p95: float
    pinecone_calls_total: int = 0
    neo4j_calls_total: int = 0
    llm_calls_total: int = 0
    uptime_seconds: int = 0

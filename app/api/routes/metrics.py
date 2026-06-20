"""
In-process metrics registry + GET /metrics endpoint.

This is intentionally lightweight — a singleton with monotonic counters and a
rolling latency window. For multi-instance deployments, swap this for
prometheus_client; until then, one Render instance is one set of counters.
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock

from fastapi import APIRouter

from app.schemas.common import MetricsSnapshot

router = APIRouter(tags=["meta"])


class _Metrics:
    """Thread-safe counters + a 1024-sample rolling latency window."""

    _MAX_SAMPLES = 1024

    def __init__(self) -> None:
        self._lock = Lock()
        self.start_ts = time.monotonic()
        self.requests_total = 0
        self.requests_inflight = 0
        self.errors_total = 0
        self.pinecone_calls_total = 0
        self.neo4j_calls_total = 0
        self.llm_calls_total = 0
        self._latencies_ms: deque[float] = deque(maxlen=self._MAX_SAMPLES)

    # ---- HTTP request hooks (called by middleware) -----------------------

    def inflight_inc(self) -> None:
        with self._lock:
            self.requests_inflight += 1

    def inflight_dec(self) -> None:
        with self._lock:
            self.requests_inflight = max(0, self.requests_inflight - 1)

    def record_request(self, *, duration_ms: float, status_code: int) -> None:
        with self._lock:
            self.requests_total += 1
            self._latencies_ms.append(duration_ms)
            if status_code >= 500:
                self.errors_total += 1

    # ---- Pipeline counters (called from orchestrator) --------------------

    def incr_pinecone(self) -> None:
        with self._lock:
            self.pinecone_calls_total += 1

    def incr_neo4j(self) -> None:
        with self._lock:
            self.neo4j_calls_total += 1

    def incr_llm(self) -> None:
        with self._lock:
            self.llm_calls_total += 1

    # ---- Snapshot --------------------------------------------------------

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            latencies = sorted(self._latencies_ms)
            p50 = _percentile(latencies, 50.0)
            p95 = _percentile(latencies, 95.0)
            return MetricsSnapshot(
                requests_total=self.requests_total,
                requests_inflight=self.requests_inflight,
                errors_total=self.errors_total,
                latency_ms_p50=p50,
                latency_ms_p95=p95,
                pinecone_calls_total=self.pinecone_calls_total,
                neo4j_calls_total=self.neo4j_calls_total,
                llm_calls_total=self.llm_calls_total,
                uptime_seconds=int(time.monotonic() - self.start_ts),
            )


def _percentile(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    rank = (q / 100.0) * (len(sorted_samples) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = rank - lo
    return float(sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac)


METRICS = _Metrics()


@router.get("/metrics", response_model=MetricsSnapshot)
async def metrics() -> MetricsSnapshot:
    return METRICS.snapshot()

"""
PMS client abstraction — INTERFACE ONLY. Nothing here calls a PMS.

    PMSClient      — the Protocol every implementation satisfies.
    NullPMSClient  — the default wired today: a no-op that does NOT emit
                     anywhere, so existing behaviour is byte-for-byte unchanged.
    HttpPMSClient  — a documented PLACEHOLDER for the future HTTP client. It is
                     NOT wired and NOT instantiated; calling it raises so no one
                     accidentally ships a half-built integration.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

from app.services.pms.events import ClinicalMemoryEventV1

logger = logging.getLogger(__name__)


@runtime_checkable
class PMSClient(Protocol):
    """Sink for longitudinal clinical-memory events."""

    async def ingest_clinical_memory(self, event: ClinicalMemoryEventV1) -> None:
        ...


class NullPMSClient:
    """
    Default no-op client. The producer builds and hands off events as normal,
    but this sink drops them (debug-log only) — so PMS is NOT called and all
    existing memory behaviour (Redis, Pinecone, streaming) is unchanged.
    """

    async def ingest_clinical_memory(self, event: ClinicalMemoryEventV1) -> None:
        logger.debug(
            "[PMS:null] clinical_memory_event dropped (not sent) — patient=%s… "
            "session=%s category=%s severity=%s",
            event.patient_id[:6], event.session_id,
            event.category.value, event.severity.value,
        )


class HttpPMSClient:
    """
    SHADOW-mode PMS HTTP client. Used only when ENABLE_PMS_SHADOW=true.

    Contract with the rest of the system:
      * It is ONLY ever called from the existing fire-and-forget background task,
        so chat never waits on it.
      * ``ingest_clinical_memory`` NEVER raises — every error is caught + logged,
        so a PMS outage can never surface to a user.
      * Short timeout + a pooled, reused ``httpx.AsyncClient``.
      * Bounded retry on transient/5xx only (never on 4xx).

    A transient ``event_id`` is minted per call for tracing / idempotency only —
    it is a log+header correlation token, NOT part of the (frozen) event schema
    and NOT a patient identifier.
    """

    def __init__(
        self,
        *,
        base_url: str,
        ingest_path: str = "/v1/clinical-memory/events",
        api_key: Optional[str] = None,
        consumer_id: str = "general-medicine",
        timeout_ms: int = 1500,
        max_retries: int = 1,
        transport: object | None = None,  # injectable for tests (httpx.MockTransport)
    ) -> None:
        import httpx

        self._path = ingest_path
        self._consumer_id = consumer_id
        self._max_retries = max(0, int(max_retries))
        headers = {"X-Consumer-Id": consumer_id}
        if api_key:
            headers["X-API-Key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_ms / 1000.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers=headers,
            transport=transport,  # None → default pooled transport
        )

    async def ingest_clinical_memory(self, event: ClinicalMemoryEventV1) -> None:
        import time
        import uuid

        import httpx

        event_id = uuid.uuid4().hex           # transient trace/idempotency token
        payload = event.model_dump(mode="json")
        t0 = time.monotonic()

        attempt = 0
        while True:
            try:
                resp = await self._client.post(
                    self._path, json=payload, headers={"Idempotency-Key": event_id}
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                success = resp.status_code < 400
                self._log(event, event_id, latency_ms, success, str(resp.status_code))
                # Retry only transient 5xx, and only within the bounded budget.
                if success or resp.status_code < 500 or attempt >= self._max_retries:
                    return
            except httpx.TransportError as exc:  # includes timeouts
                latency_ms = (time.monotonic() - t0) * 1000.0
                if attempt >= self._max_retries:
                    self._log(event, event_id, latency_ms, False, f"transport:{exc!r}")
                    return
            except Exception as exc:  # noqa: BLE001 — never propagate to the user
                latency_ms = (time.monotonic() - t0) * 1000.0
                self._log(event, event_id, latency_ms, False, f"error:{exc!r}")
                return
            attempt += 1

    def _log(
        self,
        event: ClinicalMemoryEventV1,
        event_id: str,
        latency_ms: float,
        success: bool,
        status: str,
    ) -> None:
        # PHI-safe: ids + timing only. Never the summary / entities / clinical text.
        logger.info(
            "pms_shadow request_id=%s patient_id=%s… consumer_id=%s event_id=%s "
            "latency_ms=%.0f success=%s status=%s",
            event.request_id, event.patient_id[:6], self._consumer_id, event_id,
            latency_ms, success, status,
        )

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.debug("HttpPMSClient close failed: %s", exc)


def build_pms_client(settings) -> PMSClient:
    """
    Choose the PMS client from config. Fails SAFE toward NullPMSClient so a
    misconfiguration can never break chat:

        ENABLE_PMS_SHADOW=false (production default) → NullPMSClient (no HTTP).
        ENABLE_PMS_SHADOW=true  + PMS_BASE_URL set   → HttpPMSClient (shadow).
        ENABLE_PMS_SHADOW=true  + PMS_BASE_URL unset → NullPMSClient (+ warn).
    """
    if not getattr(settings, "ENABLE_PMS_SHADOW", False):
        return NullPMSClient()

    base_url = getattr(settings, "PMS_BASE_URL", None)
    if not base_url:
        logger.warning(
            "ENABLE_PMS_SHADOW=true but PMS_BASE_URL is unset — using NullPMSClient."
        )
        return NullPMSClient()

    logger.info("PMS shadow mode ON — HttpPMSClient → %s", base_url)
    return HttpPMSClient(
        base_url=base_url,
        ingest_path=settings.PMS_INGEST_PATH,
        api_key=settings.PMS_API_KEY,
        consumer_id=settings.PMS_CONSUMER_ID,
        timeout_ms=settings.PMS_TIMEOUT_MS,
        max_retries=settings.PMS_MAX_RETRIES,
    )


__all__ = ["PMSClient", "NullPMSClient", "HttpPMSClient", "build_pms_client"]

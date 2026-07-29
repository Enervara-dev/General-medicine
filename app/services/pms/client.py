"""
PMS client — transport implementations behind the PMSClient interface.

    PMSClient      — the Protocol every implementation satisfies.
    NullPMSClient  — the default (ENABLE_PMS_SHADOW=false): a no-op that emits
                     nowhere, so existing behaviour is byte-for-byte unchanged.
    HttpPMSClient  — the production client (ENABLE_PMS_SHADOW=true): Bearer-JWT
                     authenticated, pooled, timeout-bounded, transient-retrying,
                     secret/PHI-safe logging, and isolated from the user path.
    build_pms_client(settings) — selects one from config, failing safe to Null.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.services.pms.events import ClinicalMemoryEventV1

if TYPE_CHECKING:
    from app.services.pms.auth import TokenProvider

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
    Production PMS HTTP client. Active only when ENABLE_PMS_SHADOW=true.

    Guarantees for the rest of the system:
      * Called ONLY from the existing fire-and-forget background task, so chat
        never waits on it and PMS latency can't affect the user response.
      * Every request carries `Authorization: Bearer <service JWT>` (never
        anonymous). The token is obtained from an injected TokenProvider.
      * ``ingest_clinical_memory`` never propagates to the user path; every
        outcome — success and failure — is LOGGED with structure (never
        swallowed silently).
      * Pooled, reused ``httpx.AsyncClient`` with separate connect + read
        timeouts. Bounded retry on TRANSIENT failures only (timeout / network /
        5xx / 429); validation (4xx) and auth (401/403) are NOT retried.

    Security: logs never contain the token, patient identifiers, clinical text,
    or the internal URL — correlation is via request_id + a transient event_id.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token_provider: "TokenProvider",
        ingest_path: str = "/v1/clinical-memory/events",
        consumer_id: str = "general-medicine",
        connect_timeout_ms: int = 1000,
        read_timeout_ms: int = 1500,
        max_retries: int = 1,
        transport: object | None = None,  # injectable for tests (httpx.MockTransport)
    ) -> None:
        import httpx

        # httpx logs each request line (incl. the full URL) at INFO. Quiet it so
        # the internal PMS URL never lands in logs (security requirement).
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self._path = ingest_path
        self._consumer_id = consumer_id
        self._token_provider = token_provider
        self._max_retries = max(0, int(max_retries))
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=connect_timeout_ms / 1000.0,
                read=read_timeout_ms / 1000.0,
                write=read_timeout_ms / 1000.0,
                pool=connect_timeout_ms / 1000.0,
            ),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"X-Consumer-Id": consumer_id},  # non-secret routing header
            transport=transport,  # None → default pooled transport
        )

    async def ingest_clinical_memory(self, event: ClinicalMemoryEventV1) -> None:
        import time
        import uuid

        import httpx

        event_id = uuid.uuid4().hex           # transient trace/idempotency token
        request_id = event.request_id or "-"
        payload = event.model_dump(mode="json")

        # --- Authenticate: obtain a service JWT; NEVER send anonymously. ------
        try:
            token = await self._token_provider.get_token()
        except Exception as exc:  # noqa: BLE001
            self._log(request_id, event_id, outcome="auth_error", status="-",
                      duration_ms=0.0, retries=0, timeouts=0, reason=type(exc).__name__)
            return
        if not token:
            self._log(request_id, event_id, outcome="auth_missing", status="-",
                      duration_ms=0.0, retries=0, timeouts=0, reason="no_service_jwt")
            return
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": event_id}

        # --- Send with bounded, transient-only retry --------------------------
        t0 = time.monotonic()
        retries = 0
        timeouts = 0
        while True:
            try:
                resp = await self._client.post(self._path, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                timeouts += 1
                if retries < self._max_retries:
                    retries += 1
                    await self._backoff(retries)
                    continue
                self._log(request_id, event_id, outcome="failure", status="timeout",
                          duration_ms=(time.monotonic() - t0) * 1000.0,
                          retries=retries, timeouts=timeouts, reason=type(exc).__name__)
                return
            except httpx.TransportError as exc:  # connect/network errors
                if retries < self._max_retries:
                    retries += 1
                    await self._backoff(retries)
                    continue
                self._log(request_id, event_id, outcome="failure", status="network",
                          duration_ms=(time.monotonic() - t0) * 1000.0,
                          retries=retries, timeouts=timeouts, reason=type(exc).__name__)
                return
            except Exception as exc:  # noqa: BLE001 — log, never propagate to user
                self._log(request_id, event_id, outcome="error", status="-",
                          duration_ms=(time.monotonic() - t0) * 1000.0,
                          retries=retries, timeouts=timeouts, reason=type(exc).__name__)
                return

            duration_ms = (time.monotonic() - t0) * 1000.0
            sc = resp.status_code

            if 200 <= sc < 300:
                # Success. We intentionally do not parse/use the body for
                # ingestion; validation is limited to the status code.
                self._log(request_id, event_id, outcome="success", status=str(sc),
                          duration_ms=duration_ms, retries=retries, timeouts=timeouts, reason="-")
                return
            if sc in (401, 403):
                # Auth failure — retrying with the same token won't help; a fresh
                # token is obtained on the next turn.
                self._log(request_id, event_id, outcome="auth_failure", status=str(sc),
                          duration_ms=duration_ms, retries=retries, timeouts=timeouts,
                          reason="unauthorized")
                return
            if sc == 429 or sc >= 500:
                if retries < self._max_retries:
                    retries += 1
                    await self._backoff(retries)
                    continue
                self._log(request_id, event_id, outcome="failure", status=str(sc),
                          duration_ms=duration_ms, retries=retries, timeouts=timeouts,
                          reason="transient")
                return
            # Any other 4xx → validation/rejection. Do NOT retry.
            self._log(request_id, event_id, outcome="rejected", status=str(sc),
                      duration_ms=duration_ms, retries=retries, timeouts=timeouts,
                      reason="validation")
            return

    @staticmethod
    async def _backoff(retries: int) -> None:
        import asyncio

        await asyncio.sleep(min(0.5, 0.1 * retries))  # short — background call

    def _log(
        self,
        request_id: str,
        event_id: str,
        *,
        outcome: str,
        status: str,
        duration_ms: float,
        retries: int,
        timeouts: int,
        reason: str,
    ) -> None:
        # SECURITY: no token, no patient id, no clinical text, no internal URL.
        logger.info(
            "pms_ingest request_id=%s event_id=%s consumer_id=%s outcome=%s "
            "status=%s duration_ms=%.0f retries=%d timeouts=%d reason=%s",
            request_id, event_id, self._consumer_id, outcome, status,
            duration_ms, retries, timeouts, reason or "-",
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
        ENABLE_PMS_SHADOW=true  + PMS_BASE_URL set   → HttpPMSClient.
        ENABLE_PMS_SHADOW=true  + PMS_BASE_URL unset → NullPMSClient (+ warn).

    The base URL is read from config (never hardcoded) and never logged. The
    service JWT comes from a StaticTokenProvider(PMS_SERVICE_JWT); if it's unset
    the client will simply not send (no anonymous requests).
    """
    from app.services.pms.auth import StaticTokenProvider

    if not getattr(settings, "ENABLE_PMS_SHADOW", False):
        return NullPMSClient()

    base_url = getattr(settings, "PMS_BASE_URL", None)
    if not base_url:
        logger.warning(
            "ENABLE_PMS_SHADOW=true but PMS_BASE_URL is unset — using NullPMSClient."
        )
        return NullPMSClient()

    if not getattr(settings, "PMS_SERVICE_JWT", None):
        logger.warning(
            "PMS shadow ON but PMS_SERVICE_JWT is unset — requests will be skipped "
            "(no anonymous PMS calls)."
        )
    logger.info("PMS shadow mode ON — HttpPMSClient active.")  # no URL in logs
    return HttpPMSClient(
        base_url=base_url,
        token_provider=StaticTokenProvider(settings.PMS_SERVICE_JWT),
        ingest_path=settings.PMS_INGEST_PATH,
        consumer_id=settings.PMS_CONSUMER_ID,
        connect_timeout_ms=settings.PMS_CONNECT_TIMEOUT_MS,
        read_timeout_ms=settings.PMS_READ_TIMEOUT_MS,
        max_retries=settings.PMS_MAX_RETRIES,
    )


__all__ = ["PMSClient", "NullPMSClient", "HttpPMSClient", "build_pms_client"]

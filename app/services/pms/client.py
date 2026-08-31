"""
PMS client — transport implementations behind the PMSClient interface.

    PMSClient      — the Protocol every implementation satisfies.
    NullPMSClient  — the default (ENABLE_PMS_SHADOW=false): a no-op that emits
                     nowhere, so existing behaviour is byte-for-byte unchanged.
    HttpPMSClient  — the production client (ENABLE_PMS_SHADOW=true): SigV4-signed
                     (VPC Lattice, AWS_IAM), pooled, timeout-bounded,
                     transient-retrying, PHI-safe logging, and isolated from
                     the user path.
    build_pms_client(settings) — selects one from config, failing safe to Null.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.services.pms.assertions import USER_ASSERTION_HEADER
from app.services.pms.events import PmsMemoryEventV1
from app.services.pms.signing import SigningError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx


logger = logging.getLogger(__name__)


def _retry_after_seconds(resp) -> "float | None":
    """
    Parse a ``Retry-After`` header into seconds.

    Accepts both forms RFC 9110 allows: delay-seconds, and an HTTP-date.
    Returns None when absent or unparseable, so the caller falls back to its
    own backoff. Never raises — this runs inside the fire-and-forget path.
    """
    raw = None
    try:
        raw = resp.headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - defensive; header access must not throw
        return None
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(int(raw)))
    except (TypeError, ValueError):
        pass
    try:
        import datetime as _dt
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        delta = (when - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
        return max(0.0, delta)
    except Exception:  # noqa: BLE001
        return None


@runtime_checkable
class PMSClient(Protocol):
    """Sink for longitudinal clinical-memory events."""

    async def ingest_clinical_memory(
        self, event: PmsMemoryEventV1, *, user_assertion: str | None, request_id: str = "-"
    ) -> None:
        ...


class NullPMSClient:
    """
    Default no-op client. The producer builds and hands off events as normal,
    but this sink drops them (debug-log only) — so PMS is NOT called and all
    existing memory behaviour (Redis, Pinecone, streaming) is unchanged.
    """

    async def ingest_clinical_memory(
        self, event: PmsMemoryEventV1, *, user_assertion: str | None = None, request_id: str = "-"
    ) -> None:
        logger.debug(
            "[PMS:null] memory_event dropped (not sent) — event=%s conversation=%s "
            "category=%s severity=%s",
            event.event_id, event.conversation_id,
            event.category.value, event.severity.value,
        )


class HttpPMSClient:
    """
    Production PMS HTTP client. Active only when ENABLE_PMS_SHADOW=true.

    Guarantees for the rest of the system:
      * Called ONLY from the existing fire-and-forget background task, so chat
        never waits on it and PMS latency can't affect the user response.
      * ``ingest_clinical_memory`` never propagates to the user path; every
        outcome — success and failure — is LOGGED with structure (never
        swallowed silently).
      * Pooled, reused ``httpx.AsyncClient`` with separate connect + read
        timeouts. Bounded retry on TRANSIENT failures only (timeout / network /
        5xx / 429); validation (4xx) and auth (401/403) are NOT retried.

    Security: logs never contain patient identifiers, clinical text, or the
    internal URL — correlation is via request_id + a transient event_id.

    AUTHENTICATION: every request is signed with AWS SigV4 (service
    ``vpc-lattice-svcs``) using the ECS task role, so VPC Lattice authenticates
    the calling workload. That proves WHICH WORKLOAD is calling and nothing
    more — patient scope is a separate concern, see ``assertions.py``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        ingest_path: str = "/v1/memory/events",
        consumer_id: str = "general-medicine",
        connect_timeout_ms: int = 1000,
        read_timeout_ms: int = 1500,
        max_retries: int = 1,
        max_retry_after_s: float = 5.0,
        auth: "httpx.Auth | None" = None,       # the SigV4 signer
        transport: object | None = None,  # injectable for tests (httpx.MockTransport)
    ) -> None:
        import httpx

        # httpx logs each request line (incl. the full URL) at INFO. Quiet it so
        # the internal PMS URL never lands in logs (security requirement).
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self._path = ingest_path
        # Retained as a LOG label only. No longer sent as a header: under
        # Lattice, consumer identity is derived from the authenticated
        # principal, and a self-asserted header is a second, spoofable source.
        self._consumer_id = consumer_id
        self._max_retries = max(0, int(max_retries))
        self._max_retry_after_s = max(0.0, float(max_retry_after_s))
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=connect_timeout_ms / 1000.0,
                read=read_timeout_ms / 1000.0,
                write=read_timeout_ms / 1000.0,
                pool=connect_timeout_ms / 1000.0,
            ),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            auth=auth,            # None → unsigned (tests / explicit opt-out)
            transport=transport,  # None → default pooled transport
        )

    async def ingest_clinical_memory(
        self, event: PmsMemoryEventV1, *, user_assertion: str | None, request_id: str = "-"
    ) -> None:
        import time

        import httpx

        # Deterministic: identical clinical content yields an identical key on every
        # delivery attempt, so PMS can collapse redeliveries. Doubles as the
        # correlation id, which makes a replay traceable across log lines.
        idem_key = event.idempotency_key()
        event_id = idem_key[:16]

        # HARD STOP. A missing assertion means the Backend is misconfigured, not that
        # the caller is anonymous. Falling back to an unauthenticated patient id would
        # manufacture a patient-memory write nobody authenticated, so the request is
        # not sent at all. The chat turn is unaffected (shadow memory is non-fatal).
        if not user_assertion:
            self._log(request_id, event_id, outcome="assertion_missing", status="-",
                      duration_ms=0.0, retries=0, timeouts=0,
                      reason="no user assertion; request not sent")
            return

        payload = event.model_dump(mode="json")
        headers = {
            "Idempotency-Key": idem_key,
            # Forwarded verbatim. GM never inspects, decodes, or rewrites it.
            USER_ASSERTION_HEADER: user_assertion,
        }

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
            except SigningError as exc:
                # Cannot authenticate to Lattice (no credentials / signing fault).
                # Not retried: a retry signs with the same broken chain.
                self._log(request_id, event_id, outcome="signing_error", status="-",
                          duration_ms=(time.monotonic() - t0) * 1000.0,
                          retries=retries, timeouts=timeouts, reason=str(exc)[:60])
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
                # Rejected by PMS. Not retried: an identical request would be
                # rejected identically.
                self._log(request_id, event_id, outcome="auth_failure", status=str(sc),
                          duration_ms=duration_ms, retries=retries, timeouts=timeouts,
                          reason="unauthorized")
                return
            if sc == 429 or sc >= 500:
                if retries < self._max_retries:
                    retries += 1
                    await self._backoff(retries, _retry_after_seconds(resp))
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

    async def _backoff(self, retries: int, retry_after: float | None = None) -> None:
        import asyncio

        delay = min(0.5, 0.1 * retries)  # short — background call
        if retry_after is not None:
            # Respect server pacing, but stay bounded: this runs on a background
            # task and must not hold a slot for the server's full suggestion.
            delay = min(max(retry_after, delay), self._max_retry_after_s)
        await asyncio.sleep(delay)

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
        # SECURITY: no patient id, no clinical text, no internal URL.
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
        ENABLE_PMS_SHADOW=true  + PMS_BASE_URL set   → HttpPMSClient (SigV4).
        ENABLE_PMS_SHADOW=true  + PMS_BASE_URL unset → NullPMSClient (+ warn).
        ENABLE_PMS_SHADOW=true  + non-HTTPS base URL → NullPMSClient (+ warn).
        ENABLE_PMS_SHADOW=true  + malformed base URL → NullPMSClient (+ warn).

    The base URL is read from config (never hardcoded) and never logged.

    TLS is MANDATORY. Clinical content and an AWS signature must never travel
    in plaintext, so a non-HTTPS URL degrades to NullPMSClient rather than
    downgrading the transport.

    AUTHENTICATION: SigV4 over VPC Lattice using the ECS task role. Signing is
    on by default and can be disabled ONLY via PMS_SIGV4_ENABLED=false, which
    exists for local testing against a stub and is unsafe against real PMS.
    """
    if not getattr(settings, "ENABLE_PMS_SHADOW", False):
        return NullPMSClient()

    base_url = getattr(settings, "PMS_BASE_URL", None)
    if not base_url:
        logger.warning(
            "ENABLE_PMS_SHADOW=true but PMS_BASE_URL is unset — using NullPMSClient."
        )
        return NullPMSClient()

    # TLS gate. Never log the URL itself — only the scheme.
    from urllib.parse import urlparse

    try:
        parsed = urlparse(str(base_url))
    except Exception:  # noqa: BLE001
        logger.warning("PMS_BASE_URL is unparseable — using NullPMSClient.")
        return NullPMSClient()
    if parsed.scheme != "https":
        logger.warning(
            "PMS_BASE_URL scheme is %r, not https — refusing plaintext clinical "
            "transport; using NullPMSClient.",
            parsed.scheme or "(none)",
        )
        return NullPMSClient()
    if not parsed.hostname:
        logger.warning("PMS_BASE_URL has no host — using NullPMSClient.")
        return NullPMSClient()

    # SigV4 signer. Built here (not inside the client) so the transport seam
    # stays injectable and tests can construct an unsigned client directly.
    auth = None
    if getattr(settings, "PMS_SIGV4_ENABLED", True):
        from app.services.pms.signing import SigV4RequestSigner

        auth = SigV4RequestSigner(
            service=getattr(settings, "PMS_SIGV4_SERVICE", "vpc-lattice-svcs"),
            region=getattr(settings, "PMS_SIGV4_REGION", "ap-south-1"),
        )
    else:
        logger.warning(
            "PMS_SIGV4_ENABLED=false — requests are UNSIGNED. Local testing only."
        )

    logger.info(
        "PMS shadow mode ON — HttpPMSClient active (sigv4=%s service=%s region=%s).",
        bool(auth),
        getattr(settings, "PMS_SIGV4_SERVICE", "vpc-lattice-svcs"),
        getattr(settings, "PMS_SIGV4_REGION", "ap-south-1"),
    )  # no URL in logs
    return HttpPMSClient(
        base_url=base_url,
        auth=auth,
        ingest_path=settings.PMS_INGEST_PATH,
        consumer_id=settings.PMS_CONSUMER_ID,
        connect_timeout_ms=settings.PMS_CONNECT_TIMEOUT_MS,
        read_timeout_ms=settings.PMS_READ_TIMEOUT_MS,
        max_retries=settings.PMS_MAX_RETRIES,
        max_retry_after_s=getattr(settings, "PMS_MAX_RETRY_AFTER_S", 5.0),
    )


__all__ = ["PMSClient", "NullPMSClient", "HttpPMSClient", "build_pms_client"]

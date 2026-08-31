"""
SigV4 request signing for outbound PMS calls via VPC Lattice.

GM reaches PMS through a VPC Lattice service configured with ``AWS_IAM`` auth.
Lattice authenticates the *calling workload* by verifying an AWS SigV4 signature
computed with the caller's IAM credentials — for GM that is the ECS task role,
resolved through the standard credential provider chain (container credentials
endpoint in ECS; ambient profile/env locally).

    SigV4RequestSigner — an ``httpx.Auth`` that signs each request in flight.
    SigningError       — raised when signing is impossible (no credentials).

WHY httpx.Auth AND NOT A TOKEN PROVIDER
    SigV4 is not a bearer credential. The signature covers the request itself —
    method, canonical path, query string, signed headers, and a payload hash — so
    it cannot be produced ahead of time by anything with a ``get_token()``-shaped
    interface. Signing must happen per request, after the body is known, which is
    exactly the seam ``httpx.Auth`` provides.

PAYLOAD HASHING
    VPC Lattice requires the literal ``UNSIGNED-PAYLOAD`` value in
    ``x-amz-content-sha256`` rather than a body digest. botocore's SigV4Auth
    honours a pre-set ``X-Amz-Content-SHA256`` header when building the canonical
    request, so setting it before ``add_auth`` produces the signature Lattice
    expects and keeps the header inside the signed set.

CONCURRENCY
    Credential resolution and refresh are blocking (the ECS credentials endpoint
    is an HTTP call). Signing therefore runs in a worker thread so the event loop
    serving chat is never blocked. Signing itself is HMAC-cheap; the thread hop
    matters only on refresh.

SECURITY
    Nothing here logs the secret key, the session token, or the signature.

    botocore itself logs the full canonical request — which contains the
    ``x-amz-security-token`` header — at DEBUG. That is a credential leak the
    moment the service runs at DEBUG, so the ``botocore`` loggers are pinned to
    WARNING here, mirroring how the PMS client pins ``httpx`` to keep the
    internal URL out of logs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD"

# Headers botocore must not see when building the canonical request: httpx sets
# them per-connection and Lattice does not expect them in the signed set.
_SKIP_HEADERS = frozenset({"connection", "accept-encoding", "user-agent"})


class SigningError(RuntimeError):
    """Raised when a request cannot be signed (typically: no credentials)."""


class SigV4RequestSigner(httpx.Auth):
    """
    Signs outbound requests with AWS SigV4 for a given service + region.

    Credentials come from the standard botocore chain, so in ECS this resolves
    the task role automatically with no configuration. The resolved credential
    object is cached; botocore refreshes it internally as it nears expiry, which
    means the expiry problem of the removed static-token path does not recur.
    """

    # httpx must materialise the body before the auth flow so the signer can see
    # a stable request (and set Content-Length inside the signed header set).
    requires_request_body = True

    def __init__(
        self,
        *,
        service: str = "vpc-lattice-svcs",
        region: str = "ap-south-1",
        session: Any | None = None,
    ) -> None:
        # botocore logs the canonical request (incl. x-amz-security-token) and
        # credential-provider detail at DEBUG. Pin these loggers so temporary
        # ECS task-role credentials can never reach the log stream.
        for _name in ("botocore.auth", "botocore.credentials", "botocore.endpoint"):
            logging.getLogger(_name).setLevel(logging.WARNING)

        self._service = service
        self._region = region
        self._session = session
        self._credentials: Any = None

    # -- credential resolution ------------------------------------------------

    def _resolve_credentials(self) -> Any:
        """Resolve (once) and cache the botocore credential object. Blocking."""
        if self._credentials is not None:
            return self._credentials
        try:
            if self._session is None:
                from botocore.session import Session

                self._session = Session()
            creds = self._session.get_credentials()
        except Exception as exc:  # noqa: BLE001
            raise SigningError(f"credential chain failed: {type(exc).__name__}") from exc
        if creds is None:
            raise SigningError("no AWS credentials available")
        self._credentials = creds
        return creds

    # -- signing --------------------------------------------------------------

    def _sign(
        self, method: str, url: str, headers: dict[str, str], body: bytes
    ) -> dict[str, str]:
        """
        Build an AWSRequest mirroring the outbound httpx request, sign it, and
        return ONLY the headers that must be copied back. Blocking (runs in a
        worker thread).
        """
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        creds = self._resolve_credentials()

        to_sign = {
            k: v for k, v in headers.items() if k.lower() not in _SKIP_HEADERS
        }
        # Lattice expects the literal sentinel, not a body digest. Setting it
        # before add_auth also pulls it into the signed header set.
        to_sign["X-Amz-Content-SHA256"] = UNSIGNED_PAYLOAD

        aws_req = AWSRequest(method=method, url=url, data=body, headers=to_sign)
        try:
            SigV4Auth(creds, self._service, self._region).add_auth(aws_req)
        except Exception as exc:  # noqa: BLE001
            raise SigningError(f"sigv4 signing failed: {type(exc).__name__}") from exc

        out = {"X-Amz-Content-SHA256": UNSIGNED_PAYLOAD}
        for name in ("Authorization", "X-Amz-Date", "X-Amz-Security-Token"):
            value = aws_req.headers.get(name)
            if value:
                out[name] = value
        if "Authorization" not in out:
            raise SigningError("signing produced no Authorization header")
        return out

    # -- httpx.Auth hooks -----------------------------------------------------

    def sync_auth_flow(self, request: httpx.Request):
        signed = self._sign(
            request.method, str(request.url), dict(request.headers), request.content
        )
        request.headers.update(signed)
        yield request

    async def async_auth_flow(self, request: httpx.Request):
        # Off-loop: credential refresh performs network I/O.
        signed = await asyncio.to_thread(
            self._sign,
            request.method,
            str(request.url),
            dict(request.headers),
            request.content,
        )
        request.headers.update(signed)
        yield request


__all__ = ["SigV4RequestSigner", "SigningError", "UNSIGNED_PAYLOAD"]

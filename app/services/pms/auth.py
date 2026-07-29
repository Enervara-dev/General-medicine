"""
Service-authentication for outbound PMS calls.

The HTTP client obtains a bearer token from a ``TokenProvider`` — an injected,
configured component — rather than a hardcoded token. This keeps the token
source swappable (a static provisioned JWT today; a client-credentials / OIDC
issuer with refresh later) without touching the client.

Security: providers must never log or expose the token.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TokenProvider(Protocol):
    """Supplies the service JWT for `Authorization: Bearer <token>`."""

    async def get_token(self) -> Optional[str]:
        """Return a valid token, or None when unavailable (→ do not send)."""
        ...


class StaticTokenProvider:
    """
    Service JWT provisioned out-of-band via configuration (secret manager / env).

    No network. Returns the pre-issued token as-is. Swap for a dynamic provider
    (fetch + cache + refresh) without changing HttpPMSClient. Returns None when
    no token is configured, so the client sends nothing rather than going
    anonymous.
    """

    def __init__(self, token: Optional[str]) -> None:
        self._token = (token or "").strip() or None

    async def get_token(self) -> Optional[str]:
        return self._token


__all__ = ["TokenProvider", "StaticTokenProvider"]

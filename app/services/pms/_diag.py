"""
TEMPORARY DIAGNOSTICS for the GM -> PMS boundary.  [PMS-DIAG]

=============================================================================
 THIS MODULE IS DISPOSABLE. To remove the diagnostics completely:

     1. delete this file
     2. delete every line tagged  # [PMS-DIAG]   (they are all one-liners):
            git grep -n "\\[PMS-DIAG\\]"

 Nothing else imports it and nothing depends on its return values, so removal
 cannot change behaviour.
=============================================================================

Why this exists: PMS was receiving zero events while /chat returned 200, and
the failure is silent by design (shadow memory is non-fatal). These lines make
each gate on the path state, once per turn, whether it passed.

WHAT IS LOGGED
    presence booleans only for the assertion and the patient id
    the client class name and transport mode
    the base URL scheme and hostname
    whether the ingest call was reached
    the outcome/status the client already computed

WHAT IS NEVER LOGGED
    the assertion or any token value, in whole or in part
    the patient id value, the session id, or any clinical content
    userinfo, query string, or path from the base URL
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger("pms.diag")

_TAG = "[PMS-DIAG]"

# Last transport mode resolved by build_pms_client. A module global rather than
# an attribute on the client so the call site is a single deletable line and the
# client class is untouched.
_TRANSPORT: str = "n/a"


def note_transport(mode: str) -> None:
    global _TRANSPORT
    _TRANSPORT = str(mode)


def _endpoint(client: Any) -> str:
    """Scheme + hostname only. Never userinfo, port-auth, path or query."""
    try:
        base = getattr(getattr(client, "_client", None), "base_url", None)
        if base is None:
            return "scheme=- host=-"
        parts = urlsplit(str(base))
        # `hostname` (not `netloc`) deliberately: netloc can carry user:pass@.
        return f"scheme={parts.scheme or '-'} host={parts.hostname or '-'}"
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return "scheme=? host=?"


def log_client_selected(client: Any) -> None:
    """Once at startup: which client the config actually resolved to."""
    try:
        logger.warning(
            "%s client=%s transport=%s %s",
            _TAG,
            type(client).__name__,
            _TRANSPORT,
            _endpoint(client),
        )
    except Exception:  # noqa: BLE001
        pass


def log_emit_attempt(identity: Any, client: Any) -> None:
    """Per turn, at the producer: the two gate inputs, as booleans only."""
    try:
        logger.warning(
            "%s emit patient_id_present=%s assertion_present=%s client=%s %s",
            _TAG,
            identity.patient_id is not None,
            bool(getattr(identity, "user_assertion", None)),
            type(client).__name__,
            _endpoint(client),
        )
    except Exception:  # noqa: BLE001
        pass


def log_gate(reason: str) -> None:
    """A gate stopped the send before any request was made."""
    try:
        logger.warning("%s not_sent reason=%s", _TAG, reason)
    except Exception:  # noqa: BLE001
        pass


def log_reached_ingest() -> None:
    """The ingest call was actually entered (not that it was sent)."""
    try:
        logger.warning("%s reached_ingest_call=true", _TAG)
    except Exception:  # noqa: BLE001
        pass


def log_outcome(outcome: str, status: str) -> None:
    """Terminal result. `status` is the HTTP status code or a dash."""
    try:
        logger.warning("%s outcome=%s http_status=%s", _TAG, outcome, status)
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "note_transport",
    "log_client_selected",
    "log_emit_attempt",
    "log_gate",
    "log_outcome",
    "log_reached_ingest",
]

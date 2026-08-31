"""
Identity model for the request path.

This service does NOT authenticate callers (see docs / audit): the `user_id`
arrives from the trusted upstream Backend as a request field. These types make
that identity a first-class, strongly-typed object instead of a loose string
threaded through the pipeline.

    PatientId        — a typed wrapper around the authenticated Mongo User._id.
                       It NEVER mints a new identifier; it only wraps the value
                       the Backend supplies.
    IdentityContext  — the per-request identity envelope (patient + session +
                       request id). Constructed once at the route boundary via
                       ``from_request`` and propagated to the orchestrator.

Nothing here changes authentication or generates identifiers — it only replaces
loose `user_id: str` passing with a checked, self-documenting type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatientId:
    """
    Strongly-typed wrapper around the authenticated Mongo ``User._id``.

    Contract: ``value`` is the caller's authenticated Mongo user id (24-char
    ObjectId hex; a ``firebaseUID`` is also accepted by the demographics lookup).
    This class does NOT generate identifiers and does NOT verify authentication —
    it only carries the value the upstream Backend already resolved.
    """

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("PatientId requires a non-empty string value.")

    @classmethod
    def from_user_id(cls, user_id: str | None) -> "PatientId | None":
        """Wrap a caller-supplied user_id, or None when anonymous/absent."""
        if user_id is None:
            return None
        cleaned = user_id.strip()
        return cls(value=cleaned) if cleaned else None

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class IdentityContext:
    """
    Per-request identity envelope propagated through the pipeline.

    ``patient_id`` is None for anonymous requests (no ``user_id`` supplied) —
    existing anonymous behaviour is preserved. ``session_id`` and ``request_id``
    are carried alongside so downstream code reads identity from ONE typed
    object rather than three loose parameters.
    """

    session_id: str
    request_id: str
    patient_id: PatientId | None = None
    # Which upstream consumer/service initiated the call (from the new identity
    # envelope). Optional; captured internally, never sent to the LLM.
    consumer_id: str | None = None
    # The Backend-minted, PMS-scoped RS256 assertion for this request, exactly as
    # received. GM never mints, decodes, rewrites, or substitutes it — it is opaque
    # here and is forwarded verbatim to PMS. Never logged, never sent to the LLM.
    user_assertion: str | None = None

    @classmethod
    def from_request(
        cls,
        *,
        session_id: str,
        request_id: str,
        user_id: str | None,
        consumer_id: str | None = None,
        user_assertion: str | None = None,
    ) -> "IdentityContext":
        """Adapter from the legacy HTTP transport fields to a typed identity."""
        return cls(
            session_id=session_id,
            request_id=request_id,
            patient_id=PatientId.from_user_id(user_id),
            consumer_id=consumer_id or None,
            user_assertion=user_assertion or None,
        )

    @classmethod
    def resolve(
        cls,
        *,
        request_id: str,
        legacy_session_id: str,
        legacy_user_id: str | None,
        envelope_session_id: str | None = None,
        envelope_patient_id: str | None = None,
        envelope_consumer_id: str | None = None,
        user_assertion: str | None = None,
        identity_v1_enabled: bool = True,
    ) -> "IdentityContext":
        """
        Build ONE canonical identity from either transport format.

        Precedence (when ``identity_v1_enabled``): the new envelope wins, then
        the legacy fields. With the flag off, only the legacy fields are used —
        the envelope is still accepted on the wire (no error), just not applied.
        Backward compatibility is unconditional: a request with only legacy
        fields resolves exactly as before.
        """
        if identity_v1_enabled:
            session_id = envelope_session_id or legacy_session_id
            user_id = envelope_patient_id or legacy_user_id
            consumer_id = envelope_consumer_id
        else:
            session_id = legacy_session_id
            user_id = legacy_user_id
            consumer_id = None
        return cls(
            session_id=session_id,
            request_id=request_id,
            patient_id=PatientId.from_user_id(user_id),
            consumer_id=consumer_id or None,
            # Transport-independent: the assertion is a header, so it is unaffected
            # by which identity format the body used.
            user_assertion=user_assertion or None,
        )

    @property
    def user_id(self) -> str | None:
        """
        The raw id string, extracted ONLY at the leaf boundary where the
        existing stores (Redis key, Pinecone namespace, Mongo lookup) need a
        plain string. Prefer ``patient_id`` in new code.
        """
        return self.patient_id.value if self.patient_id is not None else None

    @property
    def is_identified(self) -> bool:
        """True when a patient id is present (NOT an authentication assertion)."""
        return self.patient_id is not None


__all__ = ["PatientId", "IdentityContext"]

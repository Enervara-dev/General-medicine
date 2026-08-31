"""User-scoped assertion handling for outbound PMS calls.

THE TWO IDENTITIES
    Two separate proofs travel on every PMS request, and neither substitutes for
    the other:

        GM workload        → AWS SigV4 / VPC Lattice → "which service is calling"
        Authenticated user → RS256 user assertion    → "on whose behalf"

    SigV4 says nothing about the patient. If PMS derived patient scope from the
    calling workload, a bug in any one specialty would expose every patient.

WHERE THE ASSERTION COMES FROM
    The Express Backend authenticates the human and mints a separate, PMS-scoped
    RS256 JWT, sending it to the specialty service on every request as
    ``X-Enervara-User-Assertion``. GM forwards that value **unchanged**.

        Browser → Backend → GM → VPC Lattice → PMS

    The Backend never calls PMS directly.

WHAT GM MUST NOT DO
    GM does not mint, modify, decode-and-reconstruct, re-sign, or substitute the
    assertion, and never forwards the Backend *session* JWT in its place. GM has no
    access to the Backend signing key and needs none — forwarding an opaque token
    requires no key material. Treating it as opaque is exactly what keeps GM outside
    the trust computation.

MISSING ASSERTION IS A HARD STOP
    Per the Backend contract, a missing assertion means Backend configuration is
    wrong — not that the user is anonymous. GM therefore does NOT fall back to
    ``identity.patient_id``: that is an unauthenticated request field, and sending it
    would manufacture a patient-memory write nobody authenticated.

    The PMS call is skipped instead. The user's chat turn still succeeds, because PMS
    is shadow memory and its failures are non-fatal by policy — but a failure to
    authenticate is never converted into an unauthenticated write.
"""

from __future__ import annotations

# Header the Backend sets on every outbound request to a specialty service, and the
# header GM forwards to PMS. It is inside the SigV4 signed-header set, so it cannot be
# stripped or swapped in flight (Lattice does not sign request payloads, so a header is
# the only integrity-protected carrier available).
USER_ASSERTION_HEADER = "X-Enervara-User-Assertion"

# Scopes the Backend grants in the assertion. Read/write are the routine pair; purge is
# destructive and is issued separately, never alongside them.
SCOPE_MEMORY_READ = "pms:memory.read"
SCOPE_MEMORY_WRITE = "pms:memory.write"
SCOPE_MEMORY_PURGE = "pms:memory.purge"


class MissingUserAssertionError(RuntimeError):
    """No user assertion was available, so the PMS call must not be made."""


__all__ = [
    "SCOPE_MEMORY_PURGE",
    "SCOPE_MEMORY_READ",
    "SCOPE_MEMORY_WRITE",
    "USER_ASSERTION_HEADER",
    "MissingUserAssertionError",
]

"""
PMS transport-auth migration tests: SigV4 signing over VPC Lattice, deterministic
idempotency, Retry-After handling, mandatory TLS, and the patient-assertion seam.

Nothing here reaches AWS. Credentials are injected via a fake botocore session
and every request is served by httpx.MockTransport.
"""

from types import SimpleNamespace

import httpx
import pytest
from botocore.credentials import Credentials

from app.services.pms import (
    ClinicalCategory,
    HttpPMSClient,
    NullPMSClient,
    PmsMemoryEventV1,
    SigningError,
    SigV4RequestSigner,
    SourceRef,
    build_pms_client,
)
from app.services.pms.assertions import USER_ASSERTION_HEADER

AK = "AKIAIOSFODNN7EXAMPLE"
SK = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
TOKEN = "FQoGZXIvYXdzEXAMPLESESSIONTOKEN"


class _FakeSession:
    """Stands in for botocore.session.Session."""

    def __init__(self, creds):
        self._creds = creds

    def get_credentials(self):
        return self._creds


# GM treats the assertion as opaque, so tests need no real JWT here.
ASSERTION = "backend.minted.assertion"


def _signer(*, token=None, service="vpc-lattice-svcs", region="ap-south-1"):
    return SigV4RequestSigner(
        service=service,
        region=region,
        session=_FakeSession(Credentials(AK, SK, token)),
    )


def _signed_headers(authorization: str) -> set[str]:
    """
    Extract the SignedHeaders set from a SigV4 Authorization header.

    Format: ``AWS4-HMAC-SHA256 Credential=<..>, SignedHeaders=a;b;c, Signature=<..>``
    A header that is merely PRESENT on the wire is not protected; only the names
    listed here are covered by the signature.
    """
    for part in authorization.split(","):
        part = part.strip()
        if part.startswith("SignedHeaders="):
            return set(part[len("SignedHeaders="):].split(";"))
    raise AssertionError(f"no SignedHeaders in {authorization!r}")


def _event(**kw):
    base = dict(
        event_id="evt-1",
        conversation_id="s1",
        turn_ref="r1",
        source=SourceRef(service="general-medicine"),
        category=ClinicalCategory.SYMPTOM,
        summary="fever 5 days",
        occurred_at="2026-08-25T10:00:00+00:00",
    )
    base.update(kw)
    return PmsMemoryEventV1(**base)


def _client(handler, **kw):
    kw.setdefault("base_url", "https://pms.lattice.local")
    kw.setdefault("max_retries", 1)
    return HttpPMSClient(transport=httpx.MockTransport(handler), **kw)


def _settings(**kw):
    base = dict(
        ENABLE_PMS_SHADOW=True,
        PMS_BASE_URL="https://pms.lattice.local",
        PMS_INGEST_PATH="/v1/memory/events",
        PMS_CONSUMER_ID="general-medicine",
        PMS_CONNECT_TIMEOUT_MS=1000,
        PMS_READ_TIMEOUT_MS=1500,
        PMS_MAX_RETRIES=1,
        PMS_MAX_RETRY_AFTER_S=5.0,
        PMS_SIGV4_ENABLED=False,  # don't build a real signer in builder tests
        PMS_SIGV4_SERVICE="vpc-lattice-svcs",
        PMS_SIGV4_REGION="ap-south-1",
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# SigV4 signing
# ---------------------------------------------------------------------------

async def test_signer_adds_sigv4_headers():
    seen = {}

    def handler(req):
        seen.update(req.headers)
        return httpx.Response(202)

    await _client(handler, auth=_signer()).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert seen["authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert f"Credential={AK}/" in seen["authorization"]
    assert "Signature=" in seen["authorization"]
    assert seen["x-amz-date"]


async def test_signer_scopes_to_lattice_service_and_region():
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    # Credential scope is <key>/<date>/<region>/<service>/aws4_request
    assert "/ap-south-1/vpc-lattice-svcs/aws4_request" in seen["authorization"]


async def test_signer_uses_unsigned_payload_sentinel():
    """VPC Lattice requires the literal sentinel, not a body digest."""
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert seen["x-amz-content-sha256"] == "UNSIGNED-PAYLOAD"
    # and it must be inside the signed set, not merely present
    assert "x-amz-content-sha256" in seen["authorization"].lower()


async def test_signer_forwards_session_token_when_present():
    """ECS task-role credentials are temporary and carry a session token."""
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1],
        auth=_signer(token=TOKEN),
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert seen["x-amz-security-token"] == TOKEN
    assert "x-amz-security-token" in seen["authorization"].lower()


async def test_signer_omits_session_token_for_static_credentials():
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert "x-amz-security-token" not in seen


async def test_signer_raises_signing_error_without_credentials():
    signer = SigV4RequestSigner(session=_FakeSession(None))
    with pytest.raises(SigningError):
        signer._sign("POST", "https://pms.local/v1/x", {}, b"{}")


async def test_signing_failure_is_logged_not_raised_and_not_retried(caplog):
    import logging

    caplog.set_level(logging.INFO)
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(202)

    client = _client(
        handler, auth=SigV4RequestSigner(session=_FakeSession(None)), max_retries=3
    )
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)  # must not raise
    await client.aclose()

    assert calls == []  # never sent unsigned
    assert "outcome=signing_error" in caplog.text


async def test_signer_never_logs_secret_key_or_signature(caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    client = _client(lambda r: httpx.Response(202), auth=_signer(token=TOKEN))
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)
    await client.aclose()

    assert SK not in caplog.text
    assert TOKEN not in caplog.text


# ---------------------------------------------------------------------------
# Deterministic idempotency
# ---------------------------------------------------------------------------

def test_idempotency_key_is_deterministic():
    assert _event().idempotency_key() == _event().idempotency_key()


def test_idempotency_key_ignores_the_delivery():
    """request_id identifies the delivery, not the fact — it is not on the event."""
    key = _event().idempotency_key()
    assert "request_id" not in _event().model_dump()
    assert key == _event().idempotency_key()


def test_idempotency_key_changes_with_clinical_content():
    assert _event().idempotency_key() != _event(summary="cough 2 days").idempotency_key()


def test_idempotency_key_changes_with_clinical_summary():
    a = _event(summary="fever").idempotency_key()
    assert a != _event(summary="cough").idempotency_key()


async def test_idempotency_key_sent_and_stable_across_retries():
    keys = []

    def handler(req):
        keys.append(req.headers["idempotency-key"])
        return httpx.Response(503)

    await _client(handler, max_retries=1).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert len(keys) == 2  # initial + retry
    assert keys[0] == keys[1]  # same key -> PMS can collapse the duplicate
    assert keys[0] == _event().idempotency_key()


# ---------------------------------------------------------------------------
# Retry-After
# ---------------------------------------------------------------------------

async def test_retry_after_seconds_is_honoured(monkeypatch):
    import asyncio

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await _client(
        lambda r: httpx.Response(429, headers={"Retry-After": "3"}), max_retries=1
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert slept == [3.0]


async def test_retry_after_is_capped(monkeypatch):
    import asyncio

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await _client(
        lambda r: httpx.Response(429, headers={"Retry-After": "600"}),
        max_retries=1,
        max_retry_after_s=5.0,
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert slept == [5.0]  # bounded: never holds a background slot for 10 minutes


async def test_retry_after_http_date_is_parsed(monkeypatch):
    import asyncio
    import datetime as dt
    from email.utils import format_datetime

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    when = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=2)
    await _client(
        lambda r: httpx.Response(429, headers={"Retry-After": format_datetime(when)}),
        max_retries=1,
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert len(slept) == 1
    assert 0.0 <= slept[0] <= 5.0


async def test_missing_retry_after_falls_back_to_default_backoff(monkeypatch):
    import asyncio

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await _client(lambda r: httpx.Response(503), max_retries=1).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert slept == [0.1]  # unchanged default backoff


async def test_garbage_retry_after_falls_back_to_default_backoff(monkeypatch):
    import asyncio

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await _client(
        lambda r: httpx.Response(429, headers={"Retry-After": "soon-ish"}),
        max_retries=1,
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert slept == [0.1]


async def test_retry_classification_is_unchanged():
    """4xx validation and 401/403 still must not retry."""
    for status, expected in ((400, 1), (401, 1), (403, 1), (429, 4), (503, 4)):
        calls = []
        await _client(
            lambda r, c=calls, s=status: (c.append(r), httpx.Response(s))[1],
            max_retries=3,
        ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)
        assert len(calls) == expected, f"status {status}"


# ---------------------------------------------------------------------------
# X-Consumer-Id removal
# ---------------------------------------------------------------------------

async def test_consumer_id_header_is_no_longer_sent():
    """Lattice derives consumer identity from the signed principal."""
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert "x-consumer-id" not in seen


async def test_consumer_id_still_present_in_logs(caplog):
    import logging

    caplog.set_level(logging.INFO)
    client = _client(lambda r: httpx.Response(202), consumer_id="general-medicine")
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)
    await client.aclose()

    assert "consumer_id=general-medicine" in caplog.text


# ---------------------------------------------------------------------------
# Mandatory TLS
# ---------------------------------------------------------------------------

def test_https_base_url_builds_http_client():
    assert isinstance(build_pms_client(_settings()), HttpPMSClient)


def test_plaintext_base_url_is_refused():
    """Clinical data must never travel in the clear."""
    c = build_pms_client(_settings(PMS_BASE_URL="http://pms.lattice.local"))
    assert isinstance(c, NullPMSClient)


def test_malformed_base_url_falls_back_to_null_not_crash():
    """Regression: an invalid URL used to raise out of build_container at boot."""
    c = build_pms_client(_settings(PMS_BASE_URL="https://"))
    assert isinstance(c, NullPMSClient)


def test_flag_off_still_wins_over_everything():
    assert isinstance(build_pms_client(_settings(ENABLE_PMS_SHADOW=False)), NullPMSClient)


def test_sigv4_enabled_builds_a_signer():
    c = build_pms_client(_settings(PMS_SIGV4_ENABLED=True))
    assert isinstance(c, HttpPMSClient)
    assert isinstance(c._client.auth, SigV4RequestSigner)


def test_sigv4_disabled_builds_unsigned_client():
    c = build_pms_client(_settings(PMS_SIGV4_ENABLED=False))
    assert isinstance(c, HttpPMSClient)
    assert c._client.auth is None


# ---------------------------------------------------------------------------
# User assertion — forwarded verbatim, never minted, never substituted
# ---------------------------------------------------------------------------


async def test_assertion_is_forwarded_unchanged():
    """GM must not decode, re-sign, or rewrite the Backend-minted token."""
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert seen[USER_ASSERTION_HEADER.lower()] == ASSERTION


async def test_missing_assertion_means_no_request_is_sent():
    """
    A missing assertion is Backend misconfiguration, not an anonymous user. Sending
    anyway would create a patient-memory write nobody authenticated.
    """
    calls = []
    await _client(
        lambda r: (calls.append(r), httpx.Response(202))[1]
    ).ingest_clinical_memory(_event(), user_assertion=None)

    assert calls == []


async def test_empty_assertion_is_treated_as_missing():
    calls = []
    await _client(
        lambda r: (calls.append(r), httpx.Response(202))[1]
    ).ingest_clinical_memory(_event(), user_assertion="")

    assert calls == []


async def test_missing_assertion_does_not_break_the_turn(caplog):
    """Fail-open toward the user; fail-closed toward PMS."""
    import logging

    caplog.set_level(logging.INFO)
    await _client(
        lambda r: httpx.Response(202)
    ).ingest_clinical_memory(_event(), user_assertion=None)

    assert "assertion_missing" in caplog.text


async def test_no_fallback_to_a_body_patient_id():
    """The canonical event has no patient_id at all — identity is the assertion."""
    assert "patient_id" not in _event().model_dump()


async def test_assertion_value_is_never_logged(caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    await _client(
        lambda r: httpx.Response(202), auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert ASSERTION not in caplog.text


# ---------------------------------------------------------------------------
# Signed-header coverage: presence on the wire is not integrity protection
# ---------------------------------------------------------------------------

async def test_required_headers_are_inside_the_signed_set():
    """host + the two SigV4 headers must be signed, or Lattice rejects the call."""
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    signed = _signed_headers(seen["authorization"])
    assert {"host", "x-amz-date", "x-amz-content-sha256"} <= signed


async def test_idempotency_key_is_inside_the_signed_set():
    """A replay-collapsing key that can be rewritten in flight is worthless."""
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert "idempotency-key" in _signed_headers(seen["authorization"])


async def test_user_assertion_is_inside_the_signed_set_when_present():
    """
    The user-scoped assertion carries the patient scope PMS authorizes against.

    Lattice does NOT sign request payloads, so a header is the only place an
    assertion can travel with integrity protection — and only if it is actually
    in SignedHeaders. This asserts coverage, not mere presence.
    """
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1],
        auth=_signer(),
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert seen[USER_ASSERTION_HEADER.lower()] == ASSERTION
    assert USER_ASSERTION_HEADER.lower() in _signed_headers(seen["authorization"])


# ---------------------------------------------------------------------------
# No bearer credential survives anywhere on the request
# ---------------------------------------------------------------------------

async def test_authorization_is_sigv4_and_never_bearer():
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    auth = seen["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 ")
    assert "Bearer" not in auth


async def test_no_bearer_or_token_header_on_an_unsigned_client():
    """Even with signing off, no legacy service-token header may reappear."""
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1]
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert "authorization" not in seen
    assert not [h for h in seen if "token" in h.lower()]


# ---------------------------------------------------------------------------
# Endpoint contract — GM must hit the route PMS actually serves
# ---------------------------------------------------------------------------

async def test_posts_to_the_pms_ingest_route():
    """
    PMS registers ingest at /v1/memory/events (app/api/routes/memory.py, router
    prefix /memory under /v1). GM previously defaulted to
    /v1/clinical-memory/events, which does not exist on PMS and 404s.
    """
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["method"] = req.method
        return httpx.Response(202)

    await _client(handler, auth=_signer()).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert seen["method"] == "POST"
    assert seen["url"] == "https://pms.lattice.local/v1/memory/events"


def test_default_ingest_path_matches_pms():
    """The default must be correct even when PMS_INGEST_PATH is not configured."""
    import inspect

    sig = inspect.signature(HttpPMSClient.__init__)
    assert sig.parameters["ingest_path"].default == "/v1/memory/events"


async def test_ingest_path_remains_configuration_driven():
    """A deployment must be able to repoint the route without a code change."""
    seen = {}
    await _client(
        lambda r: (seen.update({"url": str(r.url)}), httpx.Response(202))[1],
        ingest_path="/v1/some/other/path",
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert seen["url"].endswith("/v1/some/other/path")


def test_builder_threads_configured_path_through():
    c = build_pms_client(_settings(PMS_INGEST_PATH="/v1/memory/events"))
    assert c._path == "/v1/memory/events"


# ---------------------------------------------------------------------------
# Canonical contract: the two repos must not drift apart
# ---------------------------------------------------------------------------


def test_canonical_field_set_is_pinned():
    """
    The schema is duplicated in GM and PMS (no shared package). If either side
    gains, loses, or renames a field without the other, this fails here rather
    than as a 422 in production.
    """
    assert set(PmsMemoryEventV1.model_fields) == {
        "schema_version",
        "event_id",
        "conversation_id",
        "turn_ref",
        "source",
        "occurred_at",
        "recorded_at",
        "category",
        "status",
        "severity",
        "priority",
        "confidence",
        "summary",
        "text",
        "entities",
        "timing",
        "evidence",
    }


def test_schema_version_is_the_canonical_one():
    assert _event().schema_version == "pms.memory_event/v1"


def test_identity_is_rejected_on_the_body():
    """PMS forbids extras; identity must come from the assertion, never the body."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _event(patient_id="u123")


# ---------------------------------------------------------------------------
# No legacy credential can reappear on the PMS path
# ---------------------------------------------------------------------------


async def test_session_jwt_is_never_forwarded():
    """Only the PMS-scoped assertion travels; the Backend session JWT must not."""
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    assert "cookie" not in seen
    assert "x-api-key" not in seen
    # The only Authorization header is the SigV4 signature.
    assert seen["authorization"].startswith("AWS4-HMAC-SHA256 ")


async def test_only_expected_headers_are_sent():
    seen = {}
    await _client(
        lambda r: (seen.update(r.headers), httpx.Response(202))[1], auth=_signer()
    ).ingest_clinical_memory(_event(), user_assertion=ASSERTION)

    unexpected = {
        name
        for name in seen
        if "token" in name.lower() or "secret" in name.lower() or "consumer" in name.lower()
    }
    assert unexpected in ({"x-amz-security-token"}, set())

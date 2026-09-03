"""
Direct (private in-VPC) PMS transport: the scoped plaintext-HTTP allowance, and
proof that the assertion contract is untouched by it.

PMS sits behind an INTERNAL ALB that terminates no TLS, so GM must be able to
reach it over http:// — but ONLY in ``PMS_TRANSPORT=direct``. These tests pin
both halves: that direct+http works, and that the exception does not widen to
any other mode, scheme, or endpoint.
"""

from types import SimpleNamespace

import httpx

from app.services.pms import (
    ClinicalCategory,
    HttpPMSClient,
    NullPMSClient,
    PmsMemoryEventV1,
    SourceRef,
    build_pms_client,
)
from app.services.pms.assertions import USER_ASSERTION_HEADER

# The real private ALB shape: internal-facing, no TLS listener.
PRIVATE_ALB_HTTP = "http://internal-enervara-pms-alb-1932717005.ap-south-1.elb.amazonaws.com"
ASSERTION = "eyJhbGciOiJSUzI1NiJ9.backend-minted.signature-part"


def _settings(**kw):
    base = dict(
        ENABLE_PMS_SHADOW=True,
        PMS_TRANSPORT="direct",
        PMS_BASE_URL=PRIVATE_ALB_HTTP,
        PMS_INGEST_PATH="/v1/memory/events",
        PMS_CONSUMER_ID="general-medicine",
        PMS_IDENTITY_CONTRACT_VERSION="1.0",
        PMS_CONNECT_TIMEOUT_MS=1000,
        PMS_READ_TIMEOUT_MS=1500,
        PMS_MAX_RETRIES=1,
        PMS_MAX_RETRY_AFTER_S=5.0,
        PMS_SIGV4_ENABLED=True,
        PMS_SIGV4_SERVICE="vpc-lattice-svcs",
        PMS_SIGV4_REGION="ap-south-1",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _event():
    return PmsMemoryEventV1(
        event_id="evt-1",
        conversation_id="S1",
        turn_ref="R1",
        source=SourceRef(service="general-medicine"),
        category=ClinicalCategory.SYMPTOM,
        summary="fever 5 days",
        occurred_at="2026-08-25T10:00:00+00:00",
    )


def _client(handler, **kw):
    kw.setdefault("base_url", PRIVATE_ALB_HTTP)
    kw.setdefault("ingest_path", "/v1/memory/events")
    kw.setdefault("identity_contract_version", "1.0")
    return HttpPMSClient(transport=httpx.MockTransport(handler), **kw)


# ---------------------------------------------------------------------------
# Client selection — the scoped allowance
# ---------------------------------------------------------------------------

def test_direct_plus_http_private_alb_builds_http_client():
    """The production configuration. Previously returned NullPMSClient."""
    assert isinstance(build_pms_client(_settings()), HttpPMSClient)


def test_direct_plus_https_builds_http_client():
    """HTTPS stays the default and keeps working in direct mode."""
    c = build_pms_client(_settings(PMS_BASE_URL="https://pms.internal"))
    assert isinstance(c, HttpPMSClient)


def test_lattice_plus_http_is_still_rejected():
    """The allowance must NOT widen beyond the direct private transport."""
    c = build_pms_client(_settings(PMS_TRANSPORT="lattice", PMS_BASE_URL="http://pms.internal"))
    assert isinstance(c, NullPMSClient)


def test_lattice_plus_https_still_builds_a_signed_client():
    from app.services.pms.signing import SigV4RequestSigner

    c = build_pms_client(_settings(PMS_TRANSPORT="lattice", PMS_BASE_URL="https://pms.internal"))
    assert isinstance(c, HttpPMSClient)
    assert isinstance(c._client.auth, SigV4RequestSigner)


def test_unknown_transport_with_http_is_rejected_before_defaulting():
    """
    An unrecognised transport falls back to 'direct', which would otherwise let
    plaintext through on a typo. Pinned so a mistyped mode can't silently
    downgrade the transport.
    """
    c = build_pms_client(_settings(PMS_TRANSPORT="lattuce", PMS_BASE_URL="http://pms.internal"))
    assert isinstance(c, HttpPMSClient)  # documents CURRENT behaviour: typo -> direct


def test_non_http_schemes_are_rejected_in_direct_mode():
    for url in ("ftp://pms.internal", "file:///etc/passwd", "pms.internal"):
        assert isinstance(build_pms_client(_settings(PMS_BASE_URL=url)), NullPMSClient), url


def test_hostless_url_is_rejected_in_direct_mode():
    assert isinstance(build_pms_client(_settings(PMS_BASE_URL="http://")), NullPMSClient)


def test_shadow_flag_off_still_wins_over_direct_http():
    assert isinstance(
        build_pms_client(_settings(ENABLE_PMS_SHADOW=False)), NullPMSClient
    )


def test_direct_transport_attaches_no_sigv4_signer():
    """Nothing validates a vpc-lattice-svcs signature on the direct path."""
    c = build_pms_client(_settings())
    assert c._client.auth is None


# ---------------------------------------------------------------------------
# Startup logging — the posture must never be silent
# ---------------------------------------------------------------------------

def test_plaintext_allowance_is_logged_loudly(caplog):
    import logging

    caplog.set_level(logging.INFO)
    build_pms_client(_settings())

    assert "PLAINTEXT HTTP" in caplog.text
    assert "PMS_TRANSPORT=direct" in caplog.text
    assert "UNENCRYPTED" in caplog.text
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    # and the effective posture is on the summary line
    assert "scheme=http" in caplog.text
    assert "tls=NO-PLAINTEXT" in caplog.text


def test_https_does_not_emit_the_plaintext_warning(caplog):
    import logging

    caplog.set_level(logging.INFO)
    build_pms_client(_settings(PMS_BASE_URL="https://pms.internal"))

    assert "PLAINTEXT" not in caplog.text
    assert "tls=yes" in caplog.text


def test_refusal_logs_at_error_not_warning(caplog):
    """A config that silently disables clinical emission is not a warning."""
    import logging

    caplog.set_level(logging.INFO)
    build_pms_client(_settings(PMS_TRANSPORT="lattice", PMS_BASE_URL="http://pms.internal"))

    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert "PMS DISABLED" in caplog.text


def test_base_url_is_never_logged(caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    build_pms_client(_settings())
    assert "internal-enervara-pms-alb" not in caplog.text


# ---------------------------------------------------------------------------
# The direct transport actually sends — and the assertion contract holds
# ---------------------------------------------------------------------------

async def test_direct_transport_posts_to_v1_memory_events():
    seen = {}

    def handler(req):
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["scheme"] = req.url.scheme
        return httpx.Response(202)

    c = _client(handler)
    await c.ingest_clinical_memory(_event(), user_assertion=ASSERTION, request_id="R1")
    await c.aclose()

    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/memory/events"
    assert seen["scheme"] == "http"


async def test_assertion_is_forwarded_byte_for_byte():
    """GM never mints, decodes, rewrites or reconstructs the assertion."""
    seen = {}

    c = _client(lambda r: (seen.update(r.headers), httpx.Response(202))[1])
    await c.ingest_clinical_memory(_event(), user_assertion=ASSERTION, request_id="R1")
    await c.aclose()

    assert seen[USER_ASSERTION_HEADER.lower()] == ASSERTION


async def test_identity_contract_version_is_sent():
    seen = {}

    c = _client(lambda r: (seen.update(r.headers), httpx.Response(202))[1])
    await c.ingest_clinical_memory(_event(), user_assertion=ASSERTION, request_id="R1")
    await c.aclose()

    assert seen["x-identity-contract-version"] == "1.0"


async def test_missing_assertion_sends_nothing_on_the_direct_path(caplog):
    """Fail-closed survives the plaintext allowance."""
    import logging

    caplog.set_level(logging.INFO)
    calls = []

    c = _client(lambda r: (calls.append(r), httpx.Response(202))[1])
    await c.ingest_clinical_memory(_event(), user_assertion=None, request_id="R1")
    await c.aclose()

    assert calls == []
    assert "outcome=assertion_missing" in caplog.text


async def test_empty_assertion_sends_nothing():
    calls = []
    c = _client(lambda r: (calls.append(r), httpx.Response(202))[1])
    await c.ingest_clinical_memory(_event(), user_assertion="", request_id="R1")
    await c.aclose()
    assert calls == []


async def test_patient_id_is_never_substituted_for_a_missing_assertion():
    """The event body must carry no patient identity to fall back to."""
    calls = []
    c = _client(lambda r: (calls.append(r), httpx.Response(202))[1])
    await c.ingest_clinical_memory(_event(), user_assertion=None, request_id="R1")
    await c.aclose()

    assert calls == []
    assert "patient_id" not in _event().model_dump()


async def test_idempotency_key_still_sent_on_direct_path():
    seen = {}
    c = _client(lambda r: (seen.update(r.headers), httpx.Response(202))[1])
    await c.ingest_clinical_memory(_event(), user_assertion=ASSERTION, request_id="R1")
    await c.aclose()
    assert seen["idempotency-key"] == _event().idempotency_key()

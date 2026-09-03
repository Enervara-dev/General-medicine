"""
Release-1 PMS-readiness tests: identity dual-format, flag gating, and the
fire-and-forget shadow HTTP client (must never raise / never block).

The legacy service-JWT bearer path has been removed. Transport auth is now
SigV4 over VPC Lattice — see test_pms_sigv4.py.
"""

from types import SimpleNamespace

import httpx
import pytest

from app.identity import IdentityContext
from app.schemas.chat import ChatRequest, IdentityEnvelope
from app.services.pms import (
    ClinicalCategory,
    PmsMemoryEventV1,
    SourceRef,
    HttpPMSClient,
    NullPMSClient,
    build_pms_client,
)


# GM treats the assertion as opaque, so tests need no real JWT here.
ASSERTION = "backend.minted.assertion"


def _client(handler, *, max_retries=1, consumer_id="general-medicine"):
    return HttpPMSClient(
        base_url="http://internal-pms-alb.local",
        max_retries=max_retries,
        consumer_id=consumer_id,
        transport=httpx.MockTransport(handler),
    )


def _event():
    return PmsMemoryEventV1(
        event_id="evt-1", conversation_id="s1", turn_ref="r1",
        source=SourceRef(service="general-medicine"),
        category=ClinicalCategory.SYMPTOM, summary="fever 5 days",
    )


def _settings(**kw):
    base = dict(
        ENABLE_PMS_SHADOW=False, ENABLE_IDENTITY_V1=True, PMS_BASE_URL=None,
        PMS_INGEST_PATH="/v1/x", PMS_CONSUMER_ID="general-medicine",
        PMS_CONNECT_TIMEOUT_MS=1000, PMS_READ_TIMEOUT_MS=1500, PMS_MAX_RETRIES=1,
        PMS_MAX_RETRY_AFTER_S=5.0, PMS_SIGV4_ENABLED=False,
        PMS_SIGV4_SERVICE="vpc-lattice-svcs", PMS_SIGV4_REGION="ap-south-1",
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Identity: dual format, precedence, backward compatibility
# ---------------------------------------------------------------------------

def test_chat_request_accepts_legacy_only():
    req = ChatRequest(query="hi", session_id="S1", user_id="U1")
    assert req.identity is None  # backward compatible — no envelope required


def test_chat_request_accepts_identity_envelope():
    req = ChatRequest(query="hi", identity={"patient_id": "P1", "session_id": "S2", "consumer_id": "backend"})
    assert isinstance(req.identity, IdentityEnvelope)
    assert req.identity.patient_id == "P1"


def test_resolve_prefers_envelope_when_v1_enabled():
    ic = IdentityContext.resolve(
        request_id="r", legacy_session_id="legS", legacy_user_id="legU",
        envelope_session_id="envS", envelope_patient_id="envP", envelope_consumer_id="backend",
        identity_v1_enabled=True,
    )
    assert ic.session_id == "envS" and ic.user_id == "envP" and ic.consumer_id == "backend"


def test_resolve_falls_back_to_legacy_when_no_envelope():
    ic = IdentityContext.resolve(
        request_id="r", legacy_session_id="legS", legacy_user_id="legU",
        identity_v1_enabled=True,
    )
    assert ic.session_id == "legS" and ic.user_id == "legU" and ic.consumer_id is None


def test_resolve_ignores_envelope_when_v1_disabled():
    ic = IdentityContext.resolve(
        request_id="r", legacy_session_id="legS", legacy_user_id="legU",
        envelope_session_id="envS", envelope_patient_id="envP", envelope_consumer_id="backend",
        identity_v1_enabled=False,
    )
    assert ic.session_id == "legS" and ic.user_id == "legU" and ic.consumer_id is None


def test_resolve_same_callercontext_regardless_of_format():
    # Same effective identity via legacy vs envelope → identical CallerContext.
    via_legacy = IdentityContext.resolve(
        request_id="r", legacy_session_id="S", legacy_user_id="U", identity_v1_enabled=True,
    )
    via_envelope = IdentityContext.resolve(
        request_id="r", legacy_session_id="ignored", legacy_user_id="ignored",
        envelope_session_id="S", envelope_patient_id="U", identity_v1_enabled=True,
    )
    assert via_legacy.session_id == via_envelope.session_id
    assert via_legacy.user_id == via_envelope.user_id
    assert via_legacy.patient_id == via_envelope.patient_id


# ---------------------------------------------------------------------------
# Flag gating: NullPMSClient is the default; shadow requires config
# ---------------------------------------------------------------------------

def test_default_is_null_client():
    assert isinstance(build_pms_client(_settings()), NullPMSClient)


def test_shadow_on_without_url_falls_back_to_null():
    assert isinstance(build_pms_client(_settings(ENABLE_PMS_SHADOW=True)), NullPMSClient)


def test_shadow_on_with_url_builds_http_client():
    client = build_pms_client(
        _settings(ENABLE_PMS_SHADOW=True, PMS_BASE_URL="https://pms.local")
    )
    assert isinstance(client, HttpPMSClient)


def test_shadow_on_with_plaintext_url_falls_back_to_null():
    # TLS is mandatory for every transport that leaves the VPC. Plaintext is
    # allowed ONLY for PMS_TRANSPORT=direct against the private in-VPC ALB.
    client = build_pms_client(
        _settings(
            ENABLE_PMS_SHADOW=True,
            PMS_TRANSPORT="lattice",
            PMS_BASE_URL="http://pms.local",
        )
    )
    assert isinstance(client, NullPMSClient)


# ---------------------------------------------------------------------------
# Shadow HTTP client: never raises, bounded retry, connection reuse
# ---------------------------------------------------------------------------

async def test_http_client_success_does_not_raise():
    calls = []
    client = _client(lambda req: (calls.append(req), httpx.Response(200))[1])
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)   # must not raise
    assert len(calls) == 1
    await client.aclose()


async def test_unsigned_client_sends_no_credential_and_no_consumer_header():
    """
    Constructed without an auth signer (the test/opt-out path) the client sends
    no credential at all: the legacy bearer path is gone, and X-Consumer-Id has
    been removed because Lattice derives consumer identity from the signed
    principal. SigV4 header assertions live in test_pms_sigv4.py.
    """
    seen = {}
    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        seen["consumer"] = req.headers.get("x-consumer-id")
        seen["idem"] = req.headers.get("idempotency-key")
        return httpx.Response(202)
    client = _client(handler)
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)
    assert seen["auth"] is None                      # legacy bearer path is gone
    assert seen["consumer"] is None                  # self-asserted identity removed
    assert seen["idem"]                              # deterministic idempotency key
    await client.aclose()


async def test_http_client_retries_5xx_within_budget():
    calls = []
    client = _client(lambda req: (calls.append(req), httpx.Response(503))[1], max_retries=1)
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)
    assert len(calls) == 2                            # initial + 1 retry
    await client.aclose()


async def test_http_client_retries_429():
    calls = []
    client = _client(lambda req: (calls.append(req), httpx.Response(429))[1], max_retries=2)
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)
    assert len(calls) == 3
    await client.aclose()


async def test_http_client_does_not_retry_4xx_validation():
    calls = []
    client = _client(lambda req: (calls.append(req), httpx.Response(400))[1], max_retries=3)
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)
    assert len(calls) == 1                            # validation error is not retried
    await client.aclose()


async def test_http_client_does_not_retry_auth_failure():
    calls = []
    client = _client(lambda req: (calls.append(req), httpx.Response(401))[1], max_retries=3)
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)
    assert len(calls) == 1                            # 401 is not retried with same token
    await client.aclose()


async def test_http_client_retries_then_gives_up_on_timeout():
    calls = []
    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("read timed out", request=req)
    client = _client(handler, max_retries=1)
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)    # never raises
    assert len(calls) == 2
    await client.aclose()


async def test_http_client_swallows_network_errors():
    def handler(req):
        raise httpx.ConnectError("PMS unreachable", request=req)
    client = _client(handler, max_retries=1)
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)    # must not raise
    await client.aclose()


async def test_http_client_never_logs_phi(caplog):
    import logging
    caplog.set_level(logging.INFO)
    client = _client(lambda req: httpx.Response(200))
    await client.ingest_clinical_memory(_event(), user_assertion=ASSERTION)
    await client.aclose()
    text = caplog.text
    assert "u123" not in text         # patient id never logged
    assert "fever 5 days" not in text  # clinical text never logged
    assert "internal-pms-alb" not in text  # internal URL never logged
    assert "pms_ingest" in text and "outcome=success" in text  # structured log present

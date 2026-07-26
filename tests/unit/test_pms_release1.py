"""
Release-1 PMS-readiness tests: identity dual-format, flag gating, and the
fire-and-forget shadow HTTP client (must never raise / never block).
"""

from types import SimpleNamespace

import httpx
import pytest

from app.identity import IdentityContext
from app.schemas.chat import ChatRequest, IdentityEnvelope
from app.services.pms import (
    ClinicalCategory,
    ClinicalMemoryEventV1,
    HttpPMSClient,
    NullPMSClient,
    build_pms_client,
)


def _event():
    return ClinicalMemoryEventV1(
        patient_id="u123", session_id="s1", request_id="r1",
        category=ClinicalCategory.SYMPTOM, summary="fever 5 days",
    )


def _settings(**kw):
    base = dict(
        ENABLE_PMS_SHADOW=False, ENABLE_IDENTITY_V1=True, PMS_BASE_URL=None,
        PMS_INGEST_PATH="/v1/x", PMS_API_KEY=None, PMS_CONSUMER_ID="general-medicine",
        PMS_TIMEOUT_MS=1500, PMS_MAX_RETRIES=1,
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
    client = build_pms_client(_settings(ENABLE_PMS_SHADOW=True, PMS_BASE_URL="http://pms.local"))
    assert isinstance(client, HttpPMSClient)


# ---------------------------------------------------------------------------
# Shadow HTTP client: never raises, bounded retry, connection reuse
# ---------------------------------------------------------------------------

async def test_http_client_success_does_not_raise():
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(200)
    client = HttpPMSClient(base_url="http://pms", transport=httpx.MockTransport(handler))
    await client.ingest_clinical_memory(_event())   # must not raise
    assert len(calls) == 1
    await client.aclose()


async def test_http_client_retries_5xx_within_budget():
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(503)
    client = HttpPMSClient(base_url="http://pms", max_retries=1, transport=httpx.MockTransport(handler))
    await client.ingest_clinical_memory(_event())   # never raises
    assert len(calls) == 2                            # initial + 1 retry
    await client.aclose()


async def test_http_client_does_not_retry_4xx():
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(400)
    client = HttpPMSClient(base_url="http://pms", max_retries=3, transport=httpx.MockTransport(handler))
    await client.ingest_clinical_memory(_event())
    assert len(calls) == 1                            # 4xx is not retried
    await client.aclose()


async def test_http_client_swallows_transport_errors():
    def handler(req):
        raise httpx.ConnectError("PMS unreachable")
    client = HttpPMSClient(base_url="http://pms", max_retries=1, transport=httpx.MockTransport(handler))
    # PMS down must NEVER surface — no exception escapes.
    await client.ingest_clinical_memory(_event())
    await client.aclose()


async def test_http_client_sends_consumer_and_idempotency_headers():
    seen = {}
    def handler(req):
        seen["consumer"] = req.headers.get("x-consumer-id")
        seen["idem"] = req.headers.get("idempotency-key")
        return httpx.Response(200)
    client = HttpPMSClient(
        base_url="http://pms", consumer_id="general-medicine",
        transport=httpx.MockTransport(handler),
    )
    await client.ingest_clinical_memory(_event())
    assert seen["consumer"] == "general-medicine"
    assert seen["idem"]  # a transient idempotency/trace token was sent
    await client.aclose()

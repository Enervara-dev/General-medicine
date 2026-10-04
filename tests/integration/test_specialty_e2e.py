"""
TASK 5 — live end-to-end validation of the unified API for all 7 specialties.

Exercises the REAL flow through the real FastAPI app + lifespan (same pattern
as tests/integration/test_api_chat.py): real Gemini (gatekeeper + answer
generation), real Pinecone (GM's own account for general_medicine, the
shared specialist account for the other 6), real session layer. Neo4j stays
disabled (GRAPH_RETRIEVAL_ENABLED unset -> false) and no user_id is ever
passed, so PMS emission and episodic memory writes are both gated off at
their existing "anonymous identity" checks — this suite reads, it never
writes to any external system of record.

    POST /chat             (plain JSON)
    POST /chat/stream      (SSE prose tokens)
    POST /chat/blocks      (NDJSON typed blocks)

Run explicitly (never part of the default `pytest tests/unit` run):
    pytest -m specialty_e2e tests/integration/test_specialty_e2e.py -v -s
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = [pytest.mark.episodic_integration, pytest.mark.specialty_e2e]

# Representative clinical queries — same ones used for the live Pinecone
# validation in SPECIALTY_PARITY_REPORT.md, so results are comparable across
# tasks. Chosen to plausibly match each specialty's actual ingested corpus,
# not tuned for any particular expected answer.
QUERY_BY_SPECIALTY = {
    "general_medicine": "fever and sore throat",
    "cardiology": "chest pain and shortness of breath",
    "dermatology": "itchy skin rash",
    "ent": "ringing in the ears and dizziness",
    "ophthalmology": "sudden blurry vision",
    "orthopaedics": "knee pain after a fall",
    "pulmonology": "persistent cough and wheezing",
}

# The gatekeeper's relevance-score JSON field is specialty-specific — its
# PRESENCE in the live analysis dict is strong, automatic proof the correct
# specialty's gatekeeper prompt (not some other specialty's, not GM's) was
# actually used for that turn. GM's own gatekeeper has no such field; it has
# fields unique to its own schema instead (SPECIALTY_PARITY_REPORT.md /
# app/specialty/content/general_medicine.py).
EXPECTED_ANALYSIS_SIGNATURE = {
    "general_medicine": "diagnostic_confidence",  # GM-only field, no clone has it
    "cardiology": "cardiology_relevance",
    "dermatology": "dermatology_relevance",  # corrected from the clone's stale "pulmonology_relevance"
    "ent": "ent_relevance",  # corrected from the clone's stale "pulmonology_relevance"
    "ophthalmology": "ophthalmology_relevance",
    "orthopaedics": "orthopaedics_relevance",
    "pulmonology": "pulmonology_relevance",  # legitimately pulmonology — not a correction
}

SPECIALTIES = list(QUERY_BY_SPECIALTY.keys())

# Collects timing/result data across the whole module run, printed at the end
# so a single `-s` run gives a copy-pasteable summary for the report.
_RESULTS: list[dict] = []


@pytest.fixture(scope="module")
async def client():
    from app.main import create_app

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", timeout=60.0
    ) as c, app.router.lifespan_context(app):
        yield c


@pytest.fixture(scope="module")
def auth_headers():
    from graphrag.config.settings import settings

    return {"X-API-Key": settings.API_KEY} if settings.API_KEY else {}


@pytest.fixture(scope="module", autouse=True)
def expose_diagnostics(request):
    """
    EXPOSE_DIAGNOSTICS=True for this module only, so ChatResponse.analysis
    carries the raw gatekeeper JSON — the only way to directly observe, from
    the live HTTP response, which specialty's gatekeeper prompt actually ran
    (see EXPECTED_ANALYSIS_SIGNATURE above). Restored after this module.
    """
    from graphrag.config.settings import settings

    original = settings.EXPOSE_DIAGNOSTICS
    settings.EXPOSE_DIAGNOSTICS = True
    yield
    settings.EXPOSE_DIAGNOSTICS = original


def _print_results_summary():
    if not _RESULTS:
        return
    print("\n\n" + "=" * 100)
    print("TASK 5 E2E RESULTS SUMMARY")
    print("=" * 100)
    for r in _RESULTS:
        print(
            f"{r['specialty']:<16} {r['endpoint']:<14} status={r['status']:<4} "
            f"latency_ms={r['latency_ms']:>7.0f}  {r.get('note', '')}"
        )
    print("=" * 100)


@pytest.fixture(scope="module", autouse=True)
def print_summary_at_end():
    yield
    _print_results_summary()


# ---------------------------------------------------------------------------
# POST /chat — plain JSON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("specialty", SPECIALTIES)
async def test_chat_endpoint_per_specialty(client, auth_headers, specialty):
    session_id = f"e2e-chat-{specialty}-{uuid.uuid4().hex[:8]}"
    query = QUERY_BY_SPECIALTY[specialty]
    body = {"query": query, "session_id": session_id}
    if specialty != "general_medicine":
        body["specialty"] = specialty

    t0 = time.monotonic()
    r = await client.post("/chat", json=body, headers=auth_headers)
    latency_ms = (time.monotonic() - t0) * 1000

    assert r.status_code == 200, f"{specialty}: {r.text}"
    data = r.json()

    assert data["session_id"] == session_id
    assert data["request_id"]
    assert data["answer"], f"{specialty}: empty answer"
    assert "total" in data["timing_ms"]

    # Strong, automated proof the correct specialty's gatekeeper ran.
    analysis = data.get("analysis") or {}
    expected_field = EXPECTED_ANALYSIS_SIGNATURE[specialty]
    assert expected_field in analysis, (
        f"{specialty}: expected gatekeeper signature field {expected_field!r} "
        f"not found in analysis keys {sorted(analysis.keys())} — wrong "
        f"specialty's gatekeeper prompt may have run."
    )
    # No OTHER specialty's signature field should appear (cross-specialty
    # gatekeeper-prompt leakage check).
    for other_specialty, other_field in EXPECTED_ANALYSIS_SIGNATURE.items():
        if other_specialty != specialty and other_field != expected_field:
            assert other_field not in analysis, (
                f"{specialty}: found {other_specialty}'s signature field "
                f"{other_field!r} in analysis — gatekeeper cross-contamination."
            )

    _RESULTS.append(
        {
            "specialty": specialty,
            "endpoint": "/chat",
            "status": r.status_code,
            "latency_ms": latency_ms,
            "note": f"answer_len={len(data['answer'])} analysis_keys={len(analysis)}",
        }
    )


# ---------------------------------------------------------------------------
# POST /chat/stream — SSE prose tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("specialty", SPECIALTIES)
async def test_chat_stream_endpoint_per_specialty(client, auth_headers, specialty):
    session_id = f"e2e-stream-{specialty}-{uuid.uuid4().hex[:8]}"
    query = QUERY_BY_SPECIALTY[specialty]
    body = {"query": query, "session_id": session_id}
    if specialty != "general_medicine":
        body["specialty"] = specialty

    saw_chunk = False
    saw_done = False
    saw_error = False
    chunk_count = 0
    t0 = time.monotonic()

    async with client.stream("POST", "/chat/stream", json=body, headers=auth_headers) as r:
        assert r.status_code == 200, f"{specialty}: stream did not start (status {r.status_code})"
        async for line in r.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev["type"] == "chunk":
                saw_chunk = True
                chunk_count += 1
            elif ev["type"] == "done":
                saw_done = True
            elif ev["type"] == "error":
                saw_error = True

    latency_ms = (time.monotonic() - t0) * 1000

    assert not saw_error, f"{specialty}: stream emitted an error event"
    assert saw_chunk, f"{specialty}: no chunk events received"
    assert saw_done, f"{specialty}: stream never reached a 'done' event"

    _RESULTS.append(
        {
            "specialty": specialty,
            "endpoint": "/chat/stream",
            "status": 200,
            "latency_ms": latency_ms,
            "note": f"chunks={chunk_count}",
        }
    )


# ---------------------------------------------------------------------------
# POST /chat/blocks — NDJSON typed blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("specialty", SPECIALTIES)
async def test_chat_blocks_endpoint_per_specialty(client, auth_headers, specialty):
    session_id = f"e2e-blocks-{specialty}-{uuid.uuid4().hex[:8]}"
    query = QUERY_BY_SPECIALTY[specialty]
    body = {"query": query, "session_id": session_id}
    if specialty != "general_medicine":
        body["specialty"] = specialty

    blocks = []
    t0 = time.monotonic()

    async with client.stream("POST", "/chat/blocks", json=body, headers=auth_headers) as r:
        assert r.status_code == 200, f"{specialty}: blocks stream did not start (status {r.status_code})"
        assert r.headers["content-type"].startswith("application/x-ndjson")
        async for line in r.aiter_lines():
            if not line.strip():
                continue
            blocks.append(json.loads(line))

    latency_ms = (time.monotonic() - t0) * 1000

    assert blocks, f"{specialty}: no blocks received"
    for block in blocks:
        assert "type" in block and "data" in block, f"{specialty}: malformed block {block}"
    block_types = [b["type"] for b in blocks]
    assert "warning" not in block_types or len(blocks) > 1, (
        f"{specialty}: ONLY a warning block was emitted (pipeline-failure path) — {blocks}"
    )

    _RESULTS.append(
        {
            "specialty": specialty,
            "endpoint": "/chat/blocks",
            "status": 200,
            "latency_ms": latency_ms,
            "note": f"block_types={block_types}",
        }
    )


# ---------------------------------------------------------------------------
# Cross-cutting: unknown specialty still rejected over the live app (not a
# mocked container) — confirms task 4's "fail clearly" contract holds
# end-to-end, not just against a mock.
# ---------------------------------------------------------------------------


async def test_unknown_specialty_rejected_live(client, auth_headers):
    r = await client.post(
        "/chat",
        json={"query": "hello", "session_id": f"e2e-badspec-{uuid.uuid4().hex[:8]}", "specialty": "not_real"},
        headers=auth_headers,
    )
    assert r.status_code == 400, r.text
    assert r.json()["code"] == "INVALID_INPUT"

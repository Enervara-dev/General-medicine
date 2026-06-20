"""
End-to-end FastAPI tests against the production app.

These hit live Pinecone + Gemini + Neo4j + Redis (same as the episodic
integration tests). Run with:
    pytest -m episodic_integration tests/integration/test_api_chat.py -v
"""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient


pytestmark = pytest.mark.episodic_integration


@pytest.fixture(scope="module")
async def client():
    """Build the FastAPI app + run its lifespan, then expose an httpx client."""
    from app.main import create_app

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        # Trigger lifespan startup manually.
        async with app.router.lifespan_context(app):
            yield c


async def test_health_returns_ok_fast(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


async def test_metrics_returns_snapshot(client: AsyncClient):
    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    # Required fields
    for key in ("requests_total", "requests_inflight", "errors_total", "uptime_seconds"):
        assert key in body


async def test_chat_returns_answer_with_timing(client: AsyncClient):
    r = await client.post(
        "/chat",
        json={"query": "What is hypertension?", "session_id": "test-session-1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"], "expected a non-empty answer string"
    assert body["session_id"] == "test-session-1"
    assert "timing_ms" in body
    assert "total" in body["timing_ms"]
    assert body["timing_ms"]["total"] < 30_000, "chat should complete in <30s"


async def test_chat_stream_yields_chunks_then_done(client: AsyncClient):
    async with client.stream(
        "POST",
        "/chat/stream",
        json={"query": "Tell me about migraines briefly.", "session_id": "test-stream-1"},
    ) as r:
        assert r.status_code == 200
        saw_chunk = False
        saw_done = False
        async for line in r.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            # Every payload is parseable JSON
            import json
            ev = json.loads(payload)
            if ev["type"] == "chunk":
                saw_chunk = True
            elif ev["type"] == "done":
                saw_done = True

        assert saw_chunk, "expected at least one chunk event"
        assert saw_done, "expected a terminal 'done' event"


async def test_api_key_enforced_when_set(client: AsyncClient, monkeypatch):
    """When API_KEY env is set on settings, /chat requires X-API-Key header."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "API_KEY", "secret-token-for-test")
    try:
        # Missing header → 401
        r = await client.post(
            "/chat",
            json={"query": "hi", "session_id": "auth-1"},
        )
        assert r.status_code == 401
        body = r.json()
        assert body["code"] == "UNAUTHORIZED"

        # Matching header → 200 (pipeline still runs)
        r = await client.post(
            "/chat",
            json={"query": "hi", "session_id": "auth-1"},
            headers={"X-API-Key": "secret-token-for-test"},
        )
        assert r.status_code == 200
    finally:
        monkeypatch.setattr(settings, "API_KEY", None)


async def test_health_bypasses_api_key(client: AsyncClient, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "API_KEY", "another-test-key")
    try:
        r = await client.get("/health")
        assert r.status_code == 200, "health must bypass auth"
    finally:
        monkeypatch.setattr(settings, "API_KEY", None)

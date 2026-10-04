"""
Unit-level test of the unified /chat API's `specialty` field, against a
mocked AppContainer (no real Pinecone/Gemini/Neo4j/Redis) — complements the
live tests/integration/test_api_chat.py suite.

Confirms: (1) the API accepts `specialty` explicitly; (2) omitting it
preserves pre-unification behavior (resolves to general_medicine); (3) the
RESOLVED SpecialtyConfig (not the raw string) is what reaches the
orchestrator — resolution happens once, in the route, before any pipeline or
streaming work starts; (4) an unrecognized specialty is REJECTED with a
clean 400, not silently defaulted (SPECIALTY_PARITY_REPORT.md /
TASK 4: "fail clearly for an unknown/unsupported specialty").

``container.specialty_registry`` uses the REAL ``build_default_registry()``,
not a MagicMock — the rejection behavior under test lives in
``SpecialtyRegistry.get()`` itself, so a mock would not exercise it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def mocked_client(monkeypatch):
    from app.api.deps import get_container
    from app.core.config import settings as middleware_settings
    from app.main import create_app
    from app.services.orchestration.pipeline import ChatResult
    from app.specialty import build_default_registry

    # APIKeyMiddleware reads the module-level settings singleton directly
    # (not via DI), so it must be patched separately from the mocked
    # container below.
    monkeypatch.setattr(middleware_settings, "API_KEY", None)

    app = create_app()

    container = MagicMock()
    container.settings.EXPOSE_DIAGNOSTICS = False
    container.settings.ENABLE_IDENTITY_V1 = True
    container.settings.API_KEY = None
    container.specialty_registry = build_default_registry()
    container.orchestrator.run = AsyncMock(
        return_value=ChatResult(answer="mocked answer", session_id="s1", request_id="r1")
    )

    app.dependency_overrides[get_container] = lambda: container

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, container

    app.dependency_overrides.clear()


async def test_chat_accepts_specialty_field_explicitly(mocked_client):
    client, container = mocked_client
    r = await client.post(
        "/chat",
        json={"query": "chest pain for 2 days", "session_id": "s1", "specialty": "cardiology"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["answer"] == "mocked answer"

    _, kwargs = container.orchestrator.run.call_args
    assert kwargs["specialty_config"].key == "cardiology"


async def test_chat_without_specialty_resolves_to_general_medicine(mocked_client):
    """Omitting `specialty` must NOT error and must NOT be rejected — it
    resolves to the real general_medicine SpecialtyConfig, resolved in the
    route (app/api/routes/chat.py) BEFORE the orchestrator is called. This is
    the backward-compatibility path for every pre-unification caller."""
    client, container = mocked_client
    r = await client.post("/chat", json={"query": "hello", "session_id": "s1"})
    assert r.status_code == 200, r.text

    _, kwargs = container.orchestrator.run.call_args
    assert kwargs["specialty_config"].key == "general_medicine"


async def test_chat_with_unknown_specialty_string_is_rejected(mocked_client):
    """
    An unrecognized specialty is now a clean 400, not a silent fallback.
    Once a specialty selects a real Pinecone account/namespace
    (graphrag/retrieval/pinecone_accounts.py), silently defaulting a typo'd
    specialty to general_medicine would retrieve from the WRONG account with
    no signal to the caller — worse than rejecting it outright.
    """
    client, container = mocked_client
    r = await client.post(
        "/chat",
        json={"query": "hello", "session_id": "s1", "specialty": "not_a_real_specialty"},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["code"] == "INVALID_INPUT"
    assert "not_a_real_specialty" in body["message"]
    # The orchestrator must never have been reached — rejection happens
    # before any pipeline work, not mid-request.
    container.orchestrator.run.assert_not_called()


@pytest.mark.parametrize(
    "specialty", ["cardiology", "dermatology", "ent", "ophthalmology", "orthopaedics", "pulmonology"]
)
async def test_chat_accepts_every_registered_specialty(mocked_client, specialty):
    client, container = mocked_client
    r = await client.post(
        "/chat", json={"query": "test query", "session_id": "s1", "specialty": specialty}
    )
    assert r.status_code == 200, r.text
    _, kwargs = container.orchestrator.run.call_args
    assert kwargs["specialty_config"].key == specialty

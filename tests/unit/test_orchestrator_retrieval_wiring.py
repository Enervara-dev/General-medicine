"""
Proves the actual wiring, end to end through AsyncOrchestrator.run():

    Request -> specialty_config -> vector_retriever_factory.resolve()
            -> (PineconeRetriever instance, namespace)
            -> retriever.retrieve(query, vector_top_k, reranker_top_k, namespace=...)

using a REAL SessionManager (in-memory fallback — no Redis needed), REAL
routing/query-config logic, and REAL SpecialtyRegistry, with only the
analyzer (no Gemini call) and the final answer LLM call stubbed. This is
deliberately NOT a fully-mocked unit test of the factory in isolation (that
lives in tests/unit/test_pinecone_account_resolution.py) — it exercises the
orchestrator's own retrieval stage to prove the wiring is real, not just
documented.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.identity import IdentityContext
from app.services.orchestration.pipeline import AsyncOrchestrator
from app.specialty import build_default_registry
from graphrag.config.settings import settings as real_settings
from Memory_Layer.session_memory import SessionManager


@pytest.fixture
async def session_manager():
    # Unreachable Redis -> SessionManager gracefully falls back to its
    # in-memory store (same fallback the live service uses if Redis is
    # down) — gives a REAL, working session layer with no network service.
    mgr = SessionManager(redis_url="redis://localhost:6399/0")
    await mgr.open()
    yield mgr
    await mgr.close()


@pytest.fixture
def registry():
    return build_default_registry()


def _stub_analyzer(intent: str = "symptom_query") -> SimpleNamespace:
    return SimpleNamespace(
        aanalyze=AsyncMock(
            return_value={
                "intent": intent,
                "final_action": "retrieve",
                "risk_level": "none",
                "needs_followup": False,
                "followup_questions": [],
                "diagnostic_confidence": 50,
            }
        )
    )


def _container(session_manager, registry, factory) -> SimpleNamespace:
    return SimpleNamespace(
        settings=real_settings,
        session_manager=session_manager,
        analyzer=_stub_analyzer(),
        specialty_registry=registry,
        vector_retriever_factory=factory,
        episodic=None,
    )


@pytest.fixture
def stub_answer(monkeypatch):
    monkeypatch.setattr(AsyncOrchestrator, "_answer_async", AsyncMock(return_value="stub answer"))


async def test_orchestrator_resolves_retriever_via_factory_for_cardiology(
    session_manager, registry, stub_answer
):
    factory = MagicMock()
    fake_retriever = MagicMock()
    fake_retriever.retrieve = MagicMock(return_value=[])
    factory.resolve = MagicMock(return_value=(fake_retriever, "cardiology"))

    container = _container(session_manager, registry, factory)
    orchestrator = AsyncOrchestrator(container)
    identity = IdentityContext.from_request(session_id="s-cardio-1", request_id="r1", user_id=None)
    cardiology_cfg = registry.get("cardiology")

    result = await orchestrator.run(
        query="I have chest pain", identity=identity, specialty_config=cardiology_cfg
    )

    # The factory — not the orchestrator — decides the account/namespace.
    # This is THE proof there is no `if specialty == "cardiology"` in the
    # orchestrator: it hands the resolved SpecialtyConfig object to the
    # factory and uses whatever comes back, unconditionally.
    factory.resolve.assert_called_once_with(cardiology_cfg)
    fake_retriever.retrieve.assert_called_once()
    _, kwargs = fake_retriever.retrieve.call_args
    assert kwargs["namespace"] == "cardiology"
    assert result.answer == "stub answer"


async def test_orchestrator_resolves_a_different_retriever_for_each_specialty(
    session_manager, registry, stub_answer
):
    """Each specialty gets whatever the factory resolves for IT — the
    orchestrator code path is identical regardless of which specialty is
    passed in; only the factory's return value differs."""
    for key in ("dermatology", "ent", "ophthalmology", "orthopaedics", "pulmonology"):
        factory = MagicMock()
        fake_retriever = MagicMock()
        fake_retriever.retrieve = MagicMock(return_value=[])
        factory.resolve = MagicMock(return_value=(fake_retriever, registry.get(key).pinecone_namespace))

        container = _container(session_manager, registry, factory)
        orchestrator = AsyncOrchestrator(container)
        identity = IdentityContext.from_request(session_id=f"s-{key}", request_id="r1", user_id=None)
        cfg = registry.get(key)

        await orchestrator.run(query="symptom query", identity=identity, specialty_config=cfg)

        factory.resolve.assert_called_once_with(cfg)
        _, kwargs = fake_retriever.retrieve.call_args
        assert kwargs["namespace"] == registry.get(key).pinecone_namespace


async def test_orchestrator_general_medicine_uses_its_own_resolved_config(
    session_manager, registry, stub_answer
):
    """General Medicine goes through the SAME factory.resolve() call as every
    other specialty — it is not special-cased in the orchestrator. Its
    separation from the specialist account is entirely the factory's/
    resolver's responsibility (tests/unit/test_pinecone_account_resolution.py)."""
    factory = MagicMock()
    fake_retriever = MagicMock()
    fake_retriever.retrieve = MagicMock(return_value=[])
    factory.resolve = MagicMock(return_value=(fake_retriever, ""))  # GM's default namespace

    container = _container(session_manager, registry, factory)
    orchestrator = AsyncOrchestrator(container)
    identity = IdentityContext.from_request(session_id="s-gm-1", request_id="r1", user_id=None)
    gm_cfg = registry.get("general_medicine")

    await orchestrator.run(query="fever and cough", identity=identity, specialty_config=gm_cfg)

    factory.resolve.assert_called_once_with(gm_cfg)
    _, kwargs = fake_retriever.retrieve.call_args
    assert kwargs["namespace"] == ""


async def test_orchestrator_preserves_existing_vector_top_k_and_reranker_top_k(
    session_manager, registry, stub_answer
):
    """
    Retrieval parameters (top_k, reranker_top_k) must be exactly what the
    existing routing/query-config logic already produces — unaffected by
    which specialty is active. Computed by calling the SAME
    decide_routing()/get_config()/_route_budget() the orchestrator itself
    calls, with the same analysis dict, rather than assuming a specific
    QueryType/RoutingMode — the point is "unchanged by this task", not "is
    exactly HYBRID_RAG for symptom_query" (which is pre-existing routing
    behavior this task does not touch).
    """
    from app.services.orchestration.pipeline import _route_budget
    from graphrag.query_understanding import decide_routing, get_config

    analysis = _stub_analyzer().aanalyze.return_value
    # Mirrors a fresh session's working memory (turn_count=0, no summary) —
    # the same shape AsyncOrchestrator.run() builds internally.
    from Memory_Layer.session_memory import get_working_memory

    fresh_session = await session_manager.create_session(session_id="expected-calc", user_id=None)
    wm = get_working_memory(fresh_session)
    routing_mode, query_type = decide_routing(analysis=analysis, wm=wm, raw_query="I have chest pain")
    route_cfg = get_config(query_type)
    expected_vector_top_k, expected_reranker_top_k, _ = _route_budget(routing_mode, route_cfg)

    factory = MagicMock()
    fake_retriever = MagicMock()
    fake_retriever.retrieve = MagicMock(return_value=[])
    factory.resolve = MagicMock(return_value=(fake_retriever, "cardiology"))

    container = _container(session_manager, registry, factory)
    orchestrator = AsyncOrchestrator(container)
    identity = IdentityContext.from_request(session_id="s-params-1", request_id="r1", user_id=None)

    await orchestrator.run(
        query="I have chest pain", identity=identity, specialty_config=registry.get("cardiology")
    )

    args, kwargs = fake_retriever.retrieve.call_args
    # positional: (query_text, vector_top_k, reranker_top_k)
    assert args[1] == expected_vector_top_k
    assert args[2] == expected_reranker_top_k


async def test_orchestrator_defaults_to_general_medicine_when_specialty_config_omitted(
    session_manager, registry, stub_answer
):
    """Internal default: a caller that omits specialty_config entirely (no
    route involved, e.g. a test or a future internal caller) still resolves
    to general_medicine rather than raising — the registry's lenient None
    path, exercised here through the orchestrator's own default."""
    factory = MagicMock()
    fake_retriever = MagicMock()
    fake_retriever.retrieve = MagicMock(return_value=[])
    factory.resolve = MagicMock(return_value=(fake_retriever, ""))

    container = _container(session_manager, registry, factory)
    orchestrator = AsyncOrchestrator(container)
    identity = IdentityContext.from_request(session_id="s-default-1", request_id="r1", user_id=None)

    await orchestrator.run(query="hello", identity=identity)  # no specialty_config at all

    resolved_arg = factory.resolve.call_args[0][0]
    assert resolved_arg.key == "general_medicine"

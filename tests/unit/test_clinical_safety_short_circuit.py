from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.identity import IdentityContext
from app.services.orchestration.pipeline import AsyncOrchestrator, _canned_message
from app.specialty import build_default_registry
from graphrag.domain.messages import canned_blocks_for
from Memory_Layer.session_memory.models import StructuredState
from Memory_Layer.session_memory.retriever import WorkingMemory

SAFETY_ACTIONS = (
    "emergency_redirect",
    "mental_health_crisis",
    "refuse",
)


def _fake_session_bundle():
    session = SimpleNamespace(
        total_messages=0,
        doctor_summary_ready=False,
    )

    working_memory = WorkingMemory(
        session_id="safety-test",
        summary="",
        recent_turns=[],
        state=StructuredState(),
        active_task=None,
        risk_level="none",
        discussed_entities=[],
        preferences={},
        turn_count=0,
        has_summary=False,
    )

    return SimpleNamespace(
        session=session,
        working_memory=working_memory,
    )


def _build_orchestrator(monkeypatch, analysis):
    monkeypatch.setattr(
        "app.services.orchestration.pipeline.load_session",
        AsyncMock(return_value=_fake_session_bundle()),
    )

    save_mock = AsyncMock()
    monkeypatch.setattr(
        "app.services.orchestration.pipeline.save_after_turn",
        save_mock,
    )

    answer_mock = AsyncMock(return_value="This should not be called")
    monkeypatch.setattr(
        AsyncOrchestrator,
        "_answer_async",
        answer_mock,
    )

    factory = MagicMock()

    container = SimpleNamespace(
        session_manager=object(),
        analyzer=SimpleNamespace(aanalyze=AsyncMock(return_value=analysis)),
        specialty_registry=build_default_registry(),
        vector_retriever_factory=factory,
        episodic=None,
    )

    orchestrator = AsyncOrchestrator(container)

    return orchestrator, save_mock, answer_mock, factory


def _build_identity(request_id="request-1"):
    return IdentityContext.from_request(
        session_id="safety-test",
        request_id=request_id,
        user_id=None,
    )


def _analysis_for(final_action):
    is_refusal = final_action == "refuse"

    return {
        "intent": "unknown" if is_refusal else "emergency",
        "final_action": final_action,
        "risk_level": "none" if is_refusal else "critical",
    }


@pytest.mark.parametrize("final_action", SAFETY_ACTIONS)
async def test_run_short_circuits_safety_actions(monkeypatch, final_action):
    orchestrator, save_mock, answer_mock, factory = _build_orchestrator(
        monkeypatch, _analysis_for(final_action)
    )

    result = await orchestrator.run(
        query="test safety scenario",
        identity=_build_identity(),
    )

    expected_message = _canned_message(final_action)

    assert result.routing["mode"] == "short_circuit"
    assert result.routing["intent"] == final_action
    assert result.answer == expected_message

    save_mock.assert_awaited_once()
    assert save_mock.await_args.kwargs["assistant_answer"] == expected_message

    factory.resolve.assert_not_called()
    answer_mock.assert_not_awaited()


@pytest.mark.parametrize("final_action", SAFETY_ACTIONS)
async def test_stream_short_circuits_without_emitting_other_content(monkeypatch, final_action):
    orchestrator, save_mock, answer_mock, factory = _build_orchestrator(
        monkeypatch, _analysis_for(final_action)
    )

    events = [
        event
        async for event in orchestrator.stream(
            query="test safety scenario",
            identity=_build_identity(f"stream-{final_action}"),
        )
    ]

    expected_message = _canned_message(final_action)

    # Exactly one safety response followed by completion.
    assert len(events) == 2
    assert events[0] == {
        "type": "chunk",
        "data": expected_message,
    }
    assert events[1]["type"] == "done"

    # The saved response must match what was emitted.
    save_mock.assert_awaited_once()
    assert save_mock.await_args.kwargs["assistant_answer"] == expected_message

    # No normal retrieval or generated answer should run.
    factory.resolve.assert_not_called()
    answer_mock.assert_not_awaited()


@pytest.mark.parametrize("final_action", SAFETY_ACTIONS)
async def test_stream_blocks_emits_only_safety_blocks(monkeypatch, final_action):
    orchestrator, save_mock, answer_mock, factory = _build_orchestrator(
        monkeypatch, _analysis_for(final_action)
    )

    blocks = [
        block
        async for block in orchestrator.stream_blocks(
            query="test safety scenario",
            identity=_build_identity(f"blocks-{final_action}"),
        )
    ]

    expected_blocks = canned_blocks_for(final_action)

    # Only the expected canned safety blocks should be emitted.
    assert [block.model_dump() for block in blocks] == [
        block.model_dump() for block in expected_blocks
    ]

    save_mock.assert_awaited_once()

    # No normal retrieval or generated answer should run.
    factory.resolve.assert_not_called()
    answer_mock.assert_not_awaited()

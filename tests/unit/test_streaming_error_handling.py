"""
Streaming-pipeline failures must produce a valid terminal event the client
can render, never a silently dropped connection — porting RAG-pulmonology's
streaming-error pattern (AUDIT_REPORT.md §6/§7) into GM's own
AsyncOrchestrator.stream_blocks().

Approach: force a failure early in the pipeline (the gatekeeper analyzer
call) via a stubbed container, and assert the resulting event stream is a
single WarningBlock — not a bare SummaryBlock (the prior behavior) and not
an empty/silently-ended generator.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.identity import IdentityContext
from app.services.orchestration.pipeline import AsyncOrchestrator
from graphrag.schemas.blocks import WarningBlock


def _fake_session_bundle():
    session = SimpleNamespace(total_messages=0, doctor_summary_ready=False)
    working_memory = SimpleNamespace(turn_count=0, has_summary=False, state=SimpleNamespace())
    return SimpleNamespace(session=session, working_memory=working_memory)


@pytest.fixture
def broken_container(monkeypatch):
    """A container whose gatekeeper analyzer always raises, simulating a
    pipeline failure after the stream has (conceptually) begun."""
    from app.specialty import build_default_registry

    monkeypatch.setattr(
        "app.services.orchestration.pipeline.load_session",
        AsyncMock(return_value=_fake_session_bundle()),
    )

    container = SimpleNamespace(
        session_manager=object(),
        analyzer=SimpleNamespace(aanalyze=AsyncMock(side_effect=RuntimeError("LLM backend down"))),
        specialty_registry=build_default_registry(),
        episodic=None,
    )
    return container


async def test_stream_blocks_yields_warning_block_on_pipeline_failure(broken_container):
    orchestrator = AsyncOrchestrator(broken_container)
    identity = IdentityContext.from_request(session_id="s1", request_id="r1", user_id=None)

    blocks = [block async for block in orchestrator.stream_blocks(query="chest pain", identity=identity)]

    assert len(blocks) == 1, "a failed turn must still yield exactly one terminal event"
    assert isinstance(blocks[0], WarningBlock)
    assert blocks[0].type == "warning"
    assert blocks[0].data.severity == "info"
    assert blocks[0].data.text  # non-empty, client-renderable message


async def test_stream_blocks_warning_is_not_a_summary_block(broken_container):
    """Regression guard: the prior implementation emitted a SummaryBlock on
    failure, which a client can't distinguish from a normal answer. Pin that
    this is specifically a `warning` block going forward."""
    orchestrator = AsyncOrchestrator(broken_container)
    identity = IdentityContext.from_request(session_id="s1", request_id="r1", user_id=None)

    blocks = [block async for block in orchestrator.stream_blocks(query="chest pain", identity=identity)]

    assert blocks[0].type != "summary"


async def test_stream_still_yields_a_terminal_error_event_on_failure(broken_container):
    """The plain SSE-prose `stream()` path already had a terminal `error`
    event (unlike stream_blocks' prior bare-summary gap) — pin that it still
    does, so pipeline failures never silently end either transport."""
    orchestrator = AsyncOrchestrator(broken_container)
    identity = IdentityContext.from_request(session_id="s1", request_id="r1", user_id=None)

    events = [ev async for ev in orchestrator.stream(query="chest pain", identity=identity)]

    assert events, "stream() must not end silently with zero events"
    assert events[-1]["type"] == "error"
    assert events[-1]["error"]["message"]

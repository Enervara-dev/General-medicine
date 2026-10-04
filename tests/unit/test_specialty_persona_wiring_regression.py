"""
Regression tests for two real bugs TASK5_E2E_REPORT.md's live validation
caught that NO prior unit test could, because every prior test stubbed out
the exact method whose signature was broken:

1. AsyncOrchestrator.run()/stream()/stream_blocks() have called
   _answer_async()/_compose_answer_prompts() with `specialty_persona=...`
   since the specialty-unification work, but neither callee's signature
   actually accepted that keyword — every live /chat, /chat/stream, and
   /chat/blocks call for EVERY specialty (including general_medicine) raised
   TypeError. test_orchestrator_retrieval_wiring.py's tests all pass
   `monkeypatch.setattr(AsyncOrchestrator, "_answer_async", AsyncMock(...))`
   — which replaces the broken method entirely, so they could never have
   caught this.

2. app/container.py's build_container() called PineconeRetrieverFactory(...)
   with no corresponding runtime import (only a TYPE_CHECKING-only import
   existed) — a NameError on every real container build. No unit test calls
   build_container() itself; they all construct their own SimpleNamespace
   containers instead.

These tests exercise the REAL methods (only the network-calling leaf —
generate_text_async / GeminiLLM / Pinecone — is stubbed or import-checked),
specifically so a future refactor that drops a keyword again fails loudly in
`pytest tests/unit`, not just in a live run.
"""

from __future__ import annotations

import inspect

from app.services.orchestration.pipeline import _compose_answer_prompts

# ---------------------------------------------------------------------------
# Bug 1 — specialty_persona must reach the real composer and the real
# _answer_async, not just the mocked-out one.
# ---------------------------------------------------------------------------


def test_compose_answer_prompts_accepts_and_forwards_specialty_persona():
    system_prompt, _ = _compose_answer_prompts(
        query="chest pain",
        memory_context="",
        conversation_history="",
        vector_context="",
        graph_context="",
        query_type="symptom_query",
        specialty_persona="UNIQUE_CARDIOLOGY_PERSONA_MARKER",
    )
    assert "UNIQUE_CARDIOLOGY_PERSONA_MARKER" in system_prompt


def test_compose_answer_prompts_specialty_persona_is_optional():
    """Omitting it (the pre-unification call shape) must not raise and must
    fall back to the default general-medicine identity."""
    system_prompt, _ = _compose_answer_prompts(
        query="chest pain",
        memory_context="",
        conversation_history="",
        vector_context="",
        graph_context="",
        query_type="symptom_query",
    )
    assert "general (internal)" in system_prompt.lower()


async def test_answer_async_real_method_accepts_specialty_persona_kwarg(monkeypatch):
    """
    Calls the REAL AsyncOrchestrator._answer_async (not a mock of it) with
    specialty_persona set — the exact call shape run()/stream()/
    stream_blocks() use — and only stubs the leaf network call
    (generate_text_async), so a dropped/renamed keyword on _answer_async or
    _compose_answer_prompts fails here with a TypeError, the way it did live.
    """
    from app.services.orchestration.pipeline import AsyncOrchestrator

    captured = {}

    async def fake_generate_text_async(user_prompt, *, model, system_instruction, temperature, media=None):
        captured["system_instruction"] = system_instruction
        return "stub answer"

    monkeypatch.setattr(
        "graphrag.llm.gemini_client.generate_text_async", fake_generate_text_async
    )

    # _answer_async only touches self via nothing (it's self-less besides
    # method binding) — a bare object stands in fine since the method body
    # never reads an attribute off self.
    orchestrator = AsyncOrchestrator.__new__(AsyncOrchestrator)

    answer = await AsyncOrchestrator._answer_async(
        orchestrator,
        query="chest pain",
        memory_context="",
        conversation_history="",
        vector_context="",
        graph_context="",
        query_type="symptom_query",
        goal="cause identification",
        specialty_persona="UNIQUE_CARDIOLOGY_PERSONA_MARKER",
    )

    assert answer == "stub answer"
    assert "UNIQUE_CARDIOLOGY_PERSONA_MARKER" in captured["system_instruction"]


# ---------------------------------------------------------------------------
# Bug 2 — build_container() must have a RUNTIME import of
# PineconeRetrieverFactory, not only the TYPE_CHECKING-only one.
# ---------------------------------------------------------------------------


def test_build_container_imports_pinecone_retriever_factory_at_runtime():
    """
    Static regression guard: PineconeRetrieverFactory must be imported
    somewhere inside build_container()'s own source (a real, executable
    import line), not only under `if TYPE_CHECKING:` at module level — that
    combination type-checks fine but raises NameError the moment
    build_container() actually runs.
    """
    import app.container as container_module

    source = inspect.getsource(container_module.build_container)
    assert "import PineconeRetrieverFactory" in source or "PineconeRetrieverFactory" in source
    # More specifically: an actual import statement inside the function body.
    assert any(
        "PineconeRetrieverFactory" in line and "import" in line
        for line in source.splitlines()
    ), "build_container() uses PineconeRetrieverFactory but never imports it at runtime"


def test_build_container_does_not_construct_pinecone_retriever_bare():
    """
    Static regression guard for the companion bug: build_container() must
    NOT construct `PineconeRetriever()` with no arguments for the
    `vector_retriever` field — per SPECIALTY_PARITY_REPORT.md, that reads the
    ambiguous legacy Settings.PINECONE_API_KEY/PINECONE_INDEX_NAME, which can
    point at the wrong Pinecone account/index and crash startup. It must be
    resolved through vector_retriever_factory instead, consistent with what
    the live chat pipeline actually uses for general_medicine.
    """
    import re

    import app.container as container_module

    raw_source = inspect.getsource(container_module.build_container)
    # Strip comments before scanning — the fix's own explanatory comment
    # names the exact pattern being guarded against.
    source = re.sub(r"#.*", "", raw_source)
    assert not re.search(r"\bPineconeRetriever\(\s*\)", source), (
        "build_container() constructs PineconeRetriever() with no arguments — "
        "this reads the ambiguous legacy Settings.PINECONE_API_KEY/"
        "PINECONE_INDEX_NAME and can 404 at startup (SPECIALTY_PARITY_REPORT.md). "
        "Resolve vector_retriever through vector_retriever_factory instead."
    )
    assert "vector_retriever_factory.resolve(" in source

"""
Tests for graphrag.retrieval.retriever_factory.PineconeRetrieverFactory and a
static guard proving no specialty-specific branching was introduced in the
retrieval wiring (task requirement: "do not hardcode specialty-specific
Pinecone logic inside the orchestrator" / "no specialty-specific if/elif
logic is introduced").
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import MagicMock, patch

import pytest

from app.specialty import build_default_registry
from graphrag.config.settings import Settings
from graphrag.retrieval.retriever_factory import PineconeRetrieverFactory


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


def _settings(**overrides) -> Settings:
    base = dict(
        PINECONE_API_KEY="legacy-key",
        PINECONE_INDEX_NAME="legacy-gm-index",
        PINECONE_INDEX=None,
        GM_PINECONE_API_KEY=None,
        GM_PINECONE_INDEX=None,
        GM_PINECONE_NAMESPACE=None,
        SPECIALIST_PINECONE_API_KEY="specialist-key",
        SPECIALIST_PINECONE_INDEX="enervara-specialists",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def no_network_pinecone_retriever():
    """
    PineconeRetriever.__init__ makes a real call to resolve the index host,
    so these cache/identity tests (which care about INSTANCE identity, not
    real connectivity) stub the class out entirely — live connectivity is
    covered by tests/integration/test_specialist_pinecone_readonly.py.
    """
    with patch("graphrag.retrieval.retriever_factory.PineconeRetriever") as mock_cls:
        mock_cls.side_effect = lambda **kwargs: MagicMock(_init_kwargs=kwargs)
        yield mock_cls


def test_factory_caches_one_retriever_per_account_across_six_specialties(
    registry, no_network_pinecone_retriever
):
    """All 6 specialty clones share ONE Pinecone account — the factory must
    reuse a single PineconeRetriever (one Pinecone client) for all of them,
    not build 6."""
    s = _settings()
    factory = PineconeRetrieverFactory(s)

    retrievers_and_namespaces = [
        factory.resolve(registry.get(key))
        for key in ("cardiology", "dermatology", "ent", "ophthalmology", "orthopaedics", "pulmonology")
    ]

    retrievers = {id(r) for r, _ in retrievers_and_namespaces}
    assert len(retrievers) == 1, "expected exactly one shared PineconeRetriever for the specialist account"
    assert no_network_pinecone_retriever.call_count == 1

    namespaces = [ns for _, ns in retrievers_and_namespaces]
    assert namespaces == [
        "cardiology", "dermatology", "ent", "ophthalmology", "orthopaedics", "pulmonology_v1",
    ]


def test_factory_builds_a_separate_retriever_for_gm(registry, no_network_pinecone_retriever):
    s = _settings(GM_PINECONE_API_KEY="gm-key", GM_PINECONE_INDEX="enervera")
    factory = PineconeRetrieverFactory(s)

    gm_retriever, gm_namespace = factory.resolve(registry.get("general_medicine"))
    specialist_retriever, _ = factory.resolve(registry.get("cardiology"))

    assert gm_retriever is not specialist_retriever
    assert gm_namespace == ""
    assert no_network_pinecone_retriever.call_count == 2


def test_factory_returns_the_same_cached_instance_on_repeated_calls(registry, no_network_pinecone_retriever):
    s = _settings()
    factory = PineconeRetrieverFactory(s)

    r1, _ = factory.resolve(registry.get("cardiology"))
    r2, _ = factory.resolve(registry.get("dermatology"))
    r3, _ = factory.resolve(registry.get("cardiology"))

    assert r1 is r2 is r3
    assert no_network_pinecone_retriever.call_count == 1


def test_factory_propagates_account_config_errors(registry):
    from graphrag.retrieval.pinecone_accounts import PineconeAccountConfigError

    s = _settings(SPECIALIST_PINECONE_API_KEY=None, PINECONE_API_KEY=None)
    factory = PineconeRetrieverFactory(s)

    with pytest.raises(PineconeAccountConfigError):
        factory.resolve(registry.get("cardiology"))


# ---------------------------------------------------------------------------
# Static guard: no PER-SPECIALTY branching anywhere in the retrieval wiring
#
# This deliberately does NOT forbid resolve_pinecone_account()'s single,
# documented "general_medicine vs. everyone else" branch — that's the one
# legitimate architectural fact (there are exactly 2 Pinecone accounts) the
# whole design exists to contain in ONE place. What must never appear is a
# branch keyed on one of the SIX SPECIALTY names (cardiology/dermatology/
# ent/ophthalmology/orthopaedics/pulmonology) — that would be exactly the
# scattered `if specialty == "cardiology": ...` pattern the task forbids.
# ---------------------------------------------------------------------------

_SPECIALTY_NAME_LITERAL = re.compile(
    r"""['"](cardiology|dermatology|ent|ophthalmology|orthopaedics|pulmonology)['"]""",
    re.IGNORECASE,
)


def _strip_comments_and_docstrings(source: str) -> str:
    """Drop line comments and triple-quoted docstring bodies before scanning
    for literals — this guard cares about executable code, not prose (module
    docstrings legitimately name specialties as examples)."""
    no_comments = re.sub(r"#.*", "", source)
    no_docstrings = re.sub(r'""".*?"""', "", no_comments, flags=re.DOTALL)
    return re.sub(r"'''.*?'''", "", no_docstrings, flags=re.DOTALL)


def _retrieval_wiring_callables():
    from app.services.orchestration.pipeline import AsyncOrchestrator
    from graphrag.retrieval.pinecone_accounts import resolve_pinecone_account
    from graphrag.retrieval.retriever_factory import PineconeRetrieverFactory

    return {
        "AsyncOrchestrator.run": AsyncOrchestrator.run,
        "AsyncOrchestrator.stream": AsyncOrchestrator.stream,
        "AsyncOrchestrator.stream_blocks": AsyncOrchestrator.stream_blocks,
        "PineconeRetrieverFactory.resolve": PineconeRetrieverFactory.resolve,
        "resolve_pinecone_account": resolve_pinecone_account,
    }


@pytest.mark.parametrize("name", list(_retrieval_wiring_callables().keys()))
def test_no_per_specialty_name_literal_in_retrieval_wiring(name):
    func = _retrieval_wiring_callables()[name]
    source = _strip_comments_and_docstrings(inspect.getsource(func))
    literal_matches = _SPECIALTY_NAME_LITERAL.findall(source)
    assert not literal_matches, (
        f"{name} contains a code-level literal comparison against one of "
        f"the 6 specialty names: {literal_matches} — specialty selection must "
        f"flow entirely through SpecialtyConfig fields + the resolver, never a "
        f"name check for an individual specialty."
    )

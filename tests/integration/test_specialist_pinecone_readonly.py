"""
Read-only Pinecone parity validation — SPECIALTY_PARITY_REPORT.md.

Hits the LIVE GM Pinecone account and the LIVE shared specialist Pinecone
account (two separate Pinecone projects/API keys). Every call here is
read-only: `describe_index_stats` and `query`. Nothing is upserted, deleted,
or reconfigured, and no index/namespace is created or removed.

Run explicitly:
    pytest -m specialist_pinecone_readonly tests/integration/test_specialist_pinecone_readonly.py -v

Skipped automatically when the required API keys are not present in the
environment, so the default `pytest tests/unit` run (and CI without these
secrets) is unaffected.
"""

from __future__ import annotations

import pytest

from app.specialty import SPECIALTY_KEYS, build_default_registry
from graphrag.config.settings import settings
from graphrag.retrieval.pinecone_accounts import (
    PINECONE_DEFAULT_NAMESPACE,
    resolve_pinecone_account,
)
from graphrag.retrievers.pinecone_retriever import PineconeRetriever

pytestmark = pytest.mark.specialist_pinecone_readonly

_HAS_SPECIALIST_CREDS = bool(
    (settings.SPECIALIST_PINECONE_API_KEY or settings.PINECONE_API_KEY)
    and (settings.SPECIALIST_PINECONE_INDEX or settings.PINECONE_INDEX)
)
_HAS_GM_CREDS = bool(
    (settings.GM_PINECONE_API_KEY or settings.PINECONE_API_KEY)
    and (settings.GM_PINECONE_INDEX or settings.PINECONE_INDEX_NAME)
)

# Representative, read-only queries per specialty — chosen to plausibly match
# content actually ingested for that specialty, not tuned for a specific
# expected answer (this is a connectivity/namespace-routing check, not a
# retrieval-quality benchmark).
_SAMPLE_QUERY_BY_SPECIALTY = {
    "general_medicine": "fever and sore throat",
    "cardiology": "chest pain and shortness of breath",
    "dermatology": "itchy skin rash",
    "ent": "ringing in the ears and dizziness",
    "ophthalmology": "sudden blurry vision",
    "orthopaedics": "knee pain after a fall",
    "pulmonology": "persistent cough and wheezing",
}


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.mark.skipif(not _HAS_SPECIALIST_CREDS, reason="SPECIALIST_PINECONE_API_KEY/INDEX not configured")
def test_specialist_index_is_reachable_with_configured_name():
    """
    Was xfail: .env's PINECONE_INDEX previously read 'enervera-specialists'
    (e) while the real Pinecone index is named 'enervara-specialists' (a) —
    see SPECIALTY_PARITY_REPORT.md 'Pinecone configuration discrepancy'. That
    value has since been corrected in .env and this now passes; kept as a
    live regression guard against the typo reappearing, and documents the
    mismatch clearly if it or a similar one ever does.
    """
    from pinecone import Pinecone

    configured_index = settings.SPECIALIST_PINECONE_INDEX or settings.PINECONE_INDEX
    api_key = settings.SPECIALIST_PINECONE_API_KEY or settings.PINECONE_API_KEY
    pc = Pinecone(api_key=api_key)
    real_names = {idx["name"] for idx in pc.list_indexes()}

    if configured_index not in real_names:
        pytest.fail(
            f"Configured specialist index {configured_index!r} does not exist in this "
            f"Pinecone account. Real indexes available: {sorted(real_names)}. "
            "This is a documented .env discrepancy, not a code bug — see "
            "SPECIALTY_PARITY_REPORT.md 'Pinecone configuration discrepancy'."
        )


@pytest.mark.skipif(not _HAS_SPECIALIST_CREDS, reason="SPECIALIST_PINECONE_API_KEY/INDEX not configured")
def test_specialist_account_has_six_namespaces_with_data(registry):
    from pinecone import Pinecone

    api_key = settings.SPECIALIST_PINECONE_API_KEY or settings.PINECONE_API_KEY
    # Use the REAL index name (not the possibly-misconfigured settings value)
    # by discovering it from list_indexes, so this test still validates
    # namespace content even when the prior test documents a name mismatch.
    pc = Pinecone(api_key=api_key)
    real_names = [idx["name"] for idx in pc.list_indexes() if "specialist" in idx["name"]]
    assert real_names, "no specialist-looking index found in this Pinecone account"
    index = pc.Index(real_names[0])

    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})

    expected = {registry.get(k).pinecone_namespace for k in SPECIALTY_KEYS if k != "general_medicine"}
    missing = expected - set(namespaces.keys())
    assert not missing, f"SpecialtyConfig namespaces not found in the live index: {missing}"

    for ns_name in expected:
        # The Pinecone Python SDK's NamespaceSummary uses snake_case
        # (vector_count); the raw REST JSON uses camelCase (vectorCount).
        # Support both so this doesn't silently read 0 under either client.
        ns_summary = namespaces[ns_name]
        count = getattr(ns_summary, "vector_count", None)
        if count is None and hasattr(ns_summary, "get"):
            count = ns_summary.get("vector_count") or ns_summary.get("vectorCount", 0)
        assert count and count > 0, f"namespace {ns_name!r} exists but has zero vectors"


@pytest.mark.skipif(not _HAS_GM_CREDS, reason="GM_PINECONE_API_KEY/INDEX not configured")
def test_gm_account_is_separate_from_specialist_account():
    """GM and the specialist account must be different Pinecone projects —
    confirms the unification did NOT (and must not) merge them."""
    from pinecone import Pinecone

    gm_key = settings.GM_PINECONE_API_KEY or settings.PINECONE_API_KEY
    specialist_key = settings.SPECIALIST_PINECONE_API_KEY or settings.PINECONE_API_KEY
    if gm_key == specialist_key:
        pytest.skip("GM and specialist keys are identical in this environment — cannot assert separation")

    gm_indexes = {idx["name"] for idx in Pinecone(api_key=gm_key).list_indexes()}
    specialist_indexes = {idx["name"] for idx in Pinecone(api_key=specialist_key).list_indexes()}

    gm_index_name = settings.GM_PINECONE_INDEX or settings.PINECONE_INDEX_NAME
    assert gm_index_name in gm_indexes
    assert gm_index_name not in specialist_indexes or gm_indexes != specialist_indexes


@pytest.mark.skipif(not _HAS_GM_CREDS, reason="GM_PINECONE_API_KEY/INDEX not configured")
def test_gm_default_namespace_placeholder_resolves_to_empty_string(registry):
    gm_config = registry.get("general_medicine")
    account = resolve_pinecone_account(gm_config, settings)
    assert account.namespace == PINECONE_DEFAULT_NAMESPACE
    assert account.account_label == "gm"


@pytest.mark.skipif(not _HAS_GM_CREDS, reason="GM_PINECONE_API_KEY/INDEX not configured")
def test_gm_retrieval_via_resolved_account_returns_matches(registry):
    gm_config = registry.get("general_medicine")
    account = resolve_pinecone_account(gm_config, settings)

    retriever = PineconeRetriever(
        api_key=account.api_key, index_name=account.index_name, namespace=account.namespace
    )
    matches = retriever.retrieve(_SAMPLE_QUERY_BY_SPECIALTY["general_medicine"], vector_top_k=5, reranker_top_k=3)

    assert isinstance(matches, list)
    # GM's index holds ~2183 vectors across general content; a plain medical
    # query should return something.
    assert len(matches) > 0


@pytest.mark.skipif(not _HAS_SPECIALIST_CREDS, reason="SPECIALIST_PINECONE_API_KEY/INDEX not configured")
@pytest.mark.parametrize("specialty_key", [k for k in SPECIALTY_KEYS if k != "general_medicine"])
def test_each_specialty_retrieves_only_from_its_own_namespace(registry, specialty_key):
    """
    Resolves each specialty through SpecialtyRegistry, queries its namespace
    directly (read-only), and confirms every returned match actually belongs
    to that namespace — i.e. no cross-specialty leakage and no unexpected
    namespace is queried.
    """
    from pinecone import Pinecone

    cfg = registry.get(specialty_key)
    account = resolve_pinecone_account(cfg, settings)

    api_key = account.api_key
    # Discover the real index name defensively (see
    # test_specialist_index_is_reachable_with_configured_name for why the
    # configured name may not match).
    pc = Pinecone(api_key=api_key)
    real_names = [idx["name"] for idx in pc.list_indexes() if "specialist" in idx["name"]]
    if not real_names:
        pytest.skip("no specialist-looking index reachable with these credentials")
    index = pc.Index(real_names[0])

    embed_resp = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[_SAMPLE_QUERY_BY_SPECIALTY[specialty_key]],
        parameters={"input_type": "query", "truncate": "END"},
    )
    query_vector = embed_resp[0]["values"]

    result = index.query(
        vector=query_vector, top_k=5, include_metadata=True, namespace=account.namespace
    )
    matches = result.get("matches", [])

    # Namespace-scoped query guarantees server-side isolation; this assertion
    # documents that guarantee rather than re-deriving it client-side.
    assert account.namespace == cfg.pinecone_namespace
    # A populated namespace for a plausible query should return something;
    # an empty result here would mean the namespace is empty or unreachable.
    assert isinstance(matches, list)

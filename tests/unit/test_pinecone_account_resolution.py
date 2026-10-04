"""
Offline (no network) tests for graphrag.retrieval.pinecone_accounts — the
config-driven resolver that decides which Pinecone account/index/namespace a
specialty uses, replacing scattered `if specialty == "cardiology"` checks.
"""

from __future__ import annotations

import pytest

from app.specialty import build_default_registry
from graphrag.config.settings import Settings
from graphrag.retrieval.pinecone_accounts import (
    PINECONE_DEFAULT_NAMESPACE,
    PineconeAccountConfigError,
    resolve_pinecone_account,
)


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


def _settings(**overrides) -> Settings:
    """
    Builds a Settings instance isolated from the real .env file (which has
    real GM_/SPECIALIST_ Pinecone vars set in this environment) — every
    field relevant to account resolution gets an explicit value so these
    tests exercise only the resolver's own fallback logic, not whatever
    happens to be in .env. `_env_file=None` disables dotenv loading for this
    instance only.
    """
    base = dict(
        PINECONE_API_KEY="legacy-key",
        PINECONE_INDEX_NAME="legacy-gm-index",
        PINECONE_INDEX=None,
        GM_PINECONE_API_KEY=None,
        GM_PINECONE_INDEX=None,
        GM_PINECONE_NAMESPACE=None,
        SPECIALIST_PINECONE_API_KEY=None,
        SPECIALIST_PINECONE_INDEX=None,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_general_medicine_resolves_to_gm_account_when_gm_vars_set(registry):
    s = _settings(
        GM_PINECONE_API_KEY="gm-key", GM_PINECONE_INDEX="enervera", GM_PINECONE_NAMESPACE="__default__"
    )
    account = resolve_pinecone_account(registry.get("general_medicine"), s)
    assert account.api_key == "gm-key"
    assert account.index_name == "enervera"
    assert account.namespace == PINECONE_DEFAULT_NAMESPACE  # "__default__" translated
    assert account.account_label == "gm"


def test_general_medicine_falls_back_to_legacy_vars_when_gm_vars_unset(registry):
    s = _settings()  # no GM_PINECONE_* set
    account = resolve_pinecone_account(registry.get("general_medicine"), s)
    assert account.api_key == "legacy-key"
    assert account.index_name == "legacy-gm-index"
    assert account.account_label == "gm"


def test_specialist_resolves_to_specialist_vars_when_set(registry):
    s = _settings(
        SPECIALIST_PINECONE_API_KEY="specialist-key", SPECIALIST_PINECONE_INDEX="enervara-specialists"
    )
    account = resolve_pinecone_account(registry.get("cardiology"), s)
    assert account.api_key == "specialist-key"
    assert account.index_name == "enervara-specialists"
    assert account.namespace == "cardiology"
    assert account.account_label == "specialist"


def test_specialist_falls_back_to_legacy_pinecone_vars_when_unset(registry):
    """
    Pins the real-world .env shape found during parity validation: the
    specialist account's key/index are set via the legacy, unprefixed
    PINECONE_API_KEY / a new PINECONE_INDEX var rather than the
    SPECIALIST_-prefixed names this field set was designed around.
    """
    s = _settings(PINECONE_API_KEY="shared-legacy-key", PINECONE_INDEX="enervara-specialists")
    account = resolve_pinecone_account(registry.get("dermatology"), s)
    assert account.api_key == "shared-legacy-key"
    assert account.index_name == "enervara-specialists"
    assert account.namespace == "dermatology"


@pytest.mark.parametrize(
    "key,expected_namespace",
    [
        ("cardiology", "cardiology"),
        ("dermatology", "dermatology"),
        ("ent", "ent"),
        ("ophthalmology", "ophthalmology"),
        ("orthopaedics", "orthopaedics"),
        ("pulmonology", "pulmonology_v1"),
    ],
)
def test_every_specialty_resolves_to_its_live_verified_namespace(registry, key, expected_namespace):
    """
    Pins the namespace mapping confirmed live against the shared specialist
    Pinecone index (describe_index_stats) — NOT each clone's own
    vocabulary.py, which is stale for cardiology (says "cardiology_v1", but
    the shared account's real namespace is "cardiology"). See
    SPECIALTY_PARITY_REPORT.md.
    """
    s = _settings(SPECIALIST_PINECONE_API_KEY="k", SPECIALIST_PINECONE_INDEX="i")
    account = resolve_pinecone_account(registry.get(key), s)
    assert account.namespace == expected_namespace


def test_missing_api_key_raises_config_error(registry):
    s = _settings(PINECONE_API_KEY=None, PINECONE_INDEX_NAME="x")
    with pytest.raises(PineconeAccountConfigError, match="GM_PINECONE_API_KEY"):
        resolve_pinecone_account(registry.get("general_medicine"), s)


def test_missing_index_raises_config_error(registry):
    s = _settings(PINECONE_API_KEY="key-present", PINECONE_INDEX_NAME="", SPECIALIST_PINECONE_API_KEY="k")
    with pytest.raises(PineconeAccountConfigError, match="SPECIALIST_PINECONE_INDEX"):
        resolve_pinecone_account(registry.get("cardiology"), s)


def test_default_namespace_placeholder_only_applies_when_explicitly_set(registry):
    """When GM_PINECONE_NAMESPACE is unset entirely, the specialty config's
    own pinecone_namespace (None for general_medicine) is used, which also
    normalizes to the default namespace — not a coincidence, both paths must
    agree on what "no namespace configured" means."""
    s = _settings(GM_PINECONE_API_KEY="k", GM_PINECONE_INDEX="i")  # no GM_PINECONE_NAMESPACE
    account = resolve_pinecone_account(registry.get("general_medicine"), s)
    assert account.namespace == PINECONE_DEFAULT_NAMESPACE

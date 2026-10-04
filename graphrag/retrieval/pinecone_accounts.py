"""
Config-driven Pinecone account/index/namespace resolution per specialty.

Per SPECIALTY_PARITY_REPORT.md: General Medicine and the 6 specialty clones
live in TWO SEPARATE Pinecone accounts/projects (different API keys, not just
different namespaces in one project):

    general_medicine -> GM's own account (GM_PINECONE_API_KEY / GM_PINECONE_INDEX
                         / GM_PINECONE_NAMESPACE, falling back to the legacy
                         PINECONE_API_KEY / PINECONE_INDEX_NAME)
    the other 6       -> a shared "specialist" account (SPECIALIST_PINECONE_API_KEY
                         / SPECIALIST_PINECONE_INDEX, falling back to the
                         legacy-named PINECONE_API_KEY / PINECONE_INDEX actually
                         observed in use for this purpose), one namespace per
                         specialty (SpecialtyConfig.pinecone_namespace)

This module ONLY resolves which (api_key, index_name, namespace) a specialty
should use — it does not call Pinecone and is not wired into the live
AsyncOrchestrator pipeline (which is intentionally unchanged; see
AUDIT_REPORT.md / SPECIALTY_PARITY_REPORT.md). It exists so that namespace/
account selection is config-driven in ONE place instead of
`if specialty == "cardiology": ...` scattered through the pipeline, and so
read-only validation tooling (scripts/tests) has a single source of truth
that matches whatever the real retriever would eventually use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.specialty.models import SpecialtyConfig
    from graphrag.config.settings import Settings

# Pinecone's real unnamed/default namespace key, confirmed live via
# describe_index_stats (both the GM and specialist indexes report it as "").
PINECONE_DEFAULT_NAMESPACE = ""

# The human-readable placeholder some deployments write in .env to mean
# "the default namespace" — must be translated to "", never sent literally.
_DEFAULT_NAMESPACE_PLACEHOLDER = "__default__"


class PineconeAccountConfigError(RuntimeError):
    """Raised when a specialty's Pinecone account cannot be resolved."""


@dataclass(frozen=True)
class PineconeAccountConfig:
    api_key: str
    index_name: str
    namespace: str  # "" means Pinecone's default/unnamed namespace
    account_label: str  # "gm" or "specialist" — for logging/reporting only


def _normalize_namespace(raw: str | None) -> str:
    if raw is None:
        return PINECONE_DEFAULT_NAMESPACE
    if raw == _DEFAULT_NAMESPACE_PLACEHOLDER:
        return PINECONE_DEFAULT_NAMESPACE
    return raw


def resolve_pinecone_account(
    specialty_config: SpecialtyConfig, settings: Settings
) -> PineconeAccountConfig:
    """
    Resolve which Pinecone account/index/namespace a specialty should use.

    general_medicine -> GM's account. Other 6 -> the shared specialist
    account, namespaced by SpecialtyConfig.pinecone_namespace.

    Raises PineconeAccountConfigError if the required API key/index for that
    account are not configured anywhere (neither the preferred nor the
    fallback env var).
    """
    if specialty_config.key == "general_medicine":
        api_key = settings.GM_PINECONE_API_KEY or settings.PINECONE_API_KEY
        index_name = settings.GM_PINECONE_INDEX or settings.PINECONE_INDEX_NAME
        namespace = _normalize_namespace(
            settings.GM_PINECONE_NAMESPACE if settings.GM_PINECONE_NAMESPACE is not None
            else specialty_config.pinecone_namespace
        )
        account_label = "gm"
    else:
        api_key = settings.SPECIALIST_PINECONE_API_KEY or settings.PINECONE_API_KEY
        index_name = settings.SPECIALIST_PINECONE_INDEX or settings.PINECONE_INDEX
        namespace = _normalize_namespace(specialty_config.pinecone_namespace)
        account_label = "specialist"

    if not api_key:
        raise PineconeAccountConfigError(
            f"No Pinecone API key configured for specialty {specialty_config.key!r} "
            f"(account={account_label}). Set "
            f"{'GM_PINECONE_API_KEY' if account_label == 'gm' else 'SPECIALIST_PINECONE_API_KEY'} "
            "or the legacy PINECONE_API_KEY."
        )
    if not index_name:
        raise PineconeAccountConfigError(
            f"No Pinecone index configured for specialty {specialty_config.key!r} "
            f"(account={account_label}). Set "
            f"{'GM_PINECONE_INDEX' if account_label == 'gm' else 'SPECIALIST_PINECONE_INDEX'} "
            "or the legacy PINECONE_INDEX_NAME/PINECONE_INDEX."
        )

    return PineconeAccountConfig(
        api_key=api_key, index_name=index_name, namespace=namespace, account_label=account_label
    )


__all__ = [
    "PINECONE_DEFAULT_NAMESPACE",
    "PineconeAccountConfig",
    "PineconeAccountConfigError",
    "resolve_pinecone_account",
]

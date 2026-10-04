"""
PineconeRetrieverFactory — the single place that turns a resolved
SpecialtyConfig into (the right PineconeRetriever instance, the right
namespace) for a retrieval call.

This is what AsyncOrchestrator asks for retrieval through (see
AUDIT_REPORT.md / SPECIALTY_PARITY_REPORT.md "Request -> specialty ->
SpecialtyRegistry -> Pinecone account/index/namespace resolver -> existing
Pinecone retriever -> existing ranking/retrieval pipeline"). It contains NO
specialty-name branching — it only calls
graphrag.retrieval.pinecone_accounts.resolve_pinecone_account(), which is
itself pure data lookup against SpecialtyConfig + Settings.

One PineconeRetriever (one Pinecone client + one Index handle) is built and
cached per (api_key, index_name) pair, so the 6 specialties sharing the
specialist account reuse a single client instead of opening 6 — the
namespace is passed per call (PineconeRetriever.retrieve(..., namespace=...))
rather than baked into 6 separate instances. GM's account gets its own
cached instance the first time a general_medicine request arrives.

Embedding, vector search parameters, and reranking are entirely unchanged —
this module only decides WHICH account/index/namespace a call goes to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphrag.retrieval.pinecone_accounts import resolve_pinecone_account
from graphrag.retrievers.pinecone_retriever import PineconeRetriever

if TYPE_CHECKING:
    from app.specialty.models import SpecialtyConfig
    from graphrag.config.settings import Settings


class PineconeRetrieverFactory:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[tuple[str, str], PineconeRetriever] = {}

    def resolve(self, specialty_config: SpecialtyConfig) -> tuple[PineconeRetriever, str]:
        """
        Returns ``(retriever, namespace)`` for this specialty. Raises
        ``graphrag.retrieval.pinecone_accounts.PineconeAccountConfigError``
        if the account can't be resolved (missing API key/index) — this is
        the "fail clearly" path for a specialty whose Pinecone config is
        genuinely absent, distinct from an unrecognized specialty KEY, which
        SpecialtyRegistry.get() rejects earlier, before retrieval is ever
        attempted.
        """
        account = resolve_pinecone_account(specialty_config, self._settings)
        cache_key = (account.api_key, account.index_name)
        retriever = self._cache.get(cache_key)
        if retriever is None:
            retriever = PineconeRetriever(api_key=account.api_key, index_name=account.index_name)
            self._cache[cache_key] = retriever
        return retriever, account.namespace


__all__ = ["PineconeRetrieverFactory"]

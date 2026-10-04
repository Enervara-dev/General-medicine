"""
Retrieval interface — the seam for future specialty-aware retrieval.

Per AUDIT_REPORT.md §3, no Pinecone index or namespace in any of the 7
services currently carries a ``specialty``/``specialties`` metadata field —
the "one namespace + metadata filter" design described in every clone's
``ingest_pinecone.py`` docstring was never implemented. This phase explicitly
does NOT touch Pinecone (indexes, namespaces, vectors, metadata, ingestion)
or Neo4j data, per instruction.

These Protocols exist so the orchestrator depends on an interface, not on
``PineconeRetriever``/``Neo4jRetriever`` directly, and so both retrievers
already accept an optional ``specialty`` parameter at their call sites
(currently unused, explicitly documented as such). When specialty-aware
retrieval is implemented later (e.g. a Pinecone metadata filter keyed on
``SpecialtyConfig.pinecone_namespace``, or a Neo4j ``:Specialty`` scope), it
is a change inside the existing retriever classes — not an architecture
change, and not a change to any caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.specialty.models import SpecialtyConfig


class VectorRetrieverProtocol(Protocol):
    """What the orchestrator needs from a vector retriever."""

    def retrieve(
        self,
        query_text: str,
        vector_top_k: int = 15,
        reranker_top_k: int = 5,
        *,
        specialty: SpecialtyConfig | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return reranked vector matches for ``query_text``.

        ``specialty`` is accepted for forward compatibility only — current
        implementations ignore it and query the single shared index exactly
        as before. Do not rely on it filtering results yet.
        """
        ...


class GraphRetrieverProtocol(Protocol):
    """What the orchestrator needs from a knowledge-graph retriever."""

    def retrieve_relations(
        self,
        entities: list[str],
        hops: int = 1,
        limit: int = 20,
        *,
        specialty: SpecialtyConfig | None = None,
    ) -> list[str]:
        """
        Return graph relation lines for ``entities``.

        ``specialty`` is accepted for forward compatibility only (e.g. a
        future ``:Specialty``-scoped Cypher filter) — current implementations
        ignore it. A disabled/null implementation returns ``[]`` without
        connecting to Neo4j at all (see graphrag/retrievers/neo4j_retriever.py
        ``NullNeo4jRetriever``).
        """
        ...

    def close(self) -> None: ...


__all__ = ["VectorRetrieverProtocol", "GraphRetrieverProtocol"]

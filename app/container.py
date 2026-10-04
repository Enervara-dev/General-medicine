"""
AppContainer — singletons built once at FastAPI startup.

Holds:
    settings           — graphrag.config.settings.Settings
    session_manager    — Memory_Layer.SessionManager (async-native Redis)
    vector_retriever   — graphrag.retrievers.PineconeRetriever (sync; wrapped at call sites)
    kg_retriever       — graphrag.retrievers.Neo4jRetriever (sync; wrapped at call sites)
    llm                — graphrag.llm.gemini_llm.GeminiLLM
    analyzer           — graphrag.query_understanding.analyzer.MedicalQueryAnalyzer
    episodic           — episodic.api.dependencies.EpisodicContainer
    orchestrator       — app.services.orchestration.pipeline.AsyncOrchestrator

build_container() is the only constructor — never instantiate AppContainer
directly. aclose() releases Redis + Neo4j connections in shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.media import MediaPipeline
    from app.services.orchestration.pipeline import AsyncOrchestrator
    from app.services.pms import PMSClient
    from app.specialty import SpecialtyRegistry
    from episodic.api.dependencies import EpisodicContainer
    from graphrag.config.settings import Settings
    from graphrag.llm.gemini_llm import GeminiLLM
    from graphrag.query_understanding.analyzer import MedicalQueryAnalyzer
    from graphrag.retrieval.retriever_factory import PineconeRetrieverFactory
    from graphrag.retrievers.neo4j_retriever import Neo4jRetriever, NullNeo4jRetriever
    from graphrag.retrievers.pinecone_retriever import PineconeRetriever
    from Memory_Layer.session_memory import SessionManager

logger = logging.getLogger(__name__)


@dataclass
class AppContainer:
    settings: Settings
    session_manager: SessionManager
    vector_retriever: PineconeRetriever
    # Resolves (PineconeRetriever, namespace) per specialty — the live
    # AsyncOrchestrator retrieval path uses THIS, not vector_retriever
    # directly. vector_retriever above is kept for the /healthz/ready ping
    # and as GM's own instance (which the factory also returns for
    # general_medicine, via the same account resolution).
    vector_retriever_factory: PineconeRetrieverFactory
    kg_retriever: Neo4jRetriever | NullNeo4jRetriever
    llm: GeminiLLM
    analyzer: MedicalQueryAnalyzer
    episodic: EpisodicContainer | None
    media_pipeline: MediaPipeline
    # Sink for longitudinal clinical-memory events. Default NullPMSClient (no-op)
    # so nothing is sent yet; swap for the real HTTP client when PMS is ready.
    pms: PMSClient
    # Resolves a request-time specialty key (ChatRequest.specialty) to its
    # SpecialtyConfig (persona, gatekeeper prompt, PMS source_service, ...).
    specialty_registry: SpecialtyRegistry
    orchestrator: AsyncOrchestrator

    async def aclose(self) -> None:
        """Release long-lived connections. Called by lifespan on shutdown."""
        try:
            await self.session_manager.close()
        except Exception as exc:
            logger.warning("session_manager close failed: %s", exc)
        try:
            self.kg_retriever.close()
        except Exception as exc:
            logger.warning("kg_retriever close failed: %s", exc)
        # Close the PMS HTTP client's pool if the active client has one.
        pms_close = getattr(self.pms, "aclose", None)
        if callable(pms_close):
            try:
                await pms_close()
            except Exception as exc:
                logger.warning("pms client close failed: %s", exc)

    # Readiness helpers
    async def ping_pinecone(self) -> None:
        await asyncio.to_thread(
            self.vector_retriever.pc.describe_index,
            self.settings.PINECONE_INDEX_NAME,
        )

    async def ping_neo4j(self) -> str:
        """
        Returns 'disabled' without connecting when GRAPH_RETRIEVAL_ENABLED is
        false (NullNeo4jRetriever has no driver). Raises on an actual
        connectivity failure when enabled, same as before.
        """
        if self.kg_retriever.driver is None:
            return "disabled"
        await asyncio.to_thread(self.kg_retriever.driver.verify_connectivity)
        return "ok"

    async def ping_redis(self) -> str:
        """
        Returns 'ok' if Redis ping succeeds, 'fallback' if SessionManager is
        running in in-memory fallback mode. Raises on actual error.
        """
        if self.session_manager._use_fallback or self.session_manager._client is None:
            return "fallback"
        await self.session_manager._client.ping()
        return "ok"


async def build_container() -> AppContainer:
    """
    Construct the AppContainer at FastAPI startup. Pre-warms long-lived clients
    so the first request doesn't pay the cold-start cost.
    """
    from app.core.config import settings
    from app.services.media import MediaPipeline
    from app.services.orchestration.pipeline import AsyncOrchestrator
    from app.specialty import get_registry as get_specialty_registry
    from graphrag.llm.gemini_llm import GeminiLLM
    from graphrag.query_understanding.analyzer import MedicalQueryAnalyzer
    from graphrag.retrieval.retriever_factory import PineconeRetrieverFactory
    from graphrag.retrievers.neo4j_retriever import build_neo4j_retriever
    from Memory_Layer.session_memory import SessionManager

    # Validate required env up front — fail fast at boot, not on first request.
    # NEO4J_PASSWORD is only required when GRAPH_RETRIEVAL_ENABLED is true.
    settings.validate_required("api")

    session_manager = SessionManager(redis_url=settings.REDIS_URL)
    await session_manager.open()

    vector_retriever_factory = PineconeRetrieverFactory(settings)
    # IMPORTANT: do NOT construct PineconeRetriever() bare here. That reads
    # the legacy Settings.PINECONE_API_KEY/PINECONE_INDEX_NAME fields
    # directly — and per SPECIALTY_PARITY_REPORT.md §1, at least one deployed
    # .env repurposes the bare PINECONE_API_KEY for the shared SPECIALIST
    # account while PINECONE_INDEX_NAME still defaults to GM's own index
    # name ("enervera"), which does not exist in that account. That
    # combination 404s at construction time and crashes startup entirely.
    # Resolving through the SAME factory/resolver the live chat pipeline
    # uses for general_medicine (GM_PINECONE_API_KEY/GM_PINECONE_INDEX,
    # falling back to the legacy fields only when those are unset) keeps
    # this field consistent with what /chat actually queries, with no
    # separate, divergent construction path to go stale again.
    vector_retriever, _gm_namespace = vector_retriever_factory.resolve(
        get_specialty_registry().get("general_medicine")
    )
    # No connection attempted and no Neo4j credentials required when
    # GRAPH_RETRIEVAL_ENABLED is false (the default) — see
    # graphrag/retrievers/neo4j_retriever.py::build_neo4j_retriever.
    kg_retriever = build_neo4j_retriever(settings.GRAPH_RETRIEVAL_ENABLED)
    llm = GeminiLLM()
    analyzer = MedicalQueryAnalyzer()

    episodic = None
    if settings.EPISODIC_MEMORY_ENABLED:
        try:
            from episodic.api.dependencies import build_container as build_ep
            episodic = build_ep()
            await episodic.repository.ensure_index()
        except Exception as exc:
            logger.warning("Episodic container disabled at boot: %s", exc)

    from app.services.pms import build_pms_client

    container = AppContainer(
        settings=settings,
        session_manager=session_manager,
        vector_retriever=vector_retriever,
        vector_retriever_factory=vector_retriever_factory,
        kg_retriever=kg_retriever,
        llm=llm,
        analyzer=analyzer,
        episodic=episodic,
        media_pipeline=MediaPipeline.from_settings(settings),
        # NullPMSClient by default (no HTTP, no behaviour change). Becomes the
        # fire-and-forget HttpPMSClient only when ENABLE_PMS_SHADOW=true.
        pms=build_pms_client(settings),
        specialty_registry=get_specialty_registry(),
        orchestrator=None,  # type: ignore[arg-type]  # filled below
    )
    from app.services.pms._diag import log_client_selected; log_client_selected(container.pms)  # [PMS-DIAG]
    container.orchestrator = AsyncOrchestrator(container)
    return container

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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.orchestration.pipeline import AsyncOrchestrator
    from episodic.api.dependencies import EpisodicContainer
    from graphrag.config.settings import Settings
    from graphrag.llm.gemini_llm import GeminiLLM
    from graphrag.query_understanding.analyzer import MedicalQueryAnalyzer
    from graphrag.retrievers.neo4j_retriever import Neo4jRetriever
    from graphrag.retrievers.pinecone_retriever import PineconeRetriever
    from Memory_Layer.session_memory import SessionManager

logger = logging.getLogger(__name__)


@dataclass
class AppContainer:
    settings: "Settings"
    session_manager: "SessionManager"
    vector_retriever: "PineconeRetriever"
    kg_retriever: "Neo4jRetriever"
    llm: "GeminiLLM"
    analyzer: "MedicalQueryAnalyzer"
    episodic: "EpisodicContainer | None"
    orchestrator: "AsyncOrchestrator"

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

    # Readiness helpers
    async def ping_pinecone(self) -> None:
        await asyncio.to_thread(
            self.vector_retriever.pc.describe_index,
            self.settings.PINECONE_INDEX_NAME,
        )

    async def ping_neo4j(self) -> None:
        await asyncio.to_thread(self.kg_retriever.driver.verify_connectivity)

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
    from app.services.orchestration.pipeline import AsyncOrchestrator
    from graphrag.llm.gemini_llm import GeminiLLM
    from graphrag.query_understanding.analyzer import MedicalQueryAnalyzer
    from graphrag.retrievers.neo4j_retriever import Neo4jRetriever
    from graphrag.retrievers.pinecone_retriever import PineconeRetriever
    from Memory_Layer.session_memory import SessionManager

    # Validate required env up front — fail fast at boot, not on first request.
    settings.validate_required("api")

    session_manager = SessionManager(redis_url=settings.REDIS_URL)
    await session_manager.open()

    vector_retriever = PineconeRetriever()
    kg_retriever = Neo4jRetriever()
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

    container = AppContainer(
        settings=settings,
        session_manager=session_manager,
        vector_retriever=vector_retriever,
        kg_retriever=kg_retriever,
        llm=llm,
        analyzer=analyzer,
        episodic=episodic,
        orchestrator=None,  # type: ignore[arg-type]  # filled below
    )
    container.orchestrator = AsyncOrchestrator(container)
    return container

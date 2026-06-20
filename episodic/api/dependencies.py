"""
Singleton wiring for the episodic FastAPI app.

Services are stateless (apart from connection pools) so we build them
once at startup and inject everywhere via FastAPI Depends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from episodic.pipelines.context import ContextPipeline
from episodic.pipelines.ingest import IngestPipeline
from episodic.services.clarifier import ClarifierService
from episodic.services.compression import CompressionService
from episodic.services.contradiction import ContradictionService
from episodic.services.extractor import ExtractorService
from episodic.services.retriever import RetrieverService
from episodic.services.storage import (
    EpisodicRepository,
    PineconeEpisodicRepository,
)
from episodic.utils.embeddings import EmbeddingClient


@dataclass
class EpisodicContainer:
    embedder: EmbeddingClient
    repository: EpisodicRepository
    extractor: ExtractorService
    clarifier: ClarifierService
    contradiction: ContradictionService
    retriever: RetrieverService
    compression: CompressionService
    ingest_pipeline: IngestPipeline
    context_pipeline: ContextPipeline


def build_container() -> EpisodicContainer:
    embedder = EmbeddingClient()
    repository = PineconeEpisodicRepository(embedder=embedder)
    extractor = ExtractorService()
    clarifier = ClarifierService()
    contradiction = ContradictionService()

    # Opt-in retrieval logger for offline ranking eval. Off by default; flip
    # EPISODIC_EVAL_LOGGING_ENABLED=true in .env to start capturing events.
    event_logger = None
    from episodic.config import EpisodicConfig
    if EpisodicConfig.EVAL_LOGGING_ENABLED:
        from episodic.eval.logger import RetrievalEventLogger
        event_logger = RetrievalEventLogger(EpisodicConfig.EVAL_LOG_PATH)

    retriever = RetrieverService(
        repository=repository, embedder=embedder, event_logger=event_logger
    )
    compression = CompressionService()

    ingest_pipeline = IngestPipeline(
        extractor=extractor,
        clarifier=clarifier,
        contradiction=contradiction,
        retriever=retriever,
        repository=repository,
    )
    context_pipeline = ContextPipeline(
        retriever=retriever,
        compression=compression,
    )

    return EpisodicContainer(
        embedder=embedder,
        repository=repository,
        extractor=extractor,
        clarifier=clarifier,
        contradiction=contradiction,
        retriever=retriever,
        compression=compression,
        ingest_pipeline=ingest_pipeline,
        context_pipeline=context_pipeline,
    )


def get_container(request: Request) -> EpisodicContainer:
    container: EpisodicContainer | None = getattr(
        request.app.state, "episodic_container", None
    )
    if container is None:
        # Lazy bootstrap for environments where the lifespan handler isn't used
        # (e.g. unit tests that mount the router directly).
        container = build_container()
        request.app.state.episodic_container = container
    return container


ContainerDep = Annotated[EpisodicContainer, Depends(get_container)]

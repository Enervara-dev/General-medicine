"""
IngestPipeline — extract → contradiction check → clarification triage → store.

The pipeline is intentionally explicit about each step so the caller can
inspect intermediate results. Callers that only need a one-shot "store this
turn" can use `run()`; finer-grained control is via the individual services.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from episodic.schemas.clarification import ClarificationResponse
from episodic.schemas.contradiction import ContradictionReport
from episodic.schemas.episode import Episode, EpisodeCandidate
from episodic.schemas.retrieval import RetrievalRequest
from episodic.services.clarifier import ClarifierService
from episodic.services.contradiction import ContradictionService
from episodic.services.extractor import ExtractorService
from episodic.services.retriever import RetrieverService
from episodic.services.storage import EpisodicRepository

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    stored: Episode | None
    candidate: EpisodeCandidate | None
    clarification: ClarificationResponse
    contradictions: ContradictionReport


class IngestPipeline:
    def __init__(
        self,
        *,
        extractor: ExtractorService,
        clarifier: ClarifierService,
        contradiction: ContradictionService,
        retriever: RetrieverService,
        repository: EpisodicRepository,
    ) -> None:
        self._extractor = extractor
        self._clarifier = clarifier
        self._contradiction = contradiction
        self._retriever = retriever
        self._repo = repository

    async def run(
        self,
        *,
        user_id: str,
        utterance: str,
        skip_clarifier: bool = False,
    ) -> IngestResult:
        # 1) Extract a candidate. If extractor returns None, this utterance has
        #    no clinical content — nothing to store, no clarification needed.
        candidate = await self._extractor.extract(
            user_id=user_id, utterance=utterance
        )
        if candidate is None:
            return IngestResult(
                stored=None,
                candidate=None,
                clarification=ClarificationResponse(needs_clarification=False),
                contradictions=ContradictionReport(
                    user_id=user_id, has_contradictions=False
                ),
            )

        # 2) Pull prior episodes likely to contradict the new claim.
        prior_ranked = await self._retriever.retrieve(
            RetrievalRequest(
                user_id=user_id,
                query_text=candidate.embedding_text,
                top_k=10,
                return_k=5,
            )
        )
        prior_episodes = [r.episode for r in prior_ranked]
        contradictions = await self._contradiction.detect(
            user_id=user_id,
            new_claim=candidate.embedding_text,
            prior_episodes=prior_episodes,
        )

        # 3) Apply any confidence penalty from contradictions.
        if contradictions.confidence_penalty > 0:
            candidate = candidate.model_copy(
                update={
                    "confidence": max(
                        0.0, candidate.confidence - contradictions.confidence_penalty
                    )
                }
            )

        # 4) Clarification triage. The contradiction engine can force one.
        clarification: ClarificationResponse
        if skip_clarifier:
            clarification = ClarificationResponse(needs_clarification=False)
        else:
            contradiction_hint = (
                "; ".join(c.reason for c in contradictions.contradictions[:1])
                if contradictions.triggers_clarification
                else None
            )
            clarification = await self._clarifier.evaluate(
                user_id=user_id,
                utterance=utterance,
                candidate=candidate,
                contradiction_hint=contradiction_hint,
            )

        # 5) If clarification is required, DO NOT store yet — the caller will
        #    ask the question and re-ingest after the user replies.
        if clarification.needs_clarification:
            return IngestResult(
                stored=None,
                candidate=candidate,
                clarification=clarification,
                contradictions=contradictions,
            )

        # 6) Persist.
        episode = Episode.from_candidate(candidate)
        await self._repo.upsert(episode)
        logger.info(
            "Stored episode %s (user=%s, category=%s, priority=%s)",
            episode.episode_id,
            user_id,
            episode.category.value,
            episode.clinical_priority.value,
        )

        return IngestResult(
            stored=episode,
            candidate=candidate,
            clarification=clarification,
            contradictions=contradictions,
        )

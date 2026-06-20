"""
Episodic Memory Layer routes.

Six endpoints, kept minimal and explicit:

  POST /episodic/extract        — utterance → EpisodeCandidate (no storage)
  POST /episodic/store          — store an Episode (or extract+store in one shot)
  POST /episodic/retrieve       — ranked retrieval (no compression)
  POST /episodic/clarify        — clarification triage for an utterance
  POST /episodic/context        — end-to-end: retrieve + rank + compress
  POST /episodic/contradictions — contradiction detection against prior episodes
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from episodic.api.dependencies import ContainerDep
from episodic.schemas.clarification import (
    ClarificationRequest,
    ClarificationResponse,
)
from episodic.schemas.contradiction import ContradictionReport
from episodic.schemas.episode import Episode, EpisodeCandidate
from episodic.schemas.retrieval import ContextBlock, RetrievalRequest, RetrievalResponse


router = APIRouter(prefix="/episodic", tags=["episodic-memory"])


# ---------------------------------------------------------------------------
# Request models that don't directly map to a schema in /schemas
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    user_id: str
    utterance: str


class StoreRequest(BaseModel):
    """
    Either pass an utterance (server runs extract+clarify+contradiction first)
    OR pass an already-built EpisodeCandidate to upsert directly.
    """
    user_id: str
    utterance: str | None = None
    candidate: EpisodeCandidate | None = None
    skip_clarifier: bool = False


class StoreResponse(BaseModel):
    stored: Episode | None
    candidate: EpisodeCandidate | None
    clarification: ClarificationResponse
    contradictions: ContradictionReport


class ContradictionRequest(BaseModel):
    user_id: str
    new_claim: str
    top_k: int = Field(default=10, ge=1, le=50)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/extract", response_model=Optional[EpisodeCandidate])
async def extract_endpoint(req: ExtractRequest, ctx: ContainerDep) -> EpisodeCandidate | None:
    candidate = await ctx.extractor.extract(user_id=req.user_id, utterance=req.utterance)
    return candidate


@router.post("/store", response_model=StoreResponse)
async def store_endpoint(req: StoreRequest, ctx: ContainerDep) -> StoreResponse:
    # Path A: caller already has a candidate — upsert directly (no clarifier).
    if req.candidate is not None:
        candidate = req.candidate.model_copy(update={"user_id": req.user_id})
        if not candidate.store_memory:
            return StoreResponse(
                stored=None,
                candidate=candidate,
                clarification=ClarificationResponse(needs_clarification=False),
                contradictions=ContradictionReport(
                    user_id=req.user_id, has_contradictions=False
                ),
            )
        episode = Episode.from_candidate(candidate)
        await ctx.repository.upsert(episode)
        return StoreResponse(
            stored=episode,
            candidate=candidate,
            clarification=ClarificationResponse(needs_clarification=False),
            contradictions=ContradictionReport(
                user_id=req.user_id, has_contradictions=False
            ),
        )

    # Path B: run the full ingest pipeline from the utterance.
    if not req.utterance:
        raise HTTPException(status_code=400, detail="utterance or candidate is required")

    result = await ctx.ingest_pipeline.run(
        user_id=req.user_id,
        utterance=req.utterance,
        skip_clarifier=req.skip_clarifier,
    )
    return StoreResponse(
        stored=result.stored,
        candidate=result.candidate,
        clarification=result.clarification,
        contradictions=result.contradictions,
    )


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_endpoint(req: RetrievalRequest, ctx: ContainerDep) -> RetrievalResponse:
    ranked = await ctx.retriever.retrieve(req)
    return RetrievalResponse(
        user_id=req.user_id, query_text=req.query_text, episodes=ranked
    )


@router.post("/clarify", response_model=ClarificationResponse)
async def clarify_endpoint(req: ClarificationRequest, ctx: ContainerDep) -> ClarificationResponse:
    candidate: EpisodeCandidate | None = None
    if req.candidate:
        try:
            candidate = EpisodeCandidate.model_validate(req.candidate)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid candidate: {exc}") from exc
    return await ctx.clarifier.evaluate(
        user_id=req.user_id,
        utterance=req.utterance,
        candidate=candidate,
    )


@router.post("/context", response_model=ContextBlock)
async def context_endpoint(req: RetrievalRequest, ctx: ContainerDep) -> ContextBlock:
    return await ctx.context_pipeline.build(req)


@router.post("/contradictions", response_model=ContradictionReport)
async def contradictions_endpoint(
    req: ContradictionRequest, ctx: ContainerDep
) -> ContradictionReport:
    prior_ranked = await ctx.retriever.retrieve(
        RetrievalRequest(
            user_id=req.user_id,
            query_text=req.new_claim,
            top_k=req.top_k,
            return_k=req.top_k,
        )
    )
    prior_episodes = [r.episode for r in prior_ranked]
    return await ctx.contradiction.detect(
        user_id=req.user_id,
        new_claim=req.new_claim,
        prior_episodes=prior_episodes,
    )

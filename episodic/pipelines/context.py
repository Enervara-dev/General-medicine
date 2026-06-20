"""
ContextPipeline — retrieve → rank → compress → render prompt block.

This is the entry point the Retrieval Orchestrator should call to obtain
patient-specific episodic context for a query. The rendered prompt block
is ready to drop into the answer LLM's context.
"""

from __future__ import annotations

import logging
from typing import Any

from episodic.schemas.retrieval import (
    CompressedEpisode,
    ContextBlock,
    RankedEpisode,
    RetrievalRequest,
)
from episodic.services.compression import CompressionService
from episodic.services.retriever import RetrieverService

logger = logging.getLogger(__name__)


class ContextPipeline:
    def __init__(
        self,
        *,
        retriever: RetrieverService,
        compression: CompressionService,
    ) -> None:
        self._retriever = retriever
        self._compression = compression

    async def build(self, req: RetrievalRequest) -> ContextBlock:
        ranked = await self._retriever.retrieve(req)
        if not ranked:
            return ContextBlock(
                user_id=req.user_id,
                query_text=req.query_text,
                episodes=[],
                compressed=[],
                rendered_prompt="",
                metadata={"strategy": "empty"},
            )

        kept, compressed = await self._compression.compress(ranked)
        rendered = _render_prompt_block(kept, compressed)

        return ContextBlock(
            user_id=req.user_id,
            query_text=req.query_text,
            episodes=kept,
            compressed=compressed,
            rendered_prompt=rendered,
            metadata={
                "strategy": "retrieve+rank+compress",
                "raw_count": len(ranked),
                "kept_count": len(kept),
                "compressed_count": len(compressed),
            },
        )


def _render_prompt_block(
    kept: list[RankedEpisode],
    compressed: list[CompressedEpisode],
) -> str:
    sections: list[str] = []

    if compressed:
        lines = ["=== RECURRING / LONG-RUNNING THEMES ==="]
        for c in compressed:
            window = f"{c.first_seen.date()} → {c.last_seen.date()}"
            sev = f" peak: {c.peak_severity}" if c.peak_severity else ""
            lines.append(f"- [{c.category}] {c.summary} ({window};{sev})")
        sections.append("\n".join(lines))

    if kept:
        lines = ["=== RECENT EPISODIC MEMORY ==="]
        for r in kept:
            ep = r.episode
            ts = ep.timestamp.date()
            sev = f" ({ep.severity.value})" if ep.severity.value != "unknown" else ""
            prio = f" [{ep.clinical_priority.value}]"
            lines.append(f"- {ts}{prio}{sev} {ep.summary}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)

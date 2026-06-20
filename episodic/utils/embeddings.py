"""
Pinecone hosted-embedding helper.

Episodic memory uses the same llama-text-embed-v2 model as the main RAG
index so vectors are interchangeable should you ever want to share an index.
"""

from __future__ import annotations

import asyncio
from typing import Iterable

from pinecone import Pinecone

from episodic.config import EpisodicConfig
from episodic.utils.retry import async_retry


class EmbeddingClient:
    def __init__(self, pc: Pinecone | None = None) -> None:
        if not EpisodicConfig.PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is missing. Set it in your .env."
            )
        self._pc = pc or Pinecone(api_key=EpisodicConfig.PINECONE_API_KEY)

    @async_retry(attempts=3, base_delay_s=0.5)
    async def embed_query(self, text: str) -> list[float]:
        """Embed one query string. Pinecone inference is sync → run in thread."""
        result = await asyncio.to_thread(
            self._pc.inference.embed,
            model=EpisodicConfig.PINECONE_EMBED_MODEL,
            inputs=[text],
            parameters={"input_type": "query", "truncate": "END"},
        )
        return list(result[0]["values"])

    @async_retry(attempts=3, base_delay_s=0.5)
    async def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of passage texts."""
        texts = list(texts)
        if not texts:
            return []
        result = await asyncio.to_thread(
            self._pc.inference.embed,
            model=EpisodicConfig.PINECONE_EMBED_MODEL,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        return [list(item["values"]) for item in result]

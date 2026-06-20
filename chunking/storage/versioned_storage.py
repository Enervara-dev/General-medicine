import json
import logging
from pathlib import Path
from typing import List
from chunking.schemas.models import MicroChunk
from chunking.config.settings import settings

logger = logging.getLogger(__name__)

class VersionedStorage:
    def save_chunk(self, chunk: MicroChunk, index: int, version: str = "v1"):
        category = chunk.source.book.lower().replace(" ", "_")
        doc_id = chunk.source.topic
        
        # User request: chunking/output/v1/internal_medicine/.../<chunk_id>.json
        base_dir = Path("chunking/output") / version / category / doc_id
        base_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = base_dir / f"{chunk.chunk_id}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(chunk.model_dump(), indent=2))
        logger.info(f"Saved chunk to {file_path}")

    def save_chunks(self, chunks: List[MicroChunk], version: str = "v1"):
        if not chunks:
            return
        category = chunks[0].source.book.lower().replace(" ", "_")
        base_dir = Path("data") / category
        base_dir.mkdir(parents=True, exist_ok=True)
        doc_id = chunks[0].source.topic
        file_path = base_dir / f"{doc_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps([c.model_dump() for c in chunks], indent=2))
        logger.info(f"Saved {len(chunks)} chunks to {file_path}")

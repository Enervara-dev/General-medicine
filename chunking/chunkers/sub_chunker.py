import datetime
from typing import List
from chunking.schemas.models import SemanticBlock, ExtractedClinicalData, MicroChunk, ChunkSource, ChunkMetadata
from chunking.config.settings import settings

class SubChunker:
    def create_micro_chunks(self, block: SemanticBlock, data: ExtractedClinicalData, max_tokens: int = 256) -> List[MicroChunk]:
        chunks = []
        
        source = ChunkSource(
            book=block.metadata.book_type,
            chapter=block.section.lower().replace(" ", "_") if block.section else "general",
            topic=block.metadata.doc_id,
            page="unknown"
        )
        
        metadata = ChunkMetadata(
            tokens=int(len(block.text.split()) * 1.3),
            llm_model=settings.model_primary,
            version=block.metadata.version,
            created_at=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        
        # If 3 or fewer entities, the block is focused enough to be one micro-chunk
        if len(data.entities) <= 3:
            mc = MicroChunk(
                chunk_id=f"{block.block_id}-micro-1",
                source=source,
                text=block.text,
                entities=data.entities,
                relations=data.relations,
                summary=data.summary,
                clinical_significance=data.clinical_significance,
                metadata=metadata
            )
            chunks.append(mc)
            return chunks

        primary_types = {"disease", "drug", "procedure", "symptom", "cause", "test"}
        primary_entities = [e for e in data.entities if e.type in primary_types]
        
        if not primary_entities:
            primary_entities = data.entities[:1]

        for i, focus_entity in enumerate(primary_entities):
            related_entity_names = {focus_entity.name}
            relevant_relations = []
            
            for rel in data.relations:
                if rel.source == focus_entity.name or rel.target == focus_entity.name:
                    relevant_relations.append(rel)
                    related_entity_names.add(rel.source)
                    related_entity_names.add(rel.target)
            
            relevant_entities = [e for e in data.entities if e.name in related_entity_names]
            
            mc = MicroChunk(
                chunk_id=f"{block.block_id}-focus-{i}",
                source=source,
                text=block.text,
                entities=relevant_entities,
                relations=relevant_relations,
                summary=data.summary,
                clinical_significance=data.clinical_significance,
                metadata=metadata
            )
            chunks.append(mc)

        return chunks

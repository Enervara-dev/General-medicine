from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import datetime

class DocumentMetadata(BaseModel):
    doc_id: str
    book_type: str
    version: str
    source_path: str

class ChunkSource(BaseModel):
    book: str
    chapter: str
    topic: str
    page: str

class ChunkMetadata(BaseModel):
    tokens: str | int
    model: str
    quality_check: str
    version: Optional[str] = None
    created_at: Optional[str] = None

class ClinicalEntity(BaseModel):
    name: str = Field(..., description="Name of the entity")
    type: Literal[
        "disease", "symptom", "risk_factor", "biomarker", "clinical_process", 
        "intervention", "care_model", "support_service", "policy", 
        "financial_resource", "decision_state", "socioeconomic_condition", 
        "anatomical_entity"
    ]
    normalized_name: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

class ClinicalRelation(BaseModel):
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    type: Literal[
        "causes", "contributes_to", "increases_risk_of", "reduces_risk_of", 
        "leads_to", "manifests_as", "affects", "includes", "requires", 
        "treats", "mitigates", "alleviates", "complicates", "mimics", 
        "alternative_to", "associated_with_preference_for", "increases_likelihood_of"
    ]

class MicroChunk(BaseModel):
    chunk_id: str
    source: ChunkSource
    text: str
    entities: List[ClinicalEntity]
    relations: List[ClinicalRelation]
    summary: str
    clinical_significance: str
    metadata: ChunkMetadata

    @model_validator(mode='after')
    def enforce_strict_validation(self) -> 'MicroChunk':
        num_entities = len(self.entities)
        num_relations = len(self.relations)
        
        if num_entities < 5:
            raise ValueError(f"Failure: len(entities) < 5 (found {num_entities})")
            
        if num_relations < (num_entities / 2.0):
            raise ValueError(f"Failure: len(relations) < len(entities)/2. Entities: {num_entities}, Relations: {num_relations}")
            
        # simple word-based token estimation constraint 
        approx_tokens = len(self.text.split()) * 1.3
        if approx_tokens > 400:
            raise ValueError(f"Failure: tokens > 400. This chunk text is too large (approx {int(approx_tokens)} tokens).")
            
        return self

class ExtractedClinicalData(BaseModel):
    chunks: List[MicroChunk]

class SemanticBlock(BaseModel):
    block_id: str
    text: str
    section: Optional[str]
    metadata: DocumentMetadata

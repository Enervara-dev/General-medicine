import json
import pydantic
from typing import Optional
from chunking.schemas.models import ExtractedClinicalData

class OutputValidator:
    def validate(self, raw_json: str) -> Optional[ExtractedClinicalData]:
        # JSON standard cleanup (often LLMs wrap in markdown)
        raw_json = raw_json.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
            
        try:
            # First gate: JSON Parsing Validity
            data = json.loads(raw_json)
            
            # Second gate: Strict Pydantic Schema Validation
            parsed_data = ExtractedClinicalData(**data)
            
            # Third gate: Semantic Validation
            for chunk in parsed_data.chunks:
                valid_relations = 0
                entity_names = {e.name for e in chunk.entities}
                for rel in chunk.relations:
                    if rel.source in entity_names and rel.target in entity_names:
                        valid_relations += 1
                        
                if chunk.relations and valid_relations == 0:
                    raise ValueError(f"All relations in chunk {chunk.chunk_id} have hallucinated or disconnected sources/targets")
                
            return parsed_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except pydantic.ValidationError as e:
            raise ValueError(f"Schema violation: {e}")

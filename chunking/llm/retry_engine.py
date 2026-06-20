import json
from chunking.llm.client import LLMEngine
from chunking.validators.schema_validator import OutputValidator
from chunking.schemas.models import ExtractedClinicalData
from typing import Optional

class ExtractionWithRetry:
    def __init__(self):
        self.llm = LLMEngine()
        self.validator = OutputValidator()

    def run(self, text: str) -> Optional[ExtractedClinicalData]:
        schema = ExtractedClinicalData.model_json_schema()
        schema_str = json.dumps(schema)
        
        max_retries = 2
        current_error = ""
        prompt_text = text
        
        for attempt in range(max_retries):
            force_fallback = (attempt == max_retries - 1)
            raw_output, llm_err = self.llm.extract_structured_data(prompt_text, schema_str, max_retries=1, force_fallback=force_fallback)
            
            if llm_err:
                current_error = llm_err
                continue
                
            try:
                structured_data = self.validator.validate(raw_output)
                return structured_data
            except Exception as e:
                current_error = str(e)
                prompt_text = f"""
ORIGINAL TEXT:
{text}

PREVIOUS BAD OUTPUT:
{raw_output}

VALIDATION ERROR REASON:
{current_error}

Retry extraction and correct the JSON to pass strict validation rules above.
"""
                
        # If it reaches here, it failed entirely.
        # Ensure logs/failed_blocks exists and log it
        import os
        from pathlib import Path
        failed_dir = Path("logs/failed_blocks")
        failed_dir.mkdir(parents=True, exist_ok=True)
        # We can use a hash of the text to identify the failed block
        error_file = failed_dir / f"failed_{hash(text)}.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"Failed extraction after {max_retries} attempts.\nError: {current_error}\n\nBad Output:\n{raw_output if 'raw_output' in locals() else 'None'}\n\nText:\n{text}")
            
        return None

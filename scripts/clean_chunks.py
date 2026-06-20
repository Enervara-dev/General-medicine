import os
import glob
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from chunking.schemas.models import MicroChunk
from graphrag.config.settings import settings
from graphrag.llm.gemini_client import DEFAULT_MODEL, generate_text

load_dotenv()
if not settings.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

CLEANING_MODEL = settings.CLEANING_MODEL or DEFAULT_MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SYSTEM_PROMPT = """You are a clinical knowledge normalization engine.

Your task is to refine and standardize a JSON chunk for semantic grounding, ontology alignment, and downstream graph construction.

----------------------------------------
INPUT
- A JSON object containing:
- text
- entities
- relations

----------------------------------------
CRITICAL RULES

1. DO NOT add new medical facts.
2. DO NOT invent new entities not present or clearly implied in the text.
3. DO NOT change the meaning of the content.
4. You MAY refine wording, normalize types, and improve relation semantics.
5. You MAY remove duplicate or invalid relations.
6. Preserve the original structure and all valid information.

----------------------------------------
ENTITY NORMALIZATION

Normalize entity types using a consistent clinical ontology.

Allowed types:

- disease
- symptom
- risk_factor
- biomarker
- clinical_process
- intervention
- care_model
- support_service
- policy
- financial_resource
- decision_state
- socioeconomic_condition
- anatomical_entity

Replace vague or incorrect types (e.g., "mechanism", "procedure", "test") with the closest correct type.

----------------------------------------
RELATION NORMALIZATION

Replace vague relations such as "associated_with" with precise types.

Allowed relation types:

- causes
- contributes_to
- increases_risk_of
- reduces_risk_of
- leads_to
- manifests_as
- affects
- includes
- requires
- treats
- mitigates
- alleviates
- complicates
- mimics
- alternative_to
- associated_with_preference_for
- increases_likelihood_of

Choose the most semantically accurate relation based on the text.

----------------------------------------
SAFETY + CLINICAL PRECISION

- Avoid overstating causality.
- If the relationship is associative or correlational, use:
- contributes_to
- increases_likelihood_of
- associated_with_preference_for
- Do NOT convert correlation into causation.

----------------------------------------
GRAPH CLEANING

- Remove duplicate relations
- Remove contradictory relations
- Ensure all relations reference valid entities
- Keep relations meaningful and clinically valid

----------------------------------------
OUTPUT FORMAT

Return ONLY valid JSON with the SAME structure:

{
"chunk_id": "...",
"text": "...",
"entities": [...],
"relations": [...],
"summary": "...",
"clinical_significance": "...",
"metadata": {...}
}

----------------------------------------
CONSTRAINTS

- Do not change field names
- Do not remove valid entities
- Do not add new sections
- Keep output deterministic and clean
- No explanations, only JSON

----------------------------------------
GOAL

Produce a semantically grounded, ontology-aligned, and graph-ready chunk suitable for downstream Neo4j ingestion."""

def clean_chunk(file_path: str, output_dir: str):
    basename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, basename)
    
    if os.path.exists(output_path):
        return f"Skipped {basename}"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Identify fields to pass as per schema formatting
    input_data = {
        "text": data.get("text", ""),
        "entities": data.get("entities", []),
        "relations": data.get("relations", []),
        "summary": data.get("summary", ""),
        "clinical_significance": data.get("clinical_significance", ""),
        "chunk_id": data.get("chunk_id", ""),
        "metadata": data.get("metadata", {}),
        "source": data.get("source", {})
    }

    try:
        content = generate_text(
            json.dumps(input_data),
            model=CLEANING_MODEL,
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            json_mode=True,
        )
        if not content:
            raise ValueError("Gemini returned empty content")

        cleaned_json = json.loads(content)
        
        # Reconstruct missing fields like source if the LLM drops them
        if "source" not in cleaned_json and "source" in input_data:
            cleaned_json["source"] = input_data["source"]
        if "chunk_id" not in cleaned_json and "chunk_id" in input_data:
            cleaned_json["chunk_id"] = input_data["chunk_id"]
        
        # Enforce strict validation
        MicroChunk.model_validate(cleaned_json)
        
        # Write clean output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_json, f, indent=2)
            
        return f"Processed {basename}"
        
    except Exception as e:
        # Avoid crashing whole thread on validation/API err
        error_msg = str(e)
        logging.error(f"Failed on {basename}: {error_msg}")
        with open(output_path + ".failed", "w", encoding="utf-8") as f:
            f.write(error_msg)
            
        return f"Failed {basename}"

def run_pipeline():
    input_dir = os.path.join("chunking", "output", "v1")
    output_dir = os.path.join("chunking", "output", "v2_cleaned")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all JSON chunks recursively so we get all books/categories
    all_chunks = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    logging.info(f"Found {len(all_chunks)} chunks to clean.")
    
    # Strictly respect max concurrent workers = 5 limit 
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(clean_chunk, f, output_dir): f for f in all_chunks}
        
        for future in as_completed(futures):
            try:
                res = future.result()
                logging.info(res)
            except Exception as e:
                logging.error(f"Worker exception: {e}")

if __name__ == "__main__":
    run_pipeline()
    logging.info("Cleaning pipeline fully complete.")

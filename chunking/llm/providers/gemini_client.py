import logging
from typing import Optional

from graphrag.llm.gemini_client import generate_text
from chunking.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMProvider):
    """
    Structured-JSON extraction provider backed by Google Gemini.
    """

    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        # api_key is accepted for interface symmetry; the shared client reads
        # GEMINI_API_KEY from settings.

    def generate_json(self, prompt: str, schema_json: str) -> tuple[Optional[str], str]:
        full_prompt = f"""
You are a clinical knowledge extraction engine.

Your task is to convert the given medical text into STRICT structured JSON for a Graph-based medical reasoning system.

-----------------------------------
CRITICAL RULES (NON-NEGOTIABLE)
-----------------------------------

1. OUTPUT FORMAT
- Output MUST be valid JSON only
- No explanations, no markdown, no extra text
- Must strictly follow the schema

2. FULL ENTITY RECALL (MANDATORY)
- Extract ALL clinical entities present in the text
- Do NOT miss:
  - diseases
  - symptoms
  - drugs (individual + classes)
  - procedures
  - diagnostic tests
  - biological mechanisms (e.g., receptors like 5-HT3, NK1)
  - syndromes
- If entities exist and you return less than 8 → you FAILED

3. RELATION DENSITY (MANDATORY)
- Every entity must participate in ≥1 relation if logically possible
- Extract ALL meaningful relations:
  - causes
  - treats
  - associated_with
  - diagnosed_by
  - mediated_by
  - contraindicated_with
- If relations are sparse → you FAILED

4. STRUCTURE-AWARE SPLITTING
- If input contains MULTIPLE clinical concepts:
  → SPLIT internally into MULTIPLE chunks
- Each chunk must represent:
  → ONE dominant clinical concept (disease / drug class / mechanism)

Examples of split triggers:
- new disease mentioned
- new drug class introduced
- new section (e.g., "MECHANISM", "TREATMENT")
- tables or lists

5. NOISE REMOVAL
- IGNORE:
  - figure captions
  - table labels (e.g., TABLE 6-5)
  - page headers/footers
  - broken OCR tokens

6. TOKEN CONTROL
- Each chunk MUST be concise (approx 150–350 tokens)
- NEVER merge large sections into one chunk

7. NORMALIZATION
- Normalize entity names:
  - lowercase
  - snake_case for normalized_name
- Keep original name also

-----------------------------------
OUTPUT SCHEMA (STRICT)
-----------------------------------
{schema_json}

-----------------------------------
QUALITY CHECK BEFORE OUTPUT
-----------------------------------

Before returning JSON, verify:

- Did I extract ALL entities? (not just obvious ones)
- Are relations dense and meaningful?
- Did I split multiple concepts into separate chunks?
- Is there any noise or irrelevant text included?
- Is JSON strictly valid?

If ANY answer is NO → FIX before output.

-----------------------------------
INPUT TEXT
-----------------------------------

{prompt}
"""

        try:
            content = generate_text(
                full_prompt,
                model=self.model_name,
                temperature=0.1,
                json_mode=True,
            )
            return content, ""
        except Exception as e:
            logger.warning(f"Gemini provider failed on model {self.model_name}: {e}")
            return None, str(e)

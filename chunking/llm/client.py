import logging
from typing import Optional

from chunking.config.settings import settings
from chunking.llm.providers.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self):
        # Both providers are Gemini, differing only in model size.
        self.primary_provider = GeminiClient(settings.gemini_api_key, settings.model_primary)
        self.fallback_provider = GeminiClient(settings.gemini_api_key, settings.model_fallback)

    def extract_structured_data(
        self,
        text: str,
        schema_json: str,
        max_retries: int = 3,
        force_fallback: bool = False,
    ) -> tuple[Optional[str], str]:
        """
        Coordinates the extraction across primary/fallback providers.
        """
        for attempt in range(max_retries):
            use_fallback = force_fallback or (max_retries > 1 and attempt == max_retries - 1)
            provider = self.fallback_provider if use_fallback else self.primary_provider

            # The prompt string holds the error injection on retries
            content, error = provider.generate_json(text, schema_json)

            if not error:
                return content, ""

            # If there's an error, inject it into the prompt for the next try
            logger.warning(f"Attempt {attempt+1} extraction failed: {error}")
            text = f"{text}\n\nWarning: Previous parsing failed because: {error}. Please correct this and emit valid JSON."

        return None, "Max retries exceeded"

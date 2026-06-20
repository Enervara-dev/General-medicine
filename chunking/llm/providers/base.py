import os
from abc import ABC, abstractmethod
from typing import Optional

class BaseLLMProvider(ABC):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    def generate_json(self, prompt: str, schema_json: str) -> tuple[Optional[str], str]:
        """
        Generate JSON structured data.
        Returns:
            Tuple of (extracted_json_string, error_message)
        """
        pass

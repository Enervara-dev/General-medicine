"""
Streaming Gemini-backed answer generator used by the GraphRAG pipeline.
"""


from __future__ import annotations

import time

from graphrag.config.settings import settings
from graphrag.llm.gemini_client import (
    DEFAULT_MODEL,
    generate_stream,
    get_client,
)
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLM:
    def __init__(self):
        # Fail fast if the API key is missing.
        get_client()
        self._model = settings.ANSWER_MODEL or DEFAULT_MODEL

    def generate_from_messages(self, messages: list[dict[str, str]]):
        logger.info("[3/3] Sending memory-aware structured context to LLM Engine...")
        system_instruction, user_prompt = _split_messages(messages)
        return self._stream(system_instruction=system_instruction, user_prompt=user_prompt)

    def generate_response(
        self,
        query_text: str,
        vector_context: str,
        graph_context: str,
        memory_context: str = "",
        conversation_history: str = "",
        query_type: str = "unknown",
        goal: str = "provide a medical answer",
        risk_level: str = "none",
    ):
        """
        Streaming Gemini answer for the legacy CLI path.

        The system prompt is built via the same layered composer the FastAPI
        path uses — `app.services.orchestration.prompt_layers` — so both
        paths share a single source of truth. CLI doesn't carry per-turn
        risk into this method today, so `risk_level` defaults to `"none"`.
        """
        from app.services.orchestration.prompt_layers import compose_system_prompt

        logger.info("[3/3] Sending structured context to LLM Engine...")

        has_name = "Patient name:" in memory_context
        system_prompt = compose_system_prompt(
            query_type=query_type,
            risk_level=risk_level,
            has_name=has_name,
        )

        user_prompt = f"""
USER QUESTION: {query_text}

=== STRUCTURED CLINICAL MEMORY ===
{memory_context}

=== RECENT CONVERSATION ===
{conversation_history}

=== RETRIEVED MEDICAL CONTEXT ===
{vector_context}

=== GRAPH RELATIONS ===
{graph_context}
"""

        return self._stream(system_instruction=system_prompt, user_prompt=user_prompt)

    def _stream(self, *, system_instruction: str | None, user_prompt: str) -> str | None:
        try:
            t_start = time.monotonic()

            logger.info("\n" + "=" * 80)
            logger.info("AI RESPONSE")
            logger.info("=" * 80 + "\n")

            answer = ""
            t_first_visible: float | None = None

            for piece in generate_stream(
                model=self._model,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
            ):
                if t_first_visible is None:
                    t_first_visible = time.monotonic()
                    logger.info(
                        f"⏱️  Time-to-first-visible-token: "
                        f"{(t_first_visible - t_start) * 1000:.0f}ms"
                    )
                print(piece, end="", flush=True)
                answer += piece

            t_end = time.monotonic()
            logger.info(
                f"\n⏱️  Stream complete in {(t_end - t_start) * 1000:.0f}ms "
                f"({len(answer)} visible chars)"
            )
            print("\n\n" + "=" * 80 + "\n")
            return answer

        except Exception as e:
            logger.error(f"\nLLM Error: {e}")
            return None


def _split_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    """
    Collapse an OpenAI-style messages array into (system_instruction, user_prompt)
    that Gemini's generate_content API expects.

    System messages are concatenated into the system_instruction. The remaining
    user/assistant turns are joined into a single user prompt with role prefixes
    so multi-turn context is preserved.
    """
    system_parts: list[str] = []
    body_parts: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            body_parts.append(f"Assistant: {content}")
        else:
            body_parts.append(f"User: {content}")
    system_instruction = "\n\n".join(p for p in system_parts if p).strip() or None
    user_prompt = "\n\n".join(p for p in body_parts if p).strip()
    return system_instruction, user_prompt

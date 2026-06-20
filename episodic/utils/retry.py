"""
Minimal async retry decorator. Exponential backoff, no external dependency.

Intentionally tiny — pulling in tenacity for a single decorator usage is
excess. If the retry policy ever grows non-trivial branches, swap to tenacity.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def async_retry(
    *,
    attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(attempts):
                try:
                    return await fn(*args, **kwargs)
                except retry_on as exc:
                    last_exc = exc
                    if attempt == attempts - 1:
                        raise
                    delay = min(max_delay_s, base_delay_s * (2 ** attempt))
                    delay = delay * (0.7 + 0.6 * random.random())  # jitter
                    logger.warning(
                        "Retry %d/%d for %s after %s: %s",
                        attempt + 1, attempts, fn.__name__, exc.__class__.__name__, exc,
                    )
                    await asyncio.sleep(delay)
            # Unreachable, but satisfies type checker.
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator

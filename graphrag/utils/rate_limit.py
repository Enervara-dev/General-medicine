import logging
import random
import time
from typing import Any, Callable

import requests


logger = logging.getLogger(__name__)


def _retry_after_seconds(headers: Any) -> float | None:
    if not headers:
        return None

    value = None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None

    if not value:
        return None

    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _exception_status_and_headers(exc: Exception) -> tuple[int | None, Any]:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)

    if response is not None:
        status_code = status_code or getattr(response, "status_code", None)
        return status_code, getattr(response, "headers", None)

    return status_code, None


def retry_delay(attempt: int, base_delay: float, max_delay: float, headers: Any = None) -> float:
    retry_after = _retry_after_seconds(headers)
    if retry_after is not None:
        return min(retry_after, max_delay)

    exponential = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, min(1.5, exponential * 0.25))
    return exponential + jitter


def request_with_retries(
    method: str,
    url: str,
    *,
    max_retries: int = 6,
    base_delay: float = 4.0,
    max_delay: float = 90.0,
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
    **kwargs: Any,
) -> requests.Response:
    last_response: requests.Response | None = None

    for attempt in range(max_retries):
        response = requests.request(method, url, **kwargs)
        if response.status_code not in retry_statuses:
            return response

        last_response = response
        if attempt == max_retries - 1:
            return response

        delay = retry_delay(attempt, base_delay, max_delay, response.headers)
        logger.warning(
            "HTTP %s from model provider. Retrying in %.1f seconds (%s/%s).",
            response.status_code,
            delay,
            attempt + 1,
            max_retries,
        )
        time.sleep(delay)

    return last_response  # type: ignore[return-value]


def call_with_retries(
    fn: Callable[..., Any],
    *,
    max_retries: int = 6,
    base_delay: float = 4.0,
    max_delay: float = 90.0,
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
    operation: str = "model call",
    **kwargs: Any,
) -> Any:
    for attempt in range(max_retries):
        try:
            return fn(**kwargs)
        except Exception as exc:
            status_code, headers = _exception_status_and_headers(exc)
            should_retry = status_code in retry_statuses

            if not should_retry or attempt == max_retries - 1:
                raise

            delay = retry_delay(attempt, base_delay, max_delay, headers)
            logger.warning(
                "%s failed with HTTP %s. Retrying in %.1f seconds (%s/%s).",
                operation,
                status_code,
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)


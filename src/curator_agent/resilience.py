"""Shared exponential-backoff retry decorator.

Retries only transient failures — network errors, HTTP 429 and 5xx — from any
client (requests, google-genai, groq, notion). Other 4xx errors fail fast.
A ``Retry-After`` header on a 429 overrides the backoff delay.
"""
from __future__ import annotations

import functools
import logging
import time

import requests

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
MAX_WAIT_SECONDS = 64.0

# Indirection so tests can skip real sleeping.
_sleep = time.sleep


def _status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status across SDKs that each name it differently."""
    response = getattr(exc, "response", None)
    for candidate in (
        getattr(response, "status_code", None),
        getattr(exc, "status_code", None),  # groq
        getattr(exc, "code", None),         # google-genai
        getattr(exc, "status", None),       # notion-client
    ):
        if isinstance(candidate, int):
            return candidate
    return None


def is_retriable(exc: BaseException) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    status = _status_code(exc)
    if status is not None:
        return status == 429 or status >= 500
    # SDK connection errors without a status (groq.APIConnectionError, etc.)
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        value = headers.get("Retry-After")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def retry_with_backoff(func):
    """Retry ``func`` up to MAX_ATTEMPTS times with 1, 2, 4, ... second waits."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if attempt == MAX_ATTEMPTS or not is_retriable(exc):
                    raise
                wait = _retry_after_seconds(exc) or min(2 ** (attempt - 1), MAX_WAIT_SECONDS)
                logger.warning(
                    "%s attempt %d/%d failed (%s). Retrying in %.1fs.",
                    getattr(func, "__name__", "call"), attempt, MAX_ATTEMPTS, exc, wait,
                )
                _sleep(wait)

    return wrapper

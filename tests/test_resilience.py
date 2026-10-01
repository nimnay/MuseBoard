from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from curator_agent import resilience
from curator_agent.resilience import MAX_ATTEMPTS, is_retriable, retry_with_backoff


def _http_error(status: int, headers: dict | None = None) -> requests.HTTPError:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    return requests.HTTPError(response=resp)


class SdkError(Exception):
    def __init__(self, code: int):
        super().__init__(f"sdk {code}")
        self.code = code


@pytest.mark.parametrize("exc, expected", [
    (requests.ConnectionError(), True),
    (requests.Timeout(), True),
    (_http_error(429), True),
    (_http_error(503), True),
    (_http_error(400), False),
    (_http_error(404), False),
    (SdkError(500), True),   # google-genai style `.code`
    (SdkError(401), False),
    (ValueError("bad json"), False),
])
def test_is_retriable(exc, expected):
    assert is_retriable(exc) is expected


def test_retries_transient_errors_then_succeeds(monkeypatch):
    waits = []
    monkeypatch.setattr(resilience, "_sleep", waits.append)
    fn = MagicMock(side_effect=[_http_error(503), _http_error(503), "ok"])
    assert retry_with_backoff(fn)() == "ok"
    assert waits == [1, 2]


def test_honours_retry_after(monkeypatch):
    waits = []
    monkeypatch.setattr(resilience, "_sleep", waits.append)
    fn = MagicMock(side_effect=[_http_error(429, {"Retry-After": "7"}), "ok"])
    retry_with_backoff(fn)()
    assert waits == [7.0]


def test_does_not_retry_client_errors():
    fn = MagicMock(side_effect=_http_error(400))
    with pytest.raises(requests.HTTPError):
        retry_with_backoff(fn)()
    assert fn.call_count == 1


def test_gives_up_after_max_attempts():
    fn = MagicMock(side_effect=_http_error(500))
    with pytest.raises(requests.HTTPError):
        retry_with_backoff(fn)()
    assert fn.call_count == MAX_ATTEMPTS

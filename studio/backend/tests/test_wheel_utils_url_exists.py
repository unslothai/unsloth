"""url_exists must tell a missing wheel (404) from GitHub refusing us (403/429/5xx, dropped
connection): the second is retried and reported, or the caller starts a many-minute source
build for a wheel that exists."""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import wheel_utils  # noqa: E402


class _Ok:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, responder):
    calls = []

    def fake_urlopen(req, timeout = 10):
        calls.append(req.full_url)
        return responder(len(calls))

    monkeypatch.setattr(wheel_utils.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(wheel_utils.time, "sleep", lambda s: None)
    return calls


def test_a_404_is_final_after_one_probe(monkeypatch):
    def gone(_n):
        raise urllib.error.HTTPError("u", 404, "not found", None, None)

    calls = _patch(monkeypatch, gone)
    assert wheel_utils.url_exists("https://github.com/x/releases/download/v1/w.whl") is False
    assert len(calls) == 1


def test_a_refusal_is_retried_once_then_reported(monkeypatch, caplog):
    def refused(_n):
        raise urllib.error.HTTPError("u", 503, "unavailable", None, None)

    calls = _patch(monkeypatch, refused)
    with caplog.at_level("WARNING", logger = wheel_utils._logger.name):
        assert wheel_utils.url_exists("https://github.com/x/releases/download/v1/w.whl") is False
    assert len(calls) == 2
    assert any("HTTP 503" in rec.getMessage() for rec in caplog.records)


def test_a_transient_failure_recovers_on_the_retry(monkeypatch):
    def flaky(n):
        if n == 1:
            raise urllib.error.URLError("connection reset")
        return _Ok()

    calls = _patch(monkeypatch, flaky)
    assert wheel_utils.url_exists("https://github.com/x/releases/download/v1/w.whl") is True
    assert len(calls) == 2


def test_a_timeout_is_not_retried(monkeypatch):
    def slow(_n):
        raise urllib.error.URLError(TimeoutError("timed out"))

    calls = _patch(monkeypatch, slow)
    assert wheel_utils.url_exists("https://github.com/x/releases/download/v1/w.whl") is False
    assert len(calls) == 1

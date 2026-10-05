"""Regression tests for ddgs HTTP/2 transport resets (unslothai/unsloth#12638)."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference import tools


def test_transport_reset_detection_matches_primp_h2_errors():
    exc = Exception(
        "DecodeError('error decoding response body > request or response body error "
        "> error sending request > connection reset', None)"
    )
    assert tools._is_ddgs_transport_reset(exc) is True


def test_search_failure_message_for_transport_reset():
    exc = Exception("DDGSException: connection reset")
    message = tools._search_failure_message(exc, timeout = 7)
    assert "HTTP/2 connection" in message
    assert "HTTP/1.1" in message


def test_install_ddgs_transport_fallback_forces_duckduckgo_http1(monkeypatch):
    class FakeHttpxClient:
        http2 = True

        def close(self):
            self.closed = True

    class FakeHttpClient2:
        def __init__(self, headers=None, proxy=None, timeout=10, *, verify=True):
            self.client = FakeHttpxClient()
            self._headers = headers
            self._proxy = proxy
            self._timeout = timeout
            self._verify = verify

    ddgs_http_client = types.ModuleType("ddgs.http_client")
    ddgs_http_client.HttpClient = object

    ddgs_http_client2 = types.ModuleType("ddgs.http_client2")
    ddgs_http_client2.HttpClient2 = FakeHttpClient2

    def _fake_ssl_context(*, verify=True):
        return "ssl-context" if verify else False

    ddgs_http_client2._get_random_ssl_context = _fake_ssl_context

    created: list[dict] = []

    class RecordingHttpxClient:
        http2 = False

        def __init__(self, **kwargs):
            created.append(kwargs)

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "ddgs.http_client", ddgs_http_client)
    monkeypatch.setitem(sys.modules, "ddgs.http_client2", ddgs_http_client2)

    import httpx

    monkeypatch.setattr(httpx, "Client", RecordingHttpxClient)

    tools._DDGS_TRANSPORT_FALLBACK_INSTALLED = False
    tools._install_ddgs_transport_fallback()

    FakeHttpClient2(headers = {"User-Agent": "test"}, proxy = None, timeout = 4, verify = True)
    assert len(created) == 1
    assert created[0]["http2"] is False
    assert created[0]["headers"] == {"User-Agent": "test"}
    assert created[0]["timeout"] == 4

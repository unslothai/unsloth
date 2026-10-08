# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ddgs requests replay once over HTTP/1.1 after a connection reset (unslothai/unsloth#12638)."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference import tools


class FakeDDGSException(Exception):
    pass


class FieldsResponse:
    """ddgs HttpClient2's Response, and ddgs 9.8.0's primp one."""

    def __init__(self, status_code, content, text):
        self.status_code, self.content, self.text = status_code, content, text


class WrappingResponse:
    """ddgs 9.14's primp Response, which wraps the raw response."""

    def __init__(self, resp):
        self.status_code, self.content, self.text = resp.status_code, resp.content, resp.text


class FakeReplayResponse:
    status_code = 200
    content = b"ok"
    text = "ok"


RESET = FakeDDGSException(
    "RequestError: RequestError('error sending request for url (https://x/) > connection reset')"
)


def _install_fake_ddgs(monkeypatch, primp_response):
    """Fake ddgs modules whose client ``request`` raises ``state["error"]`` when set."""
    state = {"error": None, "replay_error": None, "calls": 0}

    class Session:
        headers = {"user-agent": "UA", "accept-encoding": "gzip, br, zstd"}

        def get_cookies(self, url):
            return {"country": "us"} if url == "https://x/" else {}

    class Session2:
        headers = {"user-agent": "DDG"}
        cookies = {"kl": "us-en"}

    class HttpClient:
        def __init__(
            self,
            proxy = None,
            timeout = 10,
            *,
            verify = True,
        ):
            self.client = Session()

        def request(self, *args, **kwargs):
            state["calls"] += 1
            if state["error"] is not None:
                raise state["error"]
            return "primary"

    class HttpClient2:
        def __init__(
            self,
            headers = None,
            proxy = None,
            timeout = 10,
            *,
            verify = True,
        ):
            self.client = Session2()

        request = HttpClient.request

    pkg = types.ModuleType("ddgs")
    http_client = types.ModuleType("ddgs.http_client")
    http_client.HttpClient, http_client.Response = HttpClient, primp_response
    http_client2 = types.ModuleType("ddgs.http_client2")
    http_client2.HttpClient2, http_client2.Response = HttpClient2, FieldsResponse
    exceptions = types.ModuleType("ddgs.exceptions")
    exceptions.DDGSException = FakeDDGSException

    class DDGS:
        def __init__(self, timeout = None):
            self.timeout = timeout

        def text(
            self,
            query,
            max_results = 5,
            backend = "auto",
        ):
            resp = HttpClient(timeout = self.timeout).request(
                "GET", "https://html.duckduckgo.com/html/"
            )
            return [{"title": "Weather", "href": "https://example.com/weather", "body": resp.text}]

    engines = types.ModuleType("ddgs.engines")
    engines.ENGINES = {"text": {"duckduckgo": object}}
    pkg.DDGS = DDGS
    pkg.http_client, pkg.http_client2, pkg.exceptions = http_client, http_client2, exceptions
    pkg.engines = engines
    for name, module in {
        "ddgs": pkg,
        "ddgs.http_client": http_client,
        "ddgs.http_client2": http_client2,
        "ddgs.exceptions": exceptions,
        "ddgs.engines": engines,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    replays = []

    def fake_replay(args, kwargs, config):
        replays.append((args, kwargs, config))
        if state["replay_error"] is not None:
            raise state["replay_error"]
        return FakeReplayResponse()

    monkeypatch.setattr(tools, "_ddgs_http1_replay", fake_replay, raising = False)
    return types.SimpleNamespace(
        state = state, replays = replays, HttpClient = HttpClient, HttpClient2 = HttpClient2
    )


@pytest.fixture(params = [WrappingResponse, FieldsResponse], ids = ["ddgs-9.14", "ddgs-9.8"])
def fake_ddgs(monkeypatch, request):
    fake = _install_fake_ddgs(monkeypatch, request.param)
    tools._install_ddgs_http1_retry()
    return fake


def test_web_search_returns_results_after_a_connection_reset(monkeypatch):
    fake = _install_fake_ddgs(monkeypatch, WrappingResponse)
    fake.state["error"] = RESET

    def no_wikipedia(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(tools, "_wikipedia_search", no_wikipedia)
    out = tools._web_search("weather", timeout = 5)
    assert "Search failed" not in out
    assert "https://example.com/weather" in out
    assert len(fake.replays) == 1


def test_reset_detection():
    assert tools._is_connection_reset(RESET)
    assert tools._is_connection_reset(Exception("h2 connection driver error: connection reset"))
    assert tools._is_connection_reset(Exception("RemoteProtocolError('Server disconnected')"))
    assert tools._is_connection_reset(
        FakeDDGSException(
            "RequestError: error sending request > stream closed because of a broken pipe"
        )
    )
    for message in (
        "ConnectError: [Errno -2] Name or service not known",
        "ConnectError: [Errno 111] Connection refused",
        "No results found.",
        "Request timed out",
    ):
        assert not tools._is_connection_reset(FakeDDGSException(message))


def test_install_is_idempotent(fake_ddgs):
    request, init = fake_ddgs.HttpClient.request, fake_ddgs.HttpClient2.__init__
    tools._install_ddgs_http1_retry()
    tools._install_ddgs_http1_retry()
    assert fake_ddgs.HttpClient.request is request
    assert fake_ddgs.HttpClient2.__init__ is init
    fake_ddgs.state["error"] = RESET
    fake_ddgs.HttpClient().request("GET", "https://x/")
    assert len(fake_ddgs.replays) == 1


def test_install_never_raises_without_ddgs(monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", None)
    tools._install_ddgs_http1_retry()


@pytest.mark.parametrize("cls_name", ["HttpClient", "HttpClient2"])
def test_healthy_request_is_untouched(fake_ddgs, cls_name):
    assert getattr(fake_ddgs, cls_name)().request("GET", "https://x/") == "primary"
    assert fake_ddgs.replays == []


@pytest.mark.parametrize("cls_name", ["HttpClient", "HttpClient2"])
def test_other_failures_are_not_replayed(fake_ddgs, cls_name):
    fake_ddgs.state["error"] = FakeDDGSException("ConnectError: Name or service not known")
    with pytest.raises(FakeDDGSException, match = "Name or service"):
        getattr(fake_ddgs, cls_name)().request("GET", "https://x/")
    assert fake_ddgs.replays == []


def test_primp_reset_replays_once_with_constructor_settings(fake_ddgs):
    fake_ddgs.state["error"] = RESET
    client = fake_ddgs.HttpClient("socks5://p:1", 4, verify = False)
    resp = client.request("POST", "https://x/", data = {"q": "weather"})
    assert (resp.status_code, resp.text) == (200, "ok")
    assert fake_ddgs.state["calls"] == 1
    [(args, kwargs, config)] = fake_ddgs.replays
    assert args == ("POST", "https://x/") and kwargs == {"data": {"q": "weather"}}
    assert 0 < config.pop("timeout") <= 4
    assert config == {
        "headers": {"user-agent": "UA"},
        "cookies": {"country": "us"},
        "proxy": "socks5://p:1",
        "verify": False,
        "follow_redirects": True,
    }


def test_duckduckgo_reset_replays_without_following_redirects(fake_ddgs):
    fake_ddgs.state["error"] = FakeDDGSException("WriteError: [Errno 104] Connection reset by peer")
    client = fake_ddgs.HttpClient2(headers = {"User-Agent": "DDG"}, timeout = 6)
    resp = client.request(method = "POST", url = "https://html.duckduckgo.com/html/")
    assert (resp.status_code, resp.text) == (200, "ok")
    [(_, kwargs, config)] = fake_ddgs.replays
    assert kwargs == {"method": "POST", "url": "https://html.duckduckgo.com/html/"}
    assert 0 < config["timeout"] <= 6 and config["follow_redirects"] is False
    assert config["cookies"] == {"kl": "us-en"}


def test_reset_after_the_timeout_is_spent_is_not_replayed(fake_ddgs, monkeypatch):
    fake_ddgs.state["error"] = RESET
    clock = iter([100.0, 104.5])
    monkeypatch.setattr(tools.time, "monotonic", lambda: next(clock))
    with pytest.raises(FakeDDGSException, match = "connection reset"):
        fake_ddgs.HttpClient(timeout = 4).request("GET", "https://x/")
    assert fake_ddgs.replays == []


def test_failed_replay_raises_ddgs_exception_with_both_errors(fake_ddgs):
    fake_ddgs.state["error"] = RESET
    fake_ddgs.state["replay_error"] = OSError("Connection reset by peer")
    with pytest.raises(FakeDDGSException) as info:
        fake_ddgs.HttpClient().request("GET", "https://x/")
    assert "connection reset" in str(info.value)
    assert "HTTP/1.1 retry failed" in str(info.value)
    assert len(fake_ddgs.replays) == 1

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Search distinguishes a provider challenge from a genuinely empty result."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ddgs.ddgs import DDGS
from ddgs.engines import ENGINES
from ddgs.http_client2 import HttpClient2
from ddgs.results import TextResult

from core.inference import tools


@pytest.fixture
def providers(monkeypatch):
    """Use the real DDGS engine selection and parsing with controlled HTTP responses."""
    monkeypatch.setattr(DDGS, "_get_network_client", lambda self: None, raising = False)
    # Search installs client wrappers globally; restore them after each test.
    monkeypatch.setattr(HttpClient2, "__init__", HttpClient2.__init__)
    for marker in ("_unsloth_http1_retry", "_unsloth_connect_timeout"):
        monkeypatch.setattr(HttpClient2, marker, getattr(HttpClient2, marker, False), raising = False)
    monkeypatch.setattr(tools, "_SEARCH_ENGINE_TIERS", (("duckduckgo",), ("wikipedia",)))
    monkeypatch.setattr(tools, "_wikipedia_search", lambda *args: [])
    state = SimpleNamespace(status = 202, body = '<form action="/anomaly.js"></form>')

    def request(self, *args, **kwargs):
        return SimpleNamespace(
            status_code = state.status, text = state.body, content = state.body.encode()
        )

    monkeypatch.setattr(HttpClient2, "request", request)
    monkeypatch.setattr(ENGINES["text"]["wikipedia"], "search", lambda *args, **kwargs: [])
    return state


def test_challenge_is_not_reported_as_a_genuine_empty_search(providers, monkeypatch):
    """A 202 challenge followed by an empty approved tier is an availability problem."""
    monkeypatch.setattr(tools, "_text_search_failed_at", None)
    result = tools._web_search("example query", timeout = 10)
    assert result.startswith("Search failed:")
    assert "blocked" in result.lower() or "rate limit" in result.lower()
    assert result != tools.EMPTY_SEARCH_RESULTS[0]
    assert tools._text_search_failed_at is not None


@pytest.mark.parametrize("status", [403, 429])
def test_provider_refusal_is_reported_when_other_tiers_are_empty(providers, status):
    """Rate limits and access refusals are not indistinguishable from empty search results."""
    providers.status, providers.body = status, ""
    assert tools._web_search("example query", timeout = 10).startswith("Search failed:")


@pytest.mark.parametrize(
    "status,body",
    [
        (200, "<html></html>"),
        (200, '<script src="/anomaly.js"></script>'),
        (202, "<html>Accepted</html>"),
    ],
)
def test_genuine_empty_or_unrelated_accepted_response_is_not_called_blocked(
    providers, status, body
):
    """An empty response or unrelated 202 does not prove a provider blocked the query."""
    providers.status, providers.body = status, body
    assert tools._web_search("empty query", timeout = 10) == tools.EMPTY_SEARCH_RESULTS[0]


def test_successful_provider_response_is_not_changed(providers):
    """A healthy provider's real parsed result reaches the caller unchanged."""
    providers.status = 200
    providers.body = (
        '<div class="body"><h2>Example result</h2>'
        '<a href="https://example.com/result">Example query snippet</a></div>'
    )
    result = tools._web_search("example query", timeout = 10)
    assert "Title: Example result" in result
    assert "URL: https://example.com/result" in result


def test_reused_client_restores_request_methods_after_a_challenge(providers):
    """A client's next search must not inherit the prior search's blocked state."""
    client = DDGS(timeout = 10)
    engine = client._get_engines("text", "duckduckgo")[0]
    assert "request" not in engine.http_client.__dict__

    with pytest.raises(tools._SearchProviderBlocked):
        tools._ddgs_text(client, "first query", 3, "duckduckgo")
    assert "_get_engines" not in client.__dict__
    assert "request" not in engine.http_client.__dict__

    providers.status, providers.body = 200, "<html></html>"
    with pytest.raises(Exception, match = "No results found") as error:
        tools._ddgs_text(client, "second query", 3, "duckduckgo")
    assert type(error.value).__name__ == "DDGSException"


def test_later_approved_engine_recovers_after_a_challenge(providers, monkeypatch):
    """A healthy second tier wins over the first tier's challenge."""
    monkeypatch.setattr(
        ENGINES["text"]["wikipedia"],
        "search",
        lambda *args, **kwargs: [
            TextResult(title = "Approved result", href = "https://example.com/result", body = "Found")
        ],
    )
    result = tools._web_search("approved query", timeout = 10)
    assert "Title: Approved result" in result
    assert not result.startswith("Search failed:")


def test_independent_wikipedia_fallback_recovers_after_a_challenge(providers, monkeypatch):
    """The existing fallback remains available and clearly identifies its limited scope."""
    monkeypatch.setattr(
        tools,
        "_wikipedia_search",
        lambda *args: [
            {"title": "Wiki", "href": "https://en.wikipedia.org/wiki/Example", "body": "Info"}
        ],
    )
    result = tools._web_search("example query", timeout = 10)
    assert "Wikipedia-only" in result
    assert "Title: Wiki" in result


def test_simultaneous_searches_do_not_share_provider_outcomes(providers, monkeypatch):
    """A challenge in one DDGS request cannot make another empty search look blocked."""

    def request(self, *args, **kwargs):
        query = kwargs["data"]["q"]
        body = '<form action="/anomaly.js"></form>' if query == "blocked query" else "<html></html>"
        return SimpleNamespace(
            status_code = 202 if query == "blocked query" else 200, text = body, content = body.encode()
        )

    monkeypatch.setattr(HttpClient2, "request", request)
    with ThreadPoolExecutor(max_workers = 2) as pool:
        blocked = pool.submit(tools._web_search, "blocked query", timeout = 10)
        empty = pool.submit(tools._web_search, "empty query", timeout = 10)
    assert blocked.result().startswith("Search failed:")
    assert empty.result() == tools.EMPTY_SEARCH_RESULTS[0]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch
import pytest
from core.inference import tools


def missing_ddgs():
    return patch.dict(sys.modules, {"ddgs": None})


@pytest.mark.parametrize("args", [{}, {"query": "  "}, {"query": 123}])
def test_missing_input_does_not_contact_fallback(args):
    with patch.object(tools, "_wikipedia_search") as wiki:
        assert tools.execute_tool("web_search", args) == "No query provided."
        wiki.assert_not_called()


def test_missing_package_recovers():
    with (
        missing_ddgs(),
        patch.object(
            tools,
            "_wikipedia_search",
            return_value=[
                {
                    "title": "Python",
                    "href": "https://en.wikipedia.org/wiki/Python",
                    "body": "snippet",
                }
            ],
        ) as wiki,
    ):
        result = tools._web_search("Python", timeout=8)
        assert "Wikipedia-only" in result and "URL: https://" in result
        wiki.assert_called_once()


@pytest.mark.parametrize(
    "policy", [{"blockedDomains": ["wikipedia.org"]}, {"allowedDomains": ["python.org"]}]
)
def test_policy_skips_wikipedia(policy):
    with missing_ddgs(), patch.object(tools, "_wikipedia_search") as wiki:
        assert tools._web_search("Python", website_policy=policy).startswith("Search failed:")
        wiki.assert_not_called()


def test_cancel_skips_network():
    event = threading.Event()
    event.set()
    with patch.object(tools, "_wikipedia_search") as wiki:
        assert tools._web_search("Python", cancel_event=event) == "Search cancelled."
        wiki.assert_not_called()


def test_cancel_during_fallback_discards_results():
    event = threading.Event()

    def wiki(*a):
        event.set()
        return [
            {"title": "Python", "href": "https://en.wikipedia.org/wiki/Python", "body": "snippet"}
        ]

    with missing_ddgs(), patch.object(tools, "_wikipedia_search", side_effect=wiki):
        assert tools._web_search("Python", cancel_event=event) == "Search cancelled."


def test_both_fail_preserves_primary_error():
    with (
        missing_ddgs(),
        patch.object(tools, "_wikipedia_search", side_effect=RuntimeError("offline")),
    ):
        assert "import of ddgs halted" in tools._web_search("Python")


def test_api_html_and_url():
    response = {
        "query": {
            "search": [
                {"title": "A & B", "snippet": 'An <span class="searchmatch">A</span> &amp; B'}
            ]
        }
    }
    with patch.object(
        tools, "_fetch_url_raw", return_value=(None, json.dumps(response), "application/json")
    ) as fetch:
        result = tools._wikipedia_search("A & B", 5, 3, None, None, None)
        assert result == [
            {"title": "A & B", "href": "https://en.wikipedia.org/wiki/A_%26_B", "body": "An A & B"}
        ]
        assert fetch.call_args.kwargs["raw_bytes_max"] == 1024 * 1024


PRIMARY_RESULT = [
    {"title": "Primary", "href": "https://primary.example/page", "body": "Current information"}
]
FALLBACK_RESULT = [
    {"title": "Wikipedia", "href": "https://en.wikipedia.org/wiki/Example", "body": "Background"}
]


@pytest.fixture
def provider(monkeypatch):
    """Control provider responses without making network calls or depending on ddgs's lazy proxy."""
    state = SimpleNamespace(calls=[], respond=lambda client, query, **kwargs: PRIMARY_RESULT)

    class DDGS:
        def __init__(self, timeout):
            self.timeout = timeout

        def text(self, query, **kwargs):
            state.calls.append((query, kwargs["backend"], self.timeout, kwargs["max_results"]))
            return state.respond(self, query, **kwargs)

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=DDGS))
    monkeypatch.setitem(
        sys.modules,
        "ddgs.engines",
        SimpleNamespace(ENGINES={"text": {"wikipedia": object(), "google": object()}}),
    )
    state.wiki = Mock(return_value=FALLBACK_RESULT)
    monkeypatch.setattr(tools, "_wikipedia_search", state.wiki)
    return state


@pytest.fixture
def search_clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(tools, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


@pytest.mark.parametrize("timeout", [0.6, 9, 300, None, 0])
def test_primary_success_keeps_full_provider_timeout(provider, search_clock, timeout):
    result = tools._web_search("current information", timeout=timeout)
    assert "Title: Primary" in result
    assert len(provider.calls) == 1
    query, backend, budget, wanted = provider.calls[0]
    assert (query, backend, wanted) == ("current information", "wikipedia", 5)
    assert budget == pytest.approx(timeout) if timeout is not None else budget is None
    provider.wiki.assert_not_called()


@pytest.mark.parametrize("blocked", [False, True])
def test_primary_late_success_is_not_replaced(provider, search_clock, blocked):
    def respond(client, query, **kwargs):
        # A valid answer in the last five seconds of the normal 300-second budget.
        if client.timeout < 298:
            raise TimeoutError("provider needed 298 seconds")
        search_clock.now += 298
        return PRIMARY_RESULT

    provider.respond = respond
    policy = {"blockedDomains": ["wikipedia.org"]} if blocked else None
    result = tools._web_search("current information", website_policy=policy)
    assert "Title: Primary" in result
    provider.wiki.assert_not_called()


def test_primary_second_tier_gets_original_remaining_budget(provider, search_clock):
    def respond(client, query, **kwargs):
        if kwargs["backend"] == "wikipedia":
            search_clock.now += 297
            return []
        if client.timeout < 3:
            raise TimeoutError("second tier needs the remaining three seconds")
        return PRIMARY_RESULT

    provider.respond = respond
    result = tools._web_search("current information")
    assert "Title: Primary" in result
    assert [call[2] for call in provider.calls] == [300, 3]
    provider.wiki.assert_not_called()


def test_primary_image_search_retains_client_budget(provider, monkeypatch, search_clock):
    budgets = []

    def images(client, *args):
        budgets.append(client.timeout)
        return "\nIMAGES_FOUND" if client.timeout >= 8 else ""

    monkeypatch.setattr(tools, "_web_search_images_suffix", images)
    result = tools._web_search("current information", timeout=9, include_images=True)
    assert "IMAGES_FOUND" in result
    assert budgets == [9]
    provider.wiki.assert_not_called()


def test_primary_eight_parallel_successes_all_use_provider(provider):
    lock = threading.Lock()
    entered = []
    all_entered = threading.Event()
    release = threading.Event()

    def respond(client, query, **kwargs):
        with lock:
            entered.append(query)
            if len(entered) == 8:
                all_entered.set()
        release.wait(3)
        return PRIMARY_RESULT

    provider.respond = respond
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(tools._web_search, f"query {i}", timeout=5) for i in range(8)]
        try:
            admitted = all_entered.wait(1)
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]
    assert admitted, f"Only {len(entered)} successful requests reached the provider"
    assert all("Title: Primary" in result for result in results)
    provider.wiki.assert_not_called()


def test_fallback_uses_only_remaining_budget(provider, search_clock):
    def respond(client, query, **kwargs):
        search_clock.now += 4
        return []

    provider.respond = respond
    assert "Wikipedia-only" in tools._web_search("query", timeout=9)
    assert provider.wiki.call_args.args[2:4] == (1, 109)


@pytest.mark.parametrize("timeout", [30, None])
def test_fallback_has_its_own_five_second_cap(provider, search_clock, timeout):
    provider.respond = lambda *args, **kwargs: []
    assert "Wikipedia-only" in tools._web_search("query", timeout=timeout)
    assert provider.wiki.call_args.args[2:4] == (5, 105)


def test_fallback_does_not_extend_exhausted_budget(provider, search_clock):
    def respond(client, query, **kwargs):
        search_clock.now += 10
        raise TimeoutError("primary timeout")

    provider.respond = respond
    assert "primary timeout" in tools._web_search("query", timeout=9)
    provider.wiki.assert_not_called()


def test_primary_overrun_success_keeps_legacy_behavior(provider, search_clock):
    def respond(client, query, **kwargs):
        search_clock.now += 10
        return PRIMARY_RESULT

    provider.respond = respond
    assert "Title: Primary" in tools._web_search("query", timeout=9)
    provider.wiki.assert_not_called()


@pytest.mark.parametrize("outcome", ["empty", "error", "missing_url"])
def test_fallback_recovers_when_all_tiers_are_unusable(provider, outcome):
    def respond(client, query, **kwargs):
        if outcome == "error":
            raise RuntimeError("provider failed")
        if outcome == "missing_url":
            return [{"title": "No URL", "body": "Unusable"}]
        return []

    provider.respond = respond
    assert "Wikipedia-only" in tools._web_search("query")
    assert len(provider.calls) == 2
    provider.wiki.assert_called_once()


def test_filtered_first_tier_continues_to_usable_second_tier(provider):
    def respond(client, query, **kwargs):
        if kwargs["backend"] == "wikipedia":
            return [{"title": "Blocked", "href": "https://blocked.example/page", "body": "Blocked"}]
        return PRIMARY_RESULT

    provider.respond = respond
    result = tools._web_search("query", website_policy={"blockedDomains": ["blocked.example"]})
    assert "Title: Primary" in result
    assert "blocked.example/page" not in result
    assert len(provider.calls) == 2
    provider.wiki.assert_not_called()


def test_cancel_during_primary_discards_answer_without_fallback(provider):
    event = threading.Event()

    def respond(client, query, **kwargs):
        event.set()
        return PRIMARY_RESULT

    provider.respond = respond
    assert tools._web_search("query", cancel_event=event) == "Search cancelled."
    provider.wiki.assert_not_called()


@pytest.mark.parametrize("timeout", [None, 0])
def test_primary_without_deadline_reuses_client_between_tiers(provider, timeout):
    def respond(client, query, **kwargs):
        if kwargs["backend"] == "wikipedia":
            client.first_tier_ran = True
            return []
        assert client.first_tier_ran
        return PRIMARY_RESULT

    provider.respond = respond
    assert "Title: Primary" in tools._web_search("query", timeout = timeout)
    provider.wiki.assert_not_called()


@pytest.mark.parametrize("import_fails", [False, True])
def test_setup_time_preserves_primary_budget_but_bounds_failed_import_fallback(
    provider, search_clock, monkeypatch, import_fails
):
    import builtins

    real_import = builtins.__import__

    def delayed_import(name, *args, **kwargs):
        if name == "ddgs":
            search_clock.now += 2
            if import_fails:
                raise ImportError("ddgs unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", delayed_import)
    result = tools._web_search("query", timeout = 3)
    if import_fails:
        assert "Wikipedia-only" in result
        assert provider.wiki.call_args.args[2:4] == (1, 103)
        assert not provider.calls
    else:
        assert "Title: Primary" in result
        assert provider.calls[0][2] == 3
        provider.wiki.assert_not_called()

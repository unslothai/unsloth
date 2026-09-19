# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Built-in text search must not send queries to DDGS's other search engines."""

import pytest

from ddgs.ddgs import DDGS
from ddgs.engines import ENGINES
from ddgs.exceptions import DDGSException
from ddgs.results import TextResult

from core.inference import tools


@pytest.fixture
def engine_calls(monkeypatch):
    calls = []

    def search(engine, query, **kwargs):
        calls.append((engine.name, query))
        if engine.name == "duckduckgo":
            if query == "empty":
                return []
            if query == "failure":
                raise DDGSException("DuckDuckGo unavailable")
        return [
            TextResult(
                title = "Search result",
                href = "https://example.com/source",
                body = "A synthetic search result.",
            )
        ]

    # Keep the real DDGS selector and fallback behavior; replace every engine's
    # network boundary, including engines that must never receive the query.
    for engine in ENGINES["text"].values():
        monkeypatch.setattr(engine, "search", search)
    # 9.14.4 has an optional shared cache; 9.8.0 has no network-cache client.
    if hasattr(DDGS, "_get_network_client"):
        monkeypatch.setattr(DDGS, "_get_network_client", lambda self: None)
    return calls


@pytest.mark.parametrize("query", ["results", "empty", "failure"])
def test_web_search_uses_only_duckduckgo_even_when_it_cannot_return_results(engine_calls, query):
    result = tools.execute_tool("web_search", {"query": query})

    assert engine_calls == [("duckduckgo", query)]
    if query == "results":
        assert "URL: https://example.com/source" in result
    elif query == "empty":
        assert result == "No results found."
    else:
        assert result == "Search failed: DuckDuckGo unavailable"


@pytest.mark.parametrize("unavailable", ["missing", "disabled"])
def test_unavailable_duckduckgo_does_not_fall_back_to_auto(monkeypatch, engine_calls, unavailable):
    if unavailable == "missing":
        monkeypatch.delitem(ENGINES["text"], "duckduckgo")
    else:
        monkeypatch.setattr(ENGINES["text"]["duckduckgo"], "disabled", True)

    result = tools.execute_tool("web_search", {"query": "results"})

    assert result == "Search failed: DuckDuckGo search is unavailable."
    assert engine_calls == []

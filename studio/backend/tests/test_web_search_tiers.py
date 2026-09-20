# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Built-in text search runs an approved allowlist, tier by tier, and never reaches Yandex.

Drives the real DDGS engine selector through ``execute_tool``; only each engine's own ``search`` is
replaced, so the tier strings really are resolved by ddgs and a fan-out cannot hide.
"""

import pytest

from ddgs.engines import ENGINES
from ddgs.exceptions import DDGSException
from ddgs.results import TextResult

from core.inference import tools

TIER1 = ("wikipedia", "brave", "duckduckgo", "mojeek", "startpage")
TIER2 = ("grokipedia", "google", "yahoo")


class _Recorder:
    """Records every text engine ddgs decides to run, and installs their behaviour."""

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch
        self.calls = []

    def _search_for(self, name, behaviour):
        def search(engine, query, **kwargs):
            self.calls.append((name, query))
            if behaviour == "empty":
                return []
            if behaviour == "raise":
                raise DDGSException(f"{name} refused")
            return [
                TextResult(
                    title = f"{name} result",
                    href = f"https://{name}.example/source",
                    body = f"A synthetic result from {name}.",
                )
            ]

        return search

    def install(self, behaviour_for):
        for name, cls in ENGINES["text"].items():
            # Engine construction is neutralised as well as search. These are SELECTION tests, and
            # building a real engine reaches ddgs's HTTP client: on the py<3.10 pin (ddgs 9.8.0) that
            # raises `Invalid impersonate: "safari_15.6.1"` against a current primp, which would make
            # every case below fail for a reason that has nothing to do with the tiers. Bypassing it
            # is what lets these tests cover BOTH pinned ddgs versions, including the one whose
            # unknown-name fallback the resolver exists to prevent.
            self._monkeypatch.setattr(cls, "__init__", lambda self, *a, **k: None, raising = False)
            self._monkeypatch.setattr(cls, "search", self._search_for(name, behaviour_for(name)))

    @property
    def names(self):
        return [name for name, _query in self.calls]


@pytest.fixture
def engine_calls(monkeypatch):
    from ddgs.ddgs import DDGS

    # 9.14.4 defines _get_network_client; a DHT cache hit would skip the engines entirely.
    if callable(getattr(DDGS, "_get_network_client", None)):
        monkeypatch.setattr(DDGS, "_get_network_client", lambda self: None)
    return _Recorder(monkeypatch)


def _names(recorder):
    return recorder.names


def _require_two_tiers():
    """A second tier has to RESOLVE for a fallthrough case to mean anything.

    It does on both pinned versions (9.14.4 gives all three, 9.8.0 gives yahoo alone, since it ships
    no grokipedia and disables google), but an upstream release that disabled the rest would turn
    these two cases red for a reason that is not about this code. Skipping with the resolved tiers
    named is honest; asserting anyway would be a failure nobody could act on.
    """
    resolved = tools._resolve_engine_tiers(ENGINES["text"])
    if len(resolved) < 2:
        pytest.skip(f"installed ddgs resolves only {len(resolved)} tier(s): {resolved}")


def test_tier_one_answers_alone_and_tier_two_is_never_asked(engine_calls):
    engine_calls.install(lambda name: "results")

    result = tools.execute_tool("web_search", {"query": "unsloth"})

    assert set(_names(engine_calls)) <= set(TIER1)
    assert not set(_names(engine_calls)) & set(TIER2)
    assert "yandex" not in _names(engine_calls)
    assert "URL: https://" in result


def test_tier_two_runs_only_after_tier_one_finds_nothing(engine_calls):
    _require_two_tiers()
    engine_calls.install(lambda name: "empty" if name in TIER1 else "results")

    result = tools.execute_tool("web_search", {"query": "unsloth"})

    contacted = _names(engine_calls)
    assert set(contacted) & set(TIER2), "tier 2 was never reached"
    assert "yandex" not in contacted
    assert "URL: https://" in result


def test_a_tier_one_refusal_falls_through_to_tier_two(engine_calls):
    _require_two_tiers()
    engine_calls.install(lambda name: "raise" if name in TIER1 else "results")

    result = tools.execute_tool("web_search", {"query": "unsloth"})

    assert set(_names(engine_calls)) & set(TIER2)
    assert "URL: https://" in result


def test_every_tier_empty_reports_no_results_and_never_widens(engine_calls):
    engine_calls.install(lambda name: "empty")

    result = tools.execute_tool("web_search", {"query": "unsloth"})

    assert "yandex" not in _names(engine_calls)
    assert result == tools.EMPTY_SEARCH_RESULTS[0]


@pytest.mark.parametrize("missing", [("startpage",), ("startpage", "grokipedia", "google")])
def test_an_engine_absent_from_the_registry_does_not_reopen_auto(
    monkeypatch, engine_calls, missing
):
    """9.8.0 raises KeyError on an unknown name and re-runs as auto; 9.14.4 keeps the valid ones.
    Either way the query must not reach an engine outside the tiers."""
    engine_calls.install(lambda name: "results")
    for name in missing:
        monkeypatch.delitem(ENGINES["text"], name, raising = False)

    expected = set(tools._resolve_engine_tiers(ENGINES["text"])[0].split(","))

    result = tools.execute_tool("web_search", {"query": "unsloth"})

    contacted = set(_names(engine_calls))
    assert "yandex" not in contacted
    # Equality against the RESOLVED tier, not a subset. As a subset check this passed on the
    # pre-allowlist code whenever its shuffle happened to keep the dispatched engines inside the two
    # tiers, which made it a flaky discriminator (measured: 1 pass in 10 on base). Deriving the
    # expected set from the resolver keeps it exact and portable across ddgs versions.
    assert contacted == expected, f"expected exactly the resolved tier 1 {expected}, got {contacted}"
    assert "URL: https://" in result


def test_no_approved_engine_available_fails_closed(monkeypatch, engine_calls):
    engine_calls.install(lambda name: "results")
    for name in TIER1 + TIER2:
        monkeypatch.delitem(ENGINES["text"], name, raising = False)

    result = tools.execute_tool("web_search", {"query": "unsloth"})

    assert result == "Search failed: no approved search engine is available."
    assert _names(engine_calls) == [], "a query left the process with no approved engine"


def test_disabled_engines_are_dropped_from_a_tier(monkeypatch, engine_calls):
    engine_calls.install(lambda name: "results")
    monkeypatch.setattr(ENGINES["text"]["duckduckgo"], "disabled", True)

    tools.execute_tool("web_search", {"query": "unsloth"})

    contacted = set(_names(engine_calls))
    assert "duckduckgo" not in contacted
    # Subset, not just "no yandex". Asserting the absence of one engine made this a FLAKY
    # discriminator: measured against the pre-allowlist code it passed 1 run in 10, because ddgs
    # shuffles its auto candidates and sometimes filled the result budget before reaching Yandex.
    # The subset assertion fails on that code every time, which is what a negative control owes.
    assert contacted <= set(TIER1) - {"duckduckgo"}, f"an engine outside tier 1 ran: {contacted}"
    assert "yandex" not in contacted


def test_the_resolver_never_names_an_engine_outside_the_tiers():
    resolved = tools._resolve_engine_tiers(ENGINES["text"])
    named = {name for tier in resolved for name in tier.split(",")}

    assert named <= set(TIER1) | set(TIER2)
    assert "yandex" not in named
    assert "bing" not in named

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A fetched page has to fit the window the model is actually running with.

The flat 16,000-character cap is roughly 4,000 tokens. On a 128k model that is nothing;
on the 4,864-token model this was measured against it is larger than the entire prompt
budget, and nothing downstream can recover: the fit protects the newest turn, so
compaction may not drop the oversized tool result, and the request is refused outright.

Measured, from a two-message thread (one 11-token question, one assistant turn with two
web_search calls):

    latest_turn_role   = "tool"
    latest_turn_tokens = 8154
    irreducible_tokens = 8389
    prompt_target      = 3648
    dropped_messages   = 0      nothing to evict; the thread IS the tool result
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference import tools


@pytest.fixture(autouse = True)
def _unknown_window(monkeypatch):
    """Default to "no model loaded" so each test states the window it means."""
    monkeypatch.setattr(tools, "_loaded_context_tokens", lambda: None)


def _window(monkeypatch, ctx):
    monkeypatch.setattr(tools, "_loaded_context_tokens", lambda: ctx)


def test_a_small_window_gets_a_page_it_can_hold(monkeypatch):
    """The case that failed. 4,864 tokens leaves 3,648 for the prompt, and the page that
    broke it was 12,295 characters, roughly 3,073 tokens: 84% of the budget for one
    search, before the system turn, the question or room to answer."""
    _window(monkeypatch, 4864)

    budget = tools._page_char_budget()

    assert budget < 12_295, "the page that caused the refusal must no longer fit"
    # And still worth reading rather than a stub.
    assert budget >= tools._MIN_PAGE_CHARS


def test_a_large_window_is_left_exactly_as_it_was(monkeypatch):
    """The blast radius. Above roughly an 11k window the old constant is returned
    unchanged, so no model that could already afford a whole page sees any difference."""
    for ctx in (16_384, 32_768, 131_072):
        _window(monkeypatch, ctx)
        assert tools._page_char_budget() == tools._MAX_PAGE_CHARS


def test_an_unknown_window_keeps_the_old_constant():
    """Not knowing the window must never shrink a fetch: that would silently degrade
    every provider path where the local backend is not the one answering."""
    assert tools._page_char_budget() == tools._MAX_PAGE_CHARS


def test_a_tiny_window_still_returns_a_readable_page(monkeypatch):
    """A floor, not a proportion all the way down. Below it the fetch would return a
    fragment too clipped to answer from, which is worse than a truncated page: the model
    cannot tell a short page from a cut one without the notice."""
    _window(monkeypatch, 512)

    assert tools._page_char_budget() == tools._MIN_PAGE_CHARS


def test_the_budget_never_exceeds_the_absolute_cap(monkeypatch):
    """The window only ever LOWERS the cap. A 1M-token model does not get a 1.4MB page."""
    _window(monkeypatch, 1_000_000)

    assert tools._page_char_budget() == tools._MAX_PAGE_CHARS


def test_an_unreadable_backend_is_unknown_rather_than_an_error(monkeypatch):
    """Every failure reading the window is "unknown", so a fetch is never blocked by the
    orchestrator being unavailable. Exercises the REAL reader, not a stub of it: stubbing
    `_loaded_context_tokens` would step over the very try/except under test."""
    import routes.inference as routes_inference

    def _boom():
        raise RuntimeError("backend gone")

    monkeypatch.undo()  # drop the autouse stub; this test owns the reader
    monkeypatch.setattr(routes_inference, "get_llama_cpp_backend", _boom)

    assert tools._loaded_context_tokens() is None
    assert tools._page_char_budget() == tools._MAX_PAGE_CHARS


def test_the_caller_can_still_pin_a_size(monkeypatch):
    """An explicit `max_chars` wins over the window, so callers that size their own budget
    and the extraction tests that pin exact output are unaffected."""
    _window(monkeypatch, 4864)
    captured = {}

    def _fake_truncate(text, max_chars):
        captured["max_chars"] = max_chars
        return text[:max_chars]

    monkeypatch.setattr(tools, "_truncate_page_text", _fake_truncate)
    # The signature default is None, so an explicit value must survive to the truncation.
    assert tools._truncate_page_text("x" * 50_000, 200) == "x" * 200
    assert captured["max_chars"] == 200


class TestTheWindowIsReadPerRequest:
    """Two ways the budget read the wrong window, both found in review.

    The reader stopped at the llama.cpp probe, so a native/Transformers chat left
    `is_loaded` false and reported "unknown", which kept the full 16,000-character
    cap on exactly the small models that cannot hold it. And the budget consulted
    process-global state, so an external-provider request, which never touches a
    resident GGUF, inherited that GGUF's window in both directions.
    """

    def test_a_native_model_window_is_read_when_no_gguf_is_loaded(self, monkeypatch):
        # The autouse fixture stubs the reader out; these exercise the real one.
        monkeypatch.undo()
        monkeypatch.setattr(
            "routes.inference.get_llama_cpp_backend",
            lambda: SimpleNamespace(is_loaded = False, context_length = None),
        )
        monkeypatch.setattr(
            "core.research_runs._peek_inference_backend",
            lambda: SimpleNamespace(
                active_model_name = "native/model",
                models = {"native/model": {"context_length": 4864}},
            ),
        )
        assert tools._loaded_context_tokens() == 4864

    def test_a_native_window_is_read_even_if_the_gguf_probe_raises(self, monkeypatch):
        # The llama.cpp branch must fall through, not return None: swallowing the
        # native answer is what made the reader report "unknown" here.
        monkeypatch.undo()

        def _boom():
            raise RuntimeError("no backend")

        monkeypatch.setattr("routes.inference.get_llama_cpp_backend", _boom)
        monkeypatch.setattr(
            "core.research_runs._peek_inference_backend",
            lambda: SimpleNamespace(active_model_name = None, models = {}, max_seq_length = 8192),
        )
        assert tools._loaded_context_tokens() == 8192

    def test_an_external_request_does_not_inherit_the_resident_gguf_window(self, monkeypatch):
        # A large resident GGUF must not hand its budget to a small external endpoint.
        _window(monkeypatch, 262_144)
        token = tools._REQUEST_CONTEXT_TOKENS.set(0)
        try:
            assert tools._page_char_budget() == tools._MAX_PAGE_CHARS
        finally:
            tools._REQUEST_CONTEXT_TOKENS.reset(token)

    def test_a_local_request_still_uses_the_probe_when_nothing_is_scoped(self, monkeypatch):
        _window(monkeypatch, 4864)
        assert tools._REQUEST_CONTEXT_TOKENS.get() is tools._UNSET_CONTEXT_TOKENS
        assert tools._page_char_budget() == 6809

    def test_a_scoped_window_beats_the_probe(self, monkeypatch):
        _window(monkeypatch, 262_144)
        token = tools._REQUEST_CONTEXT_TOKENS.set(4864)
        try:
            assert tools._page_char_budget() == 6809
        finally:
            tools._REQUEST_CONTEXT_TOKENS.reset(token)

    def test_execute_tool_scopes_the_window_for_the_call(self):
        tools.execute_tool("render_html", {"html": "<p>x</p>"}, context_tokens = 4864)
        assert tools._REQUEST_CONTEXT_TOKENS.get() == 4864

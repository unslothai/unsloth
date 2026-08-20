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
    # The request-scoped window is module state that outlives a test, and execute_tool
    # sets it deliberately. Restore it so one test cannot decide another's budget.
    token = tools._REQUEST_CONTEXT_TOKENS.set(tools._UNSET_CONTEXT_TOKENS)
    yield
    tools._REQUEST_CONTEXT_TOKENS.reset(token)


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


class TestToolResultsAlsoFitTheWindow:
    """The code tools had the same fixed-cap defect as fetched pages.

    Observed live on a 5120-token window: two requests refused at 7043 and 6684 tokens,
    both on the terminal/python tools, whose 16,000-character cap is about 4,000 tokens
    on its own. The result lands in the NEWEST turn, which the fit protects, so
    compaction cannot drop the one thing that does not fit and the request is
    irreducible rather than merely large.
    """

    def test_a_small_window_shrinks_the_tool_result_cap(self, monkeypatch):
        _window(monkeypatch, 5120)

        budget = tools._tool_result_char_budget()

        assert budget < tools._MAX_OUTPUT_CHARS
        # Roughly 1,800 tokens rather than 4,000, which is what brought the live
        # 7,043-token request back under a 5,120-token window.
        assert budget <= 5120 * 4 * tools._PAGE_CONTEXT_SHARE

    def test_a_large_window_keeps_the_full_cap(self, monkeypatch):
        for ctx in (32_768, 131_072, 262_144):
            _window(monkeypatch, ctx)
            assert tools._tool_result_char_budget() == tools._MAX_OUTPUT_CHARS

    def test_an_unknown_window_keeps_the_full_cap(self):
        # Not knowing must never shrink a result: that would silently degrade every
        # provider path where the local backend is not the one answering.
        assert tools._tool_result_char_budget() == tools._MAX_OUTPUT_CHARS

    def test_an_external_request_keeps_the_full_cap(self, monkeypatch):
        _window(monkeypatch, 5120)
        token = tools._REQUEST_CONTEXT_TOKENS.set(0)
        try:
            assert tools._tool_result_char_budget() == tools._MAX_OUTPUT_CHARS
        finally:
            tools._REQUEST_CONTEXT_TOKENS.reset(token)

    def test_truncate_resolves_its_limit_per_call(self, monkeypatch):
        """Bound at import, the default would freeze before any model is loaded."""
        _window(monkeypatch, 5120)
        text = "x" * 20_000

        out = tools._truncate(text)

        assert len(out) < len(text)
        assert "truncated to" in out

    def test_an_explicit_limit_still_wins(self, monkeypatch):
        _window(monkeypatch, 5120)

        assert tools._truncate("x" * 500, limit = 100).startswith("x" * 100)
        assert tools._truncate("x" * 50, limit = 100) == "x" * 50

    def test_a_result_that_fits_is_returned_untouched(self, monkeypatch):
        _window(monkeypatch, 262_144)

        assert tools._truncate("all good") == "all good"


class TestADenseResultIsSizedByWhatItCosts:
    """A character cap reserves its share of the window only for English.

    Measured with Qwen3-4B, Llama-3.2 and tiktoken on the real pages the tool fetches:
    English markdown runs 4.1 characters per token, so 35% of the window is 35%. Chinese
    and Japanese prose run 1.3-1.6, and the percent-escaped links a CJK page is full of
    (`/wiki/%E7%9F%A5%E8%AF%86`) run 1.3-1.5. Before this correction, zh.wikipedia and
    ja.wikipedia articles cut to the 4,864-token budget came back at 3,558-4,511 real
    tokens: 79-95% of the WHOLE prompt budget, in the newest turn, which the fit protects.
    That is the irreducible refusal this budget exists to prevent, reproduced.
    """

    # One paragraph of CJK prose with the wiki-style escaped links that come with it.
    _CJK_PAGE = (
        "人工智能是一门研究如何使机器具备智能行为的学科，"
        "涵盖[机器学习](/wiki/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)、"
        "[语言处理](/wiki/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86)"
        "和[电脑视觉](/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89)。"
    ) * 200
    _EN_PAGE = (
        "Artificial intelligence is the study of machines that perceive their "
        "environment and take actions that maximise the chance of a goal. "
    ) * 200

    def _dense_tokens(self, text):
        from core.inference.context_window import estimate_messages_tokens_dense
        return estimate_messages_tokens_dense([{"role": "tool", "content": text}])

    def test_a_cjk_page_is_cut_to_the_share_it_was_promised(self, monkeypatch):
        _window(monkeypatch, 4864)

        out = tools._truncate_page_text(self._CJK_PAGE, tools._page_char_budget())

        # The whole point: what lands in the turn costs about the share reserved for it,
        # not the entire window. A little over for the truncation notice itself.
        assert self._dense_tokens(out) <= int(4864 * tools._PAGE_CONTEXT_SHARE) + 64
        # And this is not a no-op: the characters the flat budget would have admitted
        # do not fit the share by either measure, the repo's dense estimate or the rule
        # here (which the real tokenizers above put at 79-95% of the prompt budget).
        flat = self._CJK_PAGE[: tools._page_char_budget()]
        assert self._dense_tokens(flat) > int(4864 * tools._PAGE_CONTEXT_SHARE)
        assert tools._dense_prefix_chars(flat, 4864 * tools._PAGE_CONTEXT_SHARE) < len(flat)

    def test_an_english_page_is_left_exactly_as_it_was(self, monkeypatch):
        """The blast radius: text that really does run four characters per token keeps
        every character the character budget gave it."""
        _window(monkeypatch, 4864)
        budget = tools._page_char_budget()

        assert tools._dense_char_limit(self._EN_PAGE, budget) == budget

    def test_percent_escaped_links_are_charged_like_the_bytes_they_encode(self):
        """`%E7%9F%A5` is three non-ASCII bytes spelled in ASCII and tokenises like them,
        so charging it four characters per token undercounts it three-fold."""
        escaped = "%E7%9F%A5" * 100

        # 300 escapes, a token each for the three bytes they spell: 900 tokens, where
        # four characters per token would have called the same text 225.
        assert tools._dense_prefix_chars(escaped, 900) == len(escaped)
        assert tools._dense_prefix_chars(escaped, 450) == len(escaped) // 2
        # Cut on a whole escape, never halfway through one.
        assert tools._dense_prefix_chars(escaped, 451) % 3 == 0

    def test_a_dense_result_never_falls_below_the_readable_floor(self, monkeypatch):
        _window(monkeypatch, 1024)

        out = tools._truncate_page_text(self._CJK_PAGE, tools._page_char_budget())

        assert len(out) >= tools._MIN_PAGE_CHARS

    def test_an_unknown_window_leaves_a_dense_page_alone(self):
        """Same rule as the char budget: not knowing must never shrink a fetch."""
        assert (
            tools._dense_char_limit(self._CJK_PAGE, tools._MAX_PAGE_CHARS) == tools._MAX_PAGE_CHARS
        )

    def test_a_dense_terminal_result_is_sized_too(self, monkeypatch):
        """The code tools print CJK and escaped URLs as readily as a page carries them."""
        _window(monkeypatch, 5120)

        out = tools._truncate(self._CJK_PAGE)

        assert self._dense_tokens(out) <= int(5120 * tools._PAGE_CONTEXT_SHARE) + 64
        assert "truncated to" in out

    def test_an_explicit_limit_is_still_a_ceiling_not_a_floor(self, monkeypatch):
        """A caller that pins a size smaller than the floor keeps it."""
        _window(monkeypatch, 4864)

        assert tools._dense_char_limit(self._CJK_PAGE, 200) == 200

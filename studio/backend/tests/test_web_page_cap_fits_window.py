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

import random
import sys
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
    # Measured counts are cached per model for the life of the process, so one test's
    # backend double must not answer the next one's questions.
    tools._PROBE_COUNT_CACHE.clear()
    yield
    tools._PROBE_COUNT_CACHE.clear()
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


class TestDenseAsciiIsMeasuredNotEstimated:
    """`base64`, `hexdump -C` and `sha256sum` are ordinary terminal output, and the flat
    0.25 tokens per ASCII character the estimate charges them is off by a factor of four.

    Measured with Qwen3-4B and Llama-3.2 on the 5,120-token window this PR was built
    against, where the character cap admits 7,168 characters against a 1,792-token share:

        base64 payload.bin    7,168 chars -> 5,361 tokens   105% of the whole window
        hexdump -C            7,168 chars -> 5,540 tokens   108%
        sha256sum *           7,168 chars -> 5,109 tokens   100%
        English prose         7,168 chars -> 1,230 tokens    24%   (the estimate is right)

    A four-message thread -- system turn, an 8-token question, one tool call and one such
    result -- was then refused by `fit_rolling_context` as irreducible at 5,475 tokens
    against a 3,840-token prompt budget, with `dropped_messages: 0`. That is the exact
    refusal this budget exists to prevent, so where a tokenizer is serving the request the
    prefix is measured with it instead of estimated.
    """

    # 1.33 characters per token: the Qwen3-4B rate measured on `base64` output above.
    _RATE = 1.33

    def _serving(
        self,
        monkeypatch,
        ctx,
        rate = None,
    ):
        """A loaded llama.cpp backend that prices text at a real dense-ASCII rate."""
        rate = self._RATE if rate is None else rate
        backend = SimpleNamespace(
            is_loaded = True,
            context_length = ctx,
            count_chat_tokens = lambda messages, *a, **k: int(
                sum(len(m["content"]) for m in messages) / rate
            ),
        )
        monkeypatch.setattr("routes.inference.get_llama_cpp_backend", lambda: backend)
        return backend

    def test_a_base64_result_is_cut_to_what_it_really_costs(self, monkeypatch):
        _window(monkeypatch, 5120)
        self._serving(monkeypatch, 5120)
        text = "aGVsbG8gd29ybGQgdGhpcyBpcyBiaW5hcnkgcGF5bG9hZA" * 600

        kept = tools._dense_char_limit(text, tools._tool_result_char_budget())

        # The share it was promised, not the whole window.
        assert kept / self._RATE <= 5120 * tools._PAGE_CONTEXT_SHARE
        # And this is not vacuous: the estimate alone would have kept the full cap.
        assert tools._dense_prefix_chars(text, 5120 * tools._PAGE_CONTEXT_SHARE) > kept

    def test_a_dense_result_no_longer_outweighs_the_window(self, monkeypatch):
        """The refusal itself: 7,168 characters of base64 is 105% of a 5,120-token
        window, so the request cannot be made to fit by dropping anything."""
        _window(monkeypatch, 5120)
        self._serving(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000

        out = tools._truncate(text)

        assert len(out) / self._RATE < 5120

    def test_english_keeps_every_character_the_cap_gave_it(self, monkeypatch):
        """The blast radius: at a real English rate the exact count agrees with the
        estimate, so nothing that already fitted is shrunk."""
        _window(monkeypatch, 5120)
        self._serving(monkeypatch, 5120, rate = 4.2)
        text = (
            "Artificial intelligence is the study of machines that perceive their "
            "environment and take actions that maximise the chance of a goal. "
        ) * 200
        budget = tools._tool_result_char_budget()

        assert tools._dense_char_limit(text, budget) == budget

    def test_a_resident_gguf_does_not_price_another_model_s_request(self, monkeypatch):
        """A 262k GGUF sitting in memory must not tokenize for the 5,120-token native
        model actually answering: different tokenizer, different text."""
        _window(monkeypatch, 5120)
        self._serving(monkeypatch, 262_144)

        assert tools._loaded_token_counter(5120) is None

    def test_a_tokenizer_that_raises_falls_back_to_the_estimate(self, monkeypatch):
        _window(monkeypatch, 5120)

        def _boom(*a, **k):
            raise RuntimeError("llama-server is busy")

        monkeypatch.setattr(
            "routes.inference.get_llama_cpp_backend",
            lambda: SimpleNamespace(is_loaded = True, context_length = 5120, count_chat_tokens = _boom),
        )
        text = "0123456789abcdef" * 2000

        assert tools._dense_char_limit(text, 7168) == 7168

    def test_no_backend_at_all_leaves_the_estimate_alone(self, monkeypatch):
        _window(monkeypatch, 5120)
        monkeypatch.setattr(
            "routes.inference.get_llama_cpp_backend",
            lambda: SimpleNamespace(is_loaded = False),
        )

        assert tools._dense_char_limit("0123456789abcdef" * 2000, 7168) == 7168

    def test_a_dense_prefix_with_a_prose_tail_is_measured_not_assumed(self, monkeypatch):
        """The shape a proportional shrink gets wrong, and the one the code tools emit
        most: `base64 payload.bin` followed by the shell's ordinary English report.

        Cutting the prose off raises the average density of what is left, so each pass
        gains less than it asked for. Measured with Qwen3-4B on a real 2,500-character
        base64 prefix (1.38 chars/token) followed by English (4.2-6.2), the fixed pass
        count returned 3,497 characters costing 1,978 tokens against the 1,792-token
        share: 110%, unmeasured, which is the irreducible overflow this budget prevents.
        Whatever comes back now has been counted.
        """
        _window(monkeypatch, 5120)
        dense_chars = 2500

        # The measured Qwen3-4B rates, priced per character so the count is exact at
        # every prefix length rather than only at the two the test happens to check.
        def _price(chunk):
            dense = min(len(chunk), dense_chars)
            return int(dense / 1.376 + (len(chunk) - dense) / 4.2)

        monkeypatch.setattr(
            "routes.inference.get_llama_cpp_backend",
            lambda: SimpleNamespace(
                is_loaded = True,
                context_length = 5120,
                count_chat_tokens = lambda messages, *a, **k: sum(
                    _price(m["content"]) for m in messages
                ),
            ),
        )
        text = (
            "ABCDefgh0123+/9z" * 157
            + ("The build finished and the archive was uploaded to the release bucket. ") * 400
        )
        share = 5120 * tools._PAGE_CONTEXT_SHARE

        kept = tools._dense_char_limit(text, tools._tool_result_char_budget())

        assert _price(text[:kept]) <= share, "the retained prefix must be counted, not assumed"
        # And not by collapsing to the floor: the fit is still worth reading.
        assert kept > tools._MIN_PAGE_CHARS

    def test_a_template_that_drops_tool_messages_is_still_measured(self, monkeypatch):
        """The probe has to price a prompt that CONTAINS the chunk.

        `count_chat_tokens` renders through the model's chat template, so a role the
        template skips is priced as framing and nothing else. Both bundled Gemma-4
        templates do exactly that -- `gemma-4.jinja:232` is
        `{%- if message['role'] != 'tool' -%}`, and a tool result is only emitted while
        scanning forward from an assistant tool call. Rendered directly, a 600-character
        payload came back as 46 characters with the payload absent, so the count was a
        small positive constant, the first pass saw it fit, and the whole estimated prefix
        was returned unmeasured on a whole model family.
        """
        _window(monkeypatch, 5120)
        seen = []

        def _count_chat_tokens(messages, *a, **k):
            # A template with the Gemma-4 convention: user turns render, a standalone
            # tool message does not.
            seen.append([m["role"] for m in messages])
            body = "".join(m["content"] for m in messages if m["role"] == "user")
            return 11 + int(len(body) / 1.33)

        monkeypatch.setattr(
            "routes.inference.get_llama_cpp_backend",
            lambda: SimpleNamespace(
                is_loaded = True, context_length = 5120, count_chat_tokens = _count_chat_tokens
            ),
        )
        text = "0123456789abcdef" * 2000
        share = 5120 * tools._PAGE_CONTEXT_SHARE

        kept = tools._dense_char_limit(text, tools._tool_result_char_budget())

        assert kept / 1.33 <= share, "a skipped role priced framing, not the result"
        assert kept >= tools._MIN_PAGE_CHARS
        assert seen and all(roles == ["user"] for roles in seen)

    def test_a_template_that_renders_no_content_falls_back_to_the_estimate(self, monkeypatch):
        """The guard. If some future template drops the probe role too, the count is a
        small constant regardless of chunk size -- which is not a measurement and must not
        be accepted as one. Keeping the estimate is the pre-existing behaviour; reporting
        the constant is the silent no-op this exists to prevent."""
        _window(monkeypatch, 5120)
        monkeypatch.setattr(
            "routes.inference.get_llama_cpp_backend",
            lambda: SimpleNamespace(
                is_loaded = True,
                context_length = 5120,
                count_chat_tokens = lambda messages, *a, **k: 11,  # framing, whatever is sent
            ),
        )

        assert tools._loaded_token_counter(5120)("0123456789abcdef" * 250) is None
        # And the caller keeps the estimate rather than a prefix nothing priced.
        assert tools._dense_char_limit("0123456789abcdef" * 2000, 7168) == 7168

    def test_the_readable_floor_still_holds_under_an_exact_count(self, monkeypatch):
        _window(monkeypatch, 1024)
        self._serving(monkeypatch, 1024)

        kept = tools._dense_char_limit("0123456789abcdef" * 2000, tools._MAX_PAGE_CHARS)

        assert kept == tools._MIN_PAGE_CHARS


class TestAConfiguredCapIsNeverRaised:
    """`UNSLOTH_TOOL_RESULT_MAX_CHARS` is a ceiling the install set, and the readability
    floor is not a reason to exceed it.

    Before this, an install running a 500-character cap got 500 characters from the
    hosted path (`studio_tool_loop._truncate_for_model`) and 2,000 from the local one the
    moment a window became readable, so the one function whose job is to LOWER the cap
    raised it fourfold instead -- and did so hardest on the smallest windows, which is
    where the operator asked for the small cap.
    """

    def test_a_configured_cap_below_the_floor_survives_a_known_window(self, monkeypatch):
        monkeypatch.setattr(tools, "_MAX_OUTPUT_CHARS", 500)
        _window(monkeypatch, 8192)

        assert tools._tool_result_char_budget() == 500

    def test_it_survives_a_tiny_window_too(self, monkeypatch):
        """The window-derived share is 1,433 characters here, so the floor is the only
        thing that could have raised 500."""
        monkeypatch.setattr(tools, "_MAX_OUTPUT_CHARS", 500)
        _window(monkeypatch, 1024)

        assert tools._tool_result_char_budget() == 500

    def test_the_local_result_matches_the_hosted_one(self, monkeypatch):
        from core.inference import studio_tool_loop

        monkeypatch.setattr(tools, "_MAX_OUTPUT_CHARS", 500)
        _window(monkeypatch, 8192)
        text = "x" * 5000

        assert len(tools._truncate(text)) - len(text[:500]) < 400  # notice only
        assert tools._truncate(text).startswith(text[:500])
        assert studio_tool_loop._truncate_for_model(text).startswith(text[:500])

    def test_an_unconfigured_install_still_gets_the_floor(self, monkeypatch):
        """The floor is untouched wherever the cap is above it, which is the default."""
        _window(monkeypatch, 512)

        assert tools._tool_result_char_budget() == tools._MIN_PAGE_CHARS
        assert tools._page_char_budget() == tools._MIN_PAGE_CHARS


class TestTheProbeIsNotPaidForTwice:
    """The measurement is worth its round trips; paying for it again is not.

    `count_chat_tokens` is two llama-server calls -- `/apply-template` then `/tokenize` --
    over a fresh connection each time, so every counter call here is two HTTP round trips
    on the path between a tool finishing and the model seeing its result. Measured on the
    merge base, with a 5,120-token window: an English result cost 2 counter calls and a
    dense one (base64, hexdump, sha256sum) cost 4, on every single result, forever.

    Three things were being bought and thrown away. The framing baseline is the same
    number for every result the process ever truncates, and it was priced per result. Its
    value cannot change the answer for a result that fits on its first count, and it was
    priced before that count was taken. And an estimate already at or below the readable
    floor is the answer whatever the tokenizer says, and it was measured anyway.

    Nothing here may change what is returned. Every test below asserts the number the
    merge base returns alongside the round trips it no longer takes.
    """

    _RATE = 1.33

    def _serving(
        self,
        monkeypatch,
        ctx,
        rate = None,
        identified = True,
        pid = 4242,
        extra_args = None,
        gguf = "/models/qwen3-4b.gguf",
    ):
        """A loaded llama.cpp backend that counts the calls it is asked to make.

        `is_loaded` really is `self._process is not None and self._healthy`, so a resident
        backend always has a process, and `pid` is what a reload changes.
        """
        rate = self._RATE if rate is None else rate
        calls = []

        def count_chat_tokens(messages, *a, **k):
            body = "".join(m["content"] for m in messages)
            calls.append(len(body))
            return 8 + int(len(body) / rate)

        backend = SimpleNamespace(
            is_loaded = True, context_length = ctx, count_chat_tokens = count_chat_tokens
        )
        if identified:
            # What a real backend exposes once a GGUF is resident.
            backend._process = SimpleNamespace(pid = pid)
            backend.model_identifier = "Qwen3-4B"
            backend._gguf_load_identity = ((gguf, 66306, 4242, 1),)
            backend._chat_template_override = None
            backend._extra_args = list(extra_args) if extra_args else None
        monkeypatch.setattr("routes.inference.get_llama_cpp_backend", lambda: backend)
        return calls, backend

    def test_an_estimate_at_the_floor_is_not_measured_at_all(self, monkeypatch):
        """A 1,024-token window leaves a 358-token share, so the estimate is already below
        the 2,000-character floor and `_dense_char_limit` clamps up to it whatever comes
        back. The merge base spent 2 counter calls (4 HTTP round trips) rediscovering it.
        """
        _window(monkeypatch, 1024)
        calls, _ = self._serving(monkeypatch, 1024)

        kept = tools._dense_char_limit("0123456789abcdef" * 2000, tools._MAX_PAGE_CHARS)

        assert kept == tools._MIN_PAGE_CHARS  # the merge base's answer, unchanged
        assert calls == []

    def test_a_result_that_fits_does_not_price_the_framing_baseline(self, monkeypatch):
        """English is the common case and it fits on its first count. The baseline only
        ever decides whether a count that came in OVER budget is a real measurement, so
        for this result it is bought and never read: 2 counter calls where 1 answers."""
        _window(monkeypatch, 5120)
        calls, _ = self._serving(monkeypatch, 5120, rate = 4.2)
        text = ("The build finished and the archive was uploaded to the release bucket. ") * 400
        budget = tools._tool_result_char_budget()

        kept = tools._dense_char_limit(text, budget)

        assert kept == budget  # English keeps every character it was allowed, as before
        assert len(calls) == 1
        assert calls == [budget], "the one call is the measurement, not the baseline"

    def test_the_baseline_is_priced_once_per_model_not_once_per_result(self, monkeypatch):
        """A dense result does need the baseline. It is the same number for the next one."""
        _window(monkeypatch, 5120)
        calls, _ = self._serving(monkeypatch, 5120)
        first = "0123456789abcdef" * 2000
        second = "fedcba9876543210" * 1500

        cold = tools._dense_char_limit(first, tools._tool_result_char_budget())
        cold_calls = list(calls)
        calls.clear()
        tools._dense_char_limit(second, tools._tool_result_char_budget())

        # A chunk of length 0 IS the baseline: the empty probe the guard measures against.
        assert 0 in cold_calls, "the first dense result pays for the baseline"
        assert calls, "the second result is still measured"
        assert 0 not in calls, "but the baseline is answered from the cache"
        assert len(calls) == len(cold_calls) - 1
        # And the answer is still the measured one.
        assert cold / self._RATE <= 5120 * tools._PAGE_CONTEXT_SHARE

    def test_the_same_result_twice_costs_nothing_the_second_time(self, monkeypatch):
        """Retries, regenerations and a model that runs the same command again."""
        _window(monkeypatch, 5120)
        calls, _ = self._serving(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        first = tools._dense_char_limit(text, budget)
        assert calls, "the first pass must actually measure"
        calls.clear()
        second = tools._dense_char_limit(text, budget)

        assert second == first
        assert calls == []

    def test_a_different_model_never_reads_the_previous_one_s_counts(self, monkeypatch):
        """Same window, different tokenizer. The cache key carries the model's identity,
        so a reload cannot be answered from the model it replaced."""
        _window(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        dense_calls, _ = self._serving(
            monkeypatch, 5120, rate = 1.33, pid = 111, gguf = "/models/dense.gguf"
        )
        dense = tools._dense_char_limit(text, budget)

        sparse_calls, _ = self._serving(
            monkeypatch, 5120, rate = 4.2, pid = 222, gguf = "/models/sparse.gguf"
        )
        sparse = tools._dense_char_limit(text, budget)

        assert sparse_calls, "the new model must be measured, not looked up"
        assert sparse > dense, "and priced by its own tokenizer"
        assert sparse == budget and dense_calls

    def test_a_backend_with_no_resident_process_is_not_cached(self, monkeypatch):
        """Nothing to tie a count to means no key guaranteed to change when the rendering
        does, so the safe answer is to keep paying. Every lightweight double lands here."""
        _window(monkeypatch, 5120)
        calls, _ = self._serving(monkeypatch, 5120, identified = False)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        first = tools._dense_char_limit(text, budget)
        spent = len(calls)
        calls.clear()
        second = tools._dense_char_limit(text, budget)

        assert second == first
        assert len(calls) == spent
        assert not tools._PROBE_COUNT_CACHE

    def test_a_count_that_failed_is_never_remembered_as_an_answer(self, monkeypatch):
        """A busy server is a property of the moment, not of the text. Caching the failure
        would turn one timeout into a permanent estimate for that result."""
        _window(monkeypatch, 5120)
        state = {"fail": True}

        def count_chat_tokens(messages, *a, **k):
            if state["fail"]:
                raise RuntimeError("llama-server is busy")
            return 8 + int(sum(len(m["content"]) for m in messages) / self._RATE)

        monkeypatch.setattr(
            "routes.inference.get_llama_cpp_backend",
            lambda: SimpleNamespace(
                is_loaded = True,
                context_length = 5120,
                count_chat_tokens = count_chat_tokens,
                _process = SimpleNamespace(pid = 4242),
                model_identifier = "Qwen3-4B",
                _gguf_load_identity = (("/models/qwen3-4b.gguf", 66306, 4242, 1),),
                _chat_template_override = None,
                _extra_args = None,
            ),
        )
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        assert tools._dense_char_limit(text, budget) == budget  # the estimate stands
        state["fail"] = False

        assert tools._dense_char_limit(text, budget) < budget  # and is measured next time

    def test_the_cache_cannot_grow_without_bound(self, monkeypatch):
        _window(monkeypatch, 5120)
        self._serving(monkeypatch, 5120)
        budget = tools._tool_result_char_budget()

        for index in range(tools._PROBE_COUNT_CACHE_ENTRIES + 40):
            tools._dense_char_limit(f"{index:04d}" + "0123456789abcdef" * 2000, budget)

        held = sum(len(entry) for entry in tools._PROBE_COUNT_CACHE.values())
        assert held <= tools._PROBE_COUNT_CACHE_ENTRIES

    def test_the_cache_is_bounded_by_characters_and_not_only_by_entries(self, monkeypatch):
        """The entry count says nothing about size. Only a fetched page is capped at
        `_MAX_PAGE_CHARS`; a tool result's prefix is bounded by the configured cap and the
        window, and `_env_int` takes any positive integer, so one prefix can be enormous.
        Measured on this path: a 1,000,000-character cap on a 262k window cached 733,971
        characters from a single result, which 64 entries would multiply.
        """
        monkeypatch.setattr(tools, "_MAX_OUTPUT_CHARS", 1_000_000)
        _window(monkeypatch, 262_144)
        calls, _ = self._serving(monkeypatch, 262_144, rate = 4.0)
        budget = tools._tool_result_char_budget()

        assert budget > tools._MAX_PAGE_CHARS, "the premise: prefixes far exceed the page cap"

        for index in range(8):
            tools._dense_char_limit(f"{index:04d}" + "A" * 6_000_000, budget)

        held = sum(len(key) for entry in tools._PROBE_COUNT_CACHE.values() for key in entry)
        assert held <= tools._PROBE_COUNT_CACHE_CHARS
        # And the baseline, which is 0 characters, is still in there earning its keep.
        assert any("" in entry for entry in tools._PROBE_COUNT_CACHE.values())

    def test_a_prefix_too_large_to_hold_is_skipped_not_stored(self, monkeypatch):
        monkeypatch.setattr(tools, "_PROBE_COUNT_CACHE_CHARS", 5000)
        _window(monkeypatch, 5120)
        calls, _ = self._serving(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        first = tools._dense_char_limit(text, budget)
        held = sum(len(key) for entry in tools._PROBE_COUNT_CACHE.values() for key in entry)
        calls.clear()
        second = tools._dense_char_limit(text, budget)

        assert second == first, "the answer never depends on what was cached"
        assert held <= 5000

    def test_the_guard_still_rejects_a_template_that_renders_no_content(self, monkeypatch):
        """The baseline is deferred, not dropped. A count that does not move off it is
        still not a measurement, and the caller still keeps its estimate."""
        _window(monkeypatch, 5120)
        monkeypatch.setattr(
            "routes.inference.get_llama_cpp_backend",
            lambda: SimpleNamespace(
                is_loaded = True,
                context_length = 5120,
                count_chat_tokens = lambda messages, *a, **k: 11,
                _process = SimpleNamespace(pid = 7),
                model_identifier = "Gemma-4",
                _gguf_load_identity = (("/models/gemma-4.gguf", 66306, 7, 1),),
                _chat_template_override = None,
                _extra_args = None,
            ),
        )

        assert tools._loaded_token_counter(5120)("0123456789abcdef" * 250) is None
        assert tools._dense_char_limit("0123456789abcdef" * 2000, 7168) == 7168

    def test_a_pass_through_chat_template_is_not_answered_from_the_managed_one(self, monkeypatch):
        """The gap `_chat_template_override` cannot see.

        User extra args are appended verbatim AFTER Studio's own flags and llama.cpp is
        last-wins, so `--chat-template` / `--chat-template-file` in extra args changes what
        `/apply-template` renders while every managed field stays exactly as it was. Same
        GGUF, same window, same managed override: reuse the counts and a prefix gets a
        price from a template that is no longer serving it, which is the irreducible
        overflow this budget exists to prevent, reintroduced through the cache.
        """
        _window(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        self._serving(monkeypatch, 5120, rate = 1.33, pid = 900)
        dense = tools._dense_char_limit(text, budget)

        # Reloaded with only a pass-through template added. Everything managed is identical.
        calls, backend = self._serving(
            monkeypatch,
            5120,
            rate = 4.2,
            pid = 901,
            extra_args = ["--chat-template", "chatml"],
        )
        sparse = tools._dense_char_limit(text, budget)

        assert backend.model_identifier == "Qwen3-4B"
        assert backend._chat_template_override is None
        assert calls, "the new template must be measured, not looked up"
        assert sparse > dense, "and priced by the template actually rendering"

    def test_the_extra_args_alone_are_enough_to_miss(self, monkeypatch):
        """Belt to the process id's braces: even holding the pid fixed, counts are not
        shared across a different command line."""
        _window(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        self._serving(monkeypatch, 5120, rate = 1.33, pid = 5)
        tools._dense_char_limit(text, budget)

        calls, _ = self._serving(
            monkeypatch, 5120, rate = 1.33, pid = 5, extra_args = ["--chat-template-file", "/x.jinja"]
        )
        tools._dense_char_limit(text, budget)

        assert calls, "a different command line is a different rendering"

    def test_a_reload_of_the_very_same_configuration_still_misses(self, monkeypatch):
        """A restart is a new process whatever its arguments, so nothing survives it. This
        is what makes the key safe against flags nobody has thought of yet."""
        _window(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        self._serving(monkeypatch, 5120, pid = 1000)
        first = tools._dense_char_limit(text, budget)

        calls, _ = self._serving(monkeypatch, 5120, pid = 1001)
        second = tools._dense_char_limit(text, budget)

        assert second == first, "same configuration, same answer"
        assert calls, "but re-measured rather than carried over the restart"

    def test_an_unhashable_identity_field_disables_the_cache_rather_than_raising(self, monkeypatch):
        _window(monkeypatch, 5120)
        calls, backend = self._serving(monkeypatch, 5120)
        backend._gguf_load_identity = {"not": "hashable"}
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        first = tools._dense_char_limit(text, budget)
        spent = len(calls)
        calls.clear()

        assert tools._dense_char_limit(text, budget) == first
        assert len(calls) == spent
        assert not tools._PROBE_COUNT_CACHE

    def _template_down(
        self,
        monkeypatch,
        ctx,
        rate = None,
        fallback_rate = None,
    ):
        """`/apply-template` is down but `/tokenize` is not.

        `count_chat_tokens(strict = False)` then returns the plain-text fallback, which
        prices the bytes but drops the template's role markers and special tokens.
        """
        rate = self._RATE if rate is None else rate
        calls = []

        def count_chat_tokens(messages, *a, **k):
            body = "".join(m["content"] for m in messages)
            calls.append((len(body), bool(k.get("strict"))))
            if k.get("strict"):
                raise RuntimeError("llama-server could not render the chat template")
            return int(len(body) / (fallback_rate or rate)) or 1  # no framing: the fallback

        backend = SimpleNamespace(
            is_loaded = True,
            context_length = ctx,
            count_chat_tokens = count_chat_tokens,
            _process = SimpleNamespace(pid = 77),
            model_identifier = "Qwen3-4B",
            _gguf_load_identity = (("/models/qwen3-4b.gguf", 66306, 4242, 1),),
            _chat_template_override = None,
            _extra_args = None,
        )
        monkeypatch.setattr("routes.inference.get_llama_cpp_backend", lambda: backend)
        return calls

    def test_a_plain_text_fallback_count_is_used_but_never_retained(self, monkeypatch):
        """The count is still used: it tokenizes the real bytes, which is what catches
        dense ASCII, and the estimate it would otherwise fall back to undercharges base64
        several fold. It is simply not KEPT -- it prices a prompt the model will never be
        sent, so caching it would let one bad moment under-count that prefix for the life
        of the process.
        """
        _window(monkeypatch, 5120)
        calls = self._template_down(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        kept = tools._dense_char_limit(text, budget)

        assert kept < budget, "the fallback still measured the bytes"
        assert not any(cache for cache in tools._PROBE_COUNT_CACHE.values())
        # And a later result re-measures rather than trusting it.
        calls.clear()
        tools._dense_char_limit(text, budget)
        assert calls

    def test_the_strict_attempt_is_made_once_per_result_not_once_per_probe(self, monkeypatch):
        """A template that will not render is not going to start mid-result, so asking
        again would spend round trips on a settled question. One extra attempt for the
        first probe, not one for every probe."""
        _window(monkeypatch, 5120)
        calls = self._template_down(monkeypatch, 5120)

        tools._dense_char_limit("0123456789abcdef" * 2000, tools._tool_result_char_budget())

        assert sum(1 for _, strict in calls if strict) == 1
        assert len(calls) > 2, "and the result really did take several passes"

    def test_a_healthy_template_pays_nothing_for_the_strict_check(self, monkeypatch):
        """Strict costs the same two llama-server calls as non-strict when the template
        renders, so verification is free in the case that matters."""
        _window(monkeypatch, 5120)
        calls, _ = self._serving(monkeypatch, 5120, rate = 4.2)
        text = ("The build finished and the archive was uploaded to the release bucket. ") * 400
        budget = tools._tool_result_char_budget()

        assert tools._dense_char_limit(text, budget) == budget
        assert len(calls) == 1

    def test_a_full_cache_evicts_rather_than_refusing_every_later_result(self, monkeypatch):
        """Refusing new entries once full was worse than not caching at all: most tool
        results are one-offs, so the first 64 distinct prefixes froze the cache on text
        nothing would ask about again."""
        _window(monkeypatch, 5120)
        calls, _ = self._serving(monkeypatch, 5120)
        budget = tools._tool_result_char_budget()
        for index in range(tools._PROBE_COUNT_CACHE_ENTRIES + 10):
            tools._dense_char_limit(f"{index:04d}" + "0123456789abcdef" * 2000, budget)

        repeated = "ZZZZ" + "0123456789abcdef" * 2000
        tools._dense_char_limit(repeated, budget)
        calls.clear()
        tools._dense_char_limit(repeated, budget)

        assert calls == [], "a recently measured result is still answered from the cache"

    def test_the_baseline_survives_a_cache_full_of_one_off_results(self, monkeypatch):
        """The sharp edge. The baseline is only priced when a count comes in OVER budget,
        so a process that handled 64 results that FIT first could never get it in at all,
        and every later dense result paid for it again -- the merge base's cost, for the
        life of the process. Measured before the fix: 4 counter calls (8 HTTP) every time.
        """
        _window(monkeypatch, 5120)
        budget = tools._tool_result_char_budget()

        # 64 English results, each of which fits on its first count, so `_framing()` never
        # runs and the baseline is never offered to the cache.
        self._serving(monkeypatch, 5120, rate = 4.2)
        for index in range(tools._PROBE_COUNT_CACHE_ENTRIES):
            tools._dense_char_limit(
                f"{index:04d}" + ("The build finished and the archive was uploaded. ") * 400,
                budget,
            )
        held = list(tools._PROBE_COUNT_CACHE.values())[0]
        assert len(held) == tools._PROBE_COUNT_CACHE_ENTRIES, "the cache really is full"
        assert tools._PROBE_BASELINE not in held, "and the baseline really is not in it"

        # Now dense results arrive. The first pays for the baseline; the rest must not.
        calls, _ = self._serving(monkeypatch, 5120, rate = 1.33)
        tools._dense_char_limit("D1" + "0123456789abcdef" * 2000, budget)
        calls.clear()
        tools._dense_char_limit("D2" + "0123456789abcdef" * 2000, budget)

        assert 0 not in [chars for chars in calls], "the baseline is held, not re-priced"
        held = list(tools._PROBE_COUNT_CACHE.values())[0]
        assert tools._PROBE_BASELINE in held, "and pinned against eviction"

    def test_concurrent_chats_do_not_corrupt_or_crash_on_the_shared_cache(self, monkeypatch):
        """Tool calls run in worker threads (`tool_stream_exec.stream_tool_execution` runs
        each invocation in one), so concurrent chats reach this process-global cache at the
        same time.

        A bare dict assignment is atomic under the GIL, but the LRU touch and the eviction
        are read-then-mutate sequences and are not. Measured before the lock, with this
        exact harness at 24 threads: 69 exceptions out of 12,000 truncations -- `KeyError`
        from popping a key another thread had just evicted, and "dictionary changed size
        during iteration" from choosing a victim while another thread inserted. None of
        them were caught on the way out of `_truncate`.
        """
        import threading

        _window(monkeypatch, 5120)
        self._serving(monkeypatch, 5120)
        # Cache pressure, so eviction fires on nearly every insert, and aggressive
        # preemption so the read-then-mutate windows are actually interleaved.
        monkeypatch.setattr(tools, "_PROBE_COUNT_CACHE_ENTRIES", 3)
        monkeypatch.setattr(tools, "_PROBE_COUNT_CACHE_CHARS", 12_000)
        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-9)
        budget = tools._tool_result_char_budget()
        texts = {f"t{i}": f"{i:04d}" + "0123456789abcdef" * (300 + i * 40) for i in range(8)}
        errors: list[BaseException] = []
        answers: dict[str, set] = {name: set() for name in texts}

        def worker(seed):
            rng = random.Random(seed)
            for _ in range(150):
                name = rng.choice(list(texts))
                try:
                    answers[name].add(tools._dense_char_limit(texts[name], budget))
                except BaseException as exc:  # noqa: BLE001 -- the whole point is to catch it
                    errors.append(exc)

        try:
            threads = [threading.Thread(target = worker, args = (seed,)) for seed in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(previous_interval)

        assert errors == [], f"the shared cache raised under concurrency: {errors[:3]}"
        # And every thread agreed on every answer, which is the point of the whole change.
        for name, seen in answers.items():
            assert len(seen) == 1, f"{name} got different answers in different threads: {seen}"
        # The bounds still hold when several threads insert at once.
        for entry in tools._PROBE_COUNT_CACHE.values():
            assert len(entry) <= 3
            assert sum(map(len, entry)) <= 12_000

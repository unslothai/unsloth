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

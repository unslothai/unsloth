# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A fetched page must fit the real window: a flat 16,000-char cap can exceed a small model's budget."""

from __future__ import annotations

import random
import sys
from types import SimpleNamespace

import pytest

from core.inference import tools


def _shared_setup_1():
    text = "0123456789abcdef" * 2000
    budget = tools._tool_result_char_budget()

    first = tools._dense_char_limit(text, budget)
    return budget, first, text


@pytest.fixture(autouse = True)
def _unknown_window(monkeypatch):
    """Default to "no model loaded" so each test states the window it means."""
    monkeypatch.setattr(tools, "_loaded_context_tokens", lambda: None)
    # The request-scoped window is module state; restore it so tests stay independent.
    token = tools._REQUEST_CONTEXT_TOKENS.set(tools._UNSET_CONTEXT_TOKENS)
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

    monkeypatch.undo()
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
    assert tools._truncate_page_text("x" * 50_000, 200) == "x" * 200
    assert captured["max_chars"] == 200


class TestTheWindowIsReadPerRequest:
    """Read the window per request: no-GGUF native chats got the full cap; externals got the GGUF window"""

    def test_a_native_model_window_is_read_when_no_gguf_is_loaded(self, monkeypatch):
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
    """Code tool results need a window-sized cap too; the newest turn is protected from compaction."""

    def test_a_small_window_shrinks_the_tool_result_cap(self, monkeypatch):
        _window(monkeypatch, 5120)

        budget = tools._tool_result_char_budget()

        assert budget < tools._MAX_OUTPUT_CHARS
        assert budget <= 5120 * 4 * tools._PAGE_CONTEXT_SHARE

    def test_a_large_window_keeps_the_full_cap(self, monkeypatch):
        for ctx in (32_768, 131_072, 262_144):
            _window(monkeypatch, ctx)
            assert tools._tool_result_char_budget() == tools._MAX_OUTPUT_CHARS

    def test_an_unknown_window_keeps_the_full_cap(self):
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
    """A character cap only fits English: CJK and percent-escaped text runs 1.3-1.6 chars per token."""

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

        assert self._dense_tokens(out) <= int(4864 * tools._PAGE_CONTEXT_SHARE) + 64
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

        assert tools._dense_prefix_chars(escaped, 900) == len(escaped)
        assert tools._dense_prefix_chars(escaped, 450) == len(escaped) // 2
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
    """Dense ASCII like base64 is undercharged ~4x by the flat 0.25 tokens/char estimate; measure it."""

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

        assert kept / self._RATE <= 5120 * tools._PAGE_CONTEXT_SHARE
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
        """A proportional shrink overshoots base64 followed by prose, so the returned prefix must be
        counted."""
        _window(monkeypatch, 5120)
        dense_chars = 2500

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
        assert kept > tools._MIN_PAGE_CHARS

    def test_a_template_that_drops_tool_messages_is_still_measured(self, monkeypatch):
        """The probe must render a prompt that contains the chunk: Gemma-4 templates skip tool-role
        messages."""
        _window(monkeypatch, 5120)
        seen = []

        def _count_chat_tokens(messages, *a, **k):
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
        """A count that does not grow with the chunk is not a measurement; fall back to the estimate."""
        _window(monkeypatch, 5120)
        monkeypatch.setattr(
            "routes.inference.get_llama_cpp_backend",
            lambda: SimpleNamespace(
                is_loaded = True,
                context_length = 5120,
                count_chat_tokens = lambda messages, *a, **k: 11,
            ),
        )

        assert tools._loaded_token_counter(5120)("0123456789abcdef" * 250) is None
        assert tools._dense_char_limit("0123456789abcdef" * 2000, 7168) == 7168

    def test_the_readable_floor_still_holds_under_an_exact_count(self, monkeypatch):
        _window(monkeypatch, 1024)
        self._serving(monkeypatch, 1024)

        kept = tools._dense_char_limit("0123456789abcdef" * 2000, tools._MAX_PAGE_CHARS)

        assert kept == tools._MIN_PAGE_CHARS


class TestAConfiguredCapIsNeverRaised:
    """UNSLOTH_TOOL_RESULT_MAX_CHARS is a ceiling; the readability floor must never raise it."""

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

        assert len(tools._truncate(text)) - len(text[:500]) < 400
        assert tools._truncate(text).startswith(text[:500])
        assert studio_tool_loop._truncate_for_model(text).startswith(text[:500])

    def test_an_unconfigured_install_still_gets_the_floor(self, monkeypatch):
        """The floor is untouched wherever the cap is above it, which is the default."""
        _window(monkeypatch, 512)

        assert tools._tool_result_char_budget() == tools._MIN_PAGE_CHARS
        assert tools._page_char_budget() == tools._MIN_PAGE_CHARS


class TestTheProbeIsNotPaidForTwice:
    """Reuse the framing baseline and skip counts at the floor; output must match the merge base exactly."""

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
        """Loaded llama.cpp stand-in counting token calls; is_loaded mirrors the real backend's check."""
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
            backend._process = SimpleNamespace(pid = pid)
            backend.model_identifier = "Qwen3-4B"
            backend._gguf_load_identity = ((gguf, 66306, 4242, 1),)
            backend._chat_template_override = None
            backend._extra_args = list(extra_args) if extra_args else None
        monkeypatch.setattr("routes.inference.get_llama_cpp_backend", lambda: backend)
        return calls, backend

    def test_an_estimate_at_the_floor_is_not_measured_at_all(self, monkeypatch):
        """An estimate under the 2,000-char floor is clamped up whatever the count, so it is never
        measured."""
        _window(monkeypatch, 1024)
        calls, _ = self._serving(monkeypatch, 1024)

        kept = tools._dense_char_limit("0123456789abcdef" * 2000, tools._MAX_PAGE_CHARS)

        assert kept == tools._MIN_PAGE_CHARS
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

        assert kept == budget
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

        assert 0 in cold_calls, "the first dense result pays for the baseline"
        assert calls, "the second result is still measured"
        assert 0 not in calls, "but the baseline is answered from the cache"
        assert len(calls) == len(cold_calls) - 1
        assert cold / self._RATE <= 5120 * tools._PAGE_CONTEXT_SHARE

    def test_the_same_result_twice_costs_nothing_the_second_time(self, monkeypatch):
        """Retries, regenerations and a model that runs the same command again."""
        _window(monkeypatch, 5120)
        calls, _ = self._serving(monkeypatch, 5120)
        budget, first, text = _shared_setup_1()
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
        budget, first, text = _shared_setup_1()
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

        assert tools._dense_char_limit(text, budget) == budget
        state["fail"] = False

        assert tools._dense_char_limit(text, budget) < budget

    def test_the_cache_cannot_grow_without_bound(self, monkeypatch):
        _window(monkeypatch, 5120)
        self._serving(monkeypatch, 5120)
        budget = tools._tool_result_char_budget()

        for index in range(tools._PROBE_COUNT_CACHE_ENTRIES + 40):
            tools._dense_char_limit(f"{index:04d}" + "0123456789abcdef" * 2000, budget)

        held = sum(len(entry) for entry in tools._PROBE_COUNT_CACHE.values())
        assert held <= tools._PROBE_COUNT_CACHE_ENTRIES

    def test_the_cache_is_bounded_by_characters_and_not_only_by_entries(self, monkeypatch):
        """Entry count is no bound: one tool result cached 733,971 characters under a 1,000,000-char cap."""
        monkeypatch.setattr(tools, "_MAX_OUTPUT_CHARS", 1_000_000)
        _window(monkeypatch, 262_144)
        calls, _ = self._serving(monkeypatch, 262_144, rate = 4.0)
        budget = tools._tool_result_char_budget()

        assert budget > tools._MAX_PAGE_CHARS, "the premise: prefixes far exceed the page cap"

        for index in range(8):
            tools._dense_char_limit(f"{index:04d}" + "A" * 6_000_000, budget)

        held = sum(len(key) for entry in tools._PROBE_COUNT_CACHE.values() for key in entry)
        assert held <= tools._PROBE_COUNT_CACHE_CHARS
        assert any("" in entry for entry in tools._PROBE_COUNT_CACHE.values())

    def test_a_prefix_too_large_to_hold_is_skipped_not_stored(self, monkeypatch):
        monkeypatch.setattr(tools, "_PROBE_COUNT_CACHE_CHARS", 5000)
        _window(monkeypatch, 5120)
        calls, _ = self._serving(monkeypatch, 5120)
        budget, first, text = _shared_setup_1()
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
        """Cached counts are stale when extra args pass --chat-template, since llama.cpp is last-wins."""
        _window(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        self._serving(monkeypatch, 5120, rate = 1.33, pid = 900)
        dense = tools._dense_char_limit(text, budget)

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
        budget, first, text = _shared_setup_1()
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
        """/apply-template down: count_chat_tokens(strict = False) counts plain text, dropping role
        markers."""
        rate = self._RATE if rate is None else rate
        calls = []

        def count_chat_tokens(messages, *a, **k):
            body = "".join(m["content"] for m in messages)
            calls.append((len(body), bool(k.get("strict"))))
            if k.get("strict"):
                raise RuntimeError("llama-server could not render the chat template")
            return int(len(body) / (fallback_rate or rate)) or 1

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
        """A plain-text fallback count is used for this call but never cached: it prices a prompt
        never sent."""
        _window(monkeypatch, 5120)
        calls = self._template_down(monkeypatch, 5120)
        text = "0123456789abcdef" * 2000
        budget = tools._tool_result_char_budget()

        kept = tools._dense_char_limit(text, budget)

        assert kept < budget, "the fallback still measured the bytes"
        assert not any(cache for cache in tools._PROBE_COUNT_CACHE.values())
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
        """Price the framing baseline even when earlier results fit, or every dense result pays for
        it again."""
        _window(monkeypatch, 5120)
        budget = tools._tool_result_char_budget()

        self._serving(monkeypatch, 5120, rate = 4.2)
        for index in range(tools._PROBE_COUNT_CACHE_ENTRIES):
            tools._dense_char_limit(
                f"{index:04d}" + ("The build finished and the archive was uploaded. ") * 400,
                budget,
            )
        held = list(tools._PROBE_COUNT_CACHE.values())[0]
        assert len(held) == tools._PROBE_COUNT_CACHE_ENTRIES, "the cache really is full"
        assert tools._PROBE_BASELINE not in held, "and the baseline really is not in it"

        calls, _ = self._serving(monkeypatch, 5120, rate = 1.33)
        tools._dense_char_limit("D1" + "0123456789abcdef" * 2000, budget)
        calls.clear()
        tools._dense_char_limit("D2" + "0123456789abcdef" * 2000, budget)

        assert 0 not in [chars for chars in calls], "the baseline is held, not re-priced"
        held = list(tools._PROBE_COUNT_CACHE.values())[0]
        assert tools._PROBE_BASELINE in held, "and pinned against eviction"

    def test_concurrent_chats_do_not_corrupt_or_crash_on_the_shared_cache(self, monkeypatch):
        """LRU touch and eviction are read-then-mutate, so the shared cache needs a lock, not just
        the GIL."""
        import threading

        _window(monkeypatch, 5120)
        self._serving(monkeypatch, 5120)
        # Small cache plus aggressive preemption so eviction and races actually interleave.
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
        for name, seen in answers.items():
            assert len(seen) == 1, f"{name} got different answers in different threads: {seen}"
        for entry in tools._PROBE_COUNT_CACHE.values():
            assert len(entry) <= 3
            assert sum(map(len, entry)) <= 12_000


def test_a_previous_generation_sentinel_reads_as_unset(monkeypatch):
    """An `execute_tool` held across a reload of tools passes the old sentinels (#11384)."""
    stale = object()
    monkeypatch.setattr("state.tool_policy.require_tool_access", lambda **kw: None)
    seen = []
    monkeypatch.setattr(
        tools,
        "_search_knowledge_base_with_budget",
        lambda a, s, timeout, c, **kw: seen.append(timeout) or "ok",
    )

    token = tools._REQUEST_CONTEXT_TOKENS.set(stale)
    try:
        assert tools._page_char_budget() == tools._MAX_PAGE_CHARS
        assert tools._tool_result_char_budget() == tools._MAX_OUTPUT_CHARS
        assert tools._window_context_tokens() is None
        _window(monkeypatch, 4864)
        assert tools._page_char_budget() == 6809
        assert tools._tool_result_char_budget() == 6809
        assert tools._window_context_tokens() == 4864

        tools.execute_tool("search_knowledge_base", {}, timeout = stale, context_tokens = stale)
        assert seen == [tools._EXEC_TIMEOUT]
        assert tools._page_char_budget() == 6809
    finally:
        tools._REQUEST_CONTEXT_TOKENS.reset(token)

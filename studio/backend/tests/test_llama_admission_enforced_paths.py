# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bound has to hold on every path that builds a wire ``max_tokens``.

The arithmetic is proved next door; this proves the figure reaches every request, since a
tool round, the final answer, both respawn refits, the post-respawn retry and /v1/messages
each rebuilt the cap from the whole window. Driven through the real generators, so what is
asserted is the payload llama-server would have received.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import core.inference.llama_cpp as llama_cpp_mod
import routes.inference as inf_mod
from core.inference.api_monitor import ApiMonitor
from core.inference.chat_template_helpers import (
    model_markup,
    neutralize_control_markup_in_messages,
)
from core.inference.context_window import estimate_messages_tokens_dense, tool_result_budget
from core.inference.llama_admission import (
    ADMISSION_CONTROL_ENV,
    ADMISSION_KV_BUDGET_ENV,
    LlamaAdmissionCancelled,
    LlamaAdmissionConfig,
    LlamaAdmissionQueue,
    LlamaAdmissionRecostRefused,
    reset_llama_admission_queues,
)
from core.inference.llama_cpp import LlamaCppBackend
from fastapi.responses import JSONResponse

from models.inference import AnthropicMessagesRequest, ChatCompletionRequest
from routes.inference import (
    _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS,
    _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS,
    _OPENAI_LLAMA_ADMISSION_WIRE_RESERVE_TOKENS as _RESERVE,
    _build_openai_passthrough_body,
    _openai_llama_admission_retry_max_tokens,
    _openai_llama_admission_wire_prompt_tokens,
    _openai_llama_admission_enforced_max_tokens,
    _openai_llama_admission_prompt_tokens,
    _openai_llama_admission_recost,
    _openai_llama_admission_tokens,
    anthropic_messages,
)

_CTX = 4096
# Four slots on one unified cache, so a share is a quarter of the window.
_SHARE = _CTX // 4


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _done() -> str:
    return "data: [DONE]\n"


def _finish(reason: str) -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


def _finish_with_usage(reason: str, prompt_tokens: int, completion_tokens: int) -> str:
    return (
        "data: "
        + json.dumps(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )
        + "\n"
    )


def _tool_call(name: str, arguments: dict, call_id: str) -> list[str]:
    return [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
            }
        ),
        _done(),
    ]


def _make_backend(monkeypatch, streams: list[object], payloads: list[dict]):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48851
    backend._api_key = None
    backend._effective_context_length = _CTX
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream_with_retry(
        _client,
        _url,
        payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
    ):
        payloads.append(copy.deepcopy(payload))
        stream = streams.pop(0)
        if isinstance(stream, BaseException):
            raise stream
        yield type("FakeResponse", (), {"status_code": 200, "chunks": stream})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
    ):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    return backend


def _caps(payloads: list[dict]) -> list[int]:
    return [payload["max_tokens"] for payload in payloads]


def _count_by_length(messages, *_args, **_kwargs) -> int:
    """Stands in for the template render, on the estimator's four-characters-a-token."""
    return sum(len(str(message.get("content") or "")) for message in messages) // 4


_TOOL = {
    "type": "function",
    "function": {"name": "web_search", "parameters": {"type": "object", "properties": {}}},
}


def _run_tool_loop(
    monkeypatch,
    payloads,
    *,
    streams = None,
    backend = None,
    events = None,
    **kwargs,
):
    if backend is None:
        backend = _make_backend(
            monkeypatch,
            streams
            if streams is not None
            else [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10"}), _done()],
            ],
            payloads,
        )
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
    )
    produced = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Which kernel?"}],
            tools = [_TOOL],
            max_tool_iterations = 1,
            permission_mode = "off",
            **kwargs,
        )
    )
    if events is not None:
        events.extend(produced)
    return backend


class TestTheGeneratorsSendIt:
    """Every request a run makes, not only the first one of each kind."""

    def test_the_plain_stream_sends_the_bound(self, monkeypatch):
        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)

        list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hello"}],
                admission_output_allowance = _SHARE,
            )
        )

        assert _caps(payloads) == [_SHARE]

    def test_a_respawn_retry_keeps_the_bound(self, monkeypatch):
        """The retry re-enters the generator, which rebuilds an uncapped cap."""
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [httpx.ConnectError("dead"), [_sse({"content": "hi"}), _done()]],
            payloads,
        )
        monkeypatch.setattr(backend, "_respawn_if_dead", lambda *_a, **_k: True)

        list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hello"}],
                admission_output_allowance = _SHARE,
            )
        )

        assert len(payloads) == 2, "expected the first attempt and the post-respawn retry"
        assert _caps(payloads) == [_SHARE, _SHARE]

    def test_a_truncated_plain_chat_is_re_priced_from_what_it_sends(self, monkeypatch):
        """A history over the window prices at the one-token floor before the fit runs.

        The plain path has no re-cost, so that floor used to reach llama-server even
        though `truncate_oldest` had just made room, turning any overlong chat into a
        one-token reply. The bound is re-priced from the messages the fit leaves.
        """
        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)
        monkeypatch.setattr(backend, "count_chat_tokens", _count_by_length)
        stub = _backend_stub(window = _CTX, total = _CTX, slots = 4)
        history = [
            {"role": "user", "content": f"turn {index} " + "word " * 400} for index in range(10)
        ]

        def _price(messages):
            return _openai_llama_admission_enforced_max_tokens(
                _Payload(messages = messages),
                request = None,
                llama_backend = stub,
                conversation = messages,
            )

        assert _price(history) == 1, "the pre-fit prompt has to price at the floor"

        list(
            backend.generate_chat_completion(
                messages = history,
                context_overflow = "truncate_oldest",
                admission_output_allowance = _price(history),
                on_prompt_fitted = _price,
            )
        )

        sent = _caps(payloads)[0]
        fitted = _openai_llama_admission_wire_prompt_tokens(payloads[0]["messages"])
        assert sent > 1, "the fit made room and the bound never moved"
        assert fitted + sent <= _CTX, (fitted, sent)

    def test_a_plain_chat_that_fits_keeps_the_bound_it_was_priced(self, monkeypatch):
        """No truncation, so the re-price is the same figure and nothing widens."""
        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)
        stub = _backend_stub(window = _CTX, total = _CTX, slots = 4)

        def _price(messages):
            return _openai_llama_admission_enforced_max_tokens(
                _Payload(messages = messages),
                request = None,
                llama_backend = stub,
                conversation = messages,
            )

        messages = [{"role": "user", "content": "hello"}]
        list(
            backend.generate_chat_completion(
                messages = messages,
                context_overflow = "truncate_oldest",
                admission_output_allowance = _price(messages),
                on_prompt_fitted = _price,
            )
        )

        assert _caps(payloads) == [_price(messages)]

    def test_the_tool_round_and_the_final_pass_both_send_it(self, monkeypatch):
        """The final pass carries the whole run's history and skips the top of the loop."""
        payloads: list[dict] = []
        _run_tool_loop(monkeypatch, payloads, admission_output_allowance = _SHARE)

        assert len(payloads) == 2, "expected one tool round and one synthesized final pass"
        assert _caps(payloads) == [_SHARE, _SHARE]

    def test_a_re_cost_moves_the_bound_with_the_conversation(self, monkeypatch):
        """A cap frozen at the opening prompt drifts as far as the loop grows."""
        payloads: list[dict] = []
        recosted = iter([_SHARE - 100, _SHARE - 400])
        _run_tool_loop(
            monkeypatch,
            payloads,
            admission_output_allowance = _SHARE,
            on_conversation_grew = lambda _conversation, _tools: next(recosted, None),
        )

        assert _caps(payloads) == [_SHARE - 100, _SHARE - 400]

    def test_a_re_cost_that_says_nothing_leaves_the_bound_alone(self, monkeypatch):
        """Accounting that declined to re-price must not read as "no bound"."""
        payloads: list[dict] = []
        _run_tool_loop(
            monkeypatch,
            payloads,
            admission_output_allowance = _SHARE,
            on_conversation_grew = lambda _conversation, _tools: None,
        )

        assert _caps(payloads) == [_SHARE, _SHARE]

    def test_the_hook_is_told_which_catalogue_each_request_sends(self, monkeypatch):
        """Rounds carry the catalogue, the final answer sends none, and the loop narrows."""
        payloads: list[dict] = []
        seen: list = []

        def _recost(_conversation, tools):
            seen.append(tools)
            return _SHARE - (500 if tools else 100)

        _run_tool_loop(
            monkeypatch,
            payloads,
            admission_output_allowance = _SHARE,
            on_conversation_grew = _recost,
        )

        assert len(seen) == 2
        assert [tool["function"]["name"] for tool in seen[0]] == ["web_search"]
        assert seen[1] is None, "the final pass sends no tools array"
        assert _caps(payloads) == [_SHARE - 500, _SHARE - 100]

    def test_a_respawn_refit_does_not_restore_the_window(self, monkeypatch):
        """A replacement server reporting a bigger window is not a bigger reservation."""
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                httpx.ConnectError("dead"),
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10"}), _done()],
            ],
            payloads,
        )

        def _respawn_bigger(*_args, **_kwargs):
            backend._effective_context_length = _CTX * 2
            return True

        monkeypatch.setattr(backend, "_respawn_if_dead", _respawn_bigger)
        # The window is what this test is about, not the fit's token count.
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 10)
        _run_tool_loop(
            monkeypatch,
            payloads,
            backend = backend,
            context_overflow = "truncate_oldest",
            admission_output_allowance = _SHARE,
        )

        assert payloads, "no request was sent"
        assert all(cap <= _SHARE for cap in _caps(payloads)), _caps(payloads)

    def test_every_final_continuation_is_re_costed(self, monkeypatch):
        """A continuation is a bigger prompt on the same lease."""
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10 and then some"}), _finish("length"), _done()],
                [_sse({"content": " more"}), _done()],
            ],
            payloads,
        )
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 10)
        seen: list[int] = []
        recosted = iter([None, _SHARE - 100, _SHARE - 700])

        def _recost(messages, tools):
            if tools is None:
                seen.append(len(messages))
            return next(recosted, None)

        _run_tool_loop(
            monkeypatch,
            payloads,
            backend = backend,
            admission_output_allowance = _SHARE,
            on_conversation_grew = _recost,
        )

        assert len(seen) == 2, f"the final re-cost ran {len(seen)} time(s), not once per attempt"
        assert seen[1] >= seen[0], "the retry should carry at least the first attempt's prompt"
        assert _caps(payloads)[1:] == [_SHARE - 100, _SHARE - 700]

    def test_a_continuation_that_earned_a_bigger_allowance_gets_it(self, monkeypatch):
        """The re-cost is the cap, not a ceiling the previous attempt keeps lowering.

        Replaying a truncated answer moves the prompt to or above its share, where the
        allowance stops being `share - prompt` and becomes the flat unstated figure. The
        lease is re-costed for it before the hook returns, so a continuation held to the
        previous attempt's smaller cap stops short of an answer already paid for.
        """
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10 and then some"}), _finish("length"), _done()],
                [_sse({"content": " more"}), _done()],
            ],
            payloads,
        )
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 10)
        recosted = iter([None, _SHARE - 900, _SHARE - 100])
        _run_tool_loop(
            monkeypatch,
            payloads,
            backend = backend,
            admission_output_allowance = _SHARE,
            on_conversation_grew = lambda _messages, _tools: next(recosted, None),
        )

        assert _caps(payloads)[1:] == [_SHARE - 900, _SHARE - 100], _caps(payloads)


def _refuse_on_call(index: int):
    """A hook that grants every re-cost but the ``index``-th, which the ledger refuses."""
    calls = {"n": -1}

    def _hook(_conversation, _tools):
        calls["n"] += 1
        if calls["n"] == index:
            raise LlamaAdmissionRecostRefused("no room")
        return _SHARE - 100

    return _hook


def _finish_reasons(events: list) -> list:
    return [
        event["finish_reason"]
        for event in events
        if event.get("type") == "metadata" and event.get("finish_reason")
    ]


def _shown(events: list) -> str:
    """The last content event, which is what the client is left displaying: these are
    cumulative, so an event sent later replaces everything before it."""
    texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    return texts[-1] if texts else ""


class TestARefusedReCostDoesNotAuthoriseTheRequest:
    """``recost_waiting`` returning False leaves the PREVIOUS round's figure in force, so
    the prompt that asked for the growth is over the reservation. Handing it a freshly
    computed wire cap authorised exactly the aggregate the lease refused to buy."""

    def test_a_refused_round_sends_nothing_and_ends_the_turn_on_length(self, monkeypatch):
        payloads: list[dict] = []
        events: list[dict] = []
        _run_tool_loop(
            monkeypatch,
            payloads,
            events = events,
            admission_output_allowance = _SHARE,
            on_conversation_grew = _refuse_on_call(0),
        )

        assert payloads == [], "a refused re-cost still sent the larger prompt"
        assert _finish_reasons(events) == ["length"]
        # Nothing had been shown, so the turn has to say why it stopped rather than
        # render as an empty message.
        assert _shown(events).strip()

    def test_a_refused_final_attempt_keeps_the_partial_it_has(self, monkeypatch):
        """The continuation is the growth being refused, so the first answer stands."""
        payloads: list[dict] = []
        events: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10 and then some"}), _finish("length"), _done()],
                [_sse({"content": " more"}), _done()],
            ],
            payloads,
        )
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 10)
        _run_tool_loop(
            monkeypatch,
            payloads,
            backend = backend,
            events = events,
            admission_output_allowance = _SHARE,
            # Round zero, the first final attempt, then the continuation.
            on_conversation_grew = _refuse_on_call(2),
        )

        assert len(payloads) == 2, "the refused continuation was sent anyway"
        assert _finish_reasons(events)[-1] == "length"
        assert "6.10 and then some" in _shown(events), _shown(events)

    def test_a_refused_ending_reports_the_prompt_of_the_attempt_that_ran(self, monkeypatch):
        """The refusal sends nothing, so its usage is the last attempt's prompt and every
        attempt's generation, not a zero prompt."""
        payloads: list[dict] = []
        events: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [
                    _sse({"content": "6.10 and then some"}),
                    _finish_with_usage("length", 1234, 7),
                    _done(),
                ],
            ],
            payloads,
        )
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 10)
        _run_tool_loop(
            monkeypatch,
            payloads,
            backend = backend,
            events = events,
            admission_output_allowance = _SHARE,
            on_conversation_grew = _refuse_on_call(2),
        )

        metadata = [event for event in events if event.get("type") == "metadata"]
        assert metadata[-1]["finish_reason"] == "length"
        assert metadata[-1]["usage"]["prompt_tokens"] == 1234, metadata[-1]["usage"]
        assert metadata[-1]["usage"]["completion_tokens"] == 7, metadata[-1]["usage"]

    def test_a_stop_during_the_wait_ends_the_turn_as_a_stop(self, monkeypatch):
        """Waiting for room answers Stop by returning False too; that is a cancel, not a
        refusal, so the turn ends with neither the explanation nor a Continue."""
        payloads: list[dict] = []
        events: list[dict] = []

        def _cancelled(_conversation, _tools):
            raise LlamaAdmissionCancelled("stopped")

        _run_tool_loop(
            monkeypatch,
            payloads,
            events = events,
            admission_output_allowance = _SHARE,
            on_conversation_grew = _cancelled,
        )

        assert payloads == []
        assert _finish_reasons(events) == []
        assert not _shown(events).strip()

    def test_the_round_is_re_costed_for_the_prompt_it_sends(self):
        """Under truncate_oldest the fit drops history before the request is built; a
        refusal for what the fit was about to drop would end a turn that fits."""
        import inspect

        source = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
        fit = source.index("conversation, truncation = _fit_with_instruction_pins(")
        recost = source.index("on_conversation_grew(conversation, safe_tools)")
        assert fit < recost, "the round is re-costed before its compaction"


class TestWhatTheWaitReturnsFalseFor:
    """``recost_waiting`` returns False for a refusal, a Stop and a release alike; only
    the first is the ledger's answer."""

    def _lease(self, *, released = False):
        return SimpleNamespace(
            recost_waiting = lambda *_a, **_k: False,
            released = released,
        )

    def _recost(
        self,
        lease,
        cancel_event = None,
    ):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        grown = [{"role": "user", "content": "word " * 900}]
        return _openai_llama_admission_recost(
            SimpleNamespace(lease_nowait = lambda: lease),
            grown,
            request = None,
            llama_backend = backend,
            payload = _chat(max_tokens = 16384),
            output_tokens = 16384,
            cancel_event = cancel_event,
        )

    def test_a_refusal_is_raised_as_one(self):
        import threading
        with pytest.raises(LlamaAdmissionRecostRefused):
            self._recost(self._lease(), threading.Event())

    def test_a_stop_is_a_cancel(self):
        import threading

        stop = threading.Event()
        stop.set()
        with pytest.raises(LlamaAdmissionCancelled):
            self._recost(self._lease(), stop)

    def test_a_released_lease_is_a_cancel(self):
        with pytest.raises(LlamaAdmissionCancelled):
            self._recost(self._lease(released = True))

    def test_a_granted_re_cost_still_returns_the_clamped_cap(self, monkeypatch):
        """The refusal path must not cost the ordinary one its bound."""
        payloads: list[dict] = []
        _run_tool_loop(
            monkeypatch,
            payloads,
            admission_output_allowance = _SHARE,
            on_conversation_grew = _refuse_on_call(99),
        )

        assert _caps(payloads) == [_SHARE - 100, _SHARE - 100]


class TestTheLedgerBoundsTheAggregate:
    """The audited scenario, with the real queue and the real lease rather than a stub."""

    @pytest.fixture(autouse = True)
    def _isolate(self, monkeypatch):
        monkeypatch.setenv(ADMISSION_CONTROL_ENV, "1")
        monkeypatch.setenv(ADMISSION_KV_BUDGET_ENV, "1")
        reset_llama_admission_queues()
        yield
        reset_llama_admission_queues()

    def test_a_full_pool_refuses_the_grown_round_rather_than_pricing_it(self):
        # The queue books its slots against the running loop, as a route would.
        asyncio.run(self._refuses_the_grown_round())

    async def _refuses_the_grown_round(self):
        budget = 16384
        queue = LlamaAdmissionQueue("recost-refusal")
        reservations = [
            queue.reserve(
                capacity = 4,
                config = LlamaAdmissionConfig(),
                tokens = budget // 4,
                budget = budget,
            )
            for _ in range(4)
        ]
        try:
            # No idle clearing, so a round cannot yield its commitment to wait for room:
            # Studio launches exactly that way on Windows under full GPU offload.
            backend = SimpleNamespace(
                context_length = budget,
                _kv_cache_context_total = budget,
                effective_parallel_slots = 4,
                _kv_cache_unified = True,
                idle_slot_clearing_active = False,
            )
            grown = [{"role": "tool", "content": "\u4e2d" * 4992}]
            prompt = inf_mod.estimate_messages_tokens_dense(grown)
            assert prompt > budget // 4, "the round has to have grown past its share"

            with pytest.raises(LlamaAdmissionRecostRefused):
                _openai_llama_admission_recost(
                    reservations[0],
                    grown,
                    request = None,
                    llama_backend = backend,
                    payload = _chat(),
                    output_tokens = None,
                )

            # The refusal is what keeps this true: the lease still holds its share, so
            # the four leases plus this prompt would be over the pool if it were sent.
            assert reservations[0].lease_nowait()._tokens == budget // 4
            assert queue.snapshot().committed == budget
            assert 3 * (budget // 4) + prompt > budget
        finally:
            for reservation in reservations:
                reservation.lease_nowait().release()

    def test_round_zero_is_charged_what_its_opening_was_charged(self):
        asyncio.run(self._round_zero_matches_the_opening())

    async def _round_zero_matches_the_opening(self):
        """A translating route folds `system` into the conversation and prices the
        catalogue once; a re-cost that added the payload's raw `system` and `tools` on top
        asked for more than the share at round zero and, with nothing to yield, was
        refused before any generation ran."""
        budget = 16384
        share = budget // 4
        # 440, not 500: a longer system prompt lands inside the wire reserve of its share,
        # where it is deliberately priced the flat allowance instead, and the catalogue's
        # template preamble moved that line again. This test is about the path that DOES
        # fit its share, so it is sized with room to spare rather than against the edge.
        system = "You are a careful assistant. " * 440
        payload = _Payload(
            messages = [{"role": "user", "content": "hi"}],
            system = system,
            tools = _CATALOGUE,
            max_tokens = None,
        )
        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": "hi"},
        ]
        backend = SimpleNamespace(
            context_length = budget,
            _kv_cache_context_total = budget,
            effective_parallel_slots = 4,
            _kv_cache_unified = True,
            idle_slot_clearing_active = False,
        )
        opened = _openai_llama_admission_tokens(
            payload,
            budget = budget,
            capacity = 4,
            tool_loop = True,
            conversation = conversation,
            injected_tools = _CATALOGUE,
        )
        assert opened == share, opened
        queue = LlamaAdmissionQueue("recost-parity")
        reservations = [
            queue.reserve(capacity = 4, config = LlamaAdmissionConfig(), tokens = share, budget = budget)
            for _ in range(4)
        ]
        try:
            # The same conversation and catalogue the opening priced: nothing grew.
            _openai_llama_admission_recost(
                reservations[0],
                conversation,
                request = None,
                llama_backend = backend,
                payload = payload,
                output_tokens = None,
                injected_tools = _CATALOGUE,
                wire_tools = _CATALOGUE,
            )
            assert reservations[0].lease_nowait()._tokens == opened
            assert queue.snapshot().committed == budget
        finally:
            for reservation in reservations:
                reservation.lease_nowait().release()


def _backend_stub(*, window, total, slots):
    return SimpleNamespace(
        context_length = window,
        _kv_cache_context_total = total,
        effective_parallel_slots = slots,
    )


class _Payload:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


def _reservation():
    """A lease that accepts any re-cost, so a test reads the figure it hands back."""

    class _Lease:
        def recost_waiting(self, *_args, **_kwargs):
            # True is "the new figure is in force"; anything falsy is a refusal now.
            return True

    return SimpleNamespace(lease_nowait = lambda: _Lease())


def _chat(text = "hi", **fields):
    return _Payload(messages = [{"role": "user", "content": text}], **fields)


_CATALOGUE = [
    {
        "type": "function",
        "function": {
            "name": "search_conversation",
            "description": "Search the conversation. " * 60,
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


class TestEveryCallSiteCarriesIt:
    """A generation call site added without the bound is the whole defect, again. Read off
    the route source: what has to hold is that NO branch is left out."""

    def _routes_tree(self):
        import ast
        return ast.parse(Path(inf_mod.__file__).read_text(encoding = "utf-8"))

    def _generator_calls(self, tree):
        """Calls on `llama_backend`, the only receiver holding a KV lease. The safetensors
        twin decodes in-process against no cache, so it takes no reservation."""
        import ast
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None)
            receiver = getattr(getattr(node.func, "value", None), "id", None)
            if receiver != "llama_backend":
                continue
            if name in ("generate_chat_completion", "generate_chat_completion_with_tools"):
                yield name, node

    def test_no_generation_call_site_is_left_unbounded(self):
        import ast

        tree = self._routes_tree()
        unbounded = [
            name
            for name, call in self._generator_calls(tree)
            if not any(keyword.arg == "admission_output_allowance" for keyword in call.keywords)
        ]
        assert not unbounded, f"these call sites send the whole window: {unbounded}"

    def test_a_fitting_call_site_re_prices_what_the_fit_leaves(self):
        """`context_overflow` turns the fit on, and the fit moves the prompt the bound was
        priced from. The loop re-prices through its re-cost; the plain path has no re-cost,
        so it needs the fitted hook or it sends the pre-fit floor."""
        # Which hook re-prices the bound on each generator.
        _REPRICES = {
            "generate_chat_completion": "on_prompt_fitted",
            "generate_chat_completion_with_tools": "on_conversation_grew",
        }
        tree = self._routes_tree()
        blind = [
            name
            for name, call in self._generator_calls(tree)
            if any(keyword.arg == "context_overflow" for keyword in call.keywords)
            and not any(keyword.arg == _REPRICES[name] for keyword in call.keywords)
        ]
        assert not blind, f"these fit the prompt but keep the pre-fit bound: {blind}"

    def test_an_overflow_retry_re_prices_the_cap_it_kept(self):
        """The passthrough twins drop history on an upstream overflow. The cap in the body
        was priced on that history, so a retry that does not re-price sends the floor."""
        import ast

        tree = self._routes_tree()
        blind = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_apply_measured_overflow_truncation"
            and not any(keyword.arg == "reprice_max_tokens" for keyword in node.keywords)
        ]
        assert not blind, f"{len(blind)} overflow retries keep their pre-truncation bound"

    def test_a_tool_loop_bound_is_priced_with_the_catalogue_it_sends(self):
        """`payload.tools` omits Studio's server-side catalogue, which the lease charges."""
        import ast

        tree = self._routes_tree()
        priced_with_catalogue = {
            target.id: any(keyword.arg == "injected_tools" for keyword in node.value.keywords)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None)
            == "_openai_llama_admission_enforced_max_tokens"
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert priced_with_catalogue, "no allowance is computed in the route at all"

        blind = []
        for name, call in self._generator_calls(tree):
            if name != "generate_chat_completion_with_tools":
                continue
            for keyword in call.keywords:
                if keyword.arg != "admission_output_allowance":
                    continue
                bound_to = getattr(keyword.value, "id", None)
                if not priced_with_catalogue.get(bound_to, False):
                    blind.append(bound_to)
        assert not blind, f"tool-loop bounds priced without the catalogue they send: {blind}"


class TestWhatThePromptIsMeasuredAgainst:
    def test_the_injected_catalogue_is_inside_the_bound(self):
        """The catalogue is inside the bound, so no request is permitted `share + it`."""
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        prompt = _openai_llama_admission_prompt_tokens(payload, injected_tools = _CATALOGUE)
        blind = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        aware = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, injected_tools = _CATALOGUE
        )
        charged = _openai_llama_admission_tokens(
            payload,
            budget = 16384,
            capacity = 4,
            tool_loop = True,
            injected_tools = _CATALOGUE,
            context_window = 16384,
        )
        assert prompt + blind > charged, "the catalogue has to be big enough to matter"
        assert prompt + aware <= charged
        assert (prompt + aware) * 4 <= 16384

    def test_a_grown_conversation_earns_a_smaller_cap(self):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)

        reservation = _reservation()

        opening = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        grown = [{"role": "user", "content": "word " * 900}]
        recosted = _openai_llama_admission_recost(
            reservation,
            grown,
            request = None,
            llama_backend = backend,
            payload = payload,
            output_tokens = 16384,
        )
        assert recosted is not None and recosted < opening, (opening, recosted)
        prompt = _openai_llama_admission_prompt_tokens(_Payload(messages = grown))
        assert prompt + recosted <= 16384 // 4

    def test_a_client_that_named_a_cap_is_never_bounded_at_either_end(self):
        """A caller under the window is charged what it asked for, so a bound handed back
        mid-loop would cut its answer at a share with nothing to say why."""
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)

        reservation = _reservation()
        grown = [{"role": "user", "content": "word " * 900}]
        for payload, cap in (
            (_chat(max_tokens = 512), 512),
            (_chat(max_completion_tokens = 512), 512),
        ):
            assert (
                _openai_llama_admission_enforced_max_tokens(
                    payload, request = None, llama_backend = backend
                )
                is None
            )
            assert (
                _openai_llama_admission_recost(
                    reservation,
                    grown,
                    request = None,
                    llama_backend = backend,
                    payload = payload,
                    output_tokens = cap,
                )
                is None
            )


class TestWhatTheWireActuallyCarries:
    """The charge is deliberately conservative; the bound cannot be: the ledger re-adds
    terms the next request does not carry, and each would come off the answer."""

    def test_a_translated_system_prompt_is_not_charged_to_the_answer_twice(self):
        """`anthropic_messages_to_openai` folds `system` into the re-costed conversation,
        so the ledger's extra-prompt term counts it twice; on the wire that costs tokens."""
        backend = _backend_stub(window = 65536, total = 65536, slots = 4)
        system = "You are a careful assistant. " * 200
        payload = _Payload(
            messages = [{"role": "user", "content": "hi"}],
            system = system,
            max_tokens = 65536,
        )
        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": "hi"},
        ]

        reservation = _reservation()
        wire = _openai_llama_admission_recost(
            reservation,
            conversation,
            request = None,
            llama_backend = backend,
            payload = payload,
            output_tokens = 65536,
        )
        share = 65536 // 4
        conversation_tokens = _openai_llama_admission_wire_prompt_tokens(
            conversation, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        assert wire == share - _RESERVE - conversation_tokens, (wire, share, conversation_tokens)

    def test_image_transport_bytes_do_not_come_off_the_answer(self):
        """An Anthropic `image` part is priced as an image on both sides, never as base64
        text (#10669), so the raw payload and its translation agree and neither swamps the share."""
        # A share (8192) above one image's allowance but below the base64 as prompt text.
        backend = _backend_stub(window = 32768, total = 32768, slots = 4)
        data = "A" * 40000
        payload = _Payload(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": data},
                        },
                    ],
                }
            ],
            max_tokens = 32768,
        )
        translated = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                ],
            }
        ]
        raw = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        wire = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = translated
        )
        # The two shapes differ only by their JSON envelope around the same image.
        assert (
            abs(raw - wire) <= 32
        ), "the raw Anthropic image must be priced as an image, not as base64"
        assert (
            raw > _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS - _RESERVE
        ), "the base64 transport must not swamp the share"
        assert wire == 32768 // 4 - _RESERVE - _openai_llama_admission_wire_prompt_tokens(
            translated, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )

    def test_a_request_is_not_charged_a_catalogue_it_does_not_send(self):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)

        reservation = _reservation()
        conversation = [{"role": "user", "content": "word " * 700}]

        def _recost(wire_tools):
            return _openai_llama_admission_recost(
                reservation,
                conversation,
                request = None,
                llama_backend = backend,
                payload = payload,
                output_tokens = 16384,
                injected_tools = _CATALOGUE,
                wire_tools = wire_tools,
            )

        with_tools = _recost(_CATALOGUE)
        without = _recost(None)
        assert without > with_tools, (with_tools, without)
        catalogue = _openai_llama_admission_prompt_tokens(
            _Payload(messages = [{"role": "user", "content": ""}]), injected_tools = _CATALOGUE
        ) - _openai_llama_admission_prompt_tokens(
            _Payload(messages = [{"role": "user", "content": ""}])
        )
        assert without - with_tools == catalogue

    def test_a_prompt_injected_after_the_payload_is_still_inside_the_bound(self):
        """The plain GGUF path splices in a date prompt and has no re-cost to notice it."""
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        injected = [
            {"role": "system", "content": "Today's date is 2026-09-08. " * 40},
            {"role": "user", "content": "hi"},
        ]
        raw = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        wire = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = injected
        )
        assert wire < raw, (raw, wire)
        assert (
            _openai_llama_admission_wire_prompt_tokens(
                injected, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
            )
            + wire
            <= 16384 // 4
        )

    def test_audio_and_video_are_left_unenforced(self):
        """Nothing here can size their prompt KV: transport length as a prompt count would
        floor any real recording's answer at one token. Images keep a real bound."""
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        clip = "A" * 200000
        text_only = _chat("listen", max_tokens = 16384)
        assert (
            _openai_llama_admission_enforced_max_tokens(
                text_only, request = None, llama_backend = backend
            )
            is not None
        )
        with_audio = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "listen"},
                    {"type": "input_audio", "input_audio": {"data": clip, "format": "wav"}},
                ],
            }
        ]
        assert (
            _openai_llama_admission_enforced_max_tokens(
                text_only,
                request = None,
                llama_backend = backend,
                conversation = with_audio,
            )
            is None
        )
        assert (
            _openai_llama_admission_enforced_max_tokens(
                _Payload(
                    messages = [{"role": "user", "content": "listen"}],
                    audio_base64 = clip,
                    max_tokens = 16384,
                ),
                request = None,
                llama_backend = backend,
            )
            is None
        )

    def test_a_legacy_image_already_spliced_in_is_charged_once(self):
        """The splice already counts `image_base64`; the separate charge doubles it."""
        backend = _backend_stub(window = 32768, total = 32768, slots = 4)
        image = "iVBORw0KGgo="
        payload = _Payload(
            messages = [{"role": "user", "content": "what is this?"}],
            image_base64 = image,
            max_tokens = 32768,
        )
        spliced = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                ],
            }
        ]
        one_image = _openai_llama_admission_wire_prompt_tokens(
            spliced, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        text_only = _openai_llama_admission_wire_prompt_tokens(
            [{"role": "user", "content": "what is this?"}],
            image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS,
        )
        assert one_image - text_only < 2 * _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS, (
            one_image,
            text_only,
        )
        bound = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = spliced
        )
        assert bound is not None and bound > 1

    def test_the_lease_is_taken_on_the_same_finalized_messages(self):
        """Charged and permitted must be the same prompt, so both use `gguf_messages`."""
        budget, slots = 16384, 4
        share = budget // slots
        # Sized so the raw payload sits under its share and the finalized prompt over it.
        turn = "word " * 3200
        payload = _chat(turn, max_tokens = budget)
        injected = [
            {"role": "system", "content": "Today's date is 2026-09-08. " * 40},
            {"role": "user", "content": turn},
        ]
        charged = _openai_llama_admission_tokens(
            payload,
            budget = budget,
            capacity = slots,
            context_window = budget,
            conversation = injected,
        )
        backend = _backend_stub(window = budget, total = budget, slots = slots)
        bound = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = injected
        )
        prompt = _openai_llama_admission_wire_prompt_tokens(
            injected, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        raw = _openai_llama_admission_prompt_tokens(payload)
        assert raw < share < prompt, (raw, share, prompt)
        assert charged >= prompt + bound, (charged, prompt, bound)

    def test_transport_bytes_stay_on_the_ledger(self):
        """The charge must keep them: they stop an audio request sharing the cache."""
        budget, slots = 262144, 4
        clip = "A" * 400000
        conversation = [{"role": "user", "content": "listen"}]
        without = _openai_llama_admission_tokens(
            _chat("listen", max_tokens = budget),
            budget = budget,
            capacity = slots,
            context_window = budget,
            conversation = conversation,
        )
        with_clip = _openai_llama_admission_tokens(
            _Payload(
                messages = [{"role": "user", "content": "listen"}],
                audio_base64 = clip,
                max_tokens = budget,
            ),
            budget = budget,
            capacity = slots,
            context_window = budget,
            conversation = conversation,
        )
        assert with_clip > without
        assert with_clip >= len(clip) // 4

    @staticmethod
    def _marker_paste(markers: int):
        """A transcript pasted into one turn, carrying the loaded model's own markers."""
        body = "".join(f"<|im_start|>user\nq {index}<|im_end|>\n" for index in range(markers // 2))
        return [{"role": "user", "content": "Explain this log:\n" + body}]


class TestARetryThatGrewItsPrompt:
    def test_the_nudge_retry_is_bounded_by_its_own_prompt(self):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        first = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4088}
        grown = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "word " * 500},
                {"role": "user", "content": "please call the tool"},
            ],
            "max_tokens": 4088,
        }
        bound = _openai_llama_admission_retry_max_tokens(
            grown, admission_output_allowance = 4088, request = None, llama_backend = backend
        )
        assert bound is not None and bound < first["max_tokens"]
        assert (
            _openai_llama_admission_wire_prompt_tokens(
                grown["messages"], image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
            )
            + bound
            <= 16384 // 4
        )

    def test_the_retry_never_gets_a_second_allowance(self):
        """One lease covers both attempts, so the retry spends what is left of it.

        Under a 16K unified pool with four slots the first attempt is charged a share:
        prompt 3007 plus a 1089-token allowance. The model fills the allowance with an
        unparseable call, the nudge appends it and asks again, and the retry prompt is now
        past the share -- where the wire bound hands out the flat unstated allowance. That
        is 1024 tokens of KV nobody reserved, and four such chats occupy 21736 of 16384.
        """
        budget, slots = 16384, 4
        backend = _backend_stub(window = budget, total = budget, slots = slots)
        first_messages = [{"role": "user", "content": "word " * 2400}]
        payload = _chat("word " * 2400, max_tokens = budget)
        allowance = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = first_messages
        )
        charge = _openai_llama_admission_tokens(
            payload,
            budget = budget,
            capacity = slots,
            context_window = budget,
            conversation = first_messages,
        )
        first_prompt = _openai_llama_admission_wire_prompt_tokens(
            first_messages, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        # The charge IS the first attempt's wire occupancy plus the reserve it never sends.
        assert charge == budget // slots
        assert first_prompt + allowance == charge - _RESERVE

        grown = first_messages + [
            # The whole allowance came back as an unparseable call.
            {"role": "assistant", "content": "blah " * allowance},
            {"role": "user", "content": "That was not a valid tool call. Try again."},
        ]
        retry_prompt = _openai_llama_admission_wire_prompt_tokens(
            grown, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        assert retry_prompt >= budget // slots, retry_prompt

        unbounded = _openai_llama_admission_retry_max_tokens(
            {"messages": grown},
            admission_output_allowance = allowance,
            request = None,
            llama_backend = backend,
        )
        # Without the first attempt to measure against, a fresh flat allowance.
        assert unbounded == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS - _RESERVE

        bound = _openai_llama_admission_retry_max_tokens(
            {"messages": grown},
            admission_output_allowance = allowance,
            request = None,
            llama_backend = backend,
            first_messages = first_messages,
        )
        growth = retry_prompt - first_prompt
        assert bound == max(1, allowance - growth)
        assert bound < unbounded
        # What the retry adds to the prompt it inherited is what the lease still holds.
        assert bound <= max(1, charge - retry_prompt)

    def test_a_retry_that_grew_a_little_keeps_the_rest_of_its_allowance(self):
        """An over-share prompt is charged prompt + the flat allowance, and a short
        malformed answer leaves most of it, so the retry occupies exactly the charge."""
        budget, slots = 16384, 4
        backend = _backend_stub(window = budget, total = budget, slots = slots)
        # Already past a 4096 share on the first attempt.
        first_messages = [{"role": "user", "content": "word " * 4000}]
        payload = _chat("word " * 4000, max_tokens = budget)
        allowance = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = first_messages
        )
        assert allowance == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS - _RESERVE
        charge = _openai_llama_admission_tokens(
            payload,
            budget = budget,
            capacity = slots,
            context_window = budget,
            conversation = first_messages,
        )
        grown = first_messages + [
            {"role": "assistant", "content": "blah " * 40},
            {"role": "user", "content": "Try again."},
        ]
        retry_prompt = _openai_llama_admission_wire_prompt_tokens(
            grown, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        bound = _openai_llama_admission_retry_max_tokens(
            {"messages": grown},
            admission_output_allowance = allowance,
            request = None,
            llama_backend = backend,
            first_messages = first_messages,
        )
        assert bound > 1
        assert retry_prompt + bound == charge - _RESERVE

    def test_a_client_that_named_a_cap_is_left_alone(self):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        grown = {"messages": [{"role": "user", "content": "word " * 500}], "max_tokens": 512}
        assert (
            _openai_llama_admission_retry_max_tokens(
                grown, admission_output_allowance = None, request = None, llama_backend = backend
            )
            is None
        )


class TestTheOpeningLeaseIsPricedOnTheProfiledPrompt:
    """The builders neutralise against the loaded model's profile, and so does the bound.
    Charging the generic sweep instead left a lease short of the wire by every profiled
    marker the prompt repeats, and a bound priced past its share falls to the flat
    allowance: cells the ledger never booked."""

    def test_the_charge_moves_with_the_profile_exactly_as_the_wire_does(self):
        from core.inference.chat_template_helpers import model_markup

        profile = model_markup("[ZETA] {{ m }}", ["[ZETA]"])
        conversation = [{"role": "user", "content": "[ZETA] " * 40 + "hello"}]
        payload = _Payload(messages = conversation, max_tokens = 64)
        wire = lambda markup: _openai_llama_admission_wire_prompt_tokens(
            conversation, markup = markup
        )
        charge = lambda markup: _openai_llama_admission_tokens(
            payload,
            budget = 65536,
            capacity = 1,
            context_window = 65536,
            conversation = conversation,
            markup = markup,
        )
        assert wire(profile) > wire(None), "the profile is what makes the marker cost words"
        assert charge(profile) - charge(None) == wire(profile) - wire(None)
        # The raw surfaces reserve from the payload alone, and their builders neutralise on
        # the same profile.
        raw = lambda markup: _openai_llama_admission_tokens(
            payload, budget = 65536, capacity = 1, context_window = 65536, markup = markup
        )
        assert raw(profile) > raw(None)

    def test_the_reservation_hands_the_backends_profile_over(self):
        import inspect

        from routes.inference import _openai_llama_admission_reserve

        source = inspect.getsource(_openai_llama_admission_reserve)
        assert "markup = _openai_llama_admission_markup(llama_backend)," in source


class TestTheOperatorSwitches:
    """Both switches turn the reservation itself off, so there is nothing to enforce."""

    @pytest.mark.parametrize("variable", [ADMISSION_CONTROL_ENV, ADMISSION_KV_BUDGET_ENV])
    def test_admission_off_leaves_generation_alone(self, monkeypatch, variable):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        assert (
            _openai_llama_admission_enforced_max_tokens(
                payload, request = None, llama_backend = backend
            )
            is not None
        )
        monkeypatch.setenv(variable, "off")
        assert (
            _openai_llama_admission_enforced_max_tokens(
                payload, request = None, llama_backend = backend
            )
            is None
        )


class TestThePassthroughSurface:
    def test_the_body_reads_capacity_from_the_request(self):
        """`effective_parallel_slots` is unset early, so without the request capacity is 1."""
        backend = SimpleNamespace(
            context_length = 16384,
            _kv_cache_context_total = 16384,
            effective_parallel_slots = None,
            markup_profile = None,
            _request_reasoning_kwargs = lambda *_a, **_k: None,
        )
        payload = ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16384,
                "stream": False,
            }
        )
        request = SimpleNamespace(
            app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 4))
        )

        blind = _build_openai_passthrough_body(payload, backend_ctx = 16384, llama_backend = backend)
        aware = _build_openai_passthrough_body(
            payload, backend_ctx = 16384, llama_backend = backend, request = request
        )
        assert blind["max_tokens"] == 16384
        assert aware["max_tokens"] <= 16384 // 4


_ANTHROPIC_KEY = "http://llama.enforced.test:9999"


class _AnthropicRequest:
    def __init__(self):
        self.state = SimpleNamespace()
        self.url = SimpleNamespace(path = "/v1/messages")
        self.method = "POST"
        self.app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 4))

    async def is_disconnected(self):
        return False


class TestTheAnthropicSurface:
    """/v1/messages shares the slots, and its optional `max_tokens` reads as unstated."""

    @pytest.fixture(autouse = True)
    def _isolate(self, monkeypatch):
        reset_llama_admission_queues()
        monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 64))
        monkeypatch.setattr(inf_mod, "_CANCEL_REGISTRY", {})
        yield
        reset_llama_admission_queues()

    def _install(
        self,
        monkeypatch,
        seen: dict,
        *,
        supports_tool_passthrough = False,
    ):
        def _gen_plain(**kwargs):
            seen["plain"] = kwargs
            yield "ok"

        def _gen_tools(**kwargs):
            seen["tools"] = kwargs
            yield {"type": "content", "text": "ok"}

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            supports_tool_passthrough = supports_tool_passthrough,
            model_identifier = "test-model",
            context_length = 16384,
            _kv_cache_context_total = 16384,
            effective_parallel_slots = 4,
            count_chat_tokens = lambda *a, **k: 2,
            generate_chat_completion = _gen_plain,
            generate_chat_completion_with_tools = _gen_tools,
            base_url = _ANTHROPIC_KEY,
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
        return backend

    def test_the_plain_generator_is_bounded(self, monkeypatch):
        seen: dict = {}
        self._install(monkeypatch, seen)
        payload = AnthropicMessagesRequest(
            max_tokens = 16384,
            messages = [{"role": "user", "content": "hi"}],
        )

        asyncio.run(anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t"))

        allowance = seen["plain"]["admission_output_allowance"]
        assert allowance is not None and allowance <= 16384 // 4

    def test_the_client_tool_passthrough_is_bounded(self, monkeypatch):
        """Returns through the passthrough builders, yet still takes a lease."""
        seen: dict = {}
        self._install(monkeypatch, seen, supports_tool_passthrough = True)
        captured: dict = {}

        async def _fake_passthrough(
            llama_backend,
            openai_messages,
            openai_tools,
            temperature,
            top_p,
            top_k,
            max_tokens,
            *args,
            **kwargs,
        ):
            captured["max_tokens"] = max_tokens
            captured["allowance"] = kwargs.get("admission_output_allowance")
            return JSONResponse(content = {"id": "msg_x"})

        monkeypatch.setattr(inf_mod, "_anthropic_passthrough_non_streaming", _fake_passthrough)
        payload = AnthropicMessagesRequest.model_validate(
            {
                "max_tokens": 16384,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "name": "web_search",
                        "description": "Search the web.",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            }
        )

        asyncio.run(anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t"))

        assert captured["max_tokens"] is not None
        assert captured["max_tokens"] <= 16384 // 4, captured
        assert captured["allowance"] == captured["max_tokens"]

    def test_the_plain_chat_is_reserved_from_the_finalized_prompt(self, monkeypatch):
        """Charged and permitted must be the same prompt, as the GGUF chat paths do.

        The date prompt is spliced in after the payload, so a raw prompt just under its
        share is sent at or above one, where the bound is the flat unstated allowance
        rather than `share - prompt`. Reserving from the raw payload charged one share for
        a request the wire lets write a share plus another 1024.
        """
        seen: dict = {}
        self._install(monkeypatch, seen)
        monkeypatch.setattr(
            inf_mod, "current_date_prompt_line", lambda *_a, **_k: "Today's date is 2026-09-08."
        )
        charged: list[int] = []
        _tokens = inf_mod._openai_llama_admission_tokens

        def _spy(payload, **kwargs):
            value = _tokens(payload, **kwargs)
            charged.append(value)
            return value

        monkeypatch.setattr(inf_mod, "_openai_llama_admission_tokens", _spy)
        share = 16384 // 4
        # Sized so the raw payload sits under its share and the finalized prompt over it.
        payload = AnthropicMessagesRequest(
            max_tokens = 16384,
            messages = [{"role": "user", "content": "word " * 3260}],
        )
        raw = _openai_llama_admission_prompt_tokens(payload)

        asyncio.run(anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t"))

        sent = seen["plain"]["messages"]
        prompt = _openai_llama_admission_wire_prompt_tokens(sent)
        allowance = seen["plain"]["admission_output_allowance"]
        assert raw < share <= prompt, (raw, share, prompt)
        assert allowance == _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS - _RESERVE, allowance
        assert charged and charged[0] >= prompt + allowance, (charged, prompt, allowance)

    def test_the_tool_generator_is_bounded(self, monkeypatch):
        seen: dict = {}
        self._install(monkeypatch, seen)
        # Unsloth's own server-side loop; a client catalogue takes the passthrough instead.
        payload = AnthropicMessagesRequest.model_validate(
            {
                "max_tokens": 16384,
                "messages": [{"role": "user", "content": "hi"}],
                "enable_tools": True,
                # The route requires an explicit permission mode to run server tools.
                "permission_mode": "off",
            }
        )

        asyncio.run(anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t"))

        allowance = seen["tools"]["admission_output_allowance"]
        assert allowance is not None and allowance <= 16384 // 4


class TestBothPassthroughsPriceTheirRetry:
    """The wire cap the nudge retry actually goes out with, on both routes.

    The arithmetic is proved above; this proves each passthrough hands the first
    attempt's messages over, since neither can reach a lease from where it retries.
    """

    _GARBAGE = "<tool_call>call lookup somehow???"
    _TOOL = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    @pytest.fixture(autouse = True)
    def _isolate(self, monkeypatch):
        reset_llama_admission_queues()
        monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 64))
        yield
        reset_llama_admission_queues()

    @staticmethod
    def _backend():
        return SimpleNamespace(
            base_url = "http://llama.test",
            context_length = 16384,
            _kv_cache_context_total = 16384,
            effective_parallel_slots = 4,
            _request_reasoning_kwargs = lambda *_a, **_k: None,
        )

    class _Scripted:
        def __init__(self, bodies):
            self.bodies = list(bodies)
            self.posts = []

        async def post(
            self,
            _url,
            json = None,
            timeout = None,
            headers = None,
        ):
            self.posts.append(json)
            return httpx.Response(
                200, json = self.bodies[min(len(self.posts) - 1, len(self.bodies) - 1)]
            )

        async def aclose(self):
            return None

    def _reply(self, content):
        return {
            "id": "chatcmpl-up",
            "object": "chat.completion",
            "created": 1,
            "model": "gguf",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    def _assert_retry_stays_inside_the_lease(self, backend, posts):
        assert len(posts) == 2, len(posts)
        first, retry = posts
        first_prompt = _openai_llama_admission_wire_prompt_tokens(
            first["messages"], image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        retry_prompt = _openai_llama_admission_wire_prompt_tokens(
            retry["messages"], image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        # Past its share, where the wire bound would otherwise hand out a fresh 1024.
        assert retry_prompt >= 16384 // 4, retry_prompt
        assert retry["max_tokens"] < _OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS
        growth = retry_prompt - first_prompt
        assert retry["max_tokens"] <= max(1, first["max_tokens"] - growth)

    def test_the_openai_passthrough_prices_its_retry(self, monkeypatch):
        backend = self._backend()
        payload = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "word " * 2400}],
            tools = [self._TOOL],
            max_tokens = 16384,
            nudge_tool_calls = True,
        )
        client = self._Scripted(
            [self._reply(self._GARBAGE + " blah" * 1200), self._reply(self._GARBAGE)]
        )
        monkeypatch.setattr(inf_mod, "nonstreaming_client", lambda: client)

        asyncio.run(
            inf_mod._openai_passthrough_non_streaming_upstream(
                backend, payload, "gguf", monitor_id = None
            )
        )
        self._assert_retry_stays_inside_the_lease(backend, client.posts)

    def test_the_anthropic_passthrough_prices_its_retry(self, monkeypatch):
        backend = self._backend()
        messages = [{"role": "user", "content": "word " * 2400}]
        payload = _Payload(messages = messages, max_tokens = 16384)
        allowance = _openai_llama_admission_enforced_max_tokens(
            payload,
            request = None,
            llama_backend = backend,
            conversation = messages,
            injected_tools = [self._TOOL],
        )
        assert allowance is not None
        client = self._Scripted(
            [self._reply(self._GARBAGE + " blah" * 1200), self._reply(self._GARBAGE)]
        )
        monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)

        asyncio.run(
            inf_mod._anthropic_passthrough_non_streaming(
                backend,
                messages,
                [self._TOOL],
                0.7,
                0.95,
                None,
                allowance,
                "msg_test",
                "gguf",
                nudge_tool_calls = True,
                admission_output_allowance = allowance,
            )
        )
        self._assert_retry_stays_inside_the_lease(backend, client.posts)


class TestTheLoopSizesAgainstTheAdmittedAllowance:
    """The bound the wire gets is the bound the fit has to reserve for.

    A tool round clamps its payload to the admitted share and then sized everything
    else -- the fit, the recall budget, every tool-result budget -- against the caller's
    whole cap. On eight slots that reserves eight times the room the request may ever
    emit, so history is evicted and results are cut to pay for output the lease already
    forbids. The final pass had its fit right with `_final_fit_max_tokens` and priced the
    recall beside it off the caller's cap; both ends now read the admitted figure.
    """

    _ALLOWANCE = 256

    @staticmethod
    def _budget_handed_to_the_tool(monkeypatch, allowance):
        """The `result_budget_tokens` one round hands a tool, through the real loop."""
        seen: list[int] = []

        def _fake_execute_tool(
            name,
            arguments,
            *,
            result_budget_tokens = None,
            **_kwargs,
        ):
            seen.append(result_budget_tokens)
            return "Linux kernel 6.10."

        monkeypatch.setattr("core.inference.tools.execute_tool", _fake_execute_tool)
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10"}), _done()],
            ],
            payloads,
        )
        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "Which kernel?"}],
                tools = [_TOOL],
                max_tool_iterations = 1,
                permission_mode = "off",
                max_tokens = _CTX,
                admission_output_allowance = allowance,
            )
        )
        assert seen, "the tool never ran, so nothing was priced"
        return seen[0]

    def test_a_result_is_priced_against_the_share_not_the_callers_cap(self, monkeypatch):
        """`tool_result_budget(ctx, 256, spent)`, not `tool_result_budget(ctx, 4096, spent)`."""
        capped = self._budget_handed_to_the_tool(monkeypatch, self._ALLOWANCE)
        unbounded = self._budget_handed_to_the_tool(monkeypatch, None)

        # Same conversation both runs, so the prompt spends the same; read that spend back
        # off the unbounded figure rather than re-deriving the loop's exact count.
        spent = tool_result_budget(_CTX, _CTX, 0) - unbounded
        assert capped == tool_result_budget(_CTX, self._ALLOWANCE, spent)
        assert unbounded == tool_result_budget(_CTX, _CTX, spent)
        assert capped > unbounded, "reserving a cap the wire cannot use starves the result"

    def test_an_allowance_at_or_above_the_cap_changes_nothing(self, monkeypatch):
        """The clamp is a `min`, so a lease roomier than the request leaves it alone: an
        unadmitted run and a generously admitted one price a result identically."""
        assert self._budget_handed_to_the_tool(
            monkeypatch, _CTX
        ) == self._budget_handed_to_the_tool(monkeypatch, None)

    @staticmethod
    def _final_pass_recall_cap(monkeypatch, allowance):
        """The cap the synthesized final pass prices its recall against."""
        seen: list[int] = []
        _real = llama_cpp_mod._retrieval_budget

        def _recorder(context_length, max_tokens, prompt_tokens, **kwargs):
            # The final pass is the only caller that leaves `reply_returns` unset: its
            # answer never returns to the prompt.
            if not kwargs.get("reply_returns"):
                seen.append(max_tokens)
            return _real(context_length, max_tokens, prompt_tokens, **kwargs)

        monkeypatch.setattr(llama_cpp_mod, "_retrieval_budget", _recorder)
        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [[_sse({"content": "6.10"}), _done()]], payloads)
        # The stub has no server to render a template with, and the fit evicts nothing
        # without a count it can trust.
        monkeypatch.setattr(
            backend,
            "count_chat_tokens",
            lambda messages, *_a, **_k: sum(
                len(str(message.get("content") or "")) for message in (messages or [])
            )
            // 4,
        )
        list(
            backend.generate_chat_completion_with_tools(
                messages = [
                    {"role": "user", "content": "word " * 4000},
                    {"role": "assistant", "content": "word " * 4000},
                    {"role": "user", "content": "Which kernel?"},
                ],
                tools = [_TOOL],
                # No round to run, so what is exercised is the synthesized final pass.
                max_tool_iterations = 0,
                permission_mode = "off",
                max_tokens = _CTX,
                context_overflow = "truncate_oldest",
                admission_output_allowance = allowance,
            )
        )
        assert seen, "the final pass never priced a recall"
        return seen[0]

    def test_the_final_pass_recall_is_priced_against_the_share(self, monkeypatch):
        """Its fit reserved the share while the recall beside it reserved the whole cap,
        which on a small window leaves the retrieval nothing to spend."""
        assert self._final_pass_recall_cap(monkeypatch, self._ALLOWANCE) == self._ALLOWANCE

    def test_the_final_pass_recall_is_left_alone_without_an_allowance(self, monkeypatch):
        """Nothing admitted, nothing clamped: the caller's cap, as before."""
        assert self._final_pass_recall_cap(monkeypatch, None) == _CTX


class TestARoundSizesAgainstWhatItsOwnReCostEarned:
    """The re-cost runs below the fit, so everything under it has this round's figure.

    The fit has to price against the previous round's -- the re-cost cannot run until the
    prompt it charges for exists. Every sizing decision AFTER it can, and the result
    budget, the recall budget and the reply-room gates were all still reading the figure
    the fit used. A round that opened with a roomy allowance and re-costed down to a
    narrow one then cut its tool result to reserve output the wire is no longer sending.
    """

    _OPENED = 1024
    _RECOSTED = 128

    @staticmethod
    def _round(monkeypatch, *, opened, recosted):
        """The result budget one round hands its tool, and the caps it put on the wire."""
        seen: list[int] = []

        def _fake_execute_tool(
            name,
            arguments,
            *,
            result_budget_tokens = None,
            **_kwargs,
        ):
            seen.append(result_budget_tokens)
            return "Linux kernel 6.10."

        monkeypatch.setattr("core.inference.tools.execute_tool", _fake_execute_tool)
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10"}), _done()],
            ],
            payloads,
        )
        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "Which kernel?"}],
                tools = [_TOOL],
                max_tool_iterations = 1,
                permission_mode = "off",
                max_tokens = _CTX,
                admission_output_allowance = opened,
                on_conversation_grew = (
                    None if recosted is None else lambda _conversation, _tools = None: recosted
                ),
            )
        )
        assert seen, "the tool never ran, so nothing was priced"
        return seen[0], _caps(payloads)

    def test_the_result_is_priced_against_the_cap_the_round_actually_sends(self, monkeypatch):
        """Re-costed down to 128, the round sends 128; the result must be sized for 128."""
        budget, caps = self._round(monkeypatch, opened = self._OPENED, recosted = self._RECOSTED)
        assert caps[0] == self._RECOSTED, caps
        opened_there, _caps_there = self._round(
            monkeypatch, opened = self._RECOSTED, recosted = self._RECOSTED
        )
        assert budget == opened_there, (budget, opened_there)

    def test_a_round_that_kept_its_allowance_is_unchanged(self, monkeypatch):
        """A re-cost that says nothing leaves the figure alone, so nothing else moves."""
        stale, caps = self._round(monkeypatch, opened = self._OPENED, recosted = None)
        assert caps[0] == self._OPENED, caps
        assert stale == self._round(monkeypatch, opened = self._OPENED, recosted = self._OPENED)[0]
        # A narrower cap reserves less reply room, so the result gets more of the window.
        assert self._round(monkeypatch, opened = self._OPENED, recosted = self._RECOSTED)[0] > stale


class TestTheSizingSitesReadTheClampedFigure:
    """Source-level, because a seventh sizing site added against the unclamped name is
    the same defect again and no single behaviour test sees all of them."""

    _SOURCE = " ".join(Path(llama_cpp_mod.__file__).read_text(encoding = "utf-8").split())

    def test_the_iteration_fit_receives_the_clamped_figure(self):
        assert "max_tokens = _iteration_fit_max_tokens," in self._SOURCE

    def test_no_fit_is_handed_the_unclamped_figure(self):
        assert "max_tokens = _iteration_max_tokens," not in self._SOURCE

    def test_the_clamp_is_the_share_and_falls_back_to_the_window(self):
        assert (
            "_iteration_fit_max_tokens = ( min( _iteration_max_tokens "
            "if _iteration_max_tokens is not None "
            "else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR), "
            "admission_output_allowance, ) "
            "if admission_output_allowance is not None else _iteration_max_tokens )"
        ) in self._SOURCE

    def test_the_final_pass_still_uses_its_own_clamp(self):
        assert "max_tokens = _final_fit_max_tokens," in self._SOURCE

    def test_the_wire_cap_is_still_clamped_on_its_own_path(self):
        """Sizing borrows the figure; it does not take over `payload["max_tokens"]`."""
        assert (
            'payload["max_tokens"] = min(payload["max_tokens"], admission_output_allowance)'
            in self._SOURCE
        )

    _SIZERS = ("_retrieval_budget", "prompt_budget", "tool_result_budget")
    _CLAMPED = ("_iteration_fit_max_tokens", "_final_fit_max_tokens")

    def _tool_loop(self):
        import ast

        tree = ast.parse(Path(llama_cpp_mod.__file__).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "generate_chat_completion_with_tools"
            ):
                return node
        raise AssertionError("the tool loop is gone")

    def test_no_sizing_call_in_the_loop_reads_an_unclamped_cap(self):
        """Every budget in the loop, rounds and final pass alike, prices against what the
        wire is allowed to emit. Read off the source because one behaviour test cannot
        reach all of them and a new site added against the raw cap is the same defect."""
        import ast

        unclamped = []
        for call in ast.walk(self._tool_loop()):
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "id", None)
            if name in self._SIZERS and len(call.args) >= 2:
                read = getattr(call.args[1], "id", None)
                if read not in self._CLAMPED:
                    unclamped.append(f"{name}({read})")
            if name == "_fit_with_instruction_pins":
                for keyword in call.keywords:
                    if keyword.arg != "max_tokens":
                        continue
                    # Bare names only. The continuation eviction helper fits to a computed
                    # reply FLOOR on purpose, and a site regressed to the raw cap would be
                    # written as a name.
                    if not isinstance(keyword.value, ast.Name):
                        continue
                    if keyword.value.id not in self._CLAMPED:
                        unclamped.append(f"{name}(max_tokens = {keyword.value.id})")
        assert not unclamped, f"these size against a cap the wire will not send: {unclamped}"

    def test_a_re_cost_that_moves_the_allowance_re_sizes_with_it(self):
        """The fit above a re-cost cannot price against this round's allowance; the re-cost
        needs the fitted prompt first. Everything below it can, so both re-costs rebuild
        the sizing figure, and the final pass has no behaviour test that reaches its
        respawn refit."""
        import ast

        resized: set = set()
        tree = ast.parse(Path(llama_cpp_mod.__file__).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            assigned = [
                target.id
                for statement in node.body
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                if isinstance(target, ast.Name)
            ]
            if "admission_output_allowance" in assigned:
                resized.update(name for name in assigned if name in self._CLAMPED)
        assert set(self._CLAMPED) <= resized, resized

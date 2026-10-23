# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""continue_final_message makes the model extend the cut partial, not restart it."""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from pathlib import Path

import httpx


def _shared_setup_1(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _done()]),
        payloads,
    )
    return backend, payloads


def _shared_setup_2(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()],
            [
                _sse({"reasoning_content": "Let me reconsider the whole approach. " * 60}),
                _finish("length"),
                _done(),
            ],
            [_sse({"content": "and here is the rest."}), _done()],
        ],
        payloads,
    )
    return backend, payloads


def _shared_setup_3(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()]
            for _ in range(_MAX_LENGTH_CONTINUATIONS + 3)
        ],
        payloads,
    )
    return backend, payloads


def _shared_setup_4(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": " never sent"}), _done()]),
        payloads,
    )
    return backend, payloads


def _shared_setup_5(backend, monkeypatch):
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda messages, *_a, **_k: sum(
            len(str(message.get("content", ""))) for message in messages
        )
        // 2,
    )


def _shared_setup_6(_respawned, backend, monkeypatch):
    monkeypatch.setattr(backend, "_respawn_if_dead", _respawned)

    healthy = backend._stream_with_retry
    calls = {"n": 0}
    return calls, healthy


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.context_window import _reply_floor
from core.inference.llama_cpp import (
    _CONTINUE_TRUNCATED_ANSWER_STATUS,
    _MAX_LENGTH_CONTINUATIONS,
    LlamaCppBackend,
)

_HALF_AN_ANSWER = (
    "<!DOCTYPE html>\n<html>\n<body>\n<canvas id='c'></canvas>\n<script>\n"
    # Varied on purpose: verbatim repeats trip the guard's repetition line rule.
    + "".join(
        f"  ctx.lineTo({i * 3}, {i * 7 % 31});\n  ctx.stroke(); // segment {i}\n" for i in range(40)
    )
    + "  ctx.arc(6, -5, 5, 0"
)


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _finish(reason: str) -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _make_backend(monkeypatch, streams: list[object], payloads: list[dict]):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48857
    backend._api_key = None
    backend._effective_context_length = 4096
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
        yield type("FakeResponse", (), {"status_code": 200, "chunks": streams.pop(0)})()

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


def _run(backend, **kwargs):
    kwargs.setdefault("max_tool_iterations", 4)
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Show me the HTML inline"}],
            tools = [_WEB_SEARCH_TOOL],
            **kwargs,
        )
    )


def _texts(events, kind: str) -> list[str]:
    return [event["text"] for event in events if event.get("type") == kind]


def _cut_off_then(*later: list[str]) -> list[list[str]]:
    return [
        [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()],
        *later,
    ]


def test_an_answer_cut_in_half_is_finished(monkeypatch):
    backend, payloads = _shared_setup_1(monkeypatch)

    events = _run(backend)

    assert len(payloads) == 2, "the answer was left mid-sentence"
    assert "</html>" in "".join(_texts(events, "content"))


def test_the_partial_goes_back_to_be_extended_not_responded_to(monkeypatch):
    """Without this the model restarts the answer instead of resuming it."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": " done"}), _done()]),
        payloads,
    )

    _run(backend)

    assert payloads[1].get("continue_final_message") is True
    assert payloads[1]["messages"][-1]["role"] == "assistant"
    assert payloads[1]["messages"][-1]["content"].endswith("ctx.arc(6, -5, 5, 0")


def test_the_continuation_is_announced(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": " done"}), _done()]),
        payloads,
    )

    statuses = _texts(_run(backend), "status")

    assert _CONTINUE_TRUNCATED_ANSWER_STATUS in statuses
    index = statuses.index(_CONTINUE_TRUNCATED_ANSWER_STATUS)
    assert index > 0 and statuses[index - 1] == ""


def test_an_echo_is_kept_as_is_rather_than_continued(monkeypatch):
    """An echoed repetition is kept as is: continuing it grew one reply to 60,698 chars."""

    echo = "The user wants to see the HTML inline, so I will show the file now.\n" * 40
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": echo}), _finish("length"), _done()]],
        payloads,
    )

    _run(backend)

    assert len(payloads) == 1, "an echo was continued instead of being left alone"


def test_continuation_is_capped(monkeypatch):
    backend, payloads = _shared_setup_3(monkeypatch)

    _run(backend)

    assert len(payloads) == _MAX_LENGTH_CONTINUATIONS + 1


def test_a_clean_stop_is_never_continued(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": _HALF_AN_ANSWER}), _finish("stop"), _done()]],
        payloads,
    )

    _run(backend)

    assert len(payloads) == 1


def test_the_partial_is_kept_when_it_never_converges(monkeypatch):
    """Giving up must not throw away the work already streamed to the user."""

    backend, payloads = _shared_setup_3(monkeypatch)

    content = "".join(_texts(_run(backend), "content"))

    assert "<!DOCTYPE html>" in content


def _run_no_tools(backend, **kwargs):
    """Drives the FINAL generation, the path taken once the tool loop is done."""
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Show me the HTML inline"}],
            tools = [],
            max_tool_iterations = 0,
            **kwargs,
        )
    )


def test_the_final_answer_is_continued_too(monkeypatch):
    """Final answers run after the loop breaks, so the in-loop continuation never reaches them."""

    backend, payloads = _shared_setup_1(monkeypatch)

    events = _run_no_tools(backend)

    assert len(payloads) == 2, "the final answer was left mid-sentence"
    assert payloads[1].get("continue_final_message") is True
    assert "</html>" in "".join(_texts(events, "content"))


def test_the_final_continuation_turns_the_generation_prompt_off(monkeypatch):
    """llama-server rejects a request carrying both flags."""

    backend, payloads = _shared_setup_1(monkeypatch)

    _run_no_tools(backend)

    assert len(payloads) == 2, "the final answer was left mid-sentence"
    assert payloads[1]["continue_final_message"] is True
    assert payloads[1]["add_generation_prompt"] is False
    assert "add_generation_prompt" not in payloads[0]


def test_a_respawn_refit_during_a_continuation_carries_the_partial(monkeypatch):
    """The refit must restore the partial absent from `conversation`."""

    backend, payloads = _shared_setup_1(monkeypatch)
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 64)

    def _respawned() -> bool:
        backend._effective_context_length = 2048
        return True

    calls, healthy = _shared_setup_6(_respawned, backend, monkeypatch)

    @contextlib.contextmanager
    def flaky_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            payloads.append(copy.deepcopy(args[2]))
            raise httpx.RemoteProtocolError("llama-server died before the headers")
        with healthy(*args, **kwargs) as response:
            yield response

    monkeypatch.setattr(backend, "_stream_with_retry", flaky_stream)

    _run_no_tools(backend, context_overflow = "truncate_oldest")

    assert calls["n"] == 3, "the continuation has to be opened, die, and be retried"
    replayed = payloads[-1]
    assert replayed["messages"][0]["role"] == "user"
    assert replayed["messages"][-1]["role"] == "assistant"
    assert "ctx.arc(6, -5, 5, 0" in replayed["messages"][-1]["content"]
    assert replayed["continue_final_message"] is True
    assert replayed["add_generation_prompt"] is False


def test_a_respawn_refit_prices_the_carried_partial(monkeypatch):
    """The replacement window must include the restored partial."""

    backend, payloads = _shared_setup_1(monkeypatch)

    def _count(messages, *_a, **_k) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages) // 2

    monkeypatch.setattr(backend, "count_chat_tokens", _count)

    def _respawned() -> bool:
        backend._effective_context_length = 1400
        return True

    calls, healthy = _shared_setup_6(_respawned, backend, monkeypatch)

    @contextlib.contextmanager
    def flaky_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            payloads.append(copy.deepcopy(args[2]))
            raise httpx.RemoteProtocolError("llama-server died before the headers")
        with healthy(*args, **kwargs) as response:
            yield response

    monkeypatch.setattr(backend, "_stream_with_retry", flaky_stream)

    history = [
        {"role": "user", "content": "Older question " + "x" * 400},
        {"role": "assistant", "content": "Older answer " + "y" * 400},
        {"role": "user", "content": "Show me the HTML inline"},
    ]
    list(
        backend.generate_chat_completion_with_tools(
            messages = history,
            tools = [],
            max_tool_iterations = 0,
            context_overflow = "truncate_oldest",
        )
    )

    replayed = payloads[-1]
    assert replayed["messages"][-1]["role"] == "assistant", "the partial still rides across"
    assert len(replayed["messages"]) == 2, "the older exchange was not evicted for the partial"
    assert _count(replayed["messages"]) + _reply_floor(1400) <= 1400
    assert replayed["continue_final_message"] is True
    assert replayed["add_generation_prompt"] is False


def test_a_respawn_refit_does_not_replay_a_caller_prefill_twice(monkeypatch):
    """A respawn refit must not duplicate a caller prefill."""

    prefill = "Here is the beginning of my answer: "
    backend, payloads = _shared_setup_1(monkeypatch)
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 64)

    def _respawned() -> bool:
        backend._effective_context_length = 2048
        return True

    calls, healthy = _shared_setup_6(_respawned, backend, monkeypatch)

    @contextlib.contextmanager
    def flaky_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            payloads.append(copy.deepcopy(args[2]))
            raise httpx.RemoteProtocolError("llama-server died before the headers")
        with healthy(*args, **kwargs) as response:
            yield response

    monkeypatch.setattr(backend, "_stream_with_retry", flaky_stream)

    prefilled = [
        {"role": "user", "content": "Show me the HTML inline"},
        {"role": "assistant", "content": prefill},
    ]
    list(
        backend.generate_chat_completion_with_tools(
            messages = prefilled,
            tools = [],
            max_tool_iterations = 0,
            continue_final_message = True,
            context_overflow = "truncate_oldest",
        )
    )

    replayed = payloads[-1]
    assert [message["role"] for message in replayed["messages"]] == ["user", "assistant"]
    assert json.dumps(replayed["messages"]).count(prefill) == 1
    assert replayed["messages"][-1]["content"].startswith(prefill)
    assert "ctx.arc(6, -5, 5, 0" in replayed["messages"][-1]["content"]


def test_a_respawn_refit_during_the_reasoning_recovery_keeps_its_request(monkeypatch):
    """A respawn refit must preserve the reasoning recovery tail."""

    backend, payloads = _shared_setup_2(monkeypatch)
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 64)

    def _respawned() -> bool:
        backend._effective_context_length = 2048
        return True

    calls, healthy = _shared_setup_6(_respawned, backend, monkeypatch)

    @contextlib.contextmanager
    def flaky_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            payloads.append(copy.deepcopy(args[2]))
            raise httpx.RemoteProtocolError("llama-server died before the headers")
        with healthy(*args, **kwargs) as response:
            yield response

    monkeypatch.setattr(backend, "_stream_with_retry", flaky_stream)

    _run_no_tools(backend, context_overflow = "truncate_oldest")

    replayed = payloads[-1]
    assert [message["role"] for message in replayed["messages"]] == [
        "user",
        "assistant",
        "user",
    ], "the refit dropped the recovery turns"
    assert "ctx.arc(6, -5, 5, 0" in replayed["messages"][1]["content"]
    assert "continue_final_message" not in replayed
    assert "add_generation_prompt" not in replayed


def test_the_recovery_is_declined_rather_than_sent_without_its_question(monkeypatch):
    """Decline a recovery that cannot retain the question it answers."""

    question = "QUESTION_MARKER show me the HTML inline <|im_end|>" + "q" * 200
    backend, payloads = _shared_setup_2(monkeypatch)
    _shared_setup_5(backend, monkeypatch)

    healthy = backend._stream_with_retry
    calls = {"n": 0}

    @contextlib.contextmanager
    def flaky_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            backend._effective_context_length = 320
        with healthy(*args, **kwargs) as response:
            yield response

    monkeypatch.setattr(backend, "_stream_with_retry", flaky_stream)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "PREFILL_MARKER here is the start: "},
            ],
            tools = [],
            max_tool_iterations = 0,
            continue_final_message = True,
            context_overflow = "truncate_oldest",
        )
    )

    assert calls["n"] == 2, "a recovery that cannot carry its question was sent anyway"
    for payload in payloads:
        roles = [message["role"] for message in payload["messages"]]
        assert not any(
            roles[index] == roles[index + 1] == "user" for index in range(len(roles) - 1)
        ), f"adjacent user turns in {roles}"
        assert json.dumps(payload["messages"]).count("QUESTION_MARKER") == 1
    assert "ctx.arc(6, -5, 5, 0" in "".join(_texts(events, "content"))


def test_an_older_exchange_is_still_evicted_to_admit_the_recovery(monkeypatch):
    """Older history remains evictable while the recovered turn is protected."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()],
            [_sse({"reasoning_content": "Let me reconsider. " * 40}), _finish("length"), _done()],
            [_sse({"content": "and the rest."}), _done()],
        ],
        payloads,
    )
    _shared_setup_5(backend, monkeypatch)

    healthy = backend._stream_with_retry
    calls = {"n": 0}

    @contextlib.contextmanager
    def flaky_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            backend._effective_context_length = 1800
        with healthy(*args, **kwargs) as response:
            yield response

    monkeypatch.setattr(backend, "_stream_with_retry", flaky_stream)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "OLD_MARKER what is the weather " + "w" * 3000},
                {"role": "assistant", "content": "it is sunny " + "s" * 3000},
                {"role": "user", "content": "QUESTION_MARKER draw the bird"},
            ],
            tools = [],
            max_tool_iterations = 0,
            context_overflow = "truncate_oldest",
        )
    )

    assert calls["n"] == 3, "the recovery was refused instead of dropping the old exchange"
    recovery = json.dumps(payloads[-1]["messages"])
    assert "OLD_MARKER" not in recovery, "the older exchange was held back too"
    assert "QUESTION_MARKER" in recovery


def test_a_refit_eviction_keeps_the_turn_the_recovery_is_recovering(monkeypatch):
    """Refit eviction must keep the original turn behind a recovery request."""

    question = "QUESTION_MARKER show me the HTML inline <|im_end|>" + "q" * 200
    backend, payloads = _shared_setup_2(monkeypatch)
    _shared_setup_5(backend, monkeypatch)

    def _respawned() -> bool:
        backend._effective_context_length = 300
        return True

    calls, healthy = _shared_setup_6(_respawned, backend, monkeypatch)

    @contextlib.contextmanager
    def flaky_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            payloads.append(copy.deepcopy(args[2]))
            raise httpx.RemoteProtocolError("llama-server died before the headers")
        with healthy(*args, **kwargs) as response:
            yield response

    monkeypatch.setattr(backend, "_stream_with_retry", flaky_stream)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": question}],
            tools = [],
            max_tool_iterations = 0,
            context_overflow = "truncate_oldest",
        )
    )

    replayed = json.dumps(payloads[-1]["messages"])
    assert "QUESTION_MARKER" in replayed, "the eviction dropped the question being answered"
    assert "ctx.arc(6, -5, 5, 0" in replayed, "the eviction dropped the partial being continued"


def test_the_final_continuation_is_capped(monkeypatch):
    backend, payloads = _shared_setup_3(monkeypatch)

    _run_no_tools(backend)

    assert len(payloads) == _MAX_LENGTH_CONTINUATIONS + 1


def test_a_final_echo_is_not_continued(monkeypatch):
    echo = "I will show the file now, here it is in full for you to read.\n" * 40
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": echo}), _finish("length"), _done()]],
        payloads,
    )

    _run_no_tools(backend)

    assert len(payloads) == 1


def test_a_clean_final_stop_is_never_continued(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": _HALF_AN_ANSWER}), _finish("stop"), _done()]],
        payloads,
    )

    _run_no_tools(backend)

    assert len(payloads) == 1


_SECOND_HALF = (
    ", 0, Math.PI * 2);\n  ctx.fill();\n"
    + "".join(
        f"  pipes[{i}].x -= speed * {i % 5 + 1};\n  if (pipes[{i}].x < -60) recycle({i});\n"
        for i in range(40)
    )
    + "  requestAnimationFrame(fr"
)


def _usage(completion_tokens: int) -> str:
    return (
        "data: "
        + json.dumps(
            {
                "choices": [{"index": 0, "delta": {}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": completion_tokens},
            }
        )
        + "\n"
    )


def _metadata(events) -> dict:
    return [event for event in events if event.get("type") == "metadata"][-1]


def test_the_final_continuation_replays_each_fragment_once(monkeypatch):
    """_append_assistant_turn prepends, so only text new since the last replay may be sent."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()],
            [_sse({"content": _SECOND_HALF}), _finish("length"), _done()],
            [_sse({"content": "ame);\n</script>\n</html>"}), _done()],
        ],
        payloads,
    )

    events = _run_no_tools(backend)

    assert len(payloads) == 3
    replayed = payloads[2]["messages"][-1]["content"]
    assert replayed == _HALF_AN_ANSWER + _SECOND_HALF
    assert replayed.count("<!DOCTYPE html>") == 1
    # Content events are CUMULATIVE here, so the last one is the whole answer.
    shown = _texts(events, "content")[-1]
    assert shown.count("<!DOCTYPE html>") == 1
    assert shown.endswith("</html>")


def test_usage_is_kept_across_final_continuations(monkeypatch):
    """Each attempt's usage overwrites the last, so usage must be summed across continuations."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _usage(700), _finish("length"), _done()],
            [_sse({"content": _SECOND_HALF}), _usage(500), _finish("length"), _done()],
            [_sse({"content": "ame);\n</script>\n</html>"}), _usage(30), _done()],
        ],
        payloads,
    )

    usage = _metadata(_run_no_tools(backend))["usage"]

    assert usage["completion_tokens"] == 700 + 500 + 30


def test_the_replayed_prefix_keeps_the_whitespace_it_was_cut_on(monkeypatch):
    """Replayed prefix keeps trailing whitespace, since the next delta is appended to streamed text."""

    cut_on_whitespace = _HALF_AN_ANSWER + "\n  "
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": cut_on_whitespace}), _finish("length"), _done()],
            [_sse({"content": "ctx.fill();\n</script>\n</html>"}), _done()],
        ],
        payloads,
    )

    _run(backend)

    assert payloads[1]["messages"][-1]["content"] == cut_on_whitespace


def test_a_continuation_that_would_be_rejected_is_not_sent(monkeypatch):
    """Skip a continuation whose prompt would overflow the context; llama-server rejects it outright."""

    backend, payloads = _shared_setup_4(monkeypatch)
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_args, **_kwargs: 4096)

    events = _run_no_tools(backend)

    assert len(payloads) == 1
    assert "<!DOCTYPE html>" in "".join(_texts(events, "content"))


def test_a_count_that_cannot_be_taken_is_not_a_refusal(monkeypatch):
    """Failing open restores what this path did before the check, which is the safe side."""

    backend, payloads = _shared_setup_1(monkeypatch)

    def _no_count(*_args, **_kwargs):
        raise RuntimeError("llama-server is not loaded")

    monkeypatch.setattr(backend, "count_chat_tokens", _no_count)

    events = _run_no_tools(backend)

    assert len(payloads) == 2
    assert "</html>" in "".join(_texts(events, "content"))


def test_a_continuation_with_room_to_answer_in_is_still_sent(monkeypatch):
    """The gate refuses only what llama-server refuses: a continuation needs no tool-result reserve."""

    backend, payloads = _shared_setup_1(monkeypatch)
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_args, **_kwargs: 3800)

    events = _run_no_tools(backend)

    assert len(payloads) == 2, "a continuation with room to answer in was refused"
    assert "</html>" in "".join(_texts(events, "content"))


def test_a_caller_set_max_tokens_is_not_exceeded(monkeypatch):
    """finish_reason length may be the caller's max_tokens cap; only a context stop may continue."""

    backend, payloads = _shared_setup_4(monkeypatch)

    _run_no_tools(backend, max_tokens = 100)

    assert len(payloads) == 1, "the caller's output cap was overrun"


def test_a_caller_cap_with_room_left_continues_within_it(monkeypatch):
    """Not a blanket refusal: the retry gets the REMAINDER of the caller's budget."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _usage(400), _finish("length"), _done()],
            [_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _usage(20), _done()],
        ],
        payloads,
    )

    events = _run_no_tools(backend, max_tokens = 1000)

    assert len(payloads) == 2
    assert payloads[1]["max_tokens"] == 600, "the retry got a fresh cap, not the remainder"
    assert "</html>" in "".join(_texts(events, "content"))


def test_max_tokens_equal_to_the_window_is_the_context_wall(monkeypatch):
    """That is what the backend substitutes for "Max", so it is not a caller cap."""

    backend, payloads = _shared_setup_1(monkeypatch)

    _run_no_tools(backend, max_tokens = 4096)

    assert len(payloads) == 2


def test_a_refused_continuation_does_not_double_count_its_usage(monkeypatch):
    """The fold ran before the decision, and the reject path reported the same tokens
    again through _build_metadata_event."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": _HALF_AN_ANSWER}), _usage(700), _finish("length"), _done()]],
        payloads,
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 4096)

    usage = _metadata(_run_no_tools(backend))["usage"]

    assert len(payloads) == 1
    assert usage["completion_tokens"] == 700, "the refused attempt was counted twice"


def test_the_in_loop_continuation_respects_the_caller_cap(monkeypatch):
    """The in-loop continuation path must also respect the caller's max_tokens cap."""

    backend, payloads = _shared_setup_4(monkeypatch)

    _run(backend, max_tokens = 100)

    assert len(payloads) == 1, "the caller's output cap was overrun in the loop"


def test_the_in_loop_continuation_spends_the_remainder(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _usage(400), _finish("length"), _done()],
            [_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _usage(20), _done()],
        ],
        payloads,
    )

    _run(backend, max_tokens = 1000)

    assert len(payloads) == 2
    assert payloads[1]["max_tokens"] == 600, "the retry got a fresh cap, not the remainder"


def test_an_in_loop_continuation_that_would_be_rejected_is_not_sent(monkeypatch):
    """Same guard the final pass has. With one user turn there is no history to evict."""

    backend, payloads = _shared_setup_4(monkeypatch)
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 4096)

    events = _run(backend)

    assert len(payloads) == 1
    assert "<!DOCTYPE html>" in "".join(_texts(events, "content")), "the partial was lost"


def test_an_in_loop_continuation_with_room_is_still_sent(monkeypatch):
    backend, payloads = _shared_setup_1(monkeypatch)
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 3800)

    events = _run(backend)

    assert len(payloads) == 2
    assert "</html>" in "".join(_texts(events, "content"))


def test_replayed_output_is_neutralized_before_it_is_sent(monkeypatch):
    """The replay is text the MODEL produced, and the first payload neutralized
    everything it carried. Sent raw, a template delimiter inside it -- printed code
    being the obvious case -- is read back as chat structure."""

    with_delimiter = _HALF_AN_ANSWER + "\nprint('<|im_end|>')\n"
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": with_delimiter}), _finish("length"), _done()],
            [_sse({"content": "done"}), _done()],
        ],
        payloads,
    )

    _run_no_tools(backend)

    assert len(payloads) == 2
    assert "<|im_end|>" not in json.dumps(payloads[1]["messages"])


def test_a_continuation_that_stalls_in_reasoning_is_not_read_as_more_answer(monkeypatch):
    """Judge a continuation by what it put on screen itself, not the cumulative counters."""

    backend, payloads = _shared_setup_2(monkeypatch)

    events = _run_no_tools(backend)

    assert len(payloads) == 3, "the stalled continuation was not recovered"
    # Ending on a USER turn distinguishes recovery from the answer continuation.
    assert payloads[2]["messages"][-1]["role"] == "user"
    assert "continue_final_message" not in payloads[2]
    assert "add_generation_prompt" not in payloads[2]


def test_an_answer_already_on_screen_is_not_replaced_by_the_explanation(monkeypatch):
    """A give-up message must not replace a written answer, as content events here are cumulative."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()],
            [
                _sse({"reasoning_content": "Let me reconsider the whole approach. " * 60}),
                _finish("length"),
                _done(),
            ],
            [
                _sse({"reasoning_content": "Still thinking about it. " * 60}),
                _finish("length"),
                _done(),
            ],
            [
                _sse({"reasoning_content": "And again. " * 60}),
                _finish("length"),
                _done(),
            ],
        ],
        payloads,
    )

    events = _run_no_tools(backend)
    texts = [event["text"] for event in events if event.get("type") == "content"]

    assert texts, "the turn ended showing nothing"
    assert _HALF_AN_ANSWER[:40] in texts[-1], "the written answer was overwritten"


def test_the_in_loop_answer_continuation_is_priced_as_a_continuation(monkeypatch):
    """Admission must price an in-loop continuation as sent, with continue_final_message."""

    seen: list[object] = []
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": _SECOND_HALF}), _done()]),
        payloads,
    )

    real_count = backend.count_chat_tokens

    def recording_count(*args, **kwargs):
        seen.append(kwargs.get("continue_final_message"))
        return real_count(*args, **kwargs)

    monkeypatch.setattr(backend, "count_chat_tokens", recording_count)

    _run(backend)

    assert len(payloads) == 2, "the continuation was refused"
    assert payloads[1].get("continue_final_message") is True
    assert True in seen, "the continuation was admitted as an ordinary prompt"


def test_an_attempt_that_reports_no_usage_is_not_charged_the_previous_one(monkeypatch):
    """Reset usage and timing per continuation, or a silent attempt is charged the previous numbers."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _usage(700), _finish("length"), _done()],
            # No usage chunk at all, which llama-server omits on some builds.
            [_sse({"content": _SECOND_HALF}), _done()],
        ],
        payloads,
    )

    usage = _metadata(_run_no_tools(backend))["usage"]

    assert (
        usage["completion_tokens"] == 700
    ), f"the first attempt's 700 tokens were counted twice: {usage['completion_tokens']}"


def test_a_final_continuation_does_not_reset_the_route_cursor(monkeypatch):
    """An empty status clears prev_text; keep cumulative across continuations or the prefix repeats."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": _SECOND_HALF}), _done()]),
        payloads,
    )

    events = _run_no_tools(backend)

    assert len(payloads) == 2, "the answer was not continued"
    # Only what happens AFTER text is on screen matters: a blank status before the
    # first content event resets a cursor that is already empty.
    kinds = [(e.get("type"), e.get("text")) for e in events]
    first_content = next(i for i, (kind, _) in enumerate(kinds) if kind == "content")
    after = [text for kind, text in kinds[first_content:] if kind == "status"]
    assert after, "the retry was not announced at all"
    assert "" not in after, f"an iteration-boundary reset was emitted mid-answer: {after}"


def test_a_resumed_turn_that_calls_a_tool_stays_one_assistant_message(monkeypatch):
    """Keep continue_final_message through tool calls, or a resumed turn gets two assistant messages."""

    seen: list[list] = []
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()],
            [
                _sse(
                    {
                        "content": '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
                    }
                ),
                _done(),
            ],
            [_sse({"content": "Done."}), _done()],
        ],
        payloads,
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "a result")

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Show me the HTML inline"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 3,
        )
    )

    assert len(payloads) >= 3, "the tool round never happened"
    messages = payloads[-1]["messages"]
    roles = [m.get("role") for m in messages]
    for first, second in zip(roles, roles[1:]):
        assert not (
            first == "assistant" and second == "assistant"
        ), f"two assistant turns in a row: {roles}"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool that keeps returning the same answer must not be allowed to eat the turn.

Observed at a 4096 window, asked to show a 2401-byte file inline. `tool_result_budget`
collapsed to zero, so every read returned only the notice saying it had been cut:

    tool result: name=terminal budget_tokens=0 chars=109      (six of the last eight)

The model read that as a fresh failure and tried again, varying the line range each time,
for eighteen calls. Two things were missing. The budget was never rescued, though room is
exactly what compaction reclaims; and nothing noticed that the answer had stopped changing.

The guard is keyed on the RESULT, not the arguments, which is the whole point here: the
arguments differed on every one of those calls. OpenClaw's tool-loop detection keys on the
result for the same reason, and stays quiet while results are still changing so that
legitimate polling is untouched.
"""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import core.inference.llama_cpp as llama_cpp_module
from core.inference.llama_cpp import _MAX_IDENTICAL_TOOL_RESULTS, LlamaCppBackend

_TRUNCATION_NOTICE = "(truncated to 0 chars for the model; showing lines 1-11 of 63.)"


def _finish(reason: str) -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
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


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _done() -> str:
    return "data: [DONE]\n"


def _call(query: str, index: int = 0) -> str:
    return _sse(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call_{index}",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": query}),
                    },
                }
            ]
        }
    )


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
    backend._port = 48853
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
    kwargs.setdefault("max_tool_iterations", 12)
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Show me the HTML inline"}],
            tools = [_WEB_SEARCH_TOOL],
            **kwargs,
        )
    )


def _tool_results(events: list[dict]) -> list[str]:
    return [e.get("result", "") for e in events if e.get("type") == "tool_end"]


def test_a_tool_repeating_one_answer_is_told_so(monkeypatch):
    """The arguments vary every time, so only the RESULT can reveal the dead end."""

    streams = [[_call(f"attempt {i}", i), _done()] for i in range(_MAX_IDENTICAL_TOOL_RESULTS)]
    streams.append([_sse({"content": "I will work from what I have."}), _done()])
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: _TRUNCATION_NOTICE)

    results = _tool_results(_run(backend))

    assert any("it will not change" in r for r in results)
    # The notice that caused the repeats is kept: replacing it would leave the model
    # holding less than it already had.
    assert any(_TRUNCATION_NOTICE in r for r in results)


def test_the_run_is_not_stopped_only_the_model_is_told(monkeypatch):
    """Hard-stopping a turn that is otherwise healthy trades one dead end for a worse one."""

    streams = [[_call(f"attempt {i}", i), _done()] for i in range(_MAX_IDENTICAL_TOOL_RESULTS)]
    streams.append([_sse({"content": "Working from what I have."}), _done()])
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: _TRUNCATION_NOTICE)

    events = _run(backend)

    texts = "".join(e["text"] for e in events if e.get("type") == "content")
    assert "Working from what I have." in texts


def test_changing_results_are_never_interrupted(monkeypatch):
    """Polling is the case a result-keyed guard has to leave alone."""

    streams = [[_call(f"attempt {i}", i), _done()] for i in range(_MAX_IDENTICAL_TOOL_RESULTS + 2)]
    streams.append([_sse({"content": "Done."}), _done()])
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    _seq = iter(range(100))
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: f"still running, tick {next(_seq)}",
    )

    results = _tool_results(_run(backend))

    assert results, "no tool ran"
    assert not any("it will not change" in r for r in results)


def _thread_with_a_big_completed_call(body_chars: int = 9000) -> list[dict]:
    """A finished edit_file whose arguments are still being replayed in full."""

    body = "<div>x</div>" * (body_chars // 12)
    return [
        {"role": "user", "content": "Create a Flappy Bird game in HTML"},
        {
            "role": "assistant",
            "content": "Writing the file.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps(
                            {
                                "path": "flappy-bird.html",
                                "edits": [{"old_string": "", "new_string": body}],
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "edit_file",
            "content": f"Wrote {len(body)} chars to flappy-bird.html",
        },
        {"role": "user", "content": "Show me the HTML inline"},
    ]


def test_a_tool_is_not_priced_at_zero_behind_a_finished_call(monkeypatch):
    """A call priced at zero can only ever return the notice saying it returned nothing.

    Scope, stated because the name could promise more: this pins the PRICING, not the
    compaction rescue that backs it up. The rescue re-counts the prompt with the real
    tokenizer, and this harness has no llama-server to render a template, so the rescue
    bails out here by design. It is covered by the live run at a 4096 window, where the
    log line `Result budget for X was 0; compacted N completed call(s) and it is now M`
    is the evidence.
    """

    received: list[object] = []

    def _record(*_args, **kwargs):
        received.append(kwargs.get("result_budget_tokens"))
        return "the file contents"

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_call("read it"), _done()], [_sse({"content": "Here it is."}), _done()]],
        payloads,
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", _record)

    list(
        backend.generate_chat_completion_with_tools(
            messages = _thread_with_a_big_completed_call(),
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 4,
        )
    )

    assert received, "the tool never ran"
    budget = received[0]
    if budget is not None:
        assert (
            budget > 0
        ), "the call was priced at zero, so it could only ever return a truncation notice"


def test_a_repeat_that_stops_repeating_resets(monkeypatch):
    """Two identical answers either side of a different one are not a dead end."""

    streams = [[_call(f"attempt {i}", i), _done()] for i in range(4)]
    streams.append([_sse({"content": "Done."}), _done()])
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    _answers = iter(["same", "same", "different", "same"])
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: next(_answers))

    results = _tool_results(_run(backend))

    assert not any("it will not change" in r for r in results)


def test_distinct_calls_answered_with_the_same_acknowledgement_are_left_alone(monkeypatch):
    """A generic `OK` is not a dead end, and the nudge would talk the model out of the
    work it has left.

    Some tools answer every distinct mutation with the same short string. Keyed on the
    result alone, three successful writes to three different records read as one answer
    repeated, and the model is then told that different arguments will not change it.
    The window's OWN notices keep the result-only key, which is the case this guard was
    built for and is covered above.
    """

    streams = [[_call(f"record-{i}", i), _done()] for i in range(_MAX_IDENTICAL_TOOL_RESULTS + 1)]
    streams.append([_sse({"content": "All three updated."}), _done()])
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "OK")

    results = _tool_results(_run(backend))

    assert results, "no tool ran"
    assert not any("it will not change" in r for r in results)


def _starve_the_budget(monkeypatch):
    """Force every result budget under _MIN_USEFUL_RESULT_TOKENS, as a tight window does."""
    import core.inference.llama_cpp as _lc  # noqa: PLC0415
    monkeypatch.setattr(_lc, "tool_result_budget", lambda *_a, **_k: 0)


def test_a_short_result_that_fit_is_not_called_starved(monkeypatch):
    """The budget says what the window ALLOWED, not what the tool returned.

    "Created a.py" fits a few tokens completely. Telling the model it got nothing usable
    and to continue without it invites it to discard a successful write, or do it twice.
    """
    _starve_the_budget(monkeypatch)
    streams = [[_call("make the file", 0), _done()], [_sse({"content": "Done."}), _done()]]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "Created a.py")

    results = _tool_results(_run(backend))

    assert any("Created a.py" in r for r in results)
    assert not any("nothing usable" in r or "without it" in r for r in results)


def test_a_result_the_window_actually_cut_is_still_called_starved(monkeypatch):
    """The case the nudge exists for must survive the new evidence requirement."""
    _starve_the_budget(monkeypatch)
    streams = [[_call("read the file", 0), _done()], [_sse({"content": "Done."}), _done()]]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: _TRUNCATION_NOTICE)

    results = _tool_results(_run(backend))

    assert any(_TRUNCATION_NOTICE in r for r in results)
    assert any(r != _TRUNCATION_NOTICE for r in results), "the nudge was not added"


def test_the_budget_rescue_recounts_with_the_stand_in_reply_too(monkeypatch):
    """Both counts have to price the SAME prompt, or the rescue gives away real room.

    The initial sizing appends an empty `tool` stand-in because Qwen-style templates render
    an assistant tool call only once a reply follows it. The rescue re-count after
    compaction did not, so on those templates this call's own arguments dropped out of the
    total and the room they occupy was handed to the result -- the exact overcount the
    stand-in exists to prevent, reintroduced on the path that was meant to fix it.
    """

    counted: list[list] = []

    def fake_count(messages, *_args, **_kwargs):
        counted.append(list(messages))
        # Under the prompt budget, so the pre-execution fit leaves the calls alone and
        # there is still something for the rescue to compact, but close enough to it that
        # the result prices under _MIN_USEFUL_RESULT_TOKENS, which is what triggers it.
        return 3050

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_call("read it"), _done()], [_sse({"content": "Here it is."}), _done()]],
        payloads,
    )
    monkeypatch.setattr(backend, "count_chat_tokens", fake_count)
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "contents")

    rescued: list[int] = []
    real_compact = llama_cpp_module.compact_completed_tool_arguments

    def spy_compact(messages, *args, **kwargs):
        fitted, n = real_compact(messages, *args, **kwargs)
        if kwargs.get("protect_last") and n:
            rescued.append(n)
        return fitted, n

    monkeypatch.setattr(llama_cpp_module, "compact_completed_tool_arguments", spy_compact)

    # Two finished calls, because the rescue protects the newest one: with a single
    # completed call there is nothing left for it to compact and it never re-counts.
    _thread = _thread_with_a_big_completed_call()
    _older = copy.deepcopy(_thread[1:3])
    _older[0]["tool_calls"][0]["id"] = "c0"
    _older[1]["tool_call_id"] = "c0"

    list(
        backend.generate_chat_completion_with_tools(
            messages = [_thread[0], *_older, *_thread[1:]],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 4,
        )
    )

    assert rescued, "the rescue never ran, so this asserts nothing"
    ends_on_the_call = [
        messages
        for messages in counted
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls")
    ]
    assert (
        not ends_on_the_call
    ), "a prompt was priced with the pending call's own arguments rendered away"


def test_the_zero_room_stub_counts_as_a_window_notice(monkeypatch):
    """At a budget of zero there is no truncated body to append a notice to.

    `_truncate` returns `_zero_room_stub` instead, whose text carries neither the
    truncation marker nor any of the result. Missing it is exactly the case this
    classification exists for: the nudge is skipped and the no-progress key falls back to
    including the arguments, so a model reading one file in different slices gets the
    same empty stub forever without ever being told why.
    """
    from core.inference.tools import _zero_room_stub

    stub = _zero_room_stub(2401, None, True)
    assert "chars for the model;" not in stub, "fixture no longer exercises the gap"

    _starve_the_budget(monkeypatch)
    streams = [[_call("read the file", 0), _done()], [_sse({"content": "Done."}), _done()]]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: stub)

    results = _tool_results(_run(backend))

    assert any(stub in r for r in results)
    assert any(r != stub for r in results), "the starved-result nudge was not added"


def test_a_resumed_turn_prices_its_tool_result_by_what_is_left(monkeypatch):
    """The payload used the continuation's remainder; this budget still used the whole cap.

    With 100 of 1000 tokens left, the result was priced as if 1000 were still to come, so
    `tool_result_budget` reserved room the request was never going to use and could hand
    the call a zero budget -- a starvation notice for a read there was space for.
    """

    caps: list[object] = []
    import core.inference.llama_cpp as _lc  # noqa: PLC0415

    real_budget = _lc.tool_result_budget

    def recording_budget(context_length, max_tokens, spent):
        caps.append(max_tokens)
        return real_budget(context_length, max_tokens, spent)

    monkeypatch.setattr(_lc, "tool_result_budget", recording_budget)

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": "Half an answer"}), _usage(900), _finish("length"), _done()],
            [_call("read it", 0), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
        payloads,
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "contents")

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Show me the file"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 3,
            max_tokens = 1000,
        )
    )

    assert caps, "the result was never priced"
    assert 1000 not in caps, f"a resumed turn priced its result against the whole cap: {caps}"


def test_a_resumed_turn_sizes_its_recall_by_what_is_left(monkeypatch):
    """`retrieval_budget` reserves the output allowance before handing back recall room.

    Reserving the caller's whole cap on a continuation that has a fraction of it left
    returns a near-zero budget, so `search_conversation` drops context the request had
    ample room for.
    """

    caps: list[object] = []
    import core.inference.llama_cpp as _lc  # noqa: PLC0415

    real_budget = _lc._retrieval_budget

    def recording_budget(context_length, max_tokens, spent, **kwargs):
        caps.append(max_tokens)
        return real_budget(context_length, max_tokens, spent, **kwargs)

    monkeypatch.setattr(_lc, "_retrieval_budget", recording_budget)

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": "Half an answer"}), _usage(900), _finish("length"), _done()],
            [_call("read it", 0), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
        payloads,
    )

    def _accepts_everything(*_a, **_k):
        return "contents"

    _accepts_everything.__signature__ = None
    monkeypatch.setattr("core.inference.tools.execute_tool", _accepts_everything)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Show me the file"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 3,
            max_tokens = 1000,
        )
    )

    assert 1000 not in caps, f"a resumed turn sized its recall against the whole cap: {caps}"

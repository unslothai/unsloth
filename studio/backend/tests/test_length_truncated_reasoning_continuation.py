# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A turn that exhausts the window on reasoning resumes with thinking off, not an empty message."""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import (
    _CONTINUE_AFTER_LENGTH_STATUS,
    _MAX_LENGTH_CONTINUATIONS,
    LlamaCppBackend,
)

_LONG_THOUGHT = "I should write the game. " * 200


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


def _make_backend(monkeypatch, streams: list[object], payloads: list[dict]):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48851
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = True
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


def _run(backend, **kwargs):
    kwargs.setdefault("max_tool_iterations", 3)
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Create a Flappy Bird game in HTML"}],
            tools = [_WEB_SEARCH_TOOL],
            enable_thinking = True,
            **kwargs,
        )
    )


def _texts(events, kind: str) -> list[str]:
    return [event["text"] for event in events if event.get("type") == kind]


def _run_no_tools(backend, **kwargs):
    """Drives the FINAL generation, the pass taken once the tool loop is done."""
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Create a Flappy Bird game in HTML"}],
            tools = [],
            max_tool_iterations = 0,
            enable_thinking = True,
            **kwargs,
        )
    )


def _truncated_thought_then(*later: list[str]) -> list[list[str]]:
    return [
        [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()],
        *later,
    ]


def test_a_thought_that_filled_the_window_is_continued_not_abandoned(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _truncated_thought_then([_sse({"content": "Here is the game."}), _done()]),
        payloads,
    )

    events = _run(backend)

    assert len(payloads) == 2, "the turn was abandoned instead of continued"
    assert "Here is the game." in "".join(_texts(events, "content"))


def test_the_continuation_turns_thinking_off(monkeypatch):
    """Retrying with thinking on just re-runs the turn that failed."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _truncated_thought_then([_sse({"content": "Here is the game."}), _done()]),
        payloads,
    )

    _run(backend)

    assert payloads[0]["chat_template_kwargs"]["enable_thinking"] is True
    assert payloads[1]["chat_template_kwargs"]["enable_thinking"] is False


def test_the_continuation_carries_progress_without_replaying_the_whole_thought(monkeypatch):
    """Putting the thought back reproduces the ending that made it necessary."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _truncated_thought_then([_sse({"content": "Here is the game."}), _done()]),
        payloads,
    )

    _run(backend)

    resumed = json.dumps(payloads[1]["messages"])
    assert "Where I had got to:" in resumed
    assert "ran out of room while thinking" in resumed
    assert len(resumed) < len(_LONG_THOUGHT), "the whole thought was replayed"


def test_the_retry_is_announced_so_the_ui_is_not_a_hang(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _truncated_thought_then([_sse({"content": "Here is the game."}), _done()]),
        payloads,
    )

    statuses = _texts(_run(backend), "status")

    assert _CONTINUE_AFTER_LENGTH_STATUS in statuses
    index = statuses.index(_CONTINUE_AFTER_LENGTH_STATUS)
    # Blank first: the route resets its text cursor only on an empty status.
    assert index > 0 and statuses[index - 1] == ""


def test_continuation_is_capped_so_a_small_window_cannot_loop(monkeypatch):
    """If thinking-off still produces nothing, the window is too small. Stop trying."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()]
            for _ in range(_MAX_LENGTH_CONTINUATIONS + 3)
        ],
        payloads,
    )

    _run(backend)

    assert len(payloads) == _MAX_LENGTH_CONTINUATIONS + 1


def test_giving_up_says_so_instead_of_returning_an_empty_turn(monkeypatch):
    """Giving up must name the lever (effort, window, task size), never return an empty turn silently."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()]
            for _ in range(_MAX_LENGTH_CONTINUATIONS + 2)
        ],
        payloads,
    )

    content = "".join(_texts(_run(backend), "content"))

    assert "reasoning" in content
    assert "4096-token window" in content
    assert "Lower the reasoning effort" in content


def test_a_good_tool_round_restores_the_full_allowance(monkeypatch):
    """A good tool round restores the continuation allowance, so a later stall gets full retries."""

    payloads: list[dict] = []
    _truncated = [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()]
    _calls_a_tool = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_0",
                        "function": {"name": "web_search", "arguments": '{"query":"x"}'},
                    }
                ]
            }
        ),
        _done(),
    ]
    backend = _make_backend(
        monkeypatch,
        [_truncated, _calls_a_tool, _truncated, _truncated, _truncated],
        payloads,
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "results")

    _run(backend, max_tool_iterations = 8)

    assert len(payloads) == 5


def test_thinking_comes_back_on_after_a_good_tool_round(monkeypatch):
    """It was turned off to break one stall, not for the rest of the request."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()],
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_0",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query":"x"}',
                                },
                            }
                        ]
                    }
                ),
                _done(),
            ],
            [_sse({"content": "Done."}), _done()],
        ],
        payloads,
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "results")

    _run(backend, max_tool_iterations = 8)

    assert payloads[1]["chat_template_kwargs"]["enable_thinking"] is False
    assert payloads[2]["chat_template_kwargs"]["enable_thinking"] is True


def test_a_turn_that_answers_is_handled_as_an_answer_not_a_stalled_thought(monkeypatch):
    """Only an empty length stop triggers this path; a truncated answer resumes elsewhere, thinking kept."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                _sse({"reasoning_content": "Briefly."}),
                _sse({"content": "The first half of the answer"}),
                _finish("length"),
                _done(),
            ],
            [_sse({"content": " and the second half."}), _done()],
        ],
        payloads,
    )

    events = _run(backend)

    assert len(payloads) == 2
    assert payloads[1].get("continue_final_message") is True
    assert payloads[1]["chat_template_kwargs"]["enable_thinking"] is True
    assert "The first half of the answer" in "".join(_texts(events, "content"))


def test_a_clean_reasoning_only_stop_is_left_alone(monkeypatch):
    """A thought that ENDED is promoted as the answer; only a cut-off one is resumed."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"reasoning_content": "The answer is 4."}), _finish("stop"), _done()]],
        payloads,
    )

    _run(backend)

    assert len(payloads) == 1


def test_the_final_pass_continues_a_reasoning_only_stop(monkeypatch):
    """The final pass runs after the loop, so the in-loop continuation never reaches it."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()],
            [_sse({"content": "Here is the answer."}), _done()],
        ],
        payloads,
    )

    events = _run_no_tools(backend)

    assert len(payloads) == 2, "the final pass returned an empty message"
    assert payloads[1]["messages"][-1]["role"] == "user"
    assert payloads[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "Here is the answer." in "".join(_texts(events, "content"))


def test_the_final_pass_says_so_when_thinking_never_converges(monkeypatch):
    """Giving up silently is the original defect. The user needs something to act on."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()]
            for _ in range(_MAX_LENGTH_CONTINUATIONS + 2)
        ],
        payloads,
    )

    events = _run_no_tools(backend)

    assert len(payloads) == _MAX_LENGTH_CONTINUATIONS + 1
    assert "".join(_texts(events, "content")).strip(), "the turn ended showing nothing"


def test_a_final_pass_that_answers_is_left_alone(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                _sse({"reasoning_content": _LONG_THOUGHT}),
                _sse({"content": "Done."}),
                _finish("stop"),
                _done(),
            ]
        ],
        payloads,
    )

    _run_no_tools(backend)

    assert len(payloads) == 1


def _effort_backend(monkeypatch, streams, payloads):
    backend = _make_backend(monkeypatch, streams, payloads)
    backend._supports_reasoning = True
    backend._reasoning_style = "reasoning_effort"
    return backend


def test_an_explicit_effort_does_not_survive_the_continuation(monkeypatch):
    """An explicit effort overrides enable_thinking here, so the retry must clear the effort too."""

    payloads: list[dict] = []
    backend = _effort_backend(
        monkeypatch,
        _truncated_thought_then([_sse({"content": "Here is the game."}), _done()]),
        payloads,
    )

    _run(backend, reasoning_effort = "high")

    assert payloads[0]["chat_template_kwargs"] == {"reasoning_effort": "high"}
    # "low", not "none": some models in this style cannot actually disable reasoning.
    assert payloads[1]["chat_template_kwargs"] == {"reasoning_effort": "low"}


def test_the_caller_effort_comes_back_once_a_turn_gets_somewhere(monkeypatch):
    """It was dropped to break ONE stall, not for the rest of the request."""

    payloads: list[dict] = []
    backend = _effort_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()],
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": json.dumps({"query": "flappy bird"}),
                                },
                            }
                        ]
                    }
                ),
                _done(),
            ],
            [_sse({"content": "Here is the game."}), _done()],
        ],
        payloads,
    )

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "a result",
    )

    _run(backend, reasoning_effort = "high")

    assert payloads[1]["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payloads[2]["chat_template_kwargs"] == {"reasoning_effort": "high"}


def _tool_call_sse(index: int) -> str:
    return _sse(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": f"q{index}"}),
                    },
                }
            ]
        }
    )


def test_a_request_that_never_stalls_keeps_the_bound_it_always_had(monkeypatch):
    """Continuation credit is granted on use, not reserved, so requests that never stall are unchanged."""

    streams = [[_tool_call_sse(i), _done()] for i in range(6)]
    streams.append([_sse({"content": "Done."}), _done()])
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "a result",
    )

    _run(backend, max_tool_iterations = 3)

    assert len(payloads) == 4


def test_a_stall_does_not_eat_the_tool_budget(monkeypatch):
    """Stall retries must not spend the real tool budget, or the requested action is never performed."""

    streams = [
        [_sse({"content": "I will search for the prices now."}), _done()],
        [_sse({"content": "I am going to look that up for you."}), _done()],
        [_sse({"content": "Let me check the current listings."}), _done()],
        [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()],
        [_sse({"reasoning_content": _LONG_THOUGHT + " more"}), _finish("length"), _done()],
        [_tool_call_sse(0), _done()],
        [_sse({"content": "Done."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    calls: list[str] = []

    def _execute(name, arguments, **_kwargs):
        calls.append(name)
        return "a result"

    monkeypatch.setattr("core.inference.tools.execute_tool", _execute)

    _run(backend, max_tool_iterations = 1, nudge_tool_calls = True)

    assert calls == ["web_search"], "the stall spent the one tool iteration"


def test_the_final_pass_blames_the_cap_when_the_cap_is_what_was_spent(monkeypatch):
    """The final pass must blame Max Tokens when that cap was spent, not the context window."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()]],
        payloads,
    )

    events = _run_no_tools(backend, max_tokens = 200)

    assert len(payloads) == 1, "a spent cap has nothing left to continue with"
    text = "".join(_texts(events, "content"))
    assert "output allowance of 200 tokens" in text
    assert "window on reasoning" not in text


def test_the_final_pass_still_blames_the_window_when_no_cap_was_set(monkeypatch):
    """The other side of the same fork, so the fix above cannot swallow the window case."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()]
            for _ in range(_MAX_LENGTH_CONTINUATIONS + 2)
        ],
        payloads,
    )

    events = _run_no_tools(backend)

    text = "".join(_texts(events, "content"))
    assert "4096-token window on reasoning" in text
    assert "output allowance" not in text


def test_the_final_pass_retry_is_admitted_under_the_kwargs_it_will_be_sent_with(monkeypatch):
    """Admission must price the retry with the kwargs it is actually sent with, not the original turn."""

    seen: list[object] = []
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _finish("length"), _done()],
            [_sse({"content": "Here is the answer."}), _done()],
        ],
        payloads,
    )

    real_count = backend.count_chat_tokens

    def recording_count(*args, **kwargs):
        seen.append(kwargs.get("chat_template_kwargs"))
        return real_count(*args, **kwargs)

    monkeypatch.setattr(backend, "count_chat_tokens", recording_count)

    events = _run_no_tools(backend)

    assert len(payloads) == 2, "the retry was refused"
    assert payloads[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert {
        "enable_thinking": False
    } in seen, "the retry was admitted under kwargs it is not sent with"
    assert "Here is the answer." in "".join(_texts(events, "content"))


def test_the_in_loop_retry_is_admitted_under_the_kwargs_it_will_be_sent_with(monkeypatch):
    """In-loop retry admission priced the previous kwargs with thinking on, not the retry actually sent."""

    seen: list[object] = []
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _truncated_thought_then([_sse({"content": "Done."}), _done()]),
        payloads,
    )

    real_count = backend.count_chat_tokens

    def recording_count(*args, **kwargs):
        seen.append(kwargs.get("chat_template_kwargs"))
        return real_count(*args, **kwargs)

    monkeypatch.setattr(backend, "count_chat_tokens", recording_count)

    _run(backend)

    assert len(payloads) == 2, "the retry was refused"
    assert payloads[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert {
        "enable_thinking": False
    } in seen, "the retry was admitted under kwargs it is not sent with"


def test_the_in_loop_give_up_names_the_cap_when_the_last_attempt_spent_it(monkeypatch):
    """_reasoning_cap_spent is set only on refusal, so exhausting retries misnames the cap."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                _sse({"reasoning_content": _LONG_THOUGHT}),
                _usage(100),
                _finish("length"),
                _done(),
            ]
            for _ in range(_MAX_LENGTH_CONTINUATIONS + 2)
        ],
        payloads,
    )

    events = _run(backend, max_tokens = 300)

    assert len(payloads) == _MAX_LENGTH_CONTINUATIONS + 1, "a continuation was refused"
    text = "".join(_texts(events, "content"))
    assert "output allowance of 300 tokens" in text
    assert "window on reasoning" not in text


def test_a_continuation_one_eviction_short_is_not_abandoned(monkeypatch):
    """Refusing ends the turn before the next preflight, so a continuation one eviction short is kept."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _usage(20), _finish("length"), _done()],
            [_sse({"content": "Done."}), _usage(10), _done()],
        ],
        payloads,
    )

    def fake_count(messages, *_args, **_kwargs):
        return 200 + sum(len(str(m.get("content") or "")) // 4 for m in messages)

    monkeypatch.setattr(backend, "count_chat_tokens", fake_count)

    old_turns: list[dict] = []
    for index in range(30):
        old_turns.append({"role": "user", "content": f"Question {index}. " + "x" * 600})
        old_turns.append({"role": "assistant", "content": f"Answer {index}. " + "y" * 600})
    latest = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Create a Flappy Bird game"},
            {
                "type": "input_audio",
                "input_audio": {"data": "A" * 100_000, "format": "wav"},
            },
        ],
    }

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [*old_turns, latest],
            tools = [_WEB_SEARCH_TOOL],
            enable_thinking = True,
            max_tool_iterations = 3,
            max_tokens = 100,
            context_overflow = "truncate_oldest",
        )
    )

    assert len(payloads) == 2, "the continuation was abandoned instead of making room"
    assert payloads[1]["messages"][-3] == latest
    assert payloads[1]["messages"][-3]["content"][1]["input_audio"]["data"] == "A" * 100_000
    assert "Done." in "".join(_texts(events, "content"))


def test_final_pass_continuation_counts_strip_media_and_payloads_keep_it(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _truncated_thought_then([_sse({"content": "Done."}), _done()]),
        payloads,
    )
    counted: list[list[dict]] = []

    def fake_count(messages, *_args, **_kwargs):
        counted.append(copy.deepcopy(messages))
        return 100

    monkeypatch.setattr(backend, "count_chat_tokens", fake_count)
    audio_data = "A" * 100_000
    latest = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Create a Flappy Bird game"},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": "wav"},
            },
        ],
    }

    list(
        backend.generate_chat_completion_with_tools(
            messages = [latest],
            tools = [],
            max_tool_iterations = 0,
            enable_thinking = True,
            context_overflow = "truncate_oldest",
        )
    )

    assert counted
    assert all(
        part.get("type") != "input_audio"
        for candidate in counted
        for message in candidate
        for part in message.get("content", [])
        if isinstance(part, dict)
    )
    assert len(payloads) == 2
    assert payloads[0]["messages"][0] == latest
    assert payloads[1]["messages"][0] == latest
    assert payloads[1]["messages"][0]["content"][1]["input_audio"]["data"] == audio_data


def test_a_continuation_is_sized_by_what_is_left_of_the_cap(monkeypatch):
    """Preflight must see the remaining cap, since prompt_budget shrinks as max_tokens grows."""

    targets: list[int] = []
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": "Half an answer"}), _usage(900), _finish("length"), _done()],
            [_sse({"content": " and the rest."}), _usage(50), _done()],
        ],
        payloads,
    )

    import core.inference.llama_cpp as _lc  # noqa: PLC0415

    real_budget = _lc.prompt_budget

    def recording_budget(context_length, max_tokens):
        targets.append(max_tokens)
        return real_budget(context_length, max_tokens)

    monkeypatch.setattr(_lc, "prompt_budget", recording_budget)

    def fake_count(messages, *_args, **_kwargs):
        return 200 + sum(len(str(m.get("content") or "")) // 4 for m in messages)

    monkeypatch.setattr(backend, "count_chat_tokens", fake_count)

    old_turns: list[dict] = []
    for index in range(8):
        old_turns.append({"role": "user", "content": f"Question {index}. " + "x" * 600})
        old_turns.append({"role": "assistant", "content": f"Answer {index}. " + "y" * 600})

    list(
        backend.generate_chat_completion_with_tools(
            messages = [*old_turns, {"role": "user", "content": "Show me the HTML inline"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 3,
            max_tokens = 1000,
            context_overflow = "truncate_oldest",
        )
    )

    assert len(payloads) == 2, "the answer was not continued"
    assert payloads[1]["max_tokens"] == 100, "the payload did not get the remainder"
    assert 100 in targets, f"every sizing decision still used the whole cap: {sorted(set(targets))}"

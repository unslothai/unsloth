# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A turn that spends the whole window thinking must not surface as an empty message.

Observed on a 4096-token window: the model generated 2301 tokens, all of them reasoning,
and stopped on `finish_reason: length` -- exactly the room the prompt left. It never
reached a tool call or an answer, and because `_finalize_reasoning_only_cumulative`
refuses to promote a truncated thought (correctly -- it is not an answer), the thread
showed nothing at all. Twice, on consecutive turns.

Compaction is not the lever: there was no tool result to compact, and the prompt was only
1795 tokens of a 3072-token budget. The window went entirely on thinking, so the fix is to
resume with thinking off rather than to reclaim prompt room that was never the problem.
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
    """Returning silently IS the original defect, so the give-up path must not repeat it.

    Mirrors the advice hermes-agent gives from `_thinking_exhausted` and Codex gives for
    the same symptom: name the lever (effort, window, task size) rather than show nothing.
    """

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
    """NousResearch/hermes-agent#79100: a surviving counter gives a later stall fewer tries.

    Stall, continue, run a tool, then stall again. The second stall is a new problem and
    is entitled to the same allowance the first one had.
    """

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

    # 1 stall + 1 continuation that called the tool + 3 more turns once the tool round
    # reset the allowance. A counter that survived would have stopped an attempt earlier.
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
    """The trigger for THIS path is an EMPTY length stop, not any length stop.

    Retargeted rather than deleted. It used to assert that a turn producing content was
    never continued at all, which was true when the stalled-thought path was the only one.
    A truncated ANSWER is now continued too, by the sibling path in
    `test_truncated_answer_continuation.py`, and the distinction that still matters is
    which one takes it: resuming an answer must not switch thinking off, because thinking
    was never the problem, and must extend the partial rather than start a fresh turn.
    """

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
    """The in-loop continuation cannot reach this pass, which runs after the loop breaks.

    A turn that spends its last permitted tool call, or a one-shot tool that sets
    `force_final_answer`, produces its answer here. If that generation spends the window
    thinking, the user gets an empty message and no indication anything went wrong: the
    exact failure the in-loop continuation exists to prevent, on the path it does not
    cover.
    """

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
    """For this style an explicit effort WINS over enable_thinking.

    `_request_reasoning_kwargs` returns the caller's "high" and never reaches the
    enable_thinking branch, so turning thinking off for the retry changed nothing that
    llama-server could see and the retry re-ran the turn that had just spent the whole
    window thinking. The second failure then looked identical to the first.
    """

    payloads: list[dict] = []
    backend = _effort_backend(
        monkeypatch,
        _truncated_thought_then([_sse({"content": "Here is the game."}), _done()]),
        payloads,
    )

    _run(backend, reasoning_effort = "high")

    assert payloads[0]["chat_template_kwargs"] == {"reasoning_effort": "high"}
    # "low", not "none": this style covers models that cannot actually disable
    # reasoning, and that is the convention the non-continuation path already uses.
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
    """The credit is granted as continuations happen, not reserved up front.

    The loop bound moved from a `range(...)` to an explicit counter to make room for
    them. A request with no stall must be unaffected by that: same number of model
    calls, same tool budget, nothing extra.
    """

    streams = [[_tool_call_sse(i), _done()] for i in range(6)]
    streams.append([_sse({"content": "Done."}), _done()])
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "a result",
    )

    _run(backend, max_tool_iterations = 3)

    # Three tool rounds, then the tool-free final pass. Unchanged by the conversion.
    assert len(payloads) == 4


def test_a_stall_does_not_eat_the_tool_budget(monkeypatch):
    """The defect the credit exists for: at a small budget the retries spent it all.

    Codex's scenario, built literally. `max_tool_iterations=1` and `MAX_ACT_REPROMPTS=3`
    give the loop five slots. Three plan-without-action turns take three of them, two
    truncated reasoning turns take the other two, and the model has still not issued its
    call: control falls through to the tool-free final pass and the action the user asked
    for is never performed, though no real tool iteration was ever spent.
    """

    streams = [
        # Short, with an intent signal and no tool call: each earns a re-prompt. Distinct,
        # because a nudge that gets the same answer back stops the sequence.
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

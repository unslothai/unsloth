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


def test_the_final_pass_blames_the_cap_when_the_cap_is_what_was_spent(monkeypatch):
    """The in-loop give-up already told these two walls apart; this pass did not.

    A caller-set Max Tokens smaller than the window leaves no remainder to continue with,
    so the final pass gives up here. Naming the CONTEXT window then sends the user to
    raise the one setting that was never the constraint, and this text reaches the client
    as ordinary content, so nothing downstream can correct it.
    """

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
    """Admission has to price the prompt that is actually about to be sent.

    The retry goes out with thinking OFF, which renders a different prompt from the
    thinking-on kwargs the turn started with. Counting the candidate under the original
    kwargs measures a prompt nobody sends: it refuses a retry that would have fit, or
    admits one llama-server then rejects.
    """

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
    """The final pass got this right; the in-loop path, which is the one a request with
    tools actually takes, still admitted the retry under the previous attempt's kwargs.

    `_reasoning_kw` is computed once at the top of each iteration, with thinking on. The
    retry goes out with it off, a different rendered prompt on any template that reads
    `enable_thinking`, so the admission priced a request nobody sends.
    """

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
    """`_reasoning_cap_spent` is only set when a continuation is REFUSED.

    Reaching the give-up by exhausting the retry limit instead leaves it at its default,
    so a turn whose last permitted attempt finished off an explicit Max Tokens was told
    to raise the Context Length.
    """

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

    # 300 spent 100 at a time: every continuation is ADMITTED, and the cap runs out on
    # the last permitted attempt. That is the stale case -- reaching the give-up by way
    # of a refusal already sets the flag correctly.
    events = _run(backend, max_tokens = 300)

    assert len(payloads) == _MAX_LENGTH_CONTINUATIONS + 1, "a continuation was refused"
    text = "".join(_texts(events, "content"))
    assert "output allowance of 300 tokens" in text
    assert "window on reasoning" not in text


def test_a_continuation_one_eviction_short_is_not_abandoned(monkeypatch):
    """Refusing here ends the turn, so the next iteration's preflight never runs.

    The single-turn case the check was written for really does have nothing left to
    evict. A multi-turn chat under `truncate_oldest` usually does, and abandoning it
    there throws away a recoverable answer rather than dropping one old exchange.

    The gap is `prompt_budget` against the continuation's own floor: with a small
    `max_tokens` the preflight fits the chat to 3996 of a 4096 window, while the retry
    needs 3840 or less. The fit succeeded and the continuation is still unservable.
    """

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": _LONG_THOUGHT}), _usage(20), _finish("length"), _done()],
            [_sse({"content": "Done."}), _usage(10), _done()],
        ],
        payloads,
    )

    # A tokenizer this harness can actually run: llama-server is not here to render a
    # template, and the real count raises, which every fit reads as "cannot judge".
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
    """`prompt_budget` shrinks as `max_tokens` grows, so the two must agree.

    The remainder was applied to the payload only AFTER the preflight had already fitted
    the chat against the caller's original cap. A continuation with 100 of 1000 tokens
    left was therefore priced as if it could still emit 1000, and under
    `truncate_oldest` that evicts history the request never needed to lose.
    """

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

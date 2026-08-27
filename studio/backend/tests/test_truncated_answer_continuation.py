# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An answer the window cut in half must be finished, not left mid-sentence.

Sibling of `test_length_truncated_reasoning_continuation.py`, which covers a turn that
showed NOTHING. This is the turn that showed real work and stopped mid-token: observed
streaming a 2401-byte file inline at a 4096 window, cut at `ctx.arc(6, -5, 5,` with a
`<!DOCTYPE` and no closing tag.

Compaction is the wrong lever twice over here. The room that ran out belongs to the reply,
not the prompt, and the earlier fixes had already done their job: the same turn made one
tool call where an earlier build made eighteen. What is left is arithmetic, so the answer
has to span two turns.

`continue_final_message` is what makes that seamless. The partial goes back as the assistant
turn to be EXTENDED rather than as history to be responded to, so the model resumes instead
of restarting and apologising.
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
    _CONTINUE_TRUNCATED_ANSWER_STATUS,
    _MAX_LENGTH_CONTINUATIONS,
    LlamaCppBackend,
)

_HALF_AN_ANSWER = (
    "<!DOCTYPE html>\n<html>\n<body>\n<canvas id='c'></canvas>\n<script>\n"
    # Varied on purpose. Forty VERBATIM copies of one line is repetition-dominated by the
    # guard's line rule, which is correct of the guard and wrong of a fixture standing in
    # for streamed code: the first draft of this file tripped its own echo test.
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
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _done()]),
        payloads,
    )

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
    """Continuing a repetition loop stitches the echo into the answer.

    This is the incident behind hermes-agent's repetition guard: one turn produced a
    60,698-char response because the continuation nudge kept extending a repeated fragment.
    """

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
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()]
            for _ in range(_MAX_LENGTH_CONTINUATIONS + 3)
        ],
        payloads,
    )

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

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()]
            for _ in range(_MAX_LENGTH_CONTINUATIONS + 3)
        ],
        payloads,
    )

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
    """The in-loop continuation never reaches this path, which runs after the loop breaks.

    Observed live: 25 tool calls, 22387 tokens, `incomplete: length`, stopped inside
    drawBird() with the game half-written and nothing to recover it. A turn that spends
    its whole tool budget produces its answer here, not in the loop.
    """

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _done()]),
        payloads,
    )

    events = _run_no_tools(backend)

    assert len(payloads) == 2, "the final answer was left mid-sentence"
    assert payloads[1].get("continue_final_message") is True
    assert "</html>" in "".join(_texts(events, "content"))


def test_the_final_continuation_is_capped(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"content": _HALF_AN_ANSWER}), _finish("length"), _done()]
            for _ in range(_MAX_LENGTH_CONTINUATIONS + 3)
        ],
        payloads,
    )

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
    """`_append_assistant_turn` PREPENDS onto the trailing assistant text.

    So handing it the cumulative answer on the second continuation writes the first
    fragment in twice, and the model resumes from a prompt whose own output is doubled.
    Only what is new since the last replay may be sent.
    """

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
    # And the user is shown each fragment once as well. Content events are CUMULATIVE
    # here, not deltas, so the last one is the whole answer.
    shown = _texts(events, "content")[-1]
    assert shown.count("<!DOCTYPE html>") == 1
    assert shown.endswith("</html>")


def test_usage_is_kept_across_final_continuations(monkeypatch):
    """Each attempt's usage is overwritten by the next, so it has to be folded in first.

    Left unfolded, a three-attempt answer reports the tokens of its last fragment alone
    and the tokens-per-second readout beside it is computed from the same short window.
    """

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
    """Stripping the replay makes it differ from what was already streamed.

    The next delta is concatenated onto the STREAMED text, not onto the replay, so a
    stripped replay silently drops the newline and indentation the continuation is
    about to build on. Inside a code block that is the difference between a line
    starting where it should and starting flush against the previous one.
    """

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
    """The first pass hit `length` by consuming the physical context.

    Appending everything it produced makes the retry's prompt roughly context-sized, so
    reopening the stream gets it rejected before a single extra token arrives. The user
    keeps the partial either way; only one of the two paths also shows an error.
    """

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": " never sent"}), _done()]),
        payloads,
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_args, **_kwargs: 4096)

    events = _run_no_tools(backend)

    assert len(payloads) == 1
    # And the work already streamed is still the answer.
    assert "<!DOCTYPE html>" in "".join(_texts(events, "content"))


def test_a_count_that_cannot_be_taken_is_not_a_refusal(monkeypatch):
    """Failing open restores what this path did before the check, which is the safe side."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _done()]),
        payloads,
    )

    def _no_count(*_args, **_kwargs):
        raise RuntimeError("llama-server is not loaded")

    monkeypatch.setattr(backend, "count_chat_tokens", _no_count)

    events = _run_no_tools(backend)

    assert len(payloads) == 2
    assert "</html>" in "".join(_texts(events, "content"))


def test_a_continuation_with_room_to_answer_in_is_still_sent(monkeypatch):
    """The gate must refuse only what llama-server would refuse.

    Its first form borrowed `turn_is_servable`, which charges the reserve a truncated
    TOOL RESULT needs for its notice. A continuation has no tool result, so that bar
    refused prompts that would have been served and gone on to produce text -- breaking
    the very continuation it was added to protect. 3800 of a 4096 window leaves 296, and
    the reply floor is 256.
    """

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _done()]),
        payloads,
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_args, **_kwargs: 3800)

    events = _run_no_tools(backend)

    assert len(payloads) == 2, "a continuation with room to answer in was refused"
    assert "</html>" in "".join(_texts(events, "content"))


def test_a_caller_set_max_tokens_is_not_exceeded(monkeypatch):
    """`finish_reason: length` does not say WHICH wall was hit.

    A caller asking for at most 100 completion tokens gets the stop at their own cap, and
    continuing twice more returned roughly 300 against a limit the API promised. Only the
    context wall deserves a continuation.
    """

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": " never sent"}), _done()]),
        payloads,
    )

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

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _done()]),
        payloads,
    )

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
    """With tools enabled a plain length stop takes the IN-LOOP path, not the final one.

    The in-loop payload rebuilds max_tokens from the caller's value on every iteration,
    so the guard added to the final pass did not reach here: a request capped at 100
    tokens still ran two more 100-token generations.
    """

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": " never sent"}), _done()]),
        payloads,
    )

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

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": " never sent"}), _done()]),
        payloads,
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 4096)

    events = _run(backend)

    assert len(payloads) == 1
    assert "<!DOCTYPE html>" in "".join(_texts(events, "content")), "the partial was lost"


def test_an_in_loop_continuation_with_room_is_still_sent(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _done()]),
        payloads,
    )
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
    """`has_content_tokens` and `_last_emitted` are cumulative across attempts by design.

    Judging the NEXT attempt by them classified a continuation that produced nothing but
    reasoning as another truncated visible answer: it replayed an empty suffix with
    thinking still on, which is the same failing turn a third time, instead of taking the
    reasoning-off recovery. Judged on what this attempt put on screen now.
    """

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

    events = _run_no_tools(backend)

    assert len(payloads) == 3, "the stalled continuation was not recovered"
    # The recovery ends on a USER turn asking for the answer, which is what tells it
    # apart from the answer continuation: that one replays the partial and extends it.
    assert payloads[2]["messages"][-1]["role"] == "user"
    assert "continue_final_message" not in payloads[2]


def test_an_answer_already_on_screen_is_not_replaced_by_the_explanation(monkeypatch):
    """Content events on this stream are cumulative, so a lone explanation overwrites.

    Reaching the give-up with visible text is only possible now that a stalled
    continuation takes the reasoning-only path, and losing a written answer to a message
    about why there is no answer would be a worse outcome than the one being fixed.
    """

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
    """The request goes out with `continue_final_message`; the admission counted without.

    That renders a different prompt -- no generation prompt, the partial as the turn being
    extended -- so the check can refuse a continuation llama-server would have served, or
    admit one it then rejects.
    """

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
    """Only the finish reason was reset when a continuation was accepted.

    `_metadata_usage` and `_metadata_timings` survived, so an attempt that reported
    nothing was charged the previous attempt's numbers a second time: the reply's own
    total double-counts, and a cap read off that total can look spent while there is
    still room.
    """

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
    """An empty status is the OpenAI route's iteration boundary: it clears `prev_text`.

    This pass keeps `cumulative` across attempts on purpose, so the retry's first content
    event carries the whole prefix again. Diffed from a cursor the empty status has just
    reset, the client is sent the entire partial answer a second time.
    """

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
    """`append_assistant_turn` merges into a resumed partial; the reset defeated it.

    The caller's `continue_final_message` was restored at the top of the tool-execution
    block, before the call was recorded, so a turn resumed mid-answer that then called a
    tool appended a SECOND consecutive assistant message. Strict templates reject that.
    """

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

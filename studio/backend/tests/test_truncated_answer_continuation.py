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

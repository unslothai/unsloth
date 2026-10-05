# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detect possible mid-quote stops without changing response text or finish reasons."""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import (
    LlamaCppBackend,
    _ends_inside_quote,
    _quote_cut_event,
)
from routes.inference import _quote_cut_sse_chunk, produce_openai_chat_completions

# Reported and observed Qwen3.8 cuts, plus each other opener context.
_CUT_TAILS = [
    "Actually I need to check whether the actual file contains `",
    "that's actually the canonical Qwen behavior (keeps `",
    "Note the special tokens (in the actual Qwen3, the tokens are `",
    "```\nActually Qwen1.5/2? tokens: `",
    "might be `<|im_start|>` but in tokenizer, the token ID string may be `",
    "Let's recall Qwen2 tokenizer: added tokens include:\n\"",
    'Need consider if Qwen3 uses "',
    'Could be special token "\n',
    "Character by character:\n\n- `<` `|` `i` `",
    '"',
    "Qwen2's chat template uses `{{- '",
    'The token is:\n\n```python\n"',
    "The special tokens that open and close each turn are **`",
    "the tokens (`",
    "the _`",
    "tokens: [`",
    '{"',
]

# Ways finished text ends, including ones that end on a quote or backtick.
_FINISHED_TAILS = [
    "The answer is 4.",
    "Four",
    "Use `print()` to show it.",
    "def f():\n    return 1\n```",
    "```python",
    'He said "hello".',
    'the sum of the two preceding ones (e.g., $3 + 5 = 8$)."',
    "the students'",
    '```python\n"<|im_end|>"\n```',
    '"<|im_end|>"',
    "`<|im_start|>`\n`<|im_end|>`",
    "It is called a backtick, and here it is: **`**",
    "Use `x` for inline code ```",
    'He called it "**bold**"',
    "{}",
    "",
    "   \n",
]


@pytest.mark.parametrize("tail", _CUT_TAILS)
def test_text_cut_inside_an_opened_quote_is_detected(tail):
    assert _ends_inside_quote("Some earlier text.\n\n" + tail)


@pytest.mark.parametrize("tail", _FINISHED_TAILS)
def test_finished_text_is_not(tail):
    assert not _ends_inside_quote(tail)


_CUT = _CUT_TAILS[0]
_DONE = "The answer is 4."


@pytest.mark.parametrize(
    "reasoning, answer, finish, promote, expected",
    [
        (_CUT, "", "stop", True, True),
        (_DONE, _CUT, "stop", True, True),
        (_DONE, "", "stop", True, False),
        (_DONE, _DONE, "stop", True, False),
        # A complete answer overrides a suspicious reasoning tail.
        (_CUT, _DONE, "stop", True, False),
        # `length` has its own recovery and terminal state.
        (_CUT, _CUT, "length", True, False),
        (_CUT, _CUT, None, True, False),
        # The Anthropic route does not forward the event.
        (_CUT, _CUT, "stop", False, False),
    ],
    ids = [
        "cut-thought",
        "cut-answer",
        "finished-thought",
        "finished-answer",
        "cut-thought-answered",
        "length",
        "no-finish",
        "anthropic",
    ],
)
def test_the_event_reads_the_answer_else_the_promoted_thought(
    reasoning, answer, finish, promote, expected
):
    event = _quote_cut_event(reasoning, answer, finish, promote)
    assert event == ({"type": "quote_cut"} if expected else None)


def test_the_sse_chunk_is_an_empty_chunk_carrying_the_flag():
    line = _quote_cut_sse_chunk("chatcmpl-1", "qwen")
    assert line.startswith("data: ") and line.endswith("\n\n")
    data = json.loads(line[len("data: ") :])
    assert data["choices"] == []
    assert data["quote_cut"] is True
    assert data["object"] == "chat.completion.chunk"


def test_both_gguf_streams_forward_the_flag_only_to_the_ui():
    tree = ast.parse(inspect.getsource(produce_openai_chat_completions))
    forwards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "_ui_events" in ast.dump(node.test)
        and "_quote_cut_sse_chunk" in ast.dump(ast.Module(node.body, []))
    ]
    assert len(forwards) == 2


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


def _make_backend(monkeypatch, streams: list[list[str]]):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48853
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
        _payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
    ):
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


_PREFIX = "Compare the two templates. " * 20
_CUT_THOUGHT = _PREFIX + "In Qwen3 the tokens are `"
_WHOLE_REPLY_IN_THOUGHT = _PREFIX + "They differ only in spacing."
_CUT_ANSWER = "The exact strings are:\n\n- `<` `|` `i` `"
_FINISHED_ANSWER = "The exact strings are `<|im_start|>` and `<|im_end|>`."
# Literal `</think>` in an answer must not be parsed as a reasoning boundary.
_CUT_ANSWER_QUOTING_CLOSER = "The tokens are `</think>` and `"
_FINISHED_ANSWER_QUOTING_CLOSER = "The closing tag is `</think>`"

_SEARCH_TOOL = {
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


def _stream(
    thought: str = "",
    content: str = "",
    finish: str = "stop",
) -> list[str]:
    chunks = [_sse({"reasoning_content": thought})] if thought else []
    if content:
        chunks.append(_sse({"content": content}))
    return chunks + [_finish(finish), _done()]


def _plain(backend, **kwargs) -> list:
    return list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "compare these templates"}],
            enable_thinking = True,
            **kwargs,
        )
    )


def _tool_loop(backend, **kwargs) -> list:
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "compare these templates"}],
            tools = [_SEARCH_TOOL],
            max_tool_iterations = 3,
            enable_thinking = True,
            **kwargs,
        )
    )


def _final_pass(backend, **kwargs) -> list:
    """The pass taken once the tool loop is done."""
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "compare these templates"}],
            tools = [],
            max_tool_iterations = 0,
            enable_thinking = True,
            **kwargs,
        )
    )


_EXITS = {"plain": _plain, "tool-loop": _tool_loop, "final-pass": _final_pass}

_FLAGGED = {
    "cut-thought": _stream(_CUT_THOUGHT),
    "cut-answer-after-thought": _stream(_WHOLE_REPLY_IN_THOUGHT, _CUT_ANSWER),
    "cut-answer-thinking-off": _stream(content = _CUT_ANSWER),
    "cut-answer-quoting-closer": _stream(_WHOLE_REPLY_IN_THOUGHT, _CUT_ANSWER_QUOTING_CLOSER),
    "cut-answer-quoting-closer-thinking-off": _stream(content = _CUT_ANSWER_QUOTING_CLOSER),
}

_UNFLAGGED = {
    "whole-reply-in-thought": (_stream(_WHOLE_REPLY_IN_THOUGHT), {}),
    "finished-answer": (_stream(_CUT_THOUGHT, _FINISHED_ANSWER), {}),
    "finished-answer-thinking-off": (_stream(content = _FINISHED_ANSWER), {}),
    "answer-quoting-closer": (
        _stream(_WHOLE_REPLY_IN_THOUGHT, _FINISHED_ANSWER_QUOTING_CLOSER),
        {},
    ),
    "answer-quoting-closer-thinking-off": (
        _stream(content = _FINISHED_ANSWER_QUOTING_CLOSER),
        {},
    ),
    "length": (_stream(_CUT_THOUGHT, _CUT_ANSWER, finish = "length"), {}),
    "anthropic": (_stream(_CUT_THOUGHT), {"promote_reasoning_only": False}),
}


# A `length` turn is continued, so it needs a second stream to finish on.
_SPARE = _stream(content = _FINISHED_ANSWER)


def _events(kind) -> list:
    return [e for e in kind if isinstance(e, dict)]


@pytest.mark.parametrize("exit_name", list(_EXITS))
@pytest.mark.parametrize("case", list(_FLAGGED))
def test_every_exit_reports_a_cut_before_its_metadata(monkeypatch, exit_name, case):
    events = _events(_EXITS[exit_name](_make_backend(monkeypatch, [list(_FLAGGED[case]), _SPARE])))

    kinds = [e.get("type") for e in events]
    assert kinds.count("quote_cut") == 1
    assert kinds.index("quote_cut") < kinds.index("metadata")


@pytest.mark.parametrize("exit_name", list(_EXITS))
@pytest.mark.parametrize("case", list(_UNFLAGGED))
def test_every_exit_leaves_other_endings_alone(monkeypatch, exit_name, case):
    stream, kwargs = _UNFLAGGED[case]
    events = _events(
        _EXITS[exit_name](_make_backend(monkeypatch, [list(stream), _SPARE]), **kwargs)
    )

    assert not [e for e in events if e.get("type") == "quote_cut"]


def test_the_shown_text_is_unchanged(monkeypatch):
    texts = [
        e for e in _plain(_make_backend(monkeypatch, [_stream(_CUT_THOUGHT)])) if isinstance(e, str)
    ]
    # The warning preserves the promoted thought.
    assert texts[-1].endswith("</think>" + _CUT_THOUGHT)

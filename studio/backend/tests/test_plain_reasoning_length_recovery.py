# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat without tools whose thought fills the window answers once more with thinking off,
instead of ending the turn with no answer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_length_truncated_reasoning_continuation import (  # noqa: E402
    _LONG_THOUGHT,
    _done,
    _finish,
    _make_backend,
    _sse,
    _usage,
)

_ANSWER = "Here is the game."


def _thought_cut(completion_tokens: int = 3_000) -> list[str]:
    return [
        _sse({"reasoning_content": _LONG_THOUGHT}),
        _finish("length"),
        _usage(completion_tokens),
        _done(),
    ]


def _answer() -> list[str]:
    return [_sse({"content": _ANSWER}), _finish("stop"), _done()]


def _plain(backend, **kwargs) -> list:
    return list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "Create a Flappy Bird game in HTML"}],
            enable_thinking = True,
            **kwargs,
        )
    )


def _texts(events) -> list[str]:
    return [e for e in events if isinstance(e, str)]


def _finishes(events) -> list:
    return [
        e.get("finish_reason")
        for e in events
        if isinstance(e, dict) and e.get("type") == "metadata"
    ]


def test_a_thought_that_filled_the_window_is_answered_not_abandoned(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut(), _answer()], payloads)

    events = _plain(backend)

    assert len(payloads) == 2
    final = _texts(events)[-1]
    assert final.endswith(_ANSWER)
    # The thought stays in the turn, closed, ahead of the answer.
    assert final.startswith("<think>") and "</think>" in final
    assert _finishes(events) == ["stop"]


def test_the_retry_turns_thinking_off_and_asks_for_the_answer(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut(), _answer()], payloads)

    _plain(backend)

    retry = payloads[1]
    assert retry["chat_template_kwargs"]["enable_thinking"] is False
    *_, note, ask = retry["messages"]
    assert note["role"] == "assistant" and note["content"].startswith("Where I had got to:")
    # A short note stands in for the thought; replaying it whole would fill the window again.
    assert len(note["content"]) < 1_000
    assert ask["role"] == "user" and "give your answer to my request now" in ask["content"]
    assert "call a tool" not in ask["content"]


def test_every_snapshot_extends_the_last(monkeypatch):
    backend = _make_backend(monkeypatch, [_thought_cut(), _answer()], [])

    texts = _texts(_plain(backend))

    assert all(later.startswith(earlier) for earlier, later in zip(texts, texts[1:]))


def test_the_retry_happens_once(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut(), _thought_cut(), _answer()], payloads)

    events = _plain(backend)

    assert len(payloads) == 2
    assert _finishes(events) == ["length"]


def test_a_spent_max_tokens_is_not_retried(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut(completion_tokens = 512), _answer()], payloads)

    events = _plain(backend, max_tokens = 512)

    assert len(payloads) == 1
    assert _finishes(events) == ["length"]


def test_the_retry_gets_only_what_is_left_of_max_tokens(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch, [_thought_cut(completion_tokens = 3_000), _answer()], payloads
    )

    _plain(backend, max_tokens = 4_000)

    assert payloads[1]["max_tokens"] == 1_000


def test_a_clean_reasoning_only_stop_is_left_alone(monkeypatch):
    payloads: list[dict] = []
    stream = [_sse({"reasoning_content": "Short thought."}), _finish("stop"), _done()]
    backend = _make_backend(monkeypatch, [stream, _answer()], payloads)

    _plain(backend)

    assert len(payloads) == 1


def test_an_answer_cut_by_the_window_is_not_answered_again(monkeypatch):
    payloads: list[dict] = []
    stream = [
        _sse({"reasoning_content": "Planning."}),
        _sse({"content": "Part of the answer"}),
        _finish("length"),
        _done(),
    ]
    backend = _make_backend(monkeypatch, [stream, _answer()], payloads)

    _plain(backend)

    assert len(payloads) == 1


def test_a_retry_the_server_refuses_ends_the_turn_as_before(monkeypatch):
    def refused():
        raise RuntimeError("request exceeds the available context size")
        yield  # pragma: no cover

    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut(), refused()], payloads)

    events = _plain(backend)

    assert len(payloads) == 2
    assert _texts(events)[-1].endswith("</think>")
    assert _finishes(events) == ["length"]


def test_the_image_stays_on_the_question_it_came_with(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut(), _answer()], payloads)

    _plain(backend, image_b64 = "aGVsbG8=")

    question, _, ask = payloads[1]["messages"][-3:]
    assert isinstance(question["content"], list)
    assert any(part.get("type") == "image_url" for part in question["content"])
    assert isinstance(ask["content"], str)


def test_a_model_that_always_thinks_is_not_retried(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut(), _answer()], payloads)
    backend._reasoning_always_on = True

    events = _plain(backend)

    assert len(payloads) == 1
    assert _finishes(events) == ["length"]

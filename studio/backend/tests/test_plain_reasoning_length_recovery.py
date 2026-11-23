# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat without tools whose thought fills the window (no EOS) is answered instead of ending with
no answer: the thought is cut back a few tokens, closed with a literal </think>, trimmed to its tail
so an answer fits, and the turn continues as raw text. Templates without a </think> tag, image
chats and a raw attempt that fails before any text fall back to one answer with thinking off."""

from __future__ import annotations

import json
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
    # Replaying the whole thought would fill the window again.
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


def test_a_model_that_always_thinks_without_a_think_tag_is_not_retried(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut(), _answer()], payloads)
    backend._reasoning_always_on = True

    events = _plain(backend)

    assert len(payloads) == 1
    assert _finishes(events) == ["length"]


def test_the_retry_keeps_the_question_and_evicts_nothing(monkeypatch):
    backend = _make_backend(monkeypatch, [_thought_cut(), _answer()], [])
    calls: list[dict] = []
    original = backend.generate_chat_completion

    def recording(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return original(messages, **kwargs)

    monkeypatch.setattr(backend, "generate_chat_completion", recording)
    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "Create a Flappy Bird game in HTML"}],
            enable_thinking = True,
            context_overflow = "truncate_oldest",
        )
    )

    retry = calls[1]
    # A fit protects only the newest user turn, which here is the request the retry added.
    assert retry["context_overflow"] is None
    assert retry["messages"][0] == {"role": "user", "content": "Create a Flappy Bird game in HTML"}


def test_usage_counts_both_attempts(monkeypatch):
    answer = [_sse({"content": _ANSWER}), _finish("stop"), _usage(200), _done()]
    backend = _make_backend(monkeypatch, [_thought_cut(completion_tokens = 3_000), answer], [])

    [metadata] = [e for e in _plain(backend) if isinstance(e, dict) and e.get("type") == "metadata"]

    assert metadata["usage"]["completion_tokens"] == 3_200
    assert metadata["finish_reason"] == "stop"
    # The context meter shows what the window holds now: the retry's prompt and answer only.
    assert metadata["usage"]["context_tokens"] == 100 + 200


def test_the_callers_messages_are_left_as_they_were(monkeypatch):
    backend = _make_backend(monkeypatch, [_thought_cut(), _answer()], [])
    messages = [{"role": "user", "content": "Create a Flappy Bird game in HTML"}]
    before = [dict(m) for m in messages]

    list(backend.generate_chat_completion(messages = messages, enable_thinking = True))

    assert messages == before


# ── Closing the thought with a literal </think> ──

_TEMPLATE = "{% if enable_thinking %}<think>{% endif %}...</think>..."
_PROMPT = "<|im_start|>user\nCreate a Flappy Bird game in HTML<|im_end|>\n<|im_start|>assistant\n"


def _raw_answer(
    text = _ANSWER,
    stop_type = "eos",
    predicted = 5,
) -> list[str]:
    final = {
        "content": text,
        "stop": True,
        "stop_type": stop_type,
        "tokens_evaluated": 300,
        "tokens_predicted": predicted,
        "timings": {"predicted_n": predicted, "predicted_ms": 50.0},
    }
    return [f"data: {json.dumps(final)}\n"]


def _closing_backend(
    monkeypatch,
    streams,
    payloads,
    *,
    context = 4096,
    prompt = _PROMPT,
    posts = None,
):
    """Words stand in for tokens: /tokenize splits on spaces, /detokenize joins."""
    backend = _make_backend(monkeypatch, streams, payloads)
    backend._chat_template = _TEMPLATE
    backend._effective_context_length = context

    def fake_post(path, body):
        if posts is not None:
            posts.append((path, body))
        if path == "/apply-template":
            return {"prompt": prompt}
        if path == "/tokenize":
            return {"tokens": body["content"].split()}
        if path == "/detokenize":
            return {"content": " ".join(body["tokens"])}
        return None

    monkeypatch.setattr(backend, "_llama_post_json", fake_post)
    return backend


def test_a_cut_thought_is_closed_with_a_literal_tag_and_answered(monkeypatch):
    payloads: list[dict] = []
    backend = _closing_backend(monkeypatch, [_thought_cut(), _raw_answer()], payloads)

    events = _plain(backend)

    raw = payloads[1]
    assert "messages" not in raw
    assert raw["prompt"].startswith(_PROMPT + "<think>\n")
    assert raw["prompt"].endswith("\n</think>\n\n")
    final = _texts(events)[-1]
    assert final.startswith("<think>") and final.endswith("</think>" + _ANSWER)
    assert _finishes(events) == ["stop"]


def test_the_thought_is_cut_back_and_only_its_tail_is_kept(monkeypatch):
    payloads: list[dict] = []
    backend = _closing_backend(
        monkeypatch, [_thought_cut(), _raw_answer()], payloads, context = 1_000
    )

    _plain(backend)

    words = _LONG_THOUGHT.split()
    kept = payloads[1]["prompt"][len(_PROMPT + "<think>\n") : -len("\n</think>\n\n")].split()
    # The last few tokens may stop mid-word, so they go; the rest is the thought's own tail.
    assert kept == words[:-8][-len(kept) :]
    assert 0 < len(kept) < len(words) - 8


def test_the_closed_request_leaves_room_to_answer(monkeypatch):
    payloads: list[dict] = []
    backend = _closing_backend(
        monkeypatch, [_thought_cut(), _raw_answer()], payloads, context = 1_000
    )

    _plain(backend)

    raw = payloads[1]
    kept = raw["prompt"][len(_PROMPT + "<think>\n") : -len("\n</think>\n\n")].split()
    assert len(raw["prompt"].split()) + raw["n_predict"] <= 1_000
    # The thought's tail never takes more of the window than the answer gets.
    assert raw["n_predict"] >= len(kept)


def test_the_answer_gets_only_what_is_left_of_max_tokens(monkeypatch):
    payloads: list[dict] = []
    backend = _closing_backend(
        monkeypatch, [_thought_cut(completion_tokens = 3_000), _raw_answer()], payloads
    )

    _plain(backend, max_tokens = 3_200)

    assert payloads[1]["n_predict"] == 200


def test_a_template_that_opens_the_thought_itself_gets_no_second_opener(monkeypatch):
    payloads: list[dict] = []
    backend = _closing_backend(
        monkeypatch, [_thought_cut(), _raw_answer()], payloads, prompt = _PROMPT + "<think>\n"
    )

    _plain(backend)

    assert payloads[1]["prompt"].count("<think>") == 1


def test_a_model_that_always_thinks_is_answered_the_same_way(monkeypatch):
    payloads: list[dict] = []
    backend = _closing_backend(monkeypatch, [_thought_cut(), _raw_answer()], payloads)
    backend._reasoning_always_on = True

    events = _plain(backend)

    assert len(payloads) == 2 and "prompt" in payloads[1]
    assert _texts(events)[-1].endswith(_ANSWER)


def test_an_answer_cut_again_reports_length(monkeypatch):
    payloads: list[dict] = []
    backend = _closing_backend(
        monkeypatch, [_thought_cut(), _raw_answer(stop_type = "limit")], payloads
    )

    events = _plain(backend)

    assert len(payloads) == 2
    assert _finishes(events) == ["length"]


def test_usage_after_closing_counts_both_and_the_meter_only_the_last(monkeypatch):
    backend = _closing_backend(
        monkeypatch, [_thought_cut(completion_tokens = 3_000), _raw_answer(predicted = 200)], []
    )

    [metadata] = [e for e in _plain(backend) if isinstance(e, dict) and e.get("type") == "metadata"]

    assert metadata["usage"]["completion_tokens"] == 3_200
    assert metadata["usage"]["context_tokens"] == 300 + 200


def test_a_template_without_a_think_tag_answers_with_thinking_off(monkeypatch):
    payloads: list[dict] = []
    backend = _closing_backend(monkeypatch, [_thought_cut(), _answer()], payloads)
    backend._chat_template = "<|im_start|>assistant\n"

    _plain(backend)

    assert payloads[1]["chat_template_kwargs"]["enable_thinking"] is False


def test_a_raw_attempt_that_fails_first_falls_back_to_thinking_off(monkeypatch):
    def refused():
        raise RuntimeError("completion endpoint unavailable")
        yield  # pragma: no cover

    payloads: list[dict] = []
    backend = _closing_backend(monkeypatch, [_thought_cut(), refused(), _answer()], payloads)

    events = _plain(backend)

    assert len(payloads) == 3
    assert payloads[2]["chat_template_kwargs"]["enable_thinking"] is False
    assert _texts(events)[-1].endswith(_ANSWER)


def test_an_image_chat_answers_with_thinking_off(monkeypatch):
    payloads: list[dict] = []
    posts: list = []
    backend = _closing_backend(monkeypatch, [_thought_cut(), _answer()], payloads, posts = posts)

    _plain(backend, image_b64 = "aGVsbG8=")

    assert posts == []
    assert payloads[1]["chat_template_kwargs"]["enable_thinking"] is False

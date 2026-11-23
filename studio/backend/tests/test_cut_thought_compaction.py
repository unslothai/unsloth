# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A thought the context window cut before any answer (no EOS) is not resumed: the reply ends with
the thought closed, and the thread's next request compacts, archives the thought for
search_conversation and starts with a note saying so."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_length_truncated_reasoning_continuation import (  # noqa: E402
    _LONG_THOUGHT,
    _done,
    _finish,
    _make_backend,
    _sse,
    _usage,
)

from core.inference import cut_thought  # noqa: E402

THREAD = "thread-cut"


@pytest.fixture(autouse = True)
def _fresh_markers():
    cut_thought._reset_for_tests()
    yield
    cut_thought._reset_for_tests()


@pytest.fixture
def persisted(monkeypatch):
    monkeypatch.setattr(cut_thought, "_persisted", lambda thread_id: True)


def _thought_cut(completion_tokens: int = 3_000) -> list[str]:
    return [
        _sse({"reasoning_content": _LONG_THOUGHT}),
        _finish("length"),
        _usage(completion_tokens),
        _done(),
    ]


def _plain(backend, **kwargs) -> list:
    return list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "Plan the migration"}],
            enable_thinking = True,
            thread_id = THREAD,
            **kwargs,
        )
    )


# ── At the cut ──


def test_the_cut_reply_ends_with_the_thought_closed_and_nothing_more(monkeypatch, persisted):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut()], payloads)

    events = _plain(backend)

    # llama-server is not asked to finish the thought or to answer.
    assert len(payloads) == 1
    texts = [e for e in events if isinstance(e, str)]
    assert texts[-1].endswith("</think>")
    assert [
        e["finish_reason"] for e in events if isinstance(e, dict) and e.get("type") == "metadata"
    ] == ["length"]


def test_the_cut_thought_is_held_for_the_threads_next_request(monkeypatch, persisted):
    backend = _make_backend(monkeypatch, [_thought_cut()], [])

    _plain(backend)

    assert cut_thought.take(THREAD) == _LONG_THOUGHT.strip()
    # Once.
    assert cut_thought.take(THREAD) is None


@pytest.mark.parametrize(
    "stream, kwargs",
    [
        # A clean stop.
        ([_sse({"reasoning_content": "Short."}), _finish("stop"), _done()], {}),
        # An answer the window cut: there is something to continue.
        (
            [
                _sse({"reasoning_content": "Plan."}),
                _sse({"content": "Step 1"}),
                _finish("length"),
                _done(),
            ],
            {},
        ),
        # The caller's own Max Tokens ran out, not the window: resuming the thought is right.
        (_thought_cut(completion_tokens = 512), {"max_tokens": 512}),
        # The recovery request itself: once per thread.
        (_thought_cut(), {"record_cut_thought": False}),
    ],
)
def test_only_a_thought_the_window_cut_is_held(monkeypatch, persisted, stream, kwargs):
    backend = _make_backend(monkeypatch, [stream], [])

    _plain(backend, **kwargs)

    assert cut_thought.take(THREAD) is None


def test_a_thread_studio_does_not_keep_is_left_alone(monkeypatch):
    monkeypatch.setattr(cut_thought, "_persisted", lambda thread_id: False)
    backend = _make_backend(monkeypatch, [_thought_cut()], [])

    _plain(backend)

    assert cut_thought.take(THREAD) is None


def test_a_stale_cut_is_dropped(monkeypatch, persisted):
    cut_thought.mark(THREAD, "old thought")
    real = cut_thought.time.monotonic
    monkeypatch.setattr(cut_thought.time, "monotonic", lambda: real() + cut_thought._TTL_S + 1)

    assert cut_thought.take(THREAD) is None


# ── The thread's next request ──


def _request(**kwargs):
    from models.inference import ChatCompletionRequest
    return ChatCompletionRequest(model = "local", thread_id = THREAD, stream = True, **kwargs)


def _continuation():
    """What the chat's automatic continuation sends after the cut: the thought to resume."""
    return _request(
        messages = [
            {"role": "user", "content": "Plan the migration"},
            {"role": "assistant", "content": "", "reasoning_content": _LONG_THOUGHT},
        ],
        continue_final_message = True,
        context_policy = "checkpoint",
    )


@pytest.fixture
def archived(monkeypatch):
    from state import tool_policy

    stored: list = []
    monkeypatch.setattr(
        cut_thought,
        "archive",
        lambda thread_id, thought: stored.append((thread_id, thought)) or True,
    )
    monkeypatch.setattr(tool_policy, "conversation_recall_allowed", lambda: True)
    return stored


def test_the_next_request_compacts_instead_of_resuming_the_thought(persisted, archived):
    from routes import inference as inference_route

    cut_thought.mark(THREAD, _LONG_THOUGHT)
    payload = _continuation()

    note, searchable = inference_route._recover_cut_thought(payload)

    # The cut thought is what could not fit, so it is not resent.
    assert [m.role for m in payload.messages] == ["user"]
    assert payload.continue_final_message is False
    # Forced, as /compact does.
    assert payload.compact_context is True
    assert payload.context_overflow == "truncate_oldest"
    # Archived for search_conversation, which the note names.
    assert archived == [(THREAD, _LONG_THOUGHT.strip())]
    assert searchable is True
    assert note == cut_thought.NOTE_SEARCHABLE
    assert "search_conversation" in note


def test_a_typed_next_message_is_kept(persisted, archived):
    from routes import inference as inference_route

    cut_thought.mark(THREAD, _LONG_THOUGHT)
    payload = _request(
        messages = [
            {"role": "user", "content": "Plan the migration"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "go on"},
        ],
        context_policy = "checkpoint",
    )

    inference_route._recover_cut_thought(payload)

    assert [m.content for m in payload.messages if m.role == "user"] == [
        "Plan the migration",
        "go on",
    ]
    assert payload.compact_context is True


def test_without_search_the_note_does_not_name_it(persisted, monkeypatch):
    from routes import inference as inference_route

    monkeypatch.setattr(cut_thought, "archive", lambda thread_id, thought: False)
    cut_thought.mark(THREAD, _LONG_THOUGHT)

    note, searchable = inference_route._recover_cut_thought(_continuation())

    assert searchable is False
    assert "search_conversation" not in note


def test_a_request_with_no_cut_is_untouched(archived):
    from routes import inference as inference_route

    payload = _continuation()

    assert inference_route._recover_cut_thought(payload) is None
    assert payload.continue_final_message is True and len(payload.messages) == 2
    assert archived == []


def test_a_caller_that_asked_for_errors_keeps_them(persisted, archived):
    from routes import inference as inference_route

    cut_thought.mark(THREAD, _LONG_THOUGHT)
    payload = _continuation()
    payload.context_overflow = "error"

    assert inference_route._recover_cut_thought(payload) is None
    assert payload.compact_context is None


# ── search_conversation finds the thought ──


@pytest.fixture
def conn(rag_home, rag_conn, stub_embeddings):
    return rag_conn


def _save_thread(thread_id, messages):
    from storage import studio_db
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "t",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    for index, message in enumerate(messages):
        studio_db.upsert_chat_message(
            {
                "id": f"{thread_id}-{index}",
                "threadId": thread_id,
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
                "createdAt": index + 2,
            }
        )


def test_search_conversation_finds_the_cut_thought(conn):
    from core.rag import conversation_archive

    branch = [{"role": "user", "content": "Plan the migration"}]
    _save_thread(THREAD, branch)

    assert cut_thought.archive(THREAD, "Step 12 needs the ledger moved first, code ZEPHYR-4411.")

    # The live branch never carries reasoning, yet the thought is this thread's own.
    hit = conversation_archive.recall(THREAD, "ZEPHYR-4411", branch_messages = branch)
    assert hit is not None and "ZEPHYR-4411" in hit[0]


def test_an_answer_off_the_branch_is_still_filtered(conn):
    from core.rag import conversation_archive

    branch = [{"role": "user", "content": "Plan the migration"}]
    _save_thread(THREAD, branch)
    conversation_archive.archive_turns(
        THREAD,
        [
            {"role": "user", "content": "Plan the migration"},
            {"role": "assistant", "content": "A retried reply, code SIBLING-9090."},
        ],
    )

    hit = conversation_archive.recall(THREAD, "SIBLING-9090", branch_messages = branch)
    assert hit is None or "SIBLING-9090" not in hit[0]


def test_the_archive_and_the_marker_agree_on_the_thought_document():
    from core.rag import conversation_archive

    # core.rag cannot import core.inference, so the label is written in both; they must match.
    assert conversation_archive._CUT_THOUGHT_DOCUMENT == cut_thought.THOUGHT_DOCUMENT

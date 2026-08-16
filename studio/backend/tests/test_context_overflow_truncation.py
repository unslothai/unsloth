# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the opt-in ``context_overflow="truncate_middle"`` passthrough policy.

On ``exceed_context_size_error`` the passthrough drops middle turn-groups and
retries inside the real window instead of surfacing a fatal 400. Truncation
keeps the system prompt, the first turn, and recent turns, and never orphans
a tool result from its tool_calls turn. Also covers ``/v1/models`` exposing
the real post-readback context window.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.inference import (
    _apply_overflow_truncation,
    _clip_long_contents,
    _CLIP_MARKER,
    _context_truncated_sse_chunk,
    _estimate_message_tokens,
    _openai_model_objects,
    _overflow_truncation_requested,
    _parse_overflow_counts,
    _truncate_middle_messages,
    _truncate_oldest_messages,
)
from core.inference.context_window import fit_rolling_context, messages_have_media
import routes.inference as routes_mod


# Nick's actual error body from the Discord report logs.
_NICK_ERROR = (
    '{"detail":"llama-server error: {\\"error\\":{\\"code\\":400,'
    '\\"message\\":\\"request (70494 tokens) exceeds the available context size '
    '(67584 tokens), try increasing it\\",\\"type\\":\\"exceed_context_size_error\\",'
    '\\"n_prompt_tokens\\":70494,\\"n_ctx\\":67584}}"}'
)


def _tool_turn(i: int, result_chars: int = 400) -> list[dict]:
    """An assistant tool_calls turn paired with its tool result."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "read", "arguments": f'{{"filePath":"/f{i}"}}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{i}", "content": "x" * result_chars},
    ]


def _conversation(n_tool_turns: int = 12) -> list[dict]:
    msgs = [
        {"role": "system", "content": "You are an agent." * 20},
        {"role": "user", "content": "Do the big task." * 20},
    ]
    for i in range(n_tool_turns):
        msgs.extend(_tool_turn(i))
    msgs.append({"role": "assistant", "content": "halfway summary"})
    msgs.append({"role": "user", "content": "keep going"})
    return msgs


# ---------------------------------------------------------------------------
# _parse_overflow_counts
# ---------------------------------------------------------------------------


def test_parse_overflow_counts_nick_error():
    assert _parse_overflow_counts(_NICK_ERROR) == (70494, 67584)


def test_parse_overflow_counts_missing_fields():
    assert _parse_overflow_counts('{"error":"something else"}') is None


# ---------------------------------------------------------------------------
# _truncate_middle_messages
# ---------------------------------------------------------------------------


def test_truncation_drops_middle_keeps_anchors():
    msgs = _conversation()
    new, dropped = _truncate_middle_messages(msgs, keep_ratio = 0.5)
    assert dropped > 0
    assert len(new) == len(msgs) - dropped
    # System prompt and task anchor survive.
    assert new[0]["role"] == "system"
    assert new[1] == msgs[1]
    # The most recent turns survive verbatim.
    assert new[-1] == msgs[-1]
    assert new[-2] == msgs[-2]


def test_truncation_never_orphans_tool_results():
    msgs = _conversation()
    new, dropped = _truncate_middle_messages(msgs, keep_ratio = 0.4)
    assert dropped > 0
    surviving_call_ids = {
        tc["id"] for m in new if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])
    }
    for m in new:
        if m.get("role") == "tool":
            assert m["tool_call_id"] in surviving_call_ids


def test_truncation_reduces_estimated_size_toward_target():
    msgs = _conversation()
    total = sum(_estimate_message_tokens(m) for m in msgs)
    new, dropped = _truncate_middle_messages(msgs, keep_ratio = 0.5)
    new_total = sum(_estimate_message_tokens(m) for m in new)
    assert dropped > 0
    assert new_total < total
    # Should land at or below the requested share, modulo one whole group.
    biggest_group = max(
        _estimate_message_tokens(a) + _estimate_message_tokens(b)
        for a, b in zip(msgs[2:-2:2], msgs[3:-2:2])
    )
    assert new_total <= int(total * 0.5) + biggest_group


def test_truncation_noop_when_keep_ratio_full():
    msgs = _conversation()
    new, dropped = _truncate_middle_messages(msgs, keep_ratio = 1.0)
    assert dropped == 0
    assert new == msgs


def test_truncation_noop_when_only_protected_turns_remain():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        *_tool_turn(0),
        {"role": "user", "content": "latest"},
    ]
    new, dropped = _truncate_middle_messages(msgs, keep_ratio = 0.1)
    assert dropped == 0
    assert new == msgs


def test_rolling_truncation_drops_complete_oldest_turns():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "tool question"},
        *_tool_turn(1),
        {"role": "assistant", "content": "tool answer"},
        {"role": "user", "content": "latest question"},
    ]

    new, dropped = _truncate_oldest_messages(messages, keep_ratio = 0.5)

    assert dropped > 0
    assert new[0] == messages[0]
    assert new[-1] == messages[-1]
    assert "old question" not in {message.get("content") for message in new}
    # The tool exchange is either retained as a whole or removed as a whole.
    surviving_call_ids = {
        call["id"]
        for message in new
        for call in (message.get("tool_calls") or [])
    }
    assert all(
        message.get("tool_call_id") in surviving_call_ids
        for message in new
        if message.get("role") == "tool"
    )


def test_rolling_truncation_can_evict_old_tool_rounds_from_one_user_task():
    task = {"role": "user", "content": "do the task"}
    first_round = _tool_turn(1, result_chars = 4000)
    second_round = _tool_turn(2, result_chars = 4000)
    messages = [
        {"role": "system", "content": "system"},
        task,
        *first_round,
        *second_round,
    ]

    new, dropped = _truncate_oldest_messages(
        messages,
        keep_ratio = 0.6,
        protected_message_ids = {id(task)},
    )

    assert dropped == len(first_round)
    assert task in new
    assert first_round[0] not in new and first_round[1] not in new
    assert second_round[0] in new and second_round[1] in new


def test_rolling_truncation_keeps_task_when_a_synthetic_user_nudge_is_latest():
    task = {"role": "user", "content": "actual task"}
    tool_round = _tool_turn(1, result_chars = 4000)
    nudge = {"role": "user", "content": "Use an available tool now."}
    messages = [task, *tool_round, nudge]

    new, dropped = _truncate_oldest_messages(
        messages,
        keep_ratio = 0.2,
        protected_message_ids = {id(task)},
    )

    assert dropped == len(tool_round)
    assert [message["role"] for message in new] == ["user", "assistant", "user"]
    assert new[0] is task
    assert new[-1] is nudge


def test_rolling_media_detection_covers_image_and_audio_parts():
    assert messages_have_media(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
    )
    assert messages_have_media(
        [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]
    )
    assert not messages_have_media([{"role": "user", "content": "text only"}])


def test_rolling_truncation_preserves_nonleading_system_messages():
    later_system = {"role": "system", "content": "new higher-priority instruction"}
    messages = [
        {"role": "user", "content": "old" * 1000},
        {"role": "assistant", "content": "answer" * 1000},
        later_system,
        {"role": "user", "content": "latest"},
    ]

    new, dropped = _truncate_oldest_messages(messages, keep_ratio = 0.2)

    assert dropped == 2
    assert later_system in new


def test_rolling_fit_recounts_until_the_real_template_fits():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "one" * 40},
        {"role": "assistant", "content": "answer" * 40},
        {"role": "user", "content": "two" * 40},
        {"role": "assistant", "content": "answer" * 40},
        {"role": "user", "content": "latest"},
    ]

    def count_tokens(candidate):
        return sum(len(str(message.get("content", ""))) for message in candidate)

    fitted, info = fit_rolling_context(
        messages,
        context_length = 500,
        max_tokens = 100,
        count_tokens = count_tokens,
    )

    assert info is not None
    assert info["dropped_messages"] == 2
    assert info["prompt_tokens_before"] > 400
    assert info["prompt_tokens_after"] <= 400
    assert fitted[0] == messages[0]
    assert fitted[-1] == messages[-1]


def test_rolling_fit_never_clips_an_irreducible_latest_turn():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "latest" * 200},
    ]
    fitted, info = fit_rolling_context(
        messages,
        context_length = 500,
        max_tokens = 100,
        count_tokens = lambda candidate: sum(
            len(str(message.get("content", ""))) for message in candidate
        ),
    )

    assert fitted == messages
    assert info is None


def test_rolling_fit_reports_when_protected_messages_still_do_not_fit():
    latest = {"role": "user", "content": "latest" * 200}
    messages = [
        {"role": "user", "content": "old" * 100},
        {"role": "assistant", "content": "answer" * 100},
        latest,
    ]
    fitted, info = fit_rolling_context(
        messages,
        context_length = 500,
        max_tokens = 100,
        count_tokens = lambda candidate: sum(
            len(str(message.get("content", ""))) for message in candidate
        ),
    )

    assert fitted == [latest]
    assert info is not None
    assert info["fits"] is False


# ---------------------------------------------------------------------------
# _apply_overflow_truncation
# ---------------------------------------------------------------------------


def test_apply_overflow_truncation_mutates_body_and_clamps_max_tokens():
    body = {"messages": _conversation(), "max_tokens": 32000}
    assert _apply_overflow_truncation(body, _NICK_ERROR) is True
    assert len(body["messages"]) < len(_conversation())
    # Generation headroom: max_tokens clamped to the non-prompt share of n_ctx.
    assert body["max_tokens"] <= max(1024, int(67584 * 0.25))


def test_apply_overflow_truncation_returns_false_when_nothing_droppable():
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "user", "content": "latest"},
        ],
        "max_tokens": 32000,
    }
    assert _apply_overflow_truncation(body, _NICK_ERROR) is False


def test_rolling_overflow_never_clips_the_latest_request():
    latest = "latest" * 10000
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": latest},
        ]
    }

    assert _apply_overflow_truncation(body, _NICK_ERROR, "truncate_oldest") is True
    assert body["messages"][-1]["content"] == latest
    # A second recovery has no old turn left to drop and must fail cleanly.
    assert _apply_overflow_truncation(body, _NICK_ERROR, "truncate_oldest") is False


def test_context_truncation_notice_is_an_openai_compatible_empty_chunk():
    line = _context_truncated_sse_chunk(
        "chatcmpl-test",
        "model",
        {"dropped_messages": 4, "fits": True},
    )
    payload = json.loads(line.removeprefix("data: ").strip())

    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"] == []
    assert payload["context_truncated"] == {"dropped_messages": 4, "fits": True}


def test_apply_overflow_truncation_clips_giant_protected_tool_results():
    """One giant burst (few turn-groups, all protected) must still shrink:
    stage 2 clips oversized tool contents instead of giving up."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        *_tool_turn(0, result_chars = 60000),
        *_tool_turn(1, result_chars = 60000),
    ]
    body = {"messages": msgs, "max_tokens": 32000}
    n_before = len(msgs)
    assert _apply_overflow_truncation(body, _NICK_ERROR) is True
    # No message disappeared (pairing intact), but contents were clipped.
    assert len(body["messages"]) == n_before
    clipped = [m for m in body["messages"] if _CLIP_MARKER in str(m.get("content"))]
    assert clipped, "expected at least one clipped tool result"
    surviving_call_ids = {
        tc["id"]
        for m in body["messages"]
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    for m in body["messages"]:
        if m.get("role") == "tool":
            assert m["tool_call_id"] in surviving_call_ids


def test_clip_long_contents_reaches_target_and_keeps_structure():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        *_tool_turn(0, result_chars = 40000),
        {"role": "user", "content": "latest question"},
    ]
    total = sum(_estimate_message_tokens(m) for m in msgs)
    clipped = _clip_long_contents(msgs, target_est = total // 4)
    assert clipped >= 1
    assert sum(_estimate_message_tokens(m) for m in msgs) <= total // 4
    # Roles and count unchanged; the short final user message untouched.
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]
    assert msgs[-1]["content"] == "latest question"


def test_overflow_truncation_requested_reads_field(monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising = False)

    class _P:
        context_overflow = "truncate_middle"

    class _Q:
        context_overflow = None

    class _R:
        context_overflow = "truncate_oldest"

    assert _overflow_truncation_requested(_P()) is True
    assert _overflow_truncation_requested(_Q()) is False
    assert _overflow_truncation_requested(_R()) is True
    assert _overflow_truncation_requested(object()) is False


def test_overflow_truncation_server_default_env(monkeypatch):
    """UNSLOTH_CONTEXT_OVERFLOW enables the policy for clients that cannot
    send custom body fields; an explicit per-request 'error' still wins."""

    class _Unset:
        context_overflow = None

    class _ExplicitError:
        context_overflow = "error"

    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "truncate_middle")
    assert _overflow_truncation_requested(_Unset()) is True
    assert _overflow_truncation_requested(_ExplicitError()) is False
    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "error")
    assert _overflow_truncation_requested(_Unset()) is False
    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "truncate_oldest")
    assert _overflow_truncation_requested(_Unset()) is True


# ---------------------------------------------------------------------------
# /v1/models context metadata
# ---------------------------------------------------------------------------


class _FakeLlamaBackend:
    is_loaded = True
    model_identifier = "unsloth/Qwen3.6-27B-GGUF"
    context_length = 67584
    max_context_length = 262144


class _FakeEmptyBackend:
    active_model_name = None


def test_v1_models_exposes_real_context_window(monkeypatch):
    monkeypatch.setattr(routes_mod, "get_llama_cpp_backend", lambda: _FakeLlamaBackend())
    monkeypatch.setattr(routes_mod, "get_inference_backend", lambda: _FakeEmptyBackend())
    models = _openai_model_objects()
    assert len(models) == 1
    entry = models[0]
    assert entry["id"] == "unsloth/Qwen3.6-27B-GGUF"
    # The REAL (post /props readback) window, not the requested one.
    assert entry["context_length"] == 67584
    assert entry["max_context_length"] == 262144


def _conversation_with_big_reasoning(trace_chars: int = 40000) -> list[dict]:
    messages = [{"role": "system", "content": "sys"}]
    for index in range(8):
        messages.append({"role": "user", "content": f"question {index} " + "u" * 200})
        messages.append({"role": "assistant", "content": f"answer {index} " + "a" * 200})
    messages[-1]["reasoning_content"] = "t" * trace_chars
    messages.append({"role": "user", "content": "final question"})
    return messages


def test_reasoning_clip_shrinks_a_protected_turn():
    messages = _conversation_with_big_reasoning()
    before = routes_mod._estimate_messages_tokens(messages)

    assert routes_mod._clip_reasoning_contents(messages) == 1
    assert _CLIP_MARKER in messages[-2]["reasoning_content"]
    assert routes_mod._estimate_messages_tokens(messages) < before / 5


def test_reasoning_clip_alone_prevents_middle_eviction():
    body = {"messages": _conversation_with_big_reasoning()}
    before = len(body["messages"])
    error = '{"n_prompt_tokens":12000,"n_ctx":8192}'

    assert _apply_overflow_truncation(body, error) is True
    assert len(body["messages"]) == before
    assert _CLIP_MARKER in body["messages"][-2]["reasoning_content"]
    assert body["max_tokens"] == max(
        1024, int(8192 * (1.0 - routes_mod._OVERFLOW_PROMPT_TARGET_FRACTION))
    )

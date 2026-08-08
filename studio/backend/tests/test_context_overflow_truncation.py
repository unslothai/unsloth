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
    _estimate_message_tokens,
    _openai_model_objects,
    _overflow_truncation_requested,
    _parse_overflow_counts,
    _truncate_middle_messages,
)
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

    assert _overflow_truncation_requested(_P()) is True
    assert _overflow_truncation_requested(_Q()) is False
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


def _conversation_with_big_trace(trace_chars = 40000):
    """A normal turn list whose last assistant turn carries a huge reasoning trace.

    That turn is inside the protected tail, so group-dropping can never shrink it.
    """
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(8):
        msgs.append({"role": "user", "content": f"question {i} " + "u" * 200})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "a" * 200})
    msgs[-1]["reasoning_content"] = "t" * trace_chars
    msgs.append({"role": "user", "content": "final question"})
    return msgs


def test_estimate_counts_reasoning_conservatively():
    # Templates disagree on when prior reasoning renders (Qwen3.5 gates on the last user
    # turn, Kimi-K2-Thinking on the last non-tool-call assistant turn), and a GGUF can
    # carry any template, so the estimate must never assume a trace is free.
    msgs = _conversation_with_big_trace()
    assert routes_mod._estimate_messages_tokens(msgs) == sum(
        _estimate_message_tokens(m) for m in msgs
    )


def test_reasoning_clip_shrinks_a_protected_turn():
    # The trace sits in the protected tail, which group-dropping can never reach.
    msgs = _conversation_with_big_trace()
    before = routes_mod._estimate_messages_tokens(msgs)
    assert routes_mod._clip_reasoning_contents(msgs) == 1
    assert _CLIP_MARKER in msgs[-2]["reasoning_content"]
    assert routes_mod._estimate_messages_tokens(msgs) < before / 5


def test_reasoning_clip_leaves_short_traces_alone():
    msgs = [{"role": "assistant", "content": "a", "reasoning_content": "short"}]
    assert routes_mod._clip_reasoning_contents(msgs) == 0
    assert msgs[0]["reasoning_content"] == "short"


def _overflow_error(n_prompt: int, n_ctx: int) -> str:
    """An exceed_context_size_error body in the shape llama-server actually sends.

    The counts have to parse: on a body _parse_overflow_counts cannot read,
    _apply_overflow_truncation falls back to a flat keep_ratio and never
    exercises the n_ctx sizing."""
    return (
        '{"detail":"llama-server error: {\\"error\\":{\\"code\\":400,'
        '\\"message\\":\\"the request exceeds the available context size\\",'
        '\\"type\\":\\"exceed_context_size_error\\",'
        f'\\"n_prompt_tokens\\":{n_prompt},\\"n_ctx\\":{n_ctx}}}}}"}}'
    )


def test_big_trace_no_longer_evicts_the_whole_middle():
    # Regression: the trace sat in a protected turn, so the drop loop could never shed its
    # weight and evicted extra history chasing a target the trace itself had inflated.
    # Clipping reasoning first makes the outcome independent of the trace.
    err = _overflow_error(9000, 4096)
    with_trace = {"messages": _conversation_with_big_trace()}
    without_trace = {"messages": _conversation_with_big_trace(trace_chars = 0)}
    without_trace["messages"][-2].pop("reasoning_content", None)

    assert routes_mod._apply_overflow_truncation(with_trace, err) is True
    assert routes_mod._apply_overflow_truncation(without_trace, err) is True
    # The trace-free control still overflows on its own, so it is a floor, not a
    # convergence target: the trace must never cost MORE history. It was 6 groups
    # worse before the clip moved ahead of the sizing.
    assert len(with_trace["messages"]) >= len(without_trace["messages"])
    # The trace was clipped rather than paid for by dropping conversation.
    traces = [m for m in with_trace["messages"] if m.get("reasoning_content")]
    assert traces and _CLIP_MARKER in traces[0]["reasoning_content"]


def test_reasoning_clip_alone_prevents_middle_eviction():
    # Overflow driven almost entirely by the trace: once clipped the body fits, so
    # keep_ratio must be sized against the clipped body. Dividing by the pre-clip
    # n_prompt charged the shrink twice and evicted history anyway.
    err = _overflow_error(12000, 8192)
    body = {"messages": _conversation_with_big_trace()}
    before = len(body["messages"])
    assert routes_mod._apply_overflow_truncation(body, err) is True
    assert len(body["messages"]) == before  # nothing dropped, the clip alone was enough
    assert _CLIP_MARKER in body["messages"][-2]["reasoning_content"]
    assert body["max_tokens"] == max(
        1024, int(8192 * (1.0 - routes_mod._OVERFLOW_PROMPT_TARGET_FRACTION))
    )


def test_overflow_without_reasoning_keeps_legacy_keep_ratio():
    # No trace to clip means no rescale: the drop behaviour must be byte-identical
    # to before reasoning passthrough existed.
    err = _overflow_error(9000, 4096)
    msgs = _conversation_with_big_trace(trace_chars = 0)
    msgs[-2].pop("reasoning_content", None)
    body = {"messages": list(msgs)}
    assert routes_mod._apply_overflow_truncation(body, err) is True
    keep_ratio = min(0.95, (routes_mod._OVERFLOW_PROMPT_TARGET_FRACTION * 4096) / 9000)
    expected, dropped = routes_mod._truncate_middle_messages(list(msgs), keep_ratio)
    assert dropped > 0
    assert len(body["messages"]) == len(expected)


def test_truncate_middle_total_est_counts_aliased_messages():
    # The per-message estimate is memoized by id(); the total must still sum over the
    # list so a message object appearing twice is not silently counted once.
    shared = {"role": "user", "content": "x" * 4000}
    tail = [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "d"},
    ]
    head = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    aliased = head + [shared, shared] + tail
    copied = head + [dict(shared), dict(shared)] + tail
    # Two equal turns must size the same whether or not they are the same object.
    assert (
        routes_mod._truncate_middle_messages(aliased, 0.5)[1]
        == (routes_mod._truncate_middle_messages(copied, 0.5)[1])
    )

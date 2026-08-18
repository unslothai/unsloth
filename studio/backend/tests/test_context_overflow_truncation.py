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
    _accumulate_context_truncation,
    _apply_overflow_truncation,
    _apply_measured_overflow_truncation,
    _clip_long_contents,
    _CLIP_MARKER,
    _context_truncated_sse_chunk,
    _estimate_message_tokens,
    _openai_model_objects,
    _overflow_truncation_requested,
    _rolling_context_policy,
    _parse_overflow_counts,
    _truncate_middle_messages,
    _truncate_oldest_messages,
)
from core.inference import context_window
from core.inference.context_window import (
    evicted_messages,
    fit_rolling_context,
    group_turns,
    messages_have_media,
)
from models.inference import ChatCompletion
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
        call["id"] for message in new for call in (message.get("tool_calls") or [])
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


@pytest.mark.parametrize("instruction_role", ["system", "developer"])
def test_rolling_truncation_can_drop_assistant_after_instruction(instruction_role):
    instruction = {"role": instruction_role, "content": "keep this instruction"}
    greeting = {"role": "assistant", "content": "historical greeting" * 1000}
    latest = {"role": "user", "content": "latest question"}

    new, dropped = _truncate_oldest_messages([instruction, greeting, latest], keep_ratio = 0.1)

    assert dropped == 1
    assert new == [instruction, latest]


@pytest.fixture
def no_compaction_headroom(monkeypatch):
    """Pin the compaction headroom to zero.

    For tests about the MINIMUM eviction needed to fit, the headroom is noise: it drops
    more than necessary, so an exact count would assert the headroom's value rather than
    the fit's behaviour. Tests about the headroom set it explicitly.
    """
    monkeypatch.setattr(context_window, "_COMPACTION_HEADROOM_RATIO", 0.0)


def test_rolling_fit_recounts_until_the_real_template_fits(no_compaction_headroom):
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

    assert fitted is messages
    assert fitted == messages
    # Unchanged messages, but not a silent None: the fit says WHY it gave up, so the
    # user hears the single message is the problem rather than the history.
    assert info is not None and info["fits"] is False
    assert info["dropped_messages"] == 0
    assert info["latest_turn_tokens"] > info["context_length"]


def _length_counter(candidate):
    return sum(len(str(message.get("content", ""))) for message in candidate)


def test_evicted_messages_returns_dropped_turns_in_original_order(no_compaction_headroom):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "one" * 40},
        {"role": "assistant", "content": "answer" * 40},
        {"role": "user", "content": "two" * 40},
        {"role": "assistant", "content": "answer" * 40},
        {"role": "user", "content": "latest"},
    ]

    fitted, info = fit_rolling_context(
        messages,
        context_length = 500,
        max_tokens = 100,
        count_tokens = _length_counter,
    )
    gone = evicted_messages(messages, fitted)

    assert info is not None
    assert len(gone) == info["dropped_messages"]
    assert gone == [messages[1], messages[2]]


def test_evicted_messages_uses_identity_not_equality():
    """Two byte-identical turns must not collapse into one.

    An equality diff reports BOTH copies as evicted when only the older one was, so
    downstream acts on a turn the model can still see.
    """
    first = {"role": "user", "content": "same question"}
    second = {"role": "user", "content": "same question"}
    before = [first, {"role": "assistant", "content": "reply"}, second]
    after = [second]

    gone = evicted_messages(before, after)

    assert len(gone) == 2
    assert gone[0] is first
    assert all(message is not second for message in gone)


def test_group_turns_matches_the_unit_truncation_drops():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest"},
    ]

    groups = group_turns(messages)

    assert [[message["role"] for message in group] for group in groups] == [
        ["system"],
        ["user"],
        ["assistant", "tool", "assistant"],
        ["user"],
    ]


def test_reserve_tokens_does_not_trim_a_prompt_that_already_fits():
    """The reserve must never be what causes eviction.

    A conversation inside the window comes back untouched even when the reserve would
    not fit alongside it, or recall would start evicting chats nowhere near the limit.
    """
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "short answer"},
        {"role": "user", "content": "latest"},
    ]

    fitted, info = fit_rolling_context(
        messages,
        context_length = 500,
        max_tokens = 100,
        count_tokens = _length_counter,
        reserve_tokens = 380,
    )

    assert fitted is messages
    assert info is None


def test_reserve_tokens_trims_further_once_trimming_is_needed(no_compaction_headroom):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "one" * 40},
        {"role": "assistant", "content": "answer" * 40},
        {"role": "user", "content": "two" * 40},
        {"role": "assistant", "content": "answer" * 40},
        {"role": "user", "content": "latest"},
    ]

    _, plain = fit_rolling_context(
        messages,
        context_length = 500,
        max_tokens = 100,
        count_tokens = _length_counter,
    )
    _, reserved = fit_rolling_context(
        messages,
        context_length = 500,
        max_tokens = 100,
        count_tokens = _length_counter,
        reserve_tokens = 200,
    )

    assert plain is not None and reserved is not None
    assert reserved["dropped_messages"] > plain["dropped_messages"]
    assert reserved["prompt_tokens_after"] < plain["prompt_tokens_after"]


def test_rolling_fit_keeps_original_when_protected_messages_still_do_not_fit():
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

    assert fitted is messages
    assert fitted == messages
    assert info is not None and info["fits"] is False
    # The partial eviction is deliberately NOT applied: the request fails either way,
    # and dropping turns off a doomed request loses them for nothing.
    assert info["dropped_messages"] == 0
    assert info["prompt_tokens_after"] == info["prompt_tokens_before"]


def test_an_irreducible_fit_says_WHOSE_turn_does_not_fit():
    """A tool loop refits with the tool result appended.

    The turn that will not fit is then output the user never wrote and cannot edit, so
    "shorten this message" has no remedy. The role is what tells the two apart.
    """
    user_turn = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "what does this file contain"},
        {"role": "assistant", "content": "reading it"},
        {"role": "tool", "tool_call_id": "c1", "content": "output" * 200},
    ]
    _, info = fit_rolling_context(
        user_turn,
        context_length = 500,
        max_tokens = 100,
        count_tokens = _length_counter,
    )
    assert info["fits"] is False
    assert info["latest_turn_role"] == "tool"

    # And an ordinary overflowing user message still says so.
    _, info = fit_rolling_context(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "latest" * 200},
        ],
        context_length = 500,
        max_tokens = 100,
        count_tokens = _length_counter,
    )
    assert info["latest_turn_role"] == "user"


def test_an_irreducible_fit_survives_a_template_that_refuses_a_lone_tool_result():
    """The diagnosis is produced exactly where a tool loop is most likely to be.

    Strict templates refuse to render a tool result on its own, so counting that slice
    threw out of the fit and the caller fell back to the untrimmed request, telling the
    client nothing on the one path this diagnosis exists for.
    """

    def strict_counter(messages):
        if len(messages) == 1 and messages[0].get("role") == "tool":
            raise RuntimeError("a tool result must follow an assistant tool call")
        return _length_counter(messages)

    messages = [
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "python", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "output" * 500},
    ]
    _, info = fit_rolling_context(
        messages,
        context_length = 500,
        max_tokens = 100,
        count_tokens = strict_counter,
    )

    assert info is not None and info["fits"] is False
    assert info["latest_turn_role"] == "tool"
    # Estimated rather than counted: an approximation beats no diagnosis at all.
    assert info["latest_turn_tokens"] > 0


def test_an_irreducible_fit_says_whether_the_message_or_the_history_is_at_fault():
    """The two numbers that make the error actionable.

    llama-server's error reports the WHOLE conversation's size and advises shortening
    it, which cannot work when the latest turn alone is over the window.
    """
    huge_message = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "latest" * 200},
    ]
    _, info = fit_rolling_context(
        huge_message,
        context_length = 500,
        max_tokens = 100,
        count_tokens = _length_counter,
    )
    assert info["fits"] is False
    assert info["latest_turn_tokens"] > info["context_length"]

    # A conversation that fits reports nothing, not a fits:False dict a caller could
    # mistake for a failure.
    small = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]
    _, none_info = fit_rolling_context(
        small,
        context_length = 500,
        max_tokens = 100,
        count_tokens = _length_counter,
    )
    assert none_info is None


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


def test_nonstream_completion_serializes_context_truncation():
    response = ChatCompletion(choices = [])
    body = json.loads(
        routes_mod._model_json_response_with_context_truncation(
            response, {"dropped_messages": 4, "fits": True}
        ).body
    )
    assert body["context_truncated"] == {"dropped_messages": 4, "fits": True}


def test_nonstream_completion_omits_context_field_when_not_truncated():
    response = ChatCompletion(choices = [])
    body = json.loads(routes_mod._model_json_response_with_context_truncation(response, None).body)

    assert "context_truncated" not in body


def test_tool_loop_context_truncation_accumulates_dropped_messages():
    first = _accumulate_context_truncation(
        None,
        {
            "type": "context_truncated",
            "dropped_messages": 2,
            "prompt_tokens_before": 1200,
            "prompt_tokens_after": 800,
            "fits": True,
        },
    )
    combined = _accumulate_context_truncation(
        first,
        {
            "type": "context_truncated",
            "dropped_messages": 3,
            "prompt_tokens_before": 1000,
            "prompt_tokens_after": 700,
            "fits": True,
        },
    )

    assert combined == {
        "dropped_messages": 5,
        "prompt_tokens_before": 1200,
        "prompt_tokens_after": 700,
        "fits": True,
    }


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


def test_measured_overflow_truncation_distinguishes_clipping_from_no_change():
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            *_tool_turn(0, result_chars = 60000),
        ]
    }

    assert _apply_measured_overflow_truncation(body, _NICK_ERROR, "truncate_middle") == 0
    assert _CLIP_MARKER in body["messages"][-1]["content"]


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

    assert _rolling_context_policy(_P()) is None
    assert _rolling_context_policy(_R()) == "truncate_oldest"


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
    assert _rolling_context_policy(_Unset()) is None
    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "error")
    assert _overflow_truncation_requested(_Unset()) is False
    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "truncate_oldest")
    assert _overflow_truncation_requested(_Unset()) is True
    assert _rolling_context_policy(_Unset()) == "truncate_oldest"


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


def test_compaction_headroom_does_not_trim_a_prompt_that_already_fits():
    """Same rule as the reserve: headroom must never be what causes eviction.

    The headroom makes a compaction take a chunk out in one go; charging it up front
    would evict from chats that comfortably fit today.
    """
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "short answer"},
        {"role": "user", "content": "latest"},
    ]

    fitted, info = fit_rolling_context(
        messages,
        context_length = 500,
        max_tokens = 100,
        count_tokens = _length_counter,
    )

    assert fitted is messages
    assert info is None


def test_compaction_leaves_headroom_below_the_budget():
    """A compaction lands clear of the budget, not flush against it.

    Trimming to the brim makes the boundary creep every turn: the client re-sends the
    whole transcript, so an exactly fitted prompt is over again on the next turn.
    """
    messages = [{"role": "system", "content": "system"}]
    for index in range(12):
        messages.append({"role": "user", "content": f"question {index} " * 20})
        messages.append({"role": "assistant", "content": f"answer {index} " * 20})
    messages.append({"role": "user", "content": "latest"})

    _, info = fit_rolling_context(
        messages,
        context_length = 800,
        max_tokens = 100,
        count_tokens = _length_counter,
    )

    assert info is not None
    prompt_target = 800 - min(100, 800 // 4)
    assert info["prompt_tokens_after"] <= prompt_target
    # The point of the change: comfortably under, not just under.
    assert info["prompt_tokens_after"] < prompt_target * 0.9


def _long_thread(turns: int = 40):
    """A thread several times its window, in turns big enough to be evicted as units."""
    messages = [{"role": "system", "content": "system"}]
    for index in range(turns):
        messages.append({"role": "user", "content": f"question {index} " * 200})
        messages.append({"role": "assistant", "content": f"answer {index} " * 200})
    return messages


def _fit_with_appended(
    base,
    appended,
    sticky = 0,
):
    messages = list(base)
    for index in range(appended):
        messages.append({"role": "user", "content": f"follow up {index} " * 20})
        messages.append({"role": "assistant", "content": f"reply {index} " * 20})
    messages.append({"role": "user", "content": "latest"})
    _, info = fit_rolling_context(
        messages,
        context_length = 8000,
        max_tokens = 512,
        count_tokens = _length_counter,
        sticky_dropped = sticky,
    )
    return info


def test_sticky_boundary_holds_still_while_short_turns_are_appended():
    """After a compaction, ordinary turns do not push the boundary again.

    The notice depends on this, and it is why the boundary is read back rather than
    recomputed: the client re-sends the whole transcript, so a recomputed "keep the
    newest N tokens" slides forward and every reply reports a fresh compaction.
    """
    base = _long_thread()
    first = _fit_with_appended(base, 0)
    assert first is not None and first["dropped_messages"] > 0

    boundary = first["dropped_messages"]
    # Not "forever": the appended turns consume the headroom and the next test pins
    # down that it eventually moves. A handful of turns, against a baseline that moved
    # on nearly every one.
    for appended in range(1, 6):
        info = _fit_with_appended(base, appended, sticky = boundary)
        assert info is not None
        assert (
            info["dropped_messages"] == boundary
        ), f"the boundary moved after {appended} appended turns"


def test_sticky_boundary_moves_again_once_the_headroom_is_used_up():
    """It holds still, but it does not hold forever: enough new turns re-compact."""
    base = _long_thread()
    boundary = _fit_with_appended(base, 0)["dropped_messages"]

    moved = None
    for appended in range(1, 60):
        info = _fit_with_appended(base, appended, sticky = boundary)
        if info["dropped_messages"] > boundary:
            moved = appended
            break

    assert moved is not None, "the boundary never moved, so the window would overflow"
    assert moved > 4, f"the boundary moved again after only {moved} turns"


def test_sticky_boundary_never_causes_eviction_on_a_thread_that_fits():
    """A stale boundary from a longer branch must not evict a conversation that fits.

    After a rollback the saved boundary describes a branch that no longer exists. The
    fit may reapply it, but never report a compaction on a prompt that already fits.
    """
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "short answer"},
        {"role": "user", "content": "latest"},
    ]

    fitted, info = fit_rolling_context(
        messages,
        context_length = 4000,
        max_tokens = 100,
        count_tokens = _length_counter,
        sticky_dropped = 40,
    )

    assert fitted is messages
    assert info is None


def test_the_compaction_headroom_needs_a_boundary_to_be_worth_it():
    """Cutting deeper than needed buys quiet turns only if the cut is remembered.

    An incognito chat, an API request with no persisted thread, or a request whose turns
    are not saved gets neither the boundary back nor a recall of what went, so there the
    headroom is simply less history than plain eviction would have kept, on every
    overflow, and turning the archive off did not restore the old behaviour.
    """
    messages = []
    for index in range(20):
        messages.append({"role": "user", "content": f"q{index} " + "u" * 80})
        messages.append({"role": "assistant", "content": f"a{index} " + "a" * 80})
    messages.append({"role": "user", "content": "latest"})

    def _fit(keeps_boundary):
        return fit_rolling_context(
            list(messages),
            context_length = 2000,
            max_tokens = 200,
            count_tokens = _length_counter,
            keeps_boundary = keeps_boundary,
        )

    plain, plain_info = _fit(False)
    sticky, sticky_info = _fit(True)

    assert plain_info["fits"] and sticky_info["fits"]
    # The one that can restore its boundary is the one that pays for headroom.
    assert sticky_info["dropped_messages"] > plain_info["dropped_messages"]
    assert len(plain) > len(sticky)


# --- Keeping the user's standing instruction when everything else is evicted ---
#
# `truncate_oldest_messages` protects system and developer groups, the final group, and
# the newest USER group. So an instruction is safe only while it IS the newest user turn:
# one "continue" later it is the oldest evictable thing in the conversation, and the first
# to go. These cover the pin that holds it, and the bounds that stop the pin itself
# becoming the problem.


def _instruction(text = None):
    return {
        "role": "user",
        "content": text
        or (
            "Standing instruction for the rest of this task: always report results as a "
            "markdown table, and end every reply with STATUS::ZQXVARA123-ALPHA."
        ),
    }


def _filler_turns(count, chars = 800):
    """Short user nudges with long agent replies, which is the shape that produces the
    defect: the user types "keep going", the agent writes a page, and the newest USER
    group -- the only one the window protects -- is the nudge."""
    nudges = ["continue", "keep going", "yes", "ok", "go on", "proceed"]
    out = []
    for index in range(count):
        out.append({"role": "user", "content": nudges[index % len(nudges)]})
        out.append({"role": "assistant", "content": f"Section {index}. " + "x" * chars})
    return out


def test_a_governing_instruction_survives_filler_turns():
    from core.inference import instruction_pin
    from core.inference.context_window import truncate_oldest_messages

    instruction = _instruction()
    messages = (
        [
            {"role": "system", "content": "you are helpful"},
            instruction,
            {"role": "assistant", "content": "Understood."},
        ]
        + _filler_turns(6)
        + [{"role": "user", "content": "continue"}]
    )

    pinned = instruction_pin.pinned_instruction_ids(messages, groups = 2, max_tokens = 1024)
    kept, dropped = truncate_oldest_messages(messages, 0.3, protected_message_ids = pinned)

    assert any(message is instruction for message in kept)
    assert dropped >= 2
    # And with the knob at its shipped default the behaviour is exactly today's.
    kept_today, _ = truncate_oldest_messages(
        messages,
        0.3,
        protected_message_ids = instruction_pin.pinned_instruction_ids(messages, groups = 0),
    )
    assert not any(message is instruction for message in kept_today)


def test_a_pinned_instruction_cannot_starve_the_window():
    """The single enormous instruction is the thing that could starve the window, so it
    is the thing the ceiling excludes -- not partially, at all."""
    from core.inference import instruction_pin

    # ~890 tokens each, so any one of them fits under the 1024 ceiling and no two do.
    big = [_instruction("Please " + "consider this requirement carefully. " * 95) for _ in range(3)]
    messages = []
    for instruction in big:
        messages += [instruction, {"role": "assistant", "content": "ok"}]
    messages += [{"role": "user", "content": "continue"}]

    pinned = instruction_pin.pinned_instruction_ids(messages, groups = 3, max_tokens = 1024)

    assert len(pinned) == 1


def test_later_long_user_turns_crowd_out_an_older_instruction():
    """The bound is newest-first over a fixed number of groups, so a standing instruction
    with enough long user turns after it is NOT pinned. This is the same hole Zed's 80 KB
    newest-first replay has, and it is recorded here rather than left to be discovered:
    the pin protects an instruction against FILLER, not against a long conversation.
    """
    from core.inference import instruction_pin

    instruction = _instruction()
    messages = [instruction, {"role": "assistant", "content": "Understood."}]
    for index in range(3):
        messages += [
            {
                "role": "user",
                "content": f"Now review section {index} of the "
                f"report and summarise it. " + "x" * 400,
            },
            {"role": "assistant", "content": f"Section {index} reviewed."},
        ]
    messages += [{"role": "user", "content": "continue"}]

    pinned = instruction_pin.pinned_instruction_ids(messages, groups = 2, max_tokens = 4096)

    assert id(instruction) not in pinned
    # It is reachable, though, once the budget covers the turns in between.
    wide = instruction_pin.pinned_instruction_ids(messages, groups = 4, max_tokens = 4096)
    assert id(instruction) in wide


def test_a_short_follow_up_is_never_pinned():
    """ "pin the last N user turns" would pin the nudge and leave the instruction."""
    from core.inference import instruction_pin

    instruction = _instruction()
    messages = [
        instruction,
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "ok, keep going"},
        {"role": "assistant", "content": "continuing"},
        {"role": "user", "content": "continue"},
    ]

    pinned = instruction_pin.pinned_instruction_ids(messages, groups = 3, max_tokens = 4096)

    assert pinned == {id(instruction)}


def test_an_upload_is_never_treated_as_filler():
    """A one-word message with an image attached is a request, not a nudge."""
    from core.inference import instruction_pin

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }

    assert instruction_pin.is_substantive(message)


def test_a_thin_query_is_recognised_but_a_short_real_question_is_not():
    from core.inference import instruction_pin

    assert instruction_pin.is_thin_query("continue")
    assert instruction_pin.is_thin_query("yes")
    assert instruction_pin.is_thin_query("ok!")
    # Short, but it names the thing it wants: replacing this with an older instruction
    # would answer a question the user did not ask.
    assert not instruction_pin.is_thin_query("what is ZQXVARA123 now?")


def test_a_self_contained_two_word_request_is_not_thin():
    """Thin has to mean "names nothing", not "is short".

    A word count swept in every self-contained short request, and a thin query earns an
    anchor that `conversation_archive.recall` spends AHEAD of the user's own words. At
    top_k=1 -- which the over-budget backoff (4 -> 2 -> 1) and a small window both reach,
    since `_recall_top_k` is `budget // CHUNK_TOKENS` -- the anchor takes the only slot
    and the turn answering what was actually asked is never retrieved.
    """
    from core.inference import instruction_pin

    for request in ("review billing", "restart nginx", "fix authentication", "ZQXVARA123?"):
        assert not instruction_pin.is_thin_query(request), request

    # Still thin, because every word of these is a function word or a nudge: there is
    # genuinely nothing in them to search an archive for.
    for nudge in ("what about it", "and then?", "do it", "yes please", "why?", "???"):
        assert instruction_pin.is_thin_query(nudge), nudge


def test_a_pin_is_charged_for_everything_it_holds():
    """`truncate_oldest_messages` protects by GROUP, so the reply rides along with the
    instruction. Charging only the instruction let a 28-token pin hold 20037 tokens, past
    both the ceiling and the prompt-fraction cap, and `_fit_with_instruction_pins` then
    dropped every pin on the retry -- losing the instruction the pin exists to keep."""
    from core.inference import instruction_pin

    instruction = _instruction()
    messages = [
        {"role": "system", "content": "you are helpful"},
        instruction,
        {"role": "assistant", "content": "x " * 40000},
        {"role": "user", "content": "continue"},
    ]

    assert instruction_pin.pinned_instruction_ids(messages, groups = 2, max_tokens = 1024) == set()

    # And the same instruction with a reply the budget can afford is still pinned.
    modest = list(messages)
    modest[2] = {"role": "assistant", "content": "Understood."}
    assert instruction_pin.pinned_instruction_ids(modest, groups = 2, max_tokens = 1024) == {
        id(instruction)
    }


def test_a_pin_is_not_charged_for_a_tool_exchange_it_does_not_hold():
    """A trailing tool exchange is its own group, and `truncate_oldest_messages` skips a
    protected group BEFORE the `starts_user_turn` expansion, so that group stays an
    independent eviction unit and goes while the pinned instruction stays. Charging it to
    the pin would let one ordinary file read cost a one-line instruction its pin over
    tokens the pin never keeps -- and an agent run is exactly where the filler follow-up
    the pin exists for appears."""
    from core.inference import instruction_pin
    from core.inference.context_window import truncate_oldest_messages

    instruction = _instruction()
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}],
    }
    tool_result = {"role": "tool", "tool_call_id": "c1", "content": "y " * 40000}
    messages = [
        {"role": "system", "content": "you are helpful"},
        instruction,
        tool_call,
        tool_result,
        {"role": "user", "content": "continue"},
    ]

    pinned = instruction_pin.pinned_instruction_ids(messages, groups = 2, max_tokens = 1024)
    assert pinned == {id(instruction)}

    # And the 20k tokens it was being charged for are evicted anyway, with the pin on.
    kept, dropped = truncate_oldest_messages(messages, 0.01, protected_message_ids = pinned)
    assert any(message is instruction for message in kept)
    assert not any(message is tool_call for message in kept)
    assert not any(message is tool_result for message in kept)
    assert dropped == 2

    # The reply that shares the instruction's own group is still charged, so the ceiling
    # keeps working on the thing it does hold.
    with_reply = [
        {"role": "system", "content": "you are helpful"},
        instruction,
        {"role": "assistant", "content": "x " * 40000},
        {"role": "user", "content": "continue"},
    ]
    assert instruction_pin.pinned_instruction_ids(with_reply, groups = 2, max_tokens = 1024) == set()

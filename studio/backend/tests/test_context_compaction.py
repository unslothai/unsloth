# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the context-compaction module.

Coverage focuses on the structural invariants the strategies promise:

* System and first-user messages are never dropped.
* Tool-call <-> tool-result pair linkage stays valid.
* Multimodal turns are never dropped.
* Persisted history is not mutated; the strategy returns a new list.
* ``NoCompact`` is a true passthrough.
* ``SlidingWindowCompact`` is a no-op when the budget already fits.
"""

from core.inference.context_compaction import (
    NoCompact,
    SlidingWindowCompact,
    estimate_tokens,
    get_strategy,
)


def _msg(role, content = "", tool_calls = None, tool_call_id = None, name = None):
    m = {"role": role}
    if content is not None:
        m["content"] = content
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    if tool_call_id is not None:
        m["tool_call_id"] = tool_call_id
    if name is not None:
        m["name"] = name
    return m


def _long(role, length, *, content_prefix = "x"):
    return _msg(role, content_prefix * length)


class TestEstimateTokens:
    def test_empty_returns_zero(self):
        assert estimate_tokens([]) == 0

    def test_simple_string_content(self):
        # 4 chars per token: "abcd" -> 1.
        msgs = [_msg("user", "abcd")]
        assert estimate_tokens(msgs) == 1

    def test_multimodal_text_part_counts(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "abcdefgh"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            }
        ]
        # 8 chars text, image URL ignored.
        assert estimate_tokens(msgs) == 2

    def test_tool_call_arguments_count(self):
        msgs = [
            _msg(
                "assistant",
                content = "",
                tool_calls = [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"q":"hi"}'},
                    }
                ],
            )
        ]
        # 10 char arguments -> 2 tokens.
        assert estimate_tokens(msgs) >= 2


class TestNoCompact:
    def test_passthrough_returns_copy(self):
        msgs = [_msg("user", "hi")]
        out = NoCompact().compact(msgs, budget_tokens = 10)
        assert out == msgs
        # Mutating the result does not affect the original.
        out.append(_msg("assistant", "new"))
        assert len(msgs) == 1


class TestSlidingWindowUnderBudget:
    def test_no_op_when_already_fits(self):
        msgs = [
            _msg("system", "be concise"),
            _msg("user", "hi"),
            _msg("assistant", "bye"),
        ]
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 1000)
        assert out == msgs

    def test_no_op_when_budget_is_zero(self):
        # A zero/negative budget collapses to no-op (defensive).
        msgs = [_msg("user", "abcd" * 1000)]
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 0)
        assert out == msgs


class TestSlidingWindowInvariants:
    def _make_long_chat(self, n_turns, length_per_turn = 1000):
        msgs = [_msg("system", "system prompt")]
        msgs.append(_msg("user", "the original task: " + "x" * length_per_turn))
        # Alternating assistant/user follow-ups.
        for i in range(n_turns):
            msgs.append(_long("assistant", length_per_turn))
            msgs.append(_long("user", length_per_turn))
        return msgs

    def test_keeps_system_and_first_user(self):
        msgs = self._make_long_chat(n_turns = 20)
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 200)
        assert out[0]["role"] == "system"
        # First user message must survive.
        assert any(
            m.get("role") == "user" and "original task" in m.get("content", "")
            for m in out
        )

    def test_keeps_last_n_turns(self):
        msgs = self._make_long_chat(n_turns = 20)
        out = SlidingWindowCompact(keep_recent = 3).compact(msgs, budget_tokens = 200)
        # The last assistant and user pair must survive.
        assert out[-1] == msgs[-1]
        assert out[-2] == msgs[-2]

    def test_does_not_mutate_input(self):
        msgs = self._make_long_chat(n_turns = 10)
        snap = [dict(m) for m in msgs]
        _ = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 100)
        assert msgs == snap

    def test_multimodal_turn_never_dropped(self):
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 1000),
            _long("user", 1000),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            },
            _long("assistant", 1000),
            _long("user", 1000),
        ]
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 200)
        # The multimodal turn must still be present.
        assert any(isinstance(m.get("content"), list) for m in out)


class TestSlidingWindowToolPairs:
    def _make_chat_with_tool_pair(self):
        return [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 800),
            _long("user", 800),
            # The tool pair we want to test linkage on.
            _msg(
                "assistant",
                content = "",
                tool_calls = [
                    {
                        "id": "call_42",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"q":"x"}',
                        },
                    }
                ],
            ),
            _msg(
                "tool",
                content = "result for x",
                tool_call_id = "call_42",
                name = "web_search",
            ),
            _long("assistant", 800),
            _long("user", 800),
            _long("assistant", 800),
            _long("user", 800),
        ]

    def test_tool_pair_dropped_together(self):
        msgs = self._make_chat_with_tool_pair()
        # Force aggressive compaction.
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 50)
        kept_assistant_with_calls = [
            m
            for m in out
            if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list)
        ]
        kept_tool_msgs = [m for m in out if m.get("role") == "tool"]
        # If the assistant tool-call message survives, every matching
        # tool-role message must also survive (and vice versa).
        kept_ids = set()
        for m in kept_assistant_with_calls:
            for tc in m.get("tool_calls", []):
                if isinstance(tc, dict) and tc.get("id"):
                    kept_ids.add(tc["id"])
        for m in kept_tool_msgs:
            assert m.get("tool_call_id") in kept_ids

    def test_tool_pair_kept_when_in_recent_window(self):
        # Build a chat where the tool pair sits inside the recent
        # window so it must survive even when the rest is far over
        # budget. Layout: system, first-user, long pair x2, tool pair,
        # one final user turn. keep_recent=2 catches the tool pair as
        # the second-to-last group and the final user turn as the last.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 800),
            _long("user", 800),
            _long("assistant", 800),
            _long("user", 800),
            _msg(
                "assistant",
                content = "",
                tool_calls = [
                    {
                        "id": "call_42",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"q":"x"}',
                        },
                    }
                ],
            ),
            _msg("tool", content = "result", tool_call_id = "call_42", name = "web_search"),
            _msg("user", "thanks"),
        ]
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 50)
        ids = [
            (
                m.get("role"),
                m.get("tool_call_id")
                or (m.get("tool_calls") and m["tool_calls"][0].get("id")),
            )
            for m in out
        ]
        assert ("assistant", "call_42") in ids
        assert ("tool", "call_42") in ids


class TestStrategyRegistry:
    def test_get_strategy_known(self):
        assert isinstance(get_strategy("none"), NoCompact)
        assert isinstance(get_strategy("sliding"), SlidingWindowCompact)

    def test_get_strategy_unknown_falls_back_to_none(self):
        # Unknown strategy names must degrade to no-op rather than raise.
        assert isinstance(get_strategy("totally-made-up"), NoCompact)


class TestConstructorValidation:
    def test_negative_keep_recent_raises(self):
        import pytest

        with pytest.raises(ValueError):
            SlidingWindowCompact(keep_recent = -1)

    def test_invalid_threshold_raises(self):
        import pytest

        with pytest.raises(ValueError):
            SlidingWindowCompact(compact_threshold = 0.0)
        with pytest.raises(ValueError):
            SlidingWindowCompact(compact_threshold = 1.5)

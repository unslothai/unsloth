# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Edge-case coverage for the context-compaction module.

These complement ``test_context_compaction.py`` with adversarial /
boundary inputs the happy-path tests do not exercise. The goal is to
pin down the structural invariants under malformed or extreme inputs:

* Empty / tiny / oversize inputs.
* Floating / boundary constructor arguments.
* Assistant messages whose ``content`` is None or carries both text and
  tool_calls at the same time.
* Tool messages without a matching assistant, or that arrive before
  their assistant (malformed but must not crash).
* Duplicate or non-string ``tool_call_id`` values.
* ``content`` that is neither str nor list (defensive).
* No-system / no-first-user histories.
* Threshold and recent-window boundaries (``compact_threshold == 1.0``,
  ``keep_recent > len(messages)``).
* The tool-pair "drop one side, drop the other" rule under awkward
  inputs (orphan tool, tool-without-id, anchored multimodal user with
  a tool message tied to a different assistant).
"""

import pytest

from core.inference.context_compaction import (
    NoCompact,
    SlidingWindowCompact,
    _pair_linked_indices,
    estimate_tokens,
    get_strategy,
)


def _msg(
    role,
    content = "",
    tool_calls = None,
    tool_call_id = None,
    name = None,
):
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


def _long(
    role,
    length,
    *,
    content_prefix = "x",
):
    return _msg(role, content_prefix * length)


def _tool_call(
    tcid,
    name = "web_search",
    args = '{"q":"x"}',
):
    return {
        "id": tcid,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _surviving_tool_ids(messages):
    """Tool-call ids that a chat template would expect to match.

    Returns (asst_call_ids, tool_response_ids). For a structurally valid
    OpenAI chat-completions request, the two sets must be equal.
    """
    asst_ids = set()
    tool_ids = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                    asst_ids.add(tc["id"])
        elif m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if isinstance(tcid, str):
                tool_ids.add(tcid)
    return asst_ids, tool_ids


class TestEmptyAndTrivialInputs:
    def test_empty_messages_returns_empty(self):
        # estimate_tokens([]) == 0 already covered; here check compact().
        out = SlidingWindowCompact().compact([], budget_tokens = 100)
        assert out == []

    def test_single_message_under_budget_passthrough(self):
        msgs = [_msg("user", "hi")]
        out = SlidingWindowCompact().compact(msgs, budget_tokens = 100)
        assert out == msgs
        # Defensive: result is a copy, not the same list object.
        assert out is not msgs

    def test_single_anchor_over_budget_stays(self):
        # Only message is the first-user anchor. Anchors are never
        # dropped, even when alone they exceed budget. Result will be
        # over budget — that is acceptable per the strategy's docstring
        # ("nothing left to drop"), and the compactor must not loop
        # forever or crash trying.
        msgs = [_long("user", 10000)]
        out = SlidingWindowCompact(keep_recent = 0).compact(msgs, budget_tokens = 5)
        assert out == msgs


class TestKeepRecentBoundaries:
    def test_keep_recent_larger_than_message_count(self):
        # keep_recent >> len(messages): recent window covers everything,
        # so the strategy is effectively a no-op even when over budget.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 5000),
            _long("user", 5000),
        ]
        out = SlidingWindowCompact(keep_recent = 100).compact(msgs, budget_tokens = 10)
        assert out == msgs

    def test_keep_recent_equals_message_count_after_anchors(self):
        # keep_recent exactly equal to the non-anchor group count is the
        # boundary between "everything kept" and "drop at least one".
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 5000),
            _long("user", 5000),
            _long("assistant", 5000),
        ]
        out = SlidingWindowCompact(keep_recent = 3).compact(msgs, budget_tokens = 10)
        # All non-anchor groups (the trailing 3) are in the recent
        # window. Result is the full list.
        assert out == msgs


class TestThresholdBoundary:
    def test_compact_threshold_one_exact(self):
        # threshold == 1.0 is the inclusive boundary; estimate exactly
        # at budget should be considered "fits" and skip compaction.
        msgs = [_msg("user", "abcd")]  # 1 token
        out = SlidingWindowCompact(
            keep_recent = 0,
            compact_threshold = 1.0,
        ).compact(msgs, budget_tokens = 1)
        assert out == msgs

    def test_threshold_just_below_estimate_triggers_compaction(self):
        # threshold (== int(budget * compact_threshold)) below estimate
        # should compact.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 4000),
            _long("user", 4000),
            _long("assistant", 4000),
            _long("user", 4000),
        ]
        before = estimate_tokens(msgs)
        assert before > 100
        out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 100)
        assert len(out) < len(msgs)

    def test_constructor_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            SlidingWindowCompact(compact_threshold = -0.1)


class TestAssistantWithBothContentAndToolCalls:
    def test_assistant_with_text_and_tool_calls_kept_together(self):
        # OpenAI chat completions schema allows an assistant message to
        # carry BOTH non-empty content and tool_calls. The compactor
        # should treat the (content + tool_calls) message and its tool
        # response as one group.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 4000),
            _long("user", 4000),
            _msg(
                "assistant",
                content = "thinking out loud while calling a tool",
                tool_calls = [_tool_call("call_x")],
            ),
            _msg("tool", content = "result for x", tool_call_id = "call_x"),
            _msg("user", "thanks"),
        ]
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 50)
        asst_ids, tool_ids = _surviving_tool_ids(out)
        assert asst_ids == tool_ids


class TestNoneContentAssistant:
    def test_none_content_with_tool_calls(self):
        # OpenAI permits assistant content=None when tool_calls is set.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 4000),
            _long("user", 4000),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("call_n")],
            },
            _msg("tool", content = "ok", tool_call_id = "call_n"),
            _msg("user", "thanks"),
        ]
        # Must not raise. estimate_tokens treats None content as 0.
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 50)
        asst_ids, tool_ids = _surviving_tool_ids(out)
        assert asst_ids == tool_ids


class TestOutOfOrderToolMessage:
    def test_tool_message_before_its_assistant_does_not_crash(self):
        # Malformed input: tool message arrives before any assistant.
        # _pair_linked_indices walks in order; the tool message has no
        # pending id to match, so it ends up unpaired. The compactor
        # must not crash. Use distinguishable contents so we can verify
        # output order directly (identical _long() messages would
        # collide under list.index()).
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _msg("tool", content = "orphan-tool", tool_call_id = "call_z"),
            _msg("assistant", content = "a-" + "x" * 5000),
            _msg("user", "u-" + "x" * 5000),
            _msg("assistant", content = "b-" + "x" * 5000),
            _msg("user", "v-" + "x" * 5000),
        ]
        # Identity-track each output back to the source index.
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 30)
        idxs = [msgs.index(m) for m in out]
        assert idxs == sorted(idxs)


class TestOrphanToolMessageInRecentWindow:
    def test_orphan_tool_message_alone_does_not_break(self):
        # An orphan tool message (no preceding assistant tool_calls)
        # ends up in its own group. If it lands in the recent window
        # the compactor will keep it. That is malformed input on the
        # caller's side, not a compactor bug — but it must not crash.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 4000),
            _long("user", 4000),
            _msg("tool", content = "lonely", tool_call_id = "ghost"),
            _msg("user", "thanks"),
        ]
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 30)
        # Compactor returns something and does not raise.
        assert isinstance(out, list)


class TestDuplicateToolCallIds:
    def test_duplicate_ids_across_assistants(self):
        # Two assistants reuse the same tool_call_id. Only the second
        # gets the tool match (pending_ids[tid] = i overwrites). The
        # earlier assistant's tool_call has no matching tool. The
        # compactor must still produce a coherent result when the
        # second assistant + its tool fall inside the recent window.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _msg("assistant", content = "", tool_calls = [_tool_call("dup")]),
            # No tool message for the first call (malformed input).
            _long("assistant", 4000),
            _long("user", 4000),
            _msg("assistant", content = "", tool_calls = [_tool_call("dup")]),
            _msg("tool", content = "second", tool_call_id = "dup"),
            _msg("user", "thanks"),
        ]
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 30)
        # The recent-window assistant+tool group must stay linked.
        kept_asst_tool = [m for m in out if m.get("role") == "assistant" and m.get("tool_calls")]
        kept_tools = [m for m in out if m.get("role") == "tool"]
        if kept_tools:
            tool_ids_in_out = {m["tool_call_id"] for m in kept_tools}
            asst_ids_in_out = set()
            for m in kept_asst_tool:
                for tc in m.get("tool_calls") or []:
                    if isinstance(tc.get("id"), str):
                        asst_ids_in_out.add(tc["id"])
            assert tool_ids_in_out <= asst_ids_in_out


class TestPairMapInternals:
    def test_pair_map_empty_for_plain_chat(self):
        msgs = [
            _msg("system", "sys"),
            _msg("user", "hi"),
            _msg("assistant", "hello"),
        ]
        pm = _pair_linked_indices(msgs)
        # Assistant has no tool_calls so the set entry is empty.
        assert pm == {2: set()}

    def test_pair_map_tool_without_string_id_ignored(self):
        # tool_call_id is not a string — should be ignored, not crash.
        msgs = [
            _msg("assistant", content = "", tool_calls = [_tool_call("ok")]),
            {"role": "tool", "content": "x", "tool_call_id": 42},
            {"role": "tool", "content": "y", "tool_call_id": None},
        ]
        pm = _pair_linked_indices(msgs)
        # Only the string-id tool would match, and we did not include
        # one — so the assistant entry is empty.
        assert pm.get(0) == set()


class TestNonStringContent:
    def test_int_or_dict_content_does_not_crash(self):
        # Defensive: content that is neither str nor list (a stray int
        # or dict from a misbehaving caller) should be ignored by
        # estimate_tokens and treated as non-multimodal by compact().
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            {"role": "assistant", "content": 12345},
            {"role": "user", "content": {"unexpected": "shape"}},
            _long("assistant", 4000),
            _long("user", 4000),
        ]
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 30)
        assert isinstance(out, list)


class TestNoSystemNoFirstUser:
    def test_assistant_only_history(self):
        # No system, no user. The first-user anchor loop simply does
        # not match anything. Compactor must not crash.
        msgs = [
            _long("assistant", 4000),
            _long("assistant", 4000),
            _long("assistant", 4000),
            _long("assistant", 4000),
        ]
        out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 30)
        # At least one message survives (the last assistant in the
        # recent window).
        assert len(out) >= 1

    def test_first_user_not_at_index_one(self):
        # First user is not adjacent to a system message. The
        # first-user anchor must still find it by iteration order.
        msgs = [
            _msg("assistant", content = "stray pre-amble"),
            _msg("user", "the real first task"),
            _long("assistant", 4000),
            _long("user", 4000),
            _long("assistant", 4000),
        ]
        out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 30)
        # The first user must be present.
        assert any(
            m.get("role") == "user" and "real first task" in m.get("content", "") for m in out
        )


class TestMultimodalAsAnchor:
    def test_multimodal_first_user(self):
        # The very first user message is itself multimodal. Both the
        # first-user anchor and the multimodal anchor refer to the
        # same index — set semantics make the overlap a no-op.
        msgs = [
            _msg("system", "sys"),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            },
            _long("assistant", 4000),
            _long("user", 4000),
            _long("assistant", 4000),
        ]
        out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 20)
        # The multimodal first user must still be present.
        assert any(isinstance(m.get("content"), list) for m in out)


class TestPairCleanupConsistency:
    def test_drop_assistant_drops_its_tools(self):
        # The assistant tool-call message is forced out of the recent
        # window by keep_recent and is not an anchor. Its tool message
        # must be dropped with it.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _msg("assistant", content = "", tool_calls = [_tool_call("call_a")]),
            _msg("tool", content = "old result", tool_call_id = "call_a"),
            _long("assistant", 5000),
            _long("user", 5000),
            _long("assistant", 5000),
            _long("user", 5000),
        ]
        out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 20)
        asst_ids, tool_ids = _surviving_tool_ids(out)
        # No orphans either way.
        assert asst_ids == tool_ids
        # And the old pair is gone.
        assert "call_a" not in tool_ids

    def test_drop_tools_drops_assistant(self):
        # Hand-craft a scenario where the tool-side gets removed by
        # the threshold loop but the assistant would be kept. The
        # pair-cleanup pass must then drop the assistant too.
        # We exploit the fact that the threshold-driven loop runs in
        # `droppable` order (system-anchored + first-user-anchored
        # excluded). The assistant tool-call carries little text;
        # the tool response carries a lot. Dropping the tool may
        # already get us under threshold, leaving the assistant.
        # The cleanup pass must then drop the orphaned assistant.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task" + "y" * 100),
            _msg("assistant", content = "", tool_calls = [_tool_call("call_b")]),
            _msg("tool", content = "x" * 4000, tool_call_id = "call_b"),
            _long("assistant", 50),
            _long("user", 50),
        ]
        out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 100)
        asst_ids, tool_ids = _surviving_tool_ids(out)
        # Both halves of the pair are either kept or dropped together.
        assert asst_ids == tool_ids


class TestImmutability:
    def test_input_messages_never_mutated_under_pressure(self):
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _msg("assistant", content = "", tool_calls = [_tool_call("call_m")]),
            _msg("tool", content = "ok", tool_call_id = "call_m"),
            _long("assistant", 5000),
            _long("user", 5000),
        ]
        # Deep-ish snapshot: messages themselves + their tool_calls list
        # are what the compactor could conceivably mutate.
        snap = [dict(m) for m in msgs]
        snap_tcs = [list(m.get("tool_calls") or []) for m in msgs]
        _ = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 30)
        assert msgs == snap
        for m, original_tcs in zip(msgs, snap_tcs):
            assert (m.get("tool_calls") or []) == original_tcs


class TestStrategyRegistryAliases:
    def test_get_strategy_empty_name_falls_back(self):
        # The fallback path matters: a misconfigured request should
        # degrade to no-op rather than raise.
        assert isinstance(get_strategy(""), NoCompact)


class TestEstimateTokensWithBadToolCalls:
    def test_tool_call_without_function_dict(self):
        # Defensive: tool_calls list where an entry lacks "function".
        # estimate_tokens does `(tc.get("function") or {}).get(...)`,
        # which is safe — must not crash.
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "x", "type": "function"}],
            }
        ]
        assert estimate_tokens(msgs) == 0


class TestMultiToolCallSingleAssistant:
    def test_assistant_with_multiple_tool_calls_grouped(self):
        # One assistant message carries two tool_calls; both tool
        # responses must be grouped with it so the trio stays linked.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _long("assistant", 4000),
            _long("user", 4000),
            _msg(
                "assistant",
                content = "",
                tool_calls = [_tool_call("a"), _tool_call("b")],
            ),
            _msg("tool", content = "ra", tool_call_id = "a"),
            _msg("tool", content = "rb", tool_call_id = "b"),
            _msg("user", "thanks"),
        ]
        out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 50)
        asst_ids, tool_ids = _surviving_tool_ids(out)
        # All-or-nothing: the multi-call assistant brings both tool
        # responses along, or none of the three survive.
        assert asst_ids == tool_ids


class TestInterleavedUserBetweenAsstAndTool:
    def test_user_between_asst_and_tool_does_not_break_grouping(self):
        # Malformed (user message between asst tool_call and its tool
        # response). pair_map still walks in order so the tool matches
        # the most recent assistant carrying its id. The compactor
        # must not raise; downstream rendering of an "orphan" user
        # is acceptable since it was already orphaned in the input.
        msgs = [
            _msg("system", "sys"),
            _msg("user", "first task"),
            _msg("assistant", content = "", tool_calls = [_tool_call("g")]),
            _msg("user", "intermediate"),
            _msg("tool", content = "result", tool_call_id = "g"),
            _long("assistant", 5000),
            _long("user", 5000),
        ]
        out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 30)
        asst_ids, tool_ids = _surviving_tool_ids(out)
        assert asst_ids == tool_ids


class TestNoCompactWithEmpty:
    def test_nocompact_empty_messages_returns_empty(self):
        out = NoCompact().compact([], budget_tokens = 100)
        assert out == []


# ── User-boundary breaks the pair window ─────────────────────


def test_tool_message_after_user_does_not_pair_with_earlier_assistant():
    """An intervening user message ends the pending pair window. A
    later tool message arriving after the user is malformed input per
    the OpenAI chat schema and must NOT be grouped with the earlier
    assistant tool_call; otherwise the compactor would drop them
    together and corrupt the surviving template.
    """
    from core.inference.context_compaction import _pair_linked_indices

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_X",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        # User interrupts before any tool message lands. The next tool
        # message (malformed) must not pair back to assistant[2].
        {"role": "user", "content": "changed my mind, ask something else"},
        {
            "role": "tool",
            "content": "stale result",
            "tool_call_id": "call_X",
            "name": "search",
        },
    ]
    out = _pair_linked_indices(msgs)
    # Assistant at index 2 keeps an entry (it was seen) but with NO
    # tool follow-ups paired across the user boundary.
    assert 2 in out, out
    assert out[2] == set(), out


def test_paired_window_resumes_after_new_assistant_with_tool_call():
    """A fresh assistant after a user message starts a new pending
    window. Its own tool messages pair correctly, the prior assistant
    stays unpaired.
    """
    from core.inference.context_compaction import _pair_linked_indices

    msgs = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "A",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "wait, do this instead"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "B",
                    "type": "function",
                    "function": {"name": "g", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "result B",
            "tool_call_id": "B",
            "name": "g",
        },
    ]
    out = _pair_linked_indices(msgs)
    assert out[1] == set(), out
    assert out[3] == {4}, out


# ── Defensive estimate_tokens ────────────────────────────────


def test_estimate_tokens_does_not_crash_on_non_dict_tool_calls():
    """OpenAI dicts can carry non-dict tool_calls entries before
    pydantic validation; the heuristic must skip them rather than
    raise AttributeError mid-compaction.
    """
    msgs = [
        {
            "role": "assistant",
            "content": "",
            # First entry is malformed (string), second is fine.
            "tool_calls": [
                "bad-entry",
                None,
                {
                    "id": "ok",
                    "function": {"name": "f", "arguments": "{}"},
                },
            ],
        }
    ]
    assert estimate_tokens(msgs) >= 0


def test_estimate_tokens_counts_compaction_part():
    """Studio's compaction content part carries an Anthropic summary
    that can be multi-KB; counting it prevents the threshold check
    from skipping compaction when the real prompt is huge.
    """
    summary = "x" * 4000
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "intro"},
                {"type": "compaction", "content": summary},
            ],
        }
    ]
    # 5 + 4000 chars -> ceil-divided by 4 ~= 1002 tokens.
    assert estimate_tokens(msgs) >= 1000, estimate_tokens(msgs)


# ── Anomaly regressions surfaced by sim_pr5710.py fuzzing ────


def test_pair_linked_indices_skips_non_dict_tool_call_entries():
    """``_assistant_tool_call_ids`` ran inside ``_pair_linked_indices``,
    so a malformed string / None entry in ``tool_calls`` used to crash
    the compactor mid-call. Mirrors the ``estimate_tokens`` guard.
    """
    msgs = [
        {"role": "assistant", "content": "x", "tool_calls": ["bare-string"]},
        {"role": "assistant", "content": "y", "tool_calls": [None, {"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ]
    pm = _pair_linked_indices(msgs)
    assert pm[0] == set()
    assert pm[1] == {2}


def test_compact_drops_orphan_tool_left_by_user_boundary():
    """Tool message arrives after a user / system boundary, references
    an assistant tool_call_id that the boundary-clearing logic no
    longer treats as the pair root. The assistant gets dropped by the
    main sweep; without the final invariant pass the orphan tool
    survives into the output and llama-server rejects the template.
    Repro distilled from sim_pr5710.py fuzz iter 1407.
    """
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task " * 20},
        {"role": "assistant", "content": "a", "tool_calls": [_tool_call("t1")]},
        {"role": "tool", "tool_call_id": "t1", "content": "first"},
        {"role": "user", "content": "next"},
        {"role": "user", "content": "again"},
        {"role": "tool", "tool_call_id": "t1", "content": "stale"},
    ]
    out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 1)
    # Any surviving tool message must have its assistant earlier in
    # the output.
    seen_ids: set[str] = set()
    for m in out:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tcid = tc.get("id") if isinstance(tc, dict) else None
                if isinstance(tcid, str) and tcid:
                    seen_ids.add(tcid)
        elif m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            assert (
                isinstance(tcid, str) and tcid in seen_ids
            ), f"orphan tool {tcid!r} survived; seen={seen_ids}"


def test_compact_keeps_full_pair_when_tool_is_multimodal_anchor():
    """An anchored multimodal tool implies the whole asst+tools pair
    gets anchored. Earlier behavior leaked the orphan tool alone, which
    OpenAI 400s on ("orphan tool message"). Anchor propagation across
    paired indices keeps both halves so the chat template stays valid.
    """
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task " * 20},
        {"role": "assistant", "content": "a", "tool_calls": [_tool_call("t1")]},
        # Real multimodal tool message (image part) -- anchored.
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        },
        {"role": "user", "content": "more"},
        {"role": "user", "content": "even more"},
    ]
    out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 1)
    # Both halves of the pair survive: anchored tool keeps its asst.
    asst_ids, tool_ids = _surviving_tool_ids(out)
    assert "t1" in asst_ids
    assert "t1" in tool_ids
    assert asst_ids == tool_ids


def test_anchored_multimodal_asst_orphan_tool_calls_stripped():
    """Multimodal assistant carrying tool_calls whose tool follow-ups
    all get dropped: the anchor invariant keeps the assistant, but the
    leftover tool_calls field references ids with no matching tool
    response and OpenAI 400s on that shape. Strip the orphan tool_calls
    from a copy so the multimodal content survives and the request
    stays well-formed.
    """
    multimodal_asst = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ],
        "tool_calls": [_tool_call("t1"), _tool_call("t2")],
    }
    msgs = [
        _msg("system", "sys"),
        _msg("user", "task"),
        multimodal_asst,
        _msg("tool", content = "A" * 50, tool_call_id = "t1"),
        _msg("tool", content = "B" * 4000, tool_call_id = "t2"),
        _long("assistant", 50),
        _long("user", 50),
    ]
    out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 100)
    asst_ids, tool_ids = _surviving_tool_ids(out)
    assert asst_ids == tool_ids, (asst_ids, tool_ids)
    # Multimodal content preserved.
    assert any(isinstance(m.get("content"), list) for m in out)
    # Original input unchanged (we copy on rewrite).
    assert multimodal_asst["tool_calls"] == [_tool_call("t1"), _tool_call("t2")]


def test_intervening_assistant_breaks_pair_window():
    """A later assistant turn arriving before the matching tool message
    must end the pending pair window. Otherwise the compactor groups a
    stale assistant + a malformed late tool together. Mirrors the
    user/system boundary rule for the assistant boundary.
    """
    from core.inference.context_compaction import _pair_linked_indices

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "function": {"name": "f", "arguments": "{}"}},
            ],
        },
        {"role": "assistant", "content": "intervening"},
        {"role": "tool", "tool_call_id": "a", "content": "stale"},
    ]
    pm = _pair_linked_indices(msgs)
    # Asst at index 2 had its window closed by asst at index 3.
    assert pm.get(2) == set(), pm
    # The intervening assistant has no tool_calls and so an empty set.
    assert pm.get(3) == set(), pm


def test_compaction_content_part_is_not_multimodal_anchor():
    """Studio's ``{"type":"compaction","content":"..."}`` parts are the
    summary the compactor is supposed to compact away -- they must not
    pin the carrier message as a multimodal anchor. A text-only list
    (only text + compaction parts) collapses to "not multimodal" and
    stays droppable.
    """
    from core.inference.context_compaction import _is_multimodal

    # Pure compaction part.
    msg = {
        "role": "assistant",
        "content": [{"type": "compaction", "content": "OLD " * 200}],
    }
    assert _is_multimodal(msg) is False

    # Mixed text + compaction (still no media payload).
    msg2 = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "intro"},
            {"type": "compaction", "content": "OLD " * 200},
        ],
    }
    assert _is_multimodal(msg2) is False

    # Real multimodal part wins anchoring even alongside compaction.
    msg3 = {
        "role": "assistant",
        "content": [
            {"type": "compaction", "content": "OLD"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ],
    }
    assert _is_multimodal(msg3) is True

    # End-to-end: an old compaction-only message is droppable.
    msgs = [
        _msg("system", "sys"),
        _msg("user", "first task"),
        {
            "role": "assistant",
            "content": [{"type": "compaction", "content": "x" * 8000}],
        },
        _long("assistant", 100),
        _long("user", 100),
    ]
    out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 20)
    # The old compaction-only assistant should have been dropped.
    assert not any(
        m.get("role") == "assistant"
        and isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and p.get("type") == "compaction" for p in m["content"])
        for m in out
    )


def test_partial_tool_drop_strips_orphan_tool_call_id():
    """Same shape but a plain-content assistant: the per-index budget
    loop dropped only one of the two tool follow-ups. The surviving
    tool_call_id on the assistant points at a dropped tool message --
    strip it so the chat template stays valid. Plain assistant is not
    anchored, so the original "drop the assistant when all tools are
    gone" rule would have caught this if BOTH tools had been dropped.
    Here only one tool was dropped, which is the gap this guard closes.
    """
    msgs = [
        _msg("system", "sys"),
        _msg("user", "task"),
        _msg(
            "assistant",
            content = "thinking",
            tool_calls = [_tool_call("t1"), _tool_call("t2")],
        ),
        _msg("tool", content = "A" * 50, tool_call_id = "t1"),
        _msg("tool", content = "B" * 4000, tool_call_id = "t2"),
        _long("assistant", 50),
        _long("user", 50),
    ]
    out = SlidingWindowCompact(keep_recent = 1).compact(msgs, budget_tokens = 200)
    asst_ids, tool_ids = _surviving_tool_ids(out)
    assert asst_ids == tool_ids, (asst_ids, tool_ids)


def test_orphan_tool_before_asst_does_not_leave_dangling_tool_calls():
    """Malformed input: a tool message arrives BEFORE the assistant
    that references its id. The final sweep must drop the orphan tool
    AND strip the now-dangling tool_call from the later assistant so
    the chat template stays valid. Earlier behavior computed
    responded_ids once before the orphan-tool drop, so the post-drop
    asst still matched ids <= responded_ids and kept its tool_calls.
    """
    msgs = [
        _msg("system", "sys"),
        _msg("user", "first task"),
        _msg("tool", content = "early", tool_call_id = "call_a"),
        _msg(
            "assistant",
            content = "thinking",
            tool_calls = [_tool_call("call_a")],
        ),
        _long("user", 50),
    ]
    out = SlidingWindowCompact(keep_recent = 5).compact(msgs, budget_tokens = 1000)
    asst_ids, tool_ids = _surviving_tool_ids(out)
    assert asst_ids == tool_ids, (asst_ids, tool_ids)


def test_anchored_multimodal_orphan_tool_dropped():
    """Anchored multimodal tool message references a tool_call_id no
    surviving assistant declares. Keeping the anchor would leave a
    dangling `tool_call_id` in the output and 400 upstream. Pair
    validity is the hard invariant; the multimodal anchor is the soft
    quality preference, so we drop the orphan rather than violate the
    template. Earlier behavior preserved the anchor and produced an
    invalid chat template.
    """
    msgs = [
        _msg("system", "sys"),
        _msg("user", "first task"),
        {
            "role": "tool",
            "tool_call_id": "stale_call",
            "content": [
                {"type": "text", "text": "image description"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        },
        _msg("user", "follow up"),
        _long("assistant", 100),
    ]
    out = SlidingWindowCompact(keep_recent = 2).compact(msgs, budget_tokens = 5)
    asst_ids, tool_ids = _surviving_tool_ids(out)
    assert asst_ids == tool_ids, (asst_ids, tool_ids)

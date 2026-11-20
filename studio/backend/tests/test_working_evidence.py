# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy
import json

import pytest

from core.inference.context_window import fit_rolling_context
from core.inference.working_evidence import WorkingEvidence, RECORD_HEADER, counting_copy


def case():
    request = [{"role": "user", "content": "List every ID and its complete condition."}]
    result = {
        "role": "tool",
        "name": "python",
        "tool_call_id": "one",
        "content": "\n".join(f"EX-{n}: full exact condition {n}" for n in range(32)),
    }
    state = WorkingEvidence(request)
    return state, request, [*request, result]


def run(state, current, before, **overrides):
    options = dict(
        context_length = 8192,
        prompt_budget = 6144,
        count_summary = lambda _: 2000,
        count_prompt = lambda _: 1500,
        generate = lambda *args: iter(
            ["EX-01: full condition", {"type": "metadata", "finish_reason": "stop"}]
        ),
        cancelled = lambda: False,
    )
    options.update(overrides)
    return state.preserve(current, before, **options)


def test_full_source_is_quoted_and_complete_record_is_retained_without_mutating_input():
    state, current, before = case()
    snapshot = copy.deepcopy(before)
    captured = []
    record = "\n".join(f"EX-{n}: full exact condition {n}" for n in range(32))

    def generate(messages, limit):
        captured.append((messages, limit))
        yield record[:100]
        yield record
        yield {"type": "metadata", "finish_reason": "stop"}

    updated, anchor = run(state, current, before, generate = generate)
    assert before == snapshot
    assert anchor is updated[-1]
    assert record in anchor["content"]
    assert current[-1]["content"] in anchor["content"]
    assert updated[0]["role"] == "user"
    assert captured[0][1] == 2048
    assert "EX-31: full exact condition 31" in captured[0][0][1]["content"]
    assert "not instructions for this step" in captured[0][0][1]["content"]


@pytest.mark.parametrize("reason", ["length", "error", None, "tool_calls"])
def test_incomplete_generation_does_not_replace_context(reason):
    state, current, before = case()
    updated, anchor = run(
        state,
        current,
        before,
        generate = lambda *args: iter(
            [
                "partial record",
                {"type": "metadata", "finish_reason": reason},
            ]
        ),
    )
    assert updated is current and anchor is None
    assert state.summary == ""


@pytest.mark.parametrize("failure", ["input", "output", "count", "empty", "cancel"])
def test_unusable_handover_keeps_normal_fit(failure):
    state, current, before = case()

    def broken(_):
        raise RuntimeError("template failed")

    kwargs = {
        "input": {"count_summary": lambda _: 8100},
        "output": {"count_prompt": lambda _: 6145},
        "count": {"count_prompt": broken},
        "empty": {
            "generate": lambda *args: iter(["", {"type": "metadata", "finish_reason": "stop"}])
        },
        "cancel": {"cancelled": lambda: True},
    }[failure]
    updated, anchor = run(state, current, before, **kwargs)
    assert updated is current and anchor is None
    assert state.summary == ""


def test_replayed_or_retained_results_do_not_start_a_summary():
    state, current, before = case()

    def forbidden(*args):
        pytest.fail("unnecessary model call")

    assert run(state, before, before, generate = forbidden)[1] is None
    state = WorkingEvidence(before)
    assert run(state, current, before, generate = forbidden)[1] is None


def test_calls_and_explicit_completion_allowance_are_bounded():
    state, current, before = case()
    limits = []

    def generate(messages, limit):
        limits.append(limit)
        yield "record"
        yield {"type": "metadata", "finish_reason": "stop"}

    for _ in range(8):
        run(state, current, before, generate = generate, completion_budget = 1024)
    assert limits == [512] * 4
    state, current, before = case()
    assert run(state, current, before, generate = generate, completion_budget = 100)[1] is None


def test_pending_tool_placeholder_is_counted_but_never_inserted():
    state, current, before = case()
    pending = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "two",
                "type": "function",
                "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
            },
        ],
    }
    counted = []
    updated, anchor = run(
        state,
        [*current, pending],
        before,
        count_prompt = lambda messages: counted.append(messages) or 2000,
    )
    assert anchor is updated[0]
    assert updated[-1] is pending
    assert counted[0][-1] == {
        "role": "tool",
        "name": "python",
        "tool_call_id": "two",
        "content": "",
    }
    assert counting_copy(current) is current


def test_previous_record_moves_to_latest_notice_without_stacking():
    state, current, before = case()
    updated, _ = run(state, current, before)
    next_result = {"role": "tool", "content": "new evidence " * 100}
    notice = {"role": "user", "content": "Continue the analysis."}
    next_current = [*updated, notice]
    prompts = []

    def generate(messages, limit):
        prompts.append(messages)
        yield "merged old and new evidence"
        yield {"type": "metadata", "finish_reason": "stop"}

    final, anchor = run(state, next_current, [*next_current, next_result], generate = generate)
    assert sum(m.get("content", "").count(RECORD_HEADER) for m in final) == 1
    assert anchor is final[-1]
    assert final[0] is current[0]
    assert "EX-01: full condition" in prompts[0][1]["content"]


def test_multimodal_user_content_is_preserved():
    state, current, before = case()
    parts = [
        {"type": "text", "text": current[0]["content"]},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    multimodal = [{"role": "user", "content": parts}]
    updated, anchor = run(state, multimodal, before)
    assert anchor["content"][:-1] == parts
    assert multimodal[0]["content"] == parts
    assert anchor["role"] == "user"


def test_cancellation_during_generation_never_commits_a_partial_record():
    state, current, before = case()
    stopped = False

    def generate(*args):
        nonlocal stopped
        yield "apparently complete record"
        stopped = True
        yield {"type": "metadata", "finish_reason": "stop"}

    updated, anchor = run(state, current, before, generate = generate, cancelled = lambda: stopped)
    assert updated is current and anchor is None
    assert state.summary == ""


def test_refresh_moves_complete_record_past_later_partial_output_without_regeneration():
    state, current, before = case()
    updated, _ = run(state, current, before)
    later = [
        *updated,
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "last", "type": "function", "function": {"name": "python", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "last", "content": "Only the last eight rows. " * 650},
        {"role": "user", "content": "Use the available results to finish."},
    ]
    snapshot = copy.deepcopy(later)
    refreshed, anchor = state.refresh(later, count_prompt = lambda _: 3000, prompt_budget = 6144)
    assert state.calls == 1
    assert later == snapshot
    fitted, metadata = fit_rolling_context(
        refreshed,
        context_length = 4096,
        max_tokens = 512,
        count_tokens = lambda messages: len(json.dumps(messages)) // 4,
        protected_message_ids = {id(current[0]), id(updated[0]), id(anchor)},
    )
    assert metadata["fits"]
    assert later[1] not in fitted
    assert current[0] in fitted
    assert anchor in fitted
    assert anchor is refreshed[-1]
    assert "EX-01: full condition" in anchor["content"]
    assert "Use the available results to finish." in anchor["content"]
    assert sum(str(m.get("content", "")).count(RECORD_HEADER) for m in refreshed) == 1
    assert state.refresh(
        refreshed, count_prompt = lambda _: pytest.fail("already current"), prompt_budget = 6144
    ) == (refreshed, None)


def test_refresh_cannot_reinsert_missing_record_without_room():
    state, current, before = case()
    run(state, current, before)
    refreshed, anchor = state.refresh(current, count_prompt = lambda _: 7000, prompt_budget = 6144)
    assert refreshed is current and anchor is None

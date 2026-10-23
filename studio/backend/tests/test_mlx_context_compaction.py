# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MLX context fitting: the orchestrator preflight and the tool loop's per-turn refit."""

from unittest.mock import Mock

import pytest

import core.inference.tools as tools_mod
from core.inference import llama_cpp
from core.inference.context_window import retrieval_budget
from core.inference.orchestrator import InferenceOrchestrator

INSTRUCTION = "Standing instruction for the rest of this task: always answer in a markdown table."
SEARCH = {"type": "function", "function": {"name": "search_conversation"}}
WEB = {"type": "function", "function": {"name": "web_search"}}


def _count(messages, *_args, **_kwargs):
    return sum(max(1, len(str(m.get("content", ""))) // 4) for m in messages), "mlx-test"


def _backend():
    backend = InferenceOrchestrator()
    backend.active_model_name = "mlx-test"
    backend.models["mlx-test"] = {"is_mlx": True, "context_length": 1200}
    backend.count_chat_tokens = _count
    return backend


def _thread(latest = "continue"):
    messages = [{"role": "user", "content": INSTRUCTION}, {"role": "assistant", "content": "OK."}]
    for index in range(8):
        messages.append({"role": "user", "content": f"Section {index}. " + "x" * 600})
        messages.append({"role": "assistant", "content": f"Section {index} noted."})
    return messages + [{"role": "user", "content": latest}]


def _fit(backend, messages, **kwargs):
    kwargs.setdefault("context_overflow", "truncate_oldest")
    return backend.compact_chat_context(
        messages, system_prompt = "you are helpful", max_tokens = 200, **kwargs
    )


@pytest.fixture
def archive_calls(monkeypatch):
    """A saved thread with a healthy archive, so only the request's own gates decide."""
    calls = []
    idle = {"events": [], "counts": {}, "recalled": False, "anchored": []}

    def _archive(conversation, _before, **kwargs):
        calls.append({**kwargs, "branch_messages": list(kwargs["branch_messages"])})
        return {"conversation": conversation, **idle}

    def _can_reset(
        thread_id,
        supports_tools,
        *,
        tools_withheld = False,
    ):
        return bool(thread_id and supports_tools and not tools_withheld)

    monkeypatch.setattr(llama_cpp, "_archive_and_recall", _archive)
    monkeypatch.setattr(llama_cpp, "_can_reset_epoch", _can_reset)
    monkeypatch.setattr(llama_cpp, "_archive_is_degraded", lambda: False)
    return calls


# A 4200-character question is over the prompt budget even alone: a rescued refusal.
@pytest.mark.parametrize("question_chars, fits", [(8, True), (4200, False)])
def test_an_overflowing_prompt_drops_its_oldest_turns_and_reports_the_boundary(
    archive_calls, question_chars, fits
):
    messages = _thread("q" * question_chars)

    result = _fit(_backend(), messages, thread_id = "thread-1")

    [truncation] = result["events"]
    assert truncation["type"] == "context_truncated"
    assert truncation["fits"] is fits is (truncation["prompt_tokens_after"] <= 1000)
    assert truncation["boundary_messages"] == truncation["dropped_messages"] > 0
    assert archive_calls[-1]["recall_done"] is not fits
    assert result["system_prompt"] == ""
    assert result["messages"][0] == {"role": "system", "content": "you are helpful"}
    assert result["messages"][-1] is messages[-1]


@pytest.mark.parametrize(
    "case", ["not_opted_in", "not_mlx", "media", "resumed_reply", "resumed_thought", "no_count"]
)
def test_a_prompt_the_fit_must_not_touch_is_returned_unchanged(case):
    backend, messages, kwargs = _backend(), _thread(), {}
    if case == "not_opted_in":
        kwargs["context_overflow"] = "error"
    elif case == "not_mlx":
        backend.models["mlx-test"]["is_mlx"] = False
    elif case == "media":
        messages[-1] = {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
    elif case.startswith("resumed"):
        partial = {"content": "so far"} if case == "resumed_reply" else {"reasoning_content": "hm"}
        messages.append({"role": "assistant", "content": "", **partial})
        kwargs["continue_final_message"] = True
    else:
        backend.count_chat_tokens = Mock(side_effect = RuntimeError("generation in progress"))

    result = _fit(backend, messages, **kwargs)

    assert result["messages"] == messages
    assert result["system_prompt"] == "you are helpful"
    assert result["events"] == [] and "boundary_applied" not in result


def test_a_request_resets_only_where_search_can_follow_and_names_only_an_offered_tool(
    archive_calls, monkeypatch
):
    def fit(messages = None, **kwargs):
        result = _fit(
            _backend(),
            messages or _thread(),
            context_policy = "checkpoint",
            thread_id = "thread-1",
            **kwargs,
        )
        return result["events"][-1], result["messages"][0]["content"], archive_calls[-1]

    truncation, system, archived = fit(tools = [SEARCH])
    assert not truncation.get("checkpoint") and "carried_forward" not in system
    assert archived["style"] == "inline"

    # A plain request has no catalogue to miss the tool from: the route's answer alone decides.
    with monkeypatch.context() as patch:
        patch.setattr(llama_cpp, "_memory_tool_withheld", lambda thread_id, tools: True)
        truncation, system, archived = fit(recall_reachable = True)
        assert truncation["checkpoint_started"] is True and "carried_forward" in system
        assert "search_conversation tool" not in system and archived["style"] == "inline"
        assert not fit(tool_loop = True, tools = [WEB])[0].get("checkpoint")

    truncation, system, archived = fit(tool_loop = True, tools = [WEB])
    assert truncation["checkpoint_started"] is True and "carried_forward" in system
    assert "search_conversation tool" not in system
    assert archived["style"] == "inline"

    truncation, system, archived = fit(tool_loop = True, tools = [WEB, SEARCH])
    assert "search_conversation tool" in system
    assert archived["style"] == "tool"
    halved = retrieval_budget(1200, 200, truncation["prompt_tokens_after"], reply_returns = True)
    assert archived["recall_budget_tokens"] == halved
    assert halved < retrieval_budget(1200, 200, truncation["prompt_tokens_after"])

    turns = [{"role": "user", "content": "x" * size} for size in (3200, 800, 8)]
    truncation, _, archived = fit(messages = turns, tool_loop = True, recall_reachable = True)
    after = truncation["prompt_tokens_after"]
    whole = retrieval_budget(1200, 200, after)
    assert not truncation.get("checkpoint") and archived["recall_budget_tokens"] == whole
    assert whole > retrieval_budget(1200, 200, after, reply_returns = True)


QUESTION = {"role": "user", "content": "What changed in release 4?"}
ANSWERED = {"role": "assistant", "content": "OK."}
CALL = '<tool_call>{"name":"web_search","arguments":{"query":"release %d"}}</tool_call>'


def _run_loop(monkeypatch, messages, replies, result_chars, **kwargs):
    backend, turns, prompts, branches = _backend(), iter(replies), [], []

    def _generate(**call):
        prompts.append(list(call["messages"]))
        yield next(turns)

    def _execute(*_args, conversation_branch, **_kwargs):
        branches.append(list(conversation_branch))
        return f"result {len(branches)} " + "r" * result_chars

    monkeypatch.setattr(backend, "generate_chat_response", _generate)
    monkeypatch.setattr(tools_mod, "execute_tool", _execute)
    events = backend.generate_chat_completion_with_tools(
        messages = messages,
        tools = [WEB],
        max_tokens = 200,
        nudge_tool_calls = True,
        thread_id = "thread-1",
        context_overflow = "truncate_oldest",
        **kwargs,
    )
    truncated = [event for event in events if event.get("type") == "context_truncated"]
    return truncated, prompts, branches


@pytest.mark.parametrize("policy", ["checkpoint", "rolling"])
@pytest.mark.parametrize("old_result_chars, fits", [(3900, 3), (0, 2)])
def test_the_tool_loop_refits_every_turn_and_keeps_the_question_past_a_reprompt(
    monkeypatch, archive_calls, policy, old_result_chars, fits
):
    old = {"role": "user", "content": "o" * 400}
    old_exchange = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "0"}]},
        {"role": "tool", "content": "h" * old_result_chars},
    ]
    truncated, prompts, branches = _run_loop(
        monkeypatch,
        [old, *(old_exchange if old_result_chars else []), ANSWERED, QUESTION],
        ["Let me search the web for that.", CALL % 1, CALL % 2, CALL % 3, "done"],
        2400,
        max_tool_iterations = 3,
        context_policy = policy,
    )

    final = prompts[-1]
    assert final[-1]["role"] == "user" and final[-1] is not QUESTION
    assert any(m is QUESTION for m in final) and not any(m is old for m in final)
    assert [m["content"][:8] for m in final if m.get("role") == "tool"] == ["result 3"]
    assert sum(m["content"].startswith("result") for m in branches[2] if m["role"] == "tool") == 2
    last_fit = archive_calls[-1]
    ran = [m for m in last_fit["branch_messages"] if m["role"] == "tool"][-3:]
    assert [m["content"][:8] for m in ran] == ["result 1", "result 2", "result 3"]
    assert [m for m in last_fit["branch_messages"] if m["role"] == "user"] == [old, QUESTION]
    assert len(truncated) == fits and all(event["fits"] for event in truncated)
    assert bool(truncated[0].get("checkpoint_started")) is (policy == "checkpoint")
    assert last_fit["thread_id"] == "thread-1"


# A result overflowing by less than one old turn, against a boundary two turns deep; or neither.
@pytest.mark.parametrize("result_chars, saved", [(3300, 4), (8, 0)])
def test_a_turn_that_is_not_fitted_leaves_the_saved_boundary_to_the_next(
    monkeypatch, archive_calls, result_chars, saved
):
    asked = []

    def _saved_boundary(*_args, **_kwargs):
        asked.append(1)
        return saved, False

    monkeypatch.setattr(llama_cpp, "_sticky_compaction_state", _saved_boundary)
    first, second = ({"role": "user", "content": letter * 400} for letter in "ab")
    resumed = {"role": "assistant", "content": "Let me"}
    _, prompts, _ = _run_loop(
        monkeypatch,
        [first, ANSWERED, second, ANSWERED, QUESTION, resumed],
        [CALL % 1, CALL % 2, "done"],
        result_chars,
        continue_final_message = True,
        context_policy = "rolling",
    )

    assert any(m is first for m in prompts[0])
    assert any(m is second for m in prompts[1]) == (not saved)
    assert len(asked) == 1


# The cap's budget notice; then the no-op notice a repeated call earns, once, twice, and
# twice with the model's own words in between.
@pytest.mark.parametrize(
    "replies, chars, fits, dropped",
    [
        ([CALL % 1], 3900, False, 0),
        ([CALL % 1] * 2, 3900, False, 0),
        ([CALL % 1] * 3, 3550, True, 1),
        ([CALL % 1] * 2 + ["I'll check that again. " + CALL % 1], 3550, True, 2),
    ],
)
def test_the_only_tool_result_is_not_evicted_to_make_room_for_what_the_loop_says_next(
    monkeypatch, archive_calls, replies, chars, fits, dropped
):
    [truncation], prompts, _ = _run_loop(
        monkeypatch,
        [QUESTION],
        [*replies, "done"],
        chars,
        max_tool_iterations = 5 if len(replies) > 1 else 1,
    )

    # Dropping the result would fit but leave nothing to answer from.
    assert truncation["fits"] is fits and truncation["dropped_messages"] == dropped
    assert [m["role"] for m in prompts[-1]].count("tool") == 1

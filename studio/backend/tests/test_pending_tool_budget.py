# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from core.inference import llama_cpp, tools
from core.inference.working_evidence import WorkingEvidence
from test_compact_arguments_before_results import exchange
from test_llama_cpp_tool_loop import _make_backend, _tool_call_sse, _done, _sse, _finish


def count(messages, *args, **kwargs):
    return 500 + sum(len(json.dumps(m)) // 4 for m in messages)


def make_batch(monkeypatch, batch_size):
    code = "print('records')\n" + "# inspect records\n" * 340
    codes = [code + f"\n# batch {i}" for i in range(batch_size)]
    streams = [
        [_tool_call_sse("python", {"code": c}, f"new-{i}", index = i) for i, c in enumerate(codes)]
        + [_finish("tool_calls"), _done()],
        [_sse({"content": "Done."}), _done()],
    ]
    payloads = []
    backend = _make_backend(monkeypatch, streams, payloads)
    backend._effective_context_length = 8192
    backend._supports_tools = True

    monkeypatch.setattr(backend, "count_chat_tokens", count)
    return backend, codes, payloads


def document_history():
    return [
        {"role": "user", "content": "Extract all exceptions."},
        *exchange("old-0", "print(1)", "old output " * 750),
        *exchange("old-1", "print(2)", "old output " * 750),
    ]


@pytest.mark.parametrize("overflow", ["truncate_oldest", None])
def test_generated_arguments_cannot_starve_an_evictable_tool_result(monkeypatch, overflow):
    backend, codes, _ = make_batch(monkeypatch, 2)
    messages = document_history()
    original = copy.deepcopy(messages)
    calls = []

    def execute(
        name,
        arguments,
        result_budget_tokens = None,
        **kwargs,
    ):
        calls.append((name, arguments, result_budget_tokens))
        return "All extracted records."

    monkeypatch.setattr(tools, "execute_tool", execute)
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = messages,
            tools = [tools.PYTHON_TOOL],
            max_tokens = None,
            max_tool_iterations = 1,
            context_overflow = overflow,
        )
    )
    assert [c[1]["code"] for c in calls] == codes
    if overflow:
        assert all(c[2] >= 512 for c in calls)
    assert messages == original
    compacted = any(
        e.get("type") == "context_truncated" and e.get("dropped_messages", 0) > 0 for e in events
    )
    assert compacted is bool(overflow)


def test_resumed_batch_keeps_both_calls_in_the_next_prompt(monkeypatch):
    backend, _, payloads = make_batch(monkeypatch, 2)
    monkeypatch.setattr(tools, "execute_tool", lambda *args, **kwargs: "Records.")
    list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "Extract all exceptions." + " supplied facts" * 780},
                {"role": "assistant", "content": "Inspecting the records."},
            ],
            tools = [tools.PYTHON_TOOL],
            context_overflow = "truncate_oldest",
            continue_final_message = True,
            max_tool_iterations = 1,
        )
    )
    prompt = payloads[-1]["messages"]
    assistant = next(m for m in prompt if m.get("tool_calls"))
    assert [c["id"] for c in assistant["tool_calls"]] == ["new-0", "new-1"]
    assert [m["tool_call_id"] for m in prompt if m["role"] == "tool"] == ["new-0", "new-1"]


@pytest.mark.parametrize("result_size", [40, 12000])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_recalled_history_uses_only_room_left_after_the_live_result(
    monkeypatch, result_size, batch_size
):
    backend, _, _ = make_batch(monkeypatch, batch_size)
    messages = document_history()
    stages = []
    budgets = []

    pending_handover_limits = []
    preserve = WorkingEvidence.preserve

    def record_limit(state, *args, **kwargs):
        if stages == ["archive"]:
            pending_handover_limits.append(kwargs["prompt_budget"])
        return preserve(state, *args, **kwargs)

    monkeypatch.setattr(WorkingEvidence, "preserve", record_limit)

    def execute(name, arguments, **kwargs):
        stages.append("execute")
        return "L" * result_size

    def archive(conversation, before, **kwargs):
        if kwargs.get("force_recall", True):
            stages.append("recall")
            assert stages.count("execute") == batch_size
            assert any(m.get("content") == "L" * result_size for m in conversation)
            assert callable(kwargs.get("count_tokens"))
            budget = kwargs["recall_budget_tokens"]
            budgets.append(budget)
            assert budget <= max(0, 6144 - count(conversation) - 2048)
        else:
            stages.append("archive")
        return {
            "conversation": conversation,
            "events": [],
            "counts": {},
            "recalled": False,
            "anchored": [],
        }

    monkeypatch.setattr(tools, "execute_tool", execute)
    monkeypatch.setattr(llama_cpp, "_archive_and_recall", archive)
    list(
        backend.generate_chat_completion_with_tools(
            messages = messages,
            tools = [tools.PYTHON_TOOL],
            max_tokens = None,
            max_tool_iterations = 1,
            context_overflow = "truncate_oldest",
        )
    )
    assert pending_handover_limits
    assert all(
        llama_cpp.tool_result_budget(8192, None, limit) >= 2048 for limit in pending_handover_limits
    )
    assert stages[0] == "archive"
    assert stages.count("execute") == batch_size
    assert "recall" in stages
    assert budgets and (budgets[0] > 0) is (result_size == 40)

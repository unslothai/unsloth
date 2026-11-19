# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from core.inference import tools
from test_llama_cpp_tool_loop import _make_backend, _tool_call_sse, _done, _sse, _finish


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("overflow", ["truncate_oldest", None])
def test_generated_arguments_cannot_starve_an_evictable_tool_result(monkeypatch, overflow, batch_size):
    code = "print('records')\n" + "# inspect records\n" * 340
    codes = [code + f"\n# batch {i}" for i in range(batch_size)]
    streams = [
        [_tool_call_sse("python", {"code": c}, f"new-{i}", index=i) for i, c in enumerate(codes)] + [_finish("tool_calls"), _done()],
        [_sse({"content": "Done."}), _done()],
    ]
    payloads = []
    backend = _make_backend(monkeypatch, streams, payloads)
    backend._effective_context_length = 8192
    backend._supports_tools = True
    def count(messages, *args, **kwargs):
        return 500 + sum(len(json.dumps(m)) // 4 for m in messages)
    monkeypatch.setattr(backend, "count_chat_tokens", count)
    messages = [
        {"role": "system", "content": "Analyze documents with Python."},
        {"role": "user", "content": "Extract all exceptions from agreement.docx."},
    ]
    for i in range(2):
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"old-{i}", "type": "function",
                "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
            }]},
            {"role": "tool", "name": "python", "tool_call_id": f"old-{i}",
             "content": "old output " * 750},
        ]
    original = copy.deepcopy(messages)
    calls = []
    def execute(name, arguments, result_budget_tokens=None, **kwargs):
        calls.append((name, arguments, result_budget_tokens))
        return "All extracted records."
    monkeypatch.setattr(tools, "execute_tool", execute)
    events = list(backend.generate_chat_completion_with_tools(
        messages=messages, tools=[tools.PYTHON_TOOL], max_tokens=None,
        max_tool_iterations=1, context_overflow=overflow,
    ))
    assert len(calls) == batch_size
    assert [c[1]["code"] for c in calls] == codes
    if overflow:
        assert all(c[2] >= 1024 // batch_size for c in calls), [c[2] for c in calls]
    assert messages == original
    compacted = any(e.get("type") == "context_truncated" and e.get("dropped_messages", 0) > 0 for e in events)
    assert compacted is bool(overflow)


@pytest.mark.parametrize("result_size", [40, 12000])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_recalled_history_uses_only_room_left_after_the_live_result(monkeypatch, result_size, batch_size):
    from core.inference import llama_cpp
    code = "print('records')\n" + "# inspect records\n" * 340
    streams = [
        [_tool_call_sse("python", {"code": code + f"\n# {i}"}, f"new-{i}", index=i)
         for i in range(batch_size)] + [_finish("tool_calls"), _done()],
        [_sse({"content": "Done."}), _done()],
    ]
    backend = _make_backend(monkeypatch, streams, [])
    backend._effective_context_length = 8192
    backend._supports_tools = True
    def count(messages, *args, **kwargs):
        return 500 + sum(len(json.dumps(m)) // 4 for m in messages)
    monkeypatch.setattr(backend, "count_chat_tokens", count)
    messages = [
        {"role": "system", "content": "Analyze documents with Python."},
        {"role": "user", "content": "Extract all exceptions."},
    ]
    for index in range(2):
        messages.extend([
            {"role": "assistant", "tool_calls": [{
                "id": f"old-{index}", "type": "function",
                "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
            }]},
            {"role": "tool", "name": "python", "tool_call_id": f"old-{index}",
             "content": "old output " * 750},
        ])
    stages = []
    budgets = []
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
        return {"conversation": conversation, "events": [], "counts": {},
                "recalled": False, "anchored": []}
    monkeypatch.setattr(tools, "execute_tool", execute)
    monkeypatch.setattr(llama_cpp, "_archive_and_recall", archive)
    list(backend.generate_chat_completion_with_tools(
        messages=messages, tools=[tools.PYTHON_TOOL], max_tokens=None,
        max_tool_iterations=1, context_overflow="truncate_oldest",
    ))
    assert stages[0] == "archive"
    assert stages.count("execute") == batch_size
    assert "recall" in stages
    assert budgets and (budgets[0] > 0) is (result_size == 40)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from core.inference import llama_cpp, tools
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


@pytest.mark.parametrize("batch_size", [2, 3])
def test_completed_siblings_stay_compactable_for_the_result_budget_rescue(monkeypatch, batch_size):
    backend, _, payloads = make_batch(monkeypatch, batch_size)
    backend._effective_context_length = 4608
    budgets = []

    def execute(
        name,
        arguments,
        result_budget_tokens = None,
        **kwargs,
    ):
        budgets.append(result_budget_tokens)
        return "Records."

    monkeypatch.setattr(tools, "execute_tool", execute)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Extract all exceptions."}],
            tools = [tools.PYTHON_TOOL],
            max_tokens = None,
            max_tool_iterations = 1,
            context_overflow = "truncate_oldest",
        )
    )
    ids = [f"new-{i}" for i in range(batch_size)]
    assert len(budgets) == batch_size
    assert budgets[-1] >= llama_cpp._MIN_USEFUL_RESULT_TOKENS, budgets
    prompt = payloads[-1]["messages"]
    assistant = next(m for m in prompt if m.get("tool_calls"))
    assert [c["id"] for c in assistant["tool_calls"]] == ids
    assert [m["tool_call_id"] for m in prompt if m["role"] == "tool"] == ids

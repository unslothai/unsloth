# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy
import json

import pytest

from core.inference import llama_cpp
from core.inference.context_window import compact_completed_tool_arguments


def exchange(call_id, code, result):
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
         "function": {"name": "python", "arguments": json.dumps({"code": code})}}]},
        {"role": "tool", "name": "python", "tool_call_id": call_id, "content": result},
    ]


def count(messages):
    return 150 + sum(len(json.dumps(m)) // 4 for m in messages)


def test_completed_code_is_compacted_before_findings_are_evicted():
    messages = [
        {"role": "system", "content": "Analyze the document."},
        {"role": "user", "content": "Find every exception."},
        *exchange("one", "# inspect the input\n" * 800, "EARLIER FINDINGS " * 200),
        *exchange("two", "print('later findings')", "LATER FINDINGS"),
    ]
    before = copy.deepcopy(messages)
    first_result = messages[3]
    fitted, metadata = llama_cpp._fit_with_instruction_pins(
        messages, context_length=4096, max_tokens=512, count_tokens=count,
        context_policy="rolling", can_reset=False,
    )
    assert first_result in fitted
    assert fitted[-1] is messages[-1]
    assert count(fitted) < 3584
    assert metadata is None or metadata.get("dropped_messages", 0) == 0
    assert messages == before
    assert len(fitted[2]["tool_calls"][0]["function"]["arguments"]) < 500


@pytest.mark.parametrize("result", ["Exit code 1: NameError: rows", "Error: invalid data", "Traceback: failure", "Execution timed out after 20 seconds"])
def test_failed_code_remains_available_for_repair(result):
    messages = exchange("failed", "# repair this\n" * 200, result)
    fitted, changed = compact_completed_tool_arguments(messages, preserve_failed=True)
    assert changed == 0
    assert fitted is messages


def test_a_pending_call_with_a_counting_probe_is_not_compacted():
    completed = exchange("done", "# finished\n" * 200, "done")
    pending = exchange("pending", "# not run yet\n" * 400, "")
    messages = [{"role": "user", "content": "Continue."}, *completed, *pending]
    original = copy.deepcopy(pending[0])
    fitted, _ = llama_cpp._fit_with_instruction_pins(
        messages, context_length=4096, max_tokens=512, prompt_reserve_tokens=2048,
        count_tokens=count, argument_anchor_ids={id(pending[0])},
        context_policy="rolling", can_reset=False,
    )
    live = next(m for m in fitted if m.get("tool_calls") and m["tool_calls"][0]["id"] == "pending")
    assert live == original


def test_unmeasurable_or_already_fitting_history_keeps_arguments(monkeypatch):
    messages = exchange("done", "# code\n" * 200, "done")
    def fail(_):
        raise RuntimeError("tokenizer unavailable")
    for counter in (lambda _: 100, fail):
        monkeypatch.setattr(llama_cpp, "_fit_context", lambda current, **kwargs: (current, None))
        fitted, _ = llama_cpp._fit_with_instruction_pins(
            messages, context_length=4096, max_tokens=None, count_tokens=counter,
        )
        assert fitted is messages


@pytest.mark.parametrize("context_length", [None, 0, 1])
def test_unknown_context_does_not_compact_arguments(monkeypatch, context_length):
    messages = exchange("done", "# code\n" * 200, "done")
    monkeypatch.setattr(llama_cpp, "_fit_context", lambda current, **kwargs: (current, None))
    fitted, _ = llama_cpp._fit_with_instruction_pins(
        messages, context_length=context_length, max_tokens=None, count_tokens=lambda _: 9000,
    )
    assert fitted is messages


def test_optional_result_reserve_yields_before_discarding_existing_findings():
    messages = [
        {"role": "system", "content": "Analyze the document."},
        {"role": "user", "content": "List all findings."},
        *exchange("one", "print(records)", "EARLIER FINDINGS " * 1200),
        *exchange("pending", "print(sections)", ""),
    ]
    assert 4096 < count(messages) < 6144
    first_result = messages[3]
    fitted, metadata = llama_cpp._fit_with_instruction_pins(
        messages, context_length=8192, max_tokens=None, prompt_reserve_tokens=2048,
        count_tokens=count, context_policy="rolling", can_reset=False,
        argument_anchor_ids={id(messages[-2])},
        reserve_anchor_ids={id(first_result)},
    )
    assert first_result in fitted
    assert metadata is None or metadata.get("dropped_messages", 0) == 0
    assert count(fitted) < 6144

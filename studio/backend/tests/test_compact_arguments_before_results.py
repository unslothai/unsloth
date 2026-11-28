# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy
import json

import pytest

from core.inference import llama_cpp
from core.inference.context_window import compact_completed_tool_arguments


def exchange(call_id, code, result):
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "python", "arguments": json.dumps({"code": code})},
                }
            ],
        },
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
        messages,
        context_length = 4096,
        max_tokens = 512,
        count_tokens = count,
        context_policy = "rolling",
        can_reset = False,
    )
    assert first_result in fitted
    assert fitted[-1] is messages[-1]
    assert count(fitted) < 3584
    assert metadata is None or metadata.get("dropped_messages", 0) == 0
    assert messages == before
    assert len(fitted[2]["tool_calls"][0]["function"]["arguments"]) < 500


@pytest.mark.parametrize(
    "result",
    [
        "Exit code 1: NameError: rows",
        "Error: invalid data",
        "Traceback: failure",
        "Execution timed out after 20 seconds",
        [{"type": "text", "text": "Exit code 1: NameError: rows"}],
        pytest.param("row\n" * 100 + "Execution timed out after 20 seconds.", id = "late-timeout"),
    ],
)
def test_failed_code_remains_available_for_repair(result):
    messages = exchange("failed", "# repair this\n" * 200, result)
    fitted, changed = compact_completed_tool_arguments(messages, preserve_failed = True)
    assert changed == 0
    assert fitted is messages


def test_saved_boundary_counts_evictions_after_argument_compaction():
    history = [
        {"role": "user", "content": "old " * 4200},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "Inspect the source"},
        *exchange("one", "# inspect the input\n" * 800, "EARLIER FINDINGS"),
        {"role": "user", "content": "List the findings."},
    ]
    fitted, cut = llama_cpp._fit_with_instruction_pins(
        history,
        context_length = 4096,
        max_tokens = 512,
        count_tokens = count,
        anchor_ids = {id(history[2])},
        context_policy = "rolling",
        can_reset = False,
    )
    assert cut["dropped_messages"] == 2
    boundary = llama_cpp._boundary_metadata(fitted, history)
    assert boundary["boundary_messages"] == 2
    assert boundary["boundary_anchor"] == llama_cpp._anchor_text(history[2])


def test_unmeasurable_or_already_fitting_history_keeps_arguments(monkeypatch):
    messages = exchange("done", "# code\n" * 200, "done")

    def fail(_):
        raise RuntimeError("tokenizer unavailable")

    for counter in (lambda _: 100, fail):
        monkeypatch.setattr(llama_cpp, "_fit_context", lambda current, **kwargs: (current, None))
        fitted, _ = llama_cpp._fit_with_instruction_pins(
            messages,
            context_length = 4096,
            max_tokens = None,
            count_tokens = counter,
        )
        assert fitted is messages


def test_a_compaction_that_still_overflows_keeps_the_eviction_rescue():
    # Compaction lands this prompt between the budget and n_ctx; the rescue must still evict.
    partial = {"role": "assistant", "content": "x" * 15000}
    nudge = {"role": "user", "content": "go"}
    messages = [
        {"role": "user", "content": "q"},
        *exchange("a", "# step\n" * 200, "facts " * 50),
        nudge,
        partial,
    ]
    count = lambda fitted: sum(len(json.dumps(m)) // 4 for m in fitted)
    assert count(messages) >= 4096
    fitted, truncation = llama_cpp._fit_with_instruction_pins(
        messages,
        anchor_ids = {id(partial), id(nudge)},
        context_length = 4096,
        max_tokens = 1024,
        count_tokens = count,
        estimate_message = lambda message: count([message]),
    )
    assert truncation["dropped_messages"] > 0
    assert fitted[-1] is partial

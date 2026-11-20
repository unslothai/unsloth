# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from core.inference.checkpoint import fit_checkpoint_context
from core.inference.context_window import fit_rolling_context
from core.inference.llama_cpp import _fit_with_instruction_pins
from test_compact_arguments_before_results import exchange


def count(messages):
    return sum(len(json.dumps(message)) // 4 for message in messages)


def conversation():
    messages = [
        {"role": "system", "content": "Use the document to answer."},
        {"role": "user", "content": "Identify every exception in agreement.docx."},
    ]
    for index in range(3):
        messages.extend(exchange(str(index), "print(records)", "exception facts " * 400))
    return messages


@pytest.mark.parametrize("policy", ["rolling", "checkpoint"])
def test_clipped_output_recovery_reserves_space_and_keeps_latest_exchange(policy):
    messages = conversation()
    fit = fit_checkpoint_context if policy == "checkpoint" else fit_rolling_context
    options = {"can_reset": True} if policy == "checkpoint" else {}
    before, change = fit(
        messages, context_length = 8192, max_tokens = None, count_tokens = count, **options
    )
    assert change is None
    assert before == messages

    fitted, change = fit(
        messages,
        context_length = 8192,
        max_tokens = None,
        count_tokens = count,
        prompt_reserve_tokens = 2048,
        **options,
    )
    assert change["fits"]
    assert change["dropped_messages"] > 0
    assert change["prompt_tokens_after"] == count(fitted)
    assert count(fitted) <= 4096
    assert fitted[-2:] == messages[-2:]
    assert any(m.get("content") == messages[1]["content"] for m in fitted)
    calls = {c["id"] for m in fitted for c in m.get("tool_calls", [])}
    assert all(m["tool_call_id"] in calls for m in fitted if m["role"] == "tool")


def test_reserve_cannot_refuse_a_user_message_that_fits():
    messages = [{"role": "user", "content": "document " * 2100}]
    assert 4096 < count(messages) < 6144
    fitted, change = _fit_with_instruction_pins(
        messages,
        context_length = 8192,
        max_tokens = None,
        count_tokens = count,
        can_reset = False,
        context_policy = "rolling",
        prompt_reserve_tokens = 2048,
    )
    assert fitted == messages
    assert change is None or change["fits"]

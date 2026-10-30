# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Claude's later system messages must not invalidate a stable system prefix."""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.inference import AnthropicMessagesRequest
from core.inference.anthropic_compat import anthropic_messages_to_openai


REMINDER = (
    "Concise output style is active. Remember to follow the specific guidelines for this style."
)
STYLE = f"# Output Style: Concise\nUse short answers.\n\n{REMINDER}\n\nToday's date is fixed."


def note(text):
    return f"<system-reminder>\n{text}\n</system-reminder>"


def converted(payload):
    request = AnthropicMessagesRequest.model_validate(payload).model_dump()
    return anthropic_messages_to_openai(request["messages"], request["system"])


@pytest.mark.parametrize("system_is_list", [False, True])
@pytest.mark.parametrize("style_in_messages", [False, True])
def test_repeated_style_preserves_system_prefix_and_tool_round(system_is_list, style_in_messages):
    system = "Base instructions." if style_in_messages else STYLE
    if system_is_list:
        system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    payload = {"system": system, "messages": [{"role": "user", "content": "Read a file."}]}
    if style_in_messages:
        payload["messages"].insert(0, {"role": "system", "content": STYLE})
    before = converted(payload)
    followup = copy.deepcopy(payload)
    followup["messages"] += [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "read_1",
                    "name": "Read",
                    "input": {"path": "fixture.txt"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "read_1", "content": "orchard"}],
        },
        {"role": "system", "content": REMINDER},
        {"role": "system", "content": [{"type": "text", "text": REMINDER}]},
    ]
    original = copy.deepcopy(followup)
    after = converted(followup)
    assert after[: len(before)] == before
    assert after[-2]["tool_calls"][0]["id"] == "read_1"
    assert after[-1] == {"role": "tool", "tool_call_id": "read_1", "content": "orchard"}
    assert followup == original


@pytest.mark.parametrize(
    "initial,addition",
    [
        ("Base instructions.", REMINDER),
        ("# Output Style: Other\nUse long answers.\n\n" + REMINDER, REMINDER),
        ("# Output Style: Concise\nUse short answers.", REMINDER),
        (STYLE, REMINDER + " Also include examples."),
        (
            STYLE,
            "Other output style is active. Remember to follow the specific guidelines for this style.",
        ),
        (STYLE, "Use short answers."),
        (STYLE, "<total_tokens>14999000 tokens left</total_tokens>"),
        ("# Output Style: Concise\nUse short answers.\n\nQuoted: " + REMINDER, REMINDER),
        (STYLE, "New instructions.\n" + REMINDER),
    ],
)
def test_nonredundant_system_instructions_are_preserved(initial, addition):
    result = converted(
        {
            "system": initial,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": addition},
            ],
        }
    )
    assert result[0]["content"] == initial
    assert result[1] == {"role": "user", "content": f"Hello\n{note(addition)}"}


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_conversation_text_cannot_suppress_a_system_instruction(role):
    result = converted(
        {
            "system": "Base instructions.",
            "messages": [
                {"role": role, "content": STYLE},
                {"role": "system", "content": REMINDER},
            ],
        }
    )
    if role == "user":
        assert result[0]["content"] == "Base instructions."
        assert result[1] == {"role": "user", "content": f"{STYLE}\n{note(REMINDER)}"}
    else:
        assert result[0]["content"] == "Base instructions.\n\n" + REMINDER
        assert result[1:] == [{"role": "assistant", "content": STYLE}]


def test_changed_style_definition_is_preserved():
    updated = STYLE.replace("Use short answers.", "Use detailed answers.")
    result = converted(
        {
            "system": STYLE,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": updated},
                {"role": "system", "content": REMINDER},
            ],
        }
    )
    assert result[0]["content"] == STYLE
    assert result[1] == {"role": "user", "content": f"Hello\n{note(updated)}"}


def claude_code_system(billing):
    return [
        {"type": "text", "text": f"x-anthropic-billing-header: {billing}"},
        {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Base instructions.", "cache_control": {"type": "ephemeral"}},
    ]


@pytest.mark.parametrize(
    "first,second",
    [
        (
            "cc_version=2.1.312.492; cc_entrypoint=cli;",
            "cc_version=2.1.312.9cc; cc_entrypoint=cli;",
        ),
        (
            "cc_version=2.1.30.abc; cc_entrypoint=cli; cch=a1b2c;",
            "cc_version=2.1.30.abc; cc_entrypoint=cli; cch=f9e8d;",
        ),
    ],
)
def test_claude_code_billing_header_does_not_reach_the_model(first, second):
    messages = [{"role": "user", "content": "Hello"}]
    before = converted({"system": claude_code_system(first), "messages": messages})
    after = converted({"system": claude_code_system(second), "messages": messages})
    assert before == after
    assert before[0] == {"role": "system", "content": "You are Claude Code.\nBase instructions."}
    header_only = converted({"system": claude_code_system(first)[:1], "messages": messages})
    assert header_only == converted({"messages": messages})
    flat = f"x-anthropic-billing-header: {first}\nYou are Claude Code.\nBase instructions."
    expected = {"role": "system", "content": "You are Claude Code.\nBase instructions."}
    assert converted({"system": flat, "messages": messages})[0] == expected
    one_block = [{"type": "text", "text": flat}]
    assert converted({"system": one_block, "messages": messages})[0] == expected


def test_billing_text_later_in_the_prompt_is_kept():
    text = "Quote: x-anthropic-billing-header: is a Claude Code line."
    result = converted(
        {
            "system": [{"type": "text", "text": text}],
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    assert result[0] == {"role": "system", "content": text}

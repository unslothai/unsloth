# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Empty ``role="tool"`` content must be accepted on the OpenAI-compat surface.

Agentic clients send ``content: ""`` when a command produced no output;
OpenAI and llama-server both accept it. Unsloth used to 400, which standard
clients treat as non-retryable and kill the session. The validator must
normalize empty/missing tool content to ``""`` instead of raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models.inference import ChatMessage


def test_tool_message_empty_string_content_is_accepted():
    msg = ChatMessage(role = "tool", content = "", tool_call_id = "call_1")
    assert msg.content == ""


def test_tool_message_none_content_normalizes_to_empty_string():
    msg = ChatMessage(role = "tool", content = None, tool_call_id = "call_1")
    assert msg.content == ""


def test_tool_message_empty_list_content_normalizes_to_empty_string():
    msg = ChatMessage(role = "tool", content = [], tool_call_id = "call_1")
    assert msg.content == ""


def test_tool_message_real_content_is_preserved():
    msg = ChatMessage(role = "tool", content = "ok", tool_call_id = "call_1")
    assert msg.content == "ok"


def test_user_message_still_requires_content():
    with pytest.raises(ValueError):
        ChatMessage(role = "user", content = None)


def test_assistant_empty_content_still_collapses_to_none():
    msg = ChatMessage(role = "assistant", content = "")
    assert msg.content is None


def test_assistant_reasoning_content_round_trips():
    msg = ChatMessage(
        role = "assistant",
        content = "answer",
        reasoning_content = "step-by-step trace",
    )
    assert msg.model_dump(exclude_none = True)["reasoning_content"] == "step-by-step trace"


@pytest.mark.parametrize("structured", [[{"type": "reasoning", "text": "x"}], {"text": "x"}, 42])
def test_non_string_reasoning_is_ignored_instead_of_rejected(structured):
    msg = ChatMessage(role = "assistant", content = "answer", reasoning_content = structured)
    assert msg.reasoning_content is None
    assert "reasoning_content" not in msg.model_dump(exclude_none = True)

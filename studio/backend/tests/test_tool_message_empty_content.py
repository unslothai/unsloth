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


# ---------------------------------------------------------------------------
# reasoning_content: parse-level contract for the multi-turn thinking passthrough
# ---------------------------------------------------------------------------


def test_assistant_reasoning_content_round_trips_and_survives_dump():
    """A string ``reasoning_content`` on an assistant turn must survive Pydantic
    parsing and ``model_dump(exclude_none=True)`` so the proxy can forward it
    to llama-server for ``preserve_thinking=true`` rendering."""
    msg = ChatMessage(
        role = "assistant",
        content = "answer",
        reasoning_content = "x",
    )
    assert msg.reasoning_content == "x"
    dumped = msg.model_dump(exclude_none = True)
    assert dumped["reasoning_content"] == "x"


def test_assistant_non_string_reasoning_becomes_none_not_validation_error():
    """Gateways emit structured reasoning too. The before-validator must drop it
    to None instead of raising — declaring the field used to turn previously
    ignored payloads into 422s."""
    structured = [{"type": "reasoning", "text": "x"}]
    msg = ChatMessage(role = "assistant", content = "ok", reasoning_content = structured)
    assert msg.reasoning_content is None
    msg = ChatMessage(role = "assistant", content = "ok", reasoning_content = {"text": "x"})
    assert msg.reasoning_content is None
    msg = ChatMessage(role = "assistant", content = "ok", reasoning_content = 42)
    assert msg.reasoning_content is None


def test_assistant_reasoning_content_absent_from_dump_when_unset():
    """An assistant turn without ``reasoning_content`` must not gain a null
    key on dump: ``exclude_none=True`` already drops it, but ``model_dump``
    without ``exclude_none`` should also leave it absent (the field's default
    is None, and the validator returns None on unknown, not a sentinel)."""
    msg = ChatMessage(role = "assistant", content = "ok")
    assert msg.reasoning_content is None
    dumped = msg.model_dump()
    assert "reasoning_content" not in dumped or dumped["reasoning_content"] is None
    dumped_excl = msg.model_dump(exclude_none = True)
    assert "reasoning_content" not in dumped_excl

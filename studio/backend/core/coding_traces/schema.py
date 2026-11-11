# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Typed coding-trace records built from saved Studio chats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

TRACE_SOURCE = "studio_saved"

Role = Literal["system", "user", "assistant", "tool"]


class ChatTraceError(ValueError):
    """Raised when a chat cannot become a coding trace."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory = dict)


@dataclass
class TraceMessage:
    id: str
    role: Role
    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory = list)
    # Set only on role="tool": the ToolCall.id this message answers.
    tool_call_id: Optional[str] = None
    is_error: bool = False
    # Epoch milliseconds, as Studio stores them.
    created_at: Optional[int] = None


@dataclass
class TraceWarning:
    code: str
    message: str
    message_id: Optional[str] = None


@dataclass
class ChatTrace:
    thread_id: str
    title: str
    model_id: str
    created_at: Optional[int]
    updated_at: Optional[int]
    messages: list[TraceMessage]
    warnings: list[TraceWarning] = field(default_factory = list)
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    working_folder: Optional[str] = None
    source: str = TRACE_SOURCE

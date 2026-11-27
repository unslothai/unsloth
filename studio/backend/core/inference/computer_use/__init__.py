# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Computer use: a model sees this machine's screen and drives its mouse and keyboard."""

from .tool import (
    COMPUTER_TOOL_NAME,
    COMPUTER_TOOL_NAMES,
    computer_tool,
    execute_computer_tool,
    is_computer_tool,
    status_text,
)

__all__ = [
    "COMPUTER_TOOL_NAME",
    "COMPUTER_TOOL_NAMES",
    "computer_tool",
    "execute_computer_tool",
    "is_computer_tool",
    "status_text",
]

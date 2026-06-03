# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Process-level server-side tool policy.

Set by `unsloth run` at startup; consulted by the inference route gates.

  None  -> no CLI override (default). Per-request `enable_tools` is honored.
  True  -> CLI forced tools on for every request.
  False -> CLI forced tools off for every request.
"""

from typing import Optional

_tool_policy: Optional[bool] = None


def get_tool_policy() -> Optional[bool]:
    return _tool_policy


def set_tool_policy(value: Optional[bool]) -> None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(
            f"tool_policy must be Optional[bool], got {type(value).__name__}"
        )
    global _tool_policy
    _tool_policy = value


def reset_tool_policy() -> None:
    global _tool_policy
    _tool_policy = None

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Process-level server-side tool policy.

Set by `unsloth run` at startup; consulted by the inference route gates.

  None  -> no CLI override (default). Per-request `enable_tools` is honored.
  True  -> CLI forced tools on for every request.
  False -> CLI forced tools off for every request.
"""

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional

_tool_policy: Optional[bool] = None

# Per-request hard-off so public surfaces refuse tools even under a CLI `--enable-tools`.
_force_disabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tool_policy_force_disabled", default = False
)


def get_tool_policy() -> Optional[bool]:
    if _force_disabled.get():
        return False
    return _tool_policy


@contextmanager
def tools_force_disabled() -> Iterator[None]:
    """Hard-disable server-side tools for the current async context."""
    token = _force_disabled.set(True)
    try:
        yield
    finally:
        _force_disabled.reset(token)


def set_tool_policy(value: Optional[bool]) -> None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"tool_policy must be Optional[bool], got {type(value).__name__}")
    global _tool_policy
    _tool_policy = value


def reset_tool_policy() -> None:
    global _tool_policy
    _tool_policy = None

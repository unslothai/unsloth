# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Process-level server-side tool policy.

Two slots, both set at startup and consulted by the inference route gates.

The OVERRIDE (`set_tool_policy`) comes from an explicit `--enable-tools`/
`--disable-tools` and beats the request:

  None  -> no CLI override (default). Per-request `enable_tools` is honored.
  True  -> CLI forced tools on for every request. Not on /v1/messages: that channel
           cannot present a confirmation prompt, so it takes the on direction from the
           request itself (see _anthropic_selects_server_tools).
  False -> CLI forced tools off for every request, /v1/messages included.

The DEFAULT (`set_tool_policy_default`) is what an omitted `enable_tools` falls
back to. `unsloth studio run` installs True for every bind, `--secure` included,
so a plain request to a tool-capable model can use tools. It is only a default:
a request that says `enable_tools: false` (what the Unsloth UI sends with its tool
pills off) turns them off, which the override deliberately would not.

No other launcher installs it. `unsloth studio`, the desktop app and Colab leave
it unset, so an omitted `enable_tools` still means no tools there, which is what
paths like `n > 1`, `max_tool_calls_per_message: 0` and the pre-switch
passthrough guard are built around.
"""

import contextvars
from contextlib import contextmanager
from functools import partial, wraps
from typing import Iterator, Optional

_tool_policy: Optional[bool] = None
_tool_policy_default: Optional[bool] = None

# Per-request hard-off so public surfaces refuse tools even under a CLI `--enable-tools`.
_force_disabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tool_policy_force_disabled", default = False
)


def require_tool_access(
    permission_mode: Optional[str] = None,
    *,
    bypass_permissions: bool = False,
    disable_sandbox: bool = False,
) -> None:
    if permission_mode != "full" and not bypass_permissions and not disable_sandbox:
        return
    from auth.policy import full_access_permitted
    from fastapi import HTTPException

    if not full_access_permitted():
        raise HTTPException(
            status_code = 400,
            detail = "Full access is only available to the installation owner.",
        )


def normalize_tool_permissions(
    permission_mode: Optional[str], bypass_permissions: bool
) -> tuple[str, bool]:
    require_tool_access(permission_mode, bypass_permissions = bypass_permissions)
    if permission_mode == "full" or bypass_permissions:
        return "full", True
    if permission_mode is None:
        return "auto", False
    if permission_mode not in ("ask", "auto", "off"):
        return "ask", False
    return permission_mode, False


OS_SANDBOXED_TOOLS = frozenset({"python", "terminal"})


def off_mode_still_gates(name: str) -> bool:
    """Whether "off" must still ask for this tool. Cached only; unknown counts as not isolated."""
    if name not in OS_SANDBOXED_TOOLS:
        return False
    from core.inference.os_sandbox import cached_tool_isolation

    return cached_tool_isolation(name) is not True


def tool_call_may_prompt(
    *, confirm_tool_calls: bool, bypass_permissions: bool, permission_mode: Optional[str], name: str
) -> bool:
    """Before the arguments are known: whether a call to ``name`` could stop and ask."""
    if not confirm_tool_calls or bypass_permissions:
        return False
    if permission_mode == "off":
        return off_mode_still_gates(name)
    if permission_mode == "auto":
        from core.inference.tools import is_always_safe_tool
        return not is_always_safe_tool(name)
    return True


def needs_tool_confirmation(
    *,
    confirm_tool_calls: bool,
    bypass_permissions: bool,
    permission_mode: Optional[str],
    name: str,
    arguments,
    is_high_risk = None,
    never_needs = None,
) -> bool:
    if is_high_risk is None or never_needs is None:
        from core.inference import tools
        is_high_risk = is_high_risk or tools.is_high_risk_tool_call
        never_needs = never_needs or tools.never_needs_approval
    if not confirm_tool_calls or bypass_permissions or never_needs(name):
        return False
    if permission_mode == "off":
        return off_mode_still_gates(name) and is_high_risk(name, arguments)
    if permission_mode == "auto":
        return is_high_risk(name, arguments)
    return True


def requires_os_isolation(
    *,
    confirm_tool_calls: bool,
    bypass_permissions: bool,
    permission_mode: Optional[str],
    name: str,
    arguments,
    prompted: bool,
    is_high_risk = None,
) -> bool:
    """Whether this call must launch with tool_execution_mode="required" (no software fallback).

    ``prompted`` is needs_tool_confirmation's decision; re-reading the cache here could race a refresh.
    """
    if prompted or not confirm_tool_calls or bypass_permissions or permission_mode != "off":
        return False
    if name not in OS_SANDBOXED_TOOLS:
        return False
    if is_high_risk is None:
        from core.inference.tools import is_high_risk_tool_call as is_high_risk
    return bool(is_high_risk(name, arguments))


def account_tool_stream(stream):
    from utils.account_context import current_account, is_owner_context, run_as

    if is_owner_context():
        return stream
    account = current_account()

    @wraps(stream)
    def scoped(invoke, *args, **kwargs):
        return stream(partial(run_as, account, invoke), *args, **kwargs)

    return scoped


def get_tool_policy() -> Optional[bool]:
    if _force_disabled.get():
        return False
    return _tool_policy


def get_tool_policy_default() -> Optional[bool]:
    """Fallback for a request that omits `enable_tools`; None unless `unsloth
    studio run` installed one (every other launcher, embedder and library caller
    keeps the omitted-is-off read)."""
    if _force_disabled.get():
        return False
    return _tool_policy_default


def conversation_recall_allowed() -> bool:
    """Recall reads only this thread's archive: `--disable-tools` keeps it, `tools_force_disabled` does not."""
    return not _force_disabled.get()


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


def set_tool_policy_default(value: Optional[bool]) -> None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"tool_policy_default must be Optional[bool], got {type(value).__name__}")
    global _tool_policy_default
    _tool_policy_default = value


def reset_tool_policy() -> None:
    global _tool_policy, _tool_policy_default
    _tool_policy = None
    _tool_policy_default = None

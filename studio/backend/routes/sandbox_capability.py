# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the Python and Terminal tools get on this computer, for any signed-in user.

The chat permission picker reads it to say whether "Full access in sandbox" really has an OS
sandbox behind it. It answers from the cached capability and never waits on a probe: before
the first answer both tools read as not isolated, which is what the "off" gate assumes too.
"""

import asyncio
import sys
from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from auth.authentication import get_current_subject

router = APIRouter()


class SandboxCapabilityResponse(BaseModel):
    platform: str
    python_os_isolated: bool
    terminal_os_isolated: bool
    # "unknown" until the first check has answered.
    backend: str
    reason: str
    # linux-install | windows-setup, only for the owner on a direct local request; None otherwise.
    setup_action: Optional[str] = None
    # The copyable command, for everyone; empty when there is nothing to set up.
    manual_command: str = ""
    can_run_setup: bool = False
    # Why a setup exists but this caller gets no button: not_owner | not_local | no_elevation.
    setup_blocked: Optional[str] = None
    needs_consent: bool = False


def _setup_fields_for(request: Request, isolated: bool) -> dict:
    """Blocking (reads the setup plan). Nothing to offer once both tools are isolated."""
    from core.inference import sandbox_setup_plan

    try:
        fields = sandbox_setup_plan.setup_fields_for(request, available = isolated)
    except Exception:  # noqa: BLE001 - the capability stays useful without the setup hint
        return {}
    fields.pop("reason", None)
    return fields


def _refresh(force: bool) -> None:
    """Blocking: check the tools that have no answer yet (all of them when forced)."""
    from core.inference import os_sandbox

    if os_sandbox._background_probes_disabled():
        return  # UNSLOTH_DISABLE_SANDBOX_WARMUP=1: answers come only from real launches
    for tool in os_sandbox.ISOLATED_TOOLS:
        if force or not os_sandbox.has_tool_isolation_answer(tool):
            os_sandbox.refresh_tool_isolation(tool, force = force)


def _capability() -> dict:
    from core.inference.os_sandbox import cached_tool_capability

    python = cached_tool_capability("python")
    terminal = cached_tool_capability("terminal")
    known = [item for item in (python, terminal) if item is not None]
    isolated = [item for item in known if item[0]]
    backend = (isolated or known or [(False, "unknown", "")])[0][1]
    if python is None or terminal is None:
        reason = "The sandbox check has not finished yet."
    else:
        reason = python[2] if not python[0] else terminal[2]
    return {
        "platform": sys.platform,
        "python_os_isolated": bool(python and python[0]),
        "terminal_os_isolated": bool(terminal and terminal[0]),
        "backend": backend,
        "reason": reason,
    }


@router.get("/capability", response_model = SandboxCapabilityResponse)
async def sandbox_capability(
    request: Request,
    refresh: bool = False,
    current_subject: str = Depends(get_current_subject),
) -> SandboxCapabilityResponse:
    """Right after a setup or at startup the cached answer may be missing: check before answering,
    so "not isolated" is never a guess. A forced re-check is the owner's (it relaunches the probe)."""
    from utils.account_context import is_owner_context

    force = bool(refresh) and is_owner_context()
    await asyncio.to_thread(_refresh, force)
    capability = _capability()
    isolated = capability["python_os_isolated"] and capability["terminal_os_isolated"]
    setup = await asyncio.to_thread(_setup_fields_for, request, isolated)
    return SandboxCapabilityResponse(**capability, **setup)

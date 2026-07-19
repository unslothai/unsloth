# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama.cpp prebuilt update endpoints.

GET  /api/llama/update-status  -> is a newer prebuilt available + job state
POST /api/llama/update         -> download + atomically swap to the latest

Detection reuses utils.llama_cpp_freshness; the swap reuses
install_llama_prebuilt.py via utils.llama_cpp_update. Both fail open so the UI
never blocks on a missing marker / offline GitHub.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
from typing import Optional

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from loggers import get_logger
from utils.llama_cpp_update import get_update_status, start_update
from utils.update_confirm import (
    CONFIRM_TOKEN_TTL_SECONDS,
    consume_confirm_token,
    mint_confirm_token,
)

logger = get_logger(__name__)
router = APIRouter()

# Messages for a refused apply, keyed by outcome. A refusal never runs the swap.
_REFUSAL_MESSAGES = {
    "confirmation_required": (
        "Confirm the llama.cpp update before it runs. It will download and swap "
        "the binary on the machine running Studio."
    ),
    "invalid_token": (
        "The confirmation is missing or unrecognized. Re-check for the update and "
        "confirm again before it runs."
    ),
    "expired_token": ("The confirmation expired. Re-check for the update and confirm again."),
    "stale_target": (
        "The available build changed since you confirmed. Re-check for the update "
        "and confirm the new build before it runs."
    ),
}


class UpdateMachine(BaseModel):
    """Host a swap targets, so a remote operator sees which machine would change."""

    hostname: str = Field("", description = "Hostname of the machine running Studio.")
    platform: str = Field(
        "", description = "Host OS tag (sys.platform), e.g. 'linux', 'darwin', 'win32'."
    )


def _current_machine() -> UpdateMachine:
    """Best-effort host identity; cross-OS and never branched on."""
    try:
        hostname = socket.gethostname() or "unknown"
    except Exception:  # pragma: no cover
        hostname = "unknown"
    return UpdateMachine(hostname = hostname, platform = sys.platform)


class LlamaUpdateJob(BaseModel):
    state: str = Field("idle", description = "idle | running | success | error")
    message: str = ""
    from_tag: Optional[str] = None
    to_tag: Optional[str] = None
    reload_required: Optional[bool] = None
    error: Optional[str] = None
    progress: Optional[float] = Field(None, description = "0..1 while running, 1 on success.")
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class LlamaUpdateStatusResponse(BaseModel):
    supported: bool = Field(
        False,
        description = "True when the install came from an Unsloth prebuilt (has a marker).",
    )
    update_available: bool = Field(
        False, description = "True when the latest release is genuinely newer than the install."
    )
    stale: bool = Field(
        False, description = "Update available AND install older than the staleness threshold."
    )
    installed_tag: Optional[str] = None
    latest_tag: Optional[str] = None
    published_repo: Optional[str] = None
    installed_at_utc: Optional[str] = None
    age_days: Optional[int] = None
    source_build: bool = Field(
        False, description = "True when there is no marker (source build) but a prebuilt is offered."
    )
    update_size_bytes: Optional[int] = Field(
        None, description = "Download size of the prebuilt Update would fetch, in bytes."
    )
    machine: UpdateMachine = Field(
        default_factory = UpdateMachine,
        description = "The host this status describes; the machine an Update would change.",
    )
    job: LlamaUpdateJob = Field(default_factory = LlamaUpdateJob)


class LlamaUpdateRequest(BaseModel):
    """Body for POST /update. Empty body means no confirmation, so the swap is
    refused (safe default) and a stale banner or replay cannot swap the binary."""

    confirm_token: Optional[str] = Field(
        None,
        description = "Single-use token from POST /update/confirm, bound to the exact offered build.",
    )
    confirmed: bool = Field(
        False,
        description = "Explicit confirmation for non-interactive/CLI callers that do not use a token.",
    )


class LlamaUpdateConfirmResponse(BaseModel):
    """Pending swap details plus, when appliable, a confirm token for the prompt."""

    update_available: bool = False
    appliable: bool = Field(
        False, description = "True when an update exists and can actually be applied here."
    )
    reason: Optional[str] = Field(
        None, description = "Why not appliable: up_to_date | local_link | ..."
    )
    machine: UpdateMachine = Field(default_factory = UpdateMachine)
    installed_tag: Optional[str] = None
    latest_tag: Optional[str] = None
    update_size_bytes: Optional[int] = None
    confirm_token: Optional[str] = Field(
        None, description = "Echo back to POST /update to run the swap. Absent when not appliable."
    )
    confirm_expires_at: Optional[str] = None
    confirm_ttl_seconds: Optional[int] = None


class LlamaUpdateActionResponse(BaseModel):
    started: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    machine: UpdateMachine = Field(
        default_factory = UpdateMachine,
        description = "The host the swap ran (or would run) on; names the result's machine.",
    )
    installed_tag: Optional[str] = None
    latest_tag: Optional[str] = None
    update_size_bytes: Optional[int] = None
    job: LlamaUpdateJob = Field(default_factory = LlamaUpdateJob)


_llama_update_lock = threading.Lock()
_last_llama_update_step = -1


def _log_llama_update_progress(job: LlamaUpdateJob) -> None:
    """One llama_update_progress line per 10% step so a prebuilt update reports
    progress without a line per poll. Resyncs when a new update starts."""
    global _last_llama_update_step
    if job.state != "running" or job.progress is None:
        return
    step = int(max(0.0, min(float(job.progress), 1.0)) * 10)
    with _llama_update_lock:
        prev = _last_llama_update_step
        if step == prev:
            return
        _last_llama_update_step = step
        if step < prev:
            return  # new update; resync without logging
    logger.info("llama_update_progress", to_tag = job.to_tag or "", percent = step * 10)


@router.get("/update-status", response_model = LlamaUpdateStatusResponse)
async def llama_update_status(
    force_refresh: bool = Query(
        False, description = "Bypass the 24h release cache for an explicit check."
    ),
    current_subject: str = Depends(get_current_subject),
) -> LlamaUpdateStatusResponse:
    # Off the event loop: detection may probe the host and read GitHub.
    status = await asyncio.to_thread(get_update_status, force_refresh = force_refresh)
    resp = LlamaUpdateStatusResponse(machine = _current_machine(), **status)
    _log_llama_update_progress(resp.job)
    return resp


@router.post("/update/confirm", response_model = LlamaUpdateConfirmResponse)
async def llama_update_confirm(
    current_subject: str = Depends(get_current_subject),
) -> LlamaUpdateConfirmResponse:
    """Step one of the two-step apply: describe the pending swap and, when it can
    be applied, mint a single-use token bound to the offered build. Available to
    any authenticated operator; confirmation, not location, is the gate."""
    # Force-refresh so the token binds the current build, not a stale cached tag.
    status = await asyncio.to_thread(get_update_status, force_refresh = True)
    machine = _current_machine()
    installed_tag = status.get("installed_tag")
    latest_tag = status.get("latest_tag")
    size = status.get("update_size_bytes")

    # A --with-llama-cpp-dir local link reports update_available=False, so this
    # must run before the up_to_date branch below or the local_link reason is
    # masked and callers are wrongly told there is no update.
    if status.get("local_link"):
        return LlamaUpdateConfirmResponse(
            update_available = True,
            appliable = False,
            reason = "local_link",
            machine = machine,
            installed_tag = installed_tag,
            latest_tag = latest_tag,
            update_size_bytes = size,
        )
    if not status.get("update_available"):
        return LlamaUpdateConfirmResponse(
            update_available = False,
            appliable = False,
            reason = status.get("reason") or "up_to_date",
            machine = machine,
            installed_tag = installed_tag,
            latest_tag = latest_tag,
            update_size_bytes = size,
        )

    token, expires_at = mint_confirm_token(latest_tag or "")
    return LlamaUpdateConfirmResponse(
        update_available = True,
        appliable = True,
        reason = None,
        machine = machine,
        installed_tag = installed_tag,
        latest_tag = latest_tag,
        update_size_bytes = size,
        confirm_token = token,
        confirm_expires_at = expires_at,
        confirm_ttl_seconds = CONFIRM_TOKEN_TTL_SECONDS,
    )


@router.post("/update", response_model = LlamaUpdateActionResponse)
async def llama_update(
    request: Optional[LlamaUpdateRequest] = Body(default = None),
    current_subject: str = Depends(get_current_subject),
) -> LlamaUpdateActionResponse:
    """Apply the swap, but only with an explicit, fresh confirmation.

    The installer replaces the host binary, so a caller must either echo the
    single-use ``confirm_token`` from POST /update/confirm (preferred: replay-safe,
    bound to the build) or send ``confirmed=true`` (non-interactive callers). With
    neither, the swap is refused and the binary is left untouched. The gate is the
    confirmation, not the caller's location, so a headless SSH server confirms like a local one."""
    req = request or LlamaUpdateRequest()
    machine = _current_machine()
    # Force-refresh so the token is validated against the same build start_update
    # will resolve: a stale cache would accept a token minted for an older tag and
    # then install a newer one, bypassing the exact-build binding.
    status = await asyncio.to_thread(get_update_status, force_refresh = True)
    installed_tag = status.get("installed_tag")
    target_tag = status.get("latest_tag")
    size = status.get("update_size_bytes")

    confirmed = False
    refuse_reason = "confirmation_required"
    if req.confirm_token:
        ok, why = consume_confirm_token(req.confirm_token, target_tag or "")
        if ok:
            confirmed = True
        else:
            refuse_reason = why or "invalid_token"
    elif req.confirmed:
        confirmed = True

    if not confirmed:
        # Safe default: no confirmation, no swap; return a visible, actionable refusal.
        return LlamaUpdateActionResponse(
            started = False,
            reason = refuse_reason,
            message = _REFUSAL_MESSAGES.get(
                refuse_reason, _REFUSAL_MESSAGES["confirmation_required"]
            ),
            machine = machine,
            installed_tag = installed_tag,
            latest_tag = target_tag,
            update_size_bytes = size,
            job = LlamaUpdateJob(**status.get("job", {})),
        )

    # Pass the confirmed target so the updater installs exactly that build and
    # aborts if latest moved between this refresh and its own (see start_update).
    action = await asyncio.to_thread(start_update, target_tag)
    return LlamaUpdateActionResponse(
        machine = machine,
        installed_tag = installed_tag,
        latest_tag = target_tag,
        update_size_bytes = size,
        **action,
    )

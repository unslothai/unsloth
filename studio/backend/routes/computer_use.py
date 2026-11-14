# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Computer use: availability, the owner's switch, and the OS grants it needs."""

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from auth.authentication import get_current_subject
from auth.policy import OwnerOnly
from loggers import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ComputerUsePermissions(BaseModel):
    screen: Optional[bool] = None
    input: Optional[bool] = None


class ComputerUseStatus(BaseModel):
    available: bool
    reason: Optional[str] = None
    enabled: bool
    coordinates: Literal["auto", "pixels", "grid"]
    platform: str
    permissions: ComputerUsePermissions
    allowed_apps: list[str] = []
    blocked_apps: list[str] = []
    stopped_chats: int = 0


class ComputerUseAppRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str = Field(min_length = 1, max_length = 80)


class ComputerUseSettingsRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: Optional[StrictBool] = None
    coordinates: Optional[Literal["auto", "pixels", "grid"]] = None


def _status() -> ComputerUseStatus:
    from core.inference.computer_use import policy
    from core.inference.computer_use.backends import get_backend

    backend = get_backend()
    reason = policy.unavailable_reason(backend)
    permissions: dict = {"screen": None, "input": None}
    if backend.unavailable_reason() is None:
        try:
            permissions = backend.permissions()
        except Exception:  # noqa: BLE001 - the page still loads without the grants' state
            logger.warning("computer use permissions could not be read", exc_info = True)
    from state import computer_use as state
    from utils.account_context import current_account_id

    current = policy.settings()
    return ComputerUseStatus(
        available = reason is None,
        reason = reason,
        enabled = current["enabled"],
        coordinates = current["coordinates"],
        platform = backend.name,
        permissions = ComputerUsePermissions(**permissions),
        allowed_apps = current["allowed_apps"],
        blocked_apps = current["blocked_apps"],
        stopped_chats = state.stopped_count(current_account_id()),
    )


@router.get("/status", response_model = ComputerUseStatus)
async def computer_use_status(current_subject: str = Depends(get_current_subject)):
    return await asyncio.to_thread(_status)


@router.put("/settings", response_model = ComputerUseStatus, dependencies = [OwnerOnly])
async def update_computer_use_settings(
    payload: ComputerUseSettingsRequest, current_subject: str = Depends(get_current_subject)
):
    from core.inference.computer_use import policy

    if payload.enabled:
        # Turning it on is the UI's to do, on the machine it controls.
        problem = policy.host_problem()
        if problem:
            raise HTTPException(status_code = 403, detail = problem)
    try:
        policy.update_settings(enabled = payload.enabled, coordinates = payload.coordinates)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None
    if payload.enabled is not None:
        from state import computer_use as state
        from utils.account_context import current_account_id
        if payload.enabled:
            state.resume_account(current_account_id())
        else:
            state.revoke_account(current_account_id())
    logger.info(
        "settings.computer_use_updated subject=%s enabled=%s coordinates=%s",
        current_subject,
        payload.enabled,
        payload.coordinates,
    )
    return await asyncio.to_thread(_status)


@router.post("/permissions/{kind}", response_model = ComputerUseStatus, dependencies = [OwnerOnly])
async def request_computer_use_permission(
    kind: Literal["screen", "input"], current_subject: str = Depends(get_current_subject)
):
    """Raise the OS prompt for a grant. A person presses this in Settings; a tool call never can."""
    from core.inference.computer_use import policy
    from core.inference.computer_use.backends import get_backend

    problem = policy.unavailable_reason()
    if problem:
        raise HTTPException(status_code = 403, detail = problem)
    await asyncio.to_thread(get_backend().request_permission, kind)
    return await asyncio.to_thread(_status)


@router.post("/allowed-apps", response_model = ComputerUseStatus, dependencies = [OwnerOnly])
async def allow_computer_use_app(
    payload: ComputerUseAppRequest, current_subject: str = Depends(get_current_subject)
):
    """Always allow: the model acts in this app without asking, in every chat."""
    from core.inference.computer_use import policy

    problem = policy.host_problem()
    if problem:
        raise HTTPException(status_code = 403, detail = problem)
    try:
        policy.allow_app(payload.name)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None
    logger.info(
        "settings.computer_use_app_allowed subject=%s app=%s", current_subject, payload.name
    )
    return await asyncio.to_thread(_status)


@router.delete("/allowed-apps", response_model = ComputerUseStatus, dependencies = [OwnerOnly])
async def remove_computer_use_app(name: str, current_subject: str = Depends(get_current_subject)):
    from core.inference.computer_use import policy

    policy.remove_allowed_app(name)
    logger.info("settings.computer_use_app_removed subject=%s app=%s", current_subject, name)
    return await asyncio.to_thread(_status)


@router.post("/blocked-apps", response_model = ComputerUseStatus, dependencies = [OwnerOnly])
async def block_computer_use_app(
    payload: ComputerUseAppRequest, current_subject: str = Depends(get_current_subject)
):
    """Block: no model acts in this app, in any chat or permission mode."""
    from core.inference.computer_use import policy

    try:
        policy.block_app(payload.name)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None
    logger.info(
        "settings.computer_use_app_blocked subject=%s app=%s", current_subject, payload.name
    )
    return await asyncio.to_thread(_status)


@router.delete("/blocked-apps", response_model = ComputerUseStatus, dependencies = [OwnerOnly])
async def unblock_computer_use_app(name: str, current_subject: str = Depends(get_current_subject)):
    from core.inference.computer_use import policy

    # Loosening is the UI's to do, on the machine it controls.
    problem = policy.host_problem()
    if problem:
        raise HTTPException(status_code = 403, detail = problem)
    policy.unblock_app(name)
    logger.info("settings.computer_use_app_unblocked subject=%s app=%s", current_subject, name)
    return await asyncio.to_thread(_status)


@router.post("/resume", response_model = ComputerUseStatus, dependencies = [OwnerOnly])
async def resume_computer_use(current_subject: str = Depends(get_current_subject)):
    """Allow again: chats the user stopped with the pointer may use the computer once more."""
    from core.inference.computer_use import policy
    from state import computer_use as state
    from utils.account_context import current_account_id

    problem = policy.host_problem()
    if problem:
        raise HTTPException(status_code = 403, detail = problem)
    resumed = state.resume_account(current_account_id())
    logger.info("settings.computer_use_resumed subject=%s chats=%s", current_subject, resumed)
    return await asyncio.to_thread(_status)


@router.post("/revoke")
async def revoke_computer_use(current_subject: str = Depends(get_current_subject)):
    """Every chat of this account asks again, and the actions running now end."""
    from state import computer_use as state
    from utils.account_context import current_account_id

    return {"revoked": state.revoke_account(current_account_id())}

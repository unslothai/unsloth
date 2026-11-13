# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Computer use: whether this machine can be controlled, the owner's switch, and the
operating system grants it needs."""

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool

from auth.authentication import get_current_subject
from auth.policy import OwnerOnly
from loggers import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ComputerUsePermissions(BaseModel):
    # None where the operating system has no such grant to hold.
    screen: Optional[bool] = None
    input: Optional[bool] = None


class ComputerUseStatus(BaseModel):
    available: bool
    reason: Optional[str] = None
    enabled: bool
    coordinates: Literal["auto", "pixels", "grid"]
    platform: str
    permissions: ComputerUsePermissions


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
    current = policy.settings()
    return ComputerUseStatus(
        available = reason is None,
        reason = reason,
        enabled = current["enabled"],
        coordinates = current["coordinates"],
        platform = backend.name,
        permissions = ComputerUsePermissions(**permissions),
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
    if payload.enabled is False:
        from state import computer_use as state
        from utils.account_context import current_account_id
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
    """Raise the operating system's own prompt for a grant. A person presses this in Settings; a
    tool call never can."""
    from core.inference.computer_use import policy
    from core.inference.computer_use.backends import get_backend

    problem = policy.unavailable_reason()
    if problem:
        raise HTTPException(status_code = 403, detail = problem)
    await asyncio.to_thread(get_backend().request_permission, kind)
    return await asyncio.to_thread(_status)


@router.post("/revoke")
async def revoke_computer_use(current_subject: str = Depends(get_current_subject)):
    """Every chat of this account asks again before it next touches the computer."""
    from state import computer_use as state
    from utils.account_context import current_account_id

    return {"revoked": state.revoke_account(current_account_id())}

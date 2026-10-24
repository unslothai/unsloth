# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings → Agents → Claude Desktop: switch the Claude app on this Mac to Studio and back."""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from core import claude_desktop as session
from routes.desktop_app_models import (
    DesktopAppModel,
    DesktopAppModelSettings,
    available_models,
    model_settings,
)
from utils import client_ip


async def _require_installation_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


def _require_ui_session(via_api_key: bool = Depends(authenticated_via_api_key)) -> None:
    # An API key is exactly what the switch hands Claude; it must not be able to drive the switch.
    if via_api_key:
        raise HTTPException(status_code = 403, detail = "Claude Desktop needs a Studio UI session.")


def _require_local(request: Request) -> None:
    # Stricter than client_ip(): this quits and opens an app on the computer running Studio.
    if not client_ip.is_direct_local_request(request):
        raise HTTPException(
            status_code = 403,
            detail = "Connect Claude Desktop from the computer running Unsloth Studio.",
        )


router = APIRouter(
    dependencies = [Depends(_require_installation_owner), Depends(_require_ui_session)]
)


class ClaudeDesktopStatus(BaseModel):
    supported: bool
    installed: bool
    app_path: Optional[str] = None
    download_url: str
    connected: bool
    held_by: Optional[str] = None
    held_by_pid: Optional[int] = None
    running: bool
    model: Optional[str] = None
    # Claude slot id → Studio model id, or ":loaded" to follow whatever Studio has loaded.
    mappings: dict[str, str] = {}
    routes: list[dict[str, str]] = []
    local: bool


class ConnectRequest(BaseModel):
    # None: ask before restarting an open Claude. True: restart it. False: leave it open.
    restart: Optional[bool] = None


class ConnectResponse(BaseModel):
    status: str  # connected | pending_restart | needs_restart
    claude: ClaudeDesktopStatus


class MappingsRequest(BaseModel):
    mappings: dict[str, str]
    # As for Connect: None asks before restarting an open, connected Claude.
    restart: Optional[bool] = None


class MappingsResponse(BaseModel):
    status: str  # saved | connected | pending_restart | needs_restart
    claude: ClaudeDesktopStatus


class DisconnectResponse(BaseModel):
    notes: list[str]
    claude: ClaudeDesktopStatus


async def _status(request: Request) -> ClaudeDesktopStatus:
    data = await asyncio.to_thread(session.status)
    return ClaudeDesktopStatus(**data, local = client_ip.is_direct_local_request(request))


def _raise(exc: session.ClaudeDesktopError) -> None:
    raise HTTPException(status_code = exc.status, detail = exc.message) from exc


def _upstream(request: Request) -> str:
    # The gateway calls back into this server over loopback, exactly like the CLI's gateway does.
    return str(request.base_url).rstrip("/")


@router.get("", response_model = ClaudeDesktopStatus)
async def get_claude_desktop(request: Request) -> ClaudeDesktopStatus:
    return await _status(request)


@router.post("/connect", response_model = ConnectResponse)
async def connect_claude_desktop(
    payload: ConnectRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
) -> ConnectResponse:
    _require_local(request)
    try:
        result = await asyncio.to_thread(
            session.connect, _upstream(request), current_subject, payload.restart
        )
    except session.ClaudeDesktopError as exc:
        _raise(exc)
    return ConnectResponse(status = result["status"], claude = await _status(request))


@router.post("/disconnect", response_model = DisconnectResponse)
async def disconnect_claude_desktop(request: Request) -> DisconnectResponse:
    _require_local(request)
    try:
        notes = await asyncio.to_thread(session.disconnect)
    except session.ClaudeDesktopError as exc:
        _raise(exc)
    return DisconnectResponse(notes = notes, claude = await _status(request))


@router.get("/models", response_model = list[DesktopAppModel])
async def claude_desktop_models() -> list[DesktopAppModel]:
    return await available_models()


@router.get("/model-settings", response_model = DesktopAppModelSettings)
async def desktop_app_model_settings(
    model: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
) -> DesktopAppModelSettings:
    """Shared by the Claude Desktop and Codex Desktop cards: where a model's run settings live
    (the same remembered per-model settings Run settings edits) and what bounds them."""
    return await model_settings(model, request, current_subject)


@router.put("/mappings", response_model = MappingsResponse)
async def set_claude_desktop_mappings(
    payload: MappingsRequest, request: Request
) -> MappingsResponse:
    _require_local(request)
    try:
        result = await asyncio.to_thread(session.set_mappings, payload.mappings, payload.restart)
    except session.ClaudeDesktopError as exc:
        _raise(exc)
    return MappingsResponse(status = result["status"], claude = await _status(request))

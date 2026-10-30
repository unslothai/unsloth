# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings → Agents → Codex Desktop: switch the Codex app on this Mac to Studio and back."""

import asyncio
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from core import codex_desktop as session
from routes.desktop_app_models import DesktopAppModel, available_models
from utils import client_ip


async def _require_installation_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


def _require_ui_session(via_api_key: bool = Depends(authenticated_via_api_key)) -> None:
    # An API key is exactly what the switch hands Codex; it must not be able to drive the switch.
    if via_api_key:
        raise HTTPException(status_code = 403, detail = "Codex Desktop needs a Studio UI session.")


def _require_local(request: Request) -> None:
    # Stricter than client_ip(): this quits and opens an app on the computer running Studio.
    if not client_ip.is_direct_local_request(request):
        raise HTTPException(
            status_code = 403,
            detail = "Connect the Codex app from the computer running Unsloth Studio.",
        )


router = APIRouter(
    dependencies = [Depends(_require_installation_owner), Depends(_require_ui_session)]
)


ModelList = Annotated[
    list[Annotated[str, StringConstraints(max_length = 512)]],
    Field(max_length = session.MAX_MODELS),
]


class CodexDesktopStatus(BaseModel):
    supported: bool
    installed: bool
    app_path: Optional[str] = None
    download_url: str
    connected: bool
    # "cli" while `unsloth start codex --app` holds the app.
    held_by: Optional[str] = None
    held_by_pid: Optional[int] = None
    running: bool
    # Codex's default model, and every model in its catalog, while connected.
    model: Optional[str] = None
    models: list[str] = []
    # What Studio has loaded now, and the card's saved list (empty follows the loaded model).
    loaded_model: Optional[str] = None
    selected_models: list[str] = []
    max_models: int
    local: bool


class ConnectRequest(BaseModel):
    # None: ask before restarting an open Codex. True: restart it. False: leave it open.
    restart: Optional[bool] = None
    # Studio catalog ids (the first is Codex's default), [] to follow the loaded model, or None
    # for the saved list.
    models: Optional[ModelList] = None


class ModelsRequest(BaseModel):
    models: ModelList = []
    restart: Optional[bool] = None


class StatusResponse(BaseModel):
    status: str  # connected | pending_restart | needs_restart | saved
    codex: CodexDesktopStatus


class DisconnectRequest(BaseModel):
    restart: Optional[bool] = None


class DisconnectResponse(BaseModel):
    status: str  # disconnected | needs_restart
    notes: list[str]
    codex: CodexDesktopStatus


async def _status(request: Request) -> CodexDesktopStatus:
    data = await asyncio.to_thread(session.status)
    return CodexDesktopStatus(**data, local = client_ip.is_direct_local_request(request))


def _raise(exc: session.CodexDesktopError) -> None:
    raise HTTPException(status_code = exc.status, detail = exc.message) from exc


def _upstream(request: Request) -> str:
    # Codex calls this server's /v1 over loopback, as with `unsloth start codex --app`.
    return str(request.base_url).rstrip("/")


@router.get("", response_model = CodexDesktopStatus)
async def get_codex_desktop(request: Request) -> CodexDesktopStatus:
    return await _status(request)


@router.post("/connect", response_model = StatusResponse)
async def connect_codex_desktop(
    payload: ConnectRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
) -> StatusResponse:
    _require_local(request)
    try:
        result = await asyncio.to_thread(
            session.connect, _upstream(request), current_subject, payload.restart, payload.models
        )
    except session.CodexDesktopError as exc:
        _raise(exc)
    return StatusResponse(status = result["status"], codex = await _status(request))


@router.put("/models", response_model = StatusResponse)
async def set_codex_desktop_models(payload: ModelsRequest, request: Request) -> StatusResponse:
    _require_local(request)
    try:
        result = await asyncio.to_thread(session.set_models, payload.models, payload.restart)
    except session.CodexDesktopError as exc:
        _raise(exc)
    return StatusResponse(status = result["status"], codex = await _status(request))


@router.post("/disconnect", response_model = DisconnectResponse)
async def disconnect_codex_desktop(
    request: Request, payload: Optional[DisconnectRequest] = None
) -> DisconnectResponse:
    _require_local(request)
    try:
        result = await asyncio.to_thread(
            session.disconnect, payload.restart if payload is not None else None
        )
    except session.CodexDesktopError as exc:
        _raise(exc)
    return DisconnectResponse(
        status = result["status"], notes = result["notes"], codex = await _status(request)
    )


@router.get("/models", response_model = list[DesktopAppModel])
async def codex_desktop_models() -> list[DesktopAppModel]:
    return await available_models()

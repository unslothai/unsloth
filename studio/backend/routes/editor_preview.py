# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local 3D editor sessions. Frames and input do not pass through chat or remote access."""

from __future__ import annotations

import asyncio
import io
import re
import secrets
import struct
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, FiniteFloat

from auth.authentication import authenticated_via_api_key, get_current_subject, request_admitted_without_credential
from utils.host_policy import remote_connector_active
from utils.lan_access_settings import request_is_loopback

router = APIRouter()
Editor = Literal["blender", "godot", "unity", "unreal", "roblox"]
ASSETS = Path(__file__).resolve().parents[1] / "integrations" / "editor_preview"
MAX_FRAME = 4 * 1024 * 1024
MAX_SESSIONS = 8
IDLE_SECONDS = 120
STALE_SECONDS = 2
NO_FRAME = bytes(4)
SESSION_HEADERS = {"Cache-Control": "no-store"}


def require_local(request: Request):
    if not request_is_loopback(request) or remote_connector_active():
        raise HTTPException(403, "Editor preview requires a local Studio connection with Remote Access off.")


async def preview_user(
    request: Request,
    subject: str = Depends(get_current_subject),
    via_key: bool = Depends(authenticated_via_api_key),
    anonymous: bool = Depends(request_admitted_without_credential),
) -> str:
    require_local(request)
    if via_key or anonymous:
        raise HTTPException(403, "Sign in to Studio to open an editor preview.")
    return subject


class Gesture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    orbit_x: FiniteFloat = 0.0
    orbit_y: FiniteFloat = 0.0
    pan_x: FiniteFloat = 0.0
    pan_y: FiniteFloat = 0.0
    zoom: FiniteFloat = 0.0

    def merge(self, other: "Gesture") -> "Gesture":
        previous = self.model_dump()
        return Gesture(**{
            axis: max(-1.0, min(1.0, previous[axis] + max(-0.25, min(0.25, delta))))
            for axis, delta in other.model_dump().items()
        })


@dataclass
class Preview:
    owner: str
    editor: Editor
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    instance: str | None = None
    viewer: bool = False
    closed: bool = False
    generation: int = 0
    updated_at: float = field(default_factory=time.monotonic)
    frame_at: float = 0
    frame: bytes | None = None
    gesture: Gesture = field(default_factory=Gesture)
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def clear(self):
        self.generation += 1
        self.updated_at = time.monotonic()
        self.frame_at = 0
        self.frame = None
        self.gesture = Gesture()

    def close(self):
        self.closed = True
        self.clear()
        self.changed.set()


def sessions(request: Request) -> dict[str, Preview]:
    values = getattr(request.app.state, "editor_previews", None)
    if values is None:
        values = request.app.state.editor_previews = {}
    now = time.monotonic()
    for key, session in list(values.items()):
        if not session.viewer and now - session.updated_at > IDLE_SECONDS:
            values.pop(key).close()
    return values


def owned(request: Request, session_id: str, user: str) -> Preview:
    preview = sessions(request).get(session_id)
    if preview is None or preview.owner != user:
        raise HTTPException(404, "Preview session ended.")
    return preview


def paired(request: Request, editor: Editor, session_id: str) -> Preview:
    require_local(request)
    preview = sessions(request).get(session_id)
    if preview is None or preview.editor != editor or not secrets.compare_digest(
        request.headers.get("authorization", "").encode(), b"Bearer " + preview.token.encode()
    ):
        raise HTTPException(401, "Start a new preview and pair the editor again.")
    instance = request.headers.get("x-unsloth-preview-instance", "")
    if not re.fullmatch(r"[a-f0-9]{32}", instance):
        raise HTTPException(400, "The adapter must identify its editor instance.")
    if preview.instance is None and request.method == "GET":
        preview.instance = instance
    if preview.instance != instance:
        raise HTTPException(409, "This preview is paired with another editor instance.")
    return preview


@router.post("/sessions/{editor}", status_code=201)
async def create(request: Request, editor: Editor, user: str = Depends(preview_user)):
    values = sessions(request)
    if len(values) >= MAX_SESSIONS:
        raise HTTPException(429, "Close another editor preview and try again.")
    session_id = secrets.token_urlsafe(16)
    preview = values[session_id] = Preview(owner=user, editor=editor)
    host, port = request.scope["server"]
    loopback = "[::1]" if ":" in host else "127.0.0.1"
    endpoint = f"http://{loopback}:{port}/api/editor-preview/adapter/{editor}/{session_id}"
    return JSONResponse({"id": session_id, "pairing": f"{endpoint} {preview.token}"}, 201, headers=SESSION_HEADERS)


@router.delete("/sessions/{session_id}", status_code=204)
async def close(request: Request, session_id: str, user: str = Depends(preview_user)):
    owned(request, session_id, user).close()
    sessions(request).pop(session_id, None)


@router.get("/adapters/{editor}")
async def download(editor: Editor, user: str = Depends(preview_user)):
    source = ASSETS / editor
    files = [entry for entry in sorted(source.rglob("*")) if entry.is_file() and "__pycache__" not in entry.parts]
    if not files:
        raise HTTPException(503, "This Studio build is missing the editor adapter.")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in files:
            archive.write(entry, entry.relative_to(source))
    return Response(output.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="unsloth-preview-{editor}.zip"'
    })


@router.get("/sessions/{session_id}/frames")
async def stream_frames(request: Request, session_id: str, user: str = Depends(preview_user)):
    preview = owned(request, session_id, user)
    if preview.viewer:
        raise HTTPException(409, "This preview already has a viewer.")

    async def stream():
        previous = None
        try:
            while not preview.closed and not remote_connector_active():
                preview.changed.clear()
                frame = preview.frame
                if frame is not None and frame is not previous:
                    previous = frame
                    yield struct.pack(">I", len(frame)) + frame
                try:
                    await asyncio.wait_for(preview.changed.wait(), timeout=.75)
                except asyncio.TimeoutError:
                    yield NO_FRAME
        finally:
            preview.viewer = False
            preview.clear()

    preview.viewer = True
    return StreamingResponse(stream(), media_type="application/octet-stream", headers={
        "Cache-Control": "no-store, no-transform",
        "Content-Encoding": "identity",
        "X-Accel-Buffering": "no",
    })


@router.post("/sessions/{session_id}/input", status_code=204)
async def send_input(request: Request, session_id: str, data: Gesture, user: str = Depends(preview_user)):
    preview = owned(request, session_id, user)
    if preview.viewer and time.monotonic() - preview.frame_at <= STALE_SECONDS:
        preview.gesture = preview.gesture.merge(data)


@router.get("/adapter/{editor}/{session_id}/input")
async def get_input(request: Request, editor: Editor, session_id: str):
    preview = paired(request, editor, session_id)
    return take_input(preview)


def take_input(preview: Preview) -> JSONResponse:
    data, preview.gesture = preview.gesture, Gesture()
    if not preview.viewer or time.monotonic() - preview.frame_at > STALE_SECONDS:
        data = Gesture()
    return JSONResponse({"active": preview.viewer, **data.model_dump()}, headers=SESSION_HEADERS)


@router.post("/adapter/{editor}/{session_id}/frame", status_code=204)
async def upload_frame(request: Request, editor: Editor, session_id: str):
    await store_frame(request, paired(request, editor, session_id))


@router.post("/adapter/roblox/{session_id}/exchange")
async def exchange(request: Request, session_id: str):
    # Roblox Studio sends its newest frame and receives input in the same request.
    preview = paired(request, "roblox", session_id)
    await store_frame(request, preview)
    return take_input(preview)


async def store_frame(request: Request, preview: Preview):
    if not preview.viewer:
        return
    generation = preview.generation
    frame = bytearray()
    async for chunk in request.stream():
        if len(frame) + len(chunk) > MAX_FRAME:
            raise HTTPException(413, "Viewport frame is too large.")
        frame.extend(chunk)
    from PIL import Image

    try:
        with Image.open(io.BytesIO(frame), formats=("JPEG", "PNG")) as image:
            if not (1 <= image.width <= 4096 and 1 <= image.height <= 4096):
                raise ValueError("Invalid viewport dimensions")
    except (OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(415, "Send an encoded JPEG or PNG viewport frame.") from None
    if preview.viewer and not preview.closed and preview.generation == generation:
        preview.frame = bytes(frame)
        preview.frame_at = time.monotonic()
        preview.changed.set()

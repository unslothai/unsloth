# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Restyle a video (Cosmos3-Nano-Sim-Transfer on sd-server): upload a clip or a folder of frames plus a prompt,
poll progress, then fetch the fitted input and the restyled output for a side-by-side view."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from auth.authentication import get_current_subject
from core.inference import video_restyle
from core.inference.video_restyle import RestyleBusy, RestyleError
from utils.account_context import account_thread, current_account_id
from utils.upload_limits import VIDEO_RESTYLE_UPLOAD_MAX_BYTES

router = APIRouter()


def _save_to_gallery(mp4: bytes, meta: dict) -> Optional[dict]:
    from core.inference import video_gallery
    try:
        return video_gallery.save(mp4, meta)
    except Exception:  # noqa: BLE001 -- the clip is still served from the job dir
        return None


async def _read_upload(request: Request) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    try:
        form = await request.form(max_files = video_restyle.MAX_FRAMES + 8)
    except Exception:  # noqa: BLE001
        raise HTTPException(
            status_code = 400,
            detail = f"Send one clip or at most {video_restyle.MAX_FRAMES} frames as multipart/form-data.",
        )
    fields = {key: form.get(key) for key in ("prompt", "seed") if isinstance(form.get(key), str)}
    uploads = [
        (name, value)
        for name, value in form.multi_items()
        if name in ("video", "frames") and isinstance(value, UploadFile)
    ]
    # Starlette spools each part to disk and records its size, so the cap is checked before any is read.
    if sum(value.size or 0 for _, value in uploads) > VIDEO_RESTYLE_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code = 413, detail = "The upload is too large.")
    files = [(name + ":" + (value.filename or ""), await value.read()) for name, value in uploads]
    return fields, files


@router.post("/video/restyle")
async def start_restyle(request: Request, current_subject: str = Depends(get_current_subject)):
    fields, files = await _read_upload(request)
    videos = [data for name, data in files if name.startswith("video:")]
    frames_in = [
        (name.split(":", 1)[1], data) for name, data in files if name.startswith("frames:")
    ]
    seed: Optional[int] = None
    if fields.get("seed"):
        try:
            seed = int(fields["seed"])
        except ValueError:
            raise HTTPException(status_code = 400, detail = "seed must be an integer.")
    try:
        if len(videos) == 1 and not frames_in:
            frames = await asyncio.to_thread(video_restyle.frames_from_video, videos[0])
        elif frames_in and not videos:
            frames = await asyncio.to_thread(video_restyle.frames_from_images, frames_in)
        else:
            raise RestyleError("Send one video file or a folder of frames.")
        job = await asyncio.to_thread(
            video_restyle.runner.start,
            prompt = fields.get("prompt") or "",
            frames = frames,
            seed = seed,
            owner = current_account_id(),
        )
    except RestyleBusy as exc:
        raise HTTPException(status_code = 409, detail = str(exc))
    except RestyleError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    account_thread(
        target = video_restyle.runner.run,
        args = (job, _save_to_gallery),
        daemon = True,
        name = "video-restyle",
    ).start()
    return video_restyle.runner.progress(current_account_id())


@router.get("/video/restyle/progress")
async def restyle_progress(current_subject: str = Depends(get_current_subject)):
    return video_restyle.runner.progress(current_account_id())


@router.post("/video/restyle/cancel")
async def cancel_restyle(current_subject: str = Depends(get_current_subject)):
    return {"cancelled": video_restyle.runner.cancel(current_account_id())}


@router.post("/video/restyle/unload")
async def unload_restyle(current_subject: str = Depends(get_current_subject)):
    if video_restyle.runner.progress().get("active"):
        raise HTTPException(status_code = 409, detail = "A restyle is running.")
    await asyncio.to_thread(video_restyle.runner.unload)
    return {"unloaded": True}


@router.get("/video/restyle/{job_id}/{kind}.mp4")
async def restyle_media(
    job_id: str,
    kind: str,
    current_subject: str = Depends(get_current_subject),
):
    path = video_restyle.runner.media_path(job_id, kind, current_account_id())
    if path is None:
        raise HTTPException(status_code = 404, detail = "Not found.")
    return FileResponse(path, media_type = "video/mp4")

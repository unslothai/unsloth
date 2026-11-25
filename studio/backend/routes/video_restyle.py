# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video restyle routes (Cosmos3-Nano-Sim-Transfer on sd-server)."""

from __future__ import annotations

import asyncio
import contextlib
import functools
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile

from auth.authentication import get_current_subject
from core.inference import gpu_arbiter
from core.inference.chat_swap_kv import acquire_then_close, save_chat_kv_before_swap
from core.inference import video_restyle
from core.inference.video_restyle import RestyleBusy, RestyleError
from hub.services.models import account_access
from state import active_generations
from utils.account_context import account_thread, current_account_id
from utils.multi_model_settings import get_multi_model_mode
from utils.upload_limits import VIDEO_RESTYLE_UPLOAD_MAX_BYTES

router = APIRouter()


def _save_to_gallery(mp4: bytes, meta: dict) -> Optional[dict]:
    from core.inference import video_gallery
    try:
        return video_gallery.save(mp4, meta)
    except Exception:  # noqa: BLE001 -- the clip is still served from the job dir
        return None


# Each upload is read into memory (up to VIDEO_RESTYLE_UPLOAD_MAX_BYTES) and the runner restyles one clip
# at a time, so more held at once only fills memory. Counted on the event loop thread only: no lock.
MAX_HELD_UPLOADS = 4
_held_uploads = 0


@contextlib.contextmanager
def _upload_slot():
    global _held_uploads
    if _held_uploads >= MAX_HELD_UPLOADS:
        raise HTTPException(
            status_code = 429,
            detail = "Restyle is busy reading other uploads. Please try again in a moment.",
        )
    _held_uploads += 1
    try:
        yield
    finally:
        _held_uploads -= 1


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
    if sum(value.size or 0 for _, value in uploads) > VIDEO_RESTYLE_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code = 413, detail = "The upload is too large.")
    files = [(name + ":" + (value.filename or ""), await value.read()) for name, value in uploads]
    return fields, files


async def _frames_from_upload(files: list[tuple[str, bytes]]) -> tuple[list[Any], Optional[float]]:
    videos = [data for name, data in files if name.startswith("video:")]
    frames_in = [
        (name.split(":", 1)[1], data) for name, data in files if name.startswith("frames:")
    ]
    if len(videos) == 1 and not frames_in:
        frames = await asyncio.to_thread(video_restyle.frames_from_video, videos[0])
        return frames, await asyncio.to_thread(video_restyle.video_seconds, videos[0])
    if frames_in and not videos:
        return await asyncio.to_thread(video_restyle.frames_from_images, frames_in), None
    raise RestyleError("Send one video file or a folder of frames.")


@router.get("/video/restyle/status")
async def restyle_status(current_subject: str = Depends(get_current_subject)):
    state = await asyncio.to_thread(video_restyle.availability)
    if state.get("can_setup"):
        try:
            await _require_restyle_installer(current_subject)
        except HTTPException as exc:
            # Only the owner may run setup; everyone else sees why instead of a button that answers 403.
            state = {**state, "can_setup": False, "refusal": state.get("refusal") or exc.detail}
    return state


class RestyleSetupRequest(BaseModel):
    hf_token: Optional[str] = Field(default = None, max_length = 512)


async def _require_restyle_installer(current_subject: str = Depends(get_current_subject)) -> None:
    """Setup installs sd.cpp and writes ~30 GB into Studio's cache: the installation owner's call, as for every other
    install (an API key only where full API access is enabled)."""
    from auth import policy

    await policy.require_owner()
    if not policy.full_access_permitted() or policy.api_full_access_refused():
        raise HTTPException(
            status_code = 403,
            detail = "Set up Restyle from the Unsloth UI as the installation owner.",
        )


@router.post("/video/restyle/setup", dependencies = [Depends(_require_restyle_installer)])
async def setup_restyle(
    body: Optional[RestyleSetupRequest] = None, current_subject: str = Depends(get_current_subject)
):
    """Install the sd.cpp server and fetch the weights, whichever is missing; poll /video/restyle/status."""
    from core.inference import video_restyle_setup as setup

    await asyncio.to_thread(account_access.require_model_allowed, video_restyle.MODEL_ID)
    state = await asyncio.to_thread(video_restyle.availability)
    if state["available"]:
        return state
    if state.get("refusal"):
        raise HTTPException(status_code = 400, detail = state["refusal"])
    if not state.get("can_setup"):
        raise HTTPException(
            status_code = 400, detail = state.get("reason") or "Restyle cannot be set up here."
        )
    token = account_access.account_hf_token((body.hf_token if body else None) or None)
    if token is None:
        token = account_access.ambient_hf_token()
    try:
        await asyncio.to_thread(
            setup.setup_runner.start,
            hf_token = token,
            need_engine = "engine" in state["missing"],
            need_weights = "weights" in state["missing"],
        )
    except setup.SetupError as exc:
        raise HTTPException(status_code = 409, detail = str(exc))
    return await asyncio.to_thread(video_restyle.availability)


@router.post("/video/restyle/setup/cancel", dependencies = [Depends(_require_restyle_installer)])
async def cancel_restyle_setup(current_subject: str = Depends(get_current_subject)):
    from core.inference import video_restyle_setup as setup
    return {"cancelled": setup.setup_runner.cancel()}


@router.post("/video/restyle/preview")
async def preview_restyle(request: Request, current_subject: str = Depends(get_current_subject)):
    with _upload_slot():
        _, files = await _read_upload(request)
        try:
            frames, seconds = await _frames_from_upload(files)
            return await asyncio.to_thread(video_restyle.preview, frames, seconds)
        except RestyleError as exc:
            raise HTTPException(status_code = 400, detail = str(exc))


@router.post("/video/restyle")
async def start_restyle(request: Request, current_subject: str = Depends(get_current_subject)):
    if account_access.managed_account():
        await asyncio.to_thread(account_access.require_model_allowed, video_restyle.MODEL_ID)
    with _upload_slot():
        # Receive the upload before reserving: a reservation holds the GPU, so a slow or stalled body
        # must not keep every model load and training start refused while it trickles in.
        upload = await _read_upload(request)
        try:
            reservation = video_restyle.runner.reserve(current_account_id())
        except RestyleBusy as exc:
            raise HTTPException(status_code = 409, detail = str(exc))
        # Released here until the shielded start takes it over, which then releases it itself:
        # a cancelled request must not free a reservation that start() is still about to use.
        lease = {"route": True}
        try:
            return await _start_reserved(upload, reservation, lease)
        finally:
            if lease["route"]:
                video_restyle.runner.release(reservation)


def _admit_and_start(**kwargs: Any) -> dict[str, Any]:
    # Same interlock as a video load: refuse beside training, and hold the training admission while
    # the runner turns active so a training start in between sees it and refuses.
    from core.training.diffusion_training_service import TrainingActiveError
    from routes.inference import _diffusion_training_admission
    from routes.video import _guard_video_load_against_training

    mode = kwargs.pop("mode", None)
    try:
        with _diffusion_training_admission():
            _guard_video_load_against_training()
            _take_the_gpu(mode)
            job = video_restyle.runner.start(**kwargs)
    except TrainingActiveError as exc:
        raise HTTPException(status_code = 409, detail = f"{exc} Try again when it finishes.")
    finally:
        video_restyle.runner.release(kwargs.get("reservation"))
    # Start inside shielded call: a cancelled request must not leave a runner without a worker.
    account_thread(
        target = video_restyle.runner.run,
        args = (job, _save_to_gallery),
        daemon = True,
        name = "video-restyle",
    ).start()
    return job


def _take_the_gpu(mode: Optional[str] = None) -> None:
    # The restyle sd-server (~30 GB) is video work: chat and image models evict; Create video is
    # checked in admit under the arbiter lock, then unloaded in register once every other refusal
    # passed.
    from core.inference import gpu_arbiter
    from core.inference.video import generation_account_in_flight, get_video_backend
    from utils.multi_model_settings import get_multi_model_mode

    if mode is None:
        mode = get_multi_model_mode()
    video = get_video_backend()

    def admit() -> None:
        if video.loading_repo_ids():
            raise HTTPException(
                status_code = 409, detail = "A video model is loading. Try again when it finishes."
            )
        if not video.status().get("loaded"):
            return
        if generation_account_in_flight() is not None:
            raise HTTPException(
                status_code = 409, detail = "A video is generating. Try again when it finishes."
            )
        if mode == "always":
            raise HTTPException(
                status_code = 409,
                detail = "Unload the Create video model first: Keep multiple models loaded is set to Always.",
            )

    def register() -> None:
        if video.status().get("loaded"):
            video.unload()

    handoff = gpu_arbiter.restyle_handoff.set(True)
    try:
        gpu_arbiter.acquire_for_request(gpu_arbiter.VIDEO, register, policy = mode, admit = admit)
    except gpu_arbiter.MultiModelAlwaysRefusal as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    finally:
        gpu_arbiter.restyle_handoff.reset(handoff)


async def _start_reserved(upload: tuple, reservation: str, lease: dict):
    fields, files = upload
    seed: Optional[int] = None
    if fields.get("seed"):
        try:
            seed = int(fields["seed"])
        except ValueError:
            raise HTTPException(status_code = 400, detail = "seed must be an integer.")
    try:
        frames, _ = await _frames_from_upload(files)
        prompt, owner = fields.get("prompt") or "", current_account_id()
        if owner is not None and active_generations.fenced(owner):
            raise HTTPException(status_code = 403, detail = "This account is being removed.")
        await asyncio.to_thread(video_restyle.validate_request, prompt, frames)
        mode = await asyncio.to_thread(get_multi_model_mode)
        swap = await save_chat_kv_before_swap(gpu_arbiter.VIDEO, "video", policy = mode)
        lease["route"] = False
        job = await acquire_then_close(
            swap,
            functools.partial(
                _admit_and_start,
                mode = mode,
                prompt = prompt,
                frames = frames,
                seed = seed,
                owner = owner,
                reservation = reservation,
            ),
        )
    except RestyleBusy as exc:
        raise HTTPException(status_code = 409, detail = str(exc))
    except RestyleError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    return video_restyle.runner.progress(current_account_id())


@router.get("/video/restyle/progress")
async def restyle_progress(current_subject: str = Depends(get_current_subject)):
    return video_restyle.runner.progress(current_account_id())


@router.post("/video/restyle/cancel")
async def cancel_restyle(current_subject: str = Depends(get_current_subject)):
    return {"cancelled": await asyncio.to_thread(video_restyle.runner.cancel, current_account_id())}


@router.post("/video/restyle/unload")
async def unload_restyle(current_subject: str = Depends(get_current_subject)):
    # A managed account may only unload a server its own last job loaded (as video unload checks).
    owner = None
    if account_access.managed_account():
        await asyncio.to_thread(account_access.require_model_allowed, video_restyle.MODEL_ID)
        owner = current_account_id()
    try:
        unloaded = await asyncio.to_thread(video_restyle.runner.unload_if_idle, owner)
    except RestyleBusy as exc:
        raise HTTPException(status_code = 403, detail = str(exc))
    if not unloaded:
        raise HTTPException(status_code = 409, detail = "A restyle is running.")
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

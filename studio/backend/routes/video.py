# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""API routes for local text-to-video inference.

The video backend is a deliberate sibling of the diffusion (image) backend, so
these routes mirror the /images/* routes one-for-one: the same validate-before-evict
load ordering, the same GPU arbiter handoff (VIDEO owner in place of DIFFUSION),
the same error boundary mapping backend exceptions to HTTP, and the same gallery
CRUD shape. The backend runs in-process and is synchronous, so the blocking
load/generate/unload calls are offloaded with asyncio.to_thread to keep the event
loop free. This module is the single error boundary: backend methods raise, we
map to HTTP here.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import ValidationError

from auth.authentication import get_current_subject
from loggers import get_logger
from models.inference import (
    GalleryVideo,
    VideoGalleryListResponse,
    VideoGenerateProgressResponse,
    VideoGenerateRequest,
    VideoGenerateResponse,
    VideoLoadProgressResponse,
    VideoLoadRequest,
    VideoStatusResponse,
)

logger = get_logger(__name__)

router = APIRouter()


def _guard_video_load_against_training() -> None:
    """Refuse loading a video model while a training run is active. Unlike chat,
    a video pipeline's VRAM can't be cheaply estimated before the load, so the
    load is refused outright rather than fit-checked. No-op when training is
    inactive or its state can't be read. Raises HTTP 409. Mirrors the image
    load's _guard_diffusion_load_against_training."""
    from core.training import get_training_backend

    try:
        llm_active = get_training_backend().is_training_active()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not check training state for video-load guard: %s", e)
        return
    diffusion_active = False
    try:
        from core.training.diffusion_training_service import get_diffusion_training_service
        diffusion_active = get_diffusion_training_service().is_active()
    except Exception:  # noqa: BLE001
        diffusion_active = False
    # An SDXL LoRA trainer runs in its own subprocess on the same GPU, so a video
    # load must be refused while one is active too -- otherwise the resident pipeline
    # competes with the trainer for VRAM. Symmetric with the image-load interlock.
    if not llm_active and not diffusion_active:
        return
    raise HTTPException(
        status_code = 409,
        detail = (
            "Can't load a video model while training is running: the video "
            "pipeline would compete with the training run for GPU memory. Training "
            "was left untouched. Try again after training finishes."
        ),
    )


@router.post("/video/load", response_model = VideoStatusResponse)
async def load_video_model(
    request: VideoLoadRequest, current_subject: str = Depends(get_current_subject)
):
    from core.inference.diffusion_device import resolve_diffusion_device_target
    from core.inference.gpu_arbiter import VIDEO, acquire_for, release
    from core.inference.video import get_video_backend
    from utils.native_path_leases import redact_native_paths

    backend = get_video_backend()
    try:
        # Validate cheaply BEFORE touching the GPU: an unloadable pick (bad family,
        # missing local checkpoint, a non-trusted non-GGUF repo) must not evict a
        # working chat model and then 400.
        await asyncio.to_thread(
            backend.validate_load_request,
            request.model_path,
            gguf_filename = request.gguf_filename,
            family_override = request.family_override,
            model_kind = request.model_kind,
            transformer_quant = request.transformer_quant,
        )
        # Refuse while training is running: a multi-GB video pipeline would compete
        # with the training subprocess for VRAM. Mirrors the image-load guard.
        _guard_video_load_against_training()
        # Take the GPU from the chat backend only when this load will actually use it,
        # which is exactly the resolved device being non-CPU. A CPU-only load never
        # touches GPU memory, so keying off the device (not the load) avoids wrongly
        # evicting a resident chat model. Release any stale VIDEO ownership on a CPU
        # load -- release() is owner-guarded, so it is a no-op when video never owned
        # the GPU.
        device = await asyncio.to_thread(lambda: resolve_diffusion_device_target().device)
        if device != "cpu":
            await asyncio.to_thread(acquire_for, VIDEO)
        else:
            await asyncio.to_thread(release, VIDEO)
        status_dict = await asyncio.to_thread(
            backend.begin_load,
            request.model_path,
            gguf_filename = request.gguf_filename,
            base_repo = request.base_repo,
            family_override = request.family_override,
            hf_token = request.hf_token,
            memory_mode = request.memory_mode,
            speed_mode = request.speed_mode,
            attention_backend = request.attention_backend,
            transformer_cache = request.transformer_cache,
            transformer_cache_threshold = request.transformer_cache_threshold,
            transformer_quant = request.transformer_quant,
            model_kind = request.model_kind,
        )
        return VideoStatusResponse(**status_dict)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code = 400, detail = redact_native_paths(str(exc)))
    except RuntimeError as exc:
        # A video load is already in progress.
        raise HTTPException(status_code = 409, detail = str(exc))


@router.get("/video/load-progress", response_model = VideoLoadProgressResponse)
async def video_load_progress(current_subject: str = Depends(get_current_subject)):
    from core.inference.video import get_video_backend
    return VideoLoadProgressResponse(**get_video_backend().load_progress())


@router.post("/video/generate", response_model = VideoGenerateResponse)
async def generate_video(
    request: VideoGenerateRequest, current_subject: str = Depends(get_current_subject)
):
    from core.inference import video_gallery
    from core.inference.video import get_video_backend
    from core.inference.video_families import VIDEO_CANCELLED_MSG, VIDEO_NOT_LOADED_MSG

    backend = get_video_backend()
    try:
        result = await asyncio.to_thread(
            backend.generate,
            prompt = request.prompt,
            negative_prompt = request.negative_prompt,
            width = request.width,
            height = request.height,
            num_frames = request.num_frames,
            fps = request.fps,
            steps = request.steps,
            guidance = request.guidance,
            guidance_2 = request.guidance_2,
            seed = request.seed,
        )
    except ValueError as exc:
        # Bad client input (a workflow the loaded family doesn't support) -- a 400 with
        # the reason, not a generic 500.
        raise HTTPException(status_code = 400, detail = str(exc))
    except RuntimeError as exc:
        # Only "no model loaded" / user-cancelled are client-state (409). Match the
        # sentinels exactly, not as a substring, so an execution failure that merely
        # contains "cancelled" can't misroute to 409 and leak that output.
        msg = str(exc)
        if msg in (VIDEO_NOT_LOADED_MSG, VIDEO_CANCELLED_MSG):
            raise HTTPException(status_code = 409, detail = msg)
        logger.error("video.generate_failed: %s", exc, exc_info = True)
        raise HTTPException(status_code = 500, detail = "Video generation failed.")
    except Exception as exc:
        logger.error("video.generate_failed: %s", exc, exc_info = True)
        raise HTTPException(status_code = 500, detail = "Video generation failed.")

    # Persist the clip with its full recipe as the JSON sidecar the gallery reads back.
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _persist() -> dict:
        return video_gallery.save(
            result["mp4_bytes"],
            {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": result["width"],
                "height": result["height"],
                "num_frames": result["num_frames"],
                "fps": result["fps"],
                "duration_s": result["duration_s"],
                "steps": result["steps"],
                "guidance": result["guidance"],
                "guidance_2": request.guidance_2,
                "seed": result["seed"],
                "has_audio": result["has_audio"],
                "model": result["repo_id"],
                "created_at": created_at,
            },
        )

    try:
        record = await asyncio.to_thread(_persist)
    except Exception as exc:  # noqa: BLE001
        logger.error("video.persist_failed: %s", exc)
        raise HTTPException(status_code = 500, detail = "Failed to save the generated video.")

    return VideoGenerateResponse(video = GalleryVideo(**record))


@router.get("/video/generate-progress", response_model = VideoGenerateProgressResponse)
async def video_generate_progress(current_subject: str = Depends(get_current_subject)):
    from core.inference.video import get_video_backend
    return VideoGenerateProgressResponse(**get_video_backend().generate_progress())


@router.post("/video/generate/cancel")
async def cancel_video_generation(current_subject: str = Depends(get_current_subject)):
    from core.inference.video import get_video_backend
    cancelled = await asyncio.to_thread(get_video_backend().cancel_generate)
    return {"cancelled": cancelled}


@router.get("/video/status", response_model = VideoStatusResponse)
async def video_status(current_subject: str = Depends(get_current_subject)):
    from core.inference.video import get_video_backend
    return VideoStatusResponse(**get_video_backend().status())


@router.post("/video/unload", response_model = VideoStatusResponse)
async def unload_video_model(current_subject: str = Depends(get_current_subject)):
    from core.inference.gpu_arbiter import VIDEO, release
    from core.inference.video import get_video_backend

    status_dict = await asyncio.to_thread(get_video_backend().unload)
    release(VIDEO)
    return VideoStatusResponse(**status_dict)


@router.get("/video/gallery", response_model = VideoGalleryListResponse)
async def list_gallery_videos(
    limit: int = 50,
    offset: int = 0,
    current_subject: str = Depends(get_current_subject),
):
    from core.inference import video_gallery

    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    # Fetch one extra to learn whether more remain, without a second scan.
    records = await asyncio.to_thread(video_gallery.list_videos, limit + 1, offset)
    has_more = len(records) > limit
    # Build the response per record and drop any that fail schema validation: a
    # sidecar with all required keys but a wrong value type (a hand-dropped or
    # older-schema file) passes the presence-only read but would raise inside
    # GalleryVideo(**r). Skipping it keeps one bad file from 500-ing the listing.
    videos = []
    for r in records[:limit]:
        try:
            videos.append(GalleryVideo(**r))
        except ValidationError:
            continue
    return VideoGalleryListResponse(videos = videos, has_more = has_more)


@router.get("/video/gallery/{video_id}/file")
async def get_gallery_video_file(
    video_id: str, current_subject: str = Depends(get_current_subject)
):
    from core.inference import video_gallery

    path = await asyncio.to_thread(video_gallery.video_path, video_id)
    if path is None:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    from fastapi.responses import FileResponse

    # FileResponse streams from disk (no whole-clip buffering per request) and
    # serves HTTP range requests so a direct URL can seek without a full fetch.
    # Immutable content (id is unique per video), so let the browser cache it.
    return FileResponse(
        path,
        media_type = "video/mp4",
        headers = {"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.get("/video/gallery/{video_id}/export")
async def export_gallery_video(
    video_id: str,
    format: str = "webm",
    current_subject: str = Depends(get_current_subject),
):
    """Download-menu transcodes: WebM (VP9) or GIF, re-encoded on demand from the
    stored MP4 (which the /file route serves verbatim). 501 with a clear message
    when the codec/deps for the requested format are missing."""
    from core.inference import video_gallery

    fmt = format.strip().lower()
    if fmt not in ("webm", "gif"):
        raise HTTPException(status_code = 400, detail = "Unsupported format. Use webm or gif.")
    try:
        data = await asyncio.to_thread(video_gallery.transcode, video_id, fmt)
    except RuntimeError as exc:
        raise HTTPException(status_code = 501, detail = str(exc)) from exc
    if data is None:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    from fastapi.responses import Response

    return Response(
        content = data,
        media_type = "video/webm" if fmt == "webm" else "image/gif",
        # Transcodes are deterministic per id+format; let the browser cache them.
        headers = {"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.delete("/video/gallery/{video_id}")
async def delete_gallery_video(video_id: str, current_subject: str = Depends(get_current_subject)):
    from core.inference import video_gallery

    deleted = await asyncio.to_thread(video_gallery.delete, video_id)
    if not deleted:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    return {"deleted": True}


@router.delete("/video/gallery")
async def clear_gallery_videos(current_subject: str = Depends(get_current_subject)):
    from core.inference import video_gallery
    removed = await asyncio.to_thread(video_gallery.clear)
    return {"removed": removed}

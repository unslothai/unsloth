# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""API routes for local text-to-video inference.

The video backend is a deliberate sibling of the diffusion (image) backend, so
these routes mirror the /images/* routes one-for-one: the same validate-before-evict
load ordering, the same GPU arbiter handoff (VIDEO owner in place of DIFFUSION),
the same error boundary mapping backend exceptions to HTTP, and the same gallery
CRUD shape. The backend runs in-process and is synchronous, so the blocking
calls are offloaded with asyncio.to_thread to keep the event loop free; the slow
operations (load AND generate) run as background jobs whose begin_* calls return
at once, with progress + terminal outcome polled from their *-progress routes.
This module is the single error boundary: backend methods raise, we map to HTTP
here.
"""

from __future__ import annotations

import asyncio
import hashlib as _hashlib
import hmac as _hmac
import secrets as _secrets
import time as _time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import ValidationError

from auth.authentication import get_current_subject
from loggers import get_logger
from models.inference import (
    DiffusionDownloadPlanResponse,
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
        # Independent probes: an unreadable LLM backend must not disable the diffusion interlock below.
        logger.warning("Could not check training state for video-load guard: %s", e)
        llm_active = False
    diffusion_active = False
    try:
        from core.training.diffusion_training_service import get_diffusion_training_service
        diffusion_active = get_diffusion_training_service().is_active()
    except Exception:  # noqa: BLE001
        diffusion_active = False
    # An SDXL LoRA trainer runs in its own subprocess on the same GPU, so refuse a video load while one is active.
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


@router.post("/video/download-plan", response_model = DiffusionDownloadPlanResponse)
async def video_download_plan(
    request: VideoLoadRequest, current_subject: str = Depends(get_current_subject)
):
    """The repos + files this pick needs, so the frontend stages them through the Hub
    download manager instead of the load downloading inline. Mirrors /images/download-plan."""
    from core.inference.diffusion import resolve_local_single_file
    from core.inference.video import get_video_backend, resolve_video_model_kind
    from utils.native_path_leases import redact_native_paths

    backend = get_video_backend()
    try:
        kind = resolve_video_model_kind(request.gguf_filename, request.model_kind)
        if kind == "pipeline" and not request.gguf_filename:
            sole = await asyncio.to_thread(resolve_local_single_file, request.model_path)
            if sole is not None:
                request.gguf_filename = sole
                kind = resolve_video_model_kind(sole, None)
        await asyncio.to_thread(
            backend.validate_load_request,
            request.model_path,
            gguf_filename = request.gguf_filename,
            family_override = request.family_override,
            model_kind = kind,
            base_repo = request.base_repo,
        )
        plan = await asyncio.to_thread(
            backend.download_plan,
            request.model_path,
            gguf_filename = request.gguf_filename,
            base_repo = request.base_repo,
            family_override = request.family_override,
            model_kind = kind,
            hf_token = request.hf_token,
            # The plan must see the encoder policy the load will use: an fp8 request takes a hosted pre-cast encoder, so staging the dense one wastes ~49 GB on LTX-2.
            text_encoder_quant = request.text_encoder_quant,
        )
        return DiffusionDownloadPlanResponse(**plan)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code = 400, detail = redact_native_paths(str(exc)))


@router.post("/video/load", response_model = VideoStatusResponse)
async def load_video_model(
    request: VideoLoadRequest, current_subject: str = Depends(get_current_subject)
):
    from core.inference.diffusion import resolve_local_single_file
    from core.inference.diffusion_device import resolve_diffusion_device_target
    from core.inference.gpu_arbiter import VIDEO, acquire_for, release
    from core.inference.video import (
        assert_video_precision_available,
        get_video_backend,
        resolve_video_model_kind,
    )
    from utils.native_path_leases import redact_native_paths

    backend = get_video_backend()
    try:
        # Resolve the load kind once (gguf / single_file / pipeline) so validation and the load agree; a bad kind raises here, so a 400.
        kind = resolve_video_model_kind(request.gguf_filename, request.model_kind)
        # A local On-Device pick can be a bare single-file .safetensors dir the picker starts as a pipeline; if it holds exactly one checkpoint, load it as single_file. Mirrors images.
        if kind == "pipeline" and not request.gguf_filename:
            sole = await asyncio.to_thread(resolve_local_single_file, request.model_path)
            if sole is not None:
                request.gguf_filename = sole
                kind = resolve_video_model_kind(sole, None)
        # Validate cheaply BEFORE touching the GPU so an unloadable pick can't evict chat then 400.
        fam = await asyncio.to_thread(
            backend.validate_load_request,
            request.model_path,
            gguf_filename = request.gguf_filename,
            base_repo = request.base_repo,
            family_override = request.family_override,
            model_kind = kind,
            transformer_quant = request.transformer_quant,
            text_encoder_quant = request.text_encoder_quant,
        )
        # Refuse while training is running (VRAM competition) BEFORE the precision check below:
        # that check runs an uncached quantise+matmul probe on the GPU, which would initialise a
        # CUDA context and allocate alongside the training subprocess for a load that is about to
        # be rejected anyway. Mirrors the image-load route, which already guards first.
        _guard_video_load_against_training()
        # Same bar for an EXPLICIT precision this host can never honor. begin_load makes the
        # identical network-free check, but it runs inside acquire_for, which evicts chat under the
        # arbiter lock BEFORE the register callback -- so a refusal raised there arrives having
        # already taken the GPU away from the model it was meant to preserve. `auto` is never
        # refused, so a caller that left the precision to the backend cannot reach this.
        await asyncio.to_thread(
            assert_video_precision_available,
            fam,
            model_kind = kind,
            transformer_quant = request.transformer_quant,
            text_encoder_quant = request.text_encoder_quant,
        )
        # Take the GPU from chat only for a non-CPU load. Release stale VIDEO ownership on a CPU load (owner-guarded no-op).
        device = await asyncio.to_thread(lambda: resolve_diffusion_device_target().device)

        def _begin_load():
            # Kicks the (slow) load onto a background thread and returns at once; begin_load itself validates network-free.
            return backend.begin_load(
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
                text_encoder_quant = request.text_encoder_quant,
                model_kind = kind,
            )

        if device != "cpu":
            # Register the in-flight load UNDER the arbiter lock: otherwise a competing acquire in that gap evicts VIDEO before the
            # load is marked, finds nothing to cancel, and both allocate at once. The training admission wraps the same span.
            from routes.inference import _diffusion_training_admission
            def _acquire_and_begin():
                with _diffusion_training_admission():
                    return acquire_for(VIDEO, _begin_load)

            status_dict = await asyncio.to_thread(_acquire_and_begin)
        else:
            await asyncio.to_thread(release, VIDEO)
            status_dict = await asyncio.to_thread(_begin_load)
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
    """Start a generation job and return at once (the begin_load pattern): a clip
    takes minutes, and secure mode's tunnel caps the origin response window near
    100 seconds, so the response must not span the generation. The worker runs the
    generate + gallery-persist pipeline; the terminal outcome (completed with the
    saved record / failed with a client-safe error) arrives via generate-progress."""
    from core.inference.video import get_video_backend
    from core.inference.video_families import (
        VIDEO_GENERATION_BUSY_MSG,
        VIDEO_NOT_LOADED_MSG,
        VideoShapeError,
    )

    backend = get_video_backend()
    # The request bounds on VideoGenerateRequest are a coarse outer guard; the real rule is the LOADED
    # family's (its presets and frame lattice), and begin_generate applies it under the same lock that
    # reserves the state the job will run against, so a load committing concurrently cannot leave the
    # shape judged against one family and denoised by another. Unloaded still falls through to the
    # not-loaded 409; a family with no declared presets keeps the old SIZE snapping, though its frame
    # lattice is enforced either way (frame_step is declared regardless).
    try:
        await asyncio.to_thread(
            backend.begin_generate,
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
    except VideoShapeError as exc:
        # 422 before the 400 below, and it must stay first: VideoShapeError IS a ValueError. The body
        # parses and is in range, but the shape is not one this model can render.
        raise HTTPException(status_code = 422, detail = str(exc))
    except ValueError as exc:
        # Bad client input -- a 400 with the reason, not a generic 500.
        raise HTTPException(status_code = 400, detail = str(exc))
    except RuntimeError as exc:
        # Only the not-loaded / busy sentinels are client-state (409); match exactly so an unrelated failure cannot leak its message.
        msg = str(exc)
        if msg in (VIDEO_NOT_LOADED_MSG, VIDEO_GENERATION_BUSY_MSG):
            raise HTTPException(status_code = 409, detail = msg)
        logger.error("video.generate_failed: %s", exc, exc_info = True)
        raise HTTPException(status_code = 500, detail = "Video generation failed.")

    return VideoGenerateResponse()


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
    from core.inference.gpu_arbiter import VIDEO, release_if
    from core.inference.video import get_video_backend

    backend = get_video_backend()
    status_dict = await asyncio.to_thread(backend.unload)
    # Drop VIDEO ownership only if nothing is resident AND no load is in flight; the check and release must be ATOMIC (release_if). Mirrors images.
    await asyncio.to_thread(
        release_if,
        VIDEO,
        lambda: not backend.loading_repo_ids() and not backend.status()["loaded"],
    )
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

    # Validate inside the pager so offset / limit / has_more count over the accepted domain: dropping bad records only after slicing stalled infinite scroll at offset 0.
    def _valid_gallery_video(record: dict) -> bool:
        try:
            GalleryVideo(**record)
        except ValidationError:
            return False
        return True

    # Fetch one extra to learn whether more remain, without a second scan.
    records = await asyncio.to_thread(
        video_gallery.list_videos, limit + 1, offset, valid = _valid_gallery_video
    )
    has_more = len(records) > limit
    videos = [GalleryVideo(**r) for r in records[:limit]]
    return VideoGalleryListResponse(videos = videos, has_more = has_more)


@router.get("/video/gallery/{video_id}/file")
async def get_gallery_video_file(
    video_id: str, current_subject: str = Depends(get_current_subject)
):
    from core.inference import video_gallery

    # Ownership-gate the serve like delete/clear: resolve only a Studio-owned MP4, so a guessed stem cannot stream out a foreign clip.
    path = await asyncio.to_thread(video_gallery.owned_video_path, video_id)
    if path is None:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    from fastapi.responses import FileResponse

    # FileResponse streams from disk and serves range requests. Immutable per id, so let the browser cache it.
    return FileResponse(
        path,
        media_type = "video/mp4",
        headers = {"Cache-Control": "private, max-age=31536000, immutable"},
    )


# A clip is tens to hundreds of MB, so the gallery cannot fetch it into a blob like a PNG: that buffers the whole MP4, defeats seeking and
# pins the bytes in the webview. The /file route streams ranges but is bearer-gated, so mint a 12-hour HMAC link (<video> re-requests on seek).
_VIDEO_LINK_TTL = 12 * 3600
_VIDEO_LINK_SECRET = _secrets.token_bytes(32)


def _sign_video_id(video_id: str) -> str:
    exp = int(_time.time()) + _VIDEO_LINK_TTL
    payload = f"{video_id}.{exp}"
    sig = _hmac.new(_VIDEO_LINK_SECRET, payload.encode(), _hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_video_link_token(token: str) -> Optional[str]:
    """The video id a valid, unexpired token names, else None. A separate secret from the image
    links, so a token minted for one media type can never serve the other."""
    try:
        video_id, exp_s, sig = token.rsplit(".", 2)
    except ValueError:
        return None
    expected = _hmac.new(
        _VIDEO_LINK_SECRET, f"{video_id}.{exp_s}".encode(), _hashlib.sha256
    ).hexdigest()
    if not _hmac.compare_digest(sig, expected):
        return None
    try:
        if int(exp_s) < int(_time.time()):
            return None
    except ValueError:
        return None
    return video_id


@router.get("/video/gallery/{video_id}/signed-url")
async def get_gallery_video_signed_url(
    video_id: str, current_subject: str = Depends(get_current_subject)
):
    """A directly playable, range-capable link for one clip (bearer-gated to mint, HMAC to use).

    Returned as a relative URL so it works behind any proxy the page itself is served through."""
    from core.inference import video_gallery

    path = await asyncio.to_thread(video_gallery.owned_video_path, video_id)
    if path is None:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    token = _sign_video_id(video_id)
    return {"url": f"/api/inference/video/gallery/{video_id}/file-signed?token={token}"}


@router.get("/video/gallery/{video_id}/file-signed")
async def get_gallery_video_file_signed(video_id: str, token: str = Query(...)):
    """Stream one gallery MP4 gated by the HMAC token instead of the bearer, so it can be a plain
    <video src> and the browser can range-request it. Same ownership gate as the bearer route, and
    the token names the single clip it may serve."""
    from core.inference import video_gallery

    if _verify_video_link_token(token) != video_id:
        raise HTTPException(status_code = 401, detail = "Invalid or expired video link.")
    path = await asyncio.to_thread(video_gallery.owned_video_path, video_id)
    if path is None:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    from fastapi.responses import FileResponse

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
        path = await asyncio.to_thread(video_gallery.transcode_to_file, video_id, fmt)
    except RuntimeError as exc:
        raise HTTPException(status_code = 501, detail = str(exc)) from exc
    if path is None:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    def _cleanup() -> None:
        try:
            path.unlink(missing_ok = True)
        except OSError as e:  # noqa: BLE001 -- a leaked temp file must not fail the download
            logger.debug(f"Could not remove the export temp file {path}: {e}")

    # FileResponse streams from disk, so a large VP9 export is never fully resident. The temp file is deleted once sent.
    return FileResponse(
        path,
        media_type = "video/webm" if fmt == "webm" else "image/gif",
        filename = f"{video_id}.{fmt}",
        # Transcodes are deterministic per id+format; let the browser cache them.
        headers = {"Cache-Control": "private, max-age=31536000, immutable"},
        background = BackgroundTask(_cleanup),
    )


def _forget_terminal_video(video_id: Optional[str]) -> None:
    """Clear the backend's completed-job record for a clip that just left the gallery, so a page
    reload does not merge it back as a card whose file is gone. Best-effort: an unavailable backend
    only means the stale record survives, which is what happened before this call existed."""
    try:
        from core.inference.video import get_video_backend
        get_video_backend().forget_terminal_video(video_id)
    except Exception as e:  # noqa: BLE001 -- never fail a delete over progress bookkeeping
        logger.debug(f"Could not clear the terminal video record for {video_id!r}: {e}")


@router.delete("/video/gallery/{video_id}")
async def delete_gallery_video(video_id: str, current_subject: str = Depends(get_current_subject)):
    from core.inference import video_gallery

    deleted = await asyncio.to_thread(video_gallery.delete, video_id)
    if not deleted:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    _forget_terminal_video(video_id)
    return {"deleted": True}


@router.delete("/video/gallery")
async def clear_gallery_videos(current_subject: str = Depends(get_current_subject)):
    from core.inference import video_gallery

    removed = await asyncio.to_thread(video_gallery.clear)
    # Clear-all takes the terminal record's clip with it whatever its id.
    _forget_terminal_video(None)
    return {"removed": removed}

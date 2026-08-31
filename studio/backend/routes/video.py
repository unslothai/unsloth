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
import base64
import hashlib as _hashlib
import hmac as _hmac
import re as _re
import secrets as _secrets
import threading
import time as _time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from auth.authentication import get_current_subject, request_admitted_without_credential
from core.inference.model_ids import public_model_id
from hub.dependencies import get_hf_token
from loggers import get_logger
from loggers.media_progress import (
    byte_fraction,
    log_media_generation_progress,
    log_media_load_progress,
    reset_media_generation_progress,
    reset_media_load_progress,
)
from models.inference import (
    DiffusionDownloadPlanResponse,
    GalleryFlagsPatch,
    GalleryVideo,
    VideoGalleryListResponse,
    VideoGenerateProgressResponse,
    VideoGenerateRequest,
    VideoGenerateResponse,
    VideoJob,
    VideoJobCreateRequest,
    VideoJobDeleteResponse,
    VideoJobError,
    VideoJobListResponse,
    VideoLoadProgressResponse,
    VideoLoadRequest,
    VideoStatusResponse,
)
from utils.api_errors import openai_error_body
from utils.upload_limits import VIDEO_INPUT_REFERENCE_MAX_BYTES

logger = get_logger(__name__)

router = APIRouter()
openai_router = APIRouter()


def _selected_gpu_ordinal(gpu_ids, *, allow_ranking: bool = True):
    """The images route's resolver, shared so both media routes apply one rule."""
    from routes.inference import _selected_gpu_ordinal as _resolve
    return _resolve(gpu_ids, allow_ranking = allow_ranking)


def _training_is_active() -> bool:
    """The non-raising half of the load guard, for callers that must not take the GPU."""
    from routes.inference import _training_is_active as _images_training_is_active
    return _images_training_is_active()


def _derived_h3_task(gguf_filename: Optional[str], kind: str) -> Optional[str]:
    """The MiniMax-H3 partition a GGUF load resolves to from its filename, else None."""
    if kind != "gguf" or not gguf_filename:
        return None
    try:
        from core.inference.video_minimax_h3 import h3_transformer_task
        from pathlib import Path as _Path

        name = _Path(gguf_filename).name.lower()
        return h3_transformer_task(name) if name.startswith("minimax_h3_") else None
    except Exception:  # noqa: BLE001 -- a probe failure must not fail the load
        return None


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
    from core.inference.video import (
        assert_video_precision_available,
        get_video_backend,
        resolve_video_model_kind,
    )
    from utils.native_path_leases import redact_native_paths

    backend = get_video_backend()
    try:
        kind = resolve_video_model_kind(request.gguf_filename, request.model_kind)
        if kind == "pipeline" and not request.gguf_filename:
            sole = await asyncio.to_thread(resolve_local_single_file, request.model_path)
            if sole is not None:
                request.gguf_filename = sole
                kind = resolve_video_model_kind(sole, None)
        fam = await asyncio.to_thread(
            backend.validate_load_request,
            request.model_path,
            gguf_filename = request.gguf_filename,
            family_override = request.family_override,
            model_kind = kind,
            base_repo = request.base_repo,
            # Validation is quant-keyed: a scheme this family can serve only from a hosted
            # pre-quantized checkpoint has to be refused HERE, on the route that stages the
            # download, or the panel fetches ~98.7 GB before /video/load can say no.
            transformer_quant = request.transformer_quant,
            # And the partition, because one of those quant-keyed refusals is task-keyed: the
            # hosted pre-quantized H3 checkpoints are fl2va denoisers, so a quantized ref2va is
            # rejected. /video/load passes this and refuses; without it here the plan below staged
            # the 66 GB dense transformer_ref/ AND the incompatible fl2va quant first.
            h3_task = request.h3_task,
        )
        # BEFORE the plan is staged, as on the images side: /video/load refuses a precision this
        # host cannot honour, but the UI plans and downloads first, so an explicit FP8 on an
        # unsupported host paid for tens of GB of weights to be told afterwards. Network-free.
        #
        # Skipped while a trainer holds the GPU: an uncached scheme takes this into a
        # quantise-and-matmul smoke probe that initialises CUDA in the Unsloth process, and the
        # plan runs before the load's training guard can refuse. Staging needs no GPU.
        # Ranking opens a CUDA context per candidate, which the training guard exists to prevent,
        # so the RANKING waits until training is known idle. Validating and translating the ids
        # does not, so that happens either way: a plan that skipped it accepted a GPU the load
        # would refuse and sized its file set for the wrong card. ONE resolution, reused by
        # preflight and plan.
        gpu_ordinal = None
        training = fam is not None and await asyncio.to_thread(_training_is_active)
        if fam is not None:
            gpu_ordinal = await _selected_gpu_ordinal(request.gpu_ids, allow_ranking = not training)
        if fam is not None and not training:
            await asyncio.to_thread(
                assert_video_precision_available,
                fam,
                model_kind = kind,
                transformer_quant = request.transformer_quant,
                text_encoder_quant = request.text_encoder_quant,
                memory_mode = request.memory_mode,
                # Judged on the card this pick would load on, as the loader does.
                gpu_ordinal = gpu_ordinal,
            )
        plan = await asyncio.to_thread(
            backend.download_plan,
            request.model_path,
            gpu_ordinal = gpu_ordinal,
            gguf_filename = request.gguf_filename,
            base_repo = request.base_repo,
            family_override = request.family_override,
            model_kind = kind,
            hf_token = request.hf_token,
            # The plan must see the encoder policy the load will use: an fp8 request takes a hosted pre-cast encoder, so staging the dense one wastes ~49 GB on LTX-2.
            text_encoder_quant = request.text_encoder_quant,
            # And the denoiser policy, for the same reason: a scheme with a hosted pre-quantized
            # checkpoint replaces the dense DiT, so without this the plan stages 66.3 GB of shards
            # the load never opens.
            transformer_quant = request.transformer_quant,
            # And the MiniMax-H3 partition, because the two denoisers live in separate 66.28 GB
            # subfolders: a ref2va load opens transformer_ref/, which the plan would otherwise
            # miss entirely while staging the fl2va transformer/ it never opens.
            h3_task = request.h3_task,
        )
        return DiffusionDownloadPlanResponse(**plan)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code = 400, detail = redact_native_paths(str(exc)))
    except RuntimeError as exc:
        # Mirrors /video/load and /images/download-plan: the precision gate above raises
        # RuntimeError, and that refusal is a 409, not a server fault.
        raise HTTPException(status_code = 409, detail = redact_native_paths(str(exc)))


@router.post("/video/load", response_model = VideoStatusResponse)
async def load_video_model(
    request: VideoLoadRequest, current_subject: str = Depends(get_current_subject)
):
    return await load_video_model_gated(request, current_subject, user_initiated = True)


async def load_video_model_gated(
    request: VideoLoadRequest,
    current_subject: str,
    *,
    user_initiated: bool = False,
):
    """Everything ``POST /video/load`` does, plus who asked for it.

    Media auto-switch awaits this rather than the route so the idle unload can tell an
    API-loaded pipeline from one the user picked on the Video page.
    """
    from core.inference.diffusion import resolve_local_single_file
    from core.inference.diffusion_device import (
        resolve_diffusion_device_target,
        resolve_selected_cuda_ordinal,
    )
    from core.inference.gpu_arbiter import VIDEO, acquire_for, release
    from core.inference.media_keepwarm import note_load_origin
    from hub.utils.gguf import extract_quant_token
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
            h3_task = request.h3_task,
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
        # Ahead of the precision gate, which has to judge the card this pick would load on.
        # Refused here too, before anything is evicted or staged; begin_load re-checks, but only
        # after the arbiter has taken the GPU.
        gpu_ordinal = await _selected_gpu_ordinal(request.gpu_ids)
        await asyncio.to_thread(
            assert_video_precision_available,
            fam,
            model_kind = kind,
            transformer_quant = request.transformer_quant,
            text_encoder_quant = request.text_encoder_quant,
            # The memory request settles the offload policy for balanced/low_vram before
            # anything is measured, and an offloaded DiT or encoder skips the torchao build.
            memory_mode = request.memory_mode,
            gpu_ordinal = gpu_ordinal,
        )
        # Same bar again, for a speech GGUF picked out of a mixed video repo. The backend's own
        # assertion runs on the load worker, INSIDE acquire_for, so a refusal there arrives
        # having already evicted the chat model this gate exists to preserve. Off-thread because
        # the probe reads a header, and cache-only when the load is not user-initiated, matching
        # the locality promise begin_load makes below.
        from core.inference.diffusion_compat import assert_pick_is_not_speech

        await asyncio.to_thread(
            assert_pick_is_not_speech,
            request.model_path,
            request.gguf_filename,
            request.hf_token,
            user_initiated,
        )
        # Take the GPU from chat only for a non-CPU load. Release stale VIDEO ownership on a CPU load (owner-guarded no-op).
        device = await asyncio.to_thread(lambda: resolve_diffusion_device_target().device)

        def _begin_load():
            # Kicks the (slow) load onto a background thread and returns at once; begin_load itself validates network-free.
            return backend.begin_load(
                request.model_path,
                # a load nobody asked for may not reach the hub: the switch verified locality
                # from the outside, and this makes that promise the loader's own rule
                local_files_only = not user_initiated,
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
                h3_task = request.h3_task,
                gpu_ids = request.gpu_ids,
                # The winner this route already ranked and preflighted, so the load cannot pick a
                # different card from free VRAM that has moved since.
                gpu_ordinal = gpu_ordinal,
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
        # Keyed to the target: this load can still fail with the previous model resident, and
        # its origin must not be read off that model.
        note_load_origin(
            VIDEO,
            request.model_path,
            extract_quant_token(request.gguf_filename) if kind == "gguf" else None,
            # Derived when the caller left it unset, since that is what the backend publishes.
            request.h3_task or _derived_h3_task(request.gguf_filename, kind),
            user_action = user_initiated,
        )
        reset_media_load_progress("video")
        return VideoStatusResponse(**status_dict)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code = 400, detail = redact_native_paths(str(exc)))
    except RuntimeError as exc:
        # A video load is already in progress.
        raise HTTPException(status_code = 409, detail = str(exc))


@router.get("/video/load-progress", response_model = VideoLoadProgressResponse)
async def video_load_progress(current_subject: str = Depends(get_current_subject)):
    from core.inference.video import get_video_backend

    progress = get_video_backend().load_progress()
    fraction = byte_fraction(progress.get("downloaded_bytes"), progress.get("expected_bytes"))
    log_media_load_progress("video", progress.get("phase"), fraction)
    return VideoLoadProgressResponse(**progress)


@router.post("/video/generate", response_model = VideoGenerateResponse)
async def generate_video(
    request: VideoGenerateRequest,
    current_subject: str = Depends(get_current_subject),
    hf_token: Optional[str] = Depends(get_hf_token),
):
    """Start a generation job and return at once (the begin_load pattern): a clip
    takes minutes, and secure mode's tunnel caps the origin response window near
    100 seconds, so the response must not span the generation. The worker runs the
    generate + gallery-persist pipeline; the terminal outcome (completed with the
    saved record / failed with a client-safe error) arrives via generate-progress.

    With media auto-switch on, ``model`` names the video model to generate on and is loaded
    when it is not the resident one."""
    from core.inference.gpu_arbiter import VIDEO
    from core.inference.media_auto_switch import maybe_auto_switch_media_model
    from core.inference.video import get_video_backend
    from core.inference.video_families import (
        VIDEO_GENERATION_BUSY_MSG,
        VIDEO_NOT_LOADED_MSG,
        VideoShapeError,
    )

    def _refuse_unservable_request(pick) -> None:
        """Judge the request against the family being switched TO, before it evicts anything.

        begin_generate judges it against the loaded family under the lock, which is what makes
        the answer race-proof, but by then a request no model could have served has already cost
        the resident pipeline and a multi-minute load. The same rules, applied to the target's
        family and MiniMax-H3 partition, both of which the pick already determines.
        """
        from core.inference.media_model_index import expected_partition
        from core.inference.video import _detect_load_family, resolve_video_model_kind
        from core.inference.video_minimax_h3 import is_h3_native
        from core.inference.video_families import (
            validate_video_flow_controls,
            validate_video_keyframe_conditioning,
            validate_video_reference_conditioning,
            validate_video_request_shape,
        )

        fam = _detect_load_family(pick.model_path, pick.gguf_filename, None)
        if fam is None:
            return
        validate_video_request_shape(fam, request.width, request.height, request.num_frames)
        h3_task = expected_partition(pick)
        validate_video_keyframe_conditioning(
            fam, h3_task, has_keyframes = bool(request.first_frame or request.last_frame)
        )
        # the engine is only knowable up front where the pick decides it, as an h3 gguf does
        kind = resolve_video_model_kind(pick.gguf_filename, pick.model_kind)
        engine = "sd_cpp" if is_h3_native(fam, kind) else None
        validate_video_reference_conditioning(
            fam,
            h3_task,
            has_references = bool(
                request.reference_images or request.reference_videos or request.reference_audios
            ),
            reference_image_size = request.reference_image_size,
            engine = engine,
        )
        validate_video_flow_controls(
            fam, request.flow_shift, request.audio_flow_shift, engine = engine
        )

    # Before the backend is resolved: the requested model may be the one this brings up.
    try:
        await maybe_auto_switch_media_model(
            request.model,
            owner = VIDEO,
            current_subject = current_subject,
            openai_errors = False,
            hf_token = hf_token,
            before_switch = _refuse_unservable_request,
        )
    except VideoShapeError as exc:
        raise HTTPException(status_code = 422, detail = str(exc))
    except ValueError as exc:
        # the conditioning rules, which begin_generate reports the same way below
        raise HTTPException(status_code = 400, detail = str(exc))

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
            first_frame = request.first_frame,
            last_frame = request.last_frame,
            reference_images = request.reference_images,
            reference_videos = [r.model_dump() for r in request.reference_videos or []] or None,
            reference_audios = request.reference_audios,
            reference_image_size = request.reference_image_size,
            flow_shift = request.flow_shift,
            audio_flow_shift = request.audio_flow_shift,
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

    reset_media_generation_progress("video")
    return VideoGenerateResponse()


@router.get("/video/generate-progress", response_model = VideoGenerateProgressResponse)
async def video_generate_progress(current_subject: str = Depends(get_current_subject)):
    from core.inference.video import get_video_backend

    progress = get_video_backend().generate_progress()
    log_media_generation_progress("video", progress)
    return VideoGenerateProgressResponse(**progress)


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
    archived: bool = False,
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
        video_gallery.list_videos,
        limit + 1,
        offset,
        valid = _valid_gallery_video,
        archived = archived,
    )
    has_more = len(records) > limit
    videos = [GalleryVideo(**r) for r in records[:limit]]
    return VideoGalleryListResponse(videos = videos, has_more = has_more)


@router.get("/video/gallery/{video_id}/file")
async def get_gallery_video_file(
    video_id: str, current_subject: str = Depends(get_current_subject)
):
    from core.inference import video_gallery

    # Ownership-gate the serve like delete/clear: resolve only an Unsloth-owned MP4, so a guessed stem cannot stream out a foreign clip.
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
    video_id: str,
    current_subject: str = Depends(get_current_subject),
    no_credential: Annotated[bool, Depends(request_admitted_without_credential)] = False,
):
    """A directly playable, range-capable link for one clip (bearer-gated to mint, HMAC to use).

    Returned as a relative URL so it works behind any proxy the page itself is served through."""
    if no_credential:
        raise HTTPException(
            status_code = 403,
            detail = "Video links can only be created from the Unsloth UI or with an API key.",
        )

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


@router.patch("/video/gallery/{video_id}", response_model = GalleryVideo)
async def update_gallery_video_flags(
    video_id: str,
    patch: GalleryFlagsPatch,
    current_subject: str = Depends(get_current_subject),
):
    """Pin/unpin or archive/restore one clip. Omitted fields are left alone."""
    from core.inference import video_gallery

    try:
        record = await asyncio.to_thread(
            video_gallery.set_flags, video_id, pinned = patch.pinned, archived = patch.archived
        )
    except OSError as exc:
        # The client already applied this optimistically, so a silent miss would look like it stuck
        # and then quietly undo on reload.
        logger.warning("video_gallery.set_flags_failed: %s", exc)
        raise HTTPException(status_code = 500, detail = "Could not save the change to this video.")
    if record is None:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    # Archiving takes the clip off the strip, so the completed-job record must go with it: the page
    # merges that snapshot on mount, which would keep resurrecting the clip it just archived.
    if patch.archived:
        _forget_terminal_video(video_id)
    return GalleryVideo(**record)


@router.delete("/video/gallery/{video_id}")
async def delete_gallery_video(video_id: str, current_subject: str = Depends(get_current_subject)):
    from core.inference import video_gallery

    deleted = await asyncio.to_thread(video_gallery.delete, video_id)
    if not deleted:
        raise HTTPException(status_code = 404, detail = "Video not found.")
    _forget_terminal_video(video_id)
    if not await asyncio.to_thread(_forget_openai_job, video_id):
        raise HTTPException(status_code = 500, detail = "Could not delete the video job.")
    return {"deleted": True}


@router.delete("/video/gallery")
async def clear_gallery_videos(current_subject: str = Depends(get_current_subject)):
    from core.inference import video_gallery
    from core.inference.gallery_flags import FlagsUnavailable

    try:
        cleared = await asyncio.to_thread(video_gallery.clear, return_ids = True)
    except FlagsUnavailable as exc:
        # Refuse rather than delete the archive we cannot prove is archived.
        logger.warning("video_gallery.clear_blocked: %s", exc)
        raise HTTPException(
            status_code = 503,
            detail = "Could not read the gallery's pin/archive data, so clearing was stopped to "
            "avoid deleting archived videos.",
        )
    # Clear-all takes the terminal record's clip with it whatever its id.
    _forget_terminal_video(None)
    failed = await asyncio.to_thread(_forget_openai_jobs, cleared)
    if failed:
        raise HTTPException(status_code = 500, detail = "Could not delete every video job.")
    return {"removed": len(cleared)}


# ── OpenAI-compatible videos API (/v1/videos) ──

_VIDEO_SIZE_RE = _re.compile(r"^(\d{1,5})\s*x\s*(\d{1,5})$")
_VIDEO_SECONDS_MAX = 120.0
_VIDEO_JOB_ID_PREFIX = "video_"
_MAX_REMEMBERED_JOBS = 256
_VIDEO_POLL_AFTER_MS = "2000"
_NO_VIDEO_MODEL_MSG = "No video model loaded. Load a video model first."
_VIDEO_FAILED_CODE = "video_generation_failed"
_VIDEO_CREATE_FIELDS = ("prompt", "model", "seconds", "size")
# Upper bound on how long DELETE waits for a cancelled run to stop writing.
_DELETE_SETTLE_TIMEOUT_S = 10.0


@dataclass
class _VideoJob:
    id: str
    created_at: int
    prompt: str
    model: str
    size: str
    seconds: str
    status: str = "queued"
    progress: int = 0
    completed_at: Optional[int] = None
    error: Optional[dict] = None

    @property
    def terminal(self) -> bool:
        return self.status in ("completed", "failed")


_jobs: dict[str, _VideoJob] = {}
_jobs_lock = threading.Lock()


def _forget_openai_job(video_id: str) -> bool:
    """Forget disk and memory state without letting a stale poll save between them."""
    from core.inference import video_gallery

    with _jobs_lock:
        if not video_gallery.forget_job(video_id):
            return False
        _jobs.pop(video_id, None)
    return True


def _forget_openai_jobs(video_ids: list[str]) -> list[str]:
    failed = []
    for video_id in video_ids:
        if not _forget_openai_job(video_id):
            failed.append(video_id)
    return failed


def _openai_video_error(
    status: int,
    message: str,
    *,
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> HTTPException:
    return HTTPException(
        status_code = status,
        detail = openai_error_body(message, status = status, code = code, param = param),
    )


def _not_found(video_id: str) -> HTTPException:
    return _openai_video_error(
        404, f"No video found with id '{video_id[:128]}'.", code = "video_not_found", param = "video_id"
    )


def _parse_openai_video_size(size: Optional[str]) -> Optional[tuple[int, int]]:
    text = (size or "").strip().lower()
    if text in ("", "auto"):
        return None
    match = _VIDEO_SIZE_RE.match(text)
    if not match:
        raise ValueError("size must be '<width>x<height>', e.g. '768x512'.")
    width, height = int(match.group(1)), int(match.group(2))
    for label, value in (("width", width), ("height", height)):
        if not 32 <= value <= 2048:
            raise ValueError(f"size {label} must be between 32 and 2048.")
    return width, height


def _parse_openai_video_seconds(seconds: Optional[str]) -> Optional[float]:
    text = (seconds or "").strip().lower()
    if text in ("", "auto"):
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError("seconds must be a number of seconds, e.g. '4'.") from None
    if value != value or not 0 < value <= _VIDEO_SECONDS_MAX:
        raise ValueError(f"seconds must be between 0 and {_VIDEO_SECONDS_MAX:g}.")
    return value


def _frames_for_seconds(seconds: float, defaults: dict) -> int:
    fps = int(defaults.get("fps") or 24)
    step = max(1, int(defaults.get("frame_step") or 1))
    offset = max(1, int(defaults.get("frame_offset") or 1))
    wanted = max(offset, int(round(seconds * fps)))
    k = max(1, int(round((wanted - offset) / step)))
    return k * step + offset


def _format_seconds(value: float) -> str:
    return f"{round(float(value), 2):g}"


def _record_epoch(record: dict) -> int:
    raw = record.get("created_at")
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            return 0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo = timezone.utc)
        return int(parsed.timestamp())
    return 0


_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def _sniff_image_type(data: bytes) -> Optional[str]:
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _valid_gallery_video_record(record: dict) -> bool:
    try:
        GalleryVideo(**record)
    except ValidationError:
        return False
    return True


def _record_to_job(record: dict, job: Optional[_VideoJob] = None) -> VideoJob:
    saved_at = _record_epoch(record)
    created = job.created_at if job is not None else saved_at
    completed = (job.completed_at if job is not None else None) or saved_at or created
    return VideoJob(
        id = record["id"],
        model = public_model_id(
            str(record.get("model") or (job.model if job is not None else "") or "video")
        ),
        status = "completed",
        progress = 100,
        created_at = created,
        completed_at = completed,
        prompt = record.get("prompt"),
        size = f"{record.get('width')}x{record.get('height')}",
        seconds = _format_seconds(record.get("duration_s") or 0.0),
    )


def _job_to_openai(job: _VideoJob) -> VideoJob:
    return VideoJob(
        id = job.id,
        model = job.model,
        status = job.status,  # type: ignore[arg-type]
        progress = job.progress,
        created_at = job.created_at,
        completed_at = job.completed_at,
        prompt = job.prompt,
        size = job.size,
        seconds = job.seconds,
        error = VideoJobError(**job.error) if job.error else None,
    )


def _remember_job(job: _VideoJob) -> None:
    from core.inference import video_gallery

    with _jobs_lock:
        pending = [existing for existing in _jobs.values() if not existing.terminal]
    for existing in pending:
        persisted = _job_from_record(video_gallery.get_job(existing.id) or {})
        if persisted is not None and persisted.terminal:
            with _jobs_lock:
                if _jobs.get(existing.id) is existing:
                    _jobs[existing.id] = persisted
    with _jobs_lock:
        _jobs[job.id] = job
        excess = len(_jobs) - _MAX_REMEMBERED_JOBS
        if excess > 0:
            for stale in [j for j in _jobs.values() if j.terminal][:excess]:
                _jobs.pop(stale.id, None)
        try:
            video_gallery.save_job(job.id, asdict(job))
        except OSError as exc:
            logger.warning("openai_videos.persist_job_failed: %s", exc)


def _job_from_record(record: dict) -> Optional[_VideoJob]:
    from core.inference.video_families import VIDEO_CANCELLED_MSG
    try:
        stored = dict(record)
        outcome = stored.pop("_worker_outcome", None)
        job = _VideoJob(**stored)
        if isinstance(outcome, dict):
            message = outcome.get("error")
            job.completed_at = int(outcome.get("completed_at") or job.completed_at or 0) or None
            if message is None:
                job.status, job.progress, job.error = "completed", 100, None
            else:
                message = str(message)
                job.status, job.progress = "failed", 0
                job.error = {
                    "code": "cancelled" if message == VIDEO_CANCELLED_MSG else _VIDEO_FAILED_CODE,
                    "message": message,
                }
        if job.status not in ("queued", "in_progress", "completed", "failed"):
            return None
        _job_to_openai(job)
    except (TypeError, ValueError, OverflowError, ValidationError):
        return None
    return job


def _hydrate_job(video_id: str) -> None:
    from core.inference import video_gallery
    with _jobs_lock:
        if video_id in _jobs:
            return
        job = _job_from_record(video_gallery.get_job(video_id) or {})
        if job is not None:
            _jobs.setdefault(job.id, job)


def _hydrate_jobs() -> list[_VideoJob]:
    from core.inference import video_gallery
    with _jobs_lock:
        jobs = [
            job for raw in video_gallery.list_jobs() if (job := _job_from_record(raw)) is not None
        ]
        jobs.sort(key = lambda job: job.created_at, reverse = True)
        for job in jobs[:_MAX_REMEMBERED_JOBS]:
            _jobs.setdefault(job.id, job)
    return jobs


def _sync_jobs() -> None:
    from core.inference import video_gallery
    from core.inference.video import get_video_backend
    from core.inference.video_families import VIDEO_CANCELLED_MSG

    with _jobs_lock:
        open_jobs = [job for job in _jobs.values() if not job.terminal]
    if not open_jobs:
        return
    gen = get_video_backend().generate_progress()
    now = int(_time.time())
    for job in open_jobs:
        persisted_job = _job_from_record(video_gallery.get_job(job.id) or {})
        if persisted_job is not None and persisted_job.terminal:
            with _jobs_lock:
                if _jobs.get(job.id) is job:
                    _jobs[job.id] = persisted_job
            continue
        status: Optional[str] = None
        progress = 0
        completed_at: Optional[int] = None
        error: Optional[dict] = None
        if gen.get("video_id") == job.id:
            phase = gen.get("phase")
            if phase == "queued":
                status = "queued"
            elif phase in ("denoise", "export") or gen.get("active"):
                fraction = float(gen.get("fraction") or 0.0)
                status = "in_progress"
                progress = 99 if phase == "export" else max(0, min(99, int(fraction * 100)))
            elif phase == "completed":
                status, progress, completed_at = "completed", 100, now
            elif phase == "failed":
                message = str(gen.get("error") or "Video generation failed.")
                status = "failed"
                error = {
                    "code": "cancelled" if message == VIDEO_CANCELLED_MSG else _VIDEO_FAILED_CODE,
                    "message": message,
                }
        if status is None:
            if video_gallery.owned_video_path(job.id) is not None:
                status, progress, completed_at = "completed", 100, now
            else:
                status = "failed"
                error = {
                    "code": _VIDEO_FAILED_CODE,
                    "message": "The generation ended before a clip was saved.",
                }
        with _jobs_lock:
            if _jobs.get(job.id) is not job:
                continue
            if job.terminal:
                continue
            if job.status == "in_progress" and status == "queued":
                continue
            if job.status == "in_progress" and status == "in_progress":
                progress = max(job.progress, progress)
            job.status = status
            job.progress = progress
            if completed_at is not None:
                job.completed_at = completed_at
            if error is not None:
                job.error = error
            persisted = asdict(job)
            try:
                video_gallery.save_job(job.id, persisted)
            except OSError as exc:
                logger.warning("openai_videos.persist_job_failed: %s", exc)


def _await_generate_settled(video_id: str, timeout: float = _DELETE_SETTLE_TIMEOUT_S) -> bool:
    """Block until the run started for ``video_id`` is no longer in flight.

    Bounded, so a wedged backend cannot hold the request open. Returns False when the
    wait expired with the run still live: the caller must not report a deletion it
    could not observe, or the worker commits its sidecar afterwards and the clip
    reappears through retrieve/list.
    """
    from core.inference.video import get_video_backend

    backend = get_video_backend()
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        gen = backend.generate_progress()
        if gen.get("video_id") != video_id or not gen.get("active"):
            return True
        _time.sleep(0.05)
    return False


def _lookup_video(video_id: str) -> Optional[VideoJob]:
    from core.inference import video_gallery

    _hydrate_job(video_id)
    _sync_jobs()
    with _jobs_lock:
        job = _jobs.get(video_id)
    if job is not None and job.status != "completed":
        return _job_to_openai(job)
    record = video_gallery.get_record(video_id)
    if record is None:
        return None
    return _record_to_job(record, job)


def _all_videos() -> list[VideoJob]:
    from core.inference import video_gallery

    persisted_jobs = _hydrate_jobs()
    _sync_jobs()
    with _jobs_lock:
        jobs = {job.id: job for job in persisted_jobs}
        jobs.update(_jobs)
    records = video_gallery.list_videos(None, 0, valid = _valid_gallery_video_record)
    records.extend(
        video_gallery.list_videos(None, 0, valid = _valid_gallery_video_record, archived = True)
    )
    listed = [_record_to_job(record, jobs.get(record["id"])) for record in records]
    seen = {video.id for video in listed}
    listed.extend(
        _job_to_openai(job)
        for job in jobs.values()
        if job.id not in seen and job.status != "completed"
    )
    return listed


async def _read_video_create_body(request: Request) -> tuple[dict, Any]:
    ctype = (request.headers.get("content-type") or "").lower()
    if ctype.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        try:
            form = await request.form()
        except Exception:  # noqa: BLE001
            raise _openai_video_error(400, "Could not parse the multipart form body.")
        fields = {key: form.get(key) for key in _VIDEO_CREATE_FIELDS if key in form}
        reference = form.get("input_reference")
        if reference is None:
            nested_reference = {
                key: form.get(f"input_reference[{key}]")
                for key in ("image_url", "file_id")
                if f"input_reference[{key}]" in form
            }
            reference = nested_reference or None
        return fields, reference
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        raise _openai_video_error(400, "Request body must be JSON or multipart/form-data.")
    if not isinstance(data, dict):
        raise _openai_video_error(400, "Request body must be a JSON object.")
    return {key: data.get(key) for key in _VIDEO_CREATE_FIELDS if key in data}, data.get(
        "input_reference"
    )


def _validate_create_fields(fields: dict) -> VideoJobCreateRequest:
    try:
        return VideoJobCreateRequest(**fields)
    except ValidationError as exc:
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = first.get("loc") or ()
        param = str(loc[0]) if loc else None
        message = str(first.get("msg") or "Invalid request.")
        raise _openai_video_error(400, f"{param}: {message}" if param else message, param = param)


async def _reference_to_data_url(reference: Any) -> Optional[str]:
    if reference is None:
        return None
    if isinstance(reference, dict):
        if reference.get("file_id"):
            raise _openai_video_error(
                400,
                "input_reference.file_id is not supported; send the image as a file upload or a base64 data URL.",
                param = "input_reference",
            )
        reference = reference.get("image_url")
        if reference is None:
            return None
    if isinstance(reference, UploadFile):
        data = await reference.read()
        if not data:
            raise _openai_video_error(400, "input_reference is empty.", param = "input_reference")
        if len(data) > VIDEO_INPUT_REFERENCE_MAX_BYTES:
            raise _openai_video_error(
                400,
                f"input_reference is too large; the limit is {VIDEO_INPUT_REFERENCE_MAX_BYTES // (1024 * 1024)} MB.",
                param = "input_reference",
            )
        content_type = (reference.content_type or "").split(";")[0].strip().lower()
        sniffed = _sniff_image_type(data)
        if not content_type.startswith("image/"):
            if sniffed is None:
                raise _openai_video_error(
                    400, "input_reference must be an image.", param = "input_reference"
                )
            content_type = sniffed
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
    if isinstance(reference, str):
        text = reference.strip()
        if not text:
            return None
        if text.lower().startswith(("http://", "https://")):
            raise _openai_video_error(
                400,
                "Remote input_reference URLs are not fetched; send the image as a file upload or a base64 data URL.",
                param = "input_reference",
            )
        return text
    raise _openai_video_error(
        400, "input_reference must be an image file or a base64 data URL.", param = "input_reference"
    )


@openai_router.post("/videos", response_model = VideoJob)
async def openai_create_video(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    hf_token: Optional[str] = Depends(get_hf_token),
):
    # The body has to be read before the row can name a model and prompt, so the parse
    # sits out here and the rest of the handler runs inside the monitor context.
    from routes.inference import _monitored_media_request

    fields, raw_reference = await _read_video_create_body(request)
    body = _validate_create_fields(fields)
    reference = await _reference_to_data_url(raw_reference)
    from core.inference.api_monitor import api_monitor

    async with _monitored_media_request(
        request,
        model = public_model_id(body.model) or body.model or "",
        prompt = body.prompt,
        subject = current_subject,
    ) as monitor_id:
        job = await _create_openai_video(body, reference, current_subject, hf_token)
        api_monitor.relabel(monitor_id, job.model)
        return job


# Split out so the API monitor row can wrap the whole handler without reindenting it,
# exactly as the OpenAI image route does.
async def _create_openai_video(
    body: VideoJobCreateRequest,
    reference: Optional[str],
    current_subject: str,
    hf_token: Optional[str],
):
    from core.inference.gpu_arbiter import VIDEO
    from core.inference.media_auto_switch import (
        maybe_auto_switch_media_model,
        resident_answers_media_request,
    )
    from core.inference.video import get_video_backend
    from core.inference.video_families import (
        VIDEO_GENERATION_BUSY_MSG,
        VIDEO_MODEL_CHANGED_MSG,
        VIDEO_NOT_LOADED_MSG,
        VideoShapeError,
    )
    from utils.openai_auto_switch_settings import get_media_auto_switch_enabled

    try:
        size = _parse_openai_video_size(body.size)
    except ValueError as exc:
        raise _openai_video_error(400, str(exc), param = "size")
    try:
        seconds = _parse_openai_video_seconds(body.seconds)
    except ValueError as exc:
        raise _openai_video_error(400, str(exc), param = "seconds")
    width, height = size if size is not None else (None, None)
    if reference is not None:
        # Check readability before a model switch. The conditioning path applies its own
        # keyframe or Ref2VA limit later, so this uses the broadest bounded source policy.
        from core.inference.diffusion import decode_b64_image
        from core.inference.video_minimax_h3 import (
            H3_REF_IMAGE_SOURCE_MAX_PIXELS,
            H3_REF_IMAGE_SOURCE_MAX_SIDE,
        )
        try:
            await asyncio.to_thread(
                decode_b64_image,
                reference,
                mode = "RGB",
                max_side = H3_REF_IMAGE_SOURCE_MAX_SIDE,
                max_pixels = H3_REF_IMAGE_SOURCE_MAX_PIXELS,
            )
        except Exception:  # noqa: BLE001 -- any decode failure is client input feedback
            raise _openai_video_error(
                400, "input_reference is not a readable image.", param = "input_reference"
            )

    def _refuse_unservable_request(pick) -> None:
        from core.inference.media_model_index import expected_partition
        from core.inference.video import _detect_load_family
        from core.inference.video_families import (
            validate_video_keyframe_conditioning,
            validate_video_reference_conditioning,
            validate_video_request_shape,
        )
        from core.inference.video_minimax_h3 import H3_TASK_REFERENCES

        fam = _detect_load_family(pick.model_path, pick.gguf_filename, None)
        if fam is None:
            return
        # Judge the duration against the family being switched TO, using its own lattice.
        # Passing None here accepted any seconds and only refused it in begin_generate --
        # after the resident pipeline had been evicted and the target fully loaded.
        want_frames = (
            _frames_for_seconds(
                seconds,
                {
                    "fps": fam.default_fps,
                    "frame_step": fam.frame_step,
                    "frame_offset": fam.frame_offset,
                },
            )
            if seconds is not None
            else None
        )
        validate_video_request_shape(fam, width, height, want_frames)
        h3_task = expected_partition(pick)
        ref2va = h3_task == H3_TASK_REFERENCES
        validate_video_keyframe_conditioning(
            fam, h3_task, has_keyframes = reference is not None and not ref2va
        )
        validate_video_reference_conditioning(
            fam,
            h3_task,
            has_references = reference is not None and ref2va,
        )

    pin_requested_model = bool(
        isinstance(body.model, str) and body.model.strip() and get_media_auto_switch_enabled()
    )
    try:
        await maybe_auto_switch_media_model(
            body.model,
            owner = VIDEO,
            current_subject = current_subject,
            openai_errors = True,
            hf_token = hf_token,
            before_switch = _refuse_unservable_request,
        )
    except VideoShapeError as exc:
        raise _openai_video_error(
            400, str(exc), param = "seconds" if "frame count" in str(exc) else "size"
        )
    except ValueError as exc:
        raise _openai_video_error(400, str(exc), param = "input_reference")

    backend = get_video_backend()
    expected_state = None
    if pin_requested_model:
        status, expected_state = await asyncio.to_thread(backend.generation_snapshot)
        if not await asyncio.to_thread(
            resident_answers_media_request, status, body.model, owner = VIDEO
        ):
            raise _openai_video_error(
                409,
                VIDEO_MODEL_CHANGED_MSG,
                code = "model_changed",
                param = "model",
            )
    else:
        status = await asyncio.to_thread(backend.status)
    if not status.get("loaded"):
        raise HTTPException(status_code = 503, detail = _NO_VIDEO_MODEL_MSG)
    defaults = status.get("defaults") or {}
    num_frames = _frames_for_seconds(seconds, defaults) if seconds is not None else None
    video_id = _VIDEO_JOB_ID_PREFIX + uuid.uuid4().hex
    try:
        generate_kwargs = dict(
            prompt = body.prompt,
            width = width,
            height = height,
            duration_s = seconds,
            input_reference = reference,
            video_id = video_id,
        )
        if expected_state is not None:
            generate_kwargs["expected_state"] = expected_state
        resolved = await asyncio.to_thread(backend.begin_generate, **generate_kwargs)
    except VideoShapeError as exc:
        raise _openai_video_error(
            400, str(exc), param = "seconds" if "frame count" in str(exc) else "size"
        )
    except ValueError as exc:
        raise _openai_video_error(
            400, str(exc), param = "input_reference" if reference is not None else None
        )
    except RuntimeError as exc:
        msg = str(exc)
        if msg == VIDEO_NOT_LOADED_MSG:
            raise HTTPException(status_code = 503, detail = _NO_VIDEO_MODEL_MSG)
        if msg == VIDEO_GENERATION_BUSY_MSG:
            raise _openai_video_error(409, msg)
        if msg == VIDEO_MODEL_CHANGED_MSG:
            raise _openai_video_error(409, msg, code = "model_changed", param = "model")
        logger.error("openai_videos.generate_failed: %s", exc, exc_info = True)
        raise HTTPException(status_code = 500, detail = "Video generation failed.")

    # begin_generate hands back the canvas it resolved. Without it a reference-image
    # request reported the family's first preset while the clip rendered at the source
    # aspect, so the job advertised one size and the finished record another.
    if isinstance(resolved, dict) and resolved.get("width") and resolved.get("height"):
        width, height = int(resolved["width"]), int(resolved["height"])
    else:
        presets = defaults.get("resolution_presets") or []
        if size is None and presets:
            width, height = int(presets[0][0]), int(presets[0][1])
    # Describe the job from what begin_generate reserved, falling back to the status()
    # snapshot only where it said nothing. A load committing between that snapshot and
    # the reservation swaps the family underneath, and the snapshot's fps would then
    # date a frame count the new model never used.
    reserved = resolved if isinstance(resolved, dict) else {}
    run_frames = reserved.get("num_frames") or num_frames
    fps = reserved.get("fps") or defaults.get("fps")
    if run_frames and fps:
        seconds_text = _format_seconds(int(run_frames) / float(fps))
    elif seconds is None and defaults.get("num_frames") and fps:
        seconds_text = _format_seconds(int(defaults["num_frames"]) / float(fps))
    else:
        seconds_text = body.seconds or "auto"
    job = _VideoJob(
        id = video_id,
        created_at = int(_time.time()),
        prompt = body.prompt,
        model = public_model_id(
            str(reserved.get("model") or status.get("repo_id") or body.model or "video")
        ),
        size = f"{width}x{height}" if width and height else "auto",
        seconds = seconds_text,
    )
    _remember_job(job)
    return _job_to_openai(job)


@openai_router.get("/videos", response_model = VideoJobListResponse)
async def openai_list_videos(
    limit: int = Query(20, ge = 1, le = 100),
    after: Optional[str] = None,
    order: Literal["asc", "desc"] = "desc",
    current_subject: str = Depends(get_current_subject),
):
    videos = await asyncio.to_thread(_all_videos)
    videos.sort(key = lambda video: (video.created_at, video.id), reverse = order == "desc")
    if after:
        index = next((i for i, video in enumerate(videos) if video.id == after), None)
        if index is None:
            raise _openai_video_error(
                400, f"No video found with id '{after[:128]}'.", param = "after"
            )
        videos = videos[index + 1 :]
    page = videos[:limit]
    return VideoJobListResponse(
        data = page,
        first_id = page[0].id if page else None,
        last_id = page[-1].id if page else None,
        has_more = len(videos) > limit,
    )


@openai_router.get("/videos/{video_id}", response_model = VideoJob)
async def openai_retrieve_video(
    video_id: str,
    response: Response,
    current_subject: str = Depends(get_current_subject),
):
    video = await asyncio.to_thread(_lookup_video, video_id)
    if video is None:
        raise _not_found(video_id)
    if video.status in ("queued", "in_progress"):
        response.headers["openai-poll-after-ms"] = _VIDEO_POLL_AFTER_MS
    return video


@openai_router.get("/videos/{video_id}/content")
async def openai_download_video_content(
    video_id: str,
    variant: str = "video",
    current_subject: str = Depends(get_current_subject),
):
    from core.inference import video_gallery

    normalized_variant = (variant or "video").strip().lower()
    if normalized_variant not in ("video", "thumbnail"):
        raise _openai_video_error(
            400, "Unsupported variant. Use 'video' or 'thumbnail'.", param = "variant"
        )
    path = await asyncio.to_thread(video_gallery.owned_video_path, video_id)
    if path is None:
        video = await asyncio.to_thread(_lookup_video, video_id)
        if video is None:
            raise _not_found(video_id)
        if video.status == "failed":
            raise _openai_video_error(
                400,
                video.error.message if video.error else "Video generation failed.",
                code = video.error.code if video.error else _VIDEO_FAILED_CODE,
            )
        raise _openai_video_error(
            400,
            "Video is still generating; retrieve it until its status is 'completed'.",
            code = "video_not_ready",
        )
    if normalized_variant == "thumbnail":
        try:
            thumbnail = await asyncio.to_thread(video_gallery.thumbnail, video_id)
        except RuntimeError as exc:
            raise _openai_video_error(501, str(exc), code = "video_thumbnail_unavailable") from exc
        if thumbnail is None:  # The clip was deleted between the ownership check and decode.
            raise _not_found(video_id)
        return Response(
            content = thumbnail,
            media_type = "image/webp",
            headers = {
                "Cache-Control": "private, max-age=31536000, immutable",
                "Content-Disposition": f'attachment; filename="{video_id}.webp"',
            },
        )
    from fastapi.responses import FileResponse

    return FileResponse(
        path,
        media_type = "video/mp4",
        filename = f"{video_id}.mp4",
        headers = {"Cache-Control": "private, max-age=31536000, immutable"},
    )


@openai_router.delete("/videos/{video_id}", response_model = VideoJobDeleteResponse)
async def openai_delete_video(video_id: str, current_subject: str = Depends(get_current_subject)):
    from core.inference import video_gallery
    from core.inference.video import get_video_backend

    video = await asyncio.to_thread(_lookup_video, video_id)
    if video is None:
        raise _not_found(video_id)
    if video.status in ("queued", "in_progress"):
        await asyncio.to_thread(get_video_backend().cancel_generate, video_id)
        # The run can reach its terminal state between the lookup and the cancel, so a
        # refused cancellation is not proof that nothing was written. Let the worker
        # settle, then fall through to the same delete the completed branch performs --
        # otherwise the clip persists and the "deleted" job reappears through
        # retrieve/list on the very next call.
        if not await asyncio.to_thread(_await_generate_settled, video_id):
            raise _openai_video_error(
                409,
                "The video is still being written; retry the delete once it is no longer generating.",
                code = "video_not_ready",
            )
    deleted = await asyncio.to_thread(video_gallery.delete, video_id)
    if deleted:
        _forget_terminal_video(video_id)
    elif await asyncio.to_thread(video_gallery.get_record, video_id) is not None:
        raise _openai_video_error(500, "Could not delete the video; retry the request.")
    if not await asyncio.to_thread(_forget_openai_job, video_id):
        raise _openai_video_error(500, "Could not delete the video job; retry the request.")
    return VideoJobDeleteResponse(id = video_id)

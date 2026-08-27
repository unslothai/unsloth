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
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import ValidationError

from auth.authentication import get_current_subject, request_admitted_without_credential
from hub.dependencies import get_hf_token
from loggers import get_logger
from models.inference import (
    DiffusionDownloadPlanResponse,
    GalleryFlagsPatch,
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
    return {"deleted": True}


@router.delete("/video/gallery")
async def clear_gallery_videos(current_subject: str = Depends(get_current_subject)):
    from core.inference import video_gallery
    from core.inference.gallery_flags import FlagsUnavailable

    try:
        removed = await asyncio.to_thread(video_gallery.clear)
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
    return {"removed": removed}

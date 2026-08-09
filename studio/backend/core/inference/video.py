# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local text-to-video inference backend (diffusers).

A deliberate sibling of ``DiffusionBackend`` rather than a mode of it: video
pipelines take frame/fps arguments, return frame stacks (plus synchronized audio
for LTX-2) instead of PIL images, and persist MP4s -- none of the image module's
img2img/inpaint/ControlNet/LoRA surface applies. The concurrency skeleton
(load token, per-generation cancel event, split status/generate locks) is copied
from the image backend so the two cannot diverge in lifecycle behaviour, and the
hardware/optimisation layers are IMPORTED from the image stack unchanged:
device/dtype resolution, memory planning + offload tiers, attention backends,
speed profiles (regional torch.compile), and FBCache step caching all operate on
``pipe.transformer`` generically.

Video-specific behaviour lives here:
- the runtime headroom estimate is frames-aware (``estimate_video_runtime_mib``):
  the VAE decode of a whole clip is the memory peak, not the denoise;
- VAE tiling is always enabled (decode of 100+ frames at 720p-class resolutions
  spikes far beyond the image case, and tiling's quality cost is negligible);
- generation snaps num_frames to the family's temporal lattice (k * step + 1)
  and width/height to its required multiple BEFORE latents are allocated;
- the result is encoded to MP4 (H.264) via diffusers' PyAV-backed exporter,
  muxing the audio track for families that produce one.

Loads are gated to trusted repos exactly like the image backend: unsloth/*, the
family's official base repos, or a local path the user explicitly picked.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import tempfile
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

from .diffusion_attention import (
    SDPA_MATH_ONLY_MESSAGE,
    apply_attention_backend,
    install_hunyuan_attention_trim,
    sdpa_math_only,
    select_attention_backend,
)
from .diffusion_cache import (
    FBCACHE_MIN_STEPS,
    TC_AUTO,
    TC_FBCACHE,
    apply_step_cache,
    maybe_toggle_step_cache,
    normalize_transformer_cache,
)
from .diffusion_device import resolve_diffusion_device_target
from .diffusion_memory import (
    apply_memory_plan,
    estimate_gguf_resident_mib,
    estimate_safetensors_dense_mib,
    estimate_video_runtime_mib,
    file_size_mib,
    normalize_memory_mode,
    plan_diffusion_memory,
    snapshot_device_memory,
)
from .diffusion_speed import (
    SPEED_DEFAULT,
    SPEED_MAX,
    SPEED_OFF,
    apply_speed_optims,
    resolve_speed_mode,
    restore_backend_flags,
    snapshot_backend_flags,
)
from .diffusion_auto_policy import (
    _QUANT_STEADY_FACTOR,
    RESOLVED_APPLIED,
    RESOLVED_FELL_BACK,
    RESOLVED_UNSUPPORTED,
    build_resolved_record,
    precision_fallback_allowed,
    precision_refusal_message,
)
from .diffusion_transformer_quant import (
    TQ_AUTO,
    dense_transformer_supported,
    explain_unusable_scheme,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)
from .diffusion import _memory_request_forces_offload
from .diffusion_precision import (
    effective_te_quant,
    normalize_te_quant,
    quantize_text_encoders,
    te_quant_needs_resident_weights,
    te_quant_supported,
    torchao_quantize_importable,
)
from .video_families import (
    VIDEO_CANCELLED_MSG,
    VIDEO_GENERATION_BUSY_MSG,
    VIDEO_NOT_LOADED_MSG,
    VideoFamily,
    default_video_generation_params,
    detect_video_family,
    resolve_video_base_repo,
    snap_num_frames,
    snap_video_size,
    supported_video_family_names,
    validate_video_request_shape,
    video_family_prequant_repo,
    video_family_prequant_schemes,
)
from .video_minimax_h3 import (
    H3_ANCHOR_FIRST,
    H3_ANCHOR_LAST,
    H3_CANVAS_MAX_PIXELS,
    H3_CANVAS_SHORT_EDGE,
    H3_REF_SIZE_MATCH,
    H3_REF_SIZE_MAX,
    H3_TASK_REFERENCES,
    fit_h3_keyframe,
    fit_h3_reference_image,
    h3_canvas_for_aspect,
    h3_conditioning_mode,
    h3_transformer_resident_gb,
    h3_transformer_task,
)
from .video_minimax_h3_te import (
    H3_TE_QUANT_REPO,
    h3_te_quant_scheme,
    h3_te_resident_gb,
)
from utils.hardware import clear_gpu_cache

# Shared with the image backend so both pin every loader call to the same live cache root.
from core.inference.diffusion import hub_cache_dir

logger = get_logger(__name__)

# Load kinds (mirror the image backend): gguf (single-file GGUF DiT + base repo), single_file (safetensors DiT), pipeline (full diffusers repo).
_MODEL_KINDS = frozenset({"gguf", "single_file", "pipeline"})

# Vendor base repos allowed to load as full (non-GGUF) artifacts. Exact-match, lowercased, safetensors-only, no remote code.
_TRUSTED_NON_GGUF_VIDEO_REPOS = frozenset(
    {
        "lightricks/ltx-2",
        "lightricks/ltx-2.3",
        "lightricks/ltx-2.3-fp8",
        # Wan2.2 official diffusers base repos: safetensors-only, no remote code.
        "wan-ai/wan2.2-ti2v-5b-diffusers",
        "wan-ai/wan2.2-t2v-a14b-diffusers",
        # HunyuanVideo-1.5 community Diffusers repacks (tencent's own repo has no model_index.json).
        "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-480p_t2v",
        "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-720p_t2v",
        "minimaxai/minimax-h3",
    }
)


def resolve_video_model_kind(gguf_filename: Optional[str], model_kind: Optional[str]) -> str:
    """Classify a load request; explicit model_kind wins, else the filename decides."""
    if model_kind:
        kind = model_kind.strip().lower()
        if kind not in _MODEL_KINDS:
            raise ValueError(
                f"Unknown model_kind '{model_kind}'. Expected one of {sorted(_MODEL_KINDS)}."
            )
        return kind
    if not gguf_filename:
        return "pipeline"
    return "gguf" if gguf_filename.strip().lower().endswith(".gguf") else "single_file"


def assert_video_precision_available(
    fam: Any,
    *,
    model_kind: str,
    transformer_quant: Optional[str] = None,
    text_encoder_quant: Optional[str] = None,
    memory_mode: Optional[str] = None,
) -> None:
    """Raise ``RuntimeError`` (the route's 409) when an EXPLICIT precision cannot run here.

    The image twin, ``DiffusionBackend.assert_precision_available``: only the network-free
    host-level impossibilities (wrong load kind, no dense-quant path, a scheme this GPU or family
    rules out). Footprint-dependent declines belong to ``load_pipeline``; ``auto`` is never
    refused.

    Public because the ROUTE has to make this call itself, before it takes the GPU: the copy in
    ``begin_load`` runs inside ``acquire_for``, which evicts chat under the arbiter lock before the
    register callback."""
    if precision_fallback_allowed():
        return
    pinned = normalize_transformer_quant(transformer_quant)
    te_mode = normalize_te_quant(text_encoder_quant)
    if pinned is None and te_mode is None:
        return
    # One probe for both checks (it reads the live device).
    target = resolve_diffusion_device_target()
    # A modular-workflow family (MiniMax-H3) never reaches the memory plan the offload refusals
    # below reason about: load_pipeline hands it to _load_h3_modular_pipeline, which always uses
    # the ComponentsManager auto CPU offload and PINS a pre-quantized denoiser resident, so
    # 'balanced' / 'low_vram' do not put the quantised DiT under a Module.to() offload hook.
    # Refusing those picks here would 409 a combination that loads.
    forces_offload = not getattr(fam, "modular_workflow", None) and _memory_request_forces_offload(
        memory_mode, False
    )
    if pinned is not None and pinned != TQ_AUTO:
        reason = None
        if model_kind != "pipeline":
            reason = (
                f"the dense DiT quant applies to full-pipeline loads only, and this is a "
                f"'{model_kind}' load, which runs the precision its checkpoint carries"
            )
        elif not dense_transformer_supported(target):
            reason = "this device cannot run a dense torchao quant (it needs a CUDA GPU in bf16)"
        elif forces_offload:
            # balanced and low_vram name their offload policy without measuring anything, and
            # offload hooks move modules with Module.to(), which torchao tensors do not survive.
            # load_pipeline therefore skips the dense build and the strict refusal landed after
            # acquire_for and the teardown had already evicted the resident model.
            reason = (
                f"'{normalize_memory_mode(memory_mode)}' memory places the DiT under CPU "
                "offload, and torchao quantised tensors cannot be moved by the offload hooks"
            )
        elif (
            select_transformer_quant_scheme(
                target,
                pinned,
                family = getattr(fam, "name", None),
                # Pre-eviction: an OOM in the smoke probe is the resident model, not the scheme.
                unproven_ok = True,
            )
            is None
        ):
            # An explicit scheme is never swapped for another, so a None means the family's
            # measured deny list, a torchao that cannot run here, or a GPU without the kernels.
            # They read the same to the selector and need different fixes from the user.
            reason = explain_unusable_scheme(getattr(fam, "name", None), pinned)
        if reason is not None:
            raise RuntimeError(
                precision_refusal_message(
                    "transformer_quant", pinned, reason, off_label = "Off to run the DiT at bf16"
                )
            )
    # The mode the loader will ACTUALLY attempt, not the raw request: an explicit int8
    # on a family with no keep-bf16 schedule is rewritten to layerwise fp8 before
    # support is consulted, and that path needs no torchao. Refusing on the raw int8
    # rejected loads the runtime would run and report as fell_back -- Windows ROCm,
    # where the torchao stub kills int8 while fp8 still works.
    te_effective = effective_te_quant(te_mode, getattr(fam, "name", None))
    te_reason = None
    if te_quant_needs_resident_weights(te_effective) and not torchao_quantize_importable():
        # Same as the image gate: the casters import torchao only after the pipeline is built.
        te_reason = (
            "torchao is not importable on this install, and these encoder modes are torchao "
            "quantisations"
        )
    elif te_effective is not None and not te_quant_supported(target, te_effective):
        te_reason = (
            "this device does not have the tensor cores that backend needs (a CUDA GPU in "
            "bf16, plus fp8 / int8 / NVFP4 support depending on the mode)"
        )
    elif te_quant_needs_resident_weights(te_effective) and forces_offload:
        # Same fence on the encoder: the loader reports those modes unsupported once offload is
        # active, and by then the resident model is gone. Layerwise fp8 is a dtype cast.
        te_reason = (
            f"'{normalize_memory_mode(memory_mode)}' memory places the text encoder under CPU "
            "offload, and torchao quantised tensors cannot be moved by the offload hooks"
        )
    if te_reason is not None:
        raise RuntimeError(
            precision_refusal_message(
                "text_encoder_quant",
                te_mode,
                te_reason,
                off_label = "leave it unset to keep the dense bf16 encoder",
                auto_available = False,
            )
        )


def _is_trusted_video_repo(repo_id: str) -> bool:
    """Whether a NON-GGUF load may deserialise this repo (see the image twin)."""
    try:
        if Path(repo_id).expanduser().exists():
            return True
    except OSError:
        pass
    rid = repo_id.strip().lower()
    return rid.startswith("unsloth/") or rid in _TRUSTED_NON_GGUF_VIDEO_REPOS


def _picked_gguf_arch(repo_id: str, gguf_filename: str) -> Optional[str]:
    """``general.architecture`` of a picked GGUF, or None. The Video picker admits a GGUF by its
    arch (not its name) -- for a LOCAL dir (``repo_id`` is a directory) AND for a cached HUB repo
    (the cached-gguf listing tags it by arch too), so a renamed/opaquely-named file whose path
    carries no family token still shows up; reading the arch lets the loader resolve the same
    family the picker offered. Reads the local file when present, else the cached hub blob
    (network-free via try_to_load_from_cache). Header-only, bounds-checked."""
    try:
        from pathlib import Path

        path = Path(repo_id).expanduser() / gguf_filename
        if not path.is_file():
            # Not a local dir: resolve a cached HUB blob (no network). Probe active, legacy AND default cache roots, or a non-active-root GGUF 400s.
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(repo_id, gguf_filename)
            if not isinstance(cached, str):
                from hub.utils.paths import hf_default_cache_dir, legacy_hf_cache_dir
                for root_fn in (legacy_hf_cache_dir, hf_default_cache_dir):
                    try:
                        cached = try_to_load_from_cache(
                            repo_id, gguf_filename, cache_dir = str(root_fn())
                        )
                    except Exception:  # noqa: BLE001 -- a bad/absent root just falls through
                        cached = None
                    if isinstance(cached, str):
                        break
            if not isinstance(cached, str):
                return None
            path = Path(cached)
        from utils.models.gguf_metadata import read_gguf_general_metadata

        arch = (read_gguf_general_metadata(str(path)) or {}).get("general.architecture")
        return arch.strip() if isinstance(arch, str) and arch.strip() else None
    except Exception:  # noqa: BLE001 -- a header read glitch just falls through to name detection
        return None


# Enough for a 15-second reference video after base64 decoding.
_MAX_REFERENCE_MEDIA_BYTES = 96 * 1024 * 1024


def _decode_b64_media(data: Optional[str]) -> bytes:
    """Decode a base64 media payload, optionally wrapped in a data URL."""
    import base64
    import binascii

    raw = (data or "").strip()
    if not raw:
        raise ValueError("A reference was sent empty.")
    if raw.startswith("data:"):
        _, _, raw = raw.partition(",")
    try:
        blob = base64.b64decode(raw, validate = False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 media data: {exc}") from exc
    if not blob:
        raise ValueError("A reference decoded to no data.")
    if len(blob) > _MAX_REFERENCE_MEDIA_BYTES:
        raise ValueError(
            f"A reference is too large ({len(blob) / 1e6:.0f} MB); the limit is "
            f"{_MAX_REFERENCE_MEDIA_BYTES / 1e6:.0f} MB."
        )
    return blob


class _VideoGenerationCancelled(Exception):
    """Unwinds a denoise loop that has no cooperative interrupt (no step callback);
    generate() maps it to the VIDEO_CANCELLED_MSG sentinel the routes 409 on."""


@contextlib.contextmanager
def _scheduler_step_progress(pipe: Any, on_step: Any):
    """Progress + cancellation for pipelines WITHOUT callback_on_step_end.

    HunyuanVideo15Pipeline exposes no per-step callback, but every denoise step
    makes exactly one ``scheduler.step`` call, so wrapping that method gives the
    same per-step tick the callback path gets. ``on_step`` receives the 1-based
    step count and may raise (_VideoGenerationCancelled) to abort the loop. The
    original method is always restored, even when the pipeline raises.
    """
    scheduler = pipe.scheduler
    original = scheduler.step
    count = {"n": 0}

    def _step(*args: Any, **kwargs: Any) -> Any:
        count["n"] += 1
        on_step(count["n"])
        return original(*args, **kwargs)

    scheduler.step = _step
    try:
        yield
    finally:
        scheduler.step = original


def _detect_load_family(
    repo_id: str, gguf_filename: Optional[str], family_override: Optional[str]
) -> Optional[VideoFamily]:
    """Family detection shared by validate_load_request and the load worker: the
    repo id first, then the picked filename -- a local directory or generically
    named repo often carries the family token only in the checkpoint filename,
    and the worker must resolve the same family the validator accepted."""
    fam = detect_video_family(repo_id, family_override) or (
        detect_video_family(f"{repo_id}/{gguf_filename}")
        if gguf_filename and not family_override
        else None
    )
    if fam is None and gguf_filename and not family_override:
        # A renamed GGUF carries no family token, so resolve via general.architecture. No-backend archs still yield None (a 400).
        arch = _picked_gguf_arch(repo_id, gguf_filename)
        if arch:
            fam = detect_video_family(repo_id, override = arch)
    return fam


def _ensure_mp4_encoder_available() -> None:
    """Fail a load fast when PyAV is missing: the export otherwise dies AFTER a
    multi-minute denoise, which is the worst possible time to learn about it."""
    try:
        import av  # noqa: F401
    except Exception as exc:  # noqa: BLE001 -- any import failure means no encoder
        raise ValueError(
            "Video generation needs the 'av' package (PyAV) to encode MP4s. "
            "Install it with: pip install av"
        ) from exc


@dataclass(frozen = True)
class _VideoLoadState:
    """Everything about the currently-loaded video pipeline, swapped as one unit."""

    pipe: Any
    family: VideoFamily
    repo_id: str
    base_repo: str
    device: str
    dtype: str
    kind: str
    engine: str = "diffusers"
    gguf_filename: Optional[str] = None
    # Resident MiniMax-H3 denoiser partition, if any.
    h3_task: Optional[str] = None
    offload_policy: str = "none"
    vae_tiling: bool = True
    memory_mode: str = "auto"
    speed_mode: str = SPEED_OFF
    speed_optims: tuple = ()
    backend_flags: Optional[dict] = None
    attention_backend: Optional[str] = None
    transformer_cache: Optional[str] = None
    # AUTO on a cache-capable DiT: generate() toggles FBCache across FBCACHE_MIN_STEPS; an explicit request is never toggled.
    cache_auto: bool = False
    # Inputs the generation-time toggle re-applies (quantised threshold + override).
    cache_quant_active: bool = False
    cache_threshold: Optional[float] = None
    # Dense transformer quant engaged ("int8"|"fp8"|"nvfp4"|"mxfp8") or None. Pipeline-kind only; quantised in place onto the low-precision tensor cores.
    transformer_quant: Optional[str] = None
    # Text-encoder quant engaged ("fp8"|"fp8_dynamic"|"int8"|"nvfp4") or None. Often the largest resident; shrunk in place.
    text_encoder_quant: Optional[str] = None
    # MiniMax-H3 only: the pre-quantized denoiser was PINNED to the device and taken out of the
    # offload rotation, so it is resident alongside whatever component is running. The memory
    # preflight has to know, because that turns its floor from a max into a sum.
    h3_denoiser_pinned: bool = False
    resolved: Optional[dict] = None


@dataclass
class _VideoLoadingState:
    repo_id: str
    base_repo: str
    expected_bytes: Optional[int] = None
    error: Optional[str] = None
    # Companion repos this load is ALSO pulling from, beyond repo_id / base_repo (the native H3
    # GGUF and component repos). Mirrors the image backend's _SdLoading.asset_repos.
    asset_repos: tuple[str, ...] = ()


def _h3_te_canonical(repo_id: Optional[str]) -> str:
    """A repo id normalised for an EXACT identity compare: mirrors folded onto the id they copy,
    then trimmed and lowercased. Deliberately no tail-segment tolerance -- ``someone/MiniMax-H3``
    is a different repo with different weights."""
    from .diffusion_families import canonical_base

    return canonical_base(repo_id).strip().lower()


def _progress(phase: Optional[str], **extra: Any) -> dict[str, Any]:
    return {"phase": phase, **extra}


# ── dual-DiT (Wan2.2-A14B MoE) helpers ────────────────────────────────────────
# The optimisation helpers and the quantiser all act on ``pipe.transformer``. Wan2.2-A14B is a dual-expert MoE
# (transformer = high-noise, transformer_2 = low-noise), so present the second DiT AS ``pipe.transformer`` via a thin proxy.


def _transformer_names(pipe: Any, fam: VideoFamily) -> tuple[str, ...]:
    """Attribute names of the denoiser(s) on ``pipe`` to optimise. Just
    ("transformer",) for a single-DiT family; also "transformer_2" for an MoE family
    whose second expert is actually present (a checkpoint may ship only the first)."""
    names = ["transformer"]
    if fam.is_moe and getattr(pipe, "transformer_2", None) is not None:
        names.append("transformer_2")
    return tuple(names)


class _SecondDiTView:
    """A thin proxy that makes ``pipe.transformer_2`` look like ``pipe.transformer`` to a
    helper that hardcodes ``getattr(pipe, "transformer")``, while every other attribute
    (vae, components, __call__, ...) reads through to the real pipe unchanged.

    This lets the existing single-DiT helpers optimise the second expert without a fork:
    ``apply_speed_optims(_SecondDiTView(pipe), ...)`` compiles / caches / sets attention on
    ``transformer_2``. Only ever wrapped around an MoE pipe (guarded by fam.is_moe)."""

    def __init__(self, pipe: Any) -> None:
        object.__setattr__(self, "_pipe", pipe)

    @property
    def transformer(self) -> Any:
        return self._pipe.transformer_2

    def __getattr__(self, name: str) -> Any:
        # Only reached for attrs not on the instance/class, so delegate to the real pipe.
        return getattr(object.__getattribute__(self, "_pipe"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Writes land on the real pipe (else a helper reassignment vanishes); ``transformer`` mirrors onto the second expert.
        pipe = object.__getattribute__(self, "_pipe")
        setattr(pipe, "transformer_2" if name == "transformer" else name, value)


def _views_for(pipe: Any, fam: VideoFamily) -> tuple[Any, ...]:
    """The pipe view(s) to pass through the ``getattr(pipe, "transformer")`` helpers so
    they cover every denoiser: the real pipe (its ``transformer``), plus a
    ``_SecondDiTView`` (its ``transformer_2``) for a dual-DiT MoE family. A single-DiT
    load returns just ``(pipe,)``, so its behaviour is unchanged."""
    if fam.is_moe and getattr(pipe, "transformer_2", None) is not None:
        return (pipe, _SecondDiTView(pipe))
    return (pipe,)


def _log_failed_generation(request_shape: Optional[dict[str, Any]], exc: BaseException) -> None:
    """Record WHAT was being generated when it failed, next to the exception.

    ``video.generate_failed`` logs the error and nothing else, which for the OOM in #8225 meant the
    only server-side evidence was "Tried to allocate 66.54 GiB" -- the resolution and frame count
    behind it had to be recovered by dividing that number by the head count. This is the missing
    half of that record.

    Cancels and "not loaded" are outcomes, not failures, so they are not reported. Never raises:
    logging a failure must not replace it with a different one."""
    if request_shape is None:
        return
    try:
        if str(exc) in (VIDEO_CANCELLED_MSG, VIDEO_NOT_LOADED_MSG):
            return
        if type(exc).__name__ == "_VideoGenerationCancelled":
            return
        # A ValueError is client input feedback here, exactly as _run_generate treats it: it is
        # returned to the caller as a validation message and deliberately gets no
        # video.generate_failed record. Logging the request shape at ERROR for one would turn a
        # rejected resolution or frame count into a server-error entry.
        if isinstance(exc, ValueError):
            return
        logger.error(
            "video.generate_failed_request: %s",
            " ".join(f"{k}={v}" for k, v in request_shape.items()),
        )
        # An OOM under the math SDPA backend is the #8225 shape exactly: the score matrix is
        # quadratic in the token count, so the fix is fewer tokens, not a smaller checkpoint.
        # Named here so the next report arrives with the diagnosis already in it.
        #
        # Only when the run was on NATIVE SDPA. A recorded attention_backend means an explicit
        # kernel (AITER, xFormers, ...) engaged and torch's own dispatch was not what ran, so
        # probing native kernels would answer about code this generation never executed -- on a
        # ROCm card with AITER active that turns every OOM into a confident false diagnosis.
        if (
            _is_out_of_memory(exc)
            and request_shape.get("attention_backend") is None
            and sdpa_math_only(_probe_target(request_shape))
        ):
            logger.error("video.generate_failed_request: %s", SDPA_MATH_ONLY_MESSAGE)
    except Exception:  # noqa: BLE001 — diagnostics never mask the real failure
        pass


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether ``exc`` is an allocator failure, by name and text rather than by class: torch
    raises ``torch.OutOfMemoryError`` on CUDA and a plain RuntimeError on some backends."""
    if any(cls.__name__ in ("OutOfMemoryError", "OutOfMemoryError_") for cls in type(exc).__mro__):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "tried to allocate" in text


def _probe_target(request_shape: dict[str, Any]) -> Any:
    """A minimal device target for the SDPA probe, built from the recorded request.

    Carries the dtype the failed generation actually ran in. Leaving it None defaults the probe to
    fp16, but every video family promotes a resolved fp16 to fp32, so the probe would answer for a
    precision the run never used -- and the fused kernels are half-precision only, so that is the
    difference between "fused kernels exist here" and "this run had none"."""
    dtype = None
    recorded = request_shape.get("dtype")
    if recorded:
        try:
            import torch
            candidate = getattr(torch, str(recorded).replace("torch.", ""), None)
            if isinstance(candidate, torch.dtype):
                dtype = candidate
        except Exception:  # noqa: BLE001 — fall back to the probe's own default
            dtype = None
    return types.SimpleNamespace(device = request_shape.get("device"), dtype = dtype)


class VideoBackend:
    """One loaded video pipeline; loads swap it atomically (same model as images)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._state: Optional[_VideoLoadState] = None
        self._loading: Optional[_VideoLoadingState] = None
        self._load_token = 0
        self._cancel_event = threading.Event()
        self._active_generate_cancel: Optional[threading.Event] = None
        # How many unloads / superseding loads are waiting on _generate_lock to free this pipeline. A generation queued behind
        # the active one holds no cancel event yet, so without this fence it could win the lock after an eject and denoise a
        # whole new clip against a pipeline being freed. A count, so concurrent teardowns each own their own release.
        self._teardown_waiters = 0
        # Generation progress, written by the step callback / phase transitions.
        self._gen: dict[str, Any] = {"active": False}
        # True from begin_generate() until its worker records a terminal state, so a second call is refused while it runs.
        self._generate_job_active = False

    # ── validation ───────────────────────────────────────────────────────────

    def validate_load_request(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        h3_task: Optional[str] = None,
    ) -> VideoFamily:
        """Cheap, network-free validation shared by the route and the load path."""
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        # A -GGUF repo picked without a quant filename resolves to pipeline kind and would only fail in from_pretrained after eviction.
        if kind == "pipeline" and repo_id.strip().lower().rstrip("/").endswith("-gguf"):
            raise ValueError(
                f"'{repo_id}' is a GGUF repo: pick one of its .gguf files "
                "(gguf_filename) instead of loading it as a diffusers pipeline."
            )
        fam = _detect_load_family(repo_id, gguf_filename, family_override)
        if fam is None:
            raise ValueError(
                f"'{repo_id}' is not a supported text-to-video model. Supported families: "
                f"{', '.join(supported_video_family_names())}. If this is a variant of one "
                f"of them, pass family_override with that family name."
            )
        # ── modular-workflow refusals, before anything heavier.
        # Deliberately the FIRST thing after the family resolves: everything below reaches into
        # diffusers (assert_pipeline_class_available, the transformer_class probe), so a refusal
        # placed after them would be unreachable on any install whose diffusers cannot even be
        # imported -- and these two are exactly the picks that cost the most to discover late.
        if fam.modular_workflow and kind == "single_file":
            # A modular workflow has no single-file assembly: its components each load through
            # their own from_pretrained from the modular index, and nothing consumes a lone
            # .safetensors DiT. Today that only surfaces inside the loader, i.e. after ~98.7 GB
            # has downloaded AND after the resident pipeline was torn down to make room for it.
            gguf_hint = (
                f", or a .gguf checkpoint from '{fam.gguf_repo}' for a quantized single-file load"
                if fam.gguf_repo
                else ""
            )
            raise ValueError(
                f"'{fam.name}' cannot load from a single .safetensors checkpoint: it is assembled "
                f"by its Modular Diffusers workflow, which builds every component from a repo. "
                f"Load the diffusers pipeline repo '{fam.base_repo}' for the full bfloat16 "
                f"model{gguf_hint}."
            )
        if fam.modular_workflow and kind == "pipeline":
            # Same normaliser the load uses, so a malformed value raises the identical message it
            # would below and only a real scheme reaches the availability check.
            requested_scheme = normalize_transformer_quant(transformer_quant)
            # The base the modular load resolves for a pipeline kind IS repo_id, so a
            # variant-keyed checkpoint is judged against the one the load will ask for and
            # validation can never refuse a load that would have worked.
            quant_base = repo_id
            # "auto" is a request for the backend's own choice, not for a specific scheme, so it is
            # never refused: it simply stays on the released bfloat16 components.
            if requested_scheme is not None and requested_scheme != TQ_AUTO:
                # The hosted checkpoints are FL2VA denoisers. The two partitions share module
                # shapes and the same base model, so a Ref2VA load would install one, pass every
                # metadata check, and have load_components skip the real Ref2VA transformer --
                # generating from the wrong partition instead of failing. Refuse until a matching
                # checkpoint exists; there is no in-place quantise seam to fall back on.
                from .video_minimax_h3 import H3_TASK_REFERENCES
                if (h3_task or "").strip().lower() == H3_TASK_REFERENCES:
                    raise ValueError(
                        f"transformer_quant '{requested_scheme}' is unavailable for "
                        f"'{fam.name}' reference video: the hosted pre-quantized checkpoints are "
                        f"keyframe (fl2va) denoisers, and seeding one into the reference workflow "
                        f"would generate from the wrong partition. Load reference video without "
                        f"transformer_quant, or pick a keyframe task."
                    )
                if video_family_prequant_repo(fam, requested_scheme, quant_base) is None:
                    # There is no in-place quantise seam here: the workflow's own component loader
                    # builds the denoiser, so a scheme without a hosted pre-quantized checkpoint
                    # cannot be served at all, and letting it through would download ~98.7 GB of
                    # dense weights and then silently ignore the request.
                    available = video_family_prequant_schemes(fam)
                    hint = (
                        f"Use one of: {', '.join(available)}."
                        if available
                        else "Leave transformer_quant unset for this model."
                    )
                    raise ValueError(
                        f"transformer_quant '{requested_scheme}' is unavailable for "
                        f"'{fam.name}': its Modular Diffusers workflow builds the denoiser itself, "
                        f"so only schemes with a hosted pre-quantized checkpoint can be applied "
                        f"and the dense weights cannot be quantized in place. {hint}"
                    )
        from .video_minimax_h3 import is_h3_native, validate_h3_transformer_filename

        if is_h3_native(fam, kind):
            validate_h3_transformer_filename(gguf_filename or "")
            # The GGUF filename and explicit task must name the same partition.
            picked = h3_transformer_task(gguf_filename or "")
            if h3_task and h3_task != picked:
                raise ValueError(
                    f"'{Path(gguf_filename or '').name}' is the {picked} partition, but the "
                    f"load asked for {h3_task}. Pick the matching checkpoint."
                )
        else:
            # Refuse a too-old diffusers here rather than deep in the load.
            from .diffusion_families import assert_pipeline_class_available
            assert_pipeline_class_available(fam.pipeline_class, fam.name)
            if fam.modular_workflow:
                import diffusers
                if not hasattr(diffusers, fam.transformer_class):
                    raise ValueError(
                        "MiniMax-H3 needs the Diffusers revision bundled with this Studio "
                        "version. Reinstall Studio dependencies and retry."
                    )
        if kind != "gguf" and not _is_trusted_video_repo(repo_id):
            raise ValueError(
                f"Non-GGUF video loads are limited to unsloth/* repos, the official "
                f"family base repos, and local paths; '{repo_id}' is neither."
            )
        # Companions load with from_pretrained, so a base repo is held to the non-GGUF bar: a GGUF pick must not smuggle in a remote base.
        if base_repo and (base_repo or "").strip() and not _is_trusted_video_repo(base_repo):
            raise ValueError(
                f"base_repo is limited to unsloth/* repos, the official family base "
                f"repos, and local paths; '{base_repo}' is neither."
            )
        # A local base_repo loads as a full pipeline (needs model_index.json); reject a non-pipeline one here, before the load.
        from core.inference.diffusion import _assert_local_base_is_pipeline

        _assert_local_base_is_pipeline(base_repo)
        if kind in ("gguf", "single_file") and not gguf_filename:
            raise ValueError("A gguf/single_file load needs the checkpoint filename.")
        if kind in ("gguf", "single_file") and fam.is_moe:
            # A single checkpoint carries one expert; the other would load dense bf16, off-plan.
            raise ValueError(
                f"'{fam.name}' is a dual-expert model: a single {kind} file covers only "
                f"one of its two transformers. Load the diffusers pipeline repo "
                f"('{fam.base_repo}') instead."
            )
        # A missing local checkpoint must fail HERE, before the route evicts a resident model.
        if kind in ("gguf", "single_file"):
            # Fail a kind/extension mismatch before the GPU handoff: gguf needs .gguf, single_file needs .safetensors.
            is_gguf_name = (gguf_filename or "").lower().endswith(".gguf")
            if kind == "gguf" and not is_gguf_name:
                raise ValueError("a 'gguf' load requires a .gguf checkpoint name.")
            if kind == "single_file" and is_gguf_name:
                raise ValueError("a .gguf checkpoint needs model_kind 'gguf', not 'single_file'.")
            if kind == "single_file" and not (gguf_filename or "").lower().endswith(".safetensors"):
                raise ValueError(
                    f"'{gguf_filename}' is not a loadable single-file checkpoint "
                    f"(expected a .safetensors name; use a .gguf name for a GGUF load)."
                )
            root = Path(repo_id).expanduser()
            # Path-shaped: "."/".." prefix, a backslash (never in "org/name"), or an absolute path, so a missing local pick fails before the handoff.
            path_shaped = (
                repo_id.startswith(("/", "\\", "~", ".")) or "\\" in repo_id or root.is_absolute()
            )
            if root.is_dir():
                from .diffusion_families import resolve_local_gguf_child
                try:
                    resolve_local_gguf_child(root, gguf_filename or "")
                except Exception as exc:  # noqa: BLE001 -- surface as client input error
                    raise ValueError(str(exc)) from exc
            elif root.is_file():
                # The loader hands a local FILE straight through (ignoring gguf_filename), so the file's own suffix must match the kind.
                suffix = root.suffix.lower()
                if kind == "gguf" and suffix != ".gguf":
                    raise ValueError(
                        f"Local checkpoint '{repo_id}' is not a .gguf file; a 'gguf' load "
                        f"needs a .gguf checkpoint."
                    )
                if kind == "single_file" and suffix != ".safetensors":
                    raise ValueError(
                        f"Local checkpoint '{repo_id}' is not a .safetensors file; a "
                        f"'single_file' load needs a .safetensors checkpoint."
                    )
            elif path_shaped:
                raise ValueError(f"Local model path '{repo_id}' does not exist.")
        # A local pipeline pick must be a diffusers directory (model_index.json), else it would only fail after eviction.
        if kind == "pipeline":
            root = Path(repo_id).expanduser()
            # Gate on .exists() (not .is_dir()) so a local FILE picked as a pipeline is rejected too.
            indexes = (
                ("model_index.json", "modular_model_index.json")
                if fam.modular_workflow
                else ("model_index.json",)
            )
            if root.exists() and not (
                root.is_dir() and any((root / name).is_file() for name in indexes)
            ):
                raise ValueError(
                    f"Local pipeline path is not a diffusers directory "
                    f"(no {' or '.join(indexes)}): {repo_id}"
                )
        # Reject a malformed transformer_quant cheaply, before the handoff (pipeline-kind only, matching the image backend).
        normalize_transformer_quant(transformer_quant)
        # Reject a malformed text_encoder_quant the same way. Every kind: an unsupported scheme is
        # a bad request regardless of whether this family has a hosted quantized encoder for it.
        normalize_te_quant(text_encoder_quant)
        _ensure_mp4_encoder_available()
        return fam

    # ── background load + progress ───────────────────────────────────────────

    def begin_load(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        model_kind: Optional[str] = None,
        h3_task: Optional[str] = None,
    ) -> dict[str, Any]:
        """Validate, then run the (slow) load on a daemon thread. Returns at once."""
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            base_repo = base_repo,
            family_override = family_override,
            model_kind = model_kind,
            transformer_quant = transformer_quant,
            text_encoder_quant = text_encoder_quant,
            h3_task = h3_task,
        )
        # Refuse an EXPLICIT precision this host can never honor BEFORE the load starts, so the
        # route answers 409 with the reason instead of evicting the resident model and pulling
        # tens of GB first. Footprint-dependent declines still surface via load-progress. The
        # ROUTE makes the same call earlier still (see assert_video_precision_available): this one
        # runs inside acquire_for, which has already evicted chat by the time it fires.
        assert_video_precision_available(
            fam,
            model_kind = resolve_video_model_kind(gguf_filename, model_kind),
            transformer_quant = transformer_quant,
            text_encoder_quant = text_encoder_quant,
        )
        with self._lock:
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A video load is already in progress.")
            self._load_token += 1
            token = self._load_token
            # A NEW event per load, never a clear() of the shared one: unload() sets the event the running worker holds but also
            # drops _loading, so the next begin_load would clear the object that worker watches. A fresh object leaves it set.
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._loading = _VideoLoadingState(repo_id = repo_id, base_repo = fam.base_repo)

        threading.Thread(
            target = self._run_load,
            kwargs = dict(
                repo_id = repo_id,
                gguf_filename = gguf_filename,
                base_repo = base_repo,
                family_override = family_override,
                hf_token = hf_token,
                memory_mode = memory_mode,
                speed_mode = speed_mode,
                attention_backend = attention_backend,
                transformer_cache = transformer_cache,
                transformer_cache_threshold = transformer_cache_threshold,
                transformer_quant = transformer_quant,
                text_encoder_quant = text_encoder_quant,
                model_kind = model_kind,
                h3_task = h3_task,
                _load_token = token,
                _cancel_event = cancel_event,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        # This load's own event: a later load replaces self._cancel_event rather than clearing it.
        cancel_event = kwargs.pop("_cancel_event", None) or self._cancel_event
        try:
            fam = _detect_load_family(
                kwargs["repo_id"], kwargs.get("gguf_filename"), kwargs.get("family_override")
            )
            kind = resolve_video_model_kind(kwargs.get("gguf_filename"), kwargs.get("model_kind"))
            from .video_minimax_h3 import is_h3_native

            if is_h3_native(fam, kind):
                self._run_load_h3_native(
                    fam = fam,
                    token = token,
                    cancel_event = cancel_event,
                    **kwargs,
                )
                with self._lock:
                    if self._load_token == token:
                        self._loading = None
                return
            base = (
                kwargs["repo_id"]
                if kind == "pipeline"
                else resolve_video_base_repo(fam, kwargs.get("base_repo"))
            )
            kwargs["base_repo"] = base
            # An fp8 encoder request loads a hosted pre-cast checkpoint, so neither the estimate nor the pull includes those dense shards.
            te_sources = self._te_prequant_sources(fam, kwargs.get("text_encoder_quant"))
            # The same deal for the denoiser: a hosted pre-quantized checkpoint replaces the dense
            # DiT shards, so the estimate and the scoped pull below both drop them.
            skip_transformer_weights = self._denoiser_prequant_covered(
                fam, kwargs.get("transformer_quant"), base
            )
            # And for MiniMax-H3's conditioner, whose hosted quantized artifact replaces the base
            # repo's 62 GB dense text_encoder/ shards outright.
            h3_te_scheme = self._h3_te_quant_scheme(fam, kwargs.get("text_encoder_quant"), base)
            expected = self._estimate_download_bytes(
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                base,
                kwargs.get("hf_token"),
                kind,
                te_sources = te_sources,
                skip_transformer_weights = skip_transformer_weights,
                h3_te_scheme = h3_te_scheme,
            )
            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.base_repo = base
                    # The conditioner artifact comes from a THIRD repo, neither repo_id nor
                    # base_repo, so without this the delete-cached guard would let it be deleted
                    # out from under the in-flight fetch (or out from under the assembly, while
                    # the base snapshot that no longer carries a dense encoder is still coming
                    # down). Same registration the native H3 path makes for its companions.
                    if h3_te_scheme:
                        self._loading.asset_repos = self._loading.asset_repos + (H3_TE_QUANT_REPO,)
                    self._loading.expected_bytes = expected
            # Checkpoint downloads outside the lock so an unload can preempt the multi-GB pull; companions pre-download the same way.
            checkpoint_local: Optional[Path] = None
            if kwargs.get("gguf_filename") and not Path(kwargs["repo_id"]).expanduser().exists():
                from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
                checkpoint_local = Path(
                    hf_hub_download_with_xet_fallback(
                        kwargs["repo_id"],
                        kwargs["gguf_filename"],
                        kwargs.get("hf_token"),
                        cancel_event = cancel_event,
                    )
                )
            # An LTX-2.3 checkpoint supplies the VAEs/vocoder/connectors, so the base pull shrinks to scheduler + TE + tokenizer; recompute the estimate.
            ltx23 = False
            if fam is not None and fam.name == "ltx-2" and kind != "pipeline":
                from .video_ltx2 import is_ltx23_checkpoint

                probe = checkpoint_local
                if probe is None:
                    # Local repos: a bare file, or a dir child via the same resolver load_pipeline uses. Unresolvable keeps the wide pull.
                    root = Path(kwargs["repo_id"]).expanduser()
                    if root.is_file():
                        probe = root
                    elif root.is_dir():
                        try:
                            probe = self._resolve_checkpoint_path(
                                kwargs["repo_id"],
                                kwargs.get("gguf_filename"),
                                kwargs.get("hf_token"),
                            )
                        except Exception:  # noqa: BLE001 -- surfaced by load_pipeline
                            probe = None
                ltx23 = probe is not None and is_ltx23_checkpoint(probe)
                if ltx23:
                    expected = self._estimate_download_bytes(
                        kwargs["repo_id"],
                        kwargs.get("gguf_filename"),
                        base,
                        kwargs.get("hf_token"),
                        kind,
                        ltx23 = True,
                        te_sources = te_sources,
                        skip_transformer_weights = skip_transformer_weights,
                        h3_te_scheme = h3_te_scheme,
                    )
                    with self._lock:
                        if self._load_token == token and self._loading is not None:
                            self._loading.expected_bytes = expected
            # Only a pre-cast checkpoint actually on disk earns the dense skip below.
            te_skipped = self._fetch_te_prequant(
                te_sources, kwargs.get("hf_token"), cancel_event = cancel_event
            )
            kwargs["_te_prequant_skipped"] = te_skipped
            # Same rule for the H3 conditioner: the artifact has to be on disk before the base pull
            # is allowed to leave the dense encoder behind.
            h3_te_skipped = self._fetch_h3_te_quant(
                h3_te_scheme, kwargs.get("hf_token"), cancel_event = cancel_event
            )
            base_local = self._predownload_base(
                base,
                kwargs.get("hf_token"),
                kind,
                ltx23 = ltx23,
                skip_te_components = te_skipped + h3_te_skipped,
                skip_transformer_weights = skip_transformer_weights,
                cancel_event = cancel_event,
            )
            # The 2.3 assembly pulls per component from the hub id (its snapshot lacks the base VAEs), so it only gets the warmed cache.
            kwargs["_base_local_dir"] = None if ltx23 else base_local
            self.load_pipeline(**kwargs)
            with self._lock:
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            # A failed/cancelled load never commits _VideoLoadState, so roll back the process-wide speed globals here (token-scoped).
            self._rollback_precommit_globals(token)
            if self._load_token != token:
                return
            logger.error("video.load_failed: %s", exc)
            # Free the debris of a failed construction: nothing was committed, so nothing else releases the VRAM.
            try:
                clear_gpu_cache()
            except Exception:  # noqa: BLE001 -- cleanup is best-effort
                pass
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))

    def _run_load_h3_native(
        self,
        *,
        fam: VideoFamily,
        token: Optional[int],
        cancel_event: threading.Event,
        repo_id: str,
        gguf_filename: Optional[str] = None,
        hf_token: Optional[str] = None,
        memory_mode: Optional[str] = None,
        **_: Any,
    ) -> None:
        """Download and commit the four-file stable-diffusion.cpp H3 runtime."""
        from huggingface_hub import HfApi

        from .sd_cpp_args import SdCppModelFiles, offload_flags
        from .diffusion_engine_router import _install_accelerator_for
        from .sd_cpp_backend import (
            _install_allowed,
            ensure_h3_sd_cpp_binary,
            sd_cpp_lists_accelerator_device,
        )
        from .sd_cpp_engine import SdCppEngine
        from .video_minimax_h3 import (
            H3_AUDIO_VAE,
            H3_COMPONENT_REPO,
            H3_GGUF_REPO,
            H3_VIDEO_VAE,
            h3_download_error,
            h3_text_encoder_filename,
        )
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        filename = gguf_filename or ""
        qwen_filename = h3_text_encoder_filename(filename)
        requests = (
            (repo_id, filename),
            (H3_GGUF_REPO, qwen_filename),
            (H3_COMPONENT_REPO, H3_VIDEO_VAE),
            (H3_COMPONENT_REPO, H3_AUDIO_VAE),
        )
        total = 0
        try:
            api = HfApi(token = hf_token or None)
            for repo, wanted in requests:
                if Path(repo).expanduser().exists():
                    continue
                info = api.model_info(repo, files_metadata = True)
                total += sum(
                    int(s.size or 0) for s in (info.siblings or []) if s.rfilename == wanted
                )
        except Exception:  # noqa: BLE001 -- progress estimate is optional
            total = 0
        with self._lock:
            if self._load_token == token and self._loading is not None:
                self._loading.base_repo = fam.base_repo
                # The download list below pulls from the H3 companion repos too, and neither is
                # repo_id or base_repo, so without this the delete-cached guard would let one be
                # deleted out from under the in-flight load. The committed twin is loaded_repo_ids().
                self._loading.asset_repos = (H3_GGUF_REPO, H3_COMPONENT_REPO)
                self._loading.expected_bytes = total or None

        resolved: list[Path] = []
        for repo, wanted in requests:
            if cancel_event.is_set():
                raise RuntimeError(VIDEO_CANCELLED_MSG)
            root = Path(repo).expanduser()
            if root.is_file():
                local = root
            elif root.is_dir():
                from .diffusion_families import resolve_local_gguf_child
                local = resolve_local_gguf_child(root, wanted)
            else:
                try:
                    local = Path(
                        hf_hub_download_with_xet_fallback(
                            repo, wanted, hf_token, cancel_event = cancel_event
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- re-raised below, narrowed by name
                    raise h3_download_error(repo, wanted, exc) from exc
            resolved.append(local)

        target = resolve_diffusion_device_target()
        allow_install = _install_allowed()
        # The H3-gated ensure, not the plain one: a build that predates H3 runs fine and so clears
        # the version() gate below, then aborts on the first generation, i.e. after the whole
        # bundle has downloaded.
        binary = ensure_h3_sd_cpp_binary(
            allow_install = allow_install,
            accelerator = _install_accelerator_for(target.backend),
        )
        native_device = target.device
        if target.backend not in ("cpu", "mps") and not sd_cpp_lists_accelerator_device(binary):
            # Upstream currently publishes no Linux CUDA archive. Keep the picker
            # functional with the CPU prebuilt when the user has not supplied a
            # locally compiled CUDA binary through the normal sd.cpp discovery path.
            #
            # The test is what the binary actually offers, not whether one was returned: the
            # accelerator install fails only ONCE, and every later load finds the CPU binary this
            # branch put there and gets it back unchanged. "binary is truthy" therefore skipped
            # the fallback from the second load on, left native_device on the GPU, and applied GPU
            # offload policy and held the VIDEO claim while sd-cli ran wholly on the CPU -- so the
            # next chat/image acquire evicted a model to make room for one that was never there.
            binary = ensure_h3_sd_cpp_binary(allow_install = allow_install, accelerator = "cpu")
            native_device = "cpu"
        engine = SdCppEngine(binary)
        if not binary or engine.version() is None:
            raise RuntimeError(
                "stable-diffusion.cpp could not be installed or started for MiniMax-H3."
            )
        requested_mode = normalize_memory_mode(memory_mode) or "auto"
        policy = {
            "auto": "none" if native_device == "cpu" else "group",
            "fast": "none",
            "balanced": "group",
            "low_vram": "model",
        }[requested_mode]
        # NOT --vae-on-cpu. H3 ships an audio VAE, and its 1-D convolutions abort on the CPU
        # path: ggml_conv_1d hardcodes an F16 im2col destination (ggml/src/ggml.c) and
        # ggml_compute_forward_im2col_f16 then asserts the KERNEL is F16, while sd.cpp's
        # audio_conv_weight_type maps only BF16 to F16 and lets F32 through. The result is
        # GGML_ASSERT(src0->type == GGML_TYPE_F16) failed, SIGABRT, exit 134, deterministically.
        # Bisected on the flags: --vae-on-cpu with --audio-vae aborts, without --audio-vae it
        # renders in 95.87s, and converting the audio VAE checkpoint to fp16 does NOT help, so
        # the type is imposed inside sd.cpp rather than by the file and cannot be fixed by
        # shipping a different checkpoint. low_vram is the one mode a small-card user reaches
        # for, so drop the flag rather than the mode; --offload-to-cpu and --clip-on-cpu, which
        # are where the savings actually are, still apply.
        #
        # The abort itself is fixed in the Unsloth sd.cpp fork (it casts F32 conv1d kernels to
        # F16 in-graph on CPU backends), so this is no longer only a crash workaround, and it
        # should not be reverted when that fix reaches the pinned prebuilt. Measured on a build
        # carrying the fix, 640x384, 25 frames, 4 steps, q4_K, with --offload-to-cpu
        # --clip-on-cpu already on: adding --vae-on-cpu moved peak VRAM 12.42 -> 12.42 GiB and
        # wall time 20.9s -> 100.4s. Under --offload-to-cpu the peak is set by the streamed
        # denoiser, so the flag buys nothing and costs 4.8x.
        native_offload = tuple(
            offload_flags(policy, vae_tiling = False, diffusion_fa = True, vae_on_cpu = False)
        )
        from .video_minimax_h3 import MiniMaxH3NativeRuntime

        runtime = MiniMaxH3NativeRuntime(
            engine = engine,
            files = SdCppModelFiles(
                diffusion_model = str(resolved[0]),
                llm = str(resolved[1]),
                vae = str(resolved[2]),
                audio_vae = str(resolved[3]),
            ),
            offload_flags = native_offload,
        )

        with self._lock:
            if token is not None and token != self._load_token:
                raise RuntimeError("Video load was cancelled or superseded.")
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            self._teardown_waiters += 1
        with self._generate_lock:
            with self._lock:
                try:
                    if token is not None and token != self._load_token:
                        raise RuntimeError("Video load was cancelled or superseded.")
                    self._teardown_state_locked()
                    self._state = _VideoLoadState(
                        pipe = runtime,
                        family = fam,
                        repo_id = repo_id,
                        base_repo = fam.base_repo,
                        device = native_device,
                        dtype = Path(filename).stem.split("-")[-1],
                        kind = "gguf",
                        engine = "sd_cpp",
                        h3_task = h3_transformer_task(filename),
                        gguf_filename = filename,
                        offload_policy = policy,
                        vae_tiling = False,
                        memory_mode = requested_mode,
                        attention_backend = "flash",
                        resolved = build_resolved_record(
                            {
                                "memory_mode": (memory_mode, policy, "native model offload"),
                                "attention_backend": (
                                    None,
                                    "flash",
                                    "sd.cpp diffusion flash attention",
                                ),
                            }
                        ),
                    )
                finally:
                    self._teardown_waiters -= 1
        if native_device == "cpu":
            # /video/load acquired the VIDEO GPU claim off the resolved device target, but no
            # accelerator binary was available and this runtime committed to the CPU build, so it
            # holds no VRAM. Drop the stale claim, or the next chat/image acquire evicts a model
            # that is not on the GPU. release_if keeps the token check atomic against a newer load
            # that took ownership already. Mirrors /images/load's release for a CPU-only native load.
            from .gpu_arbiter import VIDEO, release_if
            release_if(VIDEO, lambda: token is None or token == self._load_token)

    def _rollback_precommit_globals(self, token: Optional[int]) -> None:
        """Restore process-wide speed globals (cudnn.benchmark / TF32 / the compiled
        GGUF dequantizer) for a load that died BEFORE committing _VideoLoadState.
        _teardown_state_locked only restores from the committed state's snapshot, so an
        uncommitted load would otherwise leak its profile into the next speed=off
        load. Token-scoped: when a newer load has already taken the snapshot slot,
        the stale worker must leave the globals alone."""
        stored = getattr(self, "_precommit_globals", None)
        if stored is None:
            return
        stored_token, flags = stored
        if token is not None and stored_token is not None and stored_token != token:
            return
        self._precommit_globals = None
        restore_backend_flags(flags)
        from . import diffusion_gguf_compile

        diffusion_gguf_compile.uninstall_all()

    # LTX-2.3 gets DiT/connectors/VAEs/vocoder from the checkpoint + extras, so only the 2.0 base scheduler / text encoder / tokenizer are pulled.
    _LTX23_BASE_PREFIXES = ("scheduler/", "text_encoder/", "tokenizer/")

    @staticmethod
    def _te_prequant_sources(fam: Any, text_encoder_quant: Optional[str]) -> dict[str, Any]:
        """``{component: source}`` for the text encoders this load will take PRE-CAST from a
        hosted checkpoint instead of the base repo's dense weights (``{}`` when none)."""
        from .diffusion_te_prequant import te_prequant_sources
        return te_prequant_sources(
            fam,
            te_quant_mode = text_encoder_quant,
            target = resolve_diffusion_device_target(),
        )

    @staticmethod
    def _h3_te_quant_scheme(
        fam: Any,
        text_encoder_quant: Optional[str],
        base: Optional[str] = None,
    ) -> Optional[str]:
        """The hosted QUANTIZED conditioner scheme this pick would load, or None.

        The one resolver the plan, the byte estimate, the pre-fetch and the load itself all ask, so
        none of them can stage an encoder another one skipped -- staging the dense conditioner for a
        quantized load wastes 62 GB, and dropping one the load actually wants costs a surprise
        mid-load pull of the same size.

        Only a modular-workflow family qualifies. Every other family casts its own dense encoder in
        place and still needs those shards.

        And only against the base the artifact was cut from. A custom derivative can keep the
        Qwen3-VL architecture and still carry different conditioner weights, and the strict load
        cannot tell the difference -- every name and shape still matches -- so it would silently
        condition on someone else's encoder. ``te_base_equivalent`` is the same test the pre-cast
        encoder path uses: the same base, or one verified to ship byte-identical component weights.
        Anything else keeps the pipeline's own dense encoder.

        Pure, never raises, never touches the network."""
        try:
            if not getattr(fam, "modular_workflow", None):
                return None
            from .diffusion_te_prequant import te_base_equivalent

            canonical = getattr(fam, "base_repo", None)
            if not canonical or not base or not te_base_equivalent(canonical, base):
                return None
            return h3_te_quant_scheme(normalize_te_quant(text_encoder_quant))
        except Exception:  # noqa: BLE001 -- an unanswerable probe keeps the dense encoder
            return None

    @staticmethod
    def _h3_te_index_source(pipe: Any) -> Optional[str]:
        """The repo THIS pipeline's own modular index names as the source of its ``text_encoder``.

        ``modular_model_index.json`` records a ``pretrained_model_name_or_path`` per component, and
        diffusers has already parsed it by the time the pipeline exists, so this is free and cannot
        disagree with what ``load_components`` would have fetched. ``repo`` is the field's
        deprecated spelling; ``ComponentSpec.__post_init__`` folds it into the new name, and the
        fallback here covers a spec built by hand.

        None means the question could not be answered. The caller treats that as a refusal: not
        knowing which conditioner a pipeline uses is not a licence to substitute one."""
        try:
            spec = pipe.get_component_spec("text_encoder")
        except Exception:  # noqa: BLE001 -- no spec, no substitution
            return None
        source = getattr(spec, "pretrained_model_name_or_path", None) or getattr(spec, "repo", None)
        # A list means several sources for one component; nothing here can vouch for that.
        return source if isinstance(source, str) and source else None

    @staticmethod
    def _h3_te_quant_hub_files(
        scheme: Optional[str], api: Any
    ) -> tuple[Optional[str], list[tuple[str, int]]]:
        """``(repo_id, [(rfilename, size)])`` for the hosted quantized conditioner, or ``(None, [])``.

        The conditioner mirror of ``_denoiser_prequant_hub_files``, and it earns its keep the same
        way: the base entry drops the dense ``text_encoder/`` shards whenever this resolves, so
        without it the artifact that replaces them is in no entry at all and the disk preflight
        passes on a volume that cannot hold the 27 GB it is about to pull."""
        from .video_minimax_h3_te import h3_te_quant_filename

        filename = h3_te_quant_filename(scheme)
        if filename is None:
            return None, []
        try:
            info = api.model_info(H3_TE_QUANT_REPO, files_metadata = True)
        except Exception:  # noqa: BLE001 -- an unavailable artifact keeps the dense encoder
            return None, []
        files = [
            (s.rfilename, int(getattr(s, "size", 0) or 0))
            for s in (info.siblings or [])
            if s.rfilename == filename
        ]
        return (H3_TE_QUANT_REPO, files) if files else (None, [])

    def _fetch_h3_te_quant(
        self,
        scheme: Optional[str],
        hf_token: Optional[str],
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[str, ...]:
        """Pre-fetch the hosted quantized conditioner; return ``("text_encoder",)`` when it landed.

        Same contract as ``_fetch_te_prequant``: only a file already on disk earns the right to drop
        the dense shards, and the pull is cancellable and resumable like every other load download
        instead of an untracked stall inside ``from_pretrained``. A failed fetch keeps the dense
        encoder, so the load still has one to fall back to.

        A file that lands and then fails to LOAD (corrupt, or a re-upload this loader does not
        implement) is slow rather than broken: the base snapshot has no dense encoder, so
        ``load_components`` pulls it from the modular index's own repo id inline. Degrading to the
        download this skip avoided is the right failure -- the alternative is refusing the load."""
        from .video_minimax_h3_te import h3_te_quant_filename

        filename = h3_te_quant_filename(scheme)
        if filename is None:
            return ()
        cancel = cancel_event if cancel_event is not None else self._cancel_event
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        try:
            hf_hub_download_with_xet_fallback(
                H3_TE_QUANT_REPO,
                filename,
                hf_token,
                cancel_event = cancel,
                cache_dir = hub_cache_dir(),
            )
        except Exception as exc:  # noqa: BLE001 -- no artifact just means the dense encoder
            if cancel.is_set():
                raise
            logger.warning(
                "video.h3_te_quant_fetch_failed: %s/%s: %s", H3_TE_QUANT_REPO, filename, exc
            )
            return ()
        return ("text_encoder",)

    @staticmethod
    def _denoiser_prequant_covered(
        fam: Any, transformer_quant: Optional[str], base: Optional[str]
    ) -> bool:
        """True when a hosted PRE-QUANTIZED denoiser checkpoint replaces the dense DiT shards.

        The one place the plan, the byte estimate and the scoped pull all ask, so none of them can
        stage weights another one skipped. Only a modular-workflow family qualifies: every other
        family quantises its own dense transformer on device and still needs those shards.

        Never raises and never touches the network -- a pure registry lookup -- because it runs on
        the download-planning path, where a failure would cost the whole plan."""
        try:
            if not getattr(fam, "modular_workflow", None):
                return False
            scheme = normalize_transformer_quant(transformer_quant)
            # "auto" leaves the choice to the backend, which keeps the released bfloat16 components
            # for a modular workflow, so it must NOT drop the dense shards that load then wants.
            if scheme is None or scheme == TQ_AUTO:
                return False
            return video_family_prequant_repo(fam, scheme, base) is not None
        except Exception:  # noqa: BLE001 -- an unanswerable probe keeps the dense shards
            return False

    @staticmethod
    def _denoiser_prequant_hub_files(
        fam: Any, transformer_quant: Optional[str], base: Optional[str], api: Any
    ) -> tuple[Optional[str], list[tuple[str, int]]]:
        """``(repo_id, [(rfilename, size)])`` for the hosted pre-quantized denoiser, or
        ``(None, [])``.

        The denoiser mirror of ``_te_prequant_hub_files``. ``_denoiser_prequant_covered`` already
        drops the dense DiT shards from the plan, so without this the checkpoint that replaces
        them is in no entry at all: the byte total under-reports by the size of the artifact
        (20.25 GB for H3 int8), the disk preflight passes on a volume that cannot hold it, and an
        offline stage completes without the one file the load needs.

        Same failure rule as the encoder helper: an unreachable repo yields no files and is only
        logged, because the plan is a staging hint and the load still falls back to dense."""
        scheme = (transformer_quant or "").strip().lower()
        if scheme in ("", "auto", "off", "none"):
            return None, []
        try:
            from .diffusion_prequant import resolve_prequant_source
            source = resolve_prequant_source(fam, scheme, base_repo = base)
        except Exception as exc:  # noqa: BLE001 -- a bad registry entry must not sink the plan
            logger.warning("video.denoiser_prequant_unresolved: %s", exc)
            return None, []
        # A local override is already on disk; only a hosted checkpoint is staged.
        if source is None or getattr(source, "kind", None) != "repo":
            return None, []
        # The root name first, then the legacy scheme name: resolve_prequant_source hands back both
        # and the load tries them in that order, so the plan must stage whichever one exists.
        wanted = [
            n
            for n in (
                getattr(source, "filename", None),
                getattr(source, "fallback_filename", None),
            )
            if n
        ]
        try:
            info = api.model_info(source.location, files_metadata = True)
        except Exception as exc:  # noqa: BLE001 -- unavailable prequant means the dense DiT
            logger.warning("video.denoiser_prequant_unavailable: %s: %s", source.location, exc)
            return None, []
        by_name = {s.rfilename: int(s.size or 0) for s in (info.siblings or [])}
        for name in wanted:
            if name in by_name:
                return source.location, [(name, by_name[name])]
        return None, []

    @staticmethod
    def _te_prequant_hub_files(
        sources: dict[str, Any], api: Any
    ) -> dict[str, list[tuple[str, int]]]:
        """``{component: [(rfilename, size)]}`` for every hosted pre-cast checkpoint that really
        resolves on the Hub.

        Only a component listed here may have its dense weights dropped from a plan or an
        estimate: an unpublished / gated / renamed artifact keeps its dense encoder, exactly as
        the load's own fallback does. Checked per source so one missing repo cannot sink the
        whole plan."""
        found: dict[str, list[tuple[str, int]]] = {}
        for component, source in sources.items():
            # A local path override is already on disk; only a hosted checkpoint is staged.
            if getattr(source, "kind", None) != "repo" or not getattr(source, "filename", None):
                continue
            try:
                info = api.model_info(source.location, files_metadata = True)
            except Exception as exc:  # noqa: BLE001 -- unavailable pre-cast means the dense encoder
                logger.warning("video.te_prequant_unavailable: %s: %s", source.location, exc)
                continue
            files = [
                (s.rfilename, int(s.size or 0))
                for s in (info.siblings or [])
                if s.rfilename == source.filename
            ]
            if files:
                found[component] = files
        return found

    @staticmethod
    def _base_download_files(
        info: Any,
        kind: str,
        *,
        ltx23: bool = False,
        skip_te_components: tuple[str, ...] = (),
        skip_transformer_weights: bool = False,
    ) -> list[tuple[str, int]]:
        """The (rfilename, size) list a load actually needs from the base repo.

        Single source of truth for the progress estimate AND the scoped pre-download,
        so the two can never disagree. Excluded on purpose:
        - root-level packaged checkpoints (ComfyUI-style singles; 170 GB of the LTX-2
          repo) -- the diffusers pipeline only reads per-component subfolders;
        - the duplicate ``text_encoder/diffusion_pytorch_model*`` shard set (the LTX-2
          base repo ships its text encoder twice; transformers loads the ``model-*``
          naming via the shard index);
        - ``transformer/`` when a GGUF/single-file checkpoint replaces the DiT;
        - everything but scheduler / text encoder / tokenizer for an LTX-2.3
          checkpoint (``ltx23``), whose VAEs/vocoder/connectors come from the
          checkpoint and its extras, not the 2.0 base;
        - the dense weight shards of any ``skip_te_components`` encoder, supplied
          instead by a hosted pre-cast fp8 checkpoint (LTX-2's Gemma3 encoder is
          ~49 GB of the base repo). Their configs stay: the pre-cast loader
          meta-inits the encoder from the base repo's component config;
        - the dense ``transformer/`` weight shards under
          ``skip_transformer_weights``, supplied instead by a hosted PRE-QUANTIZED
          denoiser checkpoint (H3's transformer is 66.3 GB of its base repo).
          ``transformer/config.json`` is kept for the same reason the pre-cast
          encoders keep theirs: the pre-quant loader meta-inits the DiT from it,
          so dropping it would break the very load that made the skip safe."""
        from .diffusion_te_prequant import is_prequant_covered_weight

        siblings = list(info.siblings or [])
        is_h3_modular = any(s.rfilename == "modular_model_index.json" for s in siblings)
        h3_prefixes = (
            "audio_scheduler/",
            "audio_vae/",
            "processor/",
            "scheduler/",
            "text_encoder/",
            "tokenizer/",
            "transformer/",
            "vae/",
        )
        files: list[tuple[str, int]] = []
        for sibling in siblings:
            name, size = sibling.rfilename, sibling.size or 0
            if is_h3_modular and not (
                name in ("model_index.json", "modular_model_index.json")
                or name.startswith(h3_prefixes)
            ):
                continue
            # .jinja: tokenizer/chat_template.jinja is needed at generation time; a snapshot without it crashes the first generation.
            if not name.endswith((".safetensors", ".json", ".model", ".txt", ".jinja")):
                continue
            if "/" not in name and name.endswith(".safetensors"):
                continue
            if kind != "pipeline" and name.startswith("transformer/"):
                continue
            # is_prequant_covered_weight is component-agnostic (it matches "<component>/" against a
            # weight suffix), so the encoder helper serves the denoiser verbatim.
            if skip_transformer_weights and is_prequant_covered_weight(name, ("transformer",)):
                continue
            if name.startswith("text_encoder/diffusion_pytorch_model"):
                continue
            if ltx23 and "/" in name and not name.startswith(VideoBackend._LTX23_BASE_PREFIXES):
                continue
            if skip_te_components and is_prequant_covered_weight(name, skip_te_components):
                continue
            files.append((name, int(size)))
        return files

    def _estimate_download_bytes(
        self,
        repo_id: str,
        gguf_filename: Optional[str],
        base: str,
        hf_token: Optional[str],
        kind: str,
        ltx23: bool = False,
        te_sources: Optional[dict[str, Any]] = None,
        skip_transformer_weights: bool = False,
        h3_te_scheme: Optional[str] = None,
    ) -> Optional[int]:
        """Total bytes this load will pull (checkpoint + companions), or None.

        Dense encoder shards covered by an available pre-cast checkpoint are excluded, since
        the pull below skips them too, and so are the dense denoiser shards a hosted
        pre-quantized checkpoint replaces (``skip_transformer_weights``) and H3's dense
        ``text_encoder/`` under ``h3_te_scheme``. None of those hosted checkpoints' own bytes are
        added: the progress bar counts cached bytes for the checkpoint and base repos only, so a
        third repo in the total would leave the bar permanently short of 100%."""
        try:
            from huggingface_hub import HfApi

            total = 0
            api = HfApi(token = hf_token or None)
            skip_te_components = tuple(self._te_prequant_hub_files(te_sources or {}, api))
            if self._h3_te_quant_hub_files(h3_te_scheme, api)[1]:
                skip_te_components = skip_te_components + ("text_encoder",)
            if gguf_filename and not Path(repo_id).expanduser().exists():
                info = api.model_info(repo_id, files_metadata = True)
                for sibling in info.siblings or []:
                    if sibling.rfilename == gguf_filename and sibling.size:
                        total += int(sibling.size)
            if base and not Path(base).expanduser().exists():
                info = api.model_info(base, files_metadata = True)
                total += sum(
                    size
                    for _, size in self._base_download_files(
                        info,
                        kind,
                        ltx23 = ltx23,
                        skip_te_components = skip_te_components,
                        skip_transformer_weights = skip_transformer_weights,
                    )
                )
            return total or None
        except Exception:  # noqa: BLE001 -- progress totals are best-effort only
            return None

    def download_plan(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
        hf_token: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        **load_kwargs: Any,
    ) -> dict[str, Any]:
        """The repos + exact files this pick needs, for staging through the Hub download
        manager. Mirrors the image backend's plan; the file list is the same scoped one
        the load itself uses, so nothing extra is pulled. Local paths yield no entries.

        ``text_encoder_quant`` is read for the same reason the image plan reads the DiT
        quant: an fp8 request loads a hosted PRE-CAST encoder, so the base repo's dense
        encoder shards must not be staged (~49 GB of Lightricks/LTX-2) and the pre-cast
        checkpoint must be.

        ``transformer_quant`` is read for the mirror-image reason on the denoiser: a scheme with a
        hosted PRE-QUANTIZED checkpoint replaces the dense DiT, so staging those shards would cost
        66.3 GB the load never opens. It used to be swallowed by ``**load_kwargs`` and the plan
        stayed dense-sized."""
        from huggingface_hub import HfApi

        fam = _detect_load_family(repo_id, gguf_filename, family_override)
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        from .video_minimax_h3 import is_h3_native

        if is_h3_native(fam, kind):
            return self._h3_native_download_plan(repo_id, gguf_filename or "", hf_token = hf_token)
        base = repo_id if kind == "pipeline" else resolve_video_base_repo(fam, base_repo)
        # Only the header tells an LTX-2.3 checkpoint from 2.0 and it is not on disk yet, so narrow the base pull by NAME: a wrong guess costs an inline pull, the wide base list costs gigabytes.
        ltx23 = self._pick_looks_like_ltx23(fam, repo_id, gguf_filename, kind)
        # Keyed by repo so a 2.3 pick's checkpoint and extras stay ONE scoped job; two entries would collide on the job key.
        entries: dict[str, dict[str, Any]] = {}
        total = 0

        def add(
            repo: str,
            files: list[tuple[str, int]],
            gguf: Optional[str] = None,
        ) -> int:
            if not files:
                return 0
            entry = entries.setdefault(
                repo, {"repo_id": repo, "files": [], "bytes": 0, "gguf_filename": None}
            )
            seen = set(entry["files"])
            added = 0
            for name, size in files:
                if name in seen:
                    continue
                seen.add(name)
                entry["files"].append(name)
                entry["bytes"] += int(size)
                added += int(size)
            if gguf:
                entry["gguf_filename"] = gguf
            return added

        try:
            api = HfApi(token = hf_token or None)
            if gguf_filename and not Path(repo_id).expanduser().exists():
                info = api.model_info(repo_id, files_metadata = True)
                sizes = [
                    (s.rfilename, int(s.size or 0))
                    for s in (info.siblings or [])
                    if s.rfilename == gguf_filename
                ]
                total += add(repo_id, sizes, gguf = gguf_filename)
                if ltx23:
                    # The 2.3 assembly reads these companion files at load; without them here they would be pulled inline, outside the panel's disk preflight.
                    from .video_ltx2 import ltx23_extras_files, LTX23_EXTRAS_REPO

                    wanted = set(ltx23_extras_files(gguf_filename))
                    extras_info = (
                        info
                        if LTX23_EXTRAS_REPO == repo_id
                        else api.model_info(LTX23_EXTRAS_REPO, files_metadata = True)
                    )
                    total += add(
                        LTX23_EXTRAS_REPO,
                        [
                            (s.rfilename, int(s.size or 0))
                            for s in (extras_info.siblings or [])
                            if s.rfilename in wanted
                        ],
                    )
            # Pre-cast encoders first: only a checkpoint that really resolves earns the right to drop the dense shards.
            te_sources = self._te_prequant_sources(fam, text_encoder_quant)
            te_files = self._te_prequant_hub_files(te_sources, api)
            for component, files in te_files.items():
                total += add(te_sources[component].location, files)
            # The denoiser's replacement artifact, for the same reason: the base entry below drops
            # the dense DiT shards whenever this resolves, so it has to be staged in their place.
            dq_repo, dq_files = self._denoiser_prequant_hub_files(fam, transformer_quant, base, api)
            if dq_repo:
                total += add(dq_repo, dq_files)
            # And H3's quantized conditioner, which replaces the base repo's dense text_encoder/.
            h3_te_repo, h3_te_files = self._h3_te_quant_hub_files(
                self._h3_te_quant_scheme(fam, text_encoder_quant, base), api
            )
            if h3_te_repo:
                total += add(h3_te_repo, h3_te_files)
            if base and not Path(base).expanduser().exists():
                info = api.model_info(base, files_metadata = True)
                total += add(
                    base,
                    self._base_download_files(
                        info,
                        kind,
                        ltx23 = ltx23,
                        skip_te_components = (
                            tuple(te_files) + (("text_encoder",) if h3_te_repo else ())
                        ),
                        skip_transformer_weights = self._denoiser_prequant_covered(
                            fam, transformer_quant, base
                        ),
                    ),
                )
        except Exception as exc:  # noqa: BLE001 -- an unavailable plan falls back to the inline pull
            logger.warning("video.download_plan_failed: %s", exc)
            return {"entries": [], "total_bytes": 0}
        return {"entries": list(entries.values()), "total_bytes": total}

    @staticmethod
    def _h3_native_download_plan(
        repo_id: str, gguf_filename: str, *, hf_token: Optional[str]
    ) -> dict[str, Any]:
        from huggingface_hub import HfApi

        from .video_minimax_h3 import (
            H3_AUDIO_VAE,
            H3_COMPONENT_REPO,
            H3_GGUF_REPO,
            H3_VIDEO_VAE,
            h3_text_encoder_filename,
            validate_h3_transformer_filename,
        )

        validate_h3_transformer_filename(gguf_filename)
        wanted = (
            (repo_id, gguf_filename),
            (H3_GGUF_REPO, h3_text_encoder_filename(gguf_filename)),
            (H3_COMPONENT_REPO, H3_VIDEO_VAE),
            (H3_COMPONENT_REPO, H3_AUDIO_VAE),
        )
        grouped: dict[str, dict[str, Any]] = {}
        total = 0
        try:
            api = HfApi(token = hf_token or None)
            for repo, filename in wanted:
                if Path(repo).expanduser().exists():
                    continue
                info = api.model_info(repo, files_metadata = True)
                match = next((s for s in (info.siblings or []) if s.rfilename == filename), None)
                if match is None:
                    raise ValueError(f"Required MiniMax-H3 component is missing: {repo}/{filename}")
                size = int(match.size or 0)
                entry = grouped.setdefault(
                    repo,
                    {"repo_id": repo, "files": [], "bytes": 0, "gguf_filename": None},
                )
                if filename not in entry["files"]:
                    entry["files"].append(filename)
                    entry["bytes"] += size
                    total += size
                if filename == gguf_filename:
                    entry["gguf_filename"] = filename
        except Exception as exc:  # noqa: BLE001 -- inline loading remains the fallback
            logger.warning("video.h3_native_download_plan_failed: %s", exc)
            return {"entries": [], "total_bytes": 0}
        return {"entries": list(grouped.values()), "total_bytes": total}

    @staticmethod
    def _pick_looks_like_ltx23(
        fam: Any, repo_id: str, gguf_filename: Optional[str], kind: str
    ) -> bool:
        """Whether a pick is an LTX-2.3 checkpoint, judged by name alone.

        The load decides this from the checkpoint header (authoritative), which the plan cannot
        read before downloading. Only the file list differs, and under-guessing merely falls back
        to the load-time pull, so a name match is the right trade here."""
        if kind == "pipeline" or fam is None or getattr(fam, "name", None) != "ltx-2":
            return False
        from .video_ltx2 import LTX23_EXTRAS_REPO

        if repo_id.strip().lower() == LTX23_EXTRAS_REPO.lower():
            return True
        text = f"{repo_id} {gguf_filename or ''}".lower()
        return "2.3" in text or "2_3" in text or "23b" in text

    def _fetch_te_prequant(
        self,
        sources: dict[str, Any],
        hf_token: Optional[str],
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[str, ...]:
        """Pre-fetch the hosted pre-cast encoder checkpoints; return the components whose
        dense weights the base pull can therefore skip.

        Downloading here rather than leaving it to the injection inside ``from_pretrained``
        is what makes that skip safe: only a checkpoint already on disk earns the right to
        drop the dense shards, and the pull becomes cancellable and resumable like every
        other load download instead of an untracked stall. A component whose fetch fails
        keeps its dense weights, so the load still has an encoder to fall back to."""
        cancel = cancel_event if cancel_event is not None else self._cancel_event
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        fetched: list[str] = []
        for component, source in sources.items():
            # A local path override is validated by the injection itself; nothing to fetch, and no basis for skipping the dense download.
            if getattr(source, "kind", None) != "repo" or not getattr(source, "filename", None):
                continue
            try:
                hf_hub_download_with_xet_fallback(
                    source.location,
                    source.filename,
                    hf_token,
                    cancel_event = cancel,
                )
            except Exception as exc:  # noqa: BLE001 -- no pre-cast file just means the dense encoder
                if cancel.is_set():
                    raise
                logger.warning(
                    "video.te_prequant_fetch_failed: %s/%s: %s",
                    source.location,
                    source.filename,
                    exc,
                )
                continue
            fetched.append(component)
        return tuple(fetched)

    def _predownload_base(
        self,
        base: str,
        hf_token: Optional[str],
        kind: str,
        *,
        ltx23: bool = False,
        skip_te_components: tuple[str, ...] = (),
        skip_transformer_weights: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[str]:
        """Pull exactly the base-repo files the load needs; return the local snapshot dir.

        A bare ``from_pretrained(repo_id)`` snapshot of Lightricks/LTX-2 downloads the
        whole 314 GB repo (root packaged checkpoints plus a second 50 GB text-encoder
        shard set) when ~93 GB is used. Downloading the scoped file list ourselves is
        also cancellable per file, and handing the local dir to from_pretrained skips
        diffusers' own expected-files sweep. None -> caller keeps the hub id (local
        path, non-diffusers layout, or any metadata failure: from_pretrained then
        resolves the repo exactly as before)."""
        cancel = cancel_event if cancel_event is not None else self._cancel_event
        try:
            if not base or Path(base).expanduser().exists():
                return None
            from huggingface_hub import HfApi

            info = HfApi(token = hf_token or None).model_info(base, files_metadata = True)
            files = self._base_download_files(
                info,
                kind,
                ltx23 = ltx23,
                skip_te_components = skip_te_components,
                skip_transformer_weights = skip_transformer_weights,
            )
            if not any(name == "model_index.json" for name, _ in files):
                return None
            from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

            snapshot_root: Optional[Path] = None
            for name, _ in files:
                # Explicit check: a cached file returns without consulting the event, so a warm-cache sweep would run to completion after a cancel.
                if cancel.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)
                local = Path(
                    hf_hub_download_with_xet_fallback(base, name, hf_token, cancel_event = cancel)
                )
                if name == "model_index.json":
                    snapshot_root = local.parent
            return str(snapshot_root) if snapshot_root is not None else None
        except Exception as exc:  # noqa: BLE001 -- fall back to from_pretrained's own pull
            if cancel.is_set():
                raise
            logger.warning("video.predownload_fallback: %s", exc)
            return None

    def _cache_bytes(self, repo_id: Optional[str]) -> int:
        """Bytes of ``repo_id`` currently in the HF blob cache (progress polling).

        Walks the repo's cache directory directly instead of ``scan_cache_dir``:
        the scanner skips in-flight ``*.incomplete`` blobs, so during a multi-GB
        shard pull the counter would freeze at the last completed blob for minutes
        while the disk keeps filling (the bar sat stuck mid-download)."""
        if not repo_id:
            return 0
        try:
            import os

            folder = Path(hub_cache_dir()) / ("models--" + repo_id.strip().replace("/", "--"))
            if not folder.is_dir():
                return 0
            total = 0
            for root, _dirs, files in os.walk(folder):
                for name in files:
                    try:
                        path = os.path.join(root, name)
                        # Snapshot entries are symlinks into blobs/; skip them so a blob is not counted twice.
                        if not os.path.islink(path):
                            total += os.path.getsize(path)
                    except OSError:
                        continue
            return int(total)
        except Exception:  # noqa: BLE001 -- cache scan is best-effort
            return 0

    def load_progress(self) -> dict[str, Any]:
        """Phase + downloaded/total bytes for the in-flight load (cache-scan based)."""
        loading = self._loading
        if loading is not None and loading.error:
            return _progress("error", error = loading.error)
        if loading is None:
            return _progress("ready" if self._state is not None else None)
        downloaded = self._cache_bytes(loading.repo_id)
        if loading.base_repo and loading.base_repo != loading.repo_id:
            downloaded += self._cache_bytes(loading.base_repo)
        expected = loading.expected_bytes
        phase = "downloading"
        if not expected:
            # No size estimate (metadata failure, or a fully cached load). ``downloaded`` counts what is PRESENT in the cache,
            # not what this load fetched, so it would show a multi-GB figure for a load that fetches nothing. Report the phase alone.
            return _progress(phase)
        if downloaded >= expected:
            phase = "finalizing"
            # The cache scan counts every blob, so the raw counter can exceed the scoped estimate; clamp to what the bar reports.
            downloaded = expected
        return _progress(
            phase,
            downloaded_bytes = int(downloaded),
            expected_bytes = int(expected) if expected else None,
        )

    def loading_repo_ids(self) -> tuple[str, ...]:
        """Repo ids an in-flight background load is downloading (empty when idle).

        The delete-cached guard needs this: during a load ``status()["loaded"]`` is
        still False, but deleting the target repo (or its companion base, or one of the
        native H3 companion repos an H3 load also downloads from) would yank blobs and
        snapshot files from under the download/assembly. The in-flight twin of
        loaded_repo_ids(); mirrors the image backend's guard
        (DiffusionBackend.loading_repo_ids)."""
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            ids = (loading.repo_id, loading.base_repo, *loading.asset_repos)
            return tuple(r for r in ids if r)

    def loaded_repo_ids(self) -> tuple[str, ...]:
        """Repo ids the COMMITTED model reads from disk (empty unless a native model is loaded).

        The one-shot sd-cli re-reads H3's Qwen encoder and both VAEs from the cache on every
        generation, and those live in companion repos that are neither ``repo_id`` nor the BF16
        ``base_repo`` ``status()`` publishes, so the delete-cached guard must refuse them too.
        Mirrors the image backend's guard (SdCppBackend.loaded_repo_ids)."""
        with self._lock:
            state = self._state
            if state is None or state.engine != "sd_cpp":
                return ()
            repo_id = state.repo_id
        from .video_minimax_h3 import H3_COMPONENT_REPO, H3_GGUF_REPO

        return (repo_id, H3_GGUF_REPO, H3_COMPONENT_REPO)

    # ── the load itself ──────────────────────────────────────────────────────

    def load_pipeline(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        model_kind: Optional[str] = None,
        h3_task: Optional[str] = None,
        _load_token: Optional[int] = None,
        _base_local_dir: Optional[str] = None,
        _te_prequant_skipped: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            base_repo = base_repo,
            family_override = family_override,
            model_kind = model_kind,
            transformer_quant = transformer_quant,
            text_encoder_quant = text_encoder_quant,
            h3_task = h3_task,
        )
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        from .video_minimax_h3 import is_h3_native

        if is_h3_native(fam, kind):
            self._run_load_h3_native(
                fam = fam,
                token = _load_token,
                cancel_event = threading.Event(),
                repo_id = repo_id,
                gguf_filename = gguf_filename,
                hf_token = hf_token,
                memory_mode = memory_mode,
            )
            return self.status()

        import diffusers
        import torch

        base = repo_id if kind == "pipeline" else resolve_video_base_repo(fam, base_repo)

        with self._lock:
            if _load_token is not None and _load_token != self._load_token:
                raise RuntimeError("Video load was cancelled or superseded.")
            # Signal a generation from the PREVIOUS model (the token check above bailed a superseded worker).
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            # Same fence unload() takes, raised BEFORE the barrier: a queued generation holds no cancel event, so the signal
            # above cannot reach it and it would slip through the moment the barrier released _generate_lock.
            self._teardown_waiters += 1
        # Barrier: wait for the signalled generation to exit before teardown, or two models coexist in VRAM.
        with self._generate_lock:
            with self._lock:
                try:
                    # The barrier wait can outlive this load (superseded by a newer load/unload); recheck before touching shared state.
                    if _load_token is not None and _load_token != self._load_token:
                        raise RuntimeError("Video load was cancelled or superseded.")
                    self._teardown_state_locked()
                finally:
                    # Released here, not at the end of the load: the old pipe is gone (or this load bailed), and a raising teardown must not leave the fence up for the life of the process.
                    self._teardown_waiters -= 1

        target = resolve_diffusion_device_target()
        device = target.device
        # Video DiTs are bf16-native; fp16 overflows, so a resolved fp16 promotes to float32. CPU stays float32.
        dtype = target.dtype
        if fam.fp16_incompatible and dtype is torch.float16:
            dtype = torch.float32
        # Size tables below are bf16 (2-byte), so scale dense estimates when the promotion lands fp32 on an accelerator.
        dtype_scale = 2.0 if device != "cpu" and dtype is torch.float32 else 1.0

        if fam.modular_workflow:
            return self._load_h3_modular_pipeline(
                diffusers = diffusers,
                torch = torch,
                fam = fam,
                repo_id = repo_id,
                base = base,
                kind = kind,
                dtype = dtype,
                device = device,
                hf_token = hf_token,
                memory_mode = memory_mode,
                # transformer_quant is only normalised BELOW this dispatch, and the modular branch
                # returns before reaching that, so normalise here with the same function rather
                # than moving the dispatch past a block written for the conventional path (its
                # speed_mode/auto-ladder defaulting means nothing to a modular workflow).
                transformer_quant = normalize_transformer_quant(transformer_quant),
                # Same reason as transformer_quant above: the conventional path normalises the
                # encoder request below this dispatch, which the modular branch never reaches.
                text_encoder_quant = normalize_te_quant(text_encoder_quant),
                h3_task = h3_task,
                _load_token = _load_token,
                _base_local_dir = _base_local_dir,
            )

        # What the CALLER asked for, before the rewrite below. The record has to report this, not
        # the internal value: an omitted precision under speed_mode="off" becomes "off" here, and
        # reporting that as the request makes the resolved record say the user pinned bf16. The
        # frontend then hides the Auto badge and reseeds the Precision select to none, so after
        # the user changes Speed and reloads, quantisation stays pinned off for no reason they
        # can see.
        transformer_quant_requested = transformer_quant
        # Precision tri-state: unset/"auto" -> hardware ladder; "none"/"off" pins dense bf16; an explicit scheme pins it. Pipeline-kind only.
        if transformer_quant is None or str(transformer_quant).strip().lower() in (
            "",
            "auto",
        ):
            # An explicit Speed="off" (bit-exact) load must stay dense bf16: auto-quant would engage int8/fp8 + compile and break the request.
            speed_off = speed_mode is not None and str(speed_mode).strip().lower() == SPEED_OFF
            transformer_quant = "off" if speed_off else TQ_AUTO

        # ── memory plan: family-table resident estimate + frames-aware headroom.
        device_memory = snapshot_device_memory(target)
        components = fam.bf16_components_gb
        mib_per_gb = 1000.0**3 / (1024.0 * 1024.0)
        # bf16_components_gb is (transformer, text_encoder, vae) at bf16. When this pick takes its
        # encoder PRE-CAST from a hosted fp8 checkpoint, the bf16 entry over-states what is loaded
        # by ~11 GB for ltx-2's Gemma3-12B, which drags every budget below with it. The VAE is
        # never quantised (quantize_text_encoders touches text encoders only, and vae_force_fp32
        # families widen it), so only components[1] is scaled.
        from .diffusion_te_prequant import te_prequant_budget_scale

        te_scale = te_prequant_budget_scale(fam, te_quant_mode = text_encoder_quant, target = target)
        transformer_mib: Optional[int] = None
        if kind != "pipeline":
            checkpoint_path = self._resolve_checkpoint_path(repo_id, gguf_filename, hf_token)
            size_mib = file_size_mib(str(checkpoint_path))
            if kind == "gguf":
                transformer_mib = estimate_gguf_resident_mib(size_mib)
            else:
                transformer_mib = estimate_safetensors_dense_mib(size_mib)
                if transformer_mib is not None:
                    transformer_mib = int(transformer_mib * dtype_scale)
        runtime_mib = estimate_video_runtime_mib(
            width = fam.resolution_presets[0][0],
            height = fam.resolution_presets[0][1],
            num_frames = fam.default_num_frames,
        )

        def _plan_for_te_scale(scale: float, *, log: bool) -> tuple[Any, Any, bool]:
            """``(plan, bf16_plan, quant_replanned)`` for a text-encoder budget of ``scale`` x
            its bf16 size. Pure and cheap (``plan_diffusion_memory`` is arithmetic), so the
            dense-encoder plan can be rebuilt below if the pre-cast injection does not land."""
            text_encoder_gb = components[1] * scale if components is not None else 0.0
            companions_gb = text_encoder_gb + components[2] if components is not None else 0.0
            if log and scale != 1.0 and components is not None:
                logger.info(
                    "video.te_prequant_budget: budgeting the pre-cast %s text encoder at %.2f of "
                    "bf16 (%.1f GB instead of %.1f GB)",
                    text_encoder_quant,
                    scale,
                    text_encoder_gb,
                    components[1],
                )
            if kind == "pipeline":
                model_dense_mib = (
                    int((components[0] + companions_gb) * mib_per_gb * dtype_scale)
                    if components is not None
                    else None
                )
                companion_mib = None
            else:
                companion_mib = (
                    int(companions_gb * mib_per_gb * dtype_scale)
                    if components is not None
                    else None
                )
                # Budget ALL weights: companions stay resident, so budgeting the transformer alone lets auto pick OFFLOAD_NONE and OOM.
                model_dense_mib = (
                    transformer_mib + (companion_mib or 0) if transformer_mib is not None else None
                )
            planned = plan_diffusion_memory(
                target = target,
                device_memory = device_memory,
                model_dense_mib = model_dense_mib,
                runtime_headroom_mib = runtime_mib,
                companion_dense_mib = companion_mib,
                requested_mode = normalize_memory_mode(memory_mode),
            )
            # Parity with the image dense-quant path: the bf16-table plan can force offload a quantised DiT would not need, so re-plan with the scheme factor and keep resident if it fits.
            dense_plan = planned
            replanned_for_quant = False
            if (
                kind == "pipeline"
                and planned.offload_policy != "none"
                and normalize_transformer_quant(transformer_quant) is not None
                and dense_transformer_supported(target)
                and components is not None
            ):
                scheme_preview = select_transformer_quant_scheme(
                    target, transformer_quant, family = fam.name
                )
                factor = _QUANT_STEADY_FACTOR.get(scheme_preview) if scheme_preview else None
                if factor is not None:
                    quant_mib = int((components[0] * factor + companions_gb) * mib_per_gb)
                    replanned = plan_diffusion_memory(
                        target = target,
                        device_memory = device_memory,
                        model_dense_mib = quant_mib,
                        runtime_headroom_mib = runtime_mib,
                        companion_dense_mib = None,
                        requested_mode = normalize_memory_mode(memory_mode),
                    )
                    if replanned.offload_policy == "none":
                        if log:
                            logger.info(
                                "video.transformer_quant: %s fits resident (%d MiB steady); "
                                "dropping the bf16 plan's '%s' offload",
                                scheme_preview,
                                quant_mib,
                                planned.offload_policy,
                            )
                        planned = replanned
                        replanned_for_quant = True
            return planned, dense_plan, replanned_for_quant

        plan, bf16_plan, quant_replanned = _plan_for_te_scale(te_scale, log = True)

        # ── build the pipeline.
        pipeline_cls = getattr(diffusers, fam.pipeline_class)
        # cache_dir pins every loader call to the live cache root, so a mid-session change cannot split one model across roots.
        pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype, "cache_dir": hub_cache_dir()}
        if getattr(fam, "vae_force_fp32", False):
            # Wan VAE must decode in float32: a scalar torch_dtype truncates its fp32 weights to bf16 and a later widen only
            # restores lossy values (banding / black frames). "default" MUST be set or unlisted components fall back to fp32.
            pipe_kwargs["torch_dtype"] = {"vae": torch.float32, "default": dtype}
        if hf_token:
            pipe_kwargs["token"] = hf_token
        # A hosted pre-cast fp8 text encoder skips the dense TE download (for LTX Gemma3-27B, ~50 GB). The cast re-applies idempotently.
        from .diffusion_te_prequant import te_prequant_pipe_kwargs

        te_injected = te_prequant_pipe_kwargs(
            fam,
            repo_id if kind == "pipeline" else base,
            te_quant_mode = text_encoder_quant,
            target = target,
            dtype = dtype,
            hf_token = hf_token,
            logger = logger,
        )
        pipe_kwargs.update(te_injected)
        # The fp8-sized budget is valid only once the pre-cast encoder is actually in hand. Injection is best-effort
        # (an unreachable / corrupt / base-mismatched checkpoint falls back to the dense encoder), and a dense encoder
        # under an fp8 budget under-states the resident requirement by ~11 GB for ltx-2, which is the direction that
        # OOMs. Placement is applied post-assembly (apply_memory_plan below), so re-plan at bf16 here, exactly as the
        # dense-quant fallback below restores bf16_plan.
        if te_scale != 1.0 and not te_injected:
            logger.warning(
                "video.te_prequant_budget: no pre-cast encoder was injected; re-planning "
                "memory at the dense bf16 text-encoder size"
            )
            plan, bf16_plan, quant_replanned = _plan_for_te_scale(1.0, log = False)
        # Injection is best-effort (a missing checkpoint falls back to the dense encoder), but the scoped pre-download already dropped those
        # shards and from_pretrained cannot re-fetch from a local snapshot dir, so top them up here. ltx23 never reaches this: it gets the hub id.
        missing_dense = [c for c in (_te_prequant_skipped or ()) if c not in pipe_kwargs]
        if missing_dense and _base_local_dir:
            logger.warning(
                "video.te_prequant: %s not injected; restoring the dense weights the "
                "pre-download skipped",
                ", ".join(missing_dense),
            )
            _base_local_dir = (
                self._predownload_base(base, hf_token, kind, ltx23 = False) or _base_local_dir
            )
        if kind == "pipeline":
            # The pre-downloaded snapshot dir keeps from_pretrained off the hub; hub id when pre-download was skipped.
            pipe = pipeline_cls.from_pretrained(_base_local_dir or repo_id, **pipe_kwargs)
        else:
            transformer_cls = getattr(diffusers, fam.transformer_class)
            # checkpoint_path was resolved (and downloaded) by the memory-planning branch above.
            sf_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "config": base,
                "subfolder": "transformer",
                "token": hf_token,
                "cache_dir": hub_cache_dir(),
            }
            if kind == "gguf":
                sf_kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(
                    compute_dtype = dtype
                )
            from .video_ltx2 import is_ltx23_checkpoint, load_ltx23_pipeline

            if fam.name == "ltx-2" and is_ltx23_checkpoint(checkpoint_path):
                # 2.3 checkpoints need the full assembly: new config flags, key renames the stock converter lacks, and the 2.3 connectors/VAEs/vocoder.
                pipe = load_ltx23_pipeline(
                    checkpoint_path,
                    base_repo = base,
                    torch_dtype = dtype,
                    is_gguf = kind == "gguf",
                    hf_token = hf_token,
                    # The 2.3 assembly builds every component itself and never sees pipe_kwargs, so hand the pre-cast encoder over explicitly.
                    text_encoder = pipe_kwargs.get("text_encoder"),
                )
            else:
                transformer = transformer_cls.from_single_file(str(checkpoint_path), **sf_kwargs)
                pipe = pipeline_cls.from_pretrained(
                    _base_local_dir or base, transformer = transformer, **pipe_kwargs
                )

        # The dtype dict already loads the Wan VAE at float32; belt-and-suspenders for any path that bypassed it (e.g. a passed-in vae=).
        if getattr(fam, "vae_force_fp32", False):
            vae = getattr(pipe, "vae", None)
            if vae is not None and getattr(vae, "dtype", None) is not torch.float32:
                vae.to(torch.float32)

        if _load_token is not None and _load_token != self._load_token:
            del pipe
            clear_gpu_cache()
            raise RuntimeError("Video load was cancelled or superseded.")

        # For a dual-DiT MoE every optimisation site below covers BOTH experts: ``views`` is (pipe, _SecondDiTView(pipe)); single-DiT resolves to (pipe,).
        views = _views_for(pipe, fam)

        # dense transformer quant (opt-in, pipeline-kind only): torchao-quantise the dense bf16 DiT in place onto the
        # low-precision tensor cores. CUDA + bf16 only, best-effort. Quant must precede compile (eager is ~30x slower).
        transformer_quant_engaged: Optional[str] = None
        quant_skipped_for_offload = False
        # An EXPLICIT scheme (not auto, not off) is a contract, so a decline below must be reported
        # and -- unless the escape hatch is set -- must stop the load rather than quietly running
        # the DiT at bf16 while the Precision dropdown still reads FP8.
        transformer_quant_pinned = normalize_transformer_quant(transformer_quant)
        if transformer_quant_pinned == TQ_AUTO:
            transformer_quant_pinned = None
        # Why the quant did not engage, in the caller's terms; threaded into `resolved`.
        transformer_quant_decline: Optional[str] = None
        transformer_quant_decline_status = RESOLVED_FELL_BACK
        if transformer_quant_pinned is not None and kind != "pipeline":
            transformer_quant_decline = (
                f"the dense DiT quant applies to full-pipeline loads only, and this is a "
                f"'{kind}' load, which runs the precision its checkpoint carries"
            )
            transformer_quant_decline_status = RESOLVED_UNSUPPORTED
        elif transformer_quant_pinned is not None and not dense_transformer_supported(target):
            transformer_quant_decline = (
                "this device cannot run a dense torchao quant (it needs a CUDA GPU in bf16)"
            )
            transformer_quant_decline_status = RESOLVED_UNSUPPORTED
        if (
            kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
            and plan.offload_policy != "none"
        ):
            # Offload hooks move modules with Module.to(), which torchao quantized tensors reject. Skip quant (dense-under-offload beats a crash); forceable via a resident mode.
            logger.info(
                "video.transformer_quant: skipped (offload policy '%s' moves the "
                "DiT via Module.to(), unsupported for torchao quantized tensors); "
                "pin a resident memory mode to combine quant with this model",
                plan.offload_policy,
            )
            quant_skipped_for_offload = True
            transformer_quant_decline = (
                f"the memory plan picked '{plan.offload_policy}' offload, which moves the DiT via "
                "Module.to(); torchao quantized tensors reject that. Pin a resident memory mode "
                "(Fast) to combine quant with this model"
            )
        elif (
            kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
        ):
            engaged = []
            for view in views:
                # Pass each expert view so both DiTs quantise with the same scheme. The family name drives _FAMILY_SCHEME_DENY.
                scheme = quantize_transformer(
                    view,
                    target,
                    mode = transformer_quant,
                    family = fam.name,
                    logger = logger,
                )
                if scheme is not None:
                    engaged.append(scheme)
            # All experts or none: the first is mutated in place, so a second-expert failure cannot fall back to dense.
            if engaged and len(engaged) < len(views):
                del pipe
                clear_gpu_cache()
                raise RuntimeError(
                    f"transformer_quant={engaged[0]} engaged on only "
                    f"{len(engaged)}/{len(views)} experts; retry without quant."
                )
            if engaged:
                transformer_quant_engaged = engaged[0]
            elif transformer_quant_pinned is not None:
                transformer_quant_decline = (
                    f"'{transformer_quant_pinned}' did not engage on family '{fam.name}' with this "
                    "GPU (the scheme is unsupported here, or the family's measured deny list "
                    "rules it out); see the server log"
                )
                transformer_quant_decline_status = RESOLVED_UNSUPPORTED
        # The quant-sized plan is valid only when quant engaged; a dense fallback keeps bf16 placement.
        if quant_replanned and transformer_quant_engaged is None:
            plan = bf16_plan

        # Fail closed on a declined EXPLICIT precision, mirroring the image loader: a clip rendered
        # at bf16 under a Precision dropdown still reading FP8 is not evidence the request ran.
        if (
            transformer_quant_engaged is None
            and transformer_quant_pinned is not None
            and not precision_fallback_allowed()
        ):
            del pipe
            clear_gpu_cache()
            raise RuntimeError(
                precision_refusal_message(
                    "transformer_quant",
                    transformer_quant_pinned,
                    transformer_quant_decline
                    or "the quantised DiT build did not engage on this host",
                    off_label = "Off to run the DiT at bf16",
                )
            )

        # dense text-encoder quant (opt-in): the companion encoder is often the largest resident, so quantise it in place before placement, for every kind. Best-effort.
        text_encoder_outcome = quantize_text_encoders(
            pipe,
            target,
            mode = text_encoder_quant,
            family = fam.name,
            offload_active = plan.offload_policy != "none",
            logger = logger,
        )
        text_encoder_quant_engaged = text_encoder_outcome.mode
        # Same contract for the other half of "the requested precision": an explicit encoder mode
        # that engaged NOTHING leaves a dense bf16 encoder the caller did not ask for, and a
        # PARTIAL cast leaves conditioning split across quantised and dense towers. A mode that
        # engaged something ELSE (int8 -> fp8) is reported by the badge instead: the encoder is
        # quantised, just not the way asked, and the reason says so.
        if (
            (text_encoder_quant_engaged is None or text_encoder_outcome.partial)
            and normalize_te_quant(text_encoder_quant) is not None
            and not precision_fallback_allowed()
        ):
            del pipe
            clear_gpu_cache()
            raise RuntimeError(
                precision_refusal_message(
                    "text_encoder_quant",
                    normalize_te_quant(text_encoder_quant) or "",
                    text_encoder_outcome.reason or "no text encoder could be cast",
                    off_label = "leave it unset to keep the dense bf16 encoder",
                    auto_available = False,
                )
            )

        # optimisation layers in the image backend order: step cache FIRST (compile keys fullgraph off an active cache),
        # then attention, speed, placement. A clip denoise runs minutes, so even a dense load amortises the compile.
        effective_speed = resolve_speed_mode(
            speed_mode, is_gguf = kind == "gguf", dense_default = SPEED_DEFAULT
        )
        # A torchao-quantised DiT must be compiled (eager is ~30x slower), so force the regional profile when quant engaged but speed was off.
        if transformer_quant_engaged is not None and effective_speed == SPEED_OFF:
            logger.info(
                "video.transformer_quant: forcing speed_mode=default "
                "(quantized transformer must be compiled; eager is ~30x slower)"
            )
            effective_speed = SPEED_DEFAULT
        backend_flags = snapshot_backend_flags()
        # Until the state commit hands ownership to _teardown_state_locked, a failure must restore these globals itself. Registered BEFORE the first mutation.
        self._precommit_globals = (_load_token, backend_flags)
        # Step cache tri-state: unset/"auto" -> FBCACHE_MIN_STEPS policy (re-checked per generation); "off"/"fbcache" pinned. Run per expert.
        cache_request = normalize_transformer_cache(transformer_cache)
        cache_auto = transformer_cache is None or cache_request == TC_AUTO
        # GGUF and torchao-quantised DiTs need the higher threshold to trigger over quant noise.
        cache_quant_active = kind == "gguf" or transformer_quant_engaged is not None
        default_cache_steps: Optional[int] = None
        if cache_auto:
            default_cache_steps, _ = default_video_generation_params(gguf_filename, repo_id, base)
            cache_request = TC_FBCACHE if default_cache_steps >= FBCACHE_MIN_STEPS else None
        cache_engaged = None
        for view in views:
            engaged = apply_step_cache(
                view,
                mode = cache_request,
                threshold = transformer_cache_threshold,
                # A quantized transformer's residuals are larger, so both engaged quant and GGUF need the higher FBCache threshold.
                quant_active = cache_quant_active,
                logger = logger,
            )
            if view is pipe:
                cache_engaged = engaged
        # The auto decision can flip at generation time, but only on a cache-capable DiT.
        cache_may_toggle = cache_auto and callable(
            getattr(getattr(pipe, "transformer", None), "enable_cache", None)
        )
        if cache_auto:
            if cache_engaged:
                cache_reason = (
                    f"auto: {default_cache_steps}-step default schedule reaches "
                    f"{FBCACHE_MIN_STEPS}; re-checked per generation"
                )
            elif cache_request is not None:
                cache_reason = "auto: model does not support step caching"
            else:
                cache_reason = (
                    f"auto: {default_cache_steps}-step default schedule is below "
                    f"{FBCACHE_MIN_STEPS}; re-checked per generation"
                )
        else:
            cache_reason = "requested"
        attention_engaged = None
        # HunyuanVideo-1.5 only, and once for the whole pipe (the installer fans out over every
        # denoiser DiT itself). Before apply_attention_backend below, so the requested kernel pins
        # onto the new processors. Held off on SPEED_OFF (which must stay bit-identical) and on
        # SPEED_MAX (its blocks compile with dynamic=False, and the trimmed text length varies per
        # prompt, so every prompt would be a fresh graph).
        attention_trim_engaged = (
            install_hunyuan_attention_trim(pipe, fam, logger = logger)
            if effective_speed not in (SPEED_OFF, SPEED_MAX)
            else False
        )
        speed_optims: tuple = ()
        for view in views:
            # Both helpers act on ``view.transformer``; call once per view (engaged values match, so record the first). is_gguf needs kind==gguf AND no quant engaged.
            gguf_transformer = kind == "gguf" and transformer_quant_engaged is None
            engaged = apply_attention_backend(
                view,
                select_attention_backend(
                    target, attention_backend, speed_active = effective_speed != SPEED_OFF
                ),
                logger = logger,
                # The probe behind this must see the dtype the pipeline actually RUNS in. Every
                # video family promotes a resolved fp16 to fp32 above, and the fused SDPA kernels
                # are half-precision only, so probing the pre-promotion fp16 would report a healthy
                # device for a generation that is in fact dispatching to the quadratic math backend.
                target = types.SimpleNamespace(device = target.device, dtype = dtype),
            )
            applied = apply_speed_optims(
                view,
                target,
                is_gguf = gguf_transformer,
                family = fam,
                speed_mode = effective_speed,
                # An auto cache that could still engage also drops fullgraph (FBCache under a fullgraph-compiled DiT crashes).
                cache_active = cache_engaged is not None or cache_may_toggle,
                offload_active = plan.offload_policy != "none",
            )
            if view is pipe:
                attention_engaged = engaged
                speed_optims = tuple(k for k, v in applied.items() if v)
                if attention_trim_engaged:
                    speed_optims += ("hunyuan_attn_trim",)
        with self._generate_lock:
            # A cancelled/superseded load must not place weights on a GPU the arbiter may have reassigned; recheck before placement.
            if _load_token is not None and _load_token != self._load_token:
                del pipe
                clear_gpu_cache()
                raise RuntimeError("Video load was cancelled or superseded.")
            offload_policy, vae_tiling = apply_memory_plan(pipe, plan, device = device, logger = logger)
            # A dual-DiT MoE needs no extra per-expert pass: apply_memory_plan covers every DiT; a second pass would duplicate-hook.
            if not vae_tiling:
                # Whole-clip decode is the video memory peak; tiling is near-free, so always on.
                try:
                    pipe.vae.enable_tiling()
                    vae_tiling = True
                except Exception as exc:  # noqa: BLE001 -- tiling is an optimisation only
                    logger.warning("video.vae_tiling_failed: %s", exc)

            resolved = build_resolved_record(
                {
                    "memory_mode": (
                        memory_mode,
                        plan.requested_mode,
                        f"planned '{plan.offload_policy}' offload from the family size table",
                    ),
                    "speed_mode": (
                        speed_mode,
                        effective_speed,
                        "quantized transformer requires compile"
                        if transformer_quant_engaged is not None
                        else "clip denoises amortise the one-time compile within a single run"
                        if speed_mode is None
                        else "requested",
                    ),
                    "attention_backend": (
                        attention_backend,
                        attention_engaged or "native",
                        "cuDNN fused attention on NVIDIA when a speed profile is active",
                    ),
                    "transformer_cache": (
                        None if cache_auto else transformer_cache,
                        cache_engaged or "off",
                        cache_reason,
                    ),
                    "transformer_quant": (
                        transformer_quant_requested,
                        transformer_quant_engaged or "off",
                        # Honest framing: the shipped torchao schemes cut load time and resident memory ~2x, but per-step GEMMs are at best bf16 parity.
                        "DiT(s) quantised (halves resident weights; hosted checkpoints cut "
                        "load time; per-step speed is roughly bf16 parity)"
                        if transformer_quant_engaged is not None
                        # A recorded decline names WHY; the generic strings are the fallback.
                        else (
                            transformer_quant_decline
                            or (
                                "skipped: offload moves the DiT, unsupported for torchao "
                                "tensors; pin a resident memory mode to combine them"
                                if quant_skipped_for_offload
                                else "not engaged (dense bf16 DiT loaded)"
                            )
                        ),
                        # Honored when the quant engaged AND when the ask was "off" (bf16 is what
                        # that asks for). Only reached with the escape hatch set for a pinned ask.
                        RESOLVED_APPLIED
                        if transformer_quant_engaged is not None or transformer_quant_pinned is None
                        else transformer_quant_decline_status,
                    ),
                    "text_encoder_quant": (
                        text_encoder_quant,
                        text_encoder_quant_engaged or "off",
                        text_encoder_outcome.reason
                        or (
                            "dense text encoder quantised in place"
                            if text_encoder_quant_engaged is not None
                            else "not engaged (dense bf16 text encoder loaded)"
                        ),
                        text_encoder_outcome.status,
                    ),
                }
            )

            with self._lock:
                if _load_token is not None and _load_token != self._load_token:
                    del pipe
                    clear_gpu_cache()
                    raise RuntimeError("Video load was cancelled or superseded.")
                self._state = _VideoLoadState(
                    pipe = pipe,
                    family = fam,
                    repo_id = repo_id,
                    base_repo = base,
                    device = device,
                    dtype = str(dtype).replace("torch.", ""),
                    kind = kind,
                    gguf_filename = gguf_filename,
                    offload_policy = offload_policy,
                    vae_tiling = vae_tiling,
                    memory_mode = plan.requested_mode,
                    speed_mode = effective_speed,
                    # Already filtered to the engaged optimisations (True names only).
                    speed_optims = speed_optims,
                    backend_flags = backend_flags,
                    attention_backend = attention_engaged,
                    transformer_cache = cache_engaged,
                    cache_auto = cache_may_toggle,
                    cache_quant_active = cache_quant_active,
                    cache_threshold = transformer_cache_threshold,
                    transformer_quant = transformer_quant_engaged,
                    text_encoder_quant = text_encoder_quant_engaged,
                    resolved = resolved,
                )
                # Ownership of the globals transferred to _state / _teardown_state_locked.
                self._precommit_globals = None
        logger.info(
            "video.loaded: %s (%s, %s, offload=%s, speed=%s, quant=%s)",
            repo_id,
            fam.name,
            kind,
            offload_policy,
            effective_speed,
            transformer_quant_engaged or "off",
        )
        return self.status()

    def _load_h3_modular_pipeline(
        self,
        *,
        diffusers: Any,
        torch: Any,
        fam: VideoFamily,
        repo_id: str,
        base: str,
        kind: str,
        dtype: Any,
        device: str,
        hf_token: Optional[str],
        memory_mode: Optional[str],
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        h3_task: Optional[str] = None,
        _load_token: Optional[int] = None,
        _base_local_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load MiniMax-H3 through its official Modular Diffusers workflow.

        ``transformer_quant`` and ``text_encoder_quant`` are ALREADY-normalised schemes (or None /
        "auto"). A scheme with a hosted pre-quantized checkpoint is built here and pre-seeded into
        the pipeline; anything else keeps the released bfloat16 components."""
        # Defence in depth: validate_load_request now refuses a single_file modular pick outright,
        # so this is unreachable from the routes. It stays because this method is also reachable
        # directly, and by the time it runs the resident pipeline has already been torn down.
        if kind != "pipeline":
            raise ValueError("MiniMax-H3 Diffusers loading requires the pipeline artifact.")
        # Load one denoiser partition. fl2va also covers text-only generation.
        workflow = h3_task or fam.modular_workflow
        manager = diffusers.ComponentsManager()
        load_kwargs: dict[str, Any] = {
            "components_manager": manager,
            "cache_dir": hub_cache_dir(),
        }
        if hf_token:
            load_kwargs["token"] = hf_token
        pipe = diffusers.ModularPipeline.from_pretrained(_base_local_dir or repo_id, **load_kwargs)
        # ── hosted pre-quantized denoiser, BEFORE load_components builds a dense one.
        transformer_quant_engaged: Optional[str] = None
        transformer_quant_reason = "released bfloat16 components"
        # "auto" asks the backend to choose, and the auto ladder quantises a DENSE module on device
        # -- something this workflow never materialises -- so it stays on the released components.
        # Only an explicitly requested scheme takes the hosted path.
        scheme = transformer_quant if transformer_quant not in (None, TQ_AUTO) else None
        # The hosted checkpoints are keyframe (fl2va) denoisers. validate_load_request refuses this
        # pairing on the route; a direct call lands here, and seeding one would silently generate
        # the wrong partition rather than fail, so drop to the released components instead.
        if scheme is not None and workflow == H3_TASK_REFERENCES:
            transformer_quant_reason = (
                f"no hosted pre-quantized {scheme} checkpoint for {fam.name} reference video"
            )
            scheme = None
        if scheme is not None:
            from .diffusion_prequant import load_prequantized_transformer, resolve_prequant_source
            from .diffusion_transformer_quant import DEFAULT_MIN_LINEAR_FEATURES
            from .video_minimax_h3_adaln import h3_prepare_prequant_model

            source = resolve_prequant_source(fam, scheme, base_repo = base)
            if source is None:
                # validate_load_request already refused this on the route; a direct call lands here.
                transformer_quant_reason = (
                    f"no hosted pre-quantized {scheme} checkpoint for {fam.name}"
                )
            else:
                transformer = load_prequantized_transformer(
                    getattr(diffusers, fam.transformer_class),
                    base,
                    source,
                    # CPU: enable_auto_cpu_offload below owns placement for every component, so a
                    # pre-placed denoiser would just be moved again (and a 66 GB pin on a device the
                    # manager wanted to offload from is how this OOMs). Same as the pre-cast text
                    # encoder path, which also hands the pipeline a CPU-resident module.
                    device = "cpu",
                    dtype = dtype,
                    hf_token = hf_token,
                    scheme = scheme,
                    # Reject a checkpoint baked under a different Linear filter, exactly as the
                    # image path does, so prequant and runtime-quant stay the same model.
                    min_features = DEFAULT_MIN_LINEAR_FEATURES,
                    # Pin the live cache root, as every other loader call does: unset, the fetch
                    # lands under huggingface_hub's import-time constant and a later cache change
                    # re-downloads multiple GB into a root Studio no longer reads.
                    cache_dir = hub_cache_dir(),
                    # The hosted H3 denoisers carry the PRUNED (curve-form) adaLN: the modulation is
                    # a rank-8 factorization of the time-embedding curve plus a shared table, which
                    # is where ~40% of the released model's parameters went. Built from the base
                    # repo's dense config, the model is 4 keys short, 1 over and 51 shapes wrong, so
                    # without this reshape the strict load fails and every hosted checkpoint falls
                    # back to the dense download it exists to avoid.
                    prepare_model = h3_prepare_prequant_model(logger),
                    logger = logger,
                )
                if transformer is None:
                    # Best-effort by contract (missing / mismatched / unreadable checkpoint), so the
                    # load continues dense rather than failing after the teardown.
                    transformer_quant_reason = (
                        f"hosted pre-quantized {scheme} checkpoint unusable; "
                        f"loaded the released bfloat16 denoiser instead"
                    )
                else:
                    # Seeding is what actually saves the download: load_components(names=None) skips
                    # every component whose attribute is already set, so the dense 66.3 GB
                    # transformer is never fetched. It must happen BEFORE load_components -- after,
                    # the dense one has already been pulled and built.
                    pipe.update_components(transformer = transformer)
                    transformer_quant_engaged = scheme
                    transformer_quant_reason = (
                        f"hosted pre-quantized {scheme} denoiser checkpoint ({source.location})"
                    )
                    logger.info(
                        "video.transformer_quant: %s pre-quantized denoiser seeded for %s",
                        scheme,
                        fam.name,
                    )
        # ── hosted quantized conditioner, for the same reason and in the same place.
        #
        # Under enable_auto_cpu_offload the VRAM floor is the LARGEST resident component, so the
        # 66.7 GB Qwen3-VL conditioner -- not the denoiser -- is what has been pinning it. Seeding
        # here rather than casting after load_components is the entire point: a runtime cast has to
        # materialise the dense encoder first, so its PEAK is the bfloat16 size it was meant to
        # avoid, and on this workflow it would also have to download the 62 GB dense component.
        text_encoder_quant_engaged: Optional[str] = None
        text_encoder_quant_reason = "released bfloat16 components"
        # Base-gated, through the same resolver the plan and the pre-fetch use, so a derivative
        # base that keeps its own conditioner can never be handed this one.
        te_scheme = self._h3_te_quant_scheme(fam, text_encoder_quant, base)
        # And gated a second time, on the identity the base NAME cannot establish. That resolver
        # compares repo ids, so a derivative stored under a directory (or repo) whose last segment
        # is also MiniMax-H3 passes it. This pipeline's own modular index names the repo its
        # conditioner would have come from, which is the actual question: a derivative that
        # retrained the encoder ships it under its own id and says so here, while one that reuses
        # the released encoder still points at it, and quantizing exactly those weights is what
        # the hosted artifact IS. Free, and it cannot disagree with the load, because it reads what
        # diffusers already parsed for it. Unanswerable is a refusal, not a pass.
        te_index_source = self._h3_te_index_source(pipe) if te_scheme is not None else None
        te_index_declined = False
        if te_scheme is not None:
            # EXACT here, unlike the plan-side comparison. te_base_equivalent falls through to
            # _same_base_model, which accepts a matching final path segment, and that tolerance is
            # the hole this gate exists to close: someone/MiniMax-H3 is a different repo with
            # different weights and would have passed it. canonical_base still folds a known mirror
            # onto the id it copies, so mirrors are not collateral. A pipeline re-saved locally
            # names local paths here and is refused, which costs the optimisation, not correctness.
            if te_index_source is None or _h3_te_canonical(te_index_source) != _h3_te_canonical(
                fam.base_repo
            ):
                te_scheme = None
                te_index_declined = True
        if text_encoder_quant is not None and te_scheme is None:
            # A valid request this workflow has no hosted artifact for, or one it has but not for
            # THIS pipeline's conditioner. Say which; do NOT erase the request.
            if te_index_declined:
                text_encoder_quant_reason = (
                    f"this pipeline's conditioner comes from "
                    f"{te_index_source or 'a source that could not be read'}, not {fam.base_repo}, "
                    f"so the hosted quantized {text_encoder_quant} conditioner is not its "
                    f"quantization; loaded this pipeline's own bfloat16 encoder instead"
                )
            elif h3_te_quant_scheme(text_encoder_quant) is not None:
                text_encoder_quant_reason = (
                    f"the hosted quantized {text_encoder_quant} conditioner is cut from "
                    f"{fam.base_repo}, not {base}; loaded this pipeline's own bfloat16 encoder "
                    f"instead"
                )
            else:
                text_encoder_quant_reason = (
                    f"no hosted quantized {text_encoder_quant} conditioner for {fam.name}; "
                    f"loaded the released bfloat16 encoder instead"
                )
        elif te_scheme is not None:
            from .video_minimax_h3_te import load_h3_quantized_text_encoder
            text_encoder = load_h3_quantized_text_encoder(
                base,
                te_scheme,
                dtype = dtype,
                hf_token = hf_token,
                # Pin the live cache root, exactly as the denoiser fetch above does.
                cache_dir = hub_cache_dir(),
                # The scoped pre-download keeps every component config, so the staged snapshot
                # already holds text_encoder/config.json: read it from there rather than sending
                # the config resolution back to the hub.
                local_base = _base_local_dir,
                logger = logger,
            )
            if text_encoder is None:
                # Best-effort by contract, like the denoiser: a missing / unreadable / mismatched
                # artifact continues dense rather than failing after the teardown.
                text_encoder_quant_reason = (
                    f"hosted quantized {te_scheme} conditioner unusable; "
                    f"loaded the released bfloat16 encoder instead"
                )
            else:
                pipe.update_components(text_encoder = text_encoder)
                text_encoder_quant_engaged = te_scheme
                text_encoder_quant_reason = (
                    f"hosted quantized {te_scheme} conditioner ({H3_TE_QUANT_REPO})"
                )
                logger.info(
                    "video.text_encoder_quant: %s quantized conditioner seeded for %s",
                    te_scheme,
                    fam.name,
                )
        # The token above only opens the modular index. Every component's own from_pretrained runs
        # here, against the repos that index names, so it has to be passed again or a gated/private
        # component load goes out anonymously (load_components only WARNS on a failed component).
        # Restrict component loading to what this workflow's blocks expect, WITHOUT pruning the
        # block graph itself (the blocks are what route each request between t2va and the
        # keyframe/reference paths). workflow= only narrows the name list load_components already
        # built, and that list skips every component whose attribute is set, so the seeding above
        # still keeps the dense 66.3 GB transformer from being fetched.
        pipe.load_components(
            workflow = workflow,
            dtype = dtype,
            **({"token": hf_token} if hf_token else {}),
        )
        # The video VAE loads at float32 and the decode runs under float16 autocast, so both
        # copies are resident for the whole decode. Pre-casting the decoder removes the pair
        # without changing a single output value. The encoder stays resident because the
        # keyframe and reference paths both encode their conditioning frames through it.

        from .video_minimax_h3 import trim_h3_video_vae

        try:
            trimmed = trim_h3_video_vae(getattr(pipe, "vae", None), workflow = workflow)
        except Exception as exc:  # noqa: BLE001 -- a saving is not worth failing a load over
            logger.warning("video.h3_vae_trim failed, keeping the full VAE: %s", exc)
        else:
            if trimmed["encoder_freed"] or trimmed["decoder_freed"]:
                logger.info(
                    "video.h3_vae_trim: freed %.2f GB (encoder %.2f, decoder pre-cast %.2f)",
                    (trimmed["encoder_freed"] + trimmed["decoder_freed"]) / 1_000_000_000,
                    trimmed["encoder_freed"] / 1_000_000_000,
                    trimmed["decoder_freed"] / 1_000_000_000,
                )
        offload_policy = "none"
        denoiser_pinned = False
        if device != "cpu":
            manager.enable_auto_cpu_offload(
                # Measured H3 activations need substantially more than the
                # official 12 GB example at 10-15 seconds. Forty GB also keeps
                # very large GPUs from retaining both 66 GB components and OOMing
                # while smaller caps succeed by offloading one of them.
                device = device,
                memory_reserve_margin = "40GB",
            )
            offload_policy = "model"
            if transformer_quant_engaged:
                # enable_auto_cpu_offload just parked every component on the CPU and will move
                # each one back inside its own pre_forward. A torchao pre-quantized denoiser does
                # not survive that mid-block move (see pin_prequantized_module), so place it now
                # and take it out of the rotation. Nothing else changes: the encoder and the VAEs
                # keep their hooks and still offload around it.
                from .diffusion_prequant import pin_prequantized_module
                denoiser_pinned = pin_prequantized_module(
                    manager, getattr(pipe, "transformer", None), device, logger = logger
                )

        resolved = build_resolved_record(
            {
                "memory_mode": (
                    memory_mode,
                    offload_policy,
                    "MiniMax-H3 ComponentsManager auto CPU offload"
                    + ("; the pre-quantized denoiser stays resident" if denoiser_pinned else ""),
                ),
                "speed_mode": (None, "off", "modular pipeline uses its native execution path"),
                "attention_backend": (None, "native", "Diffusers model default"),
                "transformer_cache": (None, "off", "not supported by this modular workflow"),
                "transformer_quant": (
                    transformer_quant,
                    transformer_quant_engaged or "off",
                    transformer_quant_reason,
                ),
                # The REQUEST on the left, not None: a declined or failed request has to read as a
                # fallback in the resolved record, never disappear into a bare "off" that looks
                # like the caller never asked.
                "text_encoder_quant": (
                    text_encoder_quant,
                    text_encoder_quant_engaged or "off",
                    text_encoder_quant_reason,
                ),
            }
        )
        with self._lock:
            if _load_token is not None and _load_token != self._load_token:
                del pipe
                clear_gpu_cache()
                raise RuntimeError("Video load was cancelled or superseded.")
            self._state = _VideoLoadState(
                pipe = pipe,
                family = fam,
                repo_id = repo_id,
                base_repo = base,
                device = device,
                dtype = str(dtype).replace("torch.", ""),
                kind = kind,
                engine = "diffusers",
                h3_task = workflow,
                offload_policy = offload_policy,
                vae_tiling = True,
                memory_mode = normalize_memory_mode(memory_mode),
                speed_mode = SPEED_OFF,
                # The ENGAGED scheme, not the requested one, exactly as the conventional path
                # records it: status() reports this field, and a dense fallback must not claim a
                # quant the resident denoiser does not have.
                transformer_quant = transformer_quant_engaged,
                # Ditto for the conditioner: the memory preflight reads this field back to size the
                # resident encoder, so a dense fallback recorded as int8 would under-state the floor
                # by 39 GB and let a doomed generation start.
                text_encoder_quant = text_encoder_quant_engaged,
                h3_denoiser_pinned = denoiser_pinned,
                resolved = resolved,
            )
        logger.info(
            "video.loaded: %s (%s, modular diffusers, transformer_quant=%s, text_encoder_quant=%s)",
            repo_id,
            fam.name,
            transformer_quant_engaged or "off",
            text_encoder_quant_engaged or "off",
        )
        return self.status()

    @staticmethod
    def _resolve_checkpoint_path(
        repo_id: str, gguf_filename: Optional[str], hf_token: Optional[str]
    ) -> Path:
        """The local checkpoint file for a gguf/single_file load (downloads if hub)."""
        from .diffusion_families import resolve_local_gguf_child

        root = Path(repo_id).expanduser()
        if root.is_dir():
            return resolve_local_gguf_child(root, gguf_filename or "")
        if root.is_file():
            return root
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        return Path(hf_hub_download_with_xet_fallback(repo_id, gguf_filename or "", hf_token))

    # ── generation ───────────────────────────────────────────────────────────

    def loaded_family(self) -> Optional[VideoFamily]:
        """The resident pipeline's family, or None when nothing is loaded.

        The generate route reads it to enforce that family's shape rules (resolution
        presets + frame lattice) at the API boundary, before the request reaches the
        worker. ``getattr`` rather than ``state.family`` on purpose: a state object
        that carries no family degrades to the old snapping path instead of raising.
        """
        with self._lock:
            state = self._state
        return getattr(state, "family", None) if state is not None else None

    @staticmethod
    def _reset_step_cache(pipe: Any) -> None:
        """Clear FBCache residuals on the resident DiT(s) before a generation.

        diffusers keys the residuals on the long-lived transformer and no pipeline
        resets them, so the next clip would compare against the previous request's
        state: a shape mismatch when the resolution changed, stale reuse otherwise.
        ``_reset_stateful_cache`` is the transformer-level entry point in diffusers
        0.39 (``reset_stateful_hooks`` lives only on the HookRegistry). Best-effort:
        an uncached transformer is a silent no-op."""
        for name in ("transformer", "transformer_2"):
            module = getattr(pipe, name, None)
            reset = getattr(module, "_reset_stateful_cache", None) or getattr(
                module, "reset_stateful_hooks", None
            )
            if callable(reset):
                try:
                    reset()
                except Exception:  # noqa: BLE001 -- reset is best-effort, never fail a generation
                    pass

    def begin_generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        guidance_2: Optional[float] = None,
        seed: Optional[int] = None,
        first_frame: Optional[str] = None,
        last_frame: Optional[str] = None,
        reference_images: Optional[list[str]] = None,
        reference_videos: Optional[list[dict]] = None,
        reference_audios: Optional[list[str]] = None,
        reference_image_size: Optional[str] = None,
        flow_shift: Optional[float] = None,
        audio_flow_shift: Optional[float] = None,
    ) -> None:
        """Validate cheaply, then run generate + gallery persist on a daemon thread.

        Returns at once, mirroring begin_load: a clip takes minutes to denoise, and
        a proxy in front of Studio (secure mode's Cloudflare tunnel) caps the origin
        response window near 100 seconds, so the HTTP call must not span the
        generation. The terminal outcome (phase "completed" with the saved gallery
        record, or "failed" with a client-safe error) is reported by
        generate_progress(); cancel_generate() keeps working against the job.
        Raises RuntimeError with VIDEO_NOT_LOADED_MSG / VIDEO_GENERATION_BUSY_MSG
        sentinels the route maps to 409.
        """
        cancel = threading.Event()
        # Snapshot load state before decoding outside the backend lock.
        with self._lock:
            if self._state is None:
                raise RuntimeError(VIDEO_NOT_LOADED_MSG)
            if self._generate_job_active:
                raise RuntimeError(VIDEO_GENERATION_BUSY_MSG)
            family = self._state.family
            task = self._state.h3_task
            engine = self._state.engine
        # Validate conditioning before creating the asynchronous job so request errors return 400.
        _, _, canvas_w, canvas_h, _ = self._resolve_keyframes(
            family, task, first_frame, last_frame, width, height
        )
        self._resolve_references(
            family,
            task,
            engine,
            reference_images,
            reference_videos,
            reference_audios,
            reference_image_size,
            canvas_w,
            canvas_h,
        )
        self._resolve_flow_shifts(family, engine, flow_shift, audio_flow_shift)
        with self._lock:
            if self._state is None:
                raise RuntimeError(VIDEO_NOT_LOADED_MSG)
            if self._generate_job_active:
                raise RuntimeError(VIDEO_GENERATION_BUSY_MSG)
            # Under the SAME lock that reserves the state this job will run against. A load
            # commits its new state here too, so judging the shape from a separate earlier read
            # could accept a size for the family being replaced and then denoise it with the new
            # one, or reject a size the new family supports. getattr, so a state carrying no
            # family degrades to the old snapping rather than raising.
            fam = getattr(self._state, "family", None)
            if fam is not None:
                validate_video_request_shape(fam, width = width, height = height, num_frames = num_frames)
            self._generate_job_active = True
            # Register BEFORE the worker starts so a cancel/unload in the spawn window still stops the run.
            self._active_generate_cancel = cancel
            self._gen = {
                "active": True,
                "phase": "queued",
                "step": 0,
                "total": 0,
                "eta_seconds": None,
            }
        threading.Thread(
            target = self._run_generate,
            kwargs = dict(
                prompt = prompt,
                negative_prompt = negative_prompt,
                width = width,
                height = height,
                num_frames = num_frames,
                fps = fps,
                steps = steps,
                guidance = guidance,
                guidance_2 = guidance_2,
                seed = seed,
                first_frame = first_frame,
                last_frame = last_frame,
                reference_images = reference_images,
                reference_videos = reference_videos,
                reference_audios = reference_audios,
                reference_image_size = reference_image_size,
                flow_shift = flow_shift,
                audio_flow_shift = audio_flow_shift,
                cancel_event = cancel,
            ),
            daemon = True,
        ).start()

    def _run_generate(self, *, cancel_event: threading.Event, **gen_kwargs: Any) -> None:
        """begin_generate's worker: generate, persist to the gallery, record the
        terminal state where generate_progress() reports it. The error mapping is
        the exact one the route applied when the call was synchronous: ValueError
        text is client input feedback, sentinel RuntimeErrors pass through, and any
        other failure is logged server-side and reported as a generic message so
        internals (CUDA state, paths) never reach the client."""
        from . import video_gallery

        try:
            result = self.generate(cancel_event = cancel_event, **gen_kwargs)
        except ValueError as exc:
            self._finish_generate_job(cancel_event = cancel_event, error = str(exc))
            return
        except RuntimeError as exc:
            msg = str(exc)
            if msg not in (VIDEO_NOT_LOADED_MSG, VIDEO_CANCELLED_MSG):
                logger.error("video.generate_failed: %s", exc, exc_info = True)
                msg = "Video generation failed."
            self._finish_generate_job(cancel_event = cancel_event, error = msg)
            return
        except Exception as exc:  # noqa: BLE001 -- worker thread: never propagate
            logger.error("video.generate_failed: %s", exc, exc_info = True)
            self._finish_generate_job(cancel_event = cancel_event, error = "Video generation failed.")
            return

        # Persist the clip with its full recipe as the JSON sidecar the gallery reads back.
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            record = video_gallery.save(
                result["mp4_bytes"],
                {
                    "prompt": gen_kwargs["prompt"],
                    "negative_prompt": gen_kwargs.get("negative_prompt"),
                    "width": result["width"],
                    "height": result["height"],
                    "num_frames": result["num_frames"],
                    "fps": result["fps"],
                    "duration_s": result["duration_s"],
                    "steps": result["steps"],
                    "guidance": result["guidance"],
                    "guidance_2": gen_kwargs.get("guidance_2"),
                    "seed": result["seed"],
                    "has_audio": result["has_audio"],
                    "conditioning": result["conditioning"],
                    "flow_shift": result.get("flow_shift"),
                    "audio_flow_shift": result.get("audio_flow_shift"),
                    "model": result["repo_id"],
                    # The load-time build, so a clip's recipe still names the precision it ran at
                    # once the model is unloaded. All engaged values; absent keys read back as null
                    # on sidecars written before this existed (they are not in _REQUIRED_META).
                    "model_kind": result.get("model_kind"),
                    "gguf_filename": result.get("gguf_filename"),
                    "transformer_quant": result.get("transformer_quant"),
                    "text_encoder_quant": result.get("text_encoder_quant"),
                    "memory_mode": result.get("memory_mode"),
                    "offload_policy": result.get("offload_policy"),
                    "created_at": created_at,
                },
            )
        except Exception as exc:  # noqa: BLE001 -- disk failure must reach the poller
            logger.error("video.persist_failed: %s", exc)
            self._finish_generate_job(
                cancel_event = cancel_event, error = "Failed to save the generated video."
            )
            return
        self._finish_generate_job(cancel_event = cancel_event, video = record, total = result["steps"])

    def _finish_generate_job(
        self,
        *,
        cancel_event: Optional[threading.Event] = None,
        video: Optional[dict] = None,
        error: Optional[str] = None,
        total: int = 0,
    ) -> None:
        """Record a job's terminal state as one atomic swap. The terminal dict
        replaces the live-progress one so a poll can never mix fields from both,
        and the busy flag drops in the same critical section so the earliest
        moment a new begin_generate() can start is after the outcome is visible."""
        with self._lock:
            self._generate_job_active = False
            if cancel_event is not None and self._active_generate_cancel is cancel_event:
                # Covers a worker that failed before reaching generate()'s finally; identity-guarded so a direct generate() keeps its handle.
                self._active_generate_cancel = None
            if error is not None:
                self._gen = {
                    "active": False,
                    "phase": "failed",
                    "error": error,
                    "step": 0,
                    "total": 0,
                    "eta_seconds": None,
                }
            else:
                self._gen = {
                    "active": False,
                    "phase": "completed",
                    "video": video,
                    "step": total,
                    "total": total,
                    "eta_seconds": None,
                }

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        guidance_2: Optional[float] = None,
        seed: Optional[int] = None,
        first_frame: Optional[str] = None,
        last_frame: Optional[str] = None,
        reference_images: Optional[list[str]] = None,
        reference_videos: Optional[list[dict]] = None,
        reference_audios: Optional[list[str]] = None,
        reference_image_size: Optional[str] = None,
        flow_shift: Optional[float] = None,
        audio_flow_shift: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        # begin_generate passes its already-registered event; a direct call makes its own.
        cancel = cancel_event if cancel_event is not None else threading.Event()
        with self._generate_lock:
            with self._lock:
                # A teardown is waiting for this lock and Python locks are not FIFO, so refuse rather than denoise against a pipeline that is already being torn down.
                if self._teardown_waiters:
                    raise RuntimeError(VIDEO_CANCELLED_MSG)
                state = self._state
                if state is None:
                    raise RuntimeError(VIDEO_NOT_LOADED_MSG)
                self._active_generate_cancel = cancel
            # Bound below, once the request is resolved. None means the failure beat the
            # resolution, and there is nothing truthful to report.
            request_shape: Optional[dict[str, Any]] = None
            try:
                fam = state.family
                first_pil, last_pil, width, height, conditioning = self._resolve_keyframes(
                    fam, state.h3_task, first_frame, last_frame, width, height
                )
                references = self._resolve_references(
                    fam,
                    state.h3_task,
                    state.engine,
                    reference_images,
                    reference_videos,
                    reference_audios,
                    reference_image_size,
                    width,
                    height,
                )
                if references:
                    conditioning = h3_conditioning_mode(has_references = True)
                frames = snap_num_frames(fam, num_frames or fam.default_num_frames)
                out_fps = (
                    fam.default_fps if fam.name == "minimax-h3" else int(fps or fam.default_fps)
                )
                default_steps, default_guidance = default_video_generation_params(
                    state.gguf_filename,
                    state.repo_id,
                    state.base_repo,
                    fallback = (fam.default_steps, fam.default_guidance),
                )
                steps = int(steps or default_steps)
                guidance = float(default_guidance if guidance is None else guidance)
                shift, audio_shift = self._resolve_flow_shifts(
                    fam, state.engine, flow_shift, audio_flow_shift
                )

                if state.engine == "sd_cpp":
                    return self._generate_h3_native(
                        state = state,
                        prompt = prompt,
                        width = width,
                        height = height,
                        frames = frames,
                        fps = out_fps,
                        steps = steps,
                        guidance = guidance,
                        seed = seed,
                        first_frame = first_pil,
                        last_frame = last_pil,
                        references = references,
                        conditioning = conditioning,
                        flow_shift = shift,
                        cancel = cancel,
                    )

                import torch

                if state.engine == "diffusers" and fam.modular_workflow and state.device != "cpu":
                    from .video_minimax_h3 import (
                        H3_TEXT_ENCODER_BF16_GB,
                        estimate_h3_diffusers_host_ram_gb,
                        estimate_h3_diffusers_vram_gb,
                    )

                    device_obj = torch.device(state.device)
                    device_module = getattr(torch, device_obj.type, None)
                    if device_module is not None and hasattr(device_module, "mem_get_info"):
                        free_bytes, _ = device_module.mem_get_info(device_obj)
                        reserved_bytes = (
                            device_module.memory_reserved(device_obj)
                            if hasattr(device_module, "memory_reserved")
                            else 0
                        )
                        available_vram_gb = (free_bytes + reserved_bytes) / 1_000_000_000
                        # Size the floor from the components this load ACTUALLY holds, not from
                        # the released bfloat16 pair. Both fields are the ENGAGED schemes (a
                        # declined or failed request is recorded as None at load), so a dense
                        # fallback keeps the dense floor.
                        required_vram_gb = estimate_h3_diffusers_vram_gb(
                            width,
                            height,
                            frames,
                            text_encoder_gb = h3_te_resident_gb(
                                state.text_encoder_quant, bf16_gb = H3_TEXT_ENCODER_BF16_GB
                            ),
                            transformer_gb = h3_transformer_resident_gb(state.transformer_quant),
                            transformer_pinned = bool(getattr(state, "h3_denoiser_pinned", False)),
                        )
                        if available_vram_gb + 0.25 < required_vram_gb:
                            raise RuntimeError(
                                f"MiniMax-H3 needs about {required_vram_gb:.1f} GB available "
                                f"VRAM for {width}x{height} at {frames} frames; "
                                f"{available_vram_gb:.1f} GB is available. Lower the resolution "
                                "or duration, or load the GGUF artifact."
                            )

                        import psutil

                        process_rss = psutil.Process().memory_info().rss
                        host_capacity_gb = (
                            psutil.virtual_memory().available + process_rss
                        ) / 1_000_000_000
                        # Same engaged components as the VRAM floor above. Sizing one from what
                        # the load holds and the other from the released pair refuses exactly the
                        # configuration the quantized components exist for.
                        required_host_gb = estimate_h3_diffusers_host_ram_gb(
                            available_vram_gb,
                            text_encoder_gb = h3_te_resident_gb(
                                state.text_encoder_quant, bf16_gb = H3_TEXT_ENCODER_BF16_GB
                            ),
                            transformer_gb = h3_transformer_resident_gb(state.transformer_quant),
                        )
                        if host_capacity_gb + 0.5 < required_host_gb:
                            raise RuntimeError(
                                f"MiniMax-H3 needs about {required_host_gb:.0f} GB available "
                                f"system RAM at this VRAM tier; {host_capacity_gb:.1f} GB is "
                                "available. Load the GGUF artifact instead."
                            )

                generator = torch.Generator(device = "cpu" if fam.modular_workflow else state.device)
                if seed is None:
                    seed = int(generator.seed()) % (2**53)
                generator = generator.manual_seed(int(seed))

                # The RESOLVED request, snapshotted the moment it is known, so a failure logs what
                # actually ran rather than the caller's Nones. Without it the whole server-side
                # record of a failed video is the exception string: #8225 reported an OOM whose
                # resolution and frame count had to be reverse-engineered out of the allocation
                # size by hand. No prompt text -- a length is enough to tell an empty prompt from
                # a long one, and the log is not the place for user content.
                request_shape = {
                    "family": fam.name,
                    "repo_id": state.repo_id,
                    "gguf": state.gguf_filename,
                    "width": width,
                    "height": height,
                    "frames": frames,
                    "fps": out_fps,
                    "steps": steps,
                    "guidance": guidance,
                    "guidance_2": guidance_2,
                    "seed": int(seed),
                    "prompt_chars": len(prompt or ""),
                    "negative_prompt_chars": len(negative_prompt or ""),
                    "device": state.device,
                    "dtype": state.dtype,
                    "offload": state.offload_policy,
                    "attention_backend": state.attention_backend,
                }

                pipe = state.pipe
                call_params = inspect.signature(pipe.__call__).parameters
                kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "width": width,
                    "height": height,
                    "num_frames": frames,
                    "generator": generator,
                }
                # The 2.3 distilled DiT was trained on a fixed 8-step sigma curve (DISTILLED_SIGMA_VALUES); at the distilled default
                # step count pass it verbatim with the scheduler re-shaping neutralised. Any other count keeps the scheduler's spacing.
                sigma_ctx: Any = contextlib.nullcontext()
                if fam.name == "ltx-2" and "sigmas" in call_params:
                    from .video_ltx2 import (
                        LTX23_DISTILLED_SIGMAS,
                        ltx2_distilled_ids,
                        ltx23_verbatim_sigmas,
                    )
                    if steps == len(LTX23_DISTILLED_SIGMAS) and ltx2_distilled_ids(
                        state.gguf_filename, state.repo_id, state.base_repo
                    ):
                        kwargs["sigmas"] = list(LTX23_DISTILLED_SIGMAS)
                        sigma_ctx = ltx23_verbatim_sigmas(pipe)
                if not fam.supports_cfg:
                    pass
                elif fam.guidance_via_guider:
                    # HunyuanVideo-1.5: __call__ has no guidance kwarg; CFG scale is a guider attribute set per request.
                    pipe.guider.guidance_scale = float(guidance)
                else:
                    kwargs[fam.cfg_kwarg] = guidance
                if negative_prompt and "negative_prompt" in call_params:
                    kwargs["negative_prompt"] = negative_prompt
                # LTX-2 takes frame_rate (shapes audio length); others fix their rate, fps only at export.
                if "frame_rate" in call_params:
                    kwargs["frame_rate"] = float(out_fps)
                # Dual-DiT MoE: pass the low-noise expert guidance only when the family declares it AND the signature accepts it (WanPipeline raises with boundary_ratio=None).
                if fam.cfg2_kwarg and fam.cfg2_kwarg in call_params and guidance_2 is not None:
                    kwargs[fam.cfg2_kwarg] = float(guidance_2)
                if fam.modular_workflow:
                    kwargs["output"] = ["videos", "audio", "sampling_rate"]
                # Auto-blocks distinguish omitted conditioning from empty conditioning.
                if first_pil is not None:
                    kwargs["image"] = first_pil
                if last_pil is not None:
                    kwargs["last_image"] = last_pil
                if references:
                    from .video_minimax_h3 import h3_diffusers_references
                    kwargs["references"] = h3_diffusers_references(references)
                # Set scheduler shifts every run so prior requests cannot leak state.
                for component, value in (
                    ("scheduler", shift),
                    ("audio_scheduler", audio_shift),
                ):
                    setter = getattr(getattr(pipe, component, None), "set_shift", None)
                    if callable(setter) and value is not None:
                        setter(float(value))

                started = time.monotonic()
                self._gen = {
                    "active": True,
                    "phase": "denoise",
                    "step": 0,
                    "total": steps,
                    "started": started,
                    "eta_seconds": None,
                    "error": None,
                }

                def _tick(done: int) -> None:
                    elapsed = time.monotonic() - started
                    self._gen.update(
                        step = done,
                        eta_seconds = (elapsed / max(1, done)) * max(0, steps - done),
                    )

                def _on_step(p, step_index, timestep, callback_kwargs):
                    if cancel.is_set():
                        p._interrupt = True
                        return callback_kwargs
                    _tick(step_index + 1)
                    return callback_kwargs

                def _on_scheduler_step(done: int) -> None:
                    # No cooperative _interrupt here, so cancellation must unwind the denoise loop via an exception.
                    if cancel.is_set():
                        raise _VideoGenerationCancelled()
                    _tick(done)

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step
                    progress_ctx = contextlib.nullcontext()
                else:
                    # HunyuanVideo-1.5 has no step callback, so wrap scheduler.step for progress + cancel and restore afterwards.
                    # Same for H3's modular workflow: ModularPipeline takes no callback (an unknown input is only warned
                    # about), but MiniMaxH3LoopSchedulerStep calls components.scheduler.step -- pipe.scheduler -- once per
                    # denoise step, so the wrapper ticks and can unwind a multi-minute run on Cancel.
                    progress_ctx = _scheduler_step_progress(pipe, _on_scheduler_step)

                # Re-check an AUTO cache decision against the ACTUAL step count; explicit choices never toggle.
                if state.cache_auto:
                    toggled = state.transformer_cache
                    for view in _views_for(pipe, fam):
                        toggled = maybe_toggle_step_cache(
                            view,
                            steps = steps,
                            quant_active = state.cache_quant_active,
                            threshold = state.cache_threshold,
                            logger = logger,
                        )
                    if toggled != state.transformer_cache:
                        # _VideoLoadState is frozen; record the pipe-level toggle so status() is truthful.
                        object.__setattr__(state, "transformer_cache", toggled)
                        entry = (state.resolved or {}).get("transformer_cache")
                        if isinstance(entry, dict):
                            entry["value"] = toggled or "off"
                            entry["reason"] = (
                                f"auto: {steps}-step generation "
                                + ("reaches" if toggled else "is below")
                                + f" {FBCACHE_MIN_STEPS}"
                            )
                if state.transformer_cache:
                    self._reset_step_cache(pipe)
                try:
                    with torch.inference_mode(), progress_ctx, sigma_ctx:
                        output = pipe(**kwargs)
                except _VideoGenerationCancelled:
                    # Unwinding by exception skips maybe_free_model_hooks(); under offload the onloaded modules would stay on the GPU.
                    free_hooks = getattr(pipe, "maybe_free_model_hooks", None)
                    if callable(free_hooks):
                        try:
                            free_hooks()
                        except Exception:  # noqa: BLE001 -- cleanup is best-effort
                            pass
                    raise RuntimeError(VIDEO_CANCELLED_MSG) from None
                if cancel.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)

                self._gen.update(phase = "export", eta_seconds = None)
                if fam.modular_workflow:
                    video_frames = output["videos"][0]
                    audio = output.get("audio")
                    audio_track = audio[0] if audio is not None else None
                    audio_sample_rate = output.get("sampling_rate")
                else:
                    video_frames = output.frames[0]
                    audio = getattr(output, "audio", None)
                    audio_track = audio[0] if fam.has_audio and audio is not None else None
                    audio_sample_rate = None
                if audio_sample_rate is None:
                    mp4_bytes = self._encode_mp4(
                        video_frames, out_fps, audio_track, pipe if fam.has_audio else None
                    )
                else:
                    mp4_bytes = self._encode_mp4(
                        video_frames,
                        out_fps,
                        audio_track,
                        pipe if fam.has_audio else None,
                        audio_sample_rate = int(audio_sample_rate),
                    )
                # A cancel during the blocking export/mux must still discard the clip; re-check before it is persisted.
                if cancel.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)
                duration_s = len(video_frames) / float(out_fps) if out_fps else 0.0
                self._gen = {"active": False}
                return {
                    "mp4_bytes": mp4_bytes,
                    "seed": int(seed),
                    "repo_id": state.repo_id,
                    "width": width,
                    "height": height,
                    "num_frames": len(video_frames),
                    "fps": out_fps,
                    "duration_s": duration_s,
                    "has_audio": bool(audio_track is not None),
                    "conditioning": conditioning,
                    "steps": steps,
                    "guidance": guidance,
                    "flow_shift": shift,
                    "audio_flow_shift": audio_shift,
                    # The BUILD this clip came off, read from the ENGAGED state and never from the
                    # request, so a saved MP4 can never be labelled with a precision that did not
                    # run. Mirrors what the image gallery already records.
                    "model_kind": state.kind,
                    "gguf_filename": state.gguf_filename,
                    "transformer_quant": state.transformer_quant,
                    "text_encoder_quant": state.text_encoder_quant,
                    "memory_mode": state.memory_mode,
                    "offload_policy": state.offload_policy,
                }
            except Exception as exc:
                self._gen = {"active": False}
                _log_failed_generation(request_shape, exc)
                raise
            finally:
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None

    @staticmethod
    def _resolve_keyframes(
        fam: VideoFamily,
        h3_task: Optional[str],
        first_frame: Optional[str],
        last_frame: Optional[str],
        width: Optional[int],
        height: Optional[int],
    ) -> tuple[Any, Any, int, int, str]:
        """Decode keyframes and resolve their shared generation canvas.

        Unsupported keyframes are rejected. Omitting both dimensions matches the source aspect.
        """
        if first_frame or last_frame:
            if not fam.supports_keyframes:
                raise ValueError(
                    f"{fam.name} generates from the prompt alone; it takes no first or last frame."
                )
            if h3_task == H3_TASK_REFERENCES:
                raise ValueError(
                    "The loaded MiniMax-H3 checkpoint is the Ref2VA partition, which conditions "
                    "on references rather than keyframes. Load a minimax_h3_fl2va checkpoint to "
                    "generate from a first or last frame."
                )
        first = last = None
        if first_frame or last_frame:
            from .diffusion import decode_b64_image
            first = decode_b64_image(first_frame, mode = "RGB") if first_frame else None
            last = decode_b64_image(last_frame, mode = "RGB") if last_frame else None

        if (width is None or height is None) and (first is not None or last is not None):
            width, height = h3_canvas_for_aspect(*(first if first is not None else last).size)
        else:
            # begin_generate runs this for EVERY family, keyframes or not, so it cannot assume a
            # family declares presets: an unusual/custom one with an empty tuple used to die here
            # with an IndexError (a 500) before the request reached the worker. Fall back to the
            # generic 768x512 the rest of the video path already treats as the default canvas.
            default_w, default_h = (
                fam.resolution_presets[0] if fam.resolution_presets else (768, 512)
            )
            width, height = snap_video_size(
                fam,
                width or default_w,
                height or default_h,
            )
        # Fit once so both engines condition on identical pixels.
        if first is not None:
            first = fit_h3_keyframe(first, width, height, anchor = H3_ANCHOR_FIRST)
        if last is not None:
            last = fit_h3_keyframe(last, width, height, anchor = H3_ANCHOR_LAST)
        conditioning = h3_conditioning_mode(has_first = first is not None, has_last = last is not None)
        return first, last, width, height, conditioning

    @staticmethod
    def _resolve_flow_shifts(
        fam: VideoFamily,
        engine: str,
        flow_shift: Optional[float],
        audio_flow_shift: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Resolve requested or released sigma shifts.

        sd.cpp rejects non-default audio shifts because it cannot apply them.
        """
        if flow_shift is not None and fam.default_flow_shift is None:
            raise ValueError(f"{fam.name} does not expose a video flow_shift control.")
        if audio_flow_shift is not None and fam.default_audio_flow_shift is None:
            raise ValueError(f"{fam.name} does not expose an audio_flow_shift control.")
        shift = flow_shift if flow_shift is not None else fam.default_flow_shift
        audio_shift = (
            audio_flow_shift if audio_flow_shift is not None else fam.default_audio_flow_shift
        )
        if (
            audio_flow_shift is not None
            and engine == "sd_cpp"
            and audio_flow_shift != fam.default_audio_flow_shift
        ):
            raise ValueError(
                "stable-diffusion.cpp derives the audio schedule against a fixed "
                f"{fam.default_audio_flow_shift:g} shift, so audio_flow_shift needs the "
                "Diffusers engine."
            )
        return shift, audio_shift

    @staticmethod
    def _resolve_references(
        fam: VideoFamily,
        h3_task: Optional[str],
        engine: str,
        reference_images: Optional[list[str]],
        reference_videos: Optional[list[dict]],
        reference_audios: Optional[list[str]],
        reference_image_size: Optional[str],
        width: int,
        height: int,
    ) -> Any:
        """Decode the request's omni references into the one object both engines consume.

        Sizing is settled here for the same reason keyframes are: sd.cpp scales every reference
        to the generation's pixel area and the Diffusers blocks scale to their own 2048px short
        edge, and both are min(1, ...) scales, so pre-fitting to the requested policy makes each
        engine's own rule a no-op and the two agree. That is also why ``max`` is refused on the
        native engine rather than accepted and quietly undone.
        """
        from .video_minimax_h3 import (
            MiniMaxH3References,
            decode_h3_reference_audio,
            decode_h3_reference_video,
        )

        images = list(reference_images or [])
        videos = list(reference_videos or [])
        audios = list(reference_audios or [])
        if not (images or videos or audios):
            if h3_task == H3_TASK_REFERENCES:
                # Ref2VA has no text-only denoiser, so every request needs an image or video.
                raise ValueError(
                    "The loaded MiniMax-H3 checkpoint is the Ref2VA partition, which generates "
                    "from references. Add at least one reference image or video, or load a "
                    "minimax_h3_fl2va checkpoint for text-to-video."
                )
            return MiniMaxH3References()
        if not fam.supports_references:
            raise ValueError(f"{fam.name} takes no reference images, videos or audio.")
        if h3_task != H3_TASK_REFERENCES:
            raise ValueError(
                "The loaded MiniMax-H3 checkpoint is the FL2VA partition, which conditions on "
                "keyframes rather than references. Load a minimax_h3_ref2va checkpoint to "
                "generate from references."
            )
        policy = (reference_image_size or H3_REF_SIZE_MATCH).strip().lower()
        if policy not in (H3_REF_SIZE_MATCH, H3_REF_SIZE_MAX):
            raise ValueError(
                f"reference_image_size must be '{H3_REF_SIZE_MATCH}' or '{H3_REF_SIZE_MAX}'."
            )
        if policy == H3_REF_SIZE_MAX and engine == "sd_cpp":
            raise ValueError(
                "stable-diffusion.cpp scales every reference to the generation's pixel area, so "
                f"'{H3_REF_SIZE_MAX}' reference sizing needs the Diffusers engine. Use "
                f"'{H3_REF_SIZE_MATCH}' with this checkpoint."
            )

        from .diffusion import decode_b64_image

        fitted_images = tuple(
            fit_h3_reference_image(
                decode_b64_image(item, mode = "RGB"),
                width = width,
                height = height,
                policy = policy,
            )
            for item in images
        )
        decoded_videos = []
        for item in videos:
            blob = _decode_b64_media(item.get("video") if isinstance(item, dict) else item)
            frames, waveform, sample_rate = decode_h3_reference_video(blob)
            override = item.get("audio") if isinstance(item, dict) else None
            if override:
                waveform, sample_rate = decode_h3_reference_audio(_decode_b64_media(override))
            decoded_videos.append((frames, waveform, sample_rate))
        if engine == "sd_cpp":
            soundtrack_gap = False
            for index, (_, waveform, _) in enumerate(decoded_videos, start = 1):
                if waveform is None:
                    soundtrack_gap = True
                elif soundtrack_gap:
                    # sd-cli pairs soundtrack flags by position, so gaps would misattach audio.
                    raise ValueError(
                        "stable-diffusion.cpp can pair reference-video soundtracks only from "
                        f"Video 1 onward without gaps, but Video {index} has audio after a "
                        "silent earlier video. Reorder the videos so ones with soundtracks "
                        "come first, or add a replacement soundtrack to each earlier video."
                    )
        decoded_audios = tuple(
            decode_h3_reference_audio(_decode_b64_media(item)) for item in audios
        )
        return MiniMaxH3References(
            images = fitted_images,
            videos = tuple(decoded_videos),
            audios = decoded_audios,
        )

    def _generate_h3_native(
        self,
        *,
        state: _VideoLoadState,
        prompt: str,
        width: int,
        height: int,
        frames: int,
        fps: int,
        steps: int,
        guidance: float,
        seed: Optional[int],
        # Canvas-fitted keyframes, and the task name they add up to (see _resolve_keyframes).
        first_frame: Any,
        last_frame: Any,
        references: Any,
        conditioning: str,
        flow_shift: Optional[float],
        cancel: threading.Event,
    ) -> dict[str, Any]:
        import random
        import re

        from .sd_cpp_args import SdCppVideoGenParams
        from .sd_cpp_engine import SdCppCancelled
        from .video_minimax_h3 import (
            inspect_video,
            stage_h3_references,
            transcode_video_to_mp4,
        )

        runtime = state.pipe
        if seed is None:
            seed = random.SystemRandom().randrange(0, 2**53)
        started = time.monotonic()
        self._gen = {
            "active": True,
            "phase": "denoise",
            "step": 0,
            "total": steps,
            "started": started,
            "eta_seconds": None,
            "error": None,
        }
        # sd.cpp reuses an unlabeled bar for loading, denoising, and VAE work. Count only
        # iteration-rated bars inside its denoise window.
        denoise_open = re.compile(r"generate_video\s+\d+x\d+x\d+")
        denoise_close = re.compile(r"sampling completed", re.IGNORECASE)
        step_bar = re.compile(r"\|\s*(\d+)\s*/\s*(\d+)\s*-\s*[\d.]+\s*(?:it/s|s/it)")
        denoising = False

        def on_log(line: str) -> None:
            nonlocal denoising
            if denoise_open.search(line):
                denoising = True
                return
            if denoise_close.search(line):
                denoising = False
                # sd.cpp still has to decode both latent streams and write its
                # intermediate video. Do not leave the UI at a completed denoise bar.
                self._gen.update(phase = "decode", eta_seconds = None)
                return
            match = step_bar.search(line) if denoising else None
            if match is None:
                return
            done, total = int(match.group(1)), int(match.group(2))
            elapsed = time.monotonic() - started
            self._gen.update(
                step = done,
                total = total,
                eta_seconds = (elapsed / max(1, done)) * max(0, total - done),
            )

        tmp = tempfile.NamedTemporaryFile(suffix = ".webm", delete = False)
        tmp.close()
        output_path = Path(tmp.name)
        # Stage in-memory conditioning in sd-cli's temporary disk format.
        with tempfile.TemporaryDirectory(prefix = "unsloth-h3-keyframes-") as scratch:

            def stage(image: Any, name: str) -> Optional[str]:
                if image is None:
                    return None
                path = Path(scratch) / f"{name}.png"
                image.save(path, format = "PNG")
                return str(path)

            init_img = stage(first_frame, "first")
            end_img = stage(last_frame, "last")
            staged = stage_h3_references(references, Path(scratch))
            try:
                try:
                    generated = runtime.engine.generate_video(
                        runtime.files,
                        SdCppVideoGenParams(
                            prompt = prompt,
                            width = width,
                            height = height,
                            num_frames = frames,
                            fps = fps,
                            steps = steps,
                            cfg_scale = 1.0,
                            seed = int(seed),
                            init_img = init_img,
                            end_img = end_img,
                            ref_images = staged.images,
                            ref_videos = staged.videos,
                            ref_video_audios = staged.video_audios,
                            ref_audios = staged.audios,
                            flow_shift = flow_shift,
                        ),
                        output_path = str(output_path),
                        offload = list(runtime.offload_flags),
                        on_log = on_log,
                        cancel_event = cancel,
                    )
                except SdCppCancelled:
                    raise RuntimeError(VIDEO_CANCELLED_MSG) from None
                if cancel.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)
                self._gen.update(phase = "export", eta_seconds = None)
                actual_width, actual_height, actual_frames, has_audio = inspect_video(generated)
                mp4_bytes = transcode_video_to_mp4(generated, fps = fps)
                if cancel.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)
                self._gen = {"active": False}
                return {
                    "mp4_bytes": mp4_bytes,
                    "seed": int(seed),
                    "repo_id": state.repo_id,
                    "width": actual_width,
                    "height": actual_height,
                    "num_frames": actual_frames,
                    "fps": fps,
                    "duration_s": actual_frames / float(fps),
                    "has_audio": has_audio,
                    "conditioning": conditioning,
                    "steps": steps,
                    "guidance": guidance,
                    "flow_shift": flow_shift,
                    # sd.cpp pins the audio schedule, so the recipe records what it actually ran.
                    "audio_flow_shift": state.family.default_audio_flow_shift,
                }
            finally:
                output_path.unlink(missing_ok = True)

    @staticmethod
    def _encode_mp4(
        video_frames,
        fps: int,
        audio,
        pipe,
        *,
        audio_sample_rate: Optional[int] = None,
    ) -> bytes:
        """Encode frames (+ optional audio) to H.264 MP4 bytes via diffusers' PyAV
        exporter. A temp file bridges the exporter's path-based API; the bytes are
        what the gallery persists."""
        from diffusers.utils.export_utils import encode_video

        tmp = tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False)
        tmp.close()
        try:
            encode_kwargs: dict[str, Any] = {}
            if audio is not None and pipe is not None:
                encode_kwargs["audio"] = audio
                sample_rate = audio_sample_rate or getattr(
                    getattr(getattr(pipe, "vocoder", None), "config", None),
                    "output_sampling_rate",
                    None,
                )
                if sample_rate:
                    encode_kwargs["audio_sample_rate"] = int(sample_rate)
            encode_video(video_frames, fps, tmp.name, **encode_kwargs)
            return Path(tmp.name).read_bytes()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def generate_progress(self) -> dict[str, Any]:
        with self._lock:
            gen = dict(self._gen)
            # generate() swaps in a bare {"active": False} before the worker records the terminal dict; report active across that gap.
            if self._generate_job_active:
                gen["active"] = True
        gen.setdefault("active", False)
        # Mirror the image endpoint field names (total_steps / fraction) alongside the native "total": the two generate-progress APIs used to disagree.
        total = int(gen.get("total") or 0)
        step = int(gen.get("step") or 0)
        gen["total_steps"] = total
        gen["fraction"] = min(1.0, step / total) if total > 0 else 0.0
        return gen

    def forget_terminal_video(self, video_id: Optional[str] = None) -> bool:
        """Drop the completed terminal record when its clip leaves the gallery.

        ``generate_progress()`` keeps the last completed job until the next one starts, and the
        Video page merges that record on mount to cover a job that finished after the gallery
        fetch. Once the clip is deleted the record points at a file that is gone, so every refresh
        resurrected a ghost card whose file request 404s until the backend restarts. ``video_id``
        None forgets whatever completed record is held (the clear-all case). A live job is left
        alone: its own terminal swap comes later."""
        with self._lock:
            gen = self._gen
            if self._generate_job_active or not isinstance(gen, dict):
                return False
            if gen.get("phase") != "completed":
                return False
            held = str((gen.get("video") or {}).get("id") or "")
            if video_id is not None and held != str(video_id):
                return False
            self._gen = {"active": False}
            return True

    def cancel_generate(self) -> bool:
        """Signal the in-flight generation to stop at its next step callback."""
        with self._lock:
            cancel = self._active_generate_cancel
            if cancel is None:
                return False
            cancel.set()
            return True

    # ── teardown + status ────────────────────────────────────────────────────

    def _teardown_state_locked(self) -> None:
        """Free the committed state. The caller holds _generate_lock (no generation
        in flight) AND _lock, so the whole teardown is one atomic step: a generation
        cannot observe a half-torn-down backend."""
        state, self._state = self._state, None
        if state is not None:
            restore_backend_flags(state.backend_flags)
            # A GGUF load may have installed the compiled GGUF dequantizer; restore the stock kernels so a later speed=off load is bit-identical.
            from . import diffusion_gguf_compile

            diffusion_gguf_compile.uninstall_all()
            del state
            clear_gpu_cache()

    def unload(self) -> dict[str, Any]:
        with self._lock:
            self._load_token += 1
            self._cancel_event.set()
            self._loading = None
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            # Fence generations queued behind the active one too: they hold no cancel event, so the signal cannot reach them.
            self._teardown_waiters += 1
        # Barrier: wait for the signalled generation to exit before freeing the pipeline, else we report the VRAM free while
        # the clip still holds it. Teardown runs INSIDE the barrier: releasing first would hand out one last clip.
        with self._generate_lock:
            with self._lock:
                try:
                    self._teardown_state_locked()
                finally:
                    # Released in a finally: _teardown_state_locked ends in clear_gpu_cache(), which raises on a sticky CUDA fault, and an un-drained fence refuses every later generation.
                    self._teardown_waiters -= 1
        logger.info("video.unloaded")
        return self.status()

    def status(self) -> dict[str, Any]:
        state = self._state
        if state is None:
            return {
                "loaded": False,
                "repo_id": None,
                "family": None,
                "base_repo": None,
                "device": None,
                "dtype": None,
                "model_kind": None,
                "engine": None,
                "gguf_variant": None,
                "offload_policy": None,
                "vae_tiling": False,
                "memory_mode": None,
                "speed_mode": None,
                "speed_optims": [],
                "attention_backend": None,
                "transformer_cache": None,
                "transformer_quant": None,
                "text_encoder_quant": None,
                "has_audio": False,
                "supports_cfg": True,
                "supports_keyframes": False,
                "supports_references": False,
                "h3_task": None,
                "defaults": None,
                "resolved": None,
            }
        from hub.utils.gguf import extract_quant_token

        fam = state.family
        default_steps, default_guidance = default_video_generation_params(
            state.gguf_filename,
            state.repo_id,
            state.base_repo,
            fallback = (fam.default_steps, fam.default_guidance),
        )
        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": fam.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": state.dtype,
            "model_kind": state.kind,
            "engine": state.engine,
            # As on the image runtime: dtype is compute precision, which labelled a Q4_K_M BF16.
            "gguf_variant": (
                extract_quant_token(state.gguf_filename)
                if state.kind == "gguf" and state.gguf_filename
                else None
            ),
            "offload_policy": state.offload_policy,
            "vae_tiling": state.vae_tiling,
            "memory_mode": state.memory_mode,
            "speed_mode": state.speed_mode,
            "speed_optims": list(state.speed_optims),
            "attention_backend": state.attention_backend,
            "transformer_cache": state.transformer_cache,
            "transformer_quant": state.transformer_quant,
            "text_encoder_quant": state.text_encoder_quant,
            "has_audio": fam.has_audio,
            "supports_cfg": fam.supports_cfg,
            # Expose only conditioning supported by the resident partition.
            "supports_keyframes": fam.supports_keyframes and state.h3_task != H3_TASK_REFERENCES,
            "supports_references": fam.supports_references and state.h3_task == H3_TASK_REFERENCES,
            "h3_task": state.h3_task,
            "defaults": {
                "steps": default_steps,
                "guidance": default_guidance,
                "num_frames": fam.default_num_frames,
                "fps": fam.default_fps,
                "frame_step": fam.frame_step,
                "frame_offset": fam.frame_offset,
                "duration_presets": list(fam.duration_presets),
                "resolution_multiple": fam.resolution_multiple,
                "resolution_presets": [list(p) for p in fam.resolution_presets],
                # Let the frontend preview the backend-owned "match source" rule.
                "canvas_short_edge": H3_CANVAS_SHORT_EDGE if fam.supports_keyframes else None,
                "canvas_max_pixels": H3_CANVAS_MAX_PIXELS if fam.supports_keyframes else None,
                "flow_shift": fam.default_flow_shift,
                "audio_flow_shift": fam.default_audio_flow_shift,
                # sd.cpp has no flag for the audio shift, so the control is only real on Diffusers.
                "supports_audio_flow_shift": fam.default_audio_flow_shift is not None
                and state.engine != "sd_cpp",
            },
            "resolved": state.resolved,
        }


_backend: Optional[VideoBackend] = None
_backend_lock = threading.Lock()


def get_video_backend() -> VideoBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = VideoBackend()
        return _backend

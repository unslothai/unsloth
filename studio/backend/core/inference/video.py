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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

from .diffusion_attention import (
    apply_attention_backend,
    install_hunyuan_attention_trim,
    select_attention_backend,
)
from .diffusion_cache import (
    FBCACHE_MIN_STEPS,
    TC_AUTO,
    TC_MAGCACHE,
    apply_step_cache,
    auto_cache_mode,
    auto_cache_quality,
    maybe_toggle_step_cache,
    normalize_cache_quality,
    normalize_transformer_cache,
)
from .diffusion_cache import _disengage_step_cache
from .diffusion_cfg_parallel import (
    maybe_enable_cfg_parallel,
    normalize_cfg_parallel,
    teardown_cfg_parallel,
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
from . import diffusion_compile_cache as compile_cache
from .diffusion_speed import (
    SPEED_DEFAULT,
    SPEED_MAX,
    SPEED_OFF,
    apply_speed_optims,
    compile_eligible,
    resolve_speed_mode,
    restore_backend_flags,
    snapshot_backend_flags,
)
from .diffusion_auto_policy import _QUANT_STEADY_FACTOR, build_resolved_record
from .diffusion_transformer_quant import (
    TQ_AUTO,
    dense_transformer_supported,
    is_int8_memory_fallback,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)
from .diffusion_precision import TE_QUANT_AUTO, normalize_te_quant, quantize_text_encoders
from .diffusion_vae_quant import VAE_QUANT_AUTO, normalize_vae_quant, quantize_vae
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
)
from utils.hardware import clear_gpu_cache

logger = get_logger(__name__)

# Load kinds, mirroring the image backend: "gguf" (single-file GGUF DiT +
# companion base repo), "single_file" (safetensors DiT, e.g. the fp8 LTX-2.3
# checkpoints), "pipeline" (a full diffusers repo via from_pretrained).
_MODEL_KINDS = frozenset({"gguf", "single_file", "pipeline"})

# Official vendor base repos allowed to load as full (non-GGUF) artifacts even
# though they are not under unsloth/. Exact-match, lowercased, safetensors-only,
# no remote code -- same bar as the image backend's allowlist.
_TRUSTED_NON_GGUF_VIDEO_REPOS = frozenset(
    {
        "lightricks/ltx-2",
        "lightricks/ltx-2.3",
        "lightricks/ltx-2.3-fp8",
        # Wan2.2 official diffusers base repos (Wan-AI org): safetensors-only, no
        # remote code, so allowed as full (pipeline-kind) loads like the LTX-2 bases.
        "wan-ai/wan2.2-ti2v-5b-diffusers",
        "wan-ai/wan2.2-t2v-a14b-diffusers",
        # HunyuanVideo-1.5 community Diffusers repacks (tencent's own repo is the
        # original non-diffusers layout: config.json, no model_index.json, so it
        # cannot load through HunyuanVideo15Pipeline at all).
        "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-480p_t2v",
        "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-720p_t2v",
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
            # Not a local dir: resolve a cached HUB blob from the HF cache (no network). The
            # cached-gguf picker only offers already-downloaded repos, so the blob is on disk --
            # but that listing scans the active, legacy, AND default cache roots, so probe all
            # three here or a GGUF cached in a non-active root would be offered yet 400 on load.
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


# ── post-load compile prewarm ─────────────────────────────────────────────────
# vLLM and SGLang guarantee that ALL compilation finishes at server startup
# (dummy batches through every compiled shape) so no request ever pays a compile
# mid-serving. The video backend's analogue: after a compiled-tier load commits,
# a background thread runs one tiny throwaway generation so the first REAL
# request starts at steady state. Measured through the real backend
# (HunyuanVideo-1.5-480p, B200, 480x288/17f/30 steps, temp/vs_warm_probe.py):
# with a warm Mega-cache bundle the first-generation extra drops 11.3 s -> 2.1 s
# (a 9.0 s background warmup absorbs the dynamo tracing + cudnn autotune the
# bundle cannot carry); on a cold start the full ~54 s compile moves off the
# user's first request entirely. Steady state is untouched (the warmup adds no
# lasting state: cache residuals are reset and the real generation re-seeds its
# own generator). The shape is deliberately tiny -- the default tier compiles
# dynamic=True, so one small trace serves every later resolution.
_PREWARM_ENV = "UNSLOTH_DIFFUSION_COMPILE_PREWARM"
_PREWARM_WIDTH = 192
_PREWARM_HEIGHT = 128
_PREWARM_FRAMES = 9
_PREWARM_STEPS = 2


def compile_prewarm_decision(
    fam: VideoFamily,
    *,
    speed_mode: str,
    speed_optims: tuple,
    offload_policy: str,
    cfg_parallel_active: bool,
) -> tuple[bool, str]:
    """Whether the post-load background compile prewarm should run, with the
    resolved-record reason. Pure so it unit-tests without the runtime."""
    if (os.environ.get(_PREWARM_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False, f"disabled via {_PREWARM_ENV}"
    if "compiled" not in speed_optims:
        return False, "not applicable (no regional compile engaged; nothing to prewarm)"
    if speed_mode != SPEED_DEFAULT:
        # speed=max compiles dynamic=False: a graph is keyed to the exact shape,
        # so a tiny warmup shape would compile a graph the user's request never
        # runs and the real shape would still pay its own compile.
        return (
            False,
            "skipped: speed=max compiles static per-shape graphs a warmup shape cannot serve",
        )
    if not bool(getattr(fam, "supports_compile_prewarm", True)):
        return False, "family opted out (supports_compile_prewarm=False)"
    if offload_policy != "none":
        # Offload wraps block forwards in disabled onload hooks (the compiled-inner
        # arming skips them) and every warmup forward would stream the full DiT
        # over PCIe -- all cost, none of the measured warmup benefit.
        return (
            False,
            "skipped: offload streams weights per forward; the warmup would only churn transfers",
        )
    if cfg_parallel_active:
        # The CFG-parallel proxy serialises compile-sensitive runs through its own
        # per-(shape, steps, cache) planner; an unplanned warmup would bypass it.
        return (
            False,
            "skipped: CFG parallel plans compile-sensitive runs through its own dispatcher",
        )
    return True, (
        "background warmup generation absorbs the first-generation compile hitch "
        "(measured 11.3s -> 2.1s warm-cache, ~54s cold, HunyuanVideo-1.5-480p on B200)"
    )


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
        # The picker admits a GGUF (local dir OR cached hub repo) by its general.architecture, but
        # its path/name may carry no whole-segment family token (e.g. a renamed "model.gguf"), so
        # the name-based detection above misses it. Resolve the same family the picker offered by
        # reading the arch -- its string ("ltxv") is a family alias. A video arch with no backend
        # family (e.g. "wan") still yields None, so an unsupported pick 400s exactly as before.
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
    gguf_filename: Optional[str] = None
    offload_policy: str = "none"
    vae_tiling: bool = True
    memory_mode: str = "auto"
    speed_mode: str = SPEED_OFF
    speed_optims: tuple = ()
    backend_flags: Optional[dict] = None
    attention_backend: Optional[str] = None
    transformer_cache: Optional[str] = None
    # True when the cache decision was AUTO on a cache-capable DiT: generate() then
    # re-checks the actual step count and toggles FBCache across FBCACHE_MIN_STEPS.
    # An explicit request (off / fbcache) is never toggled.
    cache_auto: bool = False
    # Inputs the generation-time toggle re-applies (quantised threshold + override).
    cache_quant_active: bool = False
    cache_threshold: Optional[float] = None
    # The resolved cache quality preset ("quality" | "balanced" | "fast"); the toggle
    # re-applies it so a mid-session re-engage keeps the requested preset.
    cache_quality: Optional[str] = None
    # Dense transformer quant actually engaged ("int8" | "fp8" | "nvfp4" | "mxfp8") or
    # None. Mirrors the image backend's _LoadState.transformer_quant: on a pipeline-kind
    # load the dense DiT(s) can be torchao-quantised in place onto the low-precision
    # tensor cores; None means they run at their loaded (bf16) precision.
    transformer_quant: Optional[str] = None
    # Text-encoder quant actually engaged ("fp8" | "fp8_dynamic" | "int8" | "nvfp4") or None.
    # The companion text encoder (UMT5 / Gemma3 / Qwen2.5-VL) loads dense bf16 and is often the
    # largest resident component; this shrinks it in place, mirroring the image backend.
    text_encoder_quant: Optional[str] = None
    # VAE quant actually engaged ("fp8" layerwise | "fp8_dynamic" torchao conv) or None. The
    # convolutional decoder (Conv2d/Conv3d) shrinks in place; the vae_force_fp32 families
    # (Wan) never quantise (force_fp32 -> dense). Mirrors the image backend's _LoadState.
    vae_quant: Optional[str] = None
    # Dual-GPU CFG branch parallelism: "on" when a DiT replica on a second CUDA device
    # runs the pred_cond branch (bit-identical under the family's auto step cache;
    # ~1.7x e2e). The handle is the installed proxy -- generate() plans each run's
    # dispatch on it and _teardown_state frees the replica through it.
    cfg_parallel: Optional[str] = None
    cfg_parallel_handle: Any = None
    # Pre-warmed torch.compile cache context (diffusion_compile_cache.CacheContext) when a
    # compiled tier ran begin(); generate() persists the bundle after the first compiled
    # generation (when saving is enabled) and _teardown_state restores the inductor dir.
    compile_cache_ctx: Any = None
    resolved: Optional[dict] = None


@dataclass
class _VideoLoadingState:
    repo_id: str
    base_repo: str
    expected_bytes: Optional[int] = None
    error: Optional[str] = None


def _progress(phase: Optional[str], **extra: Any) -> dict[str, Any]:
    return {"phase": phase, **extra}


# ── dual-DiT (Wan2.2-A14B MoE) helpers ────────────────────────────────────────
#
# The imported optimisation helpers (apply_speed_optims / apply_attention_backend /
# apply_step_cache) and the dense quantiser all read ``pipe.transformer`` and act on
# that ONE denoiser -- correct for every single-DiT family (LTX-2, Wan2.2-TI2V-5B).
# Wan2.2-A14B is a dual-expert MoE: ``transformer`` handles the high-noise steps and
# ``transformer_2`` the low-noise steps (pipeline_wan.py routes by boundary_ratio), so
# an optimisation applied only to ``transformer`` would leave the second expert eager /
# unquantised / on the wrong attention kernel for half the schedule. Rather than fork
# each helper, present the second DiT to them AS ``pipe.transformer`` via a thin proxy
# and call the helper a second time, so the helpers stay untouched and single-DiT loads
# are bit-identical (the proxy is only built for is_moe families).


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
        # Store on the instance dict under a name __getattr__ never fires for.
        object.__setattr__(self, "_pipe", pipe)

    @property
    def transformer(self) -> Any:
        return self._pipe.transformer_2

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the instance/class (i.e. not
        # ``transformer`` / ``_pipe``), so everything else delegates to the real pipe.
        return getattr(object.__getattribute__(self, "_pipe"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Writes must land on the real pipe, or a helper's side effect (for example
        # reassigning the transformer it optimised) would vanish with the view.
        # ``transformer`` mirrors the read property onto the second expert.
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
        # Generation progress, written by the step callback / phase transitions.
        self._gen: dict[str, Any] = {"active": False}
        # True from begin_generate() until its worker records a terminal state, so
        # a second begin_generate() is refused while the first still runs (or is
        # about to run: generate() only sets _gen after taking its locks).
        self._generate_job_active = False
        # The post-load background compile prewarm thread (None until a compiled
        # load spawns one); kept for tests/diagnostics, never joined on the hot path.
        self._prewarm_thread: Optional[threading.Thread] = None
        # The prewarm's cancel event, set (in addition to _active_generate_cancel)
        # only while the prewarm runs. Real generations signal it on entry so they
        # preempt the warmup at its next step boundary instead of queueing behind
        # it; unlike _active_generate_cancel it can never point at a real job.
        self._prewarm_cancel: Optional[threading.Event] = None

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
        vae_quant: Optional[str] = None,
    ) -> VideoFamily:
        """Cheap, network-free validation shared by the route and the load path."""
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        # A -GGUF repo picked without a quant filename resolves to the pipeline
        # kind and would only fail minutes later in from_pretrained (no
        # model_index.json), AFTER the route evicted the current GPU owner.
        # Reject it here, where failing is still free.
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
        if kind != "gguf" and not _is_trusted_video_repo(repo_id):
            raise ValueError(
                f"Non-GGUF video loads are limited to unsloth/* repos, the official "
                f"family base repos, and local paths; '{repo_id}' is neither."
            )
        # The companions load with from_pretrained too, so an explicit base repo is
        # held to the same bar as a non-GGUF repo id: a GGUF pick must not smuggle
        # in an arbitrary remote base.
        if base_repo and (base_repo or "").strip() and not _is_trusted_video_repo(base_repo):
            raise ValueError(
                f"base_repo is limited to unsloth/* repos, the official family base "
                f"repos, and local paths; '{base_repo}' is neither."
            )
        # An existing LOCAL base_repo loads as a full pipeline (from_pretrained(base) / config=base),
        # which needs a model_index.json. The pipeline-kind shape check below covers only repo_id,
        # and an explicit base_repo is only meaningful for gguf/single_file kinds, so a non-pipeline
        # local base would otherwise pass here and fail deep in the background load AFTER the route
        # evicted the resident model. Shared helper, so image/video/training stay in sync.
        from core.inference.diffusion import _assert_local_base_is_pipeline

        _assert_local_base_is_pipeline(base_repo)
        if kind in ("gguf", "single_file") and not gguf_filename:
            raise ValueError("A gguf/single_file load needs the checkpoint filename.")
        if kind in ("gguf", "single_file") and fam.is_moe:
            # A single checkpoint carries only one expert; the pipeline would then pull
            # the other expert dense bf16 from the base repo, outside the memory plan.
            raise ValueError(
                f"'{fam.name}' is a dual-expert model: a single {kind} file covers only "
                f"one of its two transformers. Load the diffusers pipeline repo "
                f"('{fam.base_repo}') instead."
            )
        # A local checkpoint that cannot exist must fail HERE, before the route evicts
        # a resident chat/image model for a load that dies at resolve time.
        if kind in ("gguf", "single_file"):
            # Fail a kind/extension mismatch before the GPU handoff instead of deep in the
            # background loader: a "gguf" load needs a .gguf file, a "single_file" load must not be
            # handed a .gguf and must name an actual .safetensors checkpoint. Mirrors the image
            # loader's kind/extension gate in diffusion.validate_load_request.
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
            # POSIX path-shaped, a "."/".." prefix (covers ./ ../ and Windows .\ ..\), a Windows
            # separator anywhere (never in a bare "org/name" id), or an absolute path on this OS
            # (covers Windows C:\ / C:/). Mirrors the image loader so a missing Windows-shaped
            # local pick fails before the GPU handoff instead of being treated as a Hub repo.
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
                # The loader hands a local FILE straight to the gguf/single_file loader
                # (_resolve_checkpoint_path returns the file itself, ignoring gguf_filename),
                # so the file's OWN suffix must match the kind. Otherwise a .gguf picked as
                # single_file (or a .safetensors picked as gguf) slips past the gguf_filename
                # checks above, evicts the resident model in the route, and only then fails
                # in from_single_file / the GGUF reader. Reject it here, before the handoff.
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
        # A local pipeline pick must be a real diffusers directory (model_index.json), or it
        # would only fail deep in from_pretrained AFTER the route evicted the resident model.
        # Mirrors the image loader's local-pipeline shape check in diffusion.validate_load_request.
        if kind == "pipeline":
            root = Path(repo_id).expanduser()
            # Gate on .exists() (not .is_dir()) so a local FILE picked as a pipeline is rejected
            # too: a bare .safetensors file is not a diffusers directory, so from_pretrained would
            # still fail in the background load after the eviction. Mirrors the image loader, which
            # uses .exists() here.
            if root.exists() and not (root.is_dir() and (root / "model_index.json").is_file()):
                raise ValueError(
                    f"Local pipeline path is not a diffusers directory "
                    f"(no model_index.json): {repo_id}"
                )
        # Reject a malformed transformer_quant scheme cheaply, before the GPU handoff
        # (normalize_transformer_quant raises ValueError on an unknown scheme). It applies
        # only on pipeline-kind loads (the dense DiT from the base repo); an ignored value
        # on a gguf/single_file load is left to the loader, matching the image backend.
        normalize_transformer_quant(transformer_quant)
        # Reject a malformed text_encoder_quant the same way (applies to any load kind: the dense
        # text encoder is resident for pipeline / gguf / single_file alike).
        normalize_te_quant(text_encoder_quant)
        # Same for vae_quant (the dense VAE is resident for every load kind).
        normalize_vae_quant(vae_quant)
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
        transformer_cache_quality: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        vae_quant: Optional[str] = None,
        cfg_parallel: Optional[str] = None,
        model_kind: Optional[str] = None,
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
            vae_quant = vae_quant,
        )
        with self._lock:
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A video load is already in progress.")
            self._load_token += 1
            token = self._load_token
            self._cancel_event.clear()
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
                transformer_cache_quality = transformer_cache_quality,
                transformer_quant = transformer_quant,
                text_encoder_quant = text_encoder_quant,
                vae_quant = vae_quant,
                cfg_parallel = cfg_parallel,
                model_kind = model_kind,
                _load_token = token,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        try:
            fam = _detect_load_family(
                kwargs["repo_id"], kwargs.get("gguf_filename"), kwargs.get("family_override")
            )
            kind = resolve_video_model_kind(kwargs.get("gguf_filename"), kwargs.get("model_kind"))
            base = (
                kwargs["repo_id"]
                if kind == "pipeline"
                else resolve_video_base_repo(fam, kwargs.get("base_repo"))
            )
            kwargs["base_repo"] = base
            expected = self._estimate_download_bytes(
                kwargs["repo_id"], kwargs.get("gguf_filename"), base, kwargs.get("hf_token"), kind
            )
            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.base_repo = base
                    self._loading.expected_bytes = expected
            # The GGUF/single-file checkpoint downloads outside the lock so an
            # unload/eviction can preempt the multi-GB pull; the pipeline
            # companions pre-download the same way (scoped file list, cancellable,
            # resumes from the cache so a cancelled pull costs nothing).
            checkpoint_local: Optional[Path] = None
            if kwargs.get("gguf_filename") and not Path(kwargs["repo_id"]).expanduser().exists():
                from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
                checkpoint_local = Path(
                    hf_hub_download_with_xet_fallback(
                        kwargs["repo_id"],
                        kwargs["gguf_filename"],
                        kwargs.get("hf_token"),
                        cancel_event = self._cancel_event,
                    )
                )
            # An LTX-2.3 checkpoint replaces the base VAEs/vocoder/connectors too, so
            # its base pull shrinks to scheduler + text encoder + tokenizer; the
            # estimate is recomputed to match (detectable only once the checkpoint
            # header is on disk, hence after the pull above).
            ltx23 = False
            if fam is not None and fam.name == "ltx-2" and kind != "pipeline":
                from .video_ltx2 import is_ltx23_checkpoint

                probe = checkpoint_local
                if probe is None:
                    # Local repos: a bare file, or a directory whose child the same
                    # resolver load_pipeline uses picks out. Unresolvable here means
                    # load_pipeline will surface the real error; keep the wide pull.
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
                    )
                    with self._lock:
                        if self._load_token == token and self._loading is not None:
                            self._loading.expected_bytes = expected
            base_local = self._predownload_base(base, kwargs.get("hf_token"), kind, ltx23 = ltx23)
            # The 2.3 assembly pulls per component from the hub id (its snapshot here
            # deliberately lacks the base VAEs), so it only gets the warmed cache; the
            # generic from_pretrained paths get the complete local snapshot.
            kwargs["_base_local_dir"] = None if ltx23 else base_local
            self.load_pipeline(**kwargs)
            with self._lock:
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            # A failed or cancelled load never commits _VideoLoadState, so the
            # teardown path has no snapshot to restore: roll back the process-wide
            # speed globals here (token-scoped, so a superseded load cannot clobber
            # the globals a newer in-flight load now owns).
            self._rollback_precommit_globals(token)
            self._rollback_precommit_cfg_parallel(token)
            self._rollback_precommit_compile_cache(token)
            if self._load_token != token:
                return
            logger.error("video.load_failed: %s", exc)
            # Free the debris of a failed construction (mirrors diffusion.py's _run_load):
            # no _VideoLoadState was committed, so no later unload releases the VRAM a
            # partially built pipeline (OOM in from_pretrained / quant / placement) left
            # reserved in the caching allocator -- which would OOM the next load. Guarded so
            # a sticky CUDA error cannot skip stamping the real error below.
            try:
                clear_gpu_cache()
            except Exception:  # noqa: BLE001 -- cleanup is best-effort
                pass
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))

    def _rollback_precommit_globals(self, token: Optional[int]) -> None:
        """Restore process-wide speed globals (cudnn.benchmark / TF32 / the compiled
        GGUF dequantizer) for a load that died BEFORE committing _VideoLoadState.
        _teardown_state only restores from the committed state's snapshot, so an
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

    def _rollback_precommit_cfg_parallel(self, token: Optional[int]) -> None:
        """Tear down a CFG-parallel proxy installed by a load that died BEFORE
        committing _VideoLoadState. The proxy owns a daemon worker, the DiT
        replica's VRAM, and possibly the process-global cuDNN attention patch;
        with no committed state, _teardown_state would never reach it. Token-scoped
        exactly like _rollback_precommit_globals."""
        stored = getattr(self, "_precommit_cfg_parallel", None)
        if stored is None:
            return
        stored_token, pipe, proxy = stored
        if token is not None and stored_token is not None and stored_token != token:
            return
        self._precommit_cfg_parallel = None
        teardown_cfg_parallel(pipe, proxy, logger = logger)

    def _rollback_precommit_compile_cache(self, token: Optional[int]) -> None:
        """Restore TORCHINDUCTOR_CACHE_DIR for a load that ran ``compile_cache.begin``
        but died BEFORE committing _VideoLoadState (the committed path restores via
        _teardown_state instead). Token-scoped exactly like _rollback_precommit_globals."""
        stored = getattr(self, "_precommit_compile_cache", None)
        if stored is None:
            return
        stored_token, ctx = stored
        if token is not None and stored_token is not None and stored_token != token:
            return
        self._precommit_compile_cache = None
        compile_cache.restore(ctx)

    # Base-repo subfolders an LTX-2.3 assembly reads: the checkpoint (plus the GGUF
    # repo's extras files) supplies the DiT, connectors, both VAEs and the vocoder,
    # so only the 2.0 base's scheduler / text encoder / tokenizer are pulled.
    _LTX23_BASE_PREFIXES = ("scheduler/", "text_encoder/", "tokenizer/")

    @staticmethod
    def _base_download_files(
        info: Any,
        kind: str,
        *,
        ltx23: bool = False,
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
          checkpoint and its extras, not the 2.0 base."""
        files: list[tuple[str, int]] = []
        for sibling in info.siblings or []:
            name, size = sibling.rfilename, sibling.size or 0
            # .jinja: tokenizer/chat_template.jinja ships as a standalone file in the
            # LTX-2 and HunyuanVideo-1.5 repos (not embedded in tokenizer_config.json)
            # and apply_chat_template needs it at generation time, so a snapshot
            # without it loads fine and then crashes the first generation.
            if not name.endswith((".safetensors", ".json", ".model", ".txt", ".jinja")):
                continue
            if "/" not in name and name.endswith(".safetensors"):
                continue
            if kind != "pipeline" and name.startswith("transformer/"):
                continue
            if name.startswith("text_encoder/diffusion_pytorch_model"):
                continue
            if ltx23 and "/" in name and not name.startswith(VideoBackend._LTX23_BASE_PREFIXES):
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
    ) -> Optional[int]:
        """Total bytes this load will pull (checkpoint + companions), or None."""
        try:
            from huggingface_hub import HfApi

            total = 0
            api = HfApi(token = hf_token or None)
            if gguf_filename and not Path(repo_id).expanduser().exists():
                info = api.model_info(repo_id, files_metadata = True)
                for sibling in info.siblings or []:
                    if sibling.rfilename == gguf_filename and sibling.size:
                        total += int(sibling.size)
            if base and not Path(base).expanduser().exists():
                info = api.model_info(base, files_metadata = True)
                total += sum(size for _, size in self._base_download_files(info, kind, ltx23 = ltx23))
            return total or None
        except Exception:  # noqa: BLE001 -- progress totals are best-effort only
            return None

    def _predownload_base(
        self,
        base: str,
        hf_token: Optional[str],
        kind: str,
        *,
        ltx23: bool = False,
    ) -> Optional[str]:
        """Pull exactly the base-repo files the load needs; return the local snapshot dir.

        A bare ``from_pretrained(repo_id)`` snapshot of Lightricks/LTX-2 downloads the
        whole 314 GB repo (root packaged checkpoints plus a second 50 GB text-encoder
        shard set) when ~93 GB is used. Downloading the scoped file list ourselves is
        also cancellable per file, and handing the local dir to from_pretrained skips
        diffusers' own expected-files sweep. None -> caller keeps the hub id (local
        path, non-diffusers layout, or any metadata failure: from_pretrained then
        resolves the repo exactly as before)."""
        try:
            if not base or Path(base).expanduser().exists():
                return None
            from huggingface_hub import HfApi

            info = HfApi(token = hf_token or None).model_info(base, files_metadata = True)
            files = self._base_download_files(info, kind, ltx23 = ltx23)
            if not any(name == "model_index.json" for name, _ in files):
                return None
            from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

            snapshot_root: Optional[Path] = None
            for name, _ in files:
                # Explicit per-file check: a fully-cached file returns without ever
                # consulting the event, so a warm-cache sweep would otherwise run to
                # completion after an unload already cancelled this load.
                if self._cancel_event.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)
                local = Path(
                    hf_hub_download_with_xet_fallback(
                        base, name, hf_token, cancel_event = self._cancel_event
                    )
                )
                if name == "model_index.json":
                    snapshot_root = local.parent
            return str(snapshot_root) if snapshot_root is not None else None
        except Exception as exc:  # noqa: BLE001 -- fall back to from_pretrained's own pull
            if self._cancel_event.is_set():
                raise
            logger.warning("video.predownload_fallback: %s", exc)
            return None

    def _cache_bytes(self, repo_id: Optional[str]) -> int:
        """Bytes of ``repo_id`` currently in the HF blob cache (progress polling)."""
        if not repo_id:
            return 0
        try:
            from huggingface_hub import scan_cache_dir
            rid = repo_id.strip()
            for repo in scan_cache_dir().repos:
                if repo.repo_id == rid:
                    return int(repo.size_on_disk)
        except Exception:  # noqa: BLE001 -- cache scan is best-effort
            return 0
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
        if expected and downloaded >= expected:
            phase = "finalizing"
            # The cache scan counts every blob of the repo(s), including files a
            # previous (or broader) pull left behind that this load never reads, so
            # the raw counter can exceed the scoped estimate. Clamp: everything the
            # load needs is present, which is what the bar reports.
            downloaded = expected
        return _progress(
            phase,
            downloaded_bytes = int(downloaded),
            expected_bytes = int(expected) if expected else None,
        )

    def loading_repo_ids(self) -> tuple[str, ...]:
        """Repo ids an in-flight background load is downloading (empty when idle).

        The delete-cached guard needs this: during a load ``status()["loaded"]`` is
        still False, but deleting the target repo (or its companion base) would yank
        blobs and snapshot files from under the download/assembly. Mirrors the image
        backend's guard (DiffusionBackend.loading_repo_ids)."""
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            return tuple(r for r in (loading.repo_id, loading.base_repo) if r)

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
        transformer_cache_quality: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        vae_quant: Optional[str] = None,
        cfg_parallel: Optional[str] = None,
        model_kind: Optional[str] = None,
        _load_token: Optional[int] = None,
        _base_local_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        import diffusers
        import torch

        fam = self.validate_load_request(
            repo_id,
            gguf_filename = gguf_filename,
            base_repo = base_repo,
            family_override = family_override,
            model_kind = model_kind,
            transformer_quant = transformer_quant,
            text_encoder_quant = text_encoder_quant,
            vae_quant = vae_quant,
        )
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        # An explicit Speed="off" (bit-exact reference) load pins the companions dense too, mirroring
        # the transformer_quant default below: promoting an UNSET TE/VAE to auto-quant would silently
        # fp8/int8 the text encoder + VAE and break the bit-exact request (an auto DEFAULT overriding
        # the EXPLICIT off control). Only an EXPLICIT off suppresses; an unset speed still auto
        # -quantises, and an explicit companion scheme still forces it.
        speed_off = speed_mode is not None and str(speed_mode).strip().lower() == SPEED_OFF
        # text_encoder_quant tri-state (mirrors the image backend + transformer_quant): UNSET
        # (None / "") or "auto" -> auto (pick the best accurate TE scheme for this GPU + family);
        # an explicit "none"/"off" pins the encoder dense; a concrete scheme forces it. So the
        # shipped default is auto. auto is backend-owned, so an explicit "auto" also goes dense
        # under Speed="off", matching transformer_quant; only a concrete scheme forces quant off.
        if text_encoder_quant is None or str(text_encoder_quant).strip().lower() in ("", "auto"):
            text_encoder_quant = "off" if speed_off else TE_QUANT_AUTO
        # vae_quant tri-state, same contract (UNSET or "auto" -> auto). The vae_force_fp32 families
        # (Wan) keep the VAE dense regardless (quantize_vae's force_fp32 gate), so auto is safe as
        # the shipped default.
        if vae_quant is None or str(vae_quant).strip().lower() in ("", "auto"):
            vae_quant = "off" if speed_off else VAE_QUANT_AUTO
        base = repo_id if kind == "pipeline" else resolve_video_base_repo(fam, base_repo)

        with self._lock:
            if _load_token is not None and _load_token != self._load_token:
                raise RuntimeError("Video load was cancelled or superseded.")
            # Signal only a generation from the PREVIOUS model; the token check
            # above already bailed a superseded worker before this point.
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        # Wait for the signalled generation to actually exit before tearing the old
        # pipeline down: the denoise loop holds its own pipe reference until the
        # next step callback, and freeing/reallocating under it would put two
        # models in VRAM at once. generate() holds _generate_lock for its full
        # body, so a bare acquire is the exit barrier (never while holding _lock).
        with self._generate_lock:
            pass
        # The barrier wait can outlive this load: an unload or a newer load may
        # have superseded it while blocked, and tearing down now would destroy
        # the model that should remain current (or waste minutes building a
        # pipeline nobody wants). Recheck before touching shared state.
        if _load_token is not None and _load_token != self._load_token:
            raise RuntimeError("Video load was cancelled or superseded.")
        self._teardown_state()

        target = resolve_diffusion_device_target()
        device = target.device
        # Video DiTs are bf16-native; fp16 overflows them, so a resolved fp16
        # promotes to float32 (the same rule as the fp16-incompatible image
        # families). CPU stays float32.
        dtype = target.dtype
        if fam.fp16_incompatible and dtype is torch.float16:
            dtype = torch.float32
        # The size tables below are bf16 (2-byte) figures. When the promotion
        # above lands fp32 weights on an accelerator (a pre-bf16 GPU), every
        # dense estimate doubles; budgeting the 2-byte figure would let auto
        # pick a resident plan that OOMs inside from_pretrained. GGUF weights
        # stay quantised on disk and in memory, so only dense estimates scale.
        dtype_scale = 2.0 if device != "cpu" and dtype is torch.float32 else 1.0

        # Precision tri-state, mirroring the image backend: an UNSET request (or
        # "auto") hands the decision to the hardware ladder -- on a dense-capable
        # GPU the quantised DiT (int8 minimum, fp8 on data-center silicon) is
        # faster at the same resident-or-better footprint. An explicit
        # "none"/"off" pins dense bf16 and an explicit scheme pins that scheme.
        # Only the pipeline kind can engage it (gguf/single_file checkpoints
        # already carry their own precision), and the offload guard below still
        # skips it when the plan moves the DiT.
        if transformer_quant is None or str(transformer_quant).strip().lower() in (
            "",
            "auto",
        ):
            # An explicit Speed="off" (bit-exact) load must stay dense bf16: promoting the unset
            # precision to auto-quant here would engage int8/fp8 + regional compile and silently
            # break the user's bit-exact request (an auto DEFAULT overriding an EXPLICIT control),
            # and the quant path below would then also force effective_speed back to default.
            # Suppress the auto default when speed was explicitly pinned off (speed_off, computed
            # above with the companions), mirroring the image backend (diffusion.py); otherwise auto
            # (the dense-capable default) applies. "off" normalizes to None (no dense quant).
            transformer_quant = "off" if speed_off else TQ_AUTO

        # ── memory plan: family-table resident estimate + frames-aware headroom.
        device_memory = snapshot_device_memory(target)
        components = fam.bf16_components_gb
        mib_per_gb = 1000.0**3 / (1024.0 * 1024.0)
        if kind == "pipeline":
            model_dense_mib = (
                int(sum(components) * mib_per_gb * dtype_scale) if components is not None else None
            )
            companion_mib = None
        else:
            checkpoint_path = self._resolve_checkpoint_path(repo_id, gguf_filename, hf_token)
            size_mib = file_size_mib(str(checkpoint_path))
            model_dense_mib = None
            if kind == "gguf":
                transformer_mib = estimate_gguf_resident_mib(size_mib)
            else:
                transformer_mib = estimate_safetensors_dense_mib(size_mib)
                if transformer_mib is not None:
                    transformer_mib = int(transformer_mib * dtype_scale)
            companion_mib = (
                int((components[1] + components[2]) * mib_per_gb * dtype_scale)
                if components is not None
                else None
            )
            # The resident check budgets ALL weights (the image backend's contract):
            # the companions stay resident even when only the transformer would fit,
            # so budgeting the transformer alone lets auto pick OFFLOAD_NONE and OOM
            # while from_pretrained loads the text encoder / VAEs.
            model_dense_mib = (
                transformer_mib + (companion_mib or 0) if transformer_mib is not None else None
            )
        runtime_mib = estimate_video_runtime_mib(
            width = fam.resolution_presets[0][0],
            height = fam.resolution_presets[0][1],
            num_frames = fam.default_num_frames,
        )
        plan = plan_diffusion_memory(
            target = target,
            device_memory = device_memory,
            model_dense_mib = model_dense_mib,
            runtime_headroom_mib = runtime_mib,
            companion_dense_mib = companion_mib,
            requested_mode = normalize_memory_mode(memory_mode),
        )
        # Parity with the image dense-quant path: the bf16-table plan can force offload
        # a quantised DiT would not need (offload also disables quant entirely). Re-plan
        # with the scheme's steady factor and keep the resident placement when it fits;
        # if quantisation later fails, the load falls back to this bf16 plan.
        bf16_plan = plan
        quant_replanned = False
        if (
            kind == "pipeline"
            and plan.offload_policy != "none"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
            and components is not None
        ):
            scheme_preview = select_transformer_quant_scheme(
                target, transformer_quant, family = fam.name
            )
            factor = _QUANT_STEADY_FACTOR.get(scheme_preview) if scheme_preview else None
            if factor is not None:
                quant_mib = int(
                    (components[0] * factor + components[1] + components[2]) * mib_per_gb
                )
                replanned = plan_diffusion_memory(
                    target = target,
                    device_memory = device_memory,
                    model_dense_mib = quant_mib,
                    runtime_headroom_mib = runtime_mib,
                    companion_dense_mib = None,
                    requested_mode = normalize_memory_mode(memory_mode),
                )
                if replanned.offload_policy == "none":
                    logger.info(
                        "video.transformer_quant: %s fits resident (%d MiB steady); "
                        "dropping the bf16 plan's '%s' offload",
                        scheme_preview,
                        quant_mib,
                        plan.offload_policy,
                    )
                    plan = replanned
                    quant_replanned = True

        # ── build the pipeline.
        pipeline_cls = getattr(diffusers, fam.pipeline_class)
        pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if getattr(fam, "vae_force_fp32", False):
            # Wan's VAE must decode in float32, but a scalar torch_dtype casts EVERY component
            # (VAE included) to the pipe dtype during load -- AutoencoderKLWan has no
            # _keep_in_fp32_modules, so from_pretrained truncates its fp32 weights to bf16 and a
            # later .to(float32) only widens the already-lossy values (banding / black frames).
            # diffusers >= 0.39 takes a per-component dtype dict, so load the VAE at fp32 directly;
            # "default" MUST be set or unlisted components fall back to fp32 (over-widening the DiT).
            pipe_kwargs["torch_dtype"] = {"vae": torch.float32, "default": dtype}
        if hf_token:
            pipe_kwargs["token"] = hf_token
        if kind == "pipeline":
            # The pre-downloaded snapshot dir keeps from_pretrained off the hub (its
            # own snapshot sweep would also pull the repo's packaged root checkpoints
            # and duplicate text-encoder shards); hub id when pre-download was skipped.
            pipe = pipeline_cls.from_pretrained(_base_local_dir or repo_id, **pipe_kwargs)
        else:
            transformer_cls = getattr(diffusers, fam.transformer_class)
            # checkpoint_path was already resolved (and downloaded) by the memory
            # planning branch above for every non-pipeline kind.
            sf_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "config": base,
                "subfolder": "transformer",
                "token": hf_token,
            }
            if kind == "gguf":
                sf_kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(
                    compute_dtype = dtype
                )
            from .video_ltx2 import is_ltx23_checkpoint, load_ltx23_pipeline

            if fam.name == "ltx-2" and is_ltx23_checkpoint(checkpoint_path):
                # 2.3 checkpoints need the full assembly: new transformer config
                # flags, key renames the stock converter lacks, and the 2.3
                # connectors/VAEs/vocoder the 2.0 base repo does not carry.
                pipe = load_ltx23_pipeline(
                    checkpoint_path,
                    base_repo = base,
                    torch_dtype = dtype,
                    is_gguf = kind == "gguf",
                    hf_token = hf_token,
                )
            else:
                transformer = transformer_cls.from_single_file(str(checkpoint_path), **sf_kwargs)
                pipe = pipeline_cls.from_pretrained(
                    _base_local_dir or base, transformer = transformer, **pipe_kwargs
                )

        # The per-component torch_dtype above already loads the Wan VAE at float32 (bf16_components_gb
        # budgets it at that fp32 size, so the memory plan stays consistent). Belt-and-suspenders for
        # any path that bypassed the dict (e.g. a passed-in vae=): re-pin an fp32-force VAE that came
        # back at a lower precision. This is a no-op on the primary path (the load already fp32'd it).
        if getattr(fam, "vae_force_fp32", False):
            vae = getattr(pipe, "vae", None)
            if vae is not None and getattr(vae, "dtype", None) is not torch.float32:
                vae.to(torch.float32)

        if _load_token is not None and _load_token != self._load_token:
            del pipe
            clear_gpu_cache()
            raise RuntimeError("Video load was cancelled or superseded.")

        # For a dual-DiT MoE family (Wan2.2-A14B), every optimisation site below must
        # cover BOTH experts: ``views`` is (pipe, _SecondDiTView(pipe)) so a helper that
        # reads ``pipe.transformer`` runs once per denoiser. A single-DiT load resolves to
        # (pipe,), so it behaves exactly as before.
        views = _views_for(pipe, fam)

        # ── dense transformer quant (opt-in, pipeline-kind only): load the dense bf16
        # DiT from the base repo and torchao-quantise it in place onto the low-precision
        # tensor cores, mirroring the image backend's transformer_quant fast path. Only
        # the pipeline kind materialises the dense weights (gguf/single_file already carry
        # their own precision), and only on CUDA + bf16. Best-effort: any failure leaves
        # the DiT dense. Quant must precede compile (dynamic quant is ~30x slower eager),
        # so it runs before apply_speed_optims below -- same order as diffusion.py.
        transformer_quant_engaged: Optional[str] = None
        quant_skipped_for_offload = False
        # Auto-quant lands on int8 for fp8-denied families (HunyuanVideo-1.5); measured on a B200,
        # int8 is ~7% slower AND less accurate than the dense bf16 + regional-compile path (fp8, the
        # only quant that also speeds up, is black-framed there). int8's sole benefit is memory, so
        # when the DENSE DiT already fits resident (bf16_plan has no offload) prefer dense and skip
        # the auto-quant. Only for an AUTO request (explicit int8/fp8 honored), only where int8 is a
        # denied fallback on a data-center fp8-capable GPU (is_int8_memory_fallback excludes consumer
        # / Ampere / fp8 families), and only when dense provably fits -- so no new OOM risk and the
        # fp8 families (Wan / LTX) still quantise for their speed win.
        quant_skipped_for_dense = (
            kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) == TQ_AUTO
            and dense_transformer_supported(target)
            and bf16_plan.offload_policy == "none"
            and is_int8_memory_fallback(target, fam.name)
        )
        if quant_skipped_for_dense:
            logger.info(
                "video.transformer_quant: skipped -- dense DiT fits resident and int8 (the fp8-denied "
                "fallback for '%s') is slower + less accurate than dense+compile here; run dense",
                fam.name,
            )
        elif (
            kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
            and plan.offload_policy != "none"
        ):
            # Offload hooks move modules with Module.to(), which torchao quantized
            # tensors reject (aten._has_compatible_shallow_copy_type is
            # unimplemented) -- observed as a hard crash on the Wan2.2-A14B gate
            # run, where the 114 GB dual DiT plans model offload. A dense DiT
            # under offload beats a crashed one, so quant is skipped, surfaced in
            # the resolved record, and the user can force it by pinning a
            # resident memory mode.
            logger.info(
                "video.transformer_quant: skipped (offload policy '%s' moves the "
                "DiT via Module.to(), unsupported for torchao quantized tensors); "
                "pin a resident memory mode to combine quant with this model",
                plan.offload_policy,
            )
            quant_skipped_for_offload = True
        elif (
            kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
        ):
            engaged = []
            for view in views:
                # quantize_transformer reads ``pipe.transformer`` and returns the scheme it
                # engaged (or None); pass each expert's view so both DiTs are quantised with
                # the same arch-chosen scheme. The family name drives the per-family deny
                # table (_FAMILY_SCHEME_DENY) exactly as on the image side.
                scheme = quantize_transformer(
                    view,
                    target,
                    mode = transformer_quant,
                    family = fam.name,
                    logger = logger,
                )
                if scheme is not None:
                    engaged.append(scheme)
            # Quant must engage on every DiT or none: the first expert is mutated in
            # place, so a second-expert failure cannot fall back to dense (the schedule
            # would run at mismatched precision with quant reported off). Fail the load
            # cleanly instead; a full miss (nothing engaged) stays best-effort dense.
            if engaged and len(engaged) < len(views):
                del pipe
                clear_gpu_cache()
                raise RuntimeError(
                    f"transformer_quant={engaged[0]} engaged on only "
                    f"{len(engaged)}/{len(views)} experts; retry without quant."
                )
            if engaged:
                transformer_quant_engaged = engaged[0]
        # The quant-sized plan is only valid when quant actually engaged; a dense
        # fallback must keep the conservative bf16 placement.
        if quant_replanned and transformer_quant_engaged is None:
            plan = bf16_plan

        # ── dense text-encoder quant (opt-in): the DiT arrives quantised in a GGUF, but the
        # companion encoder (Gemma3 / UMT5 / Qwen2.5-VL) loads dense bf16 from the base repo and
        # is often the largest resident component. Quantise it in place, mirroring the image
        # backend (diffusion.py): applied for every kind (the encoder is dense regardless of how
        # the DiT was sourced) and before placement so the offload hooks move the smaller weights.
        # Best-effort: quantize_text_encoders leaves any encoder it can't cast dense. int8 needs a
        # per-family keep-bf16 schedule, so the family name is passed.
        text_encoder_quant_engaged = quantize_text_encoders(
            pipe,
            target,
            mode = text_encoder_quant,
            family = fam.name,
            offload_active = plan.offload_policy != "none",
            logger = logger,
        )
        # Quantise the dense convolutional VAE (opt-in fp8 layerwise / fp8_dynamic torchao conv).
        # The vae_force_fp32 families (Wan) run the VAE in fp32 for numerical stability, so
        # force_fp32 pins them dense (quantising the fp32 VAE bands the decode); auto skips the
        # torchao mode under offload. Best-effort: a failure leaves the VAE dense.
        vae_quant_engaged = quantize_vae(
            pipe,
            target,
            mode = vae_quant,
            family = fam.name,
            offload_active = plan.offload_policy != "none",
            force_fp32 = getattr(fam, "vae_force_fp32", False),
            logger = logger,
        )

        # ── optimisation layers, in the image backend's order: step cache FIRST
        # (compile keys its fullgraph decision off an active cache: FBCache hooks
        # graph-break, so compiling fullgraph before installing the cache crashes
        # the first cached generation), then attention, the speed profile, and
        # placement/offload last.
        # A clip denoise runs minutes, so even a dense (non-GGUF) load amortises the
        # one-time regional compile within a single generation: unset resolves to the
        # near-lossless `default` profile for every kind. Explicit values (incl.
        # "off") are honored verbatim, and `max` is never an auto choice.
        effective_speed = resolve_speed_mode(
            speed_mode, is_gguf = kind == "gguf", dense_default = SPEED_DEFAULT
        )
        # A torchao-quantised DiT must be compiled (eager dynamic quant is ~30x slower and
        # would lose to the bf16 it replaced), so force at least the regional-compile
        # profile when quant engaged and the effective speed was off, matching diffusion.py.
        if transformer_quant_engaged is not None and effective_speed == SPEED_OFF:
            logger.info(
                "video.transformer_quant: forcing speed_mode=default "
                "(quantized transformer must be compiled; eager is ~30x slower)"
            )
            effective_speed = SPEED_DEFAULT
        backend_flags = snapshot_backend_flags()
        # Until the state commit below transfers ownership to _teardown_state, a
        # failure or cancellation must restore these process-wide globals itself
        # (_run_load's error handler calls _rollback_precommit_globals with this
        # token). Registered BEFORE the first mutating call.
        self._precommit_globals = (_load_token, backend_flags)
        # Step cache tri-state, mirroring the image backend: unset / "auto" lets the
        # step-count policy decide (engage when this model's DEFAULT schedule reaches
        # FBCACHE_MIN_STEPS, re-checked against the actual step count per generation);
        # explicit "off" / "fbcache" are pinned and never toggled. Run it per expert
        # so both denoisers cache; the engaged mode is identical across experts.
        cache_request = normalize_transformer_cache(transformer_cache)
        cache_auto = transformer_cache is None or cache_request == TC_AUTO
        # Cache quality preset tri-state: unset / "auto" -> the family's measured
        # default (the near-lossless "quality" preset for HunyuanVideo-1.5, "balanced"
        # -- the pre-knob behaviour -- elsewhere); an explicit preset is honored
        # verbatim. Resolved here so the generation-time toggle re-applies it.
        cache_quality_requested = normalize_cache_quality(transformer_cache_quality)
        cache_quality = cache_quality_requested or auto_cache_quality(fam.name)
        # Validate the dual-GPU CFG request cheaply here; the gate itself must run
        # after placement (it keys on the post-plan device layout + free VRAM).
        normalize_cfg_parallel(cfg_parallel)
        # GGUF checkpoints and torchao-quantised DiTs both need the higher quantised
        # threshold for the cache to still trigger over the quant noise.
        cache_quant_active = kind == "gguf" or transformer_quant_engaged is not None
        # Computed for every request: the auto step-count policy keys on it, and an
        # explicit magcache request needs it too (the ratio curve is interpolated over
        # the configured step count).
        default_cache_steps, _ = default_video_generation_params(gguf_filename, repo_id, base)
        if cache_auto:
            # The engaged CACHE MODE is per-family: MagCache where FBCache's uncapped
            # skipping derails the trajectory (HunyuanVideo-1.5), FBCache elsewhere.
            cache_request = (
                auto_cache_mode(fam.name) if default_cache_steps >= FBCACHE_MIN_STEPS else None
            )
        cache_engaged = None
        # Each view is zipped with the pipe attribute it exposes as ``transformer`` (the
        # expert-view iteration contract): a dual-expert MoE's second view passes
        # expert="transformer_2" so MagCache resolves THAT expert's calibrated curve --
        # the experts split the schedule at the boundary timestep, so their curves differ.
        for view, expert_name in zip(views, _transformer_names(pipe, fam)):
            engaged = apply_step_cache(
                view,
                mode = cache_request,
                threshold = transformer_cache_threshold,
                # A quantized transformer's block residuals are larger, so it needs the
                # higher FBCache trigger threshold to cache at all. Mirror the image path
                # (diffusion.py): both an engaged transformer_quant AND a GGUF checkpoint
                # (quantized weights) count as quant-active here (cache_quant_active, L1172).
                quant_active = cache_quant_active,
                family = fam.name,
                steps = default_cache_steps,
                quality = cache_quality,
                expert = expert_name,
                logger = logger,
            )
            if view is pipe:
                cache_engaged = engaged
        # The auto decision can flip at generation time, but only on a DiT that
        # supports caching at all (a non-CacheMixin transformer can never engage).
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
        attention_trim_engaged = False
        speed_optims: tuple = ()
        # A dense torchao transformer on the pipeline path is not a GGUF one, so is_gguf
        # keys off the load kind (gguf) AND no quant having engaged.
        gguf_transformer = kind == "gguf" and transformer_quant_engaged is None
        for view in views:
            # apply_attention_backend acts on ``view.transformer``; calling it once per
            # view sets the kernel on each expert. The engaged values match across
            # experts (same device/family/mode), so record the first pass.
            # HunyuanVideo-1.5 only: drop the ~99% zero-padded text tokens from the joint
            # attention so it runs the fused (cuDNN/flash) SDPA kernel instead of the dense-mask
            # fallback (~18x/DiT-forward at 121 frames, cosine ~1.0). Must precede the backend set
            # so the requested kernel pins onto the new processors. No-op for every other family.
            # A speed lever like the attention backend below, so honor an explicit Speed="off" (the
            # bit-exact reference path keeps the stock dense-mask attention).
            trim = (
                install_hunyuan_attention_trim(view, fam, logger = logger)
                if effective_speed != SPEED_OFF
                else False
            )
            engaged = apply_attention_backend(
                view,
                select_attention_backend(
                    target, attention_backend, speed_active = effective_speed != SPEED_OFF
                ),
                logger = logger,
            )
            if view is pipe:
                attention_engaged = engaged
                attention_trim_engaged = trim
        # Pre-warmed torch.compile cache (Mega-cache), mirroring the image backend: when a
        # compiled tier will run, point inductor at a per-fingerprint dir and load a matching
        # bundle BEFORE the first compiled forward. Measured on HunyuanVideo-1.5-480p (B200):
        # the first-generation compile extra drops 107.5 s -> 13.8 s from a 12.8 MB bundle
        # (fresh inductor dir), and the persistent per-key dir alone recovers a restart to
        # 11.7 s -- with the stock /tmp inductor dir that ~100 s is repaid after every reboot.
        # A miss is silent -> local compile, exactly as before. Must run AFTER the attention
        # backend set (the fingerprint keys on the engaged kernel) and BEFORE
        # apply_speed_optims (whose compile the loaded artifacts serve).
        compile_ctx = None
        if effective_speed in (SPEED_DEFAULT, SPEED_MAX) and compile_eligible(
            target, is_gguf = gguf_transformer, family = fam
        ):
            compile_ctx = compile_cache.begin(
                family = fam.name,
                transformer = getattr(pipe, "transformer", None),
                dtype = getattr(target, "dtype", None),
                quant = transformer_quant_engaged,
                attention_backend = attention_engaged,
                compile_kwargs = {
                    # Mirrors apply_speed_optims' fullgraph decision: an active step cache
                    # (or one that may still toggle on) OR a planned offload graph-breaks,
                    # so the cached bundle must be keyed on the same fullgraph setting.
                    "fullgraph": cache_engaged is None
                    and not cache_may_toggle
                    and plan.offload_policy == "none",
                    "dynamic": effective_speed != SPEED_MAX,
                    "mode": "max-autotune-no-cudagraphs"
                    if effective_speed == SPEED_MAX
                    else "default",
                },
                logger = logger,
            )
            # Until the state commit below transfers ownership to _teardown_state, a
            # failed or cancelled load must restore TORCHINDUCTOR_CACHE_DIR itself
            # (_run_load's error handler, token-scoped like the globals).
            self._precommit_compile_cache = (_load_token, compile_ctx)
        for view in views:
            applied = apply_speed_optims(
                view,
                target,
                is_gguf = gguf_transformer,
                family = fam,
                speed_mode = effective_speed,
                # An auto cache that could still engage mid-session also drops
                # fullgraph: enabling FBCache under a fullgraph-compiled DiT would
                # crash the first cached generation.
                cache_active = cache_engaged is not None or cache_may_toggle,
                offload_active = plan.offload_policy != "none",
            )
            if view is pipe:
                speed_optims = tuple(k for k, v in applied.items() if v) + (
                    ("hunyuan_attn_trim",) if attention_trim_engaged else ()
                )
        with self._generate_lock:
            # A cancelled/superseded load must not place weights on the GPU the arbiter
            # may already have handed to another backend; recheck right before placement
            # (the commit below still does the final locked check).
            if _load_token is not None and _load_token != self._load_token:
                del pipe
                clear_gpu_cache()
                raise RuntimeError("Video load was cancelled or superseded.")
            offload_policy, vae_tiling = apply_memory_plan(pipe, plan, device = device, logger = logger)
            # A dual-DiT MoE pipe (Wan2.2-A14B) needs no extra per-expert offload pass here:
            # apply_memory_plan's group tier (_apply_group_offload) already block-streams every
            # DiT it finds on the pipe -- transformer AND transformer_2 -- and model/sequential
            # offload hook every top-level module, so the second expert is covered under all tiers.
            # A second _apply_group_offload on transformer_2 would re-register the group-offload
            # hooks it already carries, which diffusers rejects with a duplicate-hook ValueError.
            if not vae_tiling:
                # Decode of a whole clip is the video memory peak; tiling is near-free
                # in quality and keeps the decode bounded, so it is always on.
                try:
                    pipe.vae.enable_tiling()
                    vae_tiling = True
                except Exception as exc:  # noqa: BLE001 -- tiling is an optimisation only
                    logger.warning("video.vae_tiling_failed: %s", exc)

            # ── dual-GPU CFG branch parallelism (auto on the measured families, else
            # opt-in). AFTER placement so the memory plan stays single-device: the DiT
            # replica lives entirely on a SECOND CUDA device and is gated on that
            # device's free VRAM; a single-GPU host, an offload plan, or a quantised
            # DiT all fall through to today's single-device path with the reason
            # surfaced in the resolved record.
            cfg_parallel_proxy, cfg_parallel_reason = maybe_enable_cfg_parallel(
                pipe,
                fam,
                requested = cfg_parallel,
                kind = kind,
                transformer_source = _base_local_dir or base,
                hf_token = hf_token,
                dtype = dtype,
                quant_engaged = transformer_quant_engaged,
                offload_active = offload_policy != "none",
                compiled = "compiled" in speed_optims,
                attention_backend = attention_engaged,
                speed_active = effective_speed != SPEED_OFF,
                logger = logger,
            )
            if cfg_parallel_proxy is not None:
                # Until the _VideoLoadState commit below, the proxy (daemon worker,
                # DiT replica VRAM, process-global cuDNN patch) is owned by this
                # stash: a cancellation or failure between install and commit is
                # torn down by _run_load's error handler via
                # _rollback_precommit_cfg_parallel, token-scoped like the globals.
                self._precommit_cfg_parallel = (_load_token, pipe, cfg_parallel_proxy)
            if cfg_parallel_proxy is not None and cache_engaged:
                # The load engaged the step cache on the primary BEFORE the proxy
                # existed; re-engage THROUGH the proxy so the replica carries the same
                # hooks and each branch's cache state matches the single-GPU run
                # exactly (the bit-identity precondition). Cheap: hook install only.
                _disengage_step_cache(
                    cfg_parallel_proxy._primary,
                    reason = "re-engaging through the cfg-parallel proxy",
                    logger = logger,
                )
                cache_engaged = apply_step_cache(
                    pipe,
                    mode = cache_request,
                    threshold = transformer_cache_threshold,
                    quant_active = cache_quant_active,
                    family = fam.name,
                    steps = default_cache_steps,
                    quality = cache_quality,
                    logger = logger,
                )

            prewarm_on, prewarm_reason = compile_prewarm_decision(
                fam,
                speed_mode = effective_speed,
                speed_optims = speed_optims,
                offload_policy = offload_policy,
                cfg_parallel_active = cfg_parallel_proxy is not None,
            )
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
                    "attention_trim": (
                        None,
                        "on" if attention_trim_engaged else "off",
                        "HunyuanVideo-1.5: padded text tokens dropped so joint attention runs "
                        "the fused SDPA kernel (~18x per DiT forward, cosine ~1.0)"
                        if attention_trim_engaged
                        else "not applicable (non-Hunyuan family)",
                    ),
                    "transformer_cache": (
                        None if cache_auto else transformer_cache,
                        cache_engaged or "off",
                        cache_reason,
                    ),
                    "transformer_cache_quality": (
                        transformer_cache_quality,
                        cache_quality,
                        "requested speed/accuracy preset (threshold + skip budget)"
                        if cache_quality_requested is not None
                        else "auto: the family's measured preset (near-lossless "
                        "'quality' for HunyuanVideo-1.5, 'balanced' elsewhere)",
                    ),
                    "cfg_parallel": (
                        cfg_parallel,
                        "on" if cfg_parallel_proxy is not None else "off",
                        cfg_parallel_reason,
                    ),
                    "transformer_quant": (
                        transformer_quant,
                        transformer_quant_engaged or "off",
                        "dense DiT(s) torchao-quantised onto the low-precision tensor cores"
                        if transformer_quant_engaged is not None
                        else (
                            "skipped: offload moves the DiT, unsupported for torchao "
                            "tensors; pin a resident memory mode to combine them"
                            if quant_skipped_for_offload
                            else "skipped: dense DiT fits resident and int8 (fp8-denied fallback) "
                            "is slower + less accurate than dense+compile here"
                            if quant_skipped_for_dense
                            else "not engaged (dense bf16 DiT loaded)"
                        ),
                    ),
                    "text_encoder_quant": (
                        text_encoder_quant,
                        text_encoder_quant_engaged or "off",
                        "dense text encoder quantised in place"
                        if text_encoder_quant_engaged is not None
                        else "not engaged (dense bf16 text encoder loaded)",
                    ),
                    "vae_quant": (
                        vae_quant,
                        vae_quant_engaged or "off",
                        "dense VAE quantised in place"
                        if vae_quant_engaged is not None
                        else "not engaged (fp32 VAE family / offload / disabled -> dense)"
                        if getattr(fam, "vae_force_fp32", False)
                        else "not engaged (dense VAE loaded)",
                    ),
                    "compile_prewarm": (
                        None,
                        "on" if prewarm_on else "off",
                        prewarm_reason,
                    ),
                }
            )

            with self._lock:
                if _load_token is not None and _load_token != self._load_token:
                    # The pre-commit CFG-parallel stash (torn down by _run_load's
                    # error handler) still references the pipe, so the proxy, its
                    # daemon worker, the replica's VRAM, and the cuDNN patch are
                    # all rolled back even though no state was committed.
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
                    # Already filtered above to only the optimisations that engaged;
                    # apply_speed_optims returns every flag True/False and the view
                    # loop keeps just the True names.
                    speed_optims = speed_optims,
                    backend_flags = backend_flags,
                    attention_backend = attention_engaged,
                    transformer_cache = cache_engaged,
                    cache_auto = cache_may_toggle,
                    cache_quant_active = cache_quant_active,
                    cache_threshold = transformer_cache_threshold,
                    cache_quality = cache_quality,
                    transformer_quant = transformer_quant_engaged,
                    text_encoder_quant = text_encoder_quant_engaged,
                    vae_quant = vae_quant_engaged,
                    cfg_parallel = "on" if cfg_parallel_proxy is not None else None,
                    cfg_parallel_handle = cfg_parallel_proxy,
                    compile_cache_ctx = compile_ctx,
                    resolved = resolved,
                )
                # Ownership of the globals, the CFG-parallel proxy, and the compile
                # cache context transferred to _state / _teardown_state.
                self._precommit_globals = None
                self._precommit_cfg_parallel = None
                self._precommit_compile_cache = None
        logger.info(
            "video.loaded: %s (%s, %s, offload=%s, speed=%s, quant=%s)",
            repo_id,
            fam.name,
            kind,
            offload_policy,
            effective_speed,
            transformer_quant_engaged or "off",
        )
        if prewarm_on:
            # Post-commit so a real request is never blocked behind an uncommitted
            # load; the thread re-checks the token and yields to any generation
            # that arrived first (which then pays -- and absorbs -- the warmup
            # itself, exactly the pre-prewarm behaviour).
            self._prewarm_thread = threading.Thread(
                target = self._compile_prewarm,
                args = (_load_token,),
                daemon = True,
                name = "video-compile-prewarm",
            )
            self._prewarm_thread.start()
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

    # ── post-load compile prewarm ─────────────────────────────────────────────

    def _compile_prewarm(self, token: Optional[int]) -> None:
        """Run one tiny throwaway generation so the first REAL request skips the
        compile/trace warmup (see the module-level prewarm constants for the
        measured numbers). Runs on a daemon thread under ``_generate_lock``,
        registers itself as the active cancellable job (so unload / a new load /
        cancel_generate can abort it at a step boundary, exactly like a real
        generation), and never touches the user-visible ``_gen`` progress. A
        failure or cancellation is logged and the load simply keeps the
        pre-prewarm behaviour: the first real generation pays the warmup."""
        import torch

        cancel = threading.Event()
        with self._generate_lock:
            with self._lock:
                state = self._state
                if state is None or (token is not None and token != self._load_token):
                    return  # superseded/unloaded before the warmup could start
                if self._generate_job_active or self._gen.get("active"):
                    return  # a real request beat us; it absorbs the warmup itself
                self._active_generate_cancel = cancel
                self._prewarm_cancel = cancel
            started = time.monotonic()
            try:
                pipe = state.pipe
                fam = state.family
                width, height = snap_video_size(fam, _PREWARM_WIDTH, _PREWARM_HEIGHT)
                frames = snap_num_frames(fam, _PREWARM_FRAMES)
                call_params = inspect.signature(pipe.__call__).parameters
                kwargs: dict[str, Any] = {
                    "prompt": "warmup",
                    "num_inference_steps": _PREWARM_STEPS,
                    "width": width,
                    "height": height,
                    "num_frames": frames,
                    "generator": torch.Generator(device = state.device).manual_seed(0),
                }
                # Default guidance keeps the CFG branch structure of a real run
                # (both guider branches trace), mirroring generate().
                if fam.guidance_via_guider:
                    pipe.guider.guidance_scale = float(fam.default_guidance)
                else:
                    kwargs[fam.cfg_kwarg] = float(fam.default_guidance)
                if "frame_rate" in call_params:
                    kwargs["frame_rate"] = float(fam.default_fps)

                def _on_step(p, step_index, timestep, callback_kwargs):
                    if cancel.is_set():
                        p._interrupt = True
                    return callback_kwargs

                def _on_scheduler_step(done: int) -> None:
                    if cancel.is_set():
                        raise _VideoGenerationCancelled()

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step
                    progress_ctx = contextlib.nullcontext()
                else:
                    progress_ctx = _scheduler_step_progress(pipe, _on_scheduler_step)
                with torch.inference_mode(), progress_ctx:
                    pipe(**kwargs)
                if cancel.is_set():
                    raise _VideoGenerationCancelled()
                # Drop the warmup's step-cache residuals so the next real
                # generation starts from the same state as an unwarmed load
                # (generate() also resets per request; this is defence in depth).
                if state.transformer_cache:
                    self._reset_step_cache(pipe)
                # The warmup just paid the compile: persist the Mega-cache bundle
                # now (env-gated, idempotent) instead of waiting for the first
                # real generation, mirroring generate()'s save point.
                try:
                    compile_cache.save(state.compile_cache_ctx, logger = logger)
                except Exception:  # noqa: BLE001 -- cache persistence is best-effort
                    pass
                logger.info(
                    "video.compile_prewarm: warmup absorbed the compile hitch in %.1fs "
                    "(%dx%d, %d frames, %d steps)",
                    time.monotonic() - started,
                    width,
                    height,
                    frames,
                    _PREWARM_STEPS,
                )
            except _VideoGenerationCancelled:
                logger.info("video.compile_prewarm: cancelled (unload / new load / user cancel)")
            except Exception as exc:  # noqa: BLE001 -- warmup is best-effort, never fatal
                logger.warning(
                    "video.compile_prewarm: failed (%s); the first generation pays the "
                    "compile warmup instead",
                    exc,
                )
            finally:
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                    if self._prewarm_cancel is cancel:
                        self._prewarm_cancel = None

    # ── generation ───────────────────────────────────────────────────────────

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
        with self._lock:
            if self._state is None:
                raise RuntimeError(VIDEO_NOT_LOADED_MSG)
            if self._generate_job_active:
                raise RuntimeError(VIDEO_GENERATION_BUSY_MSG)
            # A background compile prewarm may hold _generate_lock. Signal its
            # dedicated cancel handle BEFORE registering ours so the real job
            # preempts the warmup at its next step boundary instead of queueing
            # behind the full warmup (which also left unload/cancel pointing at
            # the wrong event once _active_generate_cancel was overwritten below).
            if self._prewarm_cancel is not None:
                self._prewarm_cancel.set()
            self._generate_job_active = True
            # Register the cancel event BEFORE the worker starts so a cancel (or an
            # unload) that lands in the spawn window still stops the run instead of
            # returning "nothing to cancel".
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
                    "model": result["repo_id"],
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
                # generate() clears its own registration; this covers a job whose
                # worker failed before (or without) reaching generate()'s finally.
                # Identity-guarded so a direct generate() that registered its own
                # event in the meantime keeps its cancel handle.
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
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        import torch

        # begin_generate passes the event it already registered (so a cancel in the
        # spawn window is honoured); a direct call makes its own.
        cancel = cancel_event if cancel_event is not None else threading.Event()
        if cancel_event is None:
            # Direct callers skip begin_generate's preemption, so signal a
            # running compile prewarm here too rather than queueing behind it.
            with self._lock:
                if self._prewarm_cancel is not None:
                    self._prewarm_cancel.set()
        with self._generate_lock:
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(VIDEO_NOT_LOADED_MSG)
                self._active_generate_cancel = cancel
            try:
                fam = state.family
                width, height = snap_video_size(
                    fam,
                    width or fam.resolution_presets[0][0],
                    height or fam.resolution_presets[0][1],
                )
                frames = snap_num_frames(fam, num_frames or fam.default_num_frames)
                out_fps = int(fps or fam.default_fps)
                default_steps, default_guidance = default_video_generation_params(
                    state.gguf_filename,
                    state.repo_id,
                    state.base_repo,
                    fallback = (fam.default_steps, fam.default_guidance),
                )
                steps = int(steps or default_steps)
                guidance = float(default_guidance if guidance is None else guidance)

                generator = torch.Generator(device = state.device)
                if seed is None:
                    seed = int(generator.seed()) % (2**53)
                generator = generator.manual_seed(int(seed))

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
                if fam.guidance_via_guider:
                    # HunyuanVideo-1.5: __call__ has no guidance kwarg at all; the
                    # CFG scale is a plain attribute on the pipeline's guider
                    # component, set per request. Near-1 scales auto-disable CFG
                    # inside the guider itself (_is_cfg_enabled's is_close check).
                    pipe.guider.guidance_scale = float(guidance)
                else:
                    kwargs[fam.cfg_kwarg] = guidance
                if negative_prompt and "negative_prompt" in call_params:
                    kwargs["negative_prompt"] = negative_prompt
                # LTX-2 takes frame_rate (it shapes the audio track length); other
                # pipelines fix their own rate and fps only matters at export.
                if "frame_rate" in call_params:
                    kwargs["frame_rate"] = float(out_fps)
                # Dual-DiT MoE (Wan2.2-A14B): the low-noise expert (transformer_2) has its
                # own guidance kwarg (cfg2_kwarg = "guidance_scale_2"). Thread it only when
                # the loaded family declares one AND the pipeline signature accepts it (the
                # same inspect.signature gate frame_rate uses), so a single-DiT pipeline is
                # never handed a kwarg its check_inputs would reject. WanPipeline raises if
                # guidance_scale_2 is passed to a pipeline with boundary_ratio=None
                # (pipeline_wan.py:322), so the gate is BOTH the family flag and the
                # signature: TI2V-5B has no cfg2_kwarg, so it never reaches here. A None
                # request lets the pipeline default it (to guidance_scale) itself.
                if fam.cfg2_kwarg and fam.cfg2_kwarg in call_params and guidance_2 is not None:
                    kwargs[fam.cfg2_kwarg] = float(guidance_2)

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
                    # No cooperative _interrupt here: without a callback the pipeline
                    # never checks it, so cancellation must unwind the denoise loop
                    # via an exception (mapped to the cancelled sentinel below).
                    if cancel.is_set():
                        raise _VideoGenerationCancelled()
                    _tick(done)

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step
                    progress_ctx = contextlib.nullcontext()
                else:
                    # HunyuanVideo-1.5 has no step callback; every scheduler.step
                    # call is exactly one denoise step, so wrap it for progress +
                    # cancel and restore it afterwards.
                    progress_ctx = _scheduler_step_progress(pipe, _on_scheduler_step)

                # An AUTO cache decision is re-checked against the ACTUAL step count,
                # mirroring the image backend: a many-step request gains FBCache even
                # when the load's default schedule kept it off, and a few-step request
                # drops it. Explicit choices never toggle. Runs per view so a dual-DiT
                # MoE toggles both experts.
                if state.cache_auto:
                    toggled = state.transformer_cache
                    for view, expert_name in zip(
                        _views_for(pipe, fam), _transformer_names(pipe, fam)
                    ):
                        toggled = maybe_toggle_step_cache(
                            view,
                            steps = steps,
                            quant_active = state.cache_quant_active,
                            threshold = state.cache_threshold,
                            mode = auto_cache_mode(fam.name),
                            family = fam.name,
                            quality = state.cache_quality,
                            expert = expert_name,
                            logger = logger,
                        )
                    if toggled != state.transformer_cache:
                        # _VideoLoadState is frozen (loads swap it as one unit); this
                        # tracks the pipe-level toggle that already happened so
                        # status() reports the true cache state.
                        object.__setattr__(state, "transformer_cache", toggled)
                        entry = (state.resolved or {}).get("transformer_cache")
                        if isinstance(entry, dict):
                            entry["value"] = toggled or "off"
                            entry["reason"] = (
                                f"auto: {steps}-step generation "
                                + ("reaches" if toggled else "is below")
                                + f" {FBCACHE_MIN_STEPS}"
                            )
                elif state.transformer_cache == TC_MAGCACHE:
                    # An EXPLICIT magcache choice never toggles off, but its ratio
                    # curve, retention window, and skip budget are interpolated over
                    # the CONFIGURED step count: a clip at a different step count
                    # re-engages (marker carries "#s{steps}") so skips stay aligned
                    # with the actual schedule. The user's on choice is preserved --
                    # this only re-sizes the already-engaged cache.
                    for view, expert_name in zip(
                        _views_for(pipe, fam), _transformer_names(pipe, fam)
                    ):
                        transformer = getattr(view, "transformer", None)
                        marker = getattr(transformer, "_unsloth_step_cache", None)
                        # endswith, not substring: "#s5" would match inside "#s50".
                        if not marker or str(marker).endswith(f"#s{int(steps)}"):
                            continue
                        _disengage_step_cache(
                            transformer,
                            reason = f"explicit magcache re-interpolating for {steps} steps",
                            logger = logger,
                        )
                        apply_step_cache(
                            view,
                            mode = TC_MAGCACHE,
                            threshold = state.cache_threshold,
                            quant_active = state.cache_quant_active,
                            family = fam.name,
                            steps = steps,
                            quality = state.cache_quality,
                            expert = expert_name,
                            logger = logger,
                        )
                if state.transformer_cache:
                    self._reset_step_cache(pipe)
                # Dual-GPU CFG parallelism: resolve this generation's routing AFTER the
                # cache toggle (the plan keys on the engaged cache state -- parallel is
                # bit-identical only with the cache on or an uncompiled stack) and
                # serialize any run that may (re)compile. Planning must never fail a
                # generation: a planner error just pins the sequential passthrough.
                cfg_proxy = state.cfg_parallel_handle
                if cfg_proxy is not None:
                    try:
                        cfg_proxy.plan_generation(
                            cache_engaged = bool(state.transformer_cache),
                            steps = steps,
                            width = width,
                            height = height,
                            frames = frames,
                        )
                    except Exception:  # noqa: BLE001 -- fall back to single-device
                        try:
                            cfg_proxy.enabled = False
                        except Exception:  # noqa: BLE001
                            pass
                try:
                    with torch.inference_mode(), progress_ctx:
                        output = pipe(**kwargs)
                except _VideoGenerationCancelled:
                    # This cancel unwinds pipe.__call__ by exception (the scheduler
                    # wrapper has no cooperative _interrupt), skipping the pipeline's
                    # end-of-call maybe_free_model_hooks(); under model/group offload
                    # the currently-onloaded modules would otherwise stay on the GPU
                    # until the next request touches them.
                    free_hooks = getattr(pipe, "maybe_free_model_hooks", None)
                    if callable(free_hooks):
                        try:
                            free_hooks()
                        except Exception:  # noqa: BLE001 -- cleanup is best-effort
                            pass
                    raise RuntimeError(VIDEO_CANCELLED_MSG) from None
                if cancel.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)
                if cfg_proxy is not None:
                    # The (shape, steps, cache) key settles only on a COMPLETED run, so
                    # a cancelled/failed one keeps the next dispatch compile-safe.
                    try:
                        cfg_proxy.note_generation_done()
                    except Exception:  # noqa: BLE001
                        pass
                # The first compiled generation just paid the compile cost; persist the
                # warm torch.compile cache bundle when saving is enabled (distributor /
                # first-run warm). Idempotent + best-effort -- never fails a generation.
                try:
                    compile_cache.save(state.compile_cache_ctx, logger = logger)
                except Exception:  # noqa: BLE001 -- cache persistence is best-effort
                    pass

                self._gen.update(phase = "export", eta_seconds = None)
                video_frames = output.frames[0]
                audio = getattr(output, "audio", None)
                audio_track = audio[0] if fam.has_audio and audio is not None else None
                mp4_bytes = self._encode_mp4(
                    video_frames, out_fps, audio_track, pipe if fam.has_audio else None
                )
                # A cancel that landed during the (blocking, uncancellable) export/mux must
                # still discard the clip: cancel_generate() already reported success for it,
                # so re-check here before it is returned and persisted to the gallery.
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
                    "steps": steps,
                    "guidance": guidance,
                }
            except Exception:
                self._gen = {"active": False}
                raise
            finally:
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None

    @staticmethod
    def _encode_mp4(video_frames, fps: int, audio, pipe) -> bytes:
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
                sample_rate = getattr(
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
            # generate() swaps in a bare {"active": False} on its own exit paths
            # before the job worker records the terminal dict; report the job as
            # still active across that gap so a poller only sees active drop
            # together with a terminal phase ("completed" / "failed").
            if self._generate_job_active:
                gen["active"] = True
        gen.setdefault("active", False)
        return gen

    def cancel_generate(self) -> bool:
        """Signal the in-flight generation to stop at its next step callback."""
        with self._lock:
            cancel = self._active_generate_cancel
            if cancel is None:
                return False
            cancel.set()
            return True

    # ── teardown + status ────────────────────────────────────────────────────

    def _teardown_state(self) -> None:
        state = None
        with self._lock:
            state, self._state = self._state, None
        if state is not None:
            restore_backend_flags(state.backend_flags)
            # Restore TORCHINDUCTOR_CACHE_DIR so a later load (or the image backend)
            # does not inherit this load's per-fingerprint inductor dir. Idempotent.
            compile_cache.restore(state.compile_cache_ctx)
            # A GGUF video load may have installed the process-wide compiled GGUF
            # dequantizer; restore the stock kernels so a later load that asked for
            # speed_mode=off gets the bit-identical path (mirrors the image unload).
            from . import diffusion_gguf_compile

            diffusion_gguf_compile.uninstall_all()
            # Free the CFG-parallel replica on ITS device and restore the pipe's
            # single-device shape (proxy out, guider forward + attention backend
            # restored) before the pipe itself is dropped.
            if state.cfg_parallel_handle is not None:
                teardown_cfg_parallel(state.pipe, state.cfg_parallel_handle, logger = logger)
            del state
            clear_gpu_cache()

    def unload(self) -> dict[str, Any]:
        with self._lock:
            self._load_token += 1
            self._cancel_event.set()
            self._loading = None
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        # Wait for the signalled generation to actually exit before freeing the
        # pipeline: the denoise loop holds its own pipe reference until the next
        # step callback, so tearing down under it would report the VRAM free (and
        # let the GPU arbiter start another multi-GB load) while this clip still
        # occupies it. generate() holds _generate_lock for its full body, so a
        # bare acquire is the exit barrier (never taken while holding _lock).
        with self._generate_lock:
            pass
        self._teardown_state()
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
                "offload_policy": None,
                "vae_tiling": False,
                "memory_mode": None,
                "speed_mode": None,
                "speed_optims": [],
                "attention_backend": None,
                "transformer_cache": None,
                "transformer_quant": None,
                "text_encoder_quant": None,
                "vae_quant": None,
                "cfg_parallel": None,
                "has_audio": False,
                "defaults": None,
                "resolved": None,
            }
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
            "offload_policy": state.offload_policy,
            "vae_tiling": state.vae_tiling,
            "memory_mode": state.memory_mode,
            "speed_mode": state.speed_mode,
            "speed_optims": list(state.speed_optims),
            "attention_backend": state.attention_backend,
            "transformer_cache": state.transformer_cache,
            "transformer_quant": state.transformer_quant,
            "text_encoder_quant": state.text_encoder_quant,
            "vae_quant": state.vae_quant,
            "cfg_parallel": state.cfg_parallel,
            "has_audio": fam.has_audio,
            "defaults": {
                "steps": default_steps,
                "guidance": default_guidance,
                "num_frames": fam.default_num_frames,
                "fps": fam.default_fps,
                "frame_step": fam.frame_step,
                "resolution_multiple": fam.resolution_multiple,
                "resolution_presets": [list(p) for p in fam.resolution_presets],
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

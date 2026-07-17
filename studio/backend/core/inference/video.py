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
    compiled_shapes_are_static,
    resolve_speed_mode,
    restore_backend_flags,
    snapshot_backend_flags,
)
from .diffusion_auto_policy import _QUANT_STEADY_FACTOR, build_resolved_record
from .diffusion_prequant import (
    load_prequantized_transformer,
    resolve_prequant_source,
)
from .diffusion_transformer_quant import (
    DEFAULT_MIN_LINEAR_FEATURES,
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

# Load kinds (mirror the image backend): gguf (single-file GGUF DiT + base repo),
# single_file (safetensors DiT, e.g. fp8 LTX-2.3), pipeline (full diffusers repo).
_MODEL_KINDS = frozenset({"gguf", "single_file", "pipeline"})

# Vendor base repos allowed to load as full (non-GGUF) artifacts despite not being
# under unsloth/. Exact-match, lowercased, safetensors-only, no remote code.
_TRUSTED_NON_GGUF_VIDEO_REPOS = frozenset(
    {
        "lightricks/ltx-2",
        "lightricks/ltx-2.3",
        "lightricks/ltx-2.3-fp8",
        # Wan2.2 official diffusers base repos: safetensors-only, no remote code.
        "wan-ai/wan2.2-ti2v-5b-diffusers",
        "wan-ai/wan2.2-t2v-a14b-diffusers",
        "wan-ai/wan2.2-i2v-a14b-diffusers",
        # HunyuanVideo-1.5 community Diffusers repacks (tencent's own repo is the
        # non-diffusers layout with no model_index.json, unloadable here).
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


def _predownload_may_skip_denoiser(
    fam: Optional[VideoFamily], kind: str, transformer_quant: Optional[str]
) -> bool:
    """Whether the scoped pre-download can skip the DiT weight shards because the hosted
    prequant checkpoints will supply the quantised expert(s) instead.

    Conservative on purpose: only an EXPLICIT int8/fp8 request (auto may legitimately land
    on dense) with the family wired for that exact scheme. Should the shortcut still fall
    through at load time (offload plan, refused checkpoint), the build resolves from the
    hub id and the dense shards download then -- slower, never wrong."""
    if kind != "pipeline" or fam is None:
        return False
    mode = normalize_transformer_quant(transformer_quant)
    if mode in (None, TQ_AUTO):
        return False
    return any(
        entry_scheme == mode for entry_scheme, _ in getattr(fam, "prequant_repos", ())
    )


def _snapshot_has_denoiser_weights(local_dir: str) -> bool:
    """True when the pre-downloaded snapshot carries DiT weight shards (a ``transformer*/``
    subfolder with at least one ``.safetensors``). A prequant-scoped pre-download skips
    them; when the shortcut then falls through, the build must resolve from the hub."""
    try:
        root = Path(local_dir)
        for sub in root.glob("transformer*"):
            if sub.is_dir() and any(sub.glob("*.safetensors")):
                return True
    except OSError:
        return False
    return False


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
            # Not a local dir: resolve a cached HUB blob (no network). Probe active, legacy,
            # AND default cache roots (as the picker's listing does) or a non-active-root GGUF 400s.
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
# After a compiled-tier load commits, a background thread runs one tiny throwaway
# generation so the first REAL request starts at steady state (the vLLM/SGLang
# startup-warmup analogue). Measured (HunyuanVideo-1.5-480p, B200, 480x288/17f/30
# steps): with a warm Mega-cache bundle the first-generation extra drops 11.3s ->
# 2.1s (a 9.0s warmup absorbs the dynamo trace + cudnn autotune the bundle cannot
# carry); cold, the full ~54s compile moves off the user's first request. Steady
# state is untouched (residuals reset, real run re-seeds its generator). The tiny
# shape suffices: the default tier compiles dynamic=True, so one trace serves all.
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
    """Whether the post-load compile prewarm should run, with its resolved-record
    reason. Pure so it unit-tests without the runtime."""
    if (os.environ.get(_PREWARM_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False, f"disabled via {_PREWARM_ENV}"
    if "compiled" not in speed_optims:
        return False, "not applicable (no regional compile engaged; nothing to prewarm)"
    if speed_mode != SPEED_DEFAULT:
        # speed=max compiles dynamic=False (per-shape graph): a warmup shape would
        # compile a graph the real request never runs, and the real shape still pays.
        return (
            False,
            "skipped: speed=max compiles static per-shape graphs a warmup shape cannot serve",
        )
    if not bool(getattr(fam, "supports_compile_prewarm", True)):
        return False, "family opted out (supports_compile_prewarm=False)"
    if offload_policy != "none":
        # Offload streams the full DiT over PCIe per forward: the warmup would be
        # all transfer cost, none of the measured benefit.
        return (
            False,
            "skipped: offload streams weights per forward; the warmup would only churn transfers",
        )
    if cfg_parallel_active:
        # CFG-parallel serialises compile-sensitive runs through its own planner;
        # an unplanned warmup would bypass it.
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
        # A renamed GGUF carries no family token in its name; resolve via general.architecture
        # (its string, e.g. "ltxv", is a family alias). No-backend archs still yield None -> 400.
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
    # AUTO on a cache-capable DiT: generate() re-checks the step count and toggles FBCache
    # across FBCACHE_MIN_STEPS. An explicit request (off / fbcache) is never toggled.
    cache_auto: bool = False
    # Inputs the generation-time toggle re-applies (quantised threshold + override).
    cache_quant_active: bool = False
    cache_threshold: Optional[float] = None
    # Resolved cache quality preset ("quality" | "balanced" | "fast"); the toggle
    # re-applies it so a mid-session re-engage keeps the requested preset.
    cache_quality: Optional[str] = None
    # Dense transformer quant engaged ("int8" | "fp8" | "nvfp4" | "mxfp8") or None
    # (dense bf16). On a pipeline load the DiT(s) are torchao-quantised in place;
    # mirrors the image backend's _LoadState.transformer_quant.
    transformer_quant: Optional[str] = None
    # Text-encoder quant engaged ("fp8" | "fp8_dynamic" | "int8" | "nvfp4") or None.
    # The dense-bf16 companion TE (UMT5 / Gemma3 / Qwen2.5-VL), often the largest
    # resident component, is shrunk in place; mirrors the image backend.
    text_encoder_quant: Optional[str] = None
    # VAE quant engaged ("fp8" layerwise | "fp8_dynamic" torchao conv) or None. The
    # conv decoder shrinks in place; vae_force_fp32 families (Wan) stay dense.
    vae_quant: Optional[str] = None
    # Dual-GPU CFG branch parallelism: "on" when a DiT replica on a second CUDA device runs the
    # pred_cond branch (bit-identical under the family's auto step cache; ~1.7x e2e). The handle is
    # the proxy -- generate() plans each run on it and _teardown_state frees the replica through it.
    cfg_parallel: Optional[str] = None
    cfg_parallel_handle: Any = None
    # Pre-warmed torch.compile cache context (CacheContext) when a compiled tier ran
    # begin(); generate() persists the bundle after the first compiled generation and
    # _teardown_state restores the inductor dir.
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
# Wan2.2-A14B is a dual-expert MoE: ``transformer`` runs the high-noise steps and
# ``transformer_2`` the low-noise steps (routed by boundary_ratio), so optimising only
# ``transformer`` leaves the second expert unoptimised for half the schedule. Two
# helper shapes:
#   - apply_speed_optims / apply_attention_backend / install_hunyuan_attention_trim fan
#     out over every denoiser DiT internally, so the loader calls them ONCE on the pipe.
#   - apply_step_cache and the dense quantiser act on ``pipe.transformer`` only (the
#     cache also needs a per-expert calibrated curve). Rather than fork them, present the
#     second DiT AS ``pipe.transformer`` via a thin proxy view and call once per expert;
#     single-DiT loads are bit-identical (proxy only built for is_moe families).


def _transformer_names(pipe: Any, fam: VideoFamily) -> tuple[str, ...]:
    """Denoiser attribute name(s) to optimise: ("transformer",), plus "transformer_2"
    for an MoE family whose second expert is present (a checkpoint may ship only one)."""
    names = ["transformer"]
    if fam.is_moe and getattr(pipe, "transformer_2", None) is not None:
        names.append("transformer_2")
    return tuple(names)


class _SecondDiTView:
    """Thin proxy exposing ``pipe.transformer_2`` as ``pipe.transformer`` to a helper that
    hardcodes ``getattr(pipe, "transformer")``; every other attribute reads through to the
    real pipe. Lets single-DiT helpers optimise the second expert without a fork. Only
    wrapped around an MoE pipe (guarded by fam.is_moe)."""

    def __init__(self, pipe: Any) -> None:
        # Store under a name __getattr__ never fires for.
        object.__setattr__(self, "_pipe", pipe)

    @property
    def transformer(self) -> Any:
        return self._pipe.transformer_2

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not on the instance/class, so delegate to the pipe.
        return getattr(object.__getattribute__(self, "_pipe"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Writes must land on the real pipe or a helper's side effect would vanish with
        # the view; ``transformer`` mirrors onto the second expert.
        pipe = object.__getattribute__(self, "_pipe")
        setattr(pipe, "transformer_2" if name == "transformer" else name, value)


def _views_for(pipe: Any, fam: VideoFamily) -> tuple[Any, ...]:
    """Pipe view(s) covering every denoiser: the real pipe, plus a ``_SecondDiTView``
    for a dual-DiT MoE family. Single-DiT returns just ``(pipe,)``."""
    if fam.is_moe and getattr(pipe, "transformer_2", None) is not None:
        return (pipe, _SecondDiTView(pipe))
    return (pipe,)


def _step_cache_all_or_none(
    pipe: Any, fam: VideoFamily, engage_fn: Any, *, logger: Any
) -> tuple[Optional[str], Optional[str]]:
    """Run ``engage_fn(view, expert_name)`` over every expert and enforce ALL-OR-NONE
    (mirroring the transactional quant loop): on a mixed outcome, disengage the engaged
    expert(s) and report the cache off. Otherwise the whole MoE reports cached while
    half the schedule runs uncached. Returns (mode-or-None, failure-reason-or-None); a
    single-DiT family can never see a mixed outcome."""
    pairs = list(zip(_views_for(pipe, fam), _transformer_names(pipe, fam)))
    results: list[tuple[Any, str, Optional[str]]] = []
    try:
        for view, expert_name in pairs:
            results.append((view, expert_name, engage_fn(view, expert_name)))
    except BaseException as exc:
        # A later expert raising mid-loop leaves earlier experts cached (an all-or-none
        # violation), so tear down every marked expert then re-raise, or raise reload-required
        # if rollback itself fails.
        rollback_failed: list[str] = []
        for view, name in pairs:
            transformer = getattr(view, "transformer", None)
            if getattr(transformer, "_unsloth_step_cache", None) is None:
                continue
            if not _disengage_step_cache(
                transformer, reason = "all-or-none transaction aborted", logger = logger
            ):
                rollback_failed.append(name)
        if rollback_failed:
            raise RuntimeError(
                "step-cache transaction raised and rollback failed for "
                + ", ".join(rollback_failed)
                + "; reload the video model before generating"
            ) from exc
        raise
    engaged = [(view, name, mode) for view, name, mode in results if mode is not None]
    if engaged and len(engaged) < len(results):
        missing = ", ".join(name for _, name, mode in results if mode is None)
        # Roll the engaged expert(s) back; a rollback that ALSO fails leaves one expert cached
        # while state/status would report the whole pipeline uncached -- a silent inconsistency,
        # so surface it as a hard reload-required error instead of returning a false "uncached".
        rollback_failed: list[str] = []
        for view, name, _mode in engaged:
            if not _disengage_step_cache(
                getattr(view, "transformer", None),
                reason = f"all-or-none rollback: cache did not engage on {missing}",
                logger = logger,
            ):
                rollback_failed.append(name)
        if rollback_failed:
            raise RuntimeError(
                "step cache engagement was partial and rollback failed for "
                + ", ".join(rollback_failed)
                + "; reload the video model before generating"
            )
        return None, (
            f"step cache engaged on only {len(engaged)}/{len(results)} experts "
            f"({missing} failed); disengaged all experts and running uncached"
        )
    return (engaged[0][2] if engaged else None), None


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
        # True from begin_generate() until its worker records a terminal state, so a second
        # begin_generate() is refused while the first still runs.
        self._generate_job_active = False
        # Post-load compile prewarm thread (None until a compiled load spawns one);
        # kept for tests/diagnostics, never joined on the hot path.
        self._prewarm_thread: Optional[threading.Thread] = None
        # The prewarm's cancel event, set only while the prewarm runs. Real generations signal it on
        # entry so they preempt the warmup at its next step boundary instead of queueing behind it;
        # unlike _active_generate_cancel it can never point at a real job.
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
        transformer_cache_quality: Optional[str] = None,
        cfg_parallel: Optional[str] = None,
    ) -> VideoFamily:
        """Cheap, network-free validation shared by the route and the load path."""
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        # A -GGUF repo picked without a quant filename resolves to pipeline kind and would
        # only fail in from_pretrained (no model_index.json) after the route evicts the owner.
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
        # Companions load with from_pretrained, so a base repo is held to the non-GGUF bar:
        # a GGUF pick must not smuggle in an arbitrary remote base.
        if base_repo and (base_repo or "").strip() and not _is_trusted_video_repo(base_repo):
            raise ValueError(
                f"base_repo is limited to unsloth/* repos, the official family base "
                f"repos, and local paths; '{base_repo}' is neither."
            )
        # A local base_repo loads as a full pipeline (needs model_index.json); reject a
        # non-pipeline local base here, before the load. Shared helper keeps image/video/training in sync.
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
            # Fail a kind/extension mismatch before the GPU handoff: gguf needs .gguf,
            # single_file needs .safetensors (mirrors the image loader's gate).
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
            # Path-shaped: "."/".." prefix, a backslash (never in "org/name"), or an absolute
            # path -- so a missing Windows-shaped local pick fails before the handoff, not as a Hub repo.
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
                # The loader hands a local FILE straight through (ignoring gguf_filename), so
                # the file's OWN suffix must match the kind; reject a mismatch before the handoff.
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
        # A local pipeline pick must be a diffusers directory (model_index.json), else it would
        # only fail in from_pretrained after eviction (mirrors the image loader).
        if kind == "pipeline":
            root = Path(repo_id).expanduser()
            # Gate on .exists() (not .is_dir()) so a local FILE picked as a pipeline is rejected too.
            if root.exists() and not (root.is_dir() and (root / "model_index.json").is_file()):
                raise ValueError(
                    f"Local pipeline path is not a diffusers directory "
                    f"(no model_index.json): {repo_id}"
                )
        # Reject a malformed transformer_quant cheaply, before the handoff (applies on
        # pipeline-kind loads; ignored on gguf/single_file, matching the image backend).
        normalize_transformer_quant(transformer_quant)
        # Reject a malformed text_encoder_quant the same way (any kind: the encoder is always dense).
        normalize_te_quant(text_encoder_quant)
        # Same for vae_quant (the dense VAE is resident for every load kind).
        normalize_vae_quant(vae_quant)
        # Reject malformed cache-quality / cfg-parallel here too: a direct backend caller
        # (bench, plugin, test) would otherwise do checkpoint/download work before failing deep.
        normalize_cache_quality(transformer_cache_quality)
        normalize_cfg_parallel(cfg_parallel)
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
            transformer_cache_quality = transformer_cache_quality,
            cfg_parallel = cfg_parallel,
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
            # Checkpoint downloads outside the lock so an unload/eviction can preempt the
            # multi-GB pull; companions pre-download the same way (scoped, cancellable, resumable).
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
            # An LTX-2.3 checkpoint supplies the VAEs/vocoder/connectors, so the base pull
            # shrinks to scheduler + text encoder + tokenizer; recompute the estimate to match
            # (detectable only once the checkpoint header is on disk).
            ltx23 = False
            if fam is not None and fam.name == "ltx-2" and kind != "pipeline":
                from .video_ltx2 import is_ltx23_checkpoint

                probe = checkpoint_local
                if probe is None:
                    # Local repos: a bare file, or a dir child via the same resolver load_pipeline
                    # uses. Unresolvable -> load_pipeline surfaces the real error; keep the wide pull.
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
            base_local = self._predownload_base(
                base,
                kwargs.get("hf_token"),
                kind,
                ltx23 = ltx23,
                # Skip the DiT weight shards when the hosted prequant checkpoints will replace
                # them (explicit int8/fp8 with a wired repo); the download progress estimate
                # deliberately stays un-shrunk (an over-estimate finishes early, never hangs).
                skip_denoiser = _predownload_may_skip_denoiser(
                    fam, kind, kwargs.get("transformer_quant")
                ),
            )
            # The 2.3 assembly pulls per component from the hub id (its snapshot lacks the base
            # VAEs), so it only gets the warmed cache; generic paths get the full local snapshot.
            kwargs["_base_local_dir"] = None if ltx23 else base_local
            self.load_pipeline(**kwargs)
            with self._lock:
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            # A failed/cancelled load never commits _VideoLoadState, so roll back the
            # process-wide speed globals here (token-scoped, so a superseded load can't clobber a newer one's).
            self._rollback_precommit_globals(token)
            self._rollback_precommit_cfg_parallel(token)
            self._rollback_precommit_compile_cache(token)
            if self._load_token != token:
                return
            logger.error("video.load_failed: %s", exc)
            # Free the debris of a failed construction (mirrors diffusion.py): no state was
            # committed, so nothing else releases the VRAM a partial pipeline reserved. Guarded
            # so a sticky CUDA error can't skip stamping the real error below.
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
        """Tear down a CFG-parallel proxy from a load that died BEFORE committing
        _VideoLoadState. The proxy owns a daemon worker, the DiT replica's VRAM, and
        possibly the process-global cuDNN attention patch, none of which _teardown_state
        would reach. Token-scoped like _rollback_precommit_globals."""
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
        but died BEFORE committing _VideoLoadState (committed loads restore via
        _teardown_state). Token-scoped like _rollback_precommit_globals."""
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
        skip_denoiser: bool = False,
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
            # .jinja: tokenizer/chat_template.jinja is a standalone file apply_chat_template
            # needs at generation time; a snapshot without it crashes the first generation.
            if not name.endswith((".safetensors", ".json", ".model", ".txt", ".jinja")):
                continue
            if "/" not in name and name.endswith(".safetensors"):
                continue
            if kind != "pipeline" and name.startswith("transformer/"):
                continue
            # A prequant-scoped pull skips every expert's weight shards (the hosted
            # checkpoints replace them) but keeps the configs for the meta-init.
            if (
                skip_denoiser
                and name.startswith(("transformer/", "transformer_2/"))
                and not name.endswith("config.json")
            ):
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
        skip_denoiser: bool = False,
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
            files = self._base_download_files(
                info, kind, ltx23 = ltx23, skip_denoiser = skip_denoiser
            )
            if not any(name == "model_index.json" for name, _ in files):
                return None
            from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

            snapshot_root: Optional[Path] = None
            for name, _ in files:
                # Explicit check: a cached file returns without consulting the event, so a
                # warm-cache sweep would otherwise run to completion after an unload cancelled.
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
        """Bytes of ``repo_id`` currently in the HF blob cache (progress polling).

        Walks the repo's cache directory directly instead of ``scan_cache_dir``:
        the scanner skips in-flight ``*.incomplete`` blobs, so during a multi-GB
        shard pull the counter would freeze at the last completed blob for minutes
        while the disk keeps filling (the bar sat stuck mid-download)."""
        if not repo_id:
            return 0
        try:
            import os

            from huggingface_hub.constants import HF_HUB_CACHE

            folder = Path(HF_HUB_CACHE) / ("models--" + repo_id.strip().replace("/", "--"))
            if not folder.is_dir():
                return 0
            total = 0
            for root, _dirs, files in os.walk(folder):
                for name in files:
                    try:
                        path = os.path.join(root, name)
                        # Snapshot entries are symlinks into blobs/; skip them so a
                        # blob is not counted twice.
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
        if expected and downloaded >= expected:
            phase = "finalizing"
            # The cache scan counts every blob (incl. files this load never reads), so the raw
            # counter can exceed the scoped estimate; clamp to what the bar reports.
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
            transformer_cache_quality = transformer_cache_quality,
            cfg_parallel = cfg_parallel,
        )
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        # An explicit Speed="off" (bit-exact) load pins the companions dense too: promoting
        # an UNSET TE/VAE to auto-quant would silently fp8/int8 them and break the request.
        # Only an EXPLICIT off suppresses; an unset speed still auto-quantises, and an
        # explicit companion scheme still forces it.
        speed_off = speed_mode is not None and str(speed_mode).strip().lower() == SPEED_OFF
        # text_encoder_quant tri-state (mirrors transformer_quant): UNSET ("" / None) or
        # "auto" -> auto; explicit "none"/"off" -> dense; a concrete scheme forces it. auto
        # is backend-owned, so "auto" also goes dense under Speed="off".
        if text_encoder_quant is None or str(text_encoder_quant).strip().lower() in ("", "auto"):
            text_encoder_quant = "off" if speed_off else TE_QUANT_AUTO
        # vae_quant tri-state, same contract. vae_force_fp32 families (Wan) stay dense
        # regardless, so auto is safe as the default.
        if vae_quant is None or str(vae_quant).strip().lower() in ("", "auto"):
            vae_quant = "off" if speed_off else VAE_QUANT_AUTO
        base = repo_id if kind == "pipeline" else resolve_video_base_repo(fam, base_repo)

        with self._lock:
            if _load_token is not None and _load_token != self._load_token:
                raise RuntimeError("Video load was cancelled or superseded.")
            # Signal a generation from the PREVIOUS model (the token check above bailed a superseded worker).
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        # Barrier: wait for the signalled generation to exit before teardown, or two models
        # coexist in VRAM (the denoise loop holds its pipe ref until the next callback).
        with self._generate_lock:
            pass
        # The barrier wait can outlive this load (a newer load / unload superseded it); recheck
        # before touching shared state so we don't destroy the current model or build a dead pipe.
        if _load_token is not None and _load_token != self._load_token:
            raise RuntimeError("Video load was cancelled or superseded.")
        self._teardown_state()

        target = resolve_diffusion_device_target()
        device = target.device
        # Video DiTs are bf16-native; fp16 overflows, so a resolved fp16 promotes to float32
        # (same rule as fp16-incompatible image families). CPU stays float32.
        dtype = target.dtype
        if fam.fp16_incompatible and dtype is torch.float16:
            dtype = torch.float32
        # Size tables below are bf16 (2-byte); when the promotion lands fp32 on an accelerator,
        # dense estimates double, so scale them (GGUF stays quantised, so only dense scales).
        dtype_scale = 2.0 if device != "cpu" and dtype is torch.float32 else 1.0

        # Precision tri-state (mirror image backend): unset/"auto" -> hardware ladder picks a
        # quantised DiT (int8 min, fp8 on datacenter silicon); "none"/"off" pins dense bf16; an
        # explicit scheme pins it. Pipeline-kind only; the offload guard below still skips it.
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
            # Budget ALL weights (image-backend contract): companions stay resident, so
            # budgeting the transformer alone lets auto pick OFFLOAD_NONE and OOM.
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
        # Parity with the image dense-quant path: the bf16-table plan can force offload a
        # quantised DiT would not need. Re-plan with the scheme's steady factor and keep the
        # resident placement if it fits; fall back to this bf16 plan if quant later fails.
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

        # ── hosted prequant shortcut (pipeline kind): when the resolved quant scheme has a
        # hosted checkpoint for EVERY expert, load those instead of materialising the dense
        # DiT(s) inside from_pretrained -- no dense transient in RAM/VRAM and, when the
        # pre-download skipped the DiT shards, ~half the transformer download. Mirrors the
        # image loader's prequant path (metadata-validated; any miss falls back to
        # dense+quantise below). Resident plans only (quant never engages under offload), and
        # the same dense-preference that gates the in-place auto-quant below applies here, so
        # prequant cannot engage where auto-quant would have declined.
        prequant_scheme: Optional[str] = None
        prequant_transformers: dict[str, Any] = {}
        if (
            kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
            and plan.offload_policy == "none"
            and not (
                normalize_transformer_quant(transformer_quant) == TQ_AUTO
                and bf16_plan.offload_policy == "none"
                and is_int8_memory_fallback(target, fam.name)
            )
        ):
            scheme_candidate = select_transformer_quant_scheme(
                target, transformer_quant, family = fam.name
            )
            if scheme_candidate is not None:
                expert_attrs = ["transformer"]
                if fam.is_moe and fam.transformer2_class:
                    expert_attrs.append("transformer_2")
                sources = []
                for attr in expert_attrs:
                    src = resolve_prequant_source(
                        fam, scheme_candidate, base_repo = base, expert = attr
                    )
                    if src is None:
                        sources = []
                        break
                    sources.append((attr, src))
                if sources:
                    transformer_cls = getattr(diffusers, fam.transformer_class)
                    loaded: dict[str, Any] = {}
                    for attr, src in sources:
                        module = load_prequantized_transformer(
                            transformer_cls,
                            # The HUB id, not the pre-download dir: the checkpoint's baked
                            # base_model_id is compared against this (a snapshot path's
                            # <sha> tail never matches), and the expert config it meta-inits
                            # from is already in the HF cache from the scoped pre-download.
                            base,
                            src,
                            device = device,
                            dtype = dtype,
                            hf_token = hf_token,
                            scheme = scheme_candidate,
                            # Same Linear filter as runtime quant, so prequant == quantize_.
                            min_features = DEFAULT_MIN_LINEAR_FEATURES,
                            # The expert's config subfolder (transformer_2 for the second DiT).
                            subfolder = attr,
                            logger = logger,
                        )
                        if module is None:
                            loaded = {}
                            break
                        loaded[attr] = module
                    if loaded:
                        prequant_transformers = loaded
                        prequant_scheme = scheme_candidate
                    else:
                        # All-or-none: a partial expert pair must not ship (mismatched
                        # precision between experts); free anything loaded and go dense.
                        clear_gpu_cache()

        # ── build the pipeline.
        pipeline_cls = getattr(diffusers, fam.pipeline_class)
        pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if getattr(fam, "vae_force_fp32", False):
            # Wan's VAE must decode in float32. A scalar torch_dtype truncates its fp32 weights
            # to bf16 (no _keep_in_fp32_modules); a later .to(float32) only widens lossy values
            # (banding / black frames). Use the per-component dtype dict; "default" MUST be set
            # or unlisted components fall back to fp32 (over-widening the DiT).
            pipe_kwargs["torch_dtype"] = {"vae": torch.float32, "default": dtype}
        if hf_token:
            pipe_kwargs["token"] = hf_token
        if kind == "pipeline":
            # The pre-downloaded snapshot dir keeps from_pretrained off the hub (its sweep would
            # also pull root checkpoints + duplicate shards); hub id when pre-download was skipped.
            if prequant_transformers:
                # Already-quantised experts ride in as component overrides; from_pretrained
                # loads everything else and never touches the dense DiT shards.
                pipe = pipeline_cls.from_pretrained(
                    _base_local_dir or repo_id, **pipe_kwargs, **prequant_transformers
                )
            else:
                build_source = _base_local_dir or repo_id
                if (
                    _base_local_dir
                    # Only a prequant-scoped pre-download deliberately omits the DiT shards;
                    # any other snapshot without them is the caller's responsibility (tests
                    # pass bare dirs) and must be used as-is.
                    and _predownload_may_skip_denoiser(fam, kind, transformer_quant)
                    and not _snapshot_has_denoiser_weights(_base_local_dir)
                ):
                    # The pre-download skipped the DiT shards expecting the prequant shortcut,
                    # which then fell through: resolve from the hub id instead (cached
                    # companions are reused; only the dense shards download now).
                    build_source = repo_id
                pipe = pipeline_cls.from_pretrained(build_source, **pipe_kwargs)
        else:
            transformer_cls = getattr(diffusers, fam.transformer_class)
            # checkpoint_path was resolved (and downloaded) by the memory-planning branch above.
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
                # 2.3 checkpoints need the full assembly: new config flags, key renames the
                # stock converter lacks, and the 2.3 connectors/VAEs/vocoder the base lacks.
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

        # The dtype dict already loads the Wan VAE at float32. Belt-and-suspenders for any path
        # that bypassed it (e.g. a passed-in vae=): re-pin an fp32-force VAE that came back lower.
        if getattr(fam, "vae_force_fp32", False):
            vae = getattr(pipe, "vae", None)
            if vae is not None and getattr(vae, "dtype", None) is not torch.float32:
                vae.to(torch.float32)

        if _load_token is not None and _load_token != self._load_token:
            del pipe
            clear_gpu_cache()
            raise RuntimeError("Video load was cancelled or superseded.")

        # For a dual-DiT MoE (Wan2.2-A14B), every optimisation site below covers BOTH experts:
        # ``views`` is (pipe, _SecondDiTView(pipe)); a single-DiT load resolves to (pipe,).
        views = _views_for(pipe, fam)

        # ── dense transformer quant (opt-in, pipeline-kind only): torchao-quantise the dense
        # bf16 DiT in place onto the low-precision tensor cores (image-backend fast path). CUDA +
        # bf16 only; best-effort. Quant must precede compile (eager dynamic quant is ~30x slower).
        transformer_quant_engaged: Optional[str] = None
        quant_skipped_for_offload = False
        if prequant_scheme is not None:
            # The hosted checkpoints already carry the quantised experts (all of them, or the
            # shortcut would not have engaged); nothing to quantise in place.
            transformer_quant_engaged = prequant_scheme
        # Auto-quant lands on int8 for fp8-denied families (HunyuanVideo-1.5), but on a B200
        # int8 is ~7% slower AND less accurate than dense bf16 + regional compile (fp8, the
        # only quant that also speeds up, is black-framed there). int8's sole benefit is
        # memory, so when the DENSE DiT already fits resident (bf16_plan has no offload)
        # prefer dense and skip auto-quant. Only for AUTO (explicit int8/fp8 honored), only
        # where int8 is a denied fallback on an fp8-capable GPU (is_int8_memory_fallback
        # excludes consumer / Ampere / fp8 families), only when dense provably fits -- so no
        # new OOM risk and fp8 families (Wan / LTX) still quantise for their speed win.
        quant_skipped_for_dense = (
            prequant_scheme is None
            and kind == "pipeline"
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
            prequant_scheme is None
            and kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
            and plan.offload_policy != "none"
        ):
            # Offload hooks move modules with Module.to(), which torchao quantized tensors reject
            # (aten._has_compatible_shallow_copy_type unimplemented) -- a hard crash on the
            # Wan2.2-A14B gate run (114 GB dual DiT plans model offload). Skip quant (dense-under-
            # offload beats a crash); surfaced in the resolved record, forceable via a resident mode.
            logger.info(
                "video.transformer_quant: skipped (offload policy '%s' moves the "
                "DiT via Module.to(), unsupported for torchao quantized tensors); "
                "pin a resident memory mode to combine quant with this model",
                plan.offload_policy,
            )
            quant_skipped_for_offload = True
        elif (
            prequant_scheme is None
            and kind == "pipeline"
            and normalize_transformer_quant(transformer_quant) is not None
            and dense_transformer_supported(target)
        ):
            engaged = []
            for view in views:
                # Pass each expert's view so both DiTs quantise with the same arch-chosen scheme.
                # The family name drives the per-family deny table (_FAMILY_SCHEME_DENY).
                scheme = quantize_transformer(
                    view,
                    target,
                    mode = transformer_quant,
                    family = fam.name,
                    logger = logger,
                )
                if scheme is not None:
                    engaged.append(scheme)
            # All experts or none: the first is mutated in place, so a second-expert failure
            # can't fall back to dense (mismatched precision). Fail cleanly; a full miss stays dense.
            if engaged and len(engaged) < len(views):
                del pipe
                clear_gpu_cache()
                raise RuntimeError(
                    f"transformer_quant={engaged[0]} engaged on only "
                    f"{len(engaged)}/{len(views)} experts; retry without quant."
                )
            if engaged:
                transformer_quant_engaged = engaged[0]
        # The quant-sized plan is valid only when quant engaged; a dense fallback keeps bf16 placement.
        if quant_replanned and transformer_quant_engaged is None:
            plan = bf16_plan

        # ── dense text-encoder quant (opt-in): the companion encoder (Gemma3/UMT5/Qwen2.5-VL)
        # loads dense bf16 and is often the largest resident. Quantise in place for every kind,
        # before placement (so offload moves the smaller weights). Best-effort; family drives int8's keep-bf16 schedule.
        text_encoder_quant_engaged = quantize_text_encoders(
            pipe,
            target,
            mode = text_encoder_quant,
            family = fam.name,
            offload_active = plan.offload_policy != "none",
            logger = logger,
        )
        # Quantise the dense conv VAE (opt-in fp8 layerwise / fp8_dynamic torchao conv).
        # vae_force_fp32 families (Wan) run fp32 for stability, so force_fp32 pins them
        # dense (quantising bands the decode); auto skips torchao under offload.
        # Best-effort: a failure leaves the VAE dense.
        vae_quant_engaged = quantize_vae(
            pipe,
            target,
            mode = vae_quant,
            family = fam.name,
            offload_active = plan.offload_policy != "none",
            force_fp32 = getattr(fam, "vae_force_fp32", False),
            logger = logger,
        )

        # ── optimisation layers in the image backend's order: step cache FIRST (compile keys
        # its fullgraph decision off an active cache; FBCache hooks graph-break), then attention,
        # speed profile, placement last. A clip denoise runs minutes, so even a dense load
        # amortises the compile: unset resolves to the near-lossless `default`; "off"/explicit honored.
        effective_speed = resolve_speed_mode(
            speed_mode, is_gguf = kind == "gguf", dense_default = SPEED_DEFAULT
        )
        # A torchao-quantised DiT must be compiled (eager is ~30x slower), so force at least
        # the regional-compile profile when quant engaged but speed was off (matches diffusion.py).
        if transformer_quant_engaged is not None and effective_speed == SPEED_OFF:
            logger.info(
                "video.transformer_quant: forcing speed_mode=default "
                "(quantized transformer must be compiled; eager is ~30x slower)"
            )
            effective_speed = SPEED_DEFAULT
        backend_flags = snapshot_backend_flags()
        # Until the state commit transfers ownership to _teardown_state, a failure must restore
        # these globals itself (via _rollback_precommit_globals). Registered BEFORE the first mutation.
        self._precommit_globals = (_load_token, backend_flags)
        # Step cache tri-state: unset/"auto" -> step-count policy decides (engage when the DEFAULT
        # schedule reaches FBCACHE_MIN_STEPS, re-checked per generation); "off"/"fbcache" pinned.
        # Run per expert so both denoisers cache.
        cache_request = normalize_transformer_cache(transformer_cache)
        cache_auto = transformer_cache is None or cache_request == TC_AUTO
        # Cache quality preset tri-state: unset / "auto" -> the family's measured default
        # (near-lossless "quality" for HunyuanVideo-1.5, "balanced" -- the pre-knob
        # behaviour -- elsewhere); explicit presets honored verbatim. Resolved here so
        # the generation-time toggle re-applies it.
        cache_quality_requested = normalize_cache_quality(transformer_cache_quality)
        cache_quality = cache_quality_requested or auto_cache_quality(fam.name)
        # Validate the dual-GPU CFG request cheaply here; the gate itself must run
        # after placement (it keys on the post-plan device layout + free VRAM).
        normalize_cfg_parallel(cfg_parallel)
        # GGUF checkpoints and torchao-quantised DiTs both need the higher quantised
        # threshold for the cache to still trigger over the quant noise.
        cache_quant_active = kind == "gguf" or transformer_quant_engaged is not None
        # Computed for every request: the auto step-count policy and an explicit magcache
        # request both key on it (the ratio curve interpolates over the step count).
        default_cache_steps, _ = default_video_generation_params(gguf_filename, repo_id, base)
        if cache_auto:
            # The engaged CACHE MODE is per-family: MagCache where FBCache's uncapped
            # skipping derails the trajectory (HunyuanVideo-1.5), FBCache elsewhere.
            cache_request = (
                auto_cache_mode(fam.name) if default_cache_steps >= FBCACHE_MIN_STEPS else None
            )

        # Each expert view passes expert="transformer_2" (for the second) so MagCache
        # resolves THAT expert's calibrated curve -- the experts split the schedule at the
        # boundary timestep, so their curves differ. All-or-none across experts (like the
        # quant loop): a mixed outcome is rolled back and reported uncached.
        def _engage_load_cache(view: Any, expert_name: str) -> Optional[str]:
            return apply_step_cache(
                view,
                mode = cache_request,
                threshold = transformer_cache_threshold,
                # A quantized transformer's larger block residuals need the higher FBCache
                # threshold to cache at all; both an engaged quant AND a GGUF checkpoint
                # count as quant-active (cache_quant_active), mirroring diffusion.py.
                quant_active = cache_quant_active,
                family = fam.name,
                steps = default_cache_steps,
                quality = cache_quality,
                expert = expert_name,
                logger = logger,
            )

        cache_engaged, cache_partial_reason = _step_cache_all_or_none(
            pipe, fam, _engage_load_cache, logger = logger
        )
        # The auto decision can flip at generation time, but only on a DiT that
        # supports caching at all (a non-CacheMixin transformer can never engage).
        cache_may_toggle = cache_auto and callable(
            getattr(getattr(pipe, "transformer", None), "enable_cache", None)
        )
        if cache_partial_reason:
            cache_reason = cache_partial_reason
        elif cache_auto:
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
        # A dense torchao transformer on the pipeline path is not a GGUF one, so is_gguf
        # keys off the load kind (gguf) AND no quant having engaged.
        gguf_transformer = kind == "gguf" and transformer_quant_engaged is None
        # install_hunyuan_attention_trim and apply_attention_backend fan out over every
        # denoiser DiT internally, so ONE pipe-level call covers a dual-expert MoE.
        # Trim is HunyuanVideo-1.5 only: drop the ~99% zero-padded text tokens from joint
        # attention so it runs the fused (cuDNN/flash) SDPA kernel instead of the dense-mask
        # fallback (~18x/DiT-forward at 121 frames, cosine ~1.0). Must precede the backend
        # set so the kernel pins onto the new processors. No-op for other families. A speed
        # lever, so honor Speed="off" (the bit-exact path keeps stock dense-mask attention).
        attention_trim_engaged = (
            install_hunyuan_attention_trim(pipe, fam, logger = logger)
            if effective_speed != SPEED_OFF
            else False
        )
        attention_engaged = apply_attention_backend(
            pipe,
            select_attention_backend(
                target, attention_backend, speed_active = effective_speed != SPEED_OFF
            ),
            logger = logger,
        )
        # Pre-warmed torch.compile cache (Mega-cache): when a compiled tier will run, point
        # inductor at a per-fingerprint dir and load a matching bundle BEFORE the first
        # compiled forward. Measured (HunyuanVideo-1.5-480p, B200): first-generation compile
        # extra drops 107.5s -> 13.8s from a 12.8 MB bundle, and the persistent per-key dir
        # alone recovers a restart to 11.7s (vs ~100s repaid every reboot with stock /tmp).
        # A miss is silent -> local compile. Must run AFTER the attention backend set (the
        # fingerprint keys on the engaged kernel) and BEFORE apply_speed_optims.
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
                    # Mirrors apply_speed_optims' fullgraph decision (an active/toggleable
                    # cache or a planned offload graph-breaks), so the bundle keys on it.
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
            # Until the state commit transfers ownership to _teardown_state, a failed
            # or cancelled load restores TORCHINDUCTOR_CACHE_DIR itself (_run_load's
            # error handler, token-scoped like the globals).
            self._precommit_compile_cache = (_load_token, compile_ctx)
        # apply_speed_optims fans out over every denoiser DiT internally, so one
        # pipe-level call covers a dual-expert MoE.
        applied = apply_speed_optims(
            pipe,
            target,
            is_gguf = gguf_transformer,
            family = fam,
            speed_mode = effective_speed,
            # An auto cache that could still engage mid-session also drops fullgraph: enabling
            # FBCache under a fullgraph-compiled DiT would crash the first cached generation.
            cache_active = cache_engaged is not None or cache_may_toggle,
            offload_active = plan.offload_policy != "none",
        )
        speed_optims = tuple(k for k, v in applied.items() if v) + (
            ("hunyuan_attn_trim",) if attention_trim_engaged else ()
        )
        with self._generate_lock:
            # A cancelled/superseded load must not place weights on a GPU the arbiter may have
            # reassigned; recheck right before placement (the commit below does the final check).
            if _load_token is not None and _load_token != self._load_token:
                del pipe
                clear_gpu_cache()
                raise RuntimeError("Video load was cancelled or superseded.")
            offload_policy, vae_tiling = apply_memory_plan(pipe, plan, device = device, logger = logger)
            # A dual-DiT MoE needs no extra per-expert pass: apply_memory_plan already covers every
            # DiT (transformer AND transformer_2) under all tiers; a second pass would duplicate-hook.
            if not vae_tiling:
                # Whole-clip decode is the video memory peak; tiling is near-free, so always on.
                try:
                    pipe.vae.enable_tiling()
                    vae_tiling = True
                except Exception as exc:  # noqa: BLE001 -- tiling is an optimisation only
                    logger.warning("video.vae_tiling_failed: %s", exc)

            # ── dual-GPU CFG branch parallelism (auto on measured families, else opt-in).
            # AFTER placement so the memory plan stays single-device: the DiT replica lives
            # entirely on a SECOND CUDA device, gated on that device's free VRAM. A
            # single-GPU host, an offload plan, or a quantised DiT all fall through to the
            # single-device path with the reason in the resolved record.
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
                speed_mode = effective_speed,
                logger = logger,
            )
            if cfg_parallel_proxy is not None:
                # Until the _VideoLoadState commit, the proxy (daemon worker, replica
                # VRAM, cuDNN patch) is owned by this stash; a cancel/failure before
                # commit is torn down via _rollback_precommit_cfg_parallel, token-scoped.
                self._precommit_cfg_parallel = (_load_token, pipe, cfg_parallel_proxy)
            if cfg_parallel_proxy is not None and cache_engaged:
                # The cache engaged on the primary BEFORE the proxy existed; re-engage
                # THROUGH the proxy so the replica carries the same hooks and each branch's
                # cache state matches the single-GPU run (the bit-identity precondition).
                # If the primary-only cache cannot be removed, reapplying through the proxy
                # double-hooks the primary and desyncs the branches, so fail the load.
                if not _disengage_step_cache(
                    cfg_parallel_proxy._primary,
                    reason = "re-engaging through the cfg-parallel proxy",
                    logger = logger,
                ):
                    raise RuntimeError(
                        "could not disable the primary-only step cache before installing "
                        "CFG-parallel cache hooks; reload the video model before generating"
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
                    # The pre-commit CFG-parallel stash still references the pipe, so
                    # _run_load's error handler rolls back the proxy, its worker, the
                    # replica VRAM, and the cuDNN patch even though nothing was committed.
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
                    # Already filtered to just the engaged optimisations.
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
            # Post-commit so a real request is never blocked behind an uncommitted load;
            # the thread re-checks the token and yields to any generation that arrived
            # first (which then absorbs the warmup itself, the pre-prewarm behaviour).
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
        compile/trace warmup (see the module-level prewarm constants for the numbers).
        Runs on a daemon thread under ``_generate_lock`` and registers as the active
        cancellable job (so unload / new load / cancel_generate abort it at a step
        boundary), never touching ``_gen`` progress. On failure or cancellation the
        load keeps the pre-prewarm behaviour: the first real generation pays."""
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
                # Default guidance keeps a real run's CFG branch structure (both
                # guider branches trace), mirroring generate().
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
                # Drop the warmup's step-cache residuals so the next real generation
                # starts like an unwarmed load (generate() also resets; defence in depth).
                if state.transformer_cache:
                    self._reset_step_cache(pipe)
                # The warmup just paid the compile: persist the Mega-cache bundle now
                # (env-gated, idempotent) instead of at the first real generation.
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
        init_image: Optional[str] = None,
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
            # Image gate up front (cheap, no decode) so the POST 400s synchronously instead
            # of reporting a failed job: an image-to-video family needs a source image, a
            # text-only family has no ``image`` kwarg to take one. generate() re-checks for
            # direct callers. getattr-guarded: tests fake _state without a family.
            fam = getattr(self._state, "family", None)
            if fam is not None:
                if getattr(fam, "image_conditioned", False) and not (init_image or "").strip():
                    raise ValueError(
                        f"{fam.name} is an image-to-video model: attach a source image "
                        "to animate."
                    )
                if init_image and not getattr(fam, "image_conditioned", False):
                    raise ValueError(
                        f"{fam.name} is a text-to-video model and does not take a source image."
                    )
            # A background compile prewarm may hold _generate_lock. Signal its dedicated
            # cancel handle BEFORE registering ours so the real job preempts the warmup at
            # its next step boundary instead of queueing behind it (and so unload/cancel
            # don't point at the wrong event once _active_generate_cancel is overwritten).
            if self._prewarm_cancel is not None:
                self._prewarm_cancel.set()
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
                init_image = init_image,
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
                # Covers a worker that failed before reaching generate()'s finally; identity-guarded
                # so a direct generate() that re-registered keeps its cancel handle.
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
        init_image: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        import torch

        # begin_generate passes its already-registered event; a direct call makes its own.
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
                # Image-conditioned families (WanImageToVideoPipeline) REQUIRE a source image;
                # text-only families have no ``image`` kwarg to feed one to. Both mismatches are
                # client input -> ValueError (the route/worker map it to a 400-style message).
                if fam.image_conditioned:
                    if not (init_image or "").strip():
                        raise ValueError(
                            f"{fam.name} is an image-to-video model: attach a source image "
                            "to animate."
                        )
                    from .diffusion import _decode_b64_image
                    from PIL import Image

                    init_pil = _decode_b64_image(init_image, mode = "RGB")
                    # The pipeline derives the latent grid from height/width and encodes the
                    # image at that size; resize here so the conditioning frame matches the
                    # snapped output size exactly (no center-crop surprises).
                    if init_pil.size != (width, height):
                        init_pil = init_pil.resize((width, height), Image.LANCZOS)
                    kwargs["image"] = init_pil
                elif init_image:
                    raise ValueError(
                        f"{fam.name} is a text-to-video model and does not take a source image."
                    )
                if fam.guidance_via_guider:
                    # HunyuanVideo-1.5: __call__ has no guidance kwarg; CFG scale is a guider
                    # attribute set per request (near-1 scales auto-disable CFG in the guider).
                    pipe.guider.guidance_scale = float(guidance)
                else:
                    kwargs[fam.cfg_kwarg] = guidance
                if negative_prompt and "negative_prompt" in call_params:
                    kwargs["negative_prompt"] = negative_prompt
                # LTX-2 takes frame_rate (shapes audio length); others fix their rate, fps only at export.
                if "frame_rate" in call_params:
                    kwargs["frame_rate"] = float(out_fps)
                # Dual-DiT MoE: thread the low-noise expert's guidance kwarg only when the family
                # declares one AND the signature accepts it -- WanPipeline raises if guidance_scale_2
                # is passed with boundary_ratio=None (pipeline_wan.py:322); TI2V-5B never reaches here.
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
                    # No cooperative _interrupt (the pipeline never checks it), so cancellation
                    # must unwind the denoise loop via an exception.
                    if cancel.is_set():
                        raise _VideoGenerationCancelled()
                    _tick(done)

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step
                    progress_ctx = contextlib.nullcontext()
                else:
                    # HunyuanVideo-1.5 has no step callback; each scheduler.step is one denoise
                    # step, so wrap it for progress + cancel and restore afterwards.
                    progress_ctx = _scheduler_step_progress(pipe, _on_scheduler_step)

                # An AUTO cache decision is re-checked against the ACTUAL step count: a
                # many-step request gains FBCache even when the load's default kept it off,
                # a few-step request drops it. Explicit choices never toggle.
                if state.cache_auto:
                    # All-or-none across MoE experts, like the load path: a mixed toggle
                    # is rolled back and reported uncached.
                    def _toggle_cache(view: Any, expert_name: str) -> Optional[str]:
                        return maybe_toggle_step_cache(
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

                    toggled, toggle_partial_reason = _step_cache_all_or_none(
                        pipe, fam, _toggle_cache, logger = logger
                    )
                    if toggled != state.transformer_cache:
                        # _VideoLoadState is frozen; track the toggle that already
                        # happened so status() reports the true cache state.
                        object.__setattr__(state, "transformer_cache", toggled)
                        entry = (state.resolved or {}).get("transformer_cache")
                        if isinstance(entry, dict):
                            entry["value"] = toggled or "off"
                            entry["reason"] = toggle_partial_reason or (
                                f"auto: {steps}-step generation "
                                + ("reaches" if toggled else "is below")
                                + f" {FBCACHE_MIN_STEPS}"
                            )
                elif state.transformer_cache == TC_MAGCACHE:
                    # An EXPLICIT magcache never toggles off, but its ratio curve, retention window,
                    # and skip budget are interpolated over the CONFIGURED step count: a clip at a
                    # different step count re-engages (marker carries "#s{steps}") to keep skips
                    # aligned. This only re-sizes the already-engaged cache; the on choice is
                    # preserved. Transactional across MoE experts (like the load / AUTO paths):
                    # refuse to stack a fresh cache over one whose removal failed, and roll back a
                    # mixed resize so status never reports MagCache over an asymmetric pair.
                    def _resize_explicit_magcache(view: Any, expert_name: str) -> Optional[str]:
                        transformer = getattr(view, "transformer", None)
                        marker = getattr(transformer, "_unsloth_step_cache", None)
                        # endswith, not substring: "#s5" would match inside "#s50".
                        if not marker or str(marker).endswith(f"#s{int(steps)}"):
                            return TC_MAGCACHE  # already sized for these steps
                        if not _disengage_step_cache(
                            transformer,
                            reason = f"explicit magcache re-interpolating for {steps} steps",
                            logger = logger,
                        ):
                            raise RuntimeError(
                                "could not disable the existing MagCache before resizing it "
                                f"for {steps} steps; reload the video model before generating"
                            )
                        return apply_step_cache(
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

                    resized, resize_reason = _step_cache_all_or_none(
                        pipe, fam, _resize_explicit_magcache, logger = logger
                    )
                    object.__setattr__(state, "transformer_cache", resized)
                    entry = (state.resolved or {}).get("transformer_cache")
                    if isinstance(entry, dict):
                        entry["value"] = resized or "off"
                        entry["reason"] = resize_reason or (
                            f"explicit MagCache resized for {steps} steps"
                            if resized
                            else f"MagCache could not be resized for {steps} steps"
                        )
                if state.transformer_cache:
                    self._reset_step_cache(pipe)
                # Dual-GPU CFG parallelism: resolve this generation's routing AFTER the
                # cache toggle (the plan keys on the engaged cache state -- parallel is
                # bit-identical only with the cache on or an uncompiled stack). Planning
                # must never fail a generation: a planner error pins the sequential path.
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
                    # Unwinding by exception skips the pipeline's end-of-call maybe_free_model_hooks();
                    # under offload the onloaded modules would stay on the GPU, so free them here.
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
                # warm bundle when saving is enabled. Idempotent + best-effort.
                try:
                    # A STATIC compile (speed=max) makes new inductor artifacts per
                    # (width, height, frames), so register this shape first: an uncovered
                    # shape re-dirties the context so the save rewrites the enriched bundle,
                    # otherwise a post-hit ctx.saved stays true and later resolutions/frame
                    # counts silently recompile every restart (mirrors the image backend).
                    compile_cache.register_shape(
                        state.compile_cache_ctx,
                        (width, height, frames),
                        static = "compiled" in (state.speed_optims or ())
                        and compiled_shapes_are_static(pipe, state.speed_mode),
                    )
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
                # A cancel during the blocking export/mux must still discard the clip; re-check
                # before it is returned and persisted.
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
            # generate() swaps in a bare {"active": False} before the worker records the terminal
            # dict; report active across that gap so a poller sees active drop only with a terminal phase.
            if self._generate_job_active:
                gen["active"] = True
        gen.setdefault("active", False)
        # Mirror the image endpoint's field names (total_steps / fraction) alongside the
        # native "total": the two generate-progress APIs used to disagree, so a client
        # polling the image shape against video read total_steps=null / fraction=0 while
        # the step counter advanced.
        total = int(gen.get("total") or 0)
        step = int(gen.get("step") or 0)
        gen["total_steps"] = total
        gen["fraction"] = min(1.0, step / total) if total > 0 else 0.0
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
            # Free the CFG-parallel replica on its device and restore the pipe's
            # single-device shape before the pipe is dropped.
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
        # Barrier: wait for the signalled generation to exit before freeing the pipeline, or we
        # report the VRAM free (and let the arbiter start another load) while the clip still holds it.
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
                "image_input": False,
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
            # True for image-to-video families: the UI shows the source-image control and
            # requires an image before submitting.
            "image_input": fam.image_conditioned,
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

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

from .diffusion_attention import apply_attention_backend, select_attention_backend
from .diffusion_cache import apply_step_cache, normalize_transformer_cache
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
    SPEED_OFF,
    apply_speed_optims,
    resolve_speed_mode,
    restore_backend_flags,
    snapshot_backend_flags,
)
from .diffusion_auto_policy import _QUANT_STEADY_FACTOR, build_resolved_record
from .diffusion_transformer_quant import (
    dense_transformer_supported,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)
from .video_families import (
    VIDEO_CANCELLED_MSG,
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
    # Dense transformer quant actually engaged ("int8" | "fp8" | "nvfp4" | "mxfp8") or
    # None. Mirrors the image backend's _LoadState.transformer_quant: on a pipeline-kind
    # load the dense DiT(s) can be torchao-quantised in place onto the low-precision
    # tensor cores; None means they run at their loaded (bf16) precision.
    transformer_quant: Optional[str] = None
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
    ) -> VideoFamily:
        """Cheap, network-free validation shared by the route and the load path."""
        kind = resolve_video_model_kind(gguf_filename, model_kind)
        fam = detect_video_family(repo_id, family_override) or (
            detect_video_family(f"{repo_id}/{gguf_filename}")
            if gguf_filename and not family_override
            else None
        )
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
            root = Path(repo_id).expanduser()
            if root.is_dir():
                from .diffusion_families import resolve_local_gguf_child

                try:
                    resolve_local_gguf_child(root, gguf_filename or "")
                except Exception as exc:  # noqa: BLE001 -- surface as client input error
                    raise ValueError(str(exc)) from exc
            elif repo_id.startswith(("/", "~", "./", "../")) and not root.is_file():
                raise ValueError(f"Local model path '{repo_id}' does not exist.")
        # Reject a malformed transformer_quant scheme cheaply, before the GPU handoff
        # (normalize_transformer_quant raises ValueError on an unknown scheme). It applies
        # only on pipeline-kind loads (the dense DiT from the base repo); an ignored value
        # on a gguf/single_file load is left to the loader, matching the image backend.
        normalize_transformer_quant(transformer_quant)
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
                transformer_quant = transformer_quant,
                model_kind = model_kind,
                _load_token = token,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        try:
            fam = detect_video_family(kwargs["repo_id"], kwargs.get("family_override"))
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
            # companions download inside from_pretrained (which resumes from the
            # cache, so a cancelled pull costs nothing).
            if kwargs.get("gguf_filename") and not Path(kwargs["repo_id"]).expanduser().exists():
                from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
                hf_hub_download_with_xet_fallback(
                    kwargs["repo_id"],
                    kwargs["gguf_filename"],
                    kwargs.get("hf_token"),
                    cancel_event = self._cancel_event,
                )
            self.load_pipeline(**kwargs)
            with self._lock:
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            if self._load_token != token:
                return
            logger.error("video.load_failed: %s", exc)
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))

    def _estimate_download_bytes(
        self,
        repo_id: str,
        gguf_filename: Optional[str],
        base: str,
        hf_token: Optional[str],
        kind: str,
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
                for sibling in info.siblings or []:
                    name, size = sibling.rfilename, sibling.size or 0
                    if not name.endswith((".safetensors", ".json", ".model", ".txt")):
                        continue
                    # A GGUF/single-file load replaces the base repo's DiT, and the
                    # LTX-2 base repo ships its text encoder TWICE (two shard
                    # namings); count only what from_pretrained will pull.
                    if kind != "pipeline" and name.startswith("transformer/"):
                        continue
                    if name.startswith("text_encoder/diffusion_pytorch_model"):
                        continue
                    total += int(size)
            return total or None
        except Exception:  # noqa: BLE001 -- progress totals are best-effort only
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
        return _progress(
            phase,
            downloaded_bytes = int(downloaded),
            expected_bytes = int(expected) if expected else None,
        )

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
        model_kind: Optional[str] = None,
        _load_token: Optional[int] = None,
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
        )
        kind = resolve_video_model_kind(gguf_filename, model_kind)
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
        self._teardown_state()

        target = resolve_diffusion_device_target()
        device = target.device
        # Video DiTs are bf16-native; fp16 overflows them, so a resolved fp16
        # promotes to float32 (the same rule as the fp16-incompatible image
        # families). CPU stays float32.
        dtype = target.dtype
        if fam.fp16_incompatible and dtype is torch.float16:
            dtype = torch.float32

        # ── memory plan: family-table resident estimate + frames-aware headroom.
        device_memory = snapshot_device_memory(target)
        components = fam.bf16_components_gb
        mib_per_gb = 1000.0**3 / (1024.0 * 1024.0)
        if kind == "pipeline":
            model_dense_mib = int(sum(components) * mib_per_gb) if components is not None else None
            companion_mib = None
        else:
            checkpoint_path = self._resolve_checkpoint_path(repo_id, gguf_filename, hf_token)
            size_mib = file_size_mib(str(checkpoint_path))
            model_dense_mib = None
            if kind == "gguf":
                transformer_mib = estimate_gguf_resident_mib(size_mib)
            else:
                transformer_mib = estimate_safetensors_dense_mib(size_mib)
            companion_mib = (
                int((components[1] + components[2]) * mib_per_gb)
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
                        scheme_preview, quant_mib, plan.offload_policy,
                    )
                    plan = replanned
                    quant_replanned = True

        # ── build the pipeline.
        pipeline_cls = getattr(diffusers, fam.pipeline_class)
        pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if hf_token:
            pipe_kwargs["token"] = hf_token
        if kind == "pipeline":
            pipe = pipeline_cls.from_pretrained(repo_id, **pipe_kwargs)
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
                pipe = pipeline_cls.from_pretrained(base, transformer = transformer, **pipe_kwargs)

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
        if (
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

        # ── optimisation layers, in the image backend's order: step cache FIRST
        # (compile keys its fullgraph decision off an active cache: FBCache hooks
        # graph-break, so compiling fullgraph before installing the cache crashes
        # the first cached generation), then attention, the speed profile, and
        # placement/offload last.
        effective_speed = resolve_speed_mode(speed_mode, is_gguf = kind == "gguf")
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
        # Run the step cache per expert so both denoisers cache; the engaged mode is
        # identical across experts.
        cache_engaged = None
        for view in views:
            engaged = apply_step_cache(
                view,
                mode = normalize_transformer_cache(transformer_cache),
                threshold = transformer_cache_threshold,
                logger = logger,
            )
            if view is pipe:
                cache_engaged = engaged
        attention_engaged = None
        speed_optims: tuple = ()
        for view in views:
            # apply_attention_backend / apply_speed_optims both act on ``view.transformer``;
            # calling them once per view sets the kernel and compiles each expert. The
            # engaged values match across experts (same device/family/mode), so record the
            # first pass; a dense torchao transformer on the pipeline path is not a GGUF one,
            # so is_gguf keys off the load kind (gguf) AND no quant having engaged.
            gguf_transformer = kind == "gguf" and transformer_quant_engaged is None
            engaged = apply_attention_backend(
                view,
                select_attention_backend(
                    target, attention_backend, speed_active = effective_speed != SPEED_OFF
                ),
                logger = logger,
            )
            applied = apply_speed_optims(
                view,
                target,
                is_gguf = gguf_transformer,
                family = fam,
                speed_mode = effective_speed,
                cache_active = cache_engaged is not None,
                offload_active = plan.offload_policy != "none",
            )
            if view is pipe:
                attention_engaged = engaged
                speed_optims = tuple(k for k, v in applied.items() if v)
        # A cancelled/superseded load must not place weights on the GPU the arbiter
        # may already have handed to another backend; recheck right before placement
        # (the commit below still does the final locked check).
        if _load_token is not None and _load_token != self._load_token:
            del pipe
            clear_gpu_cache()
            raise RuntimeError("Video load was cancelled or superseded.")
        offload_policy, vae_tiling = apply_memory_plan(pipe, plan, device = device, logger = logger)
        if offload_policy == "group" and len(views) > 1:
            # Group offload streams only ``pipe.transformer``; the second expert would
            # otherwise sit resident (~57 GB bf16 on the A14B) and defeat the tier.
            # model/sequential offload hook every top-level module, so only group needs
            # this. Applied through the view so the helper streams transformer_2.
            from .diffusion_memory import _apply_group_offload

            for view in views[1:]:
                if not _apply_group_offload(view, device, logger):
                    logger.warning(
                        "video.memory: group offload did not engage on the second "
                        "expert; it stays resident"
                    )
        if not vae_tiling:
            # Decode of a whole clip is the video memory peak; tiling is near-free
            # in quality and keeps the decode bounded, so it is always on.
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
                    else "GGUF video loads default to the near-lossless compile profile",
                ),
                "attention_backend": (
                    attention_backend,
                    attention_engaged or "native",
                    "cuDNN fused attention on NVIDIA when a speed profile is active",
                ),
                "transformer_cache": (
                    transformer_cache,
                    cache_engaged or "off",
                    "step cache engages on many-step schedules only",
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
                        else "not engaged (dense bf16 DiT loaded)"
                    ),
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
                # Only the optimisations that actually engaged: apply_speed_optims
                # returns every flag with True/False, and iterating the dict raw
                # would report disabled ones as active in /video/status.
                speed_optims = tuple(k for k, v in (speed_optims or {}).items() if v),
                backend_flags = backend_flags,
                attention_backend = attention_engaged,
                transformer_cache = cache_engaged,
                transformer_quant = transformer_quant_engaged,
                resolved = resolved,
            )
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
    ) -> dict[str, Any]:
        import torch
        cancel = threading.Event()
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
                    state.gguf_filename, state.repo_id, state.base_repo
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

                if state.transformer_cache:
                    self._reset_step_cache(pipe)
                try:
                    with torch.inference_mode(), progress_ctx:
                        output = pipe(**kwargs)
                except _VideoGenerationCancelled:
                    raise RuntimeError(VIDEO_CANCELLED_MSG) from None
                if cancel.is_set():
                    raise RuntimeError(VIDEO_CANCELLED_MSG)

                self._gen.update(phase = "export", eta_seconds = None)
                video_frames = output.frames[0]
                audio = getattr(output, "audio", None)
                audio_track = audio[0] if fam.has_audio and audio is not None else None
                mp4_bytes = self._encode_mp4(
                    video_frames, out_fps, audio_track, pipe if fam.has_audio else None
                )
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
        gen = dict(self._gen)
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
            # A GGUF video load may have installed the process-wide compiled GGUF
            # dequantizer; restore the stock kernels so a later load that asked for
            # speed_mode=off gets the bit-identical path (mirrors the image unload).
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
                "has_audio": False,
                "defaults": None,
                "resolved": None,
            }
        fam = state.family
        default_steps, default_guidance = default_video_generation_params(
            state.gguf_filename, state.repo_id, state.base_repo
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

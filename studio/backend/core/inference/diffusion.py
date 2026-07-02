# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local diffusion (text-to-image) backend.

A torch-only singleton: it dequantises a single-file GGUF on-device via
``GGUFQuantizationConfig`` and pulls the rest of the pipeline (VAE, text
encoders, scheduler) from the matching base repo. torch/diffusers are imported
lazily so this stays importable in a no-torch runtime. ``begin_load`` runs on a
background thread; poll ``load_progress`` for the download bar. GPU-handoff
policy lives in the arbiter the routes call, not here.
"""

from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger
from utils.hardware import clear_gpu_cache

from .diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    DiffusionFamily,
    detect_family_for_pick,
    resolve_base_repo,
    resolve_local_gguf_child,
)
from .diffusion_device import (
    DiffusionDeviceTarget,
    diffusion_device_target_from_torch_device,
    resolve_diffusion_device_target,
)
from .diffusion_memory import (
    OFFLOAD_NONE,
    apply_memory_plan,
    estimate_gguf_resident_mib,
    estimate_image_runtime_mib,
    file_size_mib,
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
from .diffusion_attention import (
    apply_attention_backend,
    select_attention_backend,
)
from .diffusion_cache import apply_step_cache
from .diffusion_precision import quantize_text_encoders
from .diffusion_prequant import (
    load_prequantized_transformer,
    resolve_prequant_source,
)
from .diffusion_transformer_quant import (
    DEFAULT_MIN_LINEAR_FEATURES,
    dense_transformer_supported,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)

logger = get_logger(__name__)


@dataclass(frozen = True)
class _LoadState:
    """Everything about the currently-loaded pipeline, swapped as one unit."""

    pipe: Any
    family: Any
    repo_id: str
    base_repo: str
    device: str
    dtype: str
    cpu_offload: bool
    # The resolved memory profile (Phase 2A). Appended with defaults so older
    # positional constructions (and the back-compat status shape) keep working.
    offload_policy: str = OFFLOAD_NONE
    vae_tiling: bool = False
    memory_mode: str = "auto"
    # The opt-in speed profile (Phase 3).
    speed_mode: str = SPEED_OFF
    speed_optims: tuple = ()
    # Process-wide torch backend flags (TF32 / cudnn.benchmark) captured before the
    # speed layer mutated them, restored on unload so a later `off` load is not
    # contaminated by this one's globals. None when nothing was changed.
    backend_flags_before: Optional[dict] = None
    # Text-encoder quantisation actually engaged: "fp8" | "nvfp4" | None (Phase 2B/2C).
    text_encoder_quant: Optional[str] = None
    # Transformer quant actually engaged on the opt-in dense fast path: "int8" | "fp8"
    # | "nvfp4" | "mxfp8" | None. None means the default GGUF transformer was loaded.
    transformer_quant: Optional[str] = None
    # Attention backend engaged via the diffusers dispatcher (e.g. "_native_cudnn"), or
    # None for the default SDPA. Set before compile; orthogonal to the weight quant.
    attention_backend: Optional[str] = None
    # Step cache engaged ("fbcache") or None. Opt-in, for many-step models.
    transformer_cache: Optional[str] = None


@dataclass
class _LoadingState:
    """An in-flight background load, polled for download progress."""

    repo_id: str
    base_repo: str
    expected_bytes: int = 0
    error: Optional[str] = None


@dataclass
class _GenState:
    """An in-flight generation, updated per denoising step for the progress bar."""

    total_steps: int
    step: int = 0
    # Set when the first step finishes; the ETA rate is measured from there so the
    # slower first step (warmup) doesn't skew it.
    first_step_at: float = 0.0
    # Computed once per step (in the callback) so it's stable between polls.
    eta_seconds: Optional[float] = None


def _estimate_eta(total_steps: int, step: int, first_step_at: float, now: float) -> Optional[float]:
    """Seconds remaining, from the average step time measured after the first step.
    None until at least one step has elapsed since the first."""
    steps_since_first = step - 1
    if not first_step_at or steps_since_first <= 0:
        return None
    per_step = (now - first_step_at) / steps_since_first
    return max(0.0, (total_steps - step) * per_step)


def _resolve_diffusion_compute_dtype(fam: Optional[DiffusionFamily], dtype: Any) -> Any:
    """Promote float16 -> float32 for fp16-incompatible families (e.g. Z-Image),
    whose activations overflow float16's finite range and render a black image.
    Every other dtype/family passes through unchanged."""
    if fam is None or not getattr(fam, "fp16_incompatible", False):
        return dtype
    import torch

    return torch.float32 if dtype == torch.float16 else dtype


class DiffusionBackend:
    """Holds at most one loaded diffusers pipeline. All mutations are serialised."""

    def __init__(self) -> None:
        # _lock serialises the small state mutations (the load swap, _loading,
        # _load_token, _gen). status() / load_progress() / generate_progress()
        # read those references WITHOUT it, so polling never blocks a slow load.
        self._lock = threading.Lock()
        # _generate_lock serialises generations and is the ONLY lock the denoise
        # holds, so a long generation never blocks status()/unload()/a new load.
        self._generate_lock = threading.Lock()
        self._state: Optional[_LoadState] = None
        self._loading: Optional[_LoadingState] = None
        # Bumped on every begin_load and unload so a worker whose load was
        # superseded (a new load) or cancelled (unload, incl. an arbiter eviction)
        # neither commits its pipeline nor stamps progress onto the current load.
        self._load_token = 0
        # Set by unload() to abort an in-flight download (which runs without the
        # lock, like the chat backend), so an eviction/unload can preempt a slow
        # load instead of blocking on the lock for the whole download.
        self._cancel_event = threading.Event()
        # The cancel Event of the generation currently in flight (or None). Set
        # under _lock by unload() / a superseding load to abort that specific
        # denoise (its step callback flips pipe._interrupt). Per-generation rather
        # than one shared flag the next generate would clear, so a cancel can't be
        # lost to a racing generate nor leak onto the wrong one.
        self._active_generate_cancel: Optional[threading.Event] = None
        # The callback mutates _gen and generate_progress() reads it, both lock-free,
        # so per-step progress polling stays live during a generation.
        self._gen: Optional[_GenState] = None

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    def _pick_device_and_dtype(self) -> tuple[str, Any]:
        """(device, dtype) for the current host. Thin wrapper over the device
        policy module, kept as a method so tests can still monkeypatch it."""
        target = resolve_diffusion_device_target()
        return target.device, target.dtype

    def _resolve_device_target(self, fam: Optional[DiffusionFamily]) -> DiffusionDeviceTarget:
        """The device target with the family fp16 guard applied.

        Routes through _pick_device_and_dtype() (so a monkeypatched override still
        drives the result), then promotes float16 -> float32 for fp16-incompatible
        families (Z-Image), rebuilding the target so dtype + capability flags stay
        consistent with the effective dtype.
        """
        device, dtype = self._pick_device_and_dtype()
        effective = _resolve_diffusion_compute_dtype(fam, dtype)
        if effective is not dtype:
            logger.warning(
                "diffusion.dtype_promoted: family=%s float16 -> float32 (fp16-incompatible)",
                getattr(fam, "name", None),
            )
        return diffusion_device_target_from_torch_device(device, effective)

    def _resolve_gguf_path(self, repo_id: str, gguf_filename: str, hf_token: Optional[str]) -> str:
        local_root = Path(repo_id).expanduser()
        if local_root.exists():
            return str(resolve_local_gguf_child(local_root, gguf_filename))
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id, gguf_filename, token = hf_token)

    def _prefetch_files(
        self,
        repo_id: str,
        gguf_filename: Optional[str],
        base: str,
        base_files: list[str],
        hf_token: Optional[str],
    ) -> None:
        """Pre-download the GGUF + the given ``base_files`` into the HF cache,
        WITHOUT the lock and honoring ``_cancel_event``, so load_pipeline's
        from_single_file / from_pretrained hit the cache and the heavy download can
        be preempted by an unload/eviction. Raises ``RuntimeError("Cancelled")``."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # GGUF transformer (hub repos only; a local path is already on disk).
        if gguf_filename and not Path(repo_id).expanduser().exists():
            hf_hub_download_with_xet_fallback(
                repo_id, gguf_filename, hf_token, cancel_event = self._cancel_event
            )
        # Base repo (VAE / text-encoder / scheduler); list comes from the estimate.
        for rfilename in base_files:
            if self._cancel_event.is_set():
                raise RuntimeError("Cancelled")
            hf_hub_download_with_xet_fallback(
                base, rfilename, hf_token, cancel_event = self._cancel_event
            )

    def validate_load_request(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        family_override: Optional[str] = None,
    ) -> DiffusionFamily:
        """Cheap, network-free validation shared by the route (before it evicts the
        chat model) and both load paths, so an unloadable pick fails BEFORE the GPU
        handoff. Raises ValueError for a missing gguf_filename or undetectable
        family, and ValueError/FileNotFoundError for a bad local GGUF path. Touches
        no GPU, network, or state."""
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: this backend loads single-file GGUF checkpoints only."
            )
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None:
            raise ValueError(
                f"Could not infer a diffusion family for '{repo_id}'. Pass family_override (z-image)."
            )
        # Reject a bad LOCAL pick now (the same checks the load would hit later), so
        # the route never evicts a working chat model for a request that can't load.
        # A path-shaped repo_id (absolute / ~ / ./ / ..) is meant to be on disk, so a
        # missing one is an error here; a bare "org/name" id is a remote HF repo and
        # is left for the background load to resolve.
        local_root = Path(repo_id).expanduser()
        if local_root.exists():
            resolve_local_gguf_child(local_root, gguf_filename)
        elif (
            # POSIX path-shaped, a "."/".." prefix (covers ./ ../ and their Windows .\ ..\
            # forms), a Windows separator anywhere (never present in a bare "org/name" HF
            # id), or an absolute path on this OS.
            repo_id.startswith(("/", "\\", "~", ".")) or "\\" in repo_id or local_root.is_absolute()
        ):
            raise FileNotFoundError(f"Local model path does not exist: {repo_id}")
        return fam

    # ── Background load + progress ─────────────────────────────────────────

    def begin_load(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """Validate, then run the (slow) load on a daemon thread. Returns at once."""
        # A blank token (the Studio default when none is configured) must mean
        # "anonymous", not an explicit empty credential the Hub rejects with 401.
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        fam = self.validate_load_request(
            repo_id, gguf_filename = gguf_filename, family_override = family_override
        )

        with self._lock:
            # Allow starting over a previously-failed load, but not over a live one.
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A diffusion load is already in progress.")
            self._load_token += 1
            token = self._load_token
            # Best-effort download preemption only; the token (not this event) is
            # the real guard that a superseded worker can't commit its pipeline.
            self._cancel_event.clear()
            # Seed with the family fallback; the worker resolves the real base
            # (a network lookup) and updates this, so begin_load never blocks.
            self._loading = _LoadingState(repo_id = repo_id, base_repo = fam.base_repo)

        threading.Thread(
            target = self._run_load,
            kwargs = dict(
                repo_id = repo_id,
                gguf_filename = gguf_filename,
                base_repo = base_repo,
                family_override = family_override,
                hf_token = hf_token,
                cpu_offload = cpu_offload,
                memory_mode = memory_mode,
                speed_mode = speed_mode,
                text_encoder_quant = text_encoder_quant,
                transformer_quant = transformer_quant,
                transformer_quant_fast_accum = transformer_quant_fast_accum,
                transformer_prequant_path = transformer_prequant_path,
                attention_backend = attention_backend,
                transformer_cache = transformer_cache,
                transformer_cache_threshold = transformer_cache_threshold,
                _load_token = token,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        token = kwargs.get("_load_token")
        try:
            # Resolve the base repo and estimate sizes on this thread (both network
            # calls) so begin_load returns instantly; the bar shows raw bytes until
            # the total lands. This is the only writer of _loading's fields here.
            fam = detect_family_for_pick(
                kwargs["repo_id"], kwargs.get("gguf_filename"), kwargs.get("family_override")
            )
            base = _resolve_base_repo(
                kwargs["repo_id"], kwargs.get("base_repo"), fam, kwargs.get("hf_token")
            )
            kwargs["base_repo"] = base
            expected, base_files = self._estimate_download_bytes(
                kwargs["repo_id"], kwargs.get("gguf_filename"), base, kwargs.get("hf_token")
            )
            with self._lock:
                # Stamp progress only if this load is still current; a superseding
                # load (or unload) has its own token and its own _LoadingState.
                if self._load_token == token and self._loading is not None:
                    self._loading.base_repo = base
                    self._loading.expected_bytes = expected
            # Download outside the lock so unload()/an eviction can preempt the
            # multi-GB pull; load_pipeline below then assembles from the cache.
            self._prefetch_files(
                kwargs["repo_id"],
                kwargs.get("gguf_filename"),
                base,
                base_files,
                kwargs.get("hf_token"),
            )
            self.load_pipeline(**kwargs)
            with self._lock:
                # Only clear the marker if this load is still the current one; a
                # newer begin_load (or an unload) has its own token.
                if self._load_token == token:
                    self._loading = None
        except Exception as exc:  # noqa: BLE001 — surfaced to the client via load_progress
            # A cancelled/superseded load raised below; don't log it as a failure
            # or stamp its error onto whatever load is current now.
            if self._load_token != token:
                return
            logger.error("diffusion.load_failed: %s", exc)
            # Redact native paths: this error is surfaced verbatim via the
            # load-progress poll, and Studio can run as a shared server.
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))

    def load_progress(self) -> dict[str, Any]:
        """Phase + downloaded/total bytes for the in-flight load (cache-scan based)."""
        loading = self._loading
        if loading is not None and loading.error:
            return _progress("error", error = loading.error)
        if loading is None:
            return _progress("ready" if self._state is not None else None)

        downloaded = self._cache_bytes(loading.repo_id) + self._cache_bytes(loading.base_repo)
        expected = loading.expected_bytes
        # Downloads done but pipeline still dequantising / moving to GPU. The cache
        # scan can slightly exceed the estimate (extra cached quants, blob padding),
        # so clamp the reported bytes/fraction so the bar never overshoots 100%.
        if expected > 0 and downloaded >= expected * 0.999:
            return _progress("finalizing", min(downloaded, expected), expected, 1.0)
        fraction = min(downloaded / expected, 1.0) if expected > 0 else 0.0
        return _progress("downloading", downloaded, expected, fraction)

    def loading_repo_ids(self) -> tuple[str, ...]:
        """Repo ids an in-flight background load is downloading (empty when idle).

        The delete-cached guard needs this: during a load ``status()["loaded"]`` is
        still False, but deleting the target repo (or its companion base) would yank
        blobs and snapshot files from under the download/assembly."""
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            return tuple(r for r in (loading.repo_id, loading.base_repo) if r)

    @staticmethod
    def _estimate_download_bytes(
        repo_id: str, gguf_filename: Optional[str], base_repo: str, hf_token: Optional[str]
    ) -> tuple[int, list[str]]:
        """Total download size for the progress bar, plus the base-repo files to
        fetch (the prefetch reuses this list, so the base is listed only once)."""
        from huggingface_hub import HfApi

        api = HfApi()
        total = 0
        base_files: list[str] = []
        try:
            # Skip the Hub size lookup for a LOCAL gguf path: model_info(repo_id) would
            # raise on a filesystem path and (caught below) skip the base-repo lookup too,
            # so the companion VAE/text-encoder files would never be prefetched and would
            # instead download synchronously under the load lock.
            if gguf_filename and not Path(repo_id).expanduser().exists():
                info = api.model_info(repo_id, files_metadata = True, token = hf_token)
                total += sum(s.size or 0 for s in info.siblings if s.rfilename == gguf_filename)
            base_info = api.model_info(base_repo, files_metadata = True, token = hf_token)
            for s in base_info.siblings:
                if _base_file_downloaded(s.rfilename):
                    base_files.append(s.rfilename)
                    total += s.size or 0
        except Exception as exc:  # noqa: BLE001 — estimate is best-effort
            logger.warning("diffusion.size_estimate_failed: %s", exc)
        return total, base_files

    @staticmethod
    def _cache_bytes(repo_id: str) -> int:
        from huggingface_hub import constants

        blobs = Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "blobs"
        total = 0
        try:
            for entry in blobs.iterdir():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue  # broken symlink / unreadable
        except OSError:
            return 0  # repo not in cache yet
        return total

    @staticmethod
    def _companion_cache_bytes(base: str) -> int:
        """Resident companion (VAE + text-encoder) size for the memory plan.

        For a hub base repo this is the cached blob total (``_cache_bytes``). For a
        LOCAL diffusers base directory the blob cache is empty, so sum the on-disk
        component weights instead, excluding ``transformer/`` (the GGUF supplies the
        transformer). Without this a local base folds its multi-GB VAE / text-encoder
        weights to zero and auto planning can pick a resident placement that OOMs."""
        local = Path(base).expanduser()
        if local.is_dir():
            total = 0
            for f in local.rglob("*"):
                if f.suffix.lower() not in (".safetensors", ".bin", ".pt", ".ckpt"):
                    continue
                try:
                    rel = f.relative_to(local)
                except ValueError:
                    continue
                if rel.parts and rel.parts[0] == "transformer":
                    continue  # supplied by the GGUF single-file; not resident here
                try:
                    total += f.stat().st_size
                except OSError:
                    continue
            return total
        return DiffusionBackend._cache_bytes(base)

    # ── Synchronous load / generate / unload ───────────────────────────────

    def load_pipeline(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        _load_token: Optional[int] = None,
    ) -> dict[str, Any]:
        # Validate first (cheap, no torch/diffusers) so a direct call with a bad
        # family fails with ValueError even in a no-diffusers runtime. Sanitize the
        # token here too (direct callers bypass begin_load): a blank string must
        # load anonymously, not 401 as an explicit empty credential.
        hf_token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
        fam = self.validate_load_request(
            repo_id, gguf_filename = gguf_filename, family_override = family_override
        )
        base = _resolve_base_repo(repo_id, base_repo, fam, hf_token)
        target = self._resolve_device_target(fam)
        device, dtype = target.device, target.dtype

        import diffusers

        # Signal an in-flight denoise to abort, then take _generate_lock to WAIT for
        # it to actually exit before allocating the replacement: a load is about to
        # claim VRAM, so unlike unload() it must not overlap a still-live pipeline.
        # The cancel makes that wait ~one step (or the rest of the denoise for a
        # pipeline that ignores the step callback).
        with self._lock:
            # Bail BEFORE signalling any cancel if this load was already superseded (an
            # unload/eviction or a newer load bumped the token while we were resolving /
            # downloading). Otherwise a stale worker would abort an unrelated, still-live
            # generation from the CURRENT model and only then discover it has nothing to do.
            if _load_token is not None and _load_token != self._load_token:
                raise RuntimeError("Diffusion load was cancelled.")
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        with self._generate_lock:
            with self._lock:
                # Re-check under the generate lock: a newer load/unload may have superseded
                # this one while we waited for the in-flight denoise to exit.
                if _load_token is not None and _load_token != self._load_token:
                    raise RuntimeError("Diffusion load was cancelled.")

                # Free the old pipeline before allocating the new one so two
                # checkpoints never sit in VRAM at once.
                self._unload_locked()

                gguf_path = self._resolve_gguf_path(repo_id, gguf_filename, hf_token)
                transformer_cls = getattr(diffusers, fam.transformer_class)
                pipeline_cls = getattr(diffusers, fam.pipeline_class)

                # Decide placement up front (the weights are still on CPU, so free VRAM is
                # the real budget) -- this also doubles as the dense-quant preflight: the
                # dense bf16 transformer must fit resident, so the fast path is offered only
                # when the plan is `none`.
                plan = self._plan_memory(target, gguf_path, base, fam, memory_mode, cpu_offload)

                # Opt-in fast path: load the DENSE bf16 transformer and torchao-quantise it
                # (int8 / fp8 / fp4 tensor cores), which beats GGUF's bf16-rate per-matmul
                # dequant on both speed and quality, at the cost of a higher-memory dense
                # load. Gated on CUDA + bf16 + a resident fit; ANY failure (unsupported arch
                # / scheme, OOM, partial quant) falls back to the GGUF build below.
                pipe = None
                transformer_quant_engaged = None
                if (
                    normalize_transformer_quant(transformer_quant) is not None
                    and dense_transformer_supported(target)
                    and plan.offload_policy == OFFLOAD_NONE
                ):
                    try:
                        pipe, transformer_quant_engaged = self._load_dense_quant_pipeline(
                            transformer_cls,
                            pipeline_cls,
                            base,
                            device,
                            dtype,
                            hf_token,
                            target,
                            transformer_quant,
                            transformer_quant_fast_accum,
                            fam = fam,
                            prequant_path = transformer_prequant_path,
                        )
                    except Exception as exc:  # noqa: BLE001 — fall back to the GGUF build
                        logger.warning(
                            "diffusion.transformer_quant_fallback: %s (loading GGUF)", exc
                        )
                        pipe = None
                        transformer_quant_engaged = None
                        # Drop the exception (and its traceback) BEFORE clearing the cache:
                        # exc.__traceback__ keeps _load_dense_quant_pipeline's frame -- and
                        # thus its partially-built dense bf16 transformer/pipe -- alive, so
                        # clear_gpu_cache() could not otherwise reclaim that VRAM before the
                        # GGUF build (the OOM-fallback path this cleanup exists for).
                        del exc
                        clear_gpu_cache()

                if pipe is None:
                    # Default: dequantise the single-file GGUF transformer on-device; the
                    # VAE / text-encoder / scheduler come from the base diffusers repo
                    # (GGUF is transformer-only).
                    transformer = transformer_cls.from_single_file(
                        gguf_path,
                        quantization_config = diffusers.GGUFQuantizationConfig(compute_dtype = dtype),
                        torch_dtype = dtype,
                        config = base,
                        subfolder = "transformer",
                        # Forward the token: the config is fetched from the (possibly gated)
                        # base repo before from_pretrained gets a chance to authenticate.
                        token = hf_token,
                    )

                    pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype, "transformer": transformer}
                    if hf_token:
                        pipe_kwargs["token"] = hf_token
                    pipe = pipeline_cls.from_pretrained(base, **pipe_kwargs)

                # Resolve the effective speed mode: GGUF models default to the
                # near-lossless `default` profile (compile is ~2.2x and sits below
                # the quant noise floor), dense models stay bit-identical `off`. An
                # explicit speed_mode (incl. "off") is honored verbatim.
                effective_speed = resolve_speed_mode(speed_mode, is_gguf = bool(gguf_filename))
                # A torchao-quantized dense transformer runs its matmuls through the
                # regional torch.compile; UNcompiled (eager) it is ~30x slower and would
                # lose to the GGUF fallback. A dense model otherwise resolves to `off`, so
                # force at least `default` (regional compile) whenever the quant engaged,
                # or the opt-in "fast" path silently commits an eager, pathologically slow
                # pipeline.
                if transformer_quant_engaged is not None and effective_speed == SPEED_OFF:
                    logger.info(
                        "diffusion.transformer_quant: forcing speed_mode=default "
                        "(quantized transformer must be compiled; eager is ~30x slower)"
                    )
                    effective_speed = SPEED_DEFAULT
                # Opt-in speed optims run BEFORE placement (channels_last / compile
                # must precede CPU offload). Snapshot the process-wide backend flags
                # first so unload can restore them: TF32 / cudnn.benchmark are global,
                # and a later `off` load must not inherit this load's settings.
                backend_flags_before = snapshot_backend_flags()
                # Pick the attention kernel BEFORE compile (compile traces attention). auto
                # upgrades to cuDNN fused attention on NVIDIA when a speed profile is active
                # (~1.18x, near-lossless); an explicit backend is honored, falling back to
                # the diffusers default if its kernel is unavailable. Orthogonal to the
                # weight quant -- it speeds the QK/PV matmuls torchao does not touch.
                attention_engaged = apply_attention_backend(
                    pipe,
                    select_attention_backend(
                        target, attention_backend, speed_active = effective_speed != SPEED_OFF
                    ),
                    logger = logger,
                )
                # Opt-in step caching (First-Block-Cache), also before compile. OFF by
                # default; for many-step models it reuses the transformer tail across steps
                # (~1.4x on Flux at LPIPS ~0.08). When engaged, compile must drop fullgraph
                # (the cache's per-step decision is a graph break), so pass it through.
                cache_engaged = apply_step_cache(
                    pipe,
                    mode = transformer_cache,
                    threshold = transformer_cache_threshold,
                    # GGUF transformers are quantized too (the default Studio path), so the
                    # cache needs the higher quantized threshold to still trigger -- not just
                    # the dense-quant fast path.
                    quant_active = transformer_quant_engaged is not None or bool(gguf_filename),
                    logger = logger,
                )
                # apply_speed_optims flips the process-global TF32 / cudnn.benchmark
                # flags. If a later step here (text-encoder quant, memory plan) then
                # raises -- e.g. OOM -- those flags would leak flipped and a subsequent
                # `off` load would no longer be bit-identical. Restore the snapshot
                # unless we reach the commit (unload restores on the happy path).
                committed = False
                try:
                    speed_applied = apply_speed_optims(
                        pipe,
                        target,
                        is_gguf = bool(gguf_filename),
                        family = fam,
                        speed_mode = effective_speed,
                        cache_active = cache_engaged is not None,
                        # The planned offload policy: group/model/sequential offload installs
                        # compiler-disabled onload hooks, so compile must drop fullgraph.
                        offload_active = plan.offload_policy != OFFLOAD_NONE,
                        logger = logger,
                    )
                    if transformer_quant_engaged is not None and not speed_applied.get("compiled"):
                        # Promotion above could not engage compile (e.g. the family is not
                        # compile-friendly, or compile_repeated_blocks failed): the quantized
                        # transformer is now running eager, which is far slower than the GGUF
                        # path it replaced. Surface it loudly rather than hiding the regression.
                        logger.warning(
                            "diffusion.transformer_quant: %s engaged but the transformer is NOT "
                            "compiled; eager torchao quant is ~30x slower than GGUF here",
                            transformer_quant_engaged,
                        )
                    # Quantise the dense companion text encoder(s) (opt-in fp8 / nvfp4),
                    # also before placement so the offload hooks move the smaller weights.
                    te_quant = quantize_text_encoders(
                        pipe,
                        target,
                        mode = text_encoder_quant,
                        logger = logger,
                    )

                    # Apply the placement planned above (from MEASURED free device memory vs
                    # the model's estimated resident size). apply_memory_plan returns the
                    # (policy, tiling) ACTUALLY engaged (it may fall back to whole-module
                    # offload, and tiling is a no-op on a pipeline with no tiling control), so
                    # status stays honest. The dense fast path already placed the pipe resident;
                    # for the `none` policy this is an idempotent re-placement.
                    effective_policy, effective_tiling = apply_memory_plan(
                        pipe, plan, device = device, logger = logger
                    )

                    self._state = _LoadState(
                        pipe = pipe,
                        family = fam,
                        repo_id = repo_id,
                        base_repo = base,
                        device = device,
                        dtype = str(dtype).replace("torch.", ""),
                        cpu_offload = effective_policy != OFFLOAD_NONE,
                        offload_policy = effective_policy,
                        vae_tiling = effective_tiling,
                        memory_mode = plan.requested_mode,
                        speed_mode = effective_speed,
                        speed_optims = tuple(k for k, v in speed_applied.items() if v),
                        backend_flags_before = backend_flags_before,
                        text_encoder_quant = te_quant,
                        transformer_quant = transformer_quant_engaged,
                        attention_backend = attention_engaged,
                        transformer_cache = cache_engaged,
                    )
                    committed = True
                finally:
                    if not committed:
                        # Restore the flags AND free the half-built pipe's VRAM: the
                        # failed load never commits _state, so nothing else reclaims it
                        # until the next unload.
                        restore_backend_flags(backend_flags_before)
                        clear_gpu_cache()

        logger.info(
            "diffusion.loaded: repo=%s base=%s device=%s offload=%s tiling=%s reasons=%s",
            repo_id,
            base,
            device,
            effective_policy,
            effective_tiling,
            "; ".join(plan.reasons),
        )
        return self.status()

    def _load_dense_quant_pipeline(
        self,
        transformer_cls: Any,
        pipeline_cls: Any,
        base: str,
        device: str,
        dtype: Any,
        hf_token: Optional[str],
        target: DiffusionDeviceTarget,
        mode: Optional[str],
        fast_accum: Optional[bool] = None,
        *,
        fam: Optional[DiffusionFamily] = None,
        prequant_path: Optional[str] = None,
    ) -> tuple[Any, str]:
        """Build the opt-in fast pipeline and return ``(pipe, engaged_scheme)``.

        Two ways to get the quantized transformer, in order:

        1. Pre-quantized: if a checkpoint is configured for the chosen scheme (an explicit
           ``prequant_path`` or the family's hosted repo), load the already-quantized
           weights onto the meta device and assign them in -- the dense bf16 never lands on
           the GPU, so the load peak is ~half and the download is smaller.
        2. Dense + quantise (fallback): load the DENSE bf16 transformer from the base repo,
           place it on the device, and torchao-quantise it in place.

        Raises if the scheme is unsupported or quantisation fails, so ``load_pipeline``
        catches it and falls back to the GGUF build. Quantisation runs ON the device and
        BEFORE the loader compiles the repeated block, so the order stays quantize ->
        compile -> placement."""
        # 1. Pre-quantized checkpoint, when one is configured for the resolved scheme.
        scheme = select_transformer_quant_scheme(target, mode)
        if scheme is None:
            # Bail BEFORE the (multi-GB) dense download: an explicit unsupported scheme
            # (e.g. fp8 on Ampere, nvfp4 off Blackwell) would otherwise materialise the
            # dense transformer and move the pipe to CUDA only to fail at quantize below --
            # a long finalization under the load lock after the old model was already
            # evicted. load_pipeline catches this and builds the GGUF pipeline instead.
            raise RuntimeError("transformer quant unsupported for this device/scheme")
        if fam is not None:
            source = resolve_prequant_source(fam, scheme, path_override = prequant_path)
            if source is not None:
                transformer = load_prequantized_transformer(
                    transformer_cls,
                    base,
                    source,
                    device = device,
                    dtype = dtype,
                    hf_token = hf_token,
                    scheme = scheme,
                    # Reject a checkpoint built with a different Linear filter than the
                    # dense path uses, so the prequant and runtime-quant models match.
                    min_features = DEFAULT_MIN_LINEAR_FEATURES,
                    # Only enforced when the caller forces fp8 fast-accum: a checkpoint that
                    # baked the other choice would ignore the request, so fall to the dense
                    # path (which applies it) instead of silently using the baked kernels.
                    fast_accum = fast_accum,
                    logger = logger,
                )
                if transformer is not None:
                    pipe = self._assemble_pipe(
                        pipeline_cls, base, transformer, dtype, hf_token, device
                    )
                    return pipe, scheme

        # 2. Fallback: materialise the dense bf16 transformer and quantise it on-device.
        transformer = transformer_cls.from_pretrained(
            base, subfolder = "transformer", torch_dtype = dtype, token = hf_token
        )
        pipe = self._assemble_pipe(pipeline_cls, base, transformer, dtype, hf_token, device)
        scheme = quantize_transformer(pipe, target, mode = mode, fast_accum = fast_accum, logger = logger)
        if scheme is None:
            raise RuntimeError("transformer quant unsupported for this device/scheme")
        return pipe, scheme

    @staticmethod
    def _assemble_pipe(
        pipeline_cls: Any,
        base: str,
        transformer: Any,
        dtype: Any,
        hf_token: Optional[str],
        device: str,
    ) -> Any:
        """Assemble the diffusers pipeline around ``transformer`` and place it on ``device``
        (a no-op for an already-placed pre-quantized transformer; it moves the companions)."""
        pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype, "transformer": transformer}
        if hf_token:
            pipe_kwargs["token"] = hf_token
        pipe = pipeline_cls.from_pretrained(base, **pipe_kwargs)
        pipe.to(device)
        return pipe

    def _plan_memory(
        self,
        target: DiffusionDeviceTarget,
        gguf_path: str,
        base: str,
        fam: DiffusionFamily,
        memory_mode: Optional[str],
        cpu_offload: bool,
    ):
        """Build the memory plan for this load: snapshot free device memory and
        estimate the model's resident footprint, then let the planner pick an
        offload policy + VAE memory savers. Kept on the backend so the cached base
        repo (companion text-encoder / VAE) feeds the size estimate."""
        device_memory = snapshot_device_memory(target)
        transformer_resident = estimate_gguf_resident_mib(file_size_mib(gguf_path))
        # The companion components (VAE + text encoders) load near their on-disk
        # size; sum whatever the prefetch placed in the base-repo cache, or -- for a
        # LOCAL diffusers base -- the on-disk component weights (the blob cache is
        # empty for a local path, which would otherwise fold multi-GB companions to 0
        # and let auto planning pick a resident placement that OOMs).
        companion = self._companion_cache_bytes(base)
        companion_mib = int(companion // (1024 * 1024)) if companion else None
        model_dense_mib = None
        if transformer_resident is not None:
            model_dense_mib = transformer_resident + (companion_mib or 0)
        # Feed the variant hint (gguf filename + base repo) next to the family name so
        # estimate_image_runtime_mib sees distilled markers ("turbo"/"schnell") that
        # detect_family normalizes out of fam.name -- distilled models need ~15% less
        # activation headroom, and over-reserving can force needless offload / tiling.
        variant_hint = " ".join(
            p for p in (fam.name, Path(gguf_path).name if gguf_path else "", base or "") if p
        )
        runtime_headroom = estimate_image_runtime_mib(width = None, height = None, family = variant_hint)
        return plan_diffusion_memory(
            target = target,
            device_memory = device_memory,
            model_dense_mib = model_dense_mib,
            companion_dense_mib = companion_mib,
            runtime_headroom_mib = runtime_headroom,
            requested_mode = memory_mode,
            explicit_offload = cpu_offload,
        )

    @staticmethod
    def _reset_step_cache(pipe: Any) -> None:
        """Clear the transformer's stateful step cache (FBCache) before a generation.

        diffusers keys FBCache residuals by cache context ("cond"/"uncond") on the
        long-lived transformer, and neither the pipeline nor the context exit resets
        them (``StateManager`` only clears via ``reset_stateful_hooks``, which no
        pipeline calls). This backend reuses one resident pipe across generations, so
        without a reset the next generation's first step compares its first-block
        residual against the PREVIOUS request's -- a tensor-shape mismatch when the
        resolution/batch changed, or a stale-cache reuse otherwise. Best-effort: a
        transformer without the hook (uncached load) is a silent no-op."""
        transformer = getattr(pipe, "transformer", None)
        reset = getattr(transformer, "reset_stateful_hooks", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001 — reset is best-effort, never fail a generation
                pass

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        # Fallbacks for a caller that passes nothing; the route always sends the
        # per-model values the UI seeds (few steps / no CFG for distilled models,
        # more steps / real CFG for full ones).
        steps: int = 9,
        guidance: float = 0.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        import torch

        # A per-generation cancel Event: unload()/a superseding load set THIS event
        # (registered under _lock below) to abort just this denoise. _generate_lock
        # serialises generations and is the only lock the denoise holds, so a slow
        # generation never blocks status()/unload()/a new load.
        cancel = threading.Event()
        with self._generate_lock:
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                # Register under _lock so unload()/a load can signal THIS generation.
                # A cancel that arrived before now either nulled _state (we raised
                # above) or targets an older generation, so nothing is lost.
                self._active_generate_cancel = cancel
            try:
                # Snapshot taken: the local `state` ref keeps the pipe alive even if
                # unload() nulls _state mid-denoise, so the call below needs no _lock.
                generator = torch.Generator(device = state.device)
                if seed is None:
                    # Draw a fresh random seed but keep it within JS's safe-integer
                    # range (< 2**53), so the reported seed round-trips through JSON
                    # and actually reproduces the image (a raw 64-bit seed would lose
                    # precision in the browser and the recipe couldn't be replayed).
                    seed = generator.seed() & ((1 << 53) - 1)
                else:
                    seed = int(seed)
                generator.manual_seed(seed)

                kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    # Most pipelines take guidance via "guidance_scale"; Qwen-Image
                    # uses "true_cfg_scale" (its distilled guidance is off).
                    state.family.cfg_kwarg: guidance,
                    "generator": generator,
                    # Generate the whole batch in one forward pass (VRAM-heavy). All
                    # share this call's seed, drawn sequentially from one generator.
                    "num_images_per_prompt": batch_size,
                }
                # Pipelines vary in which kwargs they accept (a distilled pipeline may
                # take neither a negative prompt nor a step callback), so only pass
                # those where the signature has them.
                call_params = inspect.signature(state.pipe.__call__).parameters
                if negative_prompt and "negative_prompt" in call_params:
                    kwargs["negative_prompt"] = negative_prompt

                gen = _GenState(total_steps = steps)

                def _on_step(pipe, step_index, timestep, callback_kwargs):
                    now = time.time()
                    gen.step = step_index + 1
                    if gen.first_step_at == 0.0:
                        gen.first_step_at = now
                    gen.eta_seconds = _estimate_eta(
                        gen.total_steps, gen.step, gen.first_step_at, now
                    )
                    # Preempt a long denoise on unload/eviction or a superseding load:
                    # diffusers checks pipe._interrupt and stops after the current step.
                    if cancel.is_set():
                        pipe._interrupt = True
                    return callback_kwargs

                if "callback_on_step_end" in call_params:
                    kwargs["callback_on_step_end"] = _on_step

                # Start each generation from a clean step cache: FBCache residuals from
                # a prior request on this resident pipe would otherwise be compared
                # against this generation's first step (shape mismatch on a resolution/
                # batch change, or stale reuse). No-op when no cache is engaged.
                if state.transformer_cache:
                    self._reset_step_cache(state.pipe)

                self._gen = gen
                try:
                    # inference_mode is strictly faster than the no_grad diffusers
                    # uses internally and numerically identical for inference.
                    with torch.inference_mode():
                        images = state.pipe(**kwargs).images
                finally:
                    self._gen = None
                # A cancelled denoise returns early with a partial/garbage image;
                # don't hand it back to be persisted.
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                # Return the PIL images (not yet encoded): the route embeds each
                # image's recipe and persists it via the gallery.
                return {"images": list(images), "seed": int(seed), "repo_id": state.repo_id}
            finally:
                # Deregister so a later unload/load can't poke a finished generation
                # (only if still ours — a newer generation may have replaced it).
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None

    def generate_progress(self) -> dict[str, Any]:
        """Live per-step progress for an in-flight generation (lock-free read)."""
        gen = self._gen
        if gen is None or gen.total_steps <= 0:
            return {
                "active": False,
                "step": 0,
                "total_steps": 0,
                "fraction": 0.0,
                "eta_seconds": None,
            }
        return {
            "active": True,
            "step": gen.step,
            "total_steps": gen.total_steps,
            "fraction": gen.step / gen.total_steps,  # step is 1..total, never over 1.0
            "eta_seconds": gen.eta_seconds,
        }

    def unload(self) -> dict[str, Any]:
        # Abort an in-flight download so unload/an eviction returns promptly instead
        # of waiting it out (the download runs without _lock and checks this event).
        self._cancel_event.set()
        with self._lock:
            # Abort an in-flight denoise too by setting ITS cancel event, so the step
            # callback stops it. unload does NOT take _generate_lock — it must return
            # promptly; the running generate keeps its own pipe reference, so freeing
            # _state here can't crash it, and its VRAM is reclaimed when it returns
            # (within ~one step thanks to the cancel).
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            self._unload_locked()
            # Cancel any in-flight load (its worker checks this token before
            # committing) and drop the marker so the next load starts clean.
            self._load_token += 1
            self._loading = None
        return self.status()

    def _unload_locked(self) -> None:
        state = self._state
        if state is None:
            return
        # Restore the process-wide backend flags (TF32 / cudnn.benchmark) this load
        # may have flipped, so the next `off` load is bit-identical again.
        restore_backend_flags(state.backend_flags_before)
        self._state = None
        del state
        clear_gpu_cache()

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
                "cpu_offload": False,
                "offload_policy": None,
                "vae_tiling": False,
                "memory_mode": None,
                "speed_mode": None,
                "speed_optims": [],
                "text_encoder_quant": None,
                "transformer_quant": None,
                "attention_backend": None,
                "transformer_cache": None,
            }
        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": state.family.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": state.dtype,
            "cpu_offload": state.cpu_offload,
            "offload_policy": state.offload_policy,
            "vae_tiling": state.vae_tiling,
            "memory_mode": state.memory_mode,
            "speed_mode": state.speed_mode,
            "speed_optims": list(state.speed_optims),
            "text_encoder_quant": state.text_encoder_quant,
            "transformer_quant": state.transformer_quant,
            "attention_backend": state.attention_backend,
            "transformer_cache": state.transformer_cache,
        }


def _resolve_base_repo(
    repo_id: str, base_repo: Optional[str], fam: DiffusionFamily, hf_token: Optional[str]
) -> str:
    """The companion diffusers repo: caller's base, else the GGUF repo's own
    ``base_model`` tag, else the family fallback. Shared by both load paths so a
    direct ``load_pipeline`` call resolves the variant base the same way."""
    return resolve_base_repo(fam, (base_repo or "").strip() or _hf_base_model(repo_id, hf_token))


def _hf_base_model(repo_id: str, hf_token: Optional[str]) -> Optional[str]:
    """The diffusers base repo from a GGUF repo's ``base_model`` tag, or None.

    Lets one family entry cover every variant (Turbo/full, schnell/dev, the
    2512 Qwen revision). Skipped for local paths; None on any lookup failure.
    """
    if Path(repo_id).expanduser().exists():
        return None
    try:
        from huggingface_hub import HfApi
        meta = HfApi().model_info(repo_id, token = hf_token).cardData or {}
    except Exception:  # noqa: BLE001 — best-effort; fall back to the family default
        return None
    base = meta.get("base_model")
    if isinstance(base, list):
        base = base[0] if base else None
    return base if isinstance(base, str) and base.strip() else None


def _base_file_downloaded(rfilename: str) -> bool:
    """True for base-repo files ``from_pretrained`` actually fetches.

    The transformer is supplied by the GGUF, and repo docs (``assets/``, the
    top-level README/PDF/images) are never downloaded — counting them would peg
    the progress estimate above what lands on disk, so the bar would sit short of
    100% for the whole pipeline-load phase instead of advancing to "finalizing".
    """
    if rfilename.startswith("transformer/"):
        return False
    if "/" not in rfilename:  # top-level: only the pipeline manifest is fetched
        return rfilename == "model_index.json"
    return not rfilename.startswith("assets/")


def _progress(
    phase: Optional[str],
    bytes_downloaded: int = 0,
    bytes_total: int = 0,
    fraction: float = 0.0,
    *,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "bytes_downloaded": bytes_downloaded,
        "bytes_total": bytes_total,
        "fraction": fraction,
        "error": error,
    }


_diffusion_backend: Optional[DiffusionBackend] = None


def get_diffusion_backend() -> DiffusionBackend:
    global _diffusion_backend
    if _diffusion_backend is None:
        _diffusion_backend = DiffusionBackend()
    return _diffusion_backend

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
    DiffusionFamily,
    detect_family,
    resolve_base_repo,
    resolve_local_gguf_child,
)
from .diffusion_device import (
    DiffusionDeviceTarget,
    diffusion_device_target_from_torch_device,
    resolve_diffusion_device_target,
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

    @staticmethod
    def _detect_family_for_pick(
        repo_id: str, gguf_filename: Optional[str], family_override: Optional[str]
    ) -> Optional[DiffusionFamily]:
        """Detect the family from the repo id, falling back to the combined
        path/filename for a direct local .gguf pick. The frontend splits such a
        pick into (parent dir, basename), so the family keyword can live only in
        the filename (e.g. /models/z-image-turbo-Q4_K_M.gguf) while the parent
        directory carries none; scan it too when the directory alone is
        undetectable. Only used as a fallback, so remote 'org/name' picks and
        explicit overrides behave exactly as before."""
        fam = detect_family(repo_id, family_override)
        if fam is None and gguf_filename and not family_override:
            fam = detect_family(f"{repo_id}/{gguf_filename}", family_override)
        return fam

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
        fam = self._detect_family_for_pick(repo_id, gguf_filename, family_override)
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

    def validate_load(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        family_override: Optional[str] = None,
    ) -> DiffusionFamily:
        """Cheap, pure-synchronous pre-flight: returns the resolved family or
        raises ValueError. No I/O, so a caller can reject a bad request before
        taking the GPU (begin_load and load_pipeline re-check on the load path)."""
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: this backend loads single-file GGUF checkpoints only."
            )
        fam = detect_family(repo_id, family_override)
        if fam is None:
            raise ValueError(
                f"'{repo_id}' isn't a supported image-generation model. "
                f"Supported: Z-Image, Qwen-Image, FLUX.1, FLUX.2-klein."
            )
        return fam

    def begin_load(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
    ) -> dict[str, Any]:
        """Validate, then run the (slow) load on a daemon thread. Returns at once."""
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
            fam = self._detect_family_for_pick(
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
        _load_token: Optional[int] = None,
    ) -> dict[str, Any]:
        # Validate first (cheap, no torch/diffusers) so a direct call with a bad
        # family fails with ValueError even in a no-diffusers runtime.
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
            # Check the token BEFORE cancelling: a superseded/cancelled worker (its
            # token already bumped by unload or a newer load) must not abort the
            # current model's unrelated in-flight generation on its way out.
            if _load_token is not None and _load_token != self._load_token:
                raise RuntimeError("Diffusion load was cancelled.")
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        with self._generate_lock:
            with self._lock:
                # Re-check under the lock (we dropped it to wait on _generate_lock): an
                # unload/eviction or a newer load may have superseded this one while we
                # waited, and the slow VRAM-heavy build below must not run if so.
                if _load_token is not None and _load_token != self._load_token:
                    raise RuntimeError("Diffusion load was cancelled.")

                # Free the old pipeline before allocating the new one so two
                # checkpoints never sit in VRAM at once.
                self._unload_locked()

                # Dequantise the GGUF transformer on-device; the VAE / text-encoder /
                # scheduler come from the base diffusers repo (GGUF is transformer-only).
                gguf_path = self._resolve_gguf_path(repo_id, gguf_filename, hf_token)
                transformer_cls = getattr(diffusers, fam.transformer_class)
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
                pipeline_cls = getattr(diffusers, fam.pipeline_class)
                pipe = pipeline_cls.from_pretrained(base, **pipe_kwargs)

                use_offload = bool(cpu_offload and target.supports_model_cpu_offload)
                if use_offload:
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(device)

                self._state = _LoadState(
                    pipe = pipe,
                    family = fam,
                    repo_id = repo_id,
                    base_repo = base,
                    device = device,
                    dtype = str(dtype).replace("torch.", ""),
                    cpu_offload = use_offload,
                )

        logger.info("diffusion.loaded: repo=%s base=%s device=%s", repo_id, base, device)
        return self.status()

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
                    raise RuntimeError("No diffusion model is loaded.")
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

                self._gen = gen
                try:
                    images = state.pipe(**kwargs).images
                finally:
                    self._gen = None
                # A cancelled denoise returns early with a partial/garbage image;
                # don't hand it back to be persisted.
                if cancel.is_set():
                    raise RuntimeError("Diffusion generation was cancelled.")
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
        # Signal an in-flight denoise, then WAIT on _generate_lock for it to exit
        # before freeing _state. The GPU arbiter releases diffusion ownership and
        # hands the card to chat the instant this returns, so chat must not start
        # allocating VRAM while the old pipeline is still resident — that transient
        # overlap is an OOM risk. The cancel bounds the wait to ~one step (the same
        # bound the replacement-load path in load_pipeline already relies on).
        with self._lock:
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
        with self._generate_lock:
            with self._lock:
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
            }
        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": state.family.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": state.dtype,
            "cpu_offload": state.cpu_offload,
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

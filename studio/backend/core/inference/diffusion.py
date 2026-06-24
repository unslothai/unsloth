# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local diffusion (text-to-image) backend.

A small torch-only singleton that loads a diffusers pipeline and generates
images. Single-file GGUF checkpoints are dequantised on-device via
``diffusers.GGUFQuantizationConfig``; the rest of the pipeline (VAE, text
encoders, scheduler) comes from the matching base repo (see
``diffusion_families``). torch/diffusers are imported lazily inside the methods
so this module imports cleanly in a no-torch runtime.

Loading runs on a background thread (``begin_load``) so the route returns
immediately and the frontend can poll ``load_progress`` for a download bar.

This backend owns no GPU-handoff policy: it assumes it is the heavy GPU user
while loaded. Coordinating with the chat backend lives in the GPU arbiter the
routes call, not here.
"""

from __future__ import annotations

import base64
import io
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger
from utils.hardware import clear_gpu_cache

from .diffusion_families import (
    detect_family,
    resolve_base_repo,
    resolve_local_gguf_child,
)

logger = get_logger(__name__)


def encode_png_base64(image: Any) -> str:
    """PIL image -> base64-encoded PNG string."""
    buf = io.BytesIO()
    image.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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


class DiffusionBackend:
    """Holds at most one loaded diffusers pipeline. All mutations are serialised."""

    def __init__(self) -> None:
        # One lock serialises load / generate / unload — a generate must not run
        # while the pipeline is being swapped out from under it. status() and
        # load_progress() read the single state references without the lock, so
        # polling never blocks a slow load.
        self._lock = threading.Lock()
        self._state: Optional[_LoadState] = None
        self._loading: Optional[_LoadingState] = None

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    def _pick_device_and_dtype(self) -> tuple[str, Any]:
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _resolve_gguf_path(self, repo_id: str, gguf_filename: str, hf_token: Optional[str]) -> str:
        local_root = Path(repo_id).expanduser()
        if local_root.exists():
            return str(resolve_local_gguf_child(local_root, gguf_filename))
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id, gguf_filename, token = hf_token)

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
    ) -> dict[str, Any]:
        """Validate, then run the (slow) load on a daemon thread. Returns at once."""
        if not gguf_filename:
            raise ValueError("gguf_filename is required: this backend loads single-file GGUF checkpoints only.")
        fam = detect_family(repo_id, family_override)
        if fam is None:
            raise ValueError(
                f"Could not infer a diffusion family for '{repo_id}'. Pass family_override (z-image)."
            )
        base = resolve_base_repo(fam, base_repo)

        with self._lock:
            # Allow starting over a previously-failed load, but not over a live one.
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A diffusion load is already in progress.")
            self._loading = _LoadingState(repo_id = repo_id, base_repo = base)

        threading.Thread(
            target = self._run_load,
            kwargs = dict(
                repo_id = repo_id,
                gguf_filename = gguf_filename,
                base_repo = base,
                family_override = family_override,
                hf_token = hf_token,
                cpu_offload = cpu_offload,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(self, **kwargs: Any) -> None:
        try:
            # Estimate sizes on this thread (a network call) so begin_load returns
            # instantly; the bar shows raw bytes until the total lands. This is the
            # only writer of _loading's fields, so no lock is needed here.
            loading = self._loading
            if loading is not None:
                loading.expected_bytes = self._estimate_download_bytes(
                    kwargs["repo_id"],
                    kwargs.get("gguf_filename"),
                    kwargs["base_repo"],
                    kwargs.get("hf_token"),
                )
            self.load_pipeline(**kwargs)
            with self._lock:
                self._loading = None
        except Exception as exc:  # noqa: BLE001 — surfaced to the client via load_progress
            logger.error("diffusion.load_failed: %s", exc)
            with self._lock:
                if self._loading is not None:
                    self._loading.error = str(exc)

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
    ) -> int:
        from huggingface_hub import HfApi

        api = HfApi()
        total = 0
        try:
            if gguf_filename:
                info = api.model_info(repo_id, files_metadata = True, token = hf_token)
                total += sum(s.size or 0 for s in info.siblings if s.rfilename == gguf_filename)
            base_info = api.model_info(base_repo, files_metadata = True, token = hf_token)
            total += sum(
                s.size or 0 for s in base_info.siblings if _base_file_downloaded(s.rfilename)
            )
        except Exception as exc:  # noqa: BLE001 — estimate is best-effort
            logger.warning("diffusion.size_estimate_failed: %s", exc)
        return total

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
    ) -> dict[str, Any]:
        import diffusers

        if not gguf_filename:
            raise ValueError("gguf_filename is required: this backend loads single-file GGUF checkpoints only.")
        fam = detect_family(repo_id, family_override)
        if fam is None:
            raise ValueError(
                f"Could not infer a diffusion family for '{repo_id}'. Pass family_override (z-image)."
            )
        base = resolve_base_repo(fam, base_repo)
        device, dtype = self._pick_device_and_dtype()

        with self._lock:
            # Free the old pipeline before allocating the new one so two
            # checkpoints never sit in VRAM at once.
            self._unload_locked()

            # Dequantise the GGUF transformer on-device; the VAE / text-encoder /
            # scheduler come from the base diffusers repo (the GGUF is transformer-only).
            gguf_path = self._resolve_gguf_path(repo_id, gguf_filename, hf_token)
            transformer_cls = getattr(diffusers, fam.transformer_class)
            transformer = transformer_cls.from_single_file(
                gguf_path,
                quantization_config = diffusers.GGUFQuantizationConfig(compute_dtype = dtype),
                torch_dtype = dtype,
                config = base,
                subfolder = "transformer",
            )

            pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype, "transformer": transformer}
            if hf_token:
                pipe_kwargs["token"] = hf_token
            pipeline_cls = getattr(diffusers, fam.pipeline_class)
            pipe = pipeline_cls.from_pretrained(base, **pipe_kwargs)

            use_offload = bool(cpu_offload and device == "cuda")
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
        steps: int = 24,
        guidance: float = 3.5,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        import torch

        with self._lock:
            state = self._state
            if state is None:
                raise RuntimeError("No diffusion model is loaded.")

            generator = torch.Generator(device = state.device)
            if seed is None:
                # Generator.seed() seeds the generator with a fresh random value
                # AND returns it, so the reported seed reproduces the image.
                seed = generator.seed()
            else:
                seed = int(seed)
                generator.manual_seed(seed)

            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator,
            }
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            image = state.pipe(**kwargs).images[0]
            return {
                "image_b64": encode_png_base64(image),
                "mime": "image/png",
                "seed": int(seed),
            }

    def unload(self) -> dict[str, Any]:
        with self._lock:
            self._unload_locked()
            # Drop a finished/failed load marker so the next load starts clean.
            if self._loading is not None and self._loading.error is not None:
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

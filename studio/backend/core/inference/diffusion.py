# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion image generation backend.

Loads Hugging Face diffusion checkpoints in either the standard
``diffusers`` layout or the single-file GGUF layout published under
``unsloth/*-GGUF`` (Flux 2, Flux 2 Klein, Qwen-Image, SD3, SDXL, ...).
GGUF files are dynamically dequantised on-device via
``diffusers.GGUFQuantizationConfig``, then the rest of the pipeline
(VAE, text encoders, scheduler) is pulled from the matching ``diffusers``
repo so end users only ever need one local file plus the metadata repo.

The module is intentionally torch-only: it never spawns a subprocess and
shares the active CUDA / MPS device with the rest of Studio. The cost of
not having a separate process is that loading a diffusion model and a
GGUF chat model at the same time can OOM on consumer GPUs; the routes
layer must therefore swap between the two as needed (the orchestrator
unloads llama-server before any diffusion load on hosts with < 24 GB).

The class deliberately exposes a small, llama-cpp-style surface:

    load_model(repo_id, ...)
    generate_image(prompt, ...) -> PIL.Image
    unload_model()
    status() -> dict

so the route layer at ``studio/backend/routes/inference.py`` can mirror
the existing llama-server lifecycle (probe + load + generate + unload)
without learning a second API.
"""

from __future__ import annotations

import asyncio
import gc
import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)


# ─── Pipeline registry ────────────────────────────────────────────────
#
# Keep this list narrow on purpose: only ship the small text-to-image
# families with first-class GGUF coverage on the Hub. Anything else is
# either video (LTX*, Wan) or research-grade (Sana, SD3.5) and can be
# added once it has a working GGUF release plus a smoke test.
#
# Each entry maps a substring of the loaded repo id (case-insensitive)
# to the (pipeline_class_name, transformer_class_name, default base
# repo for missing pieces). ``base_repo`` is what we pass to
# ``Pipeline.from_pretrained`` to pick up the VAE + text encoders when
# the user gave us a GGUF-only repo. The base_repo is documented to the
# user via ``status()`` so they understand why a second download fires.


@dataclass(frozen = True)
class DiffusionFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    # Optional: list of HF "trigger" substrings besides ``name`` that map
    # to this family (e.g. "flux1-dev" plus "flux.1-dev"). Lowercased.
    aliases: tuple[str, ...] = field(default_factory = tuple)


_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-klein",
        aliases = ("flux2-klein", "flux-2-klein", "flux.2.klein"),
    ),
    DiffusionFamily(
        name = "flux.2",
        pipeline_class = "Flux2Pipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-dev",
        aliases = ("flux2-dev", "flux-2-dev", "flux.2.dev"),
    ),
    DiffusionFamily(
        name = "flux.1",
        pipeline_class = "FluxPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-dev",
        aliases = ("flux1-dev", "flux-1-dev", "flux.1.dev", "flux-dev"),
    ),
    DiffusionFamily(
        name = "qwen-image",
        pipeline_class = "QwenImagePipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image",
        aliases = ("qwenimage", "qwen_image"),
    ),
    DiffusionFamily(
        name = "stable-diffusion-3",
        pipeline_class = "StableDiffusion3Pipeline",
        transformer_class = "SD3Transformer2DModel",
        base_repo = "stabilityai/stable-diffusion-3-medium-diffusers",
        aliases = ("sd3-medium", "stable-diffusion-3-medium", "sd3.5"),
    ),
    DiffusionFamily(
        name = "stable-diffusion-xl",
        pipeline_class = "StableDiffusionXLPipeline",
        transformer_class = "",  # SDXL uses a UNet, not a transformer
        base_repo = "stabilityai/stable-diffusion-xl-base-1.0",
        aliases = ("sdxl",),
    ),
)


def detect_family(
    repo_id: str, *, override_family: Optional[str] = None
) -> Optional[DiffusionFamily]:
    """Return the diffusion family matching ``repo_id``.

    Matching is substring-based and case-insensitive. ``override_family``
    bypasses substring matching and looks up by ``DiffusionFamily.name``.
    Returns ``None`` when no family applies so callers can surface a clear
    "unsupported model" error rather than guessing wrong.
    """
    if override_family:
        wanted = override_family.strip().lower()
        for fam in _FAMILIES:
            if fam.name == wanted:
                return fam
        return None
    needle = (repo_id or "").lower()
    if not needle:
        return None
    for fam in _FAMILIES:
        if fam.name in needle:
            return fam
        for alias in fam.aliases:
            if alias and alias in needle:
                return fam
    return None


def supported_families() -> list[dict[str, str]]:
    """Public-facing list of families for ``/api/inference/images/status``."""
    return [
        {
            "name": fam.name,
            "pipeline_class": fam.pipeline_class,
            "base_repo": fam.base_repo,
        }
        for fam in _FAMILIES
    ]


# ─── Backend ──────────────────────────────────────────────────────────


class DiffusionBackend:
    """Singleton-style diffusion backend.

    One pipeline at a time; ``load_model`` swaps the previous one out.
    Generation is mutex'd so concurrent requests serialise rather than
    racing GPU memory.
    """

    def __init__(self) -> None:
        self._pipe: Any = None
        self._lock = threading.Lock()
        self._family: Optional[DiffusionFamily] = None
        self._repo_id: Optional[str] = None
        self._gguf_path: Optional[str] = None
        self._base_repo: Optional[str] = None
        self._device: Optional[str] = None
        self._dtype: Optional[str] = None
        self._loaded_at: Optional[float] = None
        self._loading: bool = False
        self._last_error: Optional[str] = None

    # ── lifecycle ─────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def repo_id(self) -> Optional[str]:
        return self._repo_id

    def status(self) -> dict[str, Any]:
        return {
            "is_loaded": self.is_loaded,
            "is_loading": self._loading,
            "repo_id": self._repo_id,
            "family": self._family.name if self._family else None,
            "pipeline_class": self._family.pipeline_class if self._family else None,
            "base_repo": self._base_repo,
            "gguf_path": self._gguf_path,
            "device": self._device,
            "dtype": self._dtype,
            "loaded_at": self._loaded_at,
            "last_error": self._last_error,
            "supported_families": supported_families(),
        }

    def _pick_device_and_dtype(self) -> tuple[str, "Any"]:
        """Pick (device, dtype) for the current host.

        CUDA-first because that is the only path our diffusion GGUFs are
        validated on. On macOS we use MPS in float16 to keep the pipeline
        on the Metal GPU. CPU is allowed only as a last resort because
        running FLUX on CPU is unusably slow (> 10 minutes per image).
        """
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if (
            hasattr(torch, "backends")
            and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return "mps", torch.float16
        return "cpu", torch.float32

    def load_model(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
        family_override: Optional[str] = None,
        enable_model_cpu_offload: bool = True,
    ) -> dict[str, Any]:
        """Load a diffusion model.

        ``repo_id`` is the Hugging Face repo id of either a GGUF-only
        repo (e.g. ``unsloth/FLUX.2-klein-4B-GGUF``) or a full diffusers
        repo (e.g. ``black-forest-labs/FLUX.2-klein``). When the repo
        contains a GGUF, ``gguf_filename`` picks which quant to load;
        otherwise diffusers' standard config-driven load runs.

        ``base_repo`` overrides the auto-detected diffusers base used
        for VAE / text encoders. ``family_override`` short-circuits the
        substring matcher when an exotic repo name confuses it.

        Raises ``RuntimeError`` on failure with a user-facing message;
        the previous pipeline (if any) stays loaded so a failed swap
        does not leave Studio in an unusable state.
        """
        from huggingface_hub import hf_hub_download
        import diffusers
        import torch

        fam = detect_family(repo_id, override_family = family_override)
        if fam is None:
            raise RuntimeError(
                f"Could not infer a diffusion family for '{repo_id}'. "
                "Pass family_override = 'flux.2-klein' / 'flux.2' / "
                "'flux.1' / 'qwen-image' / 'stable-diffusion-3' / "
                "'stable-diffusion-xl' to disambiguate."
            )

        device, dtype = self._pick_device_and_dtype()

        with self._lock:
            self._loading = True
            self._last_error = None
        try:
            pipeline_cls = getattr(diffusers, fam.pipeline_class, None)
            if pipeline_cls is None:
                raise RuntimeError(
                    f"diffusers {diffusers.__version__} has no "
                    f"{fam.pipeline_class}; upgrade diffusers and retry."
                )
            transformer_cls = (
                getattr(diffusers, fam.transformer_class, None)
                if fam.transformer_class
                else None
            )

            effective_base = base_repo or fam.base_repo
            logger.info(
                "Loading diffusion model %s (family=%s, device=%s, dtype=%s, base=%s)",
                repo_id,
                fam.name,
                device,
                dtype,
                effective_base,
            )

            transformer = None
            local_gguf_path: Optional[str] = None
            if gguf_filename:
                if transformer_cls is None:
                    raise RuntimeError(
                        f"Family {fam.name} does not have a GGUF transformer "
                        "path; load the full repo instead."
                    )
                local_gguf_path = hf_hub_download(
                    repo_id = repo_id,
                    filename = gguf_filename,
                    token = hf_token,
                )
                quant_config = diffusers.GGUFQuantizationConfig(compute_dtype = dtype)
                transformer = transformer_cls.from_single_file(
                    local_gguf_path,
                    quantization_config = quant_config,
                    torch_dtype = dtype,
                )

            pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
            if transformer is not None:
                pipe_kwargs["transformer"] = transformer
            if hf_token:
                pipe_kwargs["token"] = hf_token

            pipe = pipeline_cls.from_pretrained(effective_base, **pipe_kwargs)
            if enable_model_cpu_offload and device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)

            # Drop the old pipeline only after the new one is in place.
            old = self._pipe
            with self._lock:
                self._pipe = pipe
                self._family = fam
                self._repo_id = repo_id
                self._gguf_path = local_gguf_path
                self._base_repo = effective_base
                self._device = device
                self._dtype = str(dtype).replace("torch.", "")
                self._loaded_at = time.time()
            _release(old)

            return self.status()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.exception("Diffusion load failed for %s", repo_id)
            raise RuntimeError(f"Failed to load diffusion model: {exc}") from exc
        finally:
            with self._lock:
                self._loading = False

    def unload_model(self) -> dict[str, Any]:
        with self._lock:
            old = self._pipe
            self._pipe = None
            self._family = None
            self._repo_id = None
            self._gguf_path = None
            self._base_repo = None
            self._device = None
            self._dtype = None
            self._loaded_at = None
        _release(old)
        return {"is_loaded": False}

    # ── generation ────────────────────────────────────────────────

    def generate_image(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 24,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> "Any":
        """Generate a single PIL image and return it.

        The mutex is held for the entire call: diffusion pipelines are
        not thread-safe, and overlapping ``__call__``s on a shared
        pipeline frequently corrupt their internal scheduler state.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")
        if num_inference_steps < 1 or num_inference_steps > 200:
            raise ValueError("num_inference_steps must be in [1, 200]")
        if width <= 0 or height <= 0 or width > 2048 or height > 2048:
            raise ValueError("width and height must be in (0, 2048]")
        # Snap to a multiple of 8: Flux / SD pipelines require it and a
        # silent crash deep in the VAE is much worse than a clear error
        # message up front.
        if width % 8 or height % 8:
            raise ValueError("width and height must be multiples of 8")

        import torch

        with self._lock:
            if self._pipe is None:
                raise RuntimeError("No diffusion model is loaded.")
            pipe = self._pipe
            device = self._device or "cpu"

            generator = None
            if seed is not None:
                # Match the device of the pipeline so determinism holds
                # across reload cycles. For CPU offload, the noise still
                # has to live on the device the diffusion forward runs on.
                gen_device = (
                    "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
                )
                generator = torch.Generator(device = gen_device).manual_seed(int(seed))

            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "width": int(width),
                "height": int(height),
            }
            if negative_prompt is not None and negative_prompt.strip():
                call_kwargs["negative_prompt"] = negative_prompt
            if generator is not None:
                call_kwargs["generator"] = generator

            out = pipe(**call_kwargs)
            images = getattr(out, "images", None) or []
            if not images:
                raise RuntimeError("Diffusion pipeline returned no images.")
            return images[0]


def encode_png_base64(pil_image: "Any") -> str:
    """Encode a PIL image to base64-encoded PNG."""
    import base64

    buf = io.BytesIO()
    pil_image.save(buf, format = "PNG", optimize = True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─── Helpers ──────────────────────────────────────────────────────────


def _release(obj: Any) -> None:
    """Best-effort GPU-memory release for a pipeline being swapped out."""
    if obj is None:
        return
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─── Module-level singleton ───────────────────────────────────────────


_singleton: Optional[DiffusionBackend] = None
_singleton_lock = threading.Lock()


def get_diffusion_backend() -> DiffusionBackend:
    """Return the process-wide diffusion backend (lazy-instantiated)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = DiffusionBackend()
    return _singleton


async def async_generate(
    backend: DiffusionBackend,
    **kwargs: Any,
) -> "Any":
    """Run ``generate_image`` in the default executor so route handlers
    do not block the event loop for the 5-30 s a diffusion step takes."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: backend.generate_image(**kwargs))

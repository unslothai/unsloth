# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2 VAE decode in torch for sd.cpp renders on a CPU-only host.

On the CPU, sd.cpp's VAE decode runs its convolutions through ggml's dense F16 GEMM, which is 2.2-3.2x
slower than oneDNN on the same cores: the FLUX.2 VAE takes 80-86 s at 1024 px against 35-39 s in torch fp32
from the same weights (Zen 2, 8 threads). When the server can return the final latent
(``features_by_mode.img_gen.output_latent``), the render asks for it and decodes here, exactly as diffusers'
FLUX.2 pipelines do after their denoise loop: batch-norm denormalize, unpatchify, ``vae.decode``.

Only for FLUX.2 VAE families on a CPU server on x86-64 (the measured case); anything else, or a decode failure,
keeps sd.cpp's own decode. ``UNSLOTH_SD_CPP_TORCH_VAE=0`` turns it off, ``=1`` also allows other CPU
architectures. ``STATS`` counts decoded images and fallbacks.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import platform
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

ENV = "UNSLOTH_SD_CPP_TORCH_VAE"
_DISABLE_TOKENS = frozenset({"0", "off", "false", "no"})
_FORCE_TOKENS = frozenset({"1", "on", "true", "yes"})
# Measured on x86-64 (oneDNN against ggml's F16 GEMM); aarch64 is unmeasured, so it needs the explicit switch.
_DEFAULT_MACHINES = frozenset({"x86_64", "amd64"})
# Families whose sd.cpp VAE is the FLUX.2 AutoencoderKLFlux2 file and whose final latent is the
# 128-channel patchified one diffusers' Flux2 pipelines decode.
FAMILIES = frozenset({"flux.2-klein", "flux.2-dev"})
SERVER_FEATURE = "output_latent"

STATS = {"decoded": 0, "fallback": 0}
_LOCK = threading.Lock()
_CACHE: dict[tuple, Any] = {}


def enabled() -> bool:
    value = (os.environ.get(ENV) or "").strip().lower()
    if value in _DISABLE_TOKENS:
        return False
    return value in _FORCE_TOKENS or platform.machine().lower() in _DEFAULT_MACHINES


def applies(state: Any, *, layers: Optional[int] = None) -> bool:
    """True when this render should take the latent and decode it here: a CPU server, a FLUX.2 VAE family with
    its VAE file on disk, a single image per latent, and torch + diffusers present. Import-light: torch is not
    imported here."""
    if not enabled() or layers:
        return False
    if getattr(state, "device", None) != "cpu" or getattr(state, "server", None) is None:
        return False
    if getattr(getattr(state, "family", None), "name", None) not in FAMILIES:
        return False
    vae = getattr(getattr(state, "files", None), "vae", None)
    if not vae or not os.path.isfile(vae):
        return False
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers"))


def _vae_config(vae_path: str) -> Optional[dict]:
    """The diffusers config next to the weights when there is one; None keeps AutoencoderKLFlux2's defaults,
    which are the FLUX.2 VAE's."""
    config = Path(vae_path).parent / "config.json"
    if not config.is_file():
        return None
    try:
        data = json.loads(config.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("_class_name") != "AutoencoderKLFlux2":
        return None
    return data


def _load_decoder(vae_path: str) -> Any:
    import torch
    from diffusers import AutoencoderKLFlux2
    from safetensors.torch import load_file

    st = os.stat(vae_path)
    key = (os.path.realpath(vae_path), st.st_size, st.st_mtime_ns)
    with _LOCK:
        vae = _CACHE.get(key)
        if vae is not None:
            return vae
        config = _vae_config(vae_path) or {}
        accepted = inspect.signature(AutoencoderKLFlux2.__init__).parameters
        vae = AutoencoderKLFlux2(**{k: v for k, v in config.items() if k in accepted})
        # Built in fp32; the copy casts a bf16 / fp16 file up.
        vae.load_state_dict(load_file(vae_path), strict = True)
        vae = vae.eval().requires_grad_(False)
        # Only the decoder runs here; the encoder would be a second copy of its weights in host RAM.
        vae.encoder = None
        vae.quant_conv = None
        vae = vae.to(memory_format = torch.channels_last)
        _CACHE.clear()
        _CACHE[key] = vae
        return vae


def prepare(vae_path: str) -> None:
    """Load the decoder for ``vae_path`` now (cached); raises when it cannot be built."""
    _load_decoder(vae_path)


def _latent_tensor(shape: Sequence[int], data: bytes) -> Any:
    import torch

    shape = tuple(int(n) for n in shape)
    if len(shape) != 4 or shape[1] % 4:
        raise ValueError(f"unexpected FLUX.2 latent shape {shape}")
    count = 1
    for n in shape:
        count *= n
    if len(data) != count * 4:
        raise ValueError(f"latent holds {len(data)} bytes, shape {shape} needs {count * 4}")
    return torch.frombuffer(bytearray(data), dtype = torch.float32).reshape(shape)


def decode(
    latents: Sequence[tuple[Sequence[int], bytes]],
    vae_path: str,
    *,
    threads: Optional[int] = None,
) -> list:
    """PIL images for sd.cpp's final FLUX.2 latents ``(shape [N, 128, H/16, W/16], float32 bytes)``."""
    import torch
    from PIL import Image

    vae = _load_decoder(vae_path)
    previous = torch.get_num_threads()
    if threads and threads > 0:
        torch.set_num_threads(int(threads))
    images = []
    try:
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        with torch.inference_mode():
            for shape, data in latents:
                z = _latent_tensor(shape, data) * std + mean
                n, c, h, w = z.shape
                # diffusers' Flux2 _unpatchify_latents: (N, C*4, h, w) -> (N, C, 2h, 2w).
                z = (
                    z.reshape(n, c // 4, 2, 2, h, w)
                    .permute(0, 1, 4, 2, 5, 3)
                    .reshape(n, c // 4, h * 2, w * 2)
                )
                out = vae.decode(
                    z.contiguous(memory_format = torch.channels_last), return_dict = False
                )[0]
                # VaeImageProcessor.postprocess(output_type="pil"): [-1, 1] -> [0, 1] -> round to uint8.
                pixels = ((out.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
                for frame in pixels.permute(0, 2, 3, 1).contiguous().numpy():
                    images.append(Image.fromarray(frame, mode = "RGB"))
    finally:
        if threads and threads > 0:
            torch.set_num_threads(previous)
    STATS["decoded"] += len(images)
    return images


def note_fallback(reason: str) -> None:
    STATS["fallback"] += 1
    logger.warning("sd_cpp.torch_vae: falling back to sd.cpp's VAE decode: %s", reason)

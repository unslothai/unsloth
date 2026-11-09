# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 reference-video VAE condition normalised per 17-frame chunk, not as one float32 clip; bit-identical
(elementwise), only over the mirrored diffusers 0.42.1 sources. ``UNSLOTH_H3_CHUNKED_REF_ENCODE=0`` disables."""

from __future__ import annotations

import hashlib
import inspect
import os
import threading
from typing import Any, Optional

H3_CHUNKED_REF_ENCODE_ENV = "UNSLOTH_H3_CHUNKED_REF_ENCODE"
_ENCODERS_MODULE = "diffusers.modular_pipelines.minimax_h3.encoders"
_FUNC = "encode_vae_condition"

# sha256[:16] of inspect.getsource of the mirrored functions (diffusers 0.42.1).
_KNOWN_SOURCES = {
    "encode_vae_condition": frozenset({"8dab1fbe60ffab90"}),
    "encode": frozenset({"7bd0682f1598a640"}),
    "_encode": frozenset({"591b051aaa5744c4"}),
}

_LOCK = threading.Lock()
_VERIFIED: dict = {}


def enabled() -> bool:
    raw = (os.environ.get(H3_CHUNKED_REF_ENCODE_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _sha(fn: Any) -> Optional[str]:
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def _vae_known(vae: Any) -> bool:
    cls = type(vae)
    known = _VERIFIED.get(cls)
    if known is None:
        known = cls.__name__ == "AutoencoderKLMiniMaxH3" and all(
            _sha(getattr(cls, name, None)) in _KNOWN_SOURCES[name] for name in ("encode", "_encode")
        )
        _VERIFIED[cls] = known
    return known and not ({"encode", "_encode"} & set(getattr(vae, "__dict__", {})))


def _eligible(vae: Any, pixels: Any) -> bool:
    import torch

    if not enabled() or not _vae_known(vae):
        return False
    config = getattr(vae, "config", None)
    clip = getattr(config, "clip_length", None)
    if not isinstance(clip, int) or clip <= 0 or getattr(vae, "encoder", None) is None:
        return False
    return (
        isinstance(pixels, torch.Tensor)
        and pixels.dtype == torch.uint8
        and pixels.ndim == 5
        and pixels.shape[0] == 1
        and pixels.shape[2] > clip
    )


def _chunked_moments(vae: Any, pixels: Any, pixel_mean: Any, pixel_std: Any) -> Any:
    """Moments of ``vae.encode`` on the normalised clip, one chunk of float32 at a time."""
    import torch
    from diffusers.models.modeling_utils import get_parameter_dtype

    hook = getattr(vae, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(vae)  # as apply_forward_hook on encode
    clip = vae.config.clip_length
    dtype = get_parameter_dtype(vae.encoder)
    num_frames = pixels.shape[2]
    moments = []
    for start in range(0, num_frames, clip):
        part = pixels[:, :, start : start + clip]
        if part.shape[2] < clip:
            # stock pads with the last normalised frame: same values
            part = torch.cat(
                [part, pixels[:, :, -1:].repeat(1, 1, clip - part.shape[2], 1, 1)], dim = 2
            )
        x = ((part.to(torch.float32).div(255.0) - pixel_mean) / pixel_std).to(dtype)
        del part
        moments.append(vae._encode_clip(x))
        del x
    out = torch.cat(moments, dim = 2)
    del moments
    if vae.config.token_drop > 0:
        out = out[:, :, : -vae.config.token_drop]
    return out


def _make_shim(original: Any) -> Any:
    def encode_vae_condition(
        vae: Any,
        pixels: Any,
        pixel_mean: tuple,
        pixel_std: tuple,
        encode_seed: int = 42,
    ) -> Any:
        if not _eligible(vae, pixels):
            return original(vae, pixels, pixel_mean, pixel_std, encode_seed)
        import torch
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(pixel_mean, device = pixels.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(pixel_std, device = pixels.device).view(1, -1, 1, 1, 1)

        posterior = DiagonalGaussianDistribution(
            _chunked_moments(vae, pixels, pixel_mean, pixel_std)
        )
        latents = posterior.sample(generator = torch.Generator().manual_seed(encode_seed))
        latents = latents.to(torch.float16).float().cpu()
        return (latents - latents_mean) / latents_std

    encode_vae_condition._unsloth_chunked_ref_encode = True
    encode_vae_condition.__wrapped__ = original
    try:
        encode_vae_condition.__signature__ = inspect.signature(original)
    except (TypeError, ValueError):
        pass
    return encode_vae_condition


def install_h3_chunked_reference_encode(logger: Any = None) -> bool:
    """Shims the module global the modular step calls; idempotent, never raises."""
    if not enabled():
        return False
    try:
        from diffusers.modular_pipelines.minimax_h3 import encoders as module
    except Exception:  # noqa: BLE001 - not an H3-capable diffusers
        return False
    with _LOCK:
        current = getattr(module, _FUNC, None)
        if not callable(current):
            return False
        if getattr(current, "_unsloth_chunked_ref_encode", False):
            return True
        if _sha(current) not in _KNOWN_SOURCES["encode_vae_condition"]:
            if logger is not None:
                logger.info(
                    "video.h3_ref_encode: unknown encode_vae_condition source, keeping stock"
                )
            return False
        setattr(module, _FUNC, _make_shim(current))
    if logger is not None:
        logger.info("video.h3_ref_encode: reference videos normalise one VAE chunk at a time")
    return True

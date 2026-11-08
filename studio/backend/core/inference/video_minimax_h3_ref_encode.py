# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 reference-video VAE condition, normalised one temporal chunk at a time.

Diffusers' ``encode_vae_condition`` turns the whole ``uint8`` reference clip into float32 on the device, and the VAE's
``_encode`` then pads that to a whole number of 17-frame chunks with a second full copy, before encoding the chunks
one by one: ~1.5 GiB of float32 live for a 124-frame 960x544 reference. Normalisation and the dtype cast are
elementwise and the padding repeats the last frame, so normalising each chunk on its own hands every chunk the same
values; the posterior, its seeded sample, the float16 rounding and the latent normalisation are unchanged.
Only for the known diffusers sources (else stock); ``UNSLOTH_H3_CHUNKED_REF_ENCODE=0`` disables."""

from __future__ import annotations

import hashlib
import inspect
import os
import threading
from typing import Any, Optional

H3_CHUNKED_REF_ENCODE_ENV = "UNSLOTH_H3_CHUNKED_REF_ENCODE"
_ENCODERS_MODULE = "diffusers.modular_pipelines.minimax_h3.encoders"
_FUNC = "encode_vae_condition"

# sha256[:16] of inspect.getsource for the functions this mirrors (diffusers 0.42.1).
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
    # an instance-level encode / _encode (another wrapper) is not the code mirrored here
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
    """``vae.encode((pixels / 255 - mean) / std)``'s moments for one clip, building one chunk of float32 at a time."""
    import torch
    from diffusers.models.modeling_utils import get_parameter_dtype

    hook = getattr(vae, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(vae)  # what ``apply_forward_hook`` on ``encode`` does
    clip = vae.config.clip_length
    dtype = get_parameter_dtype(vae.encoder)
    num_frames = pixels.shape[2]
    moments = []
    for start in range(0, num_frames, clip):
        part = pixels[:, :, start : start + clip]
        if part.shape[2] < clip:
            # stock pads the normalised clip with its last frame; padding the pixels first gives the same values
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

        # op for op the stock body, but for the chunked moments
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
    """Process-wide, idempotent: the modular reference step calls this module global. Never raises."""
    if not enabled():
        return False
    try:
        import importlib
        module = importlib.import_module(_ENCODERS_MODULE)
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

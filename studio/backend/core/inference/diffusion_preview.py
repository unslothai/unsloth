# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cheap latent-to-RGB thumbnails for an in-flight denoise.

The projection is least-squares fitted once per VAE against a few decoded random latents,
so no per-family colour constants are needed. Any failure disables previews for that VAE.
"""

from __future__ import annotations

import base64
import io
import math
import os
import weakref
from typing import Any, Optional

MIN_INTERVAL_S = 0.4
MAX_EDGE_PX = 256

_FIT_SAMPLES = 4
_FIT_GRID = 16

# Keyed by the VAE module itself: an id() key would collide after the module is freed.
_projections: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()

PREVIEWS_ENV = "UNSLOTH_DIFFUSION_PREVIEWS"
_DISABLED = ("0", "false", "off", "no")


def previews_enabled() -> bool:
    return os.environ.get(PREVIEWS_ENV, "1").strip().lower() not in _DISABLED


def reset() -> None:
    _projections.clear()


def _decoded(vae: Any, latents: Any) -> Any:
    out = vae.decode(latents)
    sample = getattr(out, "sample", None)
    if sample is not None:
        return sample
    return out[0] if isinstance(out, (tuple, list)) else out


def _latent_channels(vae: Any) -> int:
    config = getattr(vae, "config", None)
    # Qwen-Image and Krea-2 carry no latent_channels at all; theirs is z_dim. Reading only
    # the first name left channels at 0, so projection() cached None and those families
    # never previewed -- the per-channel denormalization below was never even reached.
    for name in ("latent_channels", "z_dim"):
        try:
            return int(getattr(config, name, None))
        except (TypeError, ValueError):
            continue
    return 0


def _channel_stats(vae: Any, channels: int, torch: Any) -> Optional[tuple[Any, Any]]:
    config = getattr(vae, "config", None)
    mean = getattr(config, "latents_mean", None)
    std = getattr(config, "latents_std", None)
    if mean is None or std is None:
        return None
    mean_t = torch.tensor(mean, dtype = torch.float32).flatten()
    std_t = torch.tensor(std, dtype = torch.float32).flatten()
    if mean_t.numel() != channels or std_t.numel() != channels:
        return None
    return std_t, mean_t


def _to_vae_space(grid: Any, vae: Any, channels: int, torch: Any) -> Optional[Any]:
    """Undo the denoiser-side normalization, so `grid` is what the decoder was fitted on.

    Three schemes ship in the families Studio loads. Most use the scalar pair; Qwen-Image
    and Krea-2 normalize per channel and carry no scaling_factor at all, so reading only
    the scalars left their latents untouched and their colours wrong. FLUX.2 normalizes
    with its VAE's BatchNorm statistics over patchified latents, whose channel count no
    longer matches the config, so it gets no preview rather than a miscoloured one."""
    if getattr(vae, "bn", None) is not None:
        return None
    stats = _channel_stats(vae, channels, torch)
    if stats is not None:
        std, mean = stats
        return grid * std + mean
    config = getattr(vae, "config", None)
    scaling = getattr(config, "scaling_factor", None) or 1.0
    shift = getattr(config, "shift_factor", None) or 0.0
    return grid / float(scaling) + float(shift)


def _fit(vae: Any, channels: int, device: Any, dtype: Any, torch: Any) -> Optional[Any]:
    noise = torch.randn(_FIT_SAMPLES, channels, _FIT_GRID, _FIT_GRID, device = device, dtype = dtype)
    with torch.no_grad():
        try:
            decoded = _decoded(vae, noise)
        except Exception:  # noqa: BLE001 -- 5-D decoders want an explicit frame axis
            decoded = _decoded(vae, noise.unsqueeze(2))
    if decoded.ndim == 5:
        decoded = decoded[:, :, 0]
    if decoded.ndim != 4 or decoded.shape[1] != 3:
        return None
    target = torch.nn.functional.adaptive_avg_pool2d(decoded.float(), _FIT_GRID)
    lhs = noise.float().permute(0, 2, 3, 1).reshape(-1, channels)
    lhs = torch.cat([lhs, torch.ones_like(lhs[:, :1])], dim = 1)
    rhs = target.permute(0, 2, 3, 1).reshape(-1, 3)
    # Solved on CPU: MPS ships no linalg_lstsq, and projection() swallows the
    # NotImplementedError, so on a Mac this would cache "no previews" for the whole
    # session after paying for the decode. Both matrices are tiny.
    solution = torch.linalg.lstsq(lhs.cpu(), rhs.cpu()).solution
    if not bool(torch.isfinite(solution).all()):
        return None
    return solution.cpu()


def projection(
    vae: Any,
    torch: Any,
    logger: Any = None,
) -> Optional[Any]:
    cached = _projections.get(vae, False)
    if cached is not False:
        return cached
    fitted = None
    try:
        channels = _latent_channels(vae)
        if channels:
            parameter = next(vae.parameters())
            # A force_upcast VAE is one the pipeline itself decodes in float32; fitting it at
            # the loaded float16 overflows on some cards, and the cached None then disabled
            # previews for the whole load even though the final decode would have worked.
            # Read as a value, not through the Parameter: vae.to() mutates it in place, so
            # restoring from parameter.dtype afterwards would restore float32 onto float32.
            original_dtype = parameter.dtype
            device = parameter.device
            upcast = (
                bool(getattr(getattr(vae, "config", None), "force_upcast", False))
                and original_dtype != torch.float32
            )
            if upcast:
                vae.to(dtype = torch.float32)
            try:
                fitted = _fit(
                    vae, channels, device, torch.float32 if upcast else original_dtype, torch
                )
            finally:
                if upcast:
                    vae.to(dtype = original_dtype)
    except Exception as exc:  # noqa: BLE001 -- a preview must never break a generation
        if logger is not None:
            logger.info("diffusion.preview: projection unavailable (%s)", exc)
    _projections[vae] = fitted
    return fitted


def _packed_dims(sequence: int, width: int, height: int) -> Optional[tuple[int, int]]:
    ratio = max(1, width) / max(1, height)
    rows = max(1, round(math.sqrt(sequence / ratio)))
    for delta in range(0, 64):
        for candidate in (rows - delta, rows + delta):
            if candidate >= 1 and sequence % candidate == 0:
                return candidate, sequence // candidate
    return None


def _grid(latents: Any, channels: int, width: int, height: int) -> Optional[Any]:
    if latents.ndim == 5:
        latents = latents[:, :, 0]
    if latents.ndim == 4:
        return latents[0].permute(1, 2, 0)
    if latents.ndim == 3:
        packed = latents[0]
        depth = int(packed.shape[-1])
        if not channels or depth % channels:
            return None
        dims = _packed_dims(int(packed.shape[0]), width, height)
        if dims is None:
            return None
        rows, columns = dims
        return packed.reshape(rows, columns, channels, depth // channels).mean(-1)
    return None


def _encode(pixels: Any, width: int, height: int) -> str:
    from PIL import Image

    image = Image.fromarray(pixels, mode = "RGB")
    scale = MAX_EDGE_PX / max(1, max(width, height))
    image = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR
    )
    buffer = io.BytesIO()
    image.save(buffer, format = "JPEG", quality = 72)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render(
    latents: Any,
    vae: Any,
    width: int,
    height: int,
    torch: Any,
    logger: Any = None,
) -> Optional[str]:
    """A base64 JPEG data URL for these latents, or None when previews are unavailable."""
    if latents is None:
        return None
    try:
        weights = projection(vae, torch, logger)
        if weights is None:
            return None
        channels = _latent_channels(vae)
        grid = _grid(latents.detach(), channels, width, height)
        if grid is None or int(grid.shape[-1]) != channels:
            return None
        grid = _to_vae_space(grid.float().cpu(), vae, channels, torch)
        if grid is None:
            return None
        rgb = grid @ weights[:channels] + weights[channels]
        pixels = ((rgb + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
        return _encode(pixels, width, height)
    except Exception as exc:  # noqa: BLE001 -- a preview must never break a generation
        if logger is not None:
            logger.info("diffusion.preview: render skipped (%s)", exc)
        return None

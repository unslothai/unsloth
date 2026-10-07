# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-frame fast path for the Qwen-Image causal 3D VAE: one frame with no cache only meets the last
temporal kernel slice, so each conv is exactly ``conv2d(x[:, :, 0], weight[:, :, -1], bias)``, and the
first chunk of the cached walk equals ``decoder(x, feat_cache=None)``. Gated to single-frame untiled
calls (tiled / multi-frame keep 3D convs). PSNR ~60 dB vs 3D (different cuDNN algo), so non-``off``
tiers only. Kill switch, read at install: ``UNSLOTH_DIFFUSION_VAE_SINGLE_FRAME=0``."""

from __future__ import annotations

import os
import types
from typing import Any

SINGLE_FRAME_ENV = "UNSLOTH_DIFFUSION_VAE_SINGLE_FRAME"

_VAE_CLASSES = frozenset({"AutoencoderKLQwenImage"})
_CONV_CLASSES = frozenset({"QwenImageCausalConv3d"})


class _Gate:
    def __init__(self) -> None:
        self.on = False


def single_frame_disabled() -> bool:
    return (os.environ.get(SINGLE_FRAME_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _conv_forward(
    self: Any,
    x: Any,
    cache_x: Any = None,
) -> Any:
    """``QwenImageCausalConv3d.forward`` with the one-frame, no-cache case as a 2D conv."""
    import torch.nn.functional as F

    pad = self._padding
    kt = self.kernel_size[0]
    if (
        self._unsloth_sf_gate.on
        and cache_x is None
        and x.dim() == 5
        and x.shape[2] == 1
        and self.stride[0] == 1
        and self.dilation[0] == 1
        and pad[4] == kt - 1
        and pad[5] == 0
        and self.groups == 1
    ):
        x2 = x[:, :, 0]
        if pad[0] or pad[1] or pad[2] or pad[3]:
            x2 = F.pad(x2, pad[:4])
        out = F.conv2d(
            x2,
            self.weight[:, :, -1],
            self.bias,
            stride = self.stride[1:],
            padding = 0,
            dilation = self.dilation[1:],
        )
        return out.unsqueeze(2)
    return type(self).forward(self, x, cache_x)


def _single_frame(vae: Any, z: Any) -> bool:
    try:
        return bool(z.dim() == 5 and z.shape[2] == 1)
    except Exception:  # noqa: BLE001 - not a tensor: stock path
        return False


def _tiles(vae: Any, z: Any, sample_space: bool) -> bool:
    if not getattr(vae, "use_tiling", False):
        return False
    ratio = 1 if sample_space else getattr(vae, "spatial_compression_ratio", 8)
    h = getattr(vae, "tile_sample_min_height", 1 << 30) // ratio
    w = getattr(vae, "tile_sample_min_width", 1 << 30) // ratio
    return z.shape[-1] > w or z.shape[-2] > h


def _fast_decode(
    self: Any,
    z: Any,
    return_dict: bool = True,
) -> Any:
    stock = self._unsloth_stock_decode
    if not _single_frame(self, z) or _tiles(self, z, False):
        return stock(z, return_dict = return_dict)
    import torch

    gate = self._unsloth_sf_gate
    gate.on = True
    try:
        x = self.post_quant_conv(z)
        out = torch.clamp(self.decoder(x), min = -1.0, max = 1.0)
    finally:
        gate.on = False
    if not return_dict:
        return (out,)
    from diffusers.models.autoencoders.vae import DecoderOutput

    return DecoderOutput(sample = out)


def _fast_encode(self: Any, x: Any) -> Any:
    stock = self._unsloth_stock_encode
    if not _single_frame(self, x) or _tiles(self, x, True):
        return stock(x)
    gate = self._unsloth_sf_gate
    gate.on = True
    try:
        return self.quant_conv(self.encoder(x))
    finally:
        gate.on = False


def install(vae: Any, logger: Any = None) -> bool:
    """Arm the single-frame path on one VAE instance. Idempotent; False when the class is not covered."""
    if vae is None or single_frame_disabled() or type(vae).__name__ not in _VAE_CLASSES:
        return False
    if getattr(vae, "_unsloth_single_frame", False):
        return True
    try:
        convs = [m for m in vae.modules() if type(m).__name__ in _CONV_CLASSES]
        if not convs or not callable(getattr(vae, "_decode", None)):
            return False
        for m in convs:
            _ = (m._padding[5], m.kernel_size[0], m.stride[0], m.dilation[0], m.groups)
        gate = _Gate()
        for m in convs:
            m._unsloth_sf_gate = gate
            m.forward = types.MethodType(_conv_forward, m)
        vae._unsloth_sf_gate = gate
        vae._unsloth_stock_decode = vae._decode
        vae._decode = types.MethodType(_fast_decode, vae)
        if (
            callable(getattr(vae, "_encode", None))
            and hasattr(vae, "encoder")
            and hasattr(vae, "quant_conv")
        ):
            vae._unsloth_stock_encode = vae._encode
            vae._encode = types.MethodType(_fast_encode, vae)
        vae._unsloth_single_frame = True
    except Exception as exc:  # noqa: BLE001 - an optimisation: the stock VAE still decodes
        if logger is not None:
            logger.warning("diffusion.vae: single-frame path not armed: %s", exc)
        return False
    if logger is not None:
        logger.info(
            "diffusion.vae: single-frame 2D path armed on %s (%d causal convs)",
            type(vae).__name__,
            len(convs),
        )
    return True

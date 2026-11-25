# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan 2.2 / Qwen-Image-2.1 VAE residual blocks without the input copy (UNSLOTH_VAE_RESIDUAL_NO_CLONE=0 disables).

``ResidualDownBlock.forward`` (encoder) and ``ResidualUpBlock.forward`` (decoder) start with ``x_copy = x.clone()``
for the shortcut, but no module below writes its input in place, so the copy only holds one more full activation
for the whole block (diffusers #15278). The fused CUDA decoder already drops it in its own up block
(``diffusion_vae_fused._fast_wan_residual_up``) and replaces this patch there; this one covers the encoder on every
device and the decoder wherever the fused path does not run (ROCm, MPS, CPU, fused off). Bitwise the stock output.
Patched per instance and only when the class's ``forward`` source is a known diffusers release."""

from __future__ import annotations

import hashlib
import inspect
import os
import types
from typing import Any, Optional

NO_CLONE_ENV = "UNSLOTH_VAE_RESIDUAL_NO_CLONE"
MARK = "_unsloth_vae_noclone"
_FALSE = ("0", "false", "no", "off")

# sha256[:16] of inspect.getsource(cls.forward): diffusers 0.36.0 / 0.40 (Wan), 0.41.0 / 0.42.1 (Wan, Qwen-Image-2.1).
_KNOWN_DOWN = frozenset(
    {
        "40a0086abd653ac4",  # WanResidualDownBlock 0.36.0 - 0.40
        "181739bf0a12b0e2",  # WanResidualDownBlock 0.41.0 - 0.42.1
        "f5c5a55f18959803",  # QwenImage21ResidualDownBlock 0.41.0 - 0.42.1
    }
)
_KNOWN_UP = frozenset(
    {
        "24256211ae1076cf",  # WanResidualUpBlock 0.36.0 - 0.40
        "3cb1aa9e7628a7ed",  # WanResidualUpBlock 0.41.0 - 0.42.1
        "ee550bf301ece975",  # QwenImage21ResidualUpBlock 0.41.0 - 0.42.1
    }
)
_VERIFIED: dict = {}


def disabled() -> bool:
    return (os.environ.get(NO_CLONE_ENV) or "").strip().lower() in _FALSE


def _sha(fn: Any) -> Optional[str]:
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def _kind(module: Any) -> Optional[str]:
    """ "down" / "up" for a residual block whose class forward is a known stock release, else None."""
    cls = type(module)
    if cls not in _VERIFIED:
        name = cls.__name__
        kind = None
        if name.endswith("ResidualDownBlock"):
            kind = "down" if _sha(cls.forward) in _KNOWN_DOWN else None
        elif name.endswith("ResidualUpBlock"):
            kind = "up" if _sha(cls.forward) in _KNOWN_UP else None
        _VERIFIED[cls] = kind
    return _VERIFIED[cls]


def _feat_idx_default(module: Any) -> Any:
    try:
        default = inspect.signature(type(module).forward).parameters["feat_idx"].default
    except (KeyError, TypeError, ValueError):
        return None
    return None if default is inspect.Parameter.empty else default


def _down_forward(block: Any) -> Any:
    default_idx = _feat_idx_default(block)

    def forward(
        self: Any,
        x: Any,
        feat_cache: Any = None,
        feat_idx: Any = None,
    ) -> Any:
        if feat_idx is None:
            feat_idx = default_idx if default_idx is not None else [0]
        x_in = x  # stock: x.clone(); nothing below writes its input
        for resnet in self.resnets:
            x = resnet(x, feat_cache = feat_cache, feat_idx = feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache = feat_cache, feat_idx = feat_idx)
        return x + self.avg_shortcut(x_in)

    return types.MethodType(forward, block)


def _up_forward(block: Any) -> Any:
    default_idx = _feat_idx_default(block)

    def forward(
        self: Any,
        x: Any,
        feat_cache: Any = None,
        feat_idx: Any = None,
        first_chunk: bool = False,
    ) -> Any:
        if feat_idx is None:
            feat_idx = default_idx if default_idx is not None else [0]
        x_in = x  # stock: x.clone(); nothing below writes its input
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache = feat_cache, feat_idx = feat_idx)
            else:
                x = resnet(x)
        if self.upsampler is not None:
            if feat_cache is not None:
                x = self.upsampler(x, feat_cache = feat_cache, feat_idx = feat_idx)
            else:
                x = self.upsampler(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_in, first_chunk = first_chunk)
        return x

    return types.MethodType(forward, block)


def install(vae: Any, logger: Any = None) -> int:
    """Patch ``vae``'s residual down / up blocks. Returns the number of patched blocks; idempotent. A block another
    speed path already replaced (the fused CUDA up block) is left alone."""
    if vae is None or disabled() or not callable(getattr(vae, "modules", None)):
        return 0
    patched = 0
    for module in vae.modules():
        if getattr(module, MARK, False) or "forward" in module.__dict__:
            continue
        kind = _kind(module)
        if kind is None:
            continue
        module.forward = _down_forward(module) if kind == "down" else _up_forward(module)
        setattr(module, MARK, True)
        patched += 1
    if patched and logger is not None:
        logger.info(
            "diffusion.vae_noclone: %d residual block(s) of %s skip the input copy",
            patched,
            type(vae).__name__,
        )
    return patched


def install_on_pipe(pipe: Any, logger: Any = None) -> int:
    try:
        return install(getattr(pipe, "vae", None), logger)
    except Exception as exc:  # noqa: BLE001 - the stock blocks stay
        if logger is not None:
            logger.warning("diffusion.vae_noclone: not installed: %s", exc)
        return 0

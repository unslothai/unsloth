# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm: run a large video VAE decoder Conv3d as one Conv2d per temporal tap.

On Windows gfx1151 (torch 2.11 + ROCm 7.14) a Conv3d whose input or output tensor exceeds 2 GiB runs about 10x slower
per element than the same conv below that size; measured on the LTX-2.3 decoder across 6 conv shape classes and
5 clip sizes. Splitting such a call into Conv2d calls, one per temporal tap with time folded into the batch, summed in
fp32 with the bias added once, decodes 1920x1088x121 about 1.8x faster and closer to a CPU fp32 decode. Below 2 GiB
the stock Conv3d is faster, and Linux ROCm has no cliff, so only Windows ROCm installs it and only bf16 / fp32 calls
over 2 GiB outside autocast take it. UNSLOTH_VIDEO_VAE_ROCM_PERTAP=0 turns it off (read per call).
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

PERTAP_ENV = "UNSLOTH_VIDEO_VAE_ROCM_PERTAP"
# A call takes the per-tap path when its input or output tensor is larger than this.
ENGAGE_BYTES = 2**31
# Output frames per Conv2d batch: the fp32 accumulator of one chunk stays under this.
CHUNK_BYTES = 256 * 2**20
# Device types the per-tap path runs on (ROCm torch reports "cuda").
ENGAGE_DEVICE_TYPES = ("cuda",)

_VAE_CLASSES = frozenset({"AutoencoderKLLTX2Video"})
_MARK = "_unsloth_rocm_pertap"
_CALLS = "_unsloth_rocm_pertap_calls"


def pertap_disabled() -> bool:
    return (os.environ.get(PERTAP_ENV) or "").strip().lower() in ("0", "false", "no", "off")


def platform_applies(torch: Any) -> bool:
    """Windows + ROCm torch (AMD SDK wheels included: the tag may only sit in ``__version__``)."""
    from .rocm_bf16 import is_rocm_torch
    return sys.platform == "win32" and is_rocm_torch(torch)


def _out_size(size: int, k: int, s: int, p: int, d: int) -> int:
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


def pertap_conv3d(
    x: Any,
    weight: Any,
    bias: Any,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int = 1,
) -> Any:
    """``F.conv3d(x, weight, bias, stride, padding, dilation, groups)`` (zero padding, int tuples) as one Conv2d per
    temporal tap over output-frame chunks, taps summed in fp32, bias added once; returns a contiguous tensor."""
    import torch
    import torch.nn.functional as F

    kt = int(weight.shape[2])
    st, sh, sw = (int(v) for v in stride)
    pt, ph, pw = (int(v) for v in padding)
    dt, dh, dw = (int(v) for v in dilation)
    if pt:
        x = F.pad(x, (0, 0, 0, 0, pt, pt))
    B, C, T, H, W = x.shape
    O = int(weight.shape[0])
    tout = _out_size(T, kt, st, 0, dt)
    ho = _out_size(H, int(weight.shape[3]), sh, ph, dh)
    wo = _out_size(W, int(weight.shape[4]), sw, pw, dw)
    out = torch.empty((B, O, max(tout, 0), ho, wo), device = x.device, dtype = x.dtype)
    if tout <= 0:
        return out
    chunk = max(1, min(tout, CHUNK_BYTES // max(1, B * O * ho * wo * 4)))
    bias32 = None if bias is None else bias.float().view(1, -1, 1, 1)
    for t0 in range(0, tout, chunk):
        n = min(tout, t0 + chunk) - t0
        acc = None
        for k in range(kt):
            s = t0 * st + k * dt
            frames = x[:, :, s : s + st * (n - 1) + 1 : st]  # (B, C, n, H, W)
            frames = frames.transpose(1, 2).reshape(B * n, C, H, W)
            y = F.conv2d(frames, weight[:, :, k], None, (sh, sw), (ph, pw), (dh, dw), groups)
            if acc is None:
                acc = y.float()
            else:
                acc.add_(y)
            del frames, y
        if bias32 is not None:
            acc.add_(bias32)
        out[:, :, t0 : t0 + n] = acc.view(B, n, O, ho, wo).transpose(1, 2)
        del acc
    return out


# Input dtypes the per-tap path takes: fp16 taps round (and can overflow to inf) before the fp32 sum, where stock
# Conv3d accumulates once, so a finite fp16 result could turn NaN.
ENGAGE_DTYPES = ("bfloat16", "float32")


def _autocast_on(torch: Any, device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # torch < 2.4: no device_type argument
        return bool(torch.is_autocast_enabled())


def _pertap_eligible(conv: Any, x: Any) -> bool:
    import torch

    # is_compiling first: a traced env read guards on the env value and recompiles when it flips
    if torch.compiler.is_compiling() or pertap_disabled():
        return False
    if x.dim() != 5 or x.device.type not in ENGAGE_DEVICE_TYPES or isinstance(conv.padding, str):
        return False
    # autocast: stock returns the autocast dtype, the per-tap output keeps x's dtype
    if str(x.dtype).rsplit(".", 1)[-1] not in ENGAGE_DTYPES or _autocast_on(torch, x.device.type):
        return False
    if torch.is_grad_enabled() and (x.requires_grad or conv.weight.requires_grad):
        return False
    itemsize = x.element_size()
    if x.numel() * itemsize > ENGAGE_BYTES:
        return True
    pad = (0, 0, 0) if conv.padding_mode != "zeros" else conv.padding
    B, _, T, H, W = x.shape
    if conv.padding_mode != "zeros":
        # stock pre-pads with padding_mode, then convolves unpadded
        T, H, W = (s + 2 * p for s, p in zip((T, H, W), conv.padding))
    dims = [
        _out_size(s, k, st, p, d)
        for s, k, st, p, d in zip((T, H, W), conv.kernel_size, conv.stride, pad, conv.dilation)
    ]
    return B * conv.out_channels * dims[0] * dims[1] * dims[2] * itemsize > ENGAGE_BYTES


def _forward(self: Any, x: Any) -> Any:
    stock = type(self).forward
    if not _pertap_eligible(self, x):
        return stock(self, x)
    padding = self.padding
    if self.padding_mode != "zeros":
        import torch.nn.functional as F
        x = F.pad(x, self._reversed_padding_repeated_twice, mode = self.padding_mode)
        padding = (0, 0, 0)
    setattr(self, _CALLS, getattr(self, _CALLS, 0) + 1)
    return pertap_conv3d(
        x, self.weight, self.bias, self.stride, padding, self.dilation, self.groups
    )


def _covered(vae: Any) -> bool:
    return type(vae).__name__ in _VAE_CLASSES and getattr(vae, "decoder", None) is not None


def install(vae: Any, logger: Any = None) -> int:
    """Patch ``vae.decoder``'s Conv3d modules on Windows ROCm; returns how many carry the patch (0 = not installed).
    Idempotent; skips convs whose forward something else already replaced."""
    if pertap_disabled() or not _covered(vae):
        return 0
    import torch

    if not platform_applies(torch):
        return 0
    patched = 0
    for module in vae.decoder.modules():
        if not isinstance(module, torch.nn.Conv3d):
            continue
        if getattr(module, _MARK, False):
            patched += 1
            continue
        if "forward" in module.__dict__:
            continue
        module.forward = types.MethodType(_forward, module)
        setattr(module, _MARK, True)
        patched += 1
    if patched and logger is not None:
        logger.info(
            "video.vae_rocm_pertap: %s decoder runs %d Conv3d as per-tap Conv2d when a tensor exceeds %d MiB",
            type(vae).__name__,
            patched,
            ENGAGE_BYTES // 2**20,
        )
    return patched


def install_on_pipe(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> int:
    if getattr(target, "device", None) != "cuda":
        return 0
    return install(getattr(pipe, "vae", None), logger)


def pertap_calls(vae: Any) -> int:
    """How many Conv3d calls in ``vae``'s decoder took the per-tap path since load."""
    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        return 0
    return sum(int(getattr(m, _CALLS, 0)) for m in decoder.modules())

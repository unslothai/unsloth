# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Apple MPS: a video VAE's Conv3d calls run as one Conv2d per temporal tap over (B*T, C, H, W), fp32 summed, bias once.

torch < 2.14 runs MPS Conv3d at 3-5 TFLOPS on an M5 Max while Conv2d reaches 36-46, so the per-tap form is 7.7-14x
faster at equal error (3x3x3, 128-1024 channels, T = 8-31, bf16 and fp16), and a 1.7 GB stock bf16 Conv3d
(128ch, T = 61, 256x416) hung the GPU into a kernel watchdog panic. torch 2.14.1's native Conv3d is within 1.10x or
faster except at wide channels over small frames (1.35x left at 1024ch, T = 8, 32x52), so from 2.14 only that shape
rule engages, plus the size guard: a call whose input or output tensor exceeds ``SIZE_GUARD_BYTES`` always takes the
chunked path, and every op inside it touches at most ``CHUNK_BYTES`` per tensor.

The patch wraps ``Conv3d._conv_forward`` per instance, so each VAE's own causal padding (Wan / Qwen-Image zero frames
and feature cache, LTX-2 first-frame replicate, HunyuanVideo-1.5 replicate F.pad, MiniMax-H3 zero frames) runs
unchanged ahead of it: the conv sees bitwise the stock input. Conv-level temporal zero padding is applied by dropping
the taps that land in it (no 5-D F.pad on MPS).

``UNSLOTH_VIDEO_VAE_MPS_PERTAP``: unset = the rule above, ``0`` = stock Conv3d everywhere, ``1`` = every call on MPS.
"""

from __future__ import annotations

import os
import re
import types
from typing import Any, Optional

PERTAP_ENV = "UNSLOTH_VIDEO_VAE_MPS_PERTAP"
# torch from which MPS Conv3d is native-speed (measured on 2.14.1, M5 Max).
FAST_NATIVE_TORCH = (2, 14)
# torch >= 2.14 shape rule (M5 Max, 2.14.1): at 1024ch per-tap won 1.11-1.74x up to 64x104 frames (2048ch 1.77x);
# at 512ch it was a tie and at 256ch / 128x208 native was 1.63x faster. Larger frames at 1024ch were not measured.
WIDE_CHANNELS = 1024
WIDE_MAX_HW = 8192
# Any call with an input or output tensor over this takes the chunked per-tap path (panic guard, any torch).
SIZE_GUARD_BYTES = 2**30
# Per-op tensor bound inside the per-tap path: frame chunk, Conv2d output and fp32 accumulator.
CHUNK_BYTES = 256 * 2**20
ENGAGE_DEVICE_TYPES = ("mps",)
ENGAGE_DTYPES = ("bfloat16", "float16", "float32")
OPTIM_NAME = "vae_mps_pertap_conv"

_MARK = "_unsloth_mps_pertap"
_CALLS = "_unsloth_mps_pertap_calls"
_FALSE = ("0", "false", "no", "off")
_TRUE = ("1", "true", "yes", "on")


def _mode() -> str:
    value = (os.environ.get(PERTAP_ENV) or "").strip().lower()
    if value in _FALSE:
        return "off"
    if value in _TRUE:
        return "force"
    return "auto"


def pertap_disabled() -> bool:
    return _mode() == "off"


def torch_version_tuple(version: str) -> tuple:
    """``"2.10.0"`` / ``"2.14.0.dev20261101+cpu"`` -> ``(2, 10)``; unparsable -> ``(0, 0)``."""
    match = re.match(r"\s*(\d+)\.(\d+)", str(version or ""))
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


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
    chunk_bytes: int = CHUNK_BYTES,
) -> Any:
    """``F.conv3d`` (zero padding) as one Conv2d per temporal tap, chunked over output frames.

    Taps that read temporal padding are skipped (they contribute zero), so no frame-axis pad is built. Each tap's Conv2d
    output is summed into an fp32 accumulator, bias added once in fp32, then cast to ``x.dtype``."""
    import torch
    import torch.nn.functional as F

    kt = int(weight.shape[2])
    st, sh, sw = (int(v) for v in stride)
    pt, ph, pw = (int(v) for v in padding)
    dt, dh, dw = (int(v) for v in dilation)
    B, C, T, H, W = x.shape
    O = int(weight.shape[0])
    tout = _out_size(T, kt, st, pt, dt)
    ho = _out_size(H, int(weight.shape[3]), sh, ph, dh)
    wo = _out_size(W, int(weight.shape[4]), sw, pw, dw)
    out = torch.empty((B, O, max(tout, 0), max(ho, 0), max(wo, 0)), device = x.device, dtype = x.dtype)
    if tout <= 0 or ho <= 0 or wo <= 0:
        return out
    frame_bytes = max(
        B * C * H * W * x.element_size(),  # one input frame copy
        B * O * ho * wo * 4,  # one fp32 accumulator frame (>= the Conv2d output frame)
        1,
    )
    chunk = max(1, min(tout, int(chunk_bytes) // frame_bytes))
    taps = [weight[:, :, k].contiguous() for k in range(kt)]
    acc_dtype = torch.promote_types(x.dtype, torch.float32)  # fp32, or fp64 for fp64 input
    bias32 = None if bias is None else bias.to(acc_dtype).view(1, 1, -1, 1, 1)
    for t0 in range(0, tout, chunk):
        n = min(tout, t0 + chunk) - t0
        acc = None
        for k in range(kt):
            # output j reads input frame j * st + k * dt - pt; keep the j whose frame exists
            offset = k * dt - pt
            j_lo = max(t0, -(offset // st) if offset < 0 else 0)
            j_hi = min(t0 + n, (T - 1 - offset) // st + 1) if T - 1 - offset >= 0 else t0
            m = j_hi - j_lo
            if m <= 0:
                continue
            i_lo = j_lo * st + offset
            frames = x[:, :, i_lo : i_lo + st * (m - 1) + 1 : st]  # (B, C, m, H, W)
            frames = frames.transpose(1, 2).reshape(B * m, C, H, W)
            y = F.conv2d(frames, taps[k], None, (sh, sw), (ph, pw), (dh, dw), groups)
            del frames
            y = y.view(B, m, O, ho, wo)
            if acc is None and m == n:
                acc = y if y.dtype == acc_dtype else y.to(acc_dtype)
            else:
                if acc is None:
                    acc = torch.zeros((B, n, O, ho, wo), device = x.device, dtype = acc_dtype)
                acc[:, j_lo - t0 : j_hi - t0].add_(y)
            del y
        if acc is None:
            acc = torch.zeros((B, n, O, ho, wo), device = x.device, dtype = acc_dtype)
        if bias32 is not None:
            acc.add_(bias32)
        out[:, :, t0 : t0 + n] = acc.transpose(1, 2)
        del acc
    return out


def _autocast_on(torch: Any, device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # torch < 2.4: no device_type argument
        return bool(torch.is_autocast_enabled())


def _call_bytes(conv: Any, x: Any) -> tuple:
    """(input bytes, output bytes) of this call as stock would run it."""
    itemsize = x.element_size()
    B, _, T, H, W = x.shape
    pad = conv.padding
    if conv.padding_mode != "zeros":
        T, H, W = (s + 2 * p for s, p in zip((T, H, W), pad))
        pad = (0, 0, 0)
    dims = [
        max(_out_size(s, k, st, p, d), 0)
        for s, k, st, p, d in zip((T, H, W), conv.kernel_size, conv.stride, pad, conv.dilation)
    ]
    return x.numel() * itemsize, B * conv.out_channels * dims[0] * dims[1] * dims[2] * itemsize


def _pertap_eligible(conv: Any, x: Any) -> bool:
    import torch

    # is_compiling first: a traced env read guards on the env value and recompiles when it flips
    if torch.compiler.is_compiling():
        return False
    mode = _mode()
    if mode == "off":
        return False
    if x.dim() != 5 or x.device.type not in ENGAGE_DEVICE_TYPES or isinstance(conv.padding, str):
        return False
    # autocast: stock returns the autocast dtype, the per-tap output keeps x's dtype
    if str(x.dtype).rsplit(".", 1)[-1] not in ENGAGE_DTYPES or _autocast_on(torch, x.device.type):
        return False
    if torch.is_grad_enabled() and (x.requires_grad or conv.weight.requires_grad):
        return False
    if mode == "force":
        return True
    if getattr(conv, "_unsloth_mps_pertap_old_torch", True):
        return True
    in_bytes, out_bytes = _call_bytes(conv, x)
    if max(in_bytes, out_bytes) > SIZE_GUARD_BYTES:
        return True
    return conv.in_channels >= WIDE_CHANNELS and x.shape[3] * x.shape[4] <= WIDE_MAX_HW


def _make_conv_forward(module: Any, original: Any) -> Any:
    def _conv_forward(input: Any, weight: Any, bias: Any) -> Any:
        if not _pertap_eligible(module, input):
            return original(input, weight, bias)
        padding = module.padding
        if module.padding_mode != "zeros":
            import torch.nn.functional as F

            # the stock pre-pad, verbatim
            input = F.pad(input, module._reversed_padding_repeated_twice, mode = module.padding_mode)
            padding = (0, 0, 0)
        setattr(module, _CALLS, getattr(module, _CALLS, 0) + 1)
        return pertap_conv3d(
            input, weight, bias, module.stride, padding, module.dilation, module.groups
        )

    _conv_forward._unsloth_mps_pertap = True
    return _conv_forward


def _vae_roots(vae: Any) -> list:
    return [vae] if callable(getattr(vae, "modules", None)) else []


def install(
    vae: Any,
    logger: Any = None,
    *,
    torch_version: Optional[str] = None,
) -> int:
    """Wrap every ``nn.Conv3d._conv_forward`` in ``vae`` (encoder and decoder); returns how many carry the wrap.

    Idempotent; skips convs whose ``_conv_forward`` something else already replaced. Device-agnostic: the per-call
    gate only engages on MPS."""
    if vae is None or pertap_disabled():
        return 0
    import torch

    old_torch = torch_version_tuple(torch_version or torch.__version__) < FAST_NATIVE_TORCH
    patched = 0
    for root in _vae_roots(vae):
        for module in root.modules():
            if not isinstance(module, torch.nn.Conv3d):
                continue
            if getattr(module, _MARK, False):
                patched += 1
                continue
            if "_conv_forward" in module.__dict__:
                continue
            module._conv_forward = _make_conv_forward(module, module._conv_forward)
            module._unsloth_mps_pertap_old_torch = old_torch
            setattr(module, _MARK, True)
            patched += 1
    if patched and logger is not None:
        logger.info(
            "video.vae_mps_pertap: %s runs %d Conv3d as per-tap Conv2d on MPS (%s)",
            type(vae).__name__,
            patched,
            "every call, torch < 2.14"
            if old_torch
            else f"torch >= 2.14: >= {WIDE_CHANNELS}ch over <= {WIDE_MAX_HW} px, or a tensor over "
            f"{SIZE_GUARD_BYTES // 2**20} MiB",
        )
    return patched


def install_on_pipe(
    pipe: Any,
    target: Any,
    logger: Any = None,
) -> int:
    if getattr(target, "device", None) != "mps":
        return 0
    return install(getattr(pipe, "vae", None), logger)


def pertap_calls(vae: Any) -> int:
    """How many Conv3d calls in ``vae`` took the per-tap path since load."""
    if vae is None or not callable(getattr(vae, "modules", None)):
        return 0
    return sum(int(getattr(m, _CALLS, 0)) for m in vae.modules())

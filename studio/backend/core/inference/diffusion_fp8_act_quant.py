# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One-pass static e4m3 activation quantiser for the ModelOpt FP8 Linear: byte-identical to the eager
``clamp(x.float() / input_scale, -448, 448).to(float8_e4m3fn)``, verified per device before first use."""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

FAST_QUANT_ENV = "UNSLOTH_MODELOPT_FP8_FAST_QUANT"
FP8_E4M3_MAX = 448.0
_BLOCK = 2048
_WARPS = 4

_LOCK = threading.Lock()
_STATE: dict = {"verified": {}, "calls": 0, "reason": ""}

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None

if triton is not None:

    @triton.jit
    def _fp8_static_quant_kernel(x_ptr, s_ptr, q_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask = mask, other = 0.0).to(tl.float32)
        s = tl.load(s_ptr)
        y = tl.math.div_rn(x, s)
        # torch.clamp keeps NaN; a comparison select does too (NaN compares false)
        y = tl.where(y > 448.0, 448.0, y)
        y = tl.where(y < -448.0, -448.0, y)
        # Integer RNE rounding: Triton's f32 -> e4m3 cast is not single-rounded on every arch (sm_89 differed from torch).
        bits = y.to(tl.uint32, bitcast = True)
        sign = ((bits >> 24) & 0x80).to(tl.uint8)
        mag_bits = bits & 0x7FFFFFFF
        mag = mag_bits.to(tl.float32, bitcast = True)
        # normal (|y| >= 2^-6): drop 20 mantissa bits with round-half-even, rebias the exponent from 127 to 7
        n_code = ((mag_bits + 0x7FFFF + ((mag_bits >> 20) & 1)) >> 20) - (120 << 3)
        # subnormal: |y| / 2^-9 rounded half-even by the 2^23 add (exact scaling, one rounding), 8 = 2^-6
        d_code = (mag * 512.0 + 8388608.0).to(tl.uint32, bitcast = True) - 0x4B000000
        code = tl.where(mag >= 0.015625, n_code, d_code).to(tl.uint8) | sign
        # torch writes NaN as 0x7f with the input's sign bit
        code = tl.where(y != y, sign | 0x7F, code)
        tl.store(q_ptr + offs, code, mask = mask)


def enabled() -> bool:
    return os.environ.get(FAST_QUANT_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def describe() -> dict:
    return {
        "verified": dict(_STATE["verified"]),
        "calls": _STATE["calls"],
        "reason": _STATE["reason"],
    }


def reference(x2: Any, scale: Any) -> Any:
    """The eager formula (diffusion_modelopt_fp8 ``_quantize_static`` on one chunk)."""
    import torch
    return (
        (x2.to(torch.float32) / scale).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    )


def _launch(x2: Any, scale: Any) -> Any:
    import torch

    out = torch.empty(x2.shape, dtype = torch.uint8, device = x2.device)
    n = x2.numel()
    if n:
        _fp8_static_quant_kernel[(triton.cdiv(n, _BLOCK),)](
            x2, scale, out, n, BLOCK = _BLOCK, num_warps = _WARPS
        )
    return out.view(torch.float8_e4m3fn)


def _verify(device: Any) -> bool:
    """Kernel codes must equal the eager formula's byte for byte on every bf16 / fp16 bit pattern."""
    import torch

    bits = torch.arange(-(1 << 15), 1 << 15, dtype = torch.int32, device = device).to(torch.int16)
    gen = torch.Generator(device = device).manual_seed(0)
    scales = [2.0**e for e in range(-20, 7)] + (
        torch.rand(24, generator = gen, device = device, dtype = torch.float32) * 0.5 + 1e-4
    ).tolist()
    xf = torch.randn((1 << 16) + 37, generator = gen, device = device, dtype = torch.float32) * 64
    for dt in (torch.bfloat16, torch.float16):
        x = bits.view(dt).reshape(256, 256)
        for s in scales:
            st = torch.tensor([s], dtype = torch.float32, device = device)
            if not torch.equal(
                _launch(x, st).view(torch.uint8), reference(x, st).view(torch.uint8)
            ):
                _STATE["reason"] = f"mismatch for {dt} at scale {s!r}"
                return False
    for s in scales[:6]:
        st = torch.tensor([s], dtype = torch.float32, device = device)
        if not torch.equal(_launch(xf, st).view(torch.uint8), reference(xf, st).view(torch.uint8)):
            _STATE["reason"] = f"mismatch for float32 at scale {s!r}"
            return False
    return True


def ready(device: Any) -> bool:
    if triton is None or not enabled():
        return False
    index = device.index
    state = _STATE["verified"].get(index)
    if state is not None:
        return state
    with _LOCK:
        state = _STATE["verified"].get(index)
        if state is None:
            import torch
            try:
                state = torch.cuda.get_device_capability(index) >= (8, 9) and _verify(device)
            except Exception as exc:  # noqa: BLE001 - any compile / launch failure keeps the eager formula
                _STATE["reason"] = f"{type(exc).__name__}: {exc}"[:2000]
                state = False
            _STATE["verified"][index] = state
    return state


def quantize_static(x2: Any, scale: Any) -> Optional[Any]:
    """e4m3 codes of a 2-D CUDA activation, or None: the caller runs the eager formula."""
    import torch

    if (
        triton is None
        or not x2.is_cuda
        or not x2.is_contiguous()
        or x2.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or scale.device != x2.device
        or torch.compiler.is_compiling()
    ):
        return None
    if not ready(x2.device):
        return None
    _STATE["calls"] += 1
    return _launch(x2, scale)

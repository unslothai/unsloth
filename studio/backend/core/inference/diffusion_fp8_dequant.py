# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eager widening of int8 / stored-fp8 codes with a per-row scale, without full float32 temporaries: one Triton
kernel on CUDA / ROCm, bounded row chunks elsewhere. Bitwise equal to ``(codes.float() * scale[:, None]).to(dtype)``
for every non-NaN value; torch imported lazily"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

# ``0`` widens a whole layer at once again (two float32 matrices per call).
FP8_DEQUANT_FUSED_ENV = "UNSLOTH_DIFFUSION_FP8_DEQUANT_FUSED"
# float32 scratch per chunk; a layer whose whole widen fits stays one expression.
CHUNK_SCRATCH_BYTES_CPU = 16 << 20
CHUNK_SCRATCH_BYTES_ACCEL = 64 << 20
_TRITON_BLOCK = 1024
_TRITON_STATE: dict = {"kernel": None, "failed": None}
_BF16_NAN: dict = {}
# Specialisations launched once in eager, so a CUDA-graph capture may replay-record them.
_WARM: set = set()
_FMT_ID = {None: 0, "float8_e4m3fn": 1, "float8_e5m2": 2}


def fused_dequant_enabled() -> bool:
    return os.environ.get(FP8_DEQUANT_FUSED_ENV, "1").strip() != "0"


def chunk_scratch_bytes(device: Any) -> int:
    return CHUNK_SCRATCH_BYTES_CPU if device.type == "cpu" else CHUNK_SCRATCH_BYTES_ACCEL


def chunk_rows(
    rows: int,
    cols: int,
    device: Any,
    per_elem: int = 4,
) -> int:
    """Rows per chunk under the scratch budget, or 0 when the old two-temporary widen (8 B/elem) fits it."""
    budget = chunk_scratch_bytes(device)
    if rows * cols * 8 <= budget:
        return 0
    return min(rows, max(1, budget // max(1, cols * per_elem)))


def _build_kernel() -> Any:
    from .diffusion_fp8_dequant_triton import dequant_rows_kernel
    return dequant_rows_kernel


def _bf16_nan(device: Any) -> int:
    import torch

    key = str(device)
    bits = _BF16_NAN.get(key)
    if bits is None:
        nan = torch.full((1,), float("nan"), dtype = torch.float32, device = device)
        bits = _BF16_NAN[key] = int(nan.to(torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    return bits


def launch_triton(codes: Any, scale: Any, out: Any, fmt: Optional[str]) -> None:
    """``out[r, c] = (decode(codes[r, c]) * scale[r]).to(out.dtype)``; contiguous 2-D codes / out, fp32 scale."""
    import torch
    import triton

    kernel = _TRITON_STATE["kernel"]
    if kernel is None:
        kernel = _TRITON_STATE["kernel"] = _build_kernel()
    rows, cols = codes.shape
    grid = (rows, triton.cdiv(cols, _TRITON_BLOCK))
    # No FMA contraction: the multiply rounds once, as torch's.
    kernel[grid](
        codes,
        scale,
        out,
        cols,
        FMT = _FMT_ID[fmt],
        BF16 = out.dtype == torch.bfloat16,
        BF16_NAN = _bf16_nan(out.device) if out.dtype == torch.bfloat16 else 0,
        BLOCK = _TRITON_BLOCK,
        num_warps = 4,
        enable_fp_fusion = False,
    )


def _triton_ok(codes: Any, dtype: Any) -> bool:
    import torch

    if _TRITON_STATE["failed"] is not None or not codes.is_cuda or not codes.is_contiguous():
        return False
    if dtype not in (torch.bfloat16, torch.float16, torch.float32) or codes.dim() != 2:
        return False
    rows, cols = codes.shape
    if not (0 < rows < (1 << 31) and 0 < cols and -(-cols // _TRITON_BLOCK) < 65536):
        return False
    # Under CUDA-graph capture only an already compiled specialisation (no JIT, no NaN probe sync).
    return not torch.cuda.is_current_stream_capturing() or _warm_key(codes, dtype) in _WARM


def _warm_key(codes: Any, dtype: Any) -> tuple:
    cols = codes.shape[1]
    return (str(codes.device), codes.dtype, dtype, cols % 16 == 0, cols == 1)


def _triton_failed(exc: BaseException) -> None:
    import logging
    _TRITON_STATE["failed"] = repr(exc)
    logging.getLogger(__name__).debug(
        "diffusion.fp8_dequant: Triton widen unavailable (%s); chunked torch from here on", exc
    )


def dequant_rows(
    codes: Any,
    scale: Any,
    dtype: Any,
    widen: Callable[[Any], Any],
    fmt: Optional[str] = None,
    triton_ok: bool = True,
) -> Any:
    """``(widen(codes) * scale[:, None]).to(dtype)`` for 2-D ``codes`` and an fp32 ``scale`` of shape [rows].
    ``widen`` maps a row slice to its exact float32 values; ``fmt`` names the fp8 bits (None: integer codes)."""
    import torch

    if triton_ok and _triton_ok(codes, dtype):
        out = torch.empty(codes.shape, dtype = dtype, device = codes.device)
        try:
            launch_triton(codes, scale.contiguous(), out, fmt)
            _WARM.add(_warm_key(codes, dtype))
            return out
        except Exception as exc:  # noqa: BLE001 - no Triton / launch failure: chunked torch below
            _triton_failed(exc)
        del out
    rows = codes.shape[0]
    cols = codes.numel() // max(1, rows)
    step = chunk_rows(rows, cols, codes.device, 8 if codes.device.type == "mps" else 4)
    if not step:
        return (widen(codes) * scale[:, None]).to(dtype)
    out = torch.empty(codes.shape, dtype = dtype, device = codes.device)
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        wide = widen(codes[start:stop])
        wide.mul_(scale[start:stop, None])
        out[start:stop].copy_(wide)
        del wide
    return out

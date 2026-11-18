# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton decode of packed W4A8 / W6A8 rows to the int8 grid, bit-identical to ``decode_int8_torch``.

The e4m3 step is decoded by integer ops (no fp8 convert below sm_89); the launch disables fp fusion so the float32
product rounds before the half-even decision, as in torch."""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_STATE: dict = {}
_OP: dict = {}
_BLOCK_R = 16
_BLOCK_C = 256


def _build() -> Any:
    """``(triton, kernel)`` or None when Triton is missing or this module's kernel cannot be defined."""
    with _LOCK:
        if "kernel" in _STATE:
            return _STATE["kernel"]
        try:
            import triton

            from .diffusion_grouped_int8_kernels import grouped_decode_kernel
            _STATE["kernel"] = (triton, grouped_decode_kernel)
        except Exception as exc:  # noqa: BLE001 -- no Triton: the torch decode is exact too
            _STATE["kernel"] = None
            _STATE["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return _STATE["kernel"]


def _failed(exc: BaseException) -> None:
    with _LOCK:
        _STATE["kernel"] = None
        _STATE["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"


def _interpreting() -> bool:
    return (os.environ.get("TRITON_INTERPRET") or "").strip() == "1"


def decode_int8_triton(
    qdata: Any, s_rel: Any, codebook: Any, group_size: int, bits: int
) -> Optional[Any]:
    """The int8 ``[N, K]`` grid of packed ``qdata`` / ``s_rel`` (see ``diffusion_grouped_int8``), or None when this
    kernel cannot take the inputs here (not CUDA / ROCm, not contiguous, bad shapes, no Triton, a failed build)."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    try:
        if bits not in (4, 6) or int(group_size) < 1:
            return None
        if qdata.dim() != 2 or s_rel.dim() != 2 or qdata.shape[0] != s_rel.shape[0]:
            return None
        if qdata.dtype not in (torch.int8, torch.uint8) or s_rel.dtype not in (
            torch.uint8,
            torch.float8_e4m3fn,
        ):
            return None
        device = qdata.device
        if not (qdata.is_cuda or (_interpreting() and device.type == "cpu")):
            return None
        if s_rel.device != device or not qdata.is_contiguous() or not s_rel.is_contiguous():
            return None
        rows, cols = int(qdata.shape[0]), int(qdata.shape[1])
        groups = int(s_rel.shape[1])
        k = groups * int(group_size)
        if cols * 8 != k * bits or k % 4:
            return None
        if codebook is not None:
            if (
                bits != 4
                or codebook.numel() != 16
                or codebook.dtype != torch.float32
                or codebook.device != device
                or not codebook.is_contiguous()
            ):
                return None
    except Exception:  # noqa: BLE001 -- not something this kernel takes
        return None
    built = _build()
    if built is None:
        return None
    triton, kernel = built
    out = torch.empty((rows, k), dtype = torch.int8, device = device)
    if rows == 0 or k == 0:
        return out
    q = qdata.view(torch.uint8)
    sf = s_rel.view(torch.uint8)
    cb = codebook if codebook is not None else out
    grid = (triton.cdiv(rows, _BLOCK_R), triton.cdiv(k, _BLOCK_C))
    import contextlib

    guard = torch.cuda.device(device) if qdata.is_cuda else contextlib.nullcontext()
    try:
        with guard:
            kernel[grid](
                q,
                sf,
                cb,
                out,
                rows,
                k,
                cols,
                groups,
                GROUP = int(group_size),
                BITS = int(bits),
                HAS_CB = codebook is not None,
                BLOCK_R = _BLOCK_R,
                BLOCK_C = _BLOCK_C,
                # the float32 product must round before the half-even decision, as torch's separate ops do
                enable_fp_fusion = False,
            )
    except Exception as exc:  # noqa: BLE001 -- a kernel that cannot build here: the torch decode is exact
        _failed(exc)
        return None
    return out


def _decode_eager(qdata: Any, s_rel: Any, codebook: Any, group_size: int, bits: int) -> Any:
    out = decode_int8_triton(qdata, s_rel, codebook, group_size, bits)
    if out is not None:
        return out
    from .diffusion_grouped_int8 import decode_int8_torch

    return decode_int8_torch(qdata, s_rel, codebook, group_size, bits)


def decode_int8_op() -> Any:
    """``torch.ops.unsloth_grouped_int8.decode``: opaque to the compiler (shape-only fake), one per process."""
    with _LOCK:
        if "op" in _OP:
            return _OP["op"]
        import torch

        # explicit schema: this module's annotations are strings (no module-level torch to resolve them)
        @torch.library.custom_op(
            "unsloth_grouped_int8::decode",
            mutates_args = (),
            schema = "(Tensor qdata, Tensor s_rel, Tensor? codebook, int group_size, int bits) -> Tensor",
        )
        def decode(qdata, s_rel, codebook, group_size, bits):
            return _decode_eager(qdata, s_rel, codebook, group_size, bits)

        @decode.register_fake
        def _(qdata, s_rel, codebook, group_size, bits):
            return qdata.new_empty((qdata.shape[0], s_rel.shape[1] * group_size), dtype = torch.int8)

        _OP["op"] = decode
        return decode


_EPI_BLOCK = (32, 128)


def int8_epilogue_triton(acc: Any, xs: Any, ws: Any, bias: Any) -> Optional[Any]:
    """bf16 ``[M, N]`` of torchao's epilogue on an int32 ``torch._int_mm`` result with fp32 row / column scales (and a
    bf16 bias) in one pass, bit-identical to ``diffusion_int8_gemm.reference``; None when not covered here."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    try:
        if not acc.is_cuda or acc.dtype != torch.int32 or acc.dim() != 2 or not acc.is_contiguous():
            return None
        m, n = int(acc.shape[0]), int(acc.shape[1])
        if xs.dtype != torch.float32 or ws.dtype != torch.float32:
            return None
        if xs.numel() != m or ws.numel() != n or xs.device != acc.device or ws.device != acc.device:
            return None
        if bias is not None and (
            bias.dtype != torch.bfloat16 or bias.numel() != n or bias.device != acc.device
        ):
            return None
    except Exception:  # noqa: BLE001
        return None
    with _LOCK:
        kernel = _STATE.get("epilogue", False)
        if kernel is False:
            try:
                from .diffusion_grouped_int8_kernels import int8_epilogue_kernel as kernel
            except Exception:  # noqa: BLE001
                kernel = None
            _STATE["epilogue"] = kernel
    if kernel is None:
        return None
    import triton

    out = torch.empty((m, n), dtype = torch.bfloat16, device = acc.device)
    if m == 0 or n == 0:
        return out
    bm, bn = _EPI_BLOCK
    xs = xs.reshape(-1).contiguous()
    ws = ws.reshape(-1).contiguous()
    try:
        with torch.cuda.device(acc.device):
            kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                acc,
                xs,
                ws,
                bias.contiguous() if bias is not None else ws,
                out,
                m,
                n,
                HAS_BIAS = bias is not None,
                BLOCK_M = bm,
                BLOCK_N = bn,
                enable_fp_fusion = False,
            )
    except Exception:  # noqa: BLE001 -- the eager epilogue is the same maths
        with _LOCK:
            _STATE["epilogue"] = None
        return None
    return out

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 conditioner in the reference workflow's ``nvfp4_awq`` layout, read with the reference runtime's numerics.

Each decoder projection stores an NVFP4 weight (two e2m1 codes per byte, the high nibble first), e4m3 block scales
per 16 inputs in the 128x4 swizzled layout, a float32 global scale, and on some projections an AWQ
``pre_quant_scale``. The file marks every projection ``full_precision_matrix_mult``: the reference multiplies the
input by ``pre_quant_scale`` in the compute dtype, dequantizes the weight, and runs a dense matmul (weight-only, on
every GPU). The dequantization is the reference CUDA kernel's: ``code * (block_scale * global_scale)`` in float32,
rounded once. The embedding table is int8 with a float32 scale per row; it is dequantized once at load
(``q * scale`` in float32, rounded), which is the value the reference's row lookup returns.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

NVFP4_AWQ_EXPECTED_QUANT = {"format": "nvfp4", "full_precision_matrix_mult": True}
EMBED_EXPECTED_QUANT = {"format": "int8_tensorwise"}
GLOBAL_SCALE_SUFFIX = ".weight_scale_2"
PRE_QUANT_SUFFIX = ".pre_quant_scale"
BLOCK = 16
# e2m1 code -> value (sign in bit 3)
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def is_nvfp4_awq_file(names: Any) -> bool:
    """A tensor-name set in this layout (a global scale next to each quantized weight)."""
    return any(str(n).endswith(GLOBAL_SCALE_SUFFIX) for n in names)


def validate_quant_meta(blob: Any, name: str, expected: dict) -> None:
    """Raise unless a tensor's quant-metadata blob names exactly the format this loader implements."""
    import json

    meta = json.loads(blob.cpu().numpy().tobytes().decode("utf-8").rstrip("\x00"))
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(
                f"{name}: unsupported quant metadata {field}={meta.get(field)!r} "
                f"(this loader implements {field}={value!r})"
            )


def plain_block_scales(swizzled: Any, rows: int, cols: int) -> Any:
    """The 128x4-swizzled e4m3 block scales as the plain ``[rows, cols // 16]`` matrix (done once, at load)."""
    from .diffusion_nvfp4_ops import unswizzle_sf
    return unswizzle_sf(swizzled, rows, cols).reshape(rows, cols // BLOCK).contiguous()


def dequantize_nvfp4_reference(
    weight: Any, block_scales: Any, global_scale: Any, dtype: Any
) -> Any:
    """Dense ``[N, K]`` weight from packed ``[N, K // 2]`` codes (high nibble first) and plain block scales:
    ``code * (float32(block_scale) * global_scale)`` in float32, rounded once to ``dtype``."""
    import torch

    rows, half = weight.shape
    lut = torch.tensor(_E2M1, dtype = torch.float32, device = weight.device)
    w = weight.view(torch.uint8)
    codes = torch.stack((w >> 4, w & 0x0F), dim = -1).reshape(rows, half * 2).long()
    decode = block_scales.view(torch.float8_e4m3fn).to(torch.float32) * global_scale.to(
        torch.float32
    ).reshape(1)
    values = lut[codes].reshape(rows, -1, BLOCK) * decode.unsqueeze(-1)
    return values.reshape(rows, half * 2).to(dtype)


@lru_cache(maxsize = None)
def _dequant_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def _pow2(k):
        # exact 2^k for a normal-range integer k, built from the exponent bits (no approximate exp2)
        return ((k + 127) << 23).to(tl.int32).to(tl.float32, bitcast = True)

    @triton.jit
    def _e2m1(code):
        # code: int32 0..15 -> float32 value (exact)
        mag = code & 7
        e = mag >> 1
        m = (mag & 1).to(tl.float32)
        normal = (1.0 + 0.5 * m) * _pow2(e - 1)
        val = tl.where(e == 0, 0.5 * m, normal)
        return tl.where((code & 8) != 0, -val, val)

    @triton.jit
    def _e4m3(bits):
        # bits: int32 0..255 of an e4m3fn scale -> float32 (exact; NaN codes never appear in block scales)
        e = (bits >> 3) & 15
        m = (bits & 7).to(tl.float32)
        sub = m * 0.001953125  # m / 8 * 2^-6
        normal = (1.0 + m * 0.125) * _pow2(e - 7)
        val = tl.where(e == 0, sub, normal)
        return tl.where((bits & 128) != 0, -val, val)

    @triton.jit
    def _kernel(w_ptr, s_ptr, g_ptr, out_ptr, n_cols, BLOCK_C: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = cols < n_cols
        byte = tl.load(w_ptr + row * (n_cols // 2) + cols // 2, mask = mask, other = 0).to(tl.int32)
        code = tl.where((cols % 2) == 0, byte >> 4, byte & 15)
        sbits = tl.load(s_ptr + row * (n_cols // 16) + cols // 16, mask = mask, other = 0).to(tl.int32)
        g = tl.load(g_ptr)
        decode = _e4m3(sbits) * g
        val = _e2m1(code) * decode
        tl.store(out_ptr + row * n_cols + cols, val.to(out_ptr.dtype.element_ty), mask = mask)

    return _kernel


def dequantize_nvfp4(weight: Any, block_scales: Any, global_scale: Any, dtype: Any) -> Any:
    """``dequantize_nvfp4_reference``, as one fused Triton pass on NVIDIA CUDA (bit-identical) and the torch
    math elsewhere."""
    import torch

    from .rocm_bf16 import is_rocm_torch

    if weight.is_cuda and not is_rocm_torch(torch) and dtype in (torch.bfloat16, torch.float16):
        try:
            rows, half = weight.shape
            cols = half * 2
            out = torch.empty(rows, cols, device = weight.device, dtype = dtype)
            block_c = 1024
            g = global_scale.to(device = weight.device, dtype = torch.float32).reshape(1)
            with torch.cuda.device(weight.device):
                _dequant_kernel()[(rows, (cols + block_c - 1) // block_c)](
                    weight.view(torch.uint8),
                    block_scales.view(torch.uint8),
                    g,
                    out,
                    cols,
                    BLOCK_C = block_c,
                )
            return out
        except torch.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001 -- no Triton here: the torch math
            pass
    return dequantize_nvfp4_reference(weight, block_scales, global_scale, dtype)


@lru_cache(maxsize = None)
def nvfp4_awq_linear_class() -> Any:
    """Built once: a fresh class on a second load would retrace every compiled block."""
    import torch
    from torch import nn

    class Nvfp4AwqLinear(nn.Module):
        """A bias-free Linear whose weight stays NVFP4 for its whole residency (about half the int8
        footprint). Each call dequantizes a transient dense weight, as the int8 path does."""

        def __init__(self, in_features: int, out_features: int, pre_quant: bool) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.register_buffer(
                "weight",
                torch.empty(out_features, in_features // 2, dtype = torch.uint8, device = "meta"),
                persistent = True,
            )
            self.register_buffer(
                "weight_scale",
                torch.empty(
                    out_features, in_features // BLOCK, dtype = torch.float8_e4m3fn, device = "meta"
                ),
                persistent = True,
            )
            self.register_buffer(
                "weight_scale_2",
                torch.empty((), dtype = torch.float32, device = "meta"),
                persistent = True,
            )
            if pre_quant:
                self.register_buffer(
                    "pre_quant_scale",
                    torch.empty(in_features, dtype = torch.bfloat16, device = "meta"),
                    persistent = True,
                )
            else:
                self.pre_quant_scale = None
            self.bias = None

        def forward(self, x: Any) -> Any:  # noqa: D102
            if self.pre_quant_scale is not None:
                # the reference: input * pre_quant_scale cast to the input dtype, before the matmul
                x = x * self.pre_quant_scale.to(device = x.device, dtype = x.dtype)
            weight = dequantize_nvfp4(self.weight, self.weight_scale, self.weight_scale_2, x.dtype)
            return torch.nn.functional.linear(x, weight)

        def extra_repr(self) -> str:  # noqa: D102
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"nvfp4_awq(pre_quant={self.pre_quant_scale is not None})"
            )

    return Nvfp4AwqLinear


def dequantize_int8_embedding(weight: Any, row_scale: Any, dtype: Any) -> Any:
    """The int8 table with a float32 scale per row, as the reference's row lookup returns it."""
    import torch
    return (weight.to(torch.float32) * row_scale.to(torch.float32).reshape(-1, 1)).to(dtype)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""W4A16 NVFP4: packed E2M1 weights (compressed-tensors `nvfp4-pack-quantized`) dequantized to 16 bit on the fly."""

import torch
import triton
import triton.language as tl

from .fp8 import _fp8_triton_device_context, _is_transposed_view, _opaque_under_compile

__all__ = [
    "NVFP4_GROUP_SIZE",
    "NVFP4QuantState",
    "nvfp4_dequantize",
    "NVFP4Linear_matmul",
    "nvfp4_linear",
]

NVFP4_GROUP_SIZE = 16
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class NVFP4QuantState:
    __slots__ = ("scale", "global_scale", "shape", "dtype")

    def __init__(self, scale, global_scale, shape, dtype = torch.bfloat16):
        self.scale = scale
        self.global_scale = global_scale
        self.shape = tuple(shape)
        self.dtype = dtype


def _nvfp4_dequantize_torch(packed, scale, global_scale, dtype):
    # Same arithmetic as compressed_tensors NVFP4PackedCompressor.decompress: scale / global_scale in fp32, times E2M1.
    rows, half = packed.shape
    lut = torch.tensor(_E2M1, dtype = torch.float32, device = packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim = -1).reshape(rows, half * 2)
    values = lut[(codes & 0x07).long()] * torch.where((codes & 0x08).bool(), -1.0, 1.0)
    group_scale = scale.to(torch.float32) / global_scale.to(torch.float32).reshape(-1)[:1]
    group_scale = group_scale.repeat_interleave(NVFP4_GROUP_SIZE, dim = 1)[:, : half * 2]
    return (values * group_scale).to(dtype)


@triton.jit
def _e2m1_to_float(code):
    mag = code & 7
    exp = mag >> 1
    # mag 0,1 -> 0, 0.5 ; mag >= 2 -> (2 + mantissa) * 2^(exp - 2)
    pow2 = tl.where(exp == 1, 0.5, tl.where(exp == 2, 1.0, 2.0))
    value = tl.where(mag < 2, mag.to(tl.float32) * 0.5, (2 + (mag & 1)).to(tl.float32) * pow2)
    return value * tl.where((code & 8) != 0, -1.0, 1.0)  # multiply, not negate: code 8 is -0.0


@triton.jit
def _nvfp4_dequant_kernel(
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    out_ptr,
    half_cols,
    scale_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_cols
    byte = tl.load(packed_ptr + row * half_cols + offs, mask = mask, other = 0).to(tl.int32)
    # Two columns per byte, 16 columns per scale: packed column p is in group p // 8.
    scale = tl.load(scale_ptr + row * scale_cols + offs // 8, mask = mask, other = 0.0).to(tl.float32)
    group_scale = tl.math.div_rn(scale, tl.load(global_scale_ptr).to(tl.float32))
    lo = _e2m1_to_float(byte & 15) * group_scale
    hi = _e2m1_to_float(byte >> 4) * group_scale
    out_row = out_ptr + row * half_cols * 2 + offs * 2
    tl.store(out_row, lo.to(out_ptr.dtype.element_ty), mask = mask)
    tl.store(out_row + 1, hi.to(out_ptr.dtype.element_ty), mask = mask)


@_opaque_under_compile(
    "nvfp4_dequantize_triton",
    lambda packed, scale, global_scale, dtype: packed.new_empty(
        (packed.shape[0], packed.shape[1] * 2), dtype = dtype
    ),
)
def _nvfp4_dequantize_triton(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows, half = packed.shape
    out = torch.empty((rows, half * 2), dtype = dtype, device = packed.device)
    BLOCK = 1024
    with _fp8_triton_device_context(packed):
        _nvfp4_dequant_kernel[(rows, triton.cdiv(half, BLOCK))](
            packed, scale, global_scale, out, half, scale.shape[1], BLOCK = BLOCK
        )
    return out


def nvfp4_dequantize(packed, scale, global_scale, dtype = torch.bfloat16):
    """Dense (out, in) weight of a packed (out, in / 2) NVFP4 tensor; a transposed view gives the transposed weight."""
    if _is_transposed_view(packed):
        return nvfp4_dequantize(packed.t(), scale, global_scale, dtype).t()
    if packed.shape[1] * 2 > scale.shape[1] * NVFP4_GROUP_SIZE:
        raise ValueError(
            f"Unsloth: NVFP4 weight {tuple(packed.shape)} needs {packed.shape[1] * 2 // NVFP4_GROUP_SIZE} "
            f"group scales per row, got {tuple(scale.shape)}."
        )
    if packed.device.type in ("cuda", "xpu"):
        return _nvfp4_dequantize_triton(
            packed.contiguous(), scale.contiguous(), global_scale.reshape(-1)[:1], dtype
        )
    return _nvfp4_dequantize_torch(packed, scale, global_scale, dtype)


# Opaque to torch.compile as whole matmuls: if the dequantized weight were a graph value, AOTAutograd would save it for
# backward in every layer (3.6x the NVFP4 storage each), which is exactly the memory W4A16 exists to avoid.
@_opaque_under_compile(
    "nvfp4_matmul_fwd",
    lambda X, packed, scale, global_scale: X.new_empty(X.shape[:-1] + (packed.shape[0],)),
)
def _nvfp4_matmul_fwd(
    X: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    return torch.matmul(X, nvfp4_dequantize(packed, scale, global_scale, X.dtype).t())


@_opaque_under_compile(
    "nvfp4_matmul_bwd",
    lambda grad, packed, scale, global_scale: grad.new_empty(grad.shape[:-1] + (packed.shape[1] * 2,)),
)
def _nvfp4_matmul_bwd(
    grad: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    return torch.matmul(grad, nvfp4_dequantize(packed, scale, global_scale, grad.dtype))


class NVFP4Linear_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, packed, scale, global_scale, bias = None):
        output = _nvfp4_matmul_fwd(X, packed, scale, global_scale)
        if bias is not None:
            output = output + bias
            # A float32 bias promotes; cast back only then (a no-op .to() aliases, zeroing dX under torch 2.11 compile).
            if output.dtype != X.dtype:
                output = output.to(X.dtype)
        ctx.save_for_backward(packed, scale, global_scale)
        ctx.bias_grad = bias is not None and bias.requires_grad
        ctx.dtype = X.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        packed, scale, global_scale = ctx.saved_tensors
        grad_X = None
        if ctx.needs_input_grad[0]:
            grad_X = _nvfp4_matmul_bwd(grad_output, packed, scale, global_scale)
            if grad_X.dtype != ctx.dtype:
                grad_X = grad_X.to(ctx.dtype)
        grad_bias = None
        if ctx.bias_grad:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_X, None, None, None, grad_bias


def nvfp4_linear(X, packed, scale, global_scale, bias = None):
    return NVFP4Linear_matmul.apply(X, packed, scale, global_scale, bias)

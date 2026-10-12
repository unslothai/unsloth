# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp8 compute for the frozen base linears of a LoRA run.

The weight is quantized once to e4m3 with a float scale per output row and the bf16 copy is freed (half the base
weight memory). Forward quantizes x per row; backward returns only grad_x = rowwise_quant(grad_y * s) @ Q, so no
weight gradient is computed and nothing from x is saved. torchao's Float8Linear instead re-quantizes the bf16 weight
about four times per layer per step and keeps it in bf16.

The GEMMs run with unit (tensorwise) scales and the row / column scales are applied after: torch routes rowwise
``_scaled_mm`` on sm100 to an untuned two-tile CUTLASS kernel, tensorwise to cuBLAS (aten ScaledBlas.cpp).
Only K and N must be multiples of 16; any token count works.
"""

import torch
import torch.nn as nn

_E4M3 = torch.float8_e4m3fn
_E4M3_MAX = float(torch.finfo(_E4M3).max)
_EPS = 1e-12


def _rowwise_quant(x2d):
    xf = x2d.float()
    amax = xf.abs().amax(dim = -1, keepdim = True).clamp_min(_EPS)
    q = (xf * (_E4M3_MAX / amax)).clamp(-_E4M3_MAX, _E4M3_MAX).to(_E4M3)
    return q, amax / _E4M3_MAX


def _unit_scaled_mm(a, b):
    one = torch.ones((), device = a.device, dtype = torch.float32)  # a fresh scalar: a buffer view fails to lower
    out = torch._scaled_mm(a, b, scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
    return out[0] if isinstance(out, tuple) else out  # torch < 2.5 also returned amax


@torch._dynamo.allow_in_graph  # as torchao's float8 matmul: inlining the Function trips a dynamo freevar assert
class _FrozenFp8MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, q, s_row, out_dtype):
        xq, sx = _rowwise_quant(x2d)
        y = (_unit_scaled_mm(xq, q.t()).float() * (sx * s_row)).to(out_dtype)
        ctx.save_for_backward(q, s_row)
        ctx.x_dtype = x2d.dtype
        return y

    @staticmethod
    def backward(ctx, gy):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        q, s_row = ctx.saved_tensors
        # The weight scale sits on the reduction dim of grad_y @ W, so it folds into grad_y; mat2 must be column-major.
        gq, sg = _rowwise_quant(gy.float() * s_row)
        gx = _unit_scaled_mm(gq, q.t().contiguous().t()).float() * sg
        return gx.to(ctx.x_dtype), None, None, None


class FrozenFp8Linear(nn.Module):
    """A frozen nn.Linear (a PEFT LoRA base_layer) with an fp8 weight; in / out features must be 16-aligned."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        out_f, in_f = linear.weight.shape
        if in_f % 16 or out_f % 16:
            raise ValueError(f"fp8 needs 16-aligned dims, got {out_f}x{in_f}")
        self.in_features, self.out_features = in_f, out_f
        with torch.no_grad():
            q, s = _rowwise_quant(linear.weight.detach())
        # Buffers: never trainable, never in an optimizer or PEFT's trainable scans.
        self.register_buffer("weight", q)
        self.register_buffer("weight_scale", s.reshape(1, out_f).contiguous())
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach(), requires_grad = False)

    def forward(self, x):
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        shape = x.shape
        y = _FrozenFp8MM.apply(x.reshape(-1, shape[-1]), self.weight, self.weight_scale, x.dtype)
        y = y.reshape(*shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8"


def convert_frozen_fp8(model: nn.Module, names) -> int:
    """Swap the frozen nn.Linear modules at ``names`` (fully qualified) for FrozenFp8Linear; returns the count."""
    count = 0
    for fqn in sorted(names):
        parent_name, _, child = fqn.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child)
        if type(old) is not nn.Linear or old.weight.requires_grad:
            continue
        new = FrozenFp8Linear(old)
        new._studio_lowp = "fp8"
        setattr(parent, child, new)
        count += 1
    return count

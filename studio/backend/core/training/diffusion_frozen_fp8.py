# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp8 compute for the frozen base linears of a LoRA run.

Each weight is quantized once to e4m3 (float scale per output row) and its bf16 copy freed; backward returns only
grad_x, so nothing from x is saved (torchao's Float8Linear re-quantizes the bf16 weight ~4x per layer per step).
GEMMs use unit scales with row scales applied after: rowwise ``_scaled_mm`` on sm100 routes to an untuned CUTLASS
kernel, tensorwise to cuBLAS (aten ScaledBlas.cpp).
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


@torch.no_grad()
def _quantize_weight(w, rows = 2048):
    # Row chunks bound the fp32 transient.
    q = torch.empty(w.shape, dtype = _E4M3, device = w.device)
    s = torch.empty((w.shape[0], 1), dtype = torch.float32, device = w.device)
    for i in range(0, w.shape[0], rows):
        q[i : i + rows], s[i : i + rows] = _rowwise_quant(w[i : i + rows])
    return q, s


def _unit_scaled_mm(a, b):
    one = torch.ones(
        (), device = a.device, dtype = torch.float32
    )  # a fresh scalar: a buffer view fails to lower
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
        q, s = _quantize_weight(linear.weight.detach())
        self.register_buffer("weight", q)
        self.register_buffer("weight_scale", s.reshape(1, out_f).contiguous())
        self.bias = (
            None if linear.bias is None else nn.Parameter(linear.bias.detach(), requires_grad = False)
        )

    def forward(self, x):
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        shape = x.shape
        y = _FrozenFp8MM.apply(x.reshape(-1, shape[-1]), self.weight, self.weight_scale, x.dtype)
        y = y.reshape(*shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8"


def convert_frozen_fp8(model: nn.Module, names) -> tuple[int, int]:
    """Swap each frozen nn.Linear in ``names`` for FrozenFp8Linear one at a time (bounds peak memory); a layer that
    fails stays bf16 rather than raising mid-conversion. Returns (converted, failed)."""
    done = failed = 0
    for fqn in sorted(names):
        parent_name, _, child = fqn.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child)
        if type(old) is not nn.Linear or old.weight.requires_grad:
            continue
        try:
            new = FrozenFp8Linear(old)
        except (RuntimeError, ValueError):
            failed += 1
            continue
        new._studio_lowp = "fp8"
        setattr(parent, child, new)
        done += 1
    return done, failed

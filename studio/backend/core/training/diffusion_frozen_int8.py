# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frozen base linears as int8 W8A8, quantized once: the int8 counterpart of ``diffusion_frozen_fp8`` for cards with
int8 tensor cores but no fp8 (A100 sm80, RTX 30xx sm86, T4 sm75).

The weight is stored per output row (symmetric, scale = amax / 127) and never re-quantized: it is frozen, so unlike
torchao's int8 mixed-precision training config (which re-quantizes the bf16 master every step) nothing is spent on
the weight after load. Forward: per-row (per token) dynamic activation codes, ``torch._int_mm`` into int32, both
scales applied in fp32. Backward returns only grad_x: the weight scale sits on the reduction dim of ``grad_y @ W``,
so it folds into grad_y before grad_y is quantized per row, and the same int8 weight (column-major) is the second
operand. No weight gradient, and x is not saved. Rows at or below ``torch._int_mm``'s M > 16 floor (AdaLN
modulation at batch 1) run the dequantized weight in the input dtype instead."""

from __future__ import annotations

import torch
import torch.nn as nn

_QMAX = 127.0
_EPS = 1e-12
_MIN_ROWS = 17  # torch._int_mm needs M > 16


def _rowwise_int8(x2d):
    xf = x2d.float()
    amax = xf.abs().amax(dim = -1, keepdim = True).clamp_min(_EPS)
    scale = amax / _QMAX
    q = torch.round(xf / scale).clamp(-_QMAX, _QMAX).to(torch.int8)
    return q, scale


@torch.no_grad()
def quantize_weight(w, rows = 2048):
    """Per-row symmetric int8 codes and fp32 scales [out, 1], in row chunks so the fp32 copy stays small."""
    q = torch.empty(w.shape, dtype = torch.int8, device = w.device)
    s = torch.empty((w.shape[0], 1), dtype = torch.float32, device = w.device)
    for i in range(0, w.shape[0], rows):
        q[i : i + rows], s[i : i + rows] = _rowwise_int8(w[i : i + rows])
    return q, s


def _dequant(q, s_row, dtype):
    return (q.float() * s_row.reshape(-1, 1)).to(dtype)


@torch._dynamo.allow_in_graph
class _FrozenInt8MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, q, s_row, out_dtype):
        ctx.save_for_backward(q, s_row)
        ctx.x_dtype = x2d.dtype
        if x2d.shape[0] < _MIN_ROWS:
            return x2d.to(out_dtype) @ _dequant(q, s_row, out_dtype).t()
        xq, sx = _rowwise_int8(x2d)
        acc = torch._int_mm(xq, q.t())
        return (acc.float() * sx * s_row).to(out_dtype)

    @staticmethod
    def backward(ctx, gy):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        q, s_row = ctx.saved_tensors
        if gy.shape[0] < _MIN_ROWS:
            return (gy @ _dequant(q, s_row, gy.dtype)).to(ctx.x_dtype), None, None, None
        gq, sg = _rowwise_int8(gy.float() * s_row)
        # mat2 [out, in] must be column-major for the int8 GEMM.
        acc = torch._int_mm(gq, q.t().contiguous().t())
        return (acc.float() * sg).to(ctx.x_dtype), None, None, None


class FrozenInt8Linear(nn.Module):
    """A frozen nn.Linear (a PEFT LoRA base_layer) as W8A8 int8; in / out features must be multiples of 8."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        out_f, in_f = linear.weight.shape
        if in_f % 8 or out_f % 8:
            raise ValueError(f"int8 W8A8 needs 8-aligned dims, got {out_f}x{in_f}")
        self.in_features, self.out_features = in_f, out_f
        q, s = quantize_weight(linear.weight.detach())
        self.register_buffer("weight", q)
        self.register_buffer("weight_scale", s.reshape(1, out_f).contiguous())
        self.bias = (
            None if linear.bias is None else nn.Parameter(linear.bias.detach(), requires_grad = False)
        )

    def forward(self, x):
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        shape = x.shape
        y = _FrozenInt8MM.apply(x.reshape(-1, shape[-1]), self.weight, self.weight_scale, x.dtype)
        y = y.reshape(*shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, int8 w8a8"


def convert_frozen_int8(
    model: nn.Module,
    names,
    device = None,
) -> tuple[int, int]:
    """Swap each frozen nn.Linear in ``names`` for FrozenInt8Linear one at a time; with ``device`` a host Linear
    moves there just before it is quantized, so the bf16 base is never whole on the device. A layer that fails
    stays as it was. Returns (converted, failed)."""
    done = failed = 0
    for fqn in sorted(names):
        parent_name, _, child = fqn.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child)
        if type(old) is not nn.Linear or old.weight.requires_grad:
            continue
        if device is not None and old.weight.device.type == "cpu":
            old.to(device)
        try:
            new = FrozenInt8Linear(old)
        except (RuntimeError, ValueError):
            failed += 1
            continue
        new._studio_lowp = "int8_w8a8"
        setattr(parent, child, new)
        done += 1
    return done, failed

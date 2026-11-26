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
modulation at batch 1) run the dequantized weight in the input dtype instead.

That column-major weight is kept by ``BACKWARD_WEIGHTS`` while it fits the trainer's budget, as the fp8 base does
(``Fp8BackwardWeights``), but keyed by the owning FrozenInt8Linear: an entry is served only while that module's
weight is still the tensor it was copied from, and a replaced weight rebuilds the module's one entry. int32
accumulation is exact, so grad_x is bit-identical either way; ``UNSLOTH_DIFFUSION_INT8_BWD_CACHE=0`` keeps the
inline copy.

No ``from __future__ import annotations`` here: torch 2.4's custom_op schema inference rejects the postponed
'torch.Tensor' strings, which silently left the backward cache off."""

import sys
import weakref

import torch
import torch.nn as nn
from torch.utils.weak import WeakIdKeyDictionary

from core.training.diffusion_frozen_fp8 import Fp8BackwardWeights, _column_major

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


# weight tensor -> its FrozenInt8Linear (weakly, both ways), so the backward can find the owner from the saved weight.
_OWNERS = WeakIdKeyDictionary()


class Int8BackwardWeights(Fp8BackwardWeights):
    """Column-major int8 weights for grad_x within ``budget`` bytes, one entry per FrozenInt8Linear."""

    ENV = "UNSLOTH_DIFFUSION_INT8_BWD_CACHE"
    STAT_PREFIX = "int8_bwd"

    def get(self, q):
        ref = _OWNERS.get(q)
        owner = ref() if ref is not None else None
        if owner is None:  # not a FrozenInt8Linear weight: nothing to key on
            self.misses += 1
            return _column_major(q)
        key = id(owner)
        hit = self.entries.get(key)
        if hit is not None:
            owner_ref, src_ref, ptr, version, qcm = hit
            if (
                owner_ref() is owner
                and src_ref() is q
                and q.data_ptr() == ptr
                and q._version == version
            ):
                self.hits += 1
                return qcm
            # The module's weight changed (moved, reloaded, written in place): drop its stale copy.
            del self.entries[key]
            self.used -= qcm.numel() * qcm.element_size()
        self.misses += 1
        qcm = _column_major(q)
        nbytes = qcm.numel() * qcm.element_size()
        if self.used + nbytes <= self.budget:
            self.entries[key] = (weakref.ref(owner), weakref.ref(q), q.data_ptr(), q._version, qcm)
            self.used += nbytes
        return qcm


BACKWARD_WEIGHTS = Int8BackwardWeights()


def _define_grad_input_op():
    # Opaque to the compiler, as the fp8 one: the lookup runs on every backward and a cache change never recompiles.
    try:
        return getattr(torch.ops.unsloth_studio, "frozen_int8_grad_input")
    except (AttributeError, RuntimeError):
        pass

    module = __name__

    @torch.library.custom_op("unsloth_studio::frozen_int8_grad_input", mutates_args = ())
    def grad_input(gq: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch._int_mm(gq, sys.modules[module].BACKWARD_WEIGHTS.get(q))

    @grad_input.register_fake
    def _(gq, q):
        return gq.new_empty((gq.shape[0], q.shape[1]), dtype = torch.int32)

    return grad_input


try:
    _grad_input_mm = _define_grad_input_op()
except Exception:  # noqa: BLE001 -- a torch without torch.library.custom_op keeps the inline copy
    _grad_input_mm = None


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
        if BACKWARD_WEIGHTS.active and _grad_input_mm is not None:
            acc = _grad_input_mm(gq, q)
        else:
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
        self._own_weight()

    def _own_weight(self):
        w = self._buffers.get("weight")
        if w is not None:
            _OWNERS[w] = weakref.ref(self)

    def _apply(
        self,
        fn,
        recurse = True,
    ):
        # .to() / .cuda() may swap in a new weight tensor: register it, or the backward could not find its entry.
        out = super()._apply(fn, recurse)
        self._own_weight()
        return out

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
        except torch.cuda.OutOfMemoryError:
            # Not a per-layer skip: the caller must see a base that ran out of memory partway.
            raise
        except (RuntimeError, ValueError):
            failed += 1
            continue
        new._studio_lowp = "int8_w8a8"
        if getattr(old, "_unsloth_fp32_island", False):
            # The fp16 path's fp32 island lives on the replaced module's forward: re-install it on the new one, or
            # an untargeted overflow-prone layer (Z-Image feed_forward.w2) would run under fp16 autocast again.
            from core.training.diffusion_dit_fp16 import make_fp32_island
            make_fp32_island(new, new.weight.device.type)
        setattr(parent, child, new)
        done += 1
    return done, failed

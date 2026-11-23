# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frozen base linears in fp8 e4m3, quantized once; backward returns only grad_x, so x is not saved.

grad_x needs the weight column-major (fp8 GEMMs take the reduction dim contiguous on both operands), which the
row-major forward copy is not. ``Fp8BackwardWeights`` keeps those column-major copies while they fit its budget
(planned from the room the checkpointing budget leaves after step 1); without room each backward makes it again, as
before (with a tiled transpose). Same tensor either way, so grad_x is bit-identical. Only a trainer that plans the
budget turns it on (``active``); ``UNSLOTH_DIFFUSION_FP8_BWD_CACHE=0`` keeps the inline copy."""

import os
import sys

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


_TRANSPOSE_KERNEL: list = []  # [kernel] once built, [None] when Triton is unavailable


def _transpose_kernel():
    if not _TRANSPOSE_KERNEL:
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def transpose(src, dst, M, N, BM: tl.constexpr, BN: tl.constexpr):
                # Tiled, so both the read of [M, N] and the write of [N, M] are coalesced.
                rm = tl.program_id(0) * BM + tl.arange(0, BM)
                rn = tl.program_id(1) * BN + tl.arange(0, BN)
                mask = (rm[:, None] < M) & (rn[None, :] < N)
                tile = tl.load(src + rm[:, None].to(tl.int64) * N + rn[None, :], mask = mask)
                tl.store(dst + rn[None, :].to(tl.int64) * M + rm[:, None], tile, mask = mask)

            from core.inference.triton_closure_globals import publish

            _TRANSPOSE_KERNEL.append(publish(transpose))
        except Exception:  # noqa: BLE001 -- no Triton: aten's copy below
            _TRANSPOSE_KERNEL.append(None)
    return _TRANSPOSE_KERNEL[0]


def _column_major(q):
    """``q`` [out, in] row-major as the same values column-major. aten's strided copy is several times slower than
    a tiled transpose for 1-byte elements (the compiled graph used to get inductor's tiled one)."""
    if (
        q.is_cuda
        and q.dim() == 2
        and min(q.shape) > 1
        and q.is_contiguous()
        and q.element_size() == 1
    ):
        kernel = _transpose_kernel()
        if kernel is not None:
            try:
                m, n = q.shape
                dst = torch.empty((n, m), dtype = torch.uint8, device = q.device)
                grid = ((m + 63) // 64, (n + 63) // 64)
                kernel[grid](q.view(torch.uint8), dst, m, n, BM = 64, BN = 64)
                return dst.view(q.dtype).t()
            except Exception:  # noqa: BLE001 -- e.g. a Triton that cannot build here: aten's copy
                _TRANSPOSE_KERNEL[0] = None
    return q.t().contiguous().t()


class Fp8BackwardWeights:
    """Column-major copies of frozen fp8 weights for grad_x, kept while they fit in ``budget`` bytes."""

    def __init__(self):
        # Set by the trainer before its first step: the backward then looks the copy up (traced in on step 1, so
        # this is read once per compiled graph); off, it copies inline as before.
        self.active = False
        self.budget = 0
        self.used = 0
        # data_ptr -> (source weight, column-major copy); the entry holds the source, so its storage (and the key)
        # cannot be freed and reused by another tensor while the entry lives.
        self.entries: dict = {}
        self.hits = 0
        self.misses = 0

    ENV = "UNSLOTH_DIFFUSION_FP8_BWD_CACHE"
    STAT_PREFIX = "fp8_bwd"

    @classmethod
    def enabled(cls) -> bool:
        return (os.environ.get(cls.ENV, "") or "1").strip().lower() not in (
            "0",
            "false",
            "off",
        )

    def plan(self, nbytes: int) -> None:
        self.budget = max(0, int(nbytes)) if self.enabled() else 0
        if self.used > self.budget:
            self.clear(keep_budget = True)

    def clear(self, keep_budget: bool = False) -> None:
        self.entries.clear()
        self.used = 0
        if not keep_budget:
            self.budget = 0

    def get(self, q):
        key = q.data_ptr()
        hit = self.entries.get(key)
        if hit is not None and hit[0].shape == q.shape and hit[0].dtype == q.dtype:
            self.hits += 1
            return hit[1]
        self.misses += 1
        qcm = _column_major(q)
        nbytes = qcm.numel() * qcm.element_size()
        if hit is None and self.used + nbytes <= self.budget:
            self.entries[key] = (q, qcm)
            self.used += nbytes
        return qcm

    def stats(self) -> dict:
        p = self.STAT_PREFIX
        return {
            f"{p}_cached_gib": round(self.used / (1 << 30), 3),
            f"{p}_budget_gib": round(self.budget / (1 << 30), 3),
            f"{p}_layers": len(self.entries),
            f"{p}_hits": self.hits,
            f"{p}_misses": self.misses,
        }


BACKWARD_WEIGHTS = Fp8BackwardWeights()


def fp8_backward_vram_handler(cache: Fp8BackwardWeights = BACKWARD_WEIGHTS, room = None):
    """lend_vram: give the copies back before another Studio load, keep making them per backward until returned.
    ``room()`` caps the returned budget (a step-down or the new load since the lend leaves less)."""
    lent = {"budget": 0}

    def handle(kind: str) -> dict:
        freed = 0
        if kind == "lend":
            freed = cache.used
            lent["budget"] = max(lent["budget"], cache.budget)
            cache.clear()
            if freed and torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif kind == "return":
            budget = lent["budget"]
            if room is not None:
                try:
                    budget = min(budget, int(room()))
                except Exception:  # noqa: BLE001 -- unknown room: cache nothing
                    budget = 0
            cache.plan(budget)
            lent["budget"] = 0
        return {"kind": kind, "freed_gib": round(freed / (1 << 30), 3), **cache.stats()}

    return handle


def _define_grad_input_op(op = "frozen_fp8_grad_input", cached = True):
    # Opaque to the compiler, so the cache lookup runs on every backward and a cache change never recompiles.
    name = f"unsloth_studio::{op}"
    try:
        return getattr(torch.ops.unsloth_studio, op)
    except (AttributeError, RuntimeError):
        pass

    module = __name__

    @torch.library.custom_op(name, mutates_args = ())
    def grad_input(gq: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # Looked up per call, so a reloaded module still serves its own cache.
        m = sys.modules[module]
        return m._unit_scaled_mm(gq, m.BACKWARD_WEIGHTS.get(q) if cached else m._column_major(q))

    @grad_input.register_fake
    def _(gq, q):
        return gq.new_empty((gq.shape[0], q.shape[1]), dtype = torch.bfloat16)

    return grad_input


try:
    _grad_input_mm = _define_grad_input_op()
    # A block-swapped weight lives in a ring slot that the next block of the same shape reuses, so its data_ptr is
    # no identity: the cache would hand back another block's copy. Same tiled transpose, never cached.
    _grad_input_mm_uncached = _define_grad_input_op("frozen_fp8_grad_input_uncached", cached = False)
except Exception:  # noqa: BLE001 -- a torch without torch.library.custom_op keeps the inline copy
    _grad_input_mm = _grad_input_mm_uncached = None


@torch._dynamo.allow_in_graph  # as torchao's float8 matmul: inlining the Function trips a dynamo freevar assert
class _FrozenFp8MM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x2d,
        q,
        s_row,
        out_dtype,
        swapped = False,
    ):
        xq, sx = _rowwise_quant(x2d)
        y = (_unit_scaled_mm(xq, q.t()).float() * (sx * s_row)).to(out_dtype)
        ctx.save_for_backward(q, s_row)
        ctx.x_dtype = x2d.dtype
        ctx.swapped = swapped
        return y

    @staticmethod
    def backward(ctx, gy):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        q, s_row = ctx.saved_tensors
        # _scaled_mm needs a row-major mat1; a transposed upstream op can hand back a column-major grad_y.
        gy = gy.contiguous()
        # The weight scale sits on the reduction dim of grad_y @ W, so it folds into grad_y; mat2 must be column-major.
        gq, sg = _rowwise_quant(gy.float() * s_row)
        if BACKWARD_WEIGHTS.active and _grad_input_mm is not None:
            op = _grad_input_mm_uncached if ctx.swapped else _grad_input_mm
            gx = op(gq, q).float() * sg
        else:
            gx = _unit_scaled_mm(gq, q.t().contiguous().t()).float() * sg
        return gx.to(ctx.x_dtype), None, None, None, None


class FrozenFp8Linear(nn.Module):
    """A frozen nn.Linear (a PEFT LoRA base_layer) with an fp8 weight; in / out features must be 16-aligned."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        out_f, in_f = linear.weight.shape
        if in_f % 16 or out_f % 16:
            raise ValueError(f"fp8 needs 16-aligned dims, got {out_f}x{in_f}")
        self.in_features, self.out_features = in_f, out_f
        # A Linear loaded from a hosted fp8 checkpoint carries its per-row codes already (diffusion_fp8_prequant).
        pre = getattr(linear, "_studio_fp8_prequant", None)
        q, s = pre if pre is not None else _quantize_weight(linear.weight.detach())
        self.register_buffer("weight", q)
        self.register_buffer("weight_scale", s.reshape(1, out_f).contiguous())
        self.bias = (
            None if linear.bias is None else nn.Parameter(linear.bias.detach(), requires_grad = False)
        )

    def forward(self, x):
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        shape = x.shape
        y = _FrozenFp8MM.apply(
            # Row-major for _scaled_mm's mat1; the codes keep the input's layout.
            x.reshape(-1, shape[-1]).contiguous(),
            self.weight,
            self.weight_scale,
            x.dtype,
            # Set by diffusion_block_swap when the weight streams through a ring slot (data_ptr is not identity).
            bool(getattr(self, "_studio_swapped_weight", False)),
        )
        y = y.reshape(*shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8"


def convert_frozen_fp8(
    model: nn.Module,
    names,
    device = None,
) -> tuple[int, int]:
    """Swap each frozen nn.Linear in ``names`` for FrozenFp8Linear one at a time (bounds peak memory); a layer that
    fails stays bf16 rather than raising mid-conversion. With ``device``, a host Linear moves there first, so a
    host-resident base lands as fp8 without ever being whole in bf16 on the device. Returns (converted, failed)."""
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
            new = FrozenFp8Linear(old)
        except torch.cuda.OutOfMemoryError:
            # Not a per-layer skip: the caller must see a base that ran out of memory partway.
            raise
        except (RuntimeError, ValueError):
            failed += 1
            continue
        new._studio_lowp = "fp8"
        setattr(parent, child, new)
        done += 1
    return done, failed

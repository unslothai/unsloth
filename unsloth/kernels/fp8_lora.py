# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""FP8 frozen-base GEMMs with 16-bit LoRA adapters and an exact master for saving and merging.

cuBLAS needs a K-major dgrad operand; transpose inside the opaque op, or cache it with
UNSLOTH_FP8_LORA_CACHE_T=1. Offloaded masters remain the same Parameter on the host;
decode reconstructs the FP8 weight, while save and merge restore the exact master.
"""

from __future__ import annotations

import contextlib
import functools
import os
from typing import Optional

import torch

__all__ = [
    "FP8_LORA_STATS",
    "Fp8LoRABaseState",
    "fp8_lora_fwd",
    "fp8_lora_dgrad",
    "quantize_tensorwise",
    "quantize_rowwise",
    "attach_fp8_lora_base",
    "detach_fp8_lora_base",
    "fp8_lora_memory",
    "fp8_transpose",
    "restore_fp8_lora_master",
    "has_fp8_dgrad",
    "reset_fp8_lora_stats",
]

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
_FP8_MAX = {E4M3: 448.0, E5M2: 57344.0}
FP8_LORA_STATS = {"fwd": 0, "dgrad": 0, "dgrad_16bit": 0, "fwd_16bit": 0, "quantize": 0}


def reset_fp8_lora_stats():
    for key in FP8_LORA_STATS:
        FP8_LORA_STATS[key] = 0


_SCALING = os.environ.get("UNSLOTH_FP8_LORA_SCALING", "auto").strip().lower()
_SCALING = {"rowwise": "row", "tensorwise": "tensor"}.get(_SCALING, _SCALING)
if _SCALING not in ("auto", "row", "tensor"):
    _SCALING = "auto"


def scaling_for_class(gpu_class) -> str:
    """'row' or 'tensor' for a precision-registry GPU class ('sm89', 'sm120', ...); the env var wins."""
    if _SCALING != "auto":
        return _SCALING
    return "row" if gpu_class == "sm120" else "tensor"


def _scaling_for(device):
    gpu_class = None
    try:
        if device.type == "cuda" and torch.version.hip is None:
            gpu_class = "sm120" if torch.cuda.get_device_capability(device)[0] >= 12 else "other"
    except Exception:
        pass
    return scaling_for_class(gpu_class)


def _env_dgrad():
    value = os.environ.get("UNSLOTH_FP8_LORA_DGRAD", "auto").strip().lower()
    if value in ("0", "false", "off", "no", "bf16", "16bit"):
        return None
    return {"e4m3": E4M3, "e5m2": E5M2}.get(value, "auto")


_DGRAD_DTYPE = _env_dgrad()
_FWD_FAST_ACCUM = os.environ.get("UNSLOTH_FP8_LORA_FWD_FAST_ACCUM", "1") != "0"
_DGRAD_FAST_ACCUM = os.environ.get("UNSLOTH_FP8_LORA_DGRAD_FAST_ACCUM", "0") == "1"
_CACHE_T = os.environ.get("UNSLOTH_FP8_LORA_CACHE_T", "0") == "1"
_OFFLOAD_MASTER = os.environ.get("UNSLOTH_FP8_LORA_OFFLOAD_MASTER", "1") != "0"
_OFFLOAD_CPU_WEIGHTS = False
_MIN_ROWS = int(os.environ.get("UNSLOTH_FP8_LORA_MIN_ROWS", "16"))

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
_fast_launch = None
if triton is not None:
    try:
        from .triton_launch import launch as _fast_launch, tag_compile_cache
    except ImportError:
        pass
    else:
        tag_compile_cache(__file__)


def _launch(kernel, grid, args, n_tensors, constexprs, device_index, **options):
    if _fast_launch is not None and not _compiling():
        return _fast_launch(kernel, grid, args, n_tensors, constexprs, device_index, **options)
    kernel[grid](*args, **constexprs, **options)


# Read once: device_count() inside a compiled region graph-breaks.
_MULTI_GPU = torch.cuda.is_available() and torch.cuda.device_count() > 1


if triton is not None:

    @triton.jit
    def _cast_fp8_rn(x, IS_E5M2: tl.constexpr):
        # Triton 3.8 rounds through fp16 on sm89; native f32 conversion avoids double rounding.
        if IS_E5M2:
            packed = tl.inline_asm_elementwise(
                "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $1;",
                constraints = "=h,f",
                args = [x],
                dtype = tl.uint16,
                is_pure = True,
                pack = 1,
            )
            return packed.to(tl.uint8).to(tl.float8e5, bitcast = True)
        else:
            packed = tl.inline_asm_elementwise(
                "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $1;",
                constraints = "=h,f",
                args = [x],
                dtype = tl.uint16,
                is_pure = True,
                pack = 1,
            )
            return packed.to(tl.uint8).to(tl.float8e4nv, bitcast = True)

    @triton.jit
    def _fp8_tensorwise_cast_kernel(
        x_ptr,
        y_ptr,
        amax_ptr,
        inv_ptr,
        numel,
        FP8_MAX: tl.constexpr,
        IS_E5M2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        amax = tl.load(amax_ptr).to(tl.float32)
        # NaN / inf amax propagates into the output, so a GradScaler still sees the overflow.
        amax = tl.maximum(amax, 1e-12, propagate_nan = tl.PropagateNan.ALL)
        # Match PyTorch's reciprocal-then-multiply rounding order (_tensor.py::__rdiv__).
        scale = tl.div_rn(1.0, amax) * FP8_MAX
        x = tl.load(x_ptr + offs, mask = mask, other = 0.0).to(tl.float32)
        y = tl.maximum(x * scale, -FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
        y = tl.minimum(y, FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
        tl.store(y_ptr + offs, _cast_fp8_rn(y, IS_E5M2), mask = mask)
        if pid == 0:
            tl.store(inv_ptr, amax * (1.0 / FP8_MAX))


def quantize_tensorwise(x: torch.Tensor, fp8_dtype = E4M3):
    """x -> (fp8 x with x's shape, contiguous; fp32 0-d dequant scale amax / fp8_max)."""
    fp8_max = _FP8_MAX[fp8_dtype]
    x = x.contiguous()
    amax = torch.linalg.vector_norm(x, float("inf"), dtype = torch.float32)
    if triton is not None and x.is_cuda and x.numel() > 0:
        y = torch.empty(x.shape, dtype = fp8_dtype, device = x.device)
        inv = torch.empty((), dtype = torch.float32, device = x.device)
        BLOCK = 2048
        grid = (triton.cdiv(x.numel(), BLOCK),)
        with torch.cuda.device(x.device) if _MULTI_GPU else contextlib.nullcontext():
            _launch(
                _fp8_tensorwise_cast_kernel,
                grid,
                (x, y, amax, inv, x.numel()),
                4,
                {"FP8_MAX": fp8_max, "IS_E5M2": fp8_dtype is E5M2, "BLOCK": BLOCK},
                x.device.index,
                num_warps = 4,
            )
        return y, inv
    amax = amax.clamp(min = 1e-12) if torch.isfinite(amax) else amax
    y = (x.float() * (fp8_max / amax)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return y, (amax / fp8_max).reshape(())


if triton is not None:

    @triton.jit
    def _fp8_rowwise_cast_kernel(
        x_ptr,
        y_ptr,
        inv_ptr,
        K,
        stride_xm,
        FP8_MAX: tl.constexpr,
        IS_E5M2: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ONE_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * stride_xm
        y_row = y_ptr + row * K
        offs = tl.arange(0, BLOCK_K)
        if ONE_BLOCK:
            x = tl.load(x_row + offs, mask = offs < K, other = 0.0).to(tl.float32)
            amax = tl.max(tl.abs(x), axis = 0)
            nans = tl.sum((x != x).to(tl.int32), axis = 0)
        else:
            acc = tl.zeros((BLOCK_K,), dtype = tl.float32)
            cnt = tl.zeros((BLOCK_K,), dtype = tl.int32)
            for k in range(0, K, BLOCK_K):
                xk = tl.load(x_row + k + offs, mask = k + offs < K, other = 0.0).to(tl.float32)
                acc = tl.maximum(acc, tl.abs(xk))
                cnt += (xk != xk).to(tl.int32)
            amax = tl.max(acc, axis = 0)
            nans = tl.sum(cnt, axis = 0)
        # tl.max drops NaN: put it back so a GradScaler still sees the overflow.
        amax = tl.where(nans > 0, float("nan"), amax)
        amax = tl.maximum(amax, 1e-12, propagate_nan = tl.PropagateNan.ALL)
        scale = tl.div_rn(1.0, amax) * FP8_MAX
        if ONE_BLOCK:
            y = tl.maximum(x * scale, -FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
            y = tl.minimum(y, FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
            tl.store(y_row + offs, _cast_fp8_rn(y, IS_E5M2), mask = offs < K)
        else:
            for k in range(0, K, BLOCK_K):
                mask = k + offs < K
                xk = tl.load(x_row + k + offs, mask = mask, other = 0.0).to(tl.float32)
                y = tl.maximum(xk * scale, -FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
                y = tl.minimum(y, FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
                tl.store(y_row + k + offs, _cast_fp8_rn(y, IS_E5M2), mask = mask)
        tl.store(inv_ptr + row, amax * (1.0 / FP8_MAX))


def quantize_rowwise(x: torch.Tensor, fp8_dtype = E4M3):
    """x [M, K] -> (fp8 x [M, K] contiguous; fp32 [M, 1] dequant scales, one per row)."""
    fp8_max = _FP8_MAX[fp8_dtype]
    if x.stride(-1) != 1:
        x = x.contiguous()
    M, K = x.shape
    if triton is not None and x.is_cuda and x.numel() > 0:
        y = torch.empty((M, K), dtype = fp8_dtype, device = x.device)
        inv = torch.empty((M, 1), dtype = torch.float32, device = x.device)
        one = K <= 8192
        BLOCK_K = triton.next_power_of_2(K) if one else 2048
        with torch.cuda.device(x.device) if _MULTI_GPU else contextlib.nullcontext():
            _launch(
                _fp8_rowwise_cast_kernel,
                (M,),
                (x, y, inv, K, x.stride(0)),
                3,
                {
                    "FP8_MAX": fp8_max,
                    "IS_E5M2": fp8_dtype is E5M2,
                    "BLOCK_K": BLOCK_K,
                    "ONE_BLOCK": one,
                },
                x.device.index,
                num_warps = 8 if BLOCK_K >= 4096 else 4,
            )
        return y, inv
    xf = x.float()
    amax = xf.abs().amax(dim = 1, keepdim = True)
    amax = torch.where(torch.isnan(xf).any(1, keepdim = True), torch.nan, amax).clamp(min = 1e-12)
    y = (xf * (fp8_max / amax)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return y, amax / fp8_max


if triton is not None:

    @triton.jit
    def _transpose_u8_kernel(x_ptr, y_ptr, R, C, BLOCK: tl.constexpr):
        # Square tiles on purpose: non-square 2D fp8 transposed stores write corrupt bytes on sm89 (L4, triton 3.x).
        rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        cols = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = (rows[:, None] < R) & (cols[None, :] < C)
        tile = tl.load(x_ptr + rows[:, None] * C + cols[None, :], mask = mask)
        tl.store(y_ptr + cols[None, :] * R + rows[:, None], tile, mask = mask)


_TRANSPOSE_BLOCK = 64


def fp8_transpose(w8: torch.Tensor, use_triton: Optional[bool] = None) -> torch.Tensor:
    """w8 [R, C] (contiguous, 1 byte per element) -> contiguous [C, R], byte for byte."""
    if use_triton is None:
        use_triton = triton is not None and w8.is_cuda
    if not use_triton or w8.dim() != 2 or w8.numel() == 0:
        return w8.t().contiguous()
    w8 = w8.contiguous()
    R, C = w8.shape
    out = torch.empty((C, R), dtype = w8.dtype, device = w8.device)
    grid = (triton.cdiv(R, _TRANSPOSE_BLOCK), triton.cdiv(C, _TRANSPOSE_BLOCK))
    with torch.cuda.device(w8.device) if (_MULTI_GPU and w8.is_cuda) else contextlib.nullcontext():
        _launch(
            _transpose_u8_kernel,
            grid,
            (w8.view(torch.uint8), out.view(torch.uint8), R, C),
            2,
            {"BLOCK": _TRANSPOSE_BLOCK},
            w8.device.index,
            num_warps = 4,
        )
    return out


def _out_dtype(dtype):
    return torch.bfloat16 if dtype == torch.bfloat16 else torch.float32


_ROWWISE_BF16_ONLY = set()


def _rowwise_ok(dtype, device) -> bool:
    return dtype != torch.float16 or device not in _ROWWISE_BF16_ONLY


def _rowwise_scaled_mm(a, b, scale_a, scale_b, dtype, fast_accum):
    """None for fp16 when this device's rowwise GEMM refuses float32 out: bf16 would round 65504 to an inf."""
    out_dtype = _out_dtype(dtype)
    if out_dtype == torch.float32 and a.device not in _ROWWISE_BF16_ONLY:
        try:
            return _scaled_mm(a, b, scale_a, scale_b, out_dtype, fast_accum)
        except (RuntimeError, ValueError) as exc:
            if str(exc) not in (
                "Only bf16 high precision output types are supported for row-wise scaling.",
                "Only bf16 and fp16 high precision output types are supported for row-wise scaling.",
            ):
                raise
            _ROWWISE_BF16_ONLY.add(a.device)
    if dtype == torch.float16:
        return None
    return _scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, fast_accum)


def _weight_scale(w_inv):
    return w_inv.reshape(-1)[0] if w_inv.dim() == 2 else w_inv


def _scaled_mm(a, b, scale_a, scale_b, out_dtype, fast_accum):
    out = torch._scaled_mm(
        a,
        b,
        scale_a = scale_a,
        scale_b = scale_b,
        out_dtype = out_dtype,
        use_fast_accum = fast_accum,
    )
    return out[0] if isinstance(out, tuple) else out


# Module level: a torch.library.Library deregisters its ops once collected.
_library = None


def _opaque_under_compile(name, fake):
    def decorator(fn):
        global _library
        substitute_in_graph = getattr(torch._dynamo, "substitute_in_graph", None)
        if substitute_in_graph is None or not hasattr(torch.library, "infer_schema"):
            return fn
        try:
            if _library is None:
                _library = torch.library.Library("unsloth", "FRAGMENT")
            _library.define(name + torch.library.infer_schema(fn, mutates_args = ()))
            _library.impl(name, fn, "CompositeExplicitAutograd")
            torch.library.register_fake(f"unsloth::{name}", fake, lib = _library)
            substitute_in_graph(fn)(
                functools.wraps(fn)(
                    lambda *args, **kwargs: getattr(torch.ops.unsloth, name)(*args, **kwargs)
                )
            )
        except (RuntimeError, TypeError, ValueError):  # registered by an earlier import
            pass
        return fn

    return decorator


def _fwd_fake(x, w8, w_inv):
    return x.new_empty((x.shape[0], w8.shape[0]))


@_opaque_under_compile("fp8_lora_fwd", _fwd_fake)
def fp8_lora_fwd(x: torch.Tensor, w8: torch.Tensor, w_inv: torch.Tensor) -> torch.Tensor:
    """x [M, K] (16-bit) @ dequant(w8 [N, K]).T -> [M, N] in x.dtype. w_inv: 0-d (tensorwise activations) or
    [1, N] (rowwise activations: the weight's tensorwise scale per output column)."""
    FP8_LORA_STATS["fwd"] += 1
    out = None
    if w_inv.dim() == 2 and _rowwise_ok(x.dtype, x.device):
        x8, x_inv = quantize_rowwise(x, E4M3)
        out = _rowwise_scaled_mm(x8, w8.t(), x_inv, w_inv, x.dtype, _FWD_FAST_ACCUM)
    if out is None:
        x8, x_inv = quantize_tensorwise(x, E4M3)
        out = _scaled_mm(
            x8, w8.t(), x_inv, _weight_scale(w_inv), _out_dtype(x.dtype), _FWD_FAST_ACCUM
        )
    return out if out.dtype == x.dtype else out.to(x.dtype)


def _dgrad_fake(dy, w8, w_inv, w_is_t):
    return dy.new_empty((dy.shape[0], w8.shape[0] if w_is_t else w8.shape[1]))


@_opaque_under_compile("fp8_lora_dgrad", _dgrad_fake)
def fp8_lora_dgrad(
    dy: torch.Tensor, w8: torch.Tensor, w_inv: torch.Tensor, w_is_t: bool
) -> torch.Tensor:
    """dy [M, N] (16-bit) @ dequant(W [N, K]) -> dx [M, K] in dy.dtype. w8 is the cached K-major copy [K, N]
    (w_is_t) or the forward copy [N, K], transposed here by the square-tile kernel inside the op, never by an
    Inductor-generated fp8 transpose (those write corrupt bytes for non-square tile configs on sm89)."""
    FP8_LORA_STATS["dgrad"] += 1
    w8t = w8 if w_is_t else fp8_transpose(w8)
    out = None
    if w_inv.dim() == 2 and _rowwise_ok(dy.dtype, dy.device):
        dy8, dy_inv = quantize_rowwise(dy, E4M3)
        out = _rowwise_scaled_mm(dy8, w8t.t(), dy_inv, w_inv, dy.dtype, _DGRAD_FAST_ACCUM)
    if out is None:
        dy8, dy_inv = quantize_tensorwise(
            dy, E5M2 if _DGRAD_DTYPE in (None, "auto") else _DGRAD_DTYPE
        )
        out = _scaled_mm(
            dy8, w8t.t(), dy_inv, _weight_scale(w_inv), _out_dtype(dy.dtype), _DGRAD_FAST_ACCUM
        )
    return out if out.dtype == dy.dtype else out.to(dy.dtype)


if triton is not None:

    @triton.jit(do_not_specialize = ["N", "numel", "raw_s0", "raw_s1", "delta_s0", "delta_s1"])
    def _fp8_accum_epilogue_kernel(
        raw_ptr,
        x_inv_ptr,
        w_inv_ptr,
        delta_ptr,
        out_ptr,
        N,
        numel,
        raw_s0,
        raw_s1,
        delta_s0,
        delta_s1,
        ROW_SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        row, col = offset // N, offset % N
        mask = offset < numel
        raw = tl.load(raw_ptr + row * raw_s0 + col * raw_s1, mask, other = 0)
        sx = tl.load(x_inv_ptr + tl.where(ROW_SCALE, row, 0), mask, other = 0)
        sw = tl.load(w_inv_ptr + tl.where(ROW_SCALE, col, 0), mask, other = 0)
        delta = tl.load(delta_ptr + row * delta_s0 + col * delta_s1, mask, other = 0).to(tl.float32)
        out = tl.inline_asm_elementwise(
            "{ .reg .f32 v; mul.rn.f32 v, $1, $2; mul.rn.f32 v, v, $3; add.rn.f32 $0, v, $4; }",
            constraints = "=f,f,f,f,f",
            args = (raw, sx, sw, delta),
            dtype = tl.float32,
            is_pure = True,
            pack = 1,
        )
        tl.store(out_ptr + offset, out, mask)


def _fp8_accum_epilogue(raw, x_inv, w_inv, delta):
    if triton is None or not raw.is_cuda or torch.version.hip is not None:
        return (raw * x_inv * w_inv + delta.float()).to(delta.dtype)
    out = torch.empty(raw.shape, device = raw.device, dtype = delta.dtype)
    with torch.cuda.device(raw.device) if _MULTI_GPU else contextlib.nullcontext():
        _launch(
            _fp8_accum_epilogue_kernel,
            ((raw.numel() + 1023) // 1024,),
            (
                raw,
                x_inv,
                w_inv,
                delta,
                out,
                raw.shape[1],
                raw.numel(),
                *raw.stride(),
                *delta.stride(),
            ),
            5,
            {"ROW_SCALE": w_inv.dim() == 2, "BLOCK": 1024},
            raw.device.index,
            num_warps = 4,
        )
    return out


def _accum_fake(x, w8, w_inv, delta, unit_scale, backward, transpose_weight):
    return delta.new_empty(delta.shape)


@_opaque_under_compile("fp8_lora_accum", _accum_fake)
def fp8_lora_accum(
    x: torch.Tensor,
    w8: torch.Tensor,
    w_inv: torch.Tensor,
    delta: torch.Tensor,
    unit_scale: torch.Tensor,
    backward: bool,
    transpose_weight: bool,
) -> torch.Tensor:
    FP8_LORA_STATS["dgrad" if backward else "fwd"] += 1
    if transpose_weight:
        w8 = fp8_transpose(w8)
    if w_inv.dim() == 2:
        x8, x_inv = quantize_rowwise(x, E4M3)
    else:
        dtype = (
            E5M2
            if backward and _DGRAD_DTYPE in (None, "auto")
            else (_DGRAD_DTYPE if backward else E4M3)
        )
        x8, x_inv = quantize_tensorwise(x, dtype)
    raw = _scaled_mm(
        x8,
        w8.t(),
        unit_scale,
        unit_scale,
        torch.float32,
        _DGRAD_FAST_ACCUM if backward else _FWD_FAST_ACCUM,
    )
    return _fp8_accum_epilogue(raw, x_inv, w_inv, delta)


def _first_fake(x, w8, w_inv, xa, bt, alpha, unit_scale, backward, transpose_weight):
    return x.new_empty((x.shape[0], bt.shape[1]))


@_opaque_under_compile("fp8_lora_first", _first_fake)
def fp8_lora_first(
    x: torch.Tensor,
    w8: torch.Tensor,
    w_inv: torch.Tensor,
    xa: torch.Tensor,
    bt: torch.Tensor,
    alpha: float,
    unit_scale: torch.Tensor,
    backward: bool,
    transpose_weight: bool,
) -> torch.Tensor:
    delta = x.new_empty((x.shape[0], bt.shape[1]))
    delta.addmm_(xa, bt, alpha = alpha, beta = 0)
    return fp8_lora_accum(x, w8, w_inv, delta, unit_scale, backward, transpose_weight)


def _is_transposed(W):
    return W.dim() == 2 and W.stride(0) == 1 and W.stride(1) != 1


def _compiling():
    try:
        return torch.compiler.is_compiling()
    except Exception:
        return False


class Fp8LoRABaseState:
    """Quant state of a frozen LoRA base in Unsloth's LoRA kernels (a packed weight state): ``W`` is the
    16-bit weight [out, in] (forward) or its ``.t()`` view (input gradient)."""

    _unsloth_packed_weight_state = True
    _unsloth_fp8_dgrad = True

    def __init__(self, weight: torch.Tensor):
        self.dtype = weight.dtype
        self.device = weight.device  # where the fp8 copy and the GEMMs live
        self.w8 = self.w8t = self.w_inv = self.w_inv_fwd = self.w_inv_dgrad = self.unit_scale = None
        self.scaling = "tensor"
        self.offloaded = False  # True: the module's 16-bit weight is the host copy
        self._key = None
        self.quantizations = 0
        self.refresh(weight)

    @staticmethod
    def _weight_key(weight):
        return (
            weight.data_ptr(),
            weight._version,
            weight.device,
            weight.dtype,
            tuple(weight.shape),
        )

    def refresh(self, weight: torch.Tensor):
        """Quantize `weight` unless the cache already holds it (same storage, version, device)."""
        if _compiling():
            return self
        key = self._weight_key(weight)
        if key == self._key:
            return self
        with torch.inference_mode(False), torch.no_grad():
            w = weight.detach()
            if self.offloaded:
                w = w.to(self.device)
            else:
                self.device = w.device
            self.w8, self.w_inv = quantize_tensorwise(w, E4M3)
            self.unit_scale = torch.ones((), device = w.device, dtype = torch.float32)
            self.w8t = fp8_transpose(self.w8) if _CACHE_T else None
            self.scaling = _scaling_for(w.device)
            if self.scaling == "row":
                self.w_inv_fwd = self.w_inv.reshape(1, 1).expand(1, w.shape[0]).contiguous()
                self.w_inv_dgrad = self.w_inv.reshape(1, 1).expand(1, w.shape[1]).contiguous()
            else:
                self.w_inv_fwd = self.w_inv_dgrad = self.w_inv
        self.dtype = weight.dtype
        self._key = key
        self.quantizations += 1
        FP8_LORA_STATS["quantize"] += 1
        return self

    def _dgrad_weight(self):
        return (self.w8t, True) if self.w8t is not None else (self.w8, False)

    def offload(self, module):
        """Move module.weight (the 16-bit master) to host RAM, keeping the Parameter object."""
        weight = module.weight
        if self.offloaded or (weight.device.type == "cpu" and not _OFFLOAD_CPU_WEIGHTS):
            return False
        self.refresh(weight)
        host = None
        if weight.is_cuda:
            try:
                host = torch.empty(weight.shape, dtype = weight.dtype, device = "cpu", pin_memory = True)
            except (
                RuntimeError
            ):  # pinned host memory exhausted: pageable works, restores are slower
                host = None
        if host is None:
            host = torch.empty(weight.shape, dtype = weight.dtype, device = "cpu")
        host.copy_(weight.detach())
        weight.data = host
        self.offloaded = True
        self._key = self._weight_key(weight)
        return True

    def restore(self, module):
        if not self.offloaded:
            return False
        weight = module.weight
        with torch.inference_mode(False), torch.no_grad():
            weight.data = weight.data.to(self.device, copy = True)
        self.offloaded = False
        self._key = self._weight_key(weight)
        return True

    def _dense(self, W):
        """A 16-bit operand on the compute device for the non-fp8 fallbacks: W itself, or (master on the host)
        the dequantized fp8 weight in W's orientation, which is what the fp8 forward uses anyway."""
        if W is not None and not self.offloaded and W.device == self.device:
            return W
        dense = self.w8.to(torch.float32) * self.w_inv
        dense = dense.to(self.dtype if W is None else W.dtype)
        return dense.t() if W is not None and _is_transposed(W) else dense

    def resident_bytes(self, module = None):
        fp8 = sum(
            t.numel() * t.element_size()
            for t in {
                id(t): t
                for t in (
                    self.w8,
                    self.w8t,
                    self.w_inv,
                    self.w_inv_fwd,
                    self.w_inv_dgrad,
                    self.unit_scale,
                )
                if t is not None
            }.values()
        )
        out = {"fp8": fp8, "master_device": 0, "master_host": 0}
        if module is not None:
            w = module.weight
            nbytes = w.numel() * w.element_size()
            if self.offloaded:
                out["master_host"] = nbytes
            elif w.device == self.device:
                out["master_device"] = nbytes
        return out

    def dequantize(
        self,
        W = None,
        dtype = None,
    ):
        if W is not None and W.device != self.device:
            return W.to(self.device)
        return W

    def _use_fp8(self, X):
        return X.dtype in (torch.bfloat16, torch.float16, torch.float32) and (
            X.numel() // max(X.shape[-1], 1) >= _MIN_ROWS
        )

    def matmul(
        self,
        X,
        W,
        out = None,
    ):
        """``X @ W.t()``: the forward for the stored W, the input gradient for its transposed view."""
        shape = X.shape
        X2 = X.reshape(-1, shape[-1])
        if not self._use_fp8(X2):
            FP8_LORA_STATS["fwd_16bit"] += 1
            result = torch.matmul(X2, self._dense(W).t().to(X2.dtype))
        elif _is_transposed(W):
            result = self.dgrad(X2, W.t())
        else:
            result = fp8_lora_fwd(X2, self.w8, self.w_inv_fwd)
        result = result.view(*shape[:-1], result.shape[-1])
        return result if out is None else out.copy_(result)

    def lora_first(
        self,
        X,
        W,
        XA,
        Bt,
        alpha,
        out = None,
    ):
        backward = _is_transposed(W)
        if not self._use_fp8(X) or (backward and _DGRAD_DTYPE is None):
            delta = X.new_empty((X.shape[0], Bt.shape[1]))
            delta.addmm_(XA, Bt, alpha = alpha, beta = 0)
            result = delta.addmm_(X, self._dense(W).t().to(X.dtype))
        else:
            w8 = self._dgrad_weight()[0] if backward else self.w8
            inv = self.w_inv_dgrad if backward else self.w_inv_fwd
            result = fp8_lora_first(
                X, w8, inv, XA, Bt, alpha, self.unit_scale, backward, backward and self.w8t is None
            )
        return result if out is None else out.copy_(result)

    def dgrad_accum(self, dY, W, delta):
        if _DGRAD_DTYPE is None or not self._use_fp8(dY):
            return delta.addmm_(dY, self._dense(W).to(dY.dtype))
        w8, transposed = self._dgrad_weight()
        return fp8_lora_accum(
            dY, w8, self.w_inv_dgrad, delta, self.unit_scale, True, not transposed
        )

    def dgrad(
        self,
        dY,
        W = None,
    ):
        """dY [..., out] @ W -> [..., in]: the base layer's input gradient."""
        shape = dY.shape
        dY2 = dY.reshape(-1, shape[-1])
        if _DGRAD_DTYPE is None or not self._use_fp8(dY2):
            FP8_LORA_STATS["dgrad_16bit"] += 1
            if W is None and not self.offloaded:
                raise RuntimeError("Unsloth: 16-bit dgrad needs the base weight")
            result = torch.matmul(dY2, self._dense(W).to(dY2.dtype))
        else:
            w8, w_is_t = self._dgrad_weight()
            result = fp8_lora_dgrad(dY2, w8, self.w_inv_dgrad, w_is_t)
        return result.view(*shape[:-1], result.shape[-1])


def has_fp8_dgrad(W_quant) -> bool:
    return getattr(type(W_quant), "_unsloth_fp8_dgrad", False)


class _Fp8FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, w8, w8t, w_inv_fwd, w_inv_dgrad):
        ctx.w_is_t = w8t is not None
        ctx.save_for_backward(
            weight,
            w8t if w8t is not None else w8,
            w_inv_fwd if w_inv_dgrad is None else w_inv_dgrad,
        )
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = fp8_lora_fwd(x2, w8, w_inv_fwd)
        return y.view(*shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        weight, w8t, w_inv = ctx.saved_tensors
        shape = dy.shape
        dy2 = dy.reshape(-1, shape[-1])
        if _DGRAD_DTYPE is None:
            FP8_LORA_STATS["dgrad_16bit"] += 1
            dx = torch.matmul(dy2, weight.to(dy2.dtype))
        else:
            dx = fp8_lora_dgrad(dy2, w8t, w_inv, ctx.w_is_t)
        return dx.view(*shape[:-1], dx.shape[-1]), None, None, None, None, None


def _fp8_base_forward(self, x):
    state = self.__dict__.get("_unsloth_fp8_lora")
    weight = self.weight
    if state is None or weight.requires_grad or not state._use_fp8(x.reshape(-1, x.shape[-1])):
        FP8_LORA_STATS["fwd_16bit"] += 1
        if state is not None and state.offloaded:
            if weight.requires_grad:  # unfrozen: train the exact weight
                state.restore(self)
            else:
                weight = state._dense(weight)
        return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)
    state.refresh(weight)
    # Tensorwise: one scale serves both GEMMs (Dynamo rejects the same tensor passed twice to apply).
    dgrad_scale = None if state.w_inv_dgrad is state.w_inv_fwd else state.w_inv_dgrad
    y = _Fp8FrozenLinear.apply(x, weight, state.w8, state.w8t, state.w_inv_fwd, dgrad_scale)
    if self.bias is not None:
        y = y + self.bias.to(y.dtype)
    return y


def eligible_base(module) -> Optional[str]:
    if type(module) is not torch.nn.Linear:
        return f"{type(module).__name__} is not a plain nn.Linear"
    weight = module.weight
    if type(weight) not in (torch.Tensor, torch.nn.Parameter):
        return f"weight is a {type(weight).__name__}"
    if getattr(weight, "_unsloth_fsdp2_preload", None) is not None or tuple(weight.shape) != (
        module.out_features,
        module.in_features,
    ):
        return f"weight is a {tuple(weight.shape)} shard of {module.out_features} x {module.in_features}"
    if not weight.is_contiguous():
        return "weight is not contiguous"
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return f"weight dtype {weight.dtype}"
    if weight.requires_grad:
        return "weight is trainable"
    if module.in_features % 16 or module.out_features % 16:
        return f"{module.in_features} x {module.out_features} is not a multiple of 16"
    if getattr(module, "weight_fake_quantizer", None) is not None:
        return "QAT fake-quantized"
    return None


def offload_supported() -> bool:
    """The host master needs a quantized-weight dgrad and one process owning the weights."""
    if not _OFFLOAD_MASTER or _DGRAD_DTYPE is None:
        return False
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            return False
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return False
    except Exception:
        return False
    return True


def _fp8_base_apply(
    self,
    fn,
    recurse = True,
):
    state = self.__dict__.get("_unsloth_fp8_lora")
    if state is not None and state.offloaded:
        probe = fn(torch.empty(0, dtype = state.dtype, device = state.device))
        if probe.device == state.device and probe.dtype == state.dtype:
            weight = self._parameters.pop("weight")
            try:
                return torch.nn.Module._apply(self, fn, recurse)
            finally:
                self._parameters["weight"] = weight
        state.restore(self)
    out = torch.nn.Module._apply(self, fn, recurse)
    if state is not None:
        state._key = None
        state.refresh(self.weight)
    return out


def attach_fp8_lora_base(module, offload: Optional[bool] = None) -> Fp8LoRABaseState:
    """Quantize the frozen base `module` (nn.Linear) once and route its GEMMs through fp8. offload: move the
    16-bit master to host RAM (None: offload_supported())."""
    import types

    state = module.__dict__.get("_unsloth_fp8_lora")
    if state is None:
        state = Fp8LoRABaseState(module.weight)
        module.__dict__["_unsloth_fp8_lora"] = state
        module.forward = types.MethodType(_fp8_base_forward, module)
        module._apply = types.MethodType(_fp8_base_apply, module)
        module.__dict__["_unsloth_fp8_lora_load_hook"] = module.register_load_state_dict_post_hook(
            _fp8_requantize_after_load
        )
    if offload is None:
        offload = offload_supported()
    if offload:
        state.offload(module)
    return state


def _fp8_requantize_after_load(module, incompatible_keys):
    state = module.__dict__.get("_unsloth_fp8_lora")
    if state is not None:
        state._key = None
        state.refresh(module.weight)


def restore_fp8_lora_master(module) -> bool:
    """Bring a base layer's exact 16-bit weight back to its compute device (PEFT merge, export)."""
    state = module.__dict__.get("_unsloth_fp8_lora")
    return state.restore(module) if state is not None else False


def detach_fp8_lora_base(module):
    restore_fp8_lora_master(module)
    module.__dict__.pop("_unsloth_fp8_lora", None)
    hook = module.__dict__.pop("_unsloth_fp8_lora_load_hook", None)
    if hook is not None:
        hook.remove()
    module.__dict__.pop("forward", None)
    module.__dict__.pop("_apply", None)


def fp8_lora_memory(model) -> dict:
    """fp8 / device-master / host-master bytes per converted base layer and in total."""
    layers, total = {}, {"fp8": 0, "master_device": 0, "master_host": 0}
    for name, module in model.named_modules():
        state = module.__dict__.get("_unsloth_fp8_lora")
        if state is None:
            continue
        layers[name] = state.resident_bytes(module)
        for key in total:
            total[key] += layers[name][key]
    return {"layers": layers, "total": total}

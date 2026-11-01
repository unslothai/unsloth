# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP8 LoRA on a frozen 16-bit base: the base GEMMs (forward y = x W^T and input gradient dx = dy W) run in
fp8 through torch._scaled_mm, the LoRA branch stays in the activation dtype (Megatron-Bridge TEFusedLoRALinear
split). The base weight is frozen, so it is quantized once (tensorwise e4m3) and cached; the 16-bit weight stays
the source of truth for merge / save. A weight changed in place (merge, unmerge, load) is requantized on its
next eager use. cuBLAS fp8 GEMMs are TN only, so dgrad needs a K-major copy: transposed per backward inside the
opaque op (UNSLOTH_FP8_LORA_CACHE_T=1 caches it instead: +1 byte per weight, 1-7% faster).

Scales (UNSLOTH_FP8_LORA_SCALING): tensor = activations e4m3 / output gradients e5m2, one scale per tensor;
row = one e4m3 scale per token for both. auto picks row on compute capability 12.x and tensor elsewhere (L4's
rowwise _scaled_mm is slower than bf16). UNSLOTH_FP8_LORA_DGRAD=0 keeps the input gradient in 16 bit. The fp8
GEMM writes bfloat16 for float16 activations (then casts), never float16 directly.

Only torch and (optionally) triton are imported here, so the module loads standalone."""

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
    "has_fp8_dgrad",
    "reset_fp8_lora_stats",
]

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
_FP8_MAX = {E4M3: 448.0, E5M2: 57344.0}
# Engagement: fp8 GEMM calls and weight quantizations (eager counters, also counted inside compiled graphs since
# the ops are opaque there).
FP8_LORA_STATS = {"fwd": 0, "dgrad": 0, "dgrad_16bit": 0, "fwd_16bit": 0, "quantize": 0}


def reset_fp8_lora_stats():
    for key in FP8_LORA_STATS:
        FP8_LORA_STATS[key] = 0


# tensor: one scale per activation / gradient tensor (torchao tensorwise); row: one per token (row), the weight
# keeping its tensorwise scale broadcast over the output columns. auto (default): row on compute capability 12.x
# (RTX PRO 6000: 1.24x vs tensorwise 1.17x at Qwen3-1.7B), tensor elsewhere (L4: rowwise _scaled_mm is slower
# than bf16, tensorwise 1.29x).
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
    # None = 16-bit dgrad; "auto" = e5m2 for tensorwise gradients (torchao), e4m3 for rowwise.
    value = os.environ.get("UNSLOTH_FP8_LORA_DGRAD", "auto").strip().lower()
    if value in ("0", "false", "off", "no", "bf16", "16bit"):
        return None
    return {"e4m3": E4M3, "e5m2": E5M2}.get(value, "auto")


_DGRAD_DTYPE = _env_dgrad()
_FWD_FAST_ACCUM = os.environ.get("UNSLOTH_FP8_LORA_FWD_FAST_ACCUM", "1") != "0"
_DGRAD_FAST_ACCUM = os.environ.get("UNSLOTH_FP8_LORA_DGRAD_FAST_ACCUM", "0") == "1"
# 1 keeps a transposed fp8 copy for dgrad (+1 byte / weight); default 0 transposes per backward inside the
# opaque op: 1% slower tensorwise (L4, RTX PRO 6000), 7% slower rowwise at Qwen3-1.7B (RTX PRO 6000), and
# 0.4 / 1.3 GB less peak memory at Qwen3-0.6B / 1.7B.
_CACHE_T = os.environ.get("UNSLOTH_FP8_LORA_CACHE_T", "0") == "1"
# Fewer rows than this (decode) use the 16-bit weight: an fp8 GEMM there is launch bound and less exact.
_MIN_ROWS = int(os.environ.get("UNSLOTH_FP8_LORA_MIN_ROWS", "16"))

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
# Read once: device_count() inside a compiled region graph-breaks.
_MULTI_GPU = torch.cuda.is_available() and torch.cuda.device_count() > 1


if triton is not None:

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
        scale = FP8_MAX / amax
        x = tl.load(x_ptr + offs, mask = mask, other = 0.0).to(tl.float32)
        y = tl.maximum(x * scale, -FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
        y = tl.minimum(y, FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
        if IS_E5M2:
            tl.store(y_ptr + offs, y.to(tl.float8e5), mask = mask)
        else:
            tl.store(y_ptr + offs, y.to(tl.float8e4nv), mask = mask)
        if pid == 0:
            tl.store(inv_ptr, amax / FP8_MAX)


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
            _fp8_tensorwise_cast_kernel[grid](
                x,
                y,
                amax,
                inv,
                x.numel(),
                FP8_MAX = fp8_max,
                IS_E5M2 = fp8_dtype is E5M2,
                BLOCK = BLOCK,
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
        scale = FP8_MAX / amax
        if ONE_BLOCK:
            y = tl.maximum(x * scale, -FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
            y = tl.minimum(y, FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
            if IS_E5M2:
                tl.store(y_row + offs, y.to(tl.float8e5), mask = offs < K)
            else:
                tl.store(y_row + offs, y.to(tl.float8e4nv), mask = offs < K)
        else:
            for k in range(0, K, BLOCK_K):
                mask = k + offs < K
                xk = tl.load(x_row + k + offs, mask = mask, other = 0.0).to(tl.float32)
                y = tl.maximum(xk * scale, -FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
                y = tl.minimum(y, FP8_MAX, propagate_nan = tl.PropagateNan.ALL)
                if IS_E5M2:
                    tl.store(y_row + k + offs, y.to(tl.float8e5), mask = mask)
                else:
                    tl.store(y_row + k + offs, y.to(tl.float8e4nv), mask = mask)
        tl.store(inv_ptr + row, amax / FP8_MAX)


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
            _fp8_rowwise_cast_kernel[(M,)](
                x,
                y,
                inv,
                K,
                x.stride(0),
                FP8_MAX = fp8_max,
                IS_E5M2 = fp8_dtype is E5M2,
                BLOCK_K = BLOCK_K,
                ONE_BLOCK = one,
                num_warps = 8 if BLOCK_K >= 4096 else 4,
            )
        return y, inv
    xf = x.float()
    amax = xf.abs().amax(dim = 1, keepdim = True)
    amax = torch.where(torch.isnan(xf).any(1, keepdim = True), torch.nan, amax).clamp(min = 1e-12)
    y = (xf * (fp8_max / amax)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return y, amax / fp8_max


def _out_dtype(dtype):
    # The fp8 GEMM never writes float16 (outputs past 65504 overflow there); bf16 out, the caller casts back.
    return torch.float32 if dtype == torch.float32 else torch.bfloat16


def _scaled_mm(a, b, scale_a, scale_b, out_dtype, fast_accum):
    return torch._scaled_mm(
        a,
        b,
        scale_a = scale_a,
        scale_b = scale_b,
        out_dtype = out_dtype,
        use_fast_accum = fast_accum,
    )


# Module level: a torch.library.Library deregisters its ops once collected.
_library = None


def _opaque_under_compile(name, fake):
    # Eager calls run the plain function; under torch.compile the call becomes one registered op, so the Triton
    # cast and _scaled_mm stay out of the graph (no graph break, no retrace of the launcher).
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
    if w_inv.dim() == 2:
        x8, x_inv = quantize_rowwise(x, E4M3)
        out_dtype = torch.bfloat16
    else:
        x8, x_inv = quantize_tensorwise(x, E4M3)
        out_dtype = _out_dtype(x.dtype)
    out = _scaled_mm(x8, w8.t(), x_inv, w_inv, out_dtype, _FWD_FAST_ACCUM)
    return out if out.dtype == x.dtype else out.to(x.dtype)


def _dgrad_fake(dy, w8, w_inv, w_is_t):
    return dy.new_empty((dy.shape[0], w8.shape[0] if w_is_t else w8.shape[1]))


@_opaque_under_compile("fp8_lora_dgrad", _dgrad_fake)
def fp8_lora_dgrad(
    dy: torch.Tensor, w8: torch.Tensor, w_inv: torch.Tensor, w_is_t: bool
) -> torch.Tensor:
    """dy [M, N] (16-bit) @ dequant(W [N, K]) -> dx [M, K] in dy.dtype. w8 is the cached K-major copy [K, N]
    (w_is_t) or the forward copy [N, K], transposed here: inside the op it is an eager aten copy, never an
    Inductor-generated fp8 transpose (those write corrupt bytes for some tile configs on sm89)."""
    FP8_LORA_STATS["dgrad"] += 1
    w8t = w8 if w_is_t else w8.t().contiguous()
    if w_inv.dim() == 2:
        # Rowwise gradients are e4m3 (torchao's rowwise recipe; per-token scales need no e5m2 range).
        dy8, dy_inv = quantize_rowwise(dy, E4M3)
        out_dtype = torch.bfloat16
    else:
        dy8, dy_inv = quantize_tensorwise(dy, E5M2 if _DGRAD_DTYPE in (None, "auto") else _DGRAD_DTYPE)
        out_dtype = _out_dtype(dy.dtype)
    out = _scaled_mm(dy8, w8t.t(), dy_inv, w_inv, out_dtype, _DGRAD_FAST_ACCUM)
    return out if out.dtype == dy.dtype else out.to(dy.dtype)


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
        self.w8 = self.w8t = self.w_inv = self.w_inv_fwd = self.w_inv_dgrad = None
        self.scaling = "tensor"
        self._key = None
        self.quantizations = 0
        self.refresh(weight)

    # ---- cache ---------------------------------------------------------------------------------------
    @staticmethod
    def _weight_key(weight):
        return (weight.data_ptr(), weight._version, weight.device, weight.dtype, tuple(weight.shape))

    def refresh(self, weight: torch.Tensor):
        """Quantize `weight` unless the cache already holds it (same storage, version, device)."""
        if _compiling():
            return self
        key = self._weight_key(weight)
        if key == self._key:
            return self
        with torch.no_grad():
            w = weight.detach()
            self.w8, self.w_inv = quantize_tensorwise(w, E4M3)
            self.w8t = self.w8.t().contiguous() if _CACHE_T else None
            self.scaling = _scaling_for(w.device)
            if self.scaling == "row":
                # _scaled_mm's rowwise form takes [M, 1] x [1, N] scales: the weight's one scale per column.
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
        """(fp8 weight, is it the K-major copy) for fp8_lora_dgrad."""
        return (self.w8t, True) if self.w8t is not None else (self.w8, False)

    # ---- kernels.utils packed-state protocol ------------------------------------------------------------
    def dequantize(self, W = None, dtype = None):
        # W is the 16-bit weight (or its .t()): exact, nothing to decode.
        return W

    def _use_fp8(self, X):
        return X.dtype in (torch.bfloat16, torch.float16, torch.float32) and (
            X.numel() // max(X.shape[-1], 1) >= _MIN_ROWS
        )

    def matmul(self, X, W, out = None):
        """``X @ W.t()`` as matmul_lora computes it: the forward when W is the stored weight, the input
        gradient when W is its transposed view."""
        shape = X.shape
        X2 = X.reshape(-1, shape[-1])
        if not self._use_fp8(X2):
            FP8_LORA_STATS["fwd_16bit"] += 1
            result = torch.matmul(X2, W.t().to(X2.dtype))
        elif _is_transposed(W):
            result = self.dgrad(X2, W.t())
        else:
            result = fp8_lora_fwd(X2, self.w8, self.w_inv_fwd)
        result = result.view(*shape[:-1], result.shape[-1])
        return result if out is None else out.copy_(result)

    def dgrad(self, dY, W = None):
        """dY [..., out] @ W -> [..., in]: the base layer's input gradient."""
        shape = dY.shape
        dY2 = dY.reshape(-1, shape[-1])
        if _DGRAD_DTYPE is None or not self._use_fp8(dY2):
            FP8_LORA_STATS["dgrad_16bit"] += 1
            if W is None:
                raise RuntimeError("Unsloth: 16-bit dgrad needs the base weight")
            result = torch.matmul(dY2, W.to(dY2.dtype))
        else:
            w8, w_is_t = self._dgrad_weight()
            result = fp8_lora_dgrad(dY2, w8, self.w_inv_dgrad, w_is_t)
        return result.view(*shape[:-1], result.shape[-1])


def has_fp8_dgrad(W_quant) -> bool:
    return getattr(type(W_quant), "_unsloth_fp8_dgrad", False)


class _Fp8FrozenLinear(torch.autograd.Function):
    """PEFT path (base_layer(x)): y = fp8(x) @ fp8(W).T, dx = fp8(dy) @ fp8(W); no weight gradient."""

    @staticmethod
    def forward(ctx, x, weight, w8, w8t, w_inv_fwd, w_inv_dgrad):
        # w8t None: the backward transposes w8 inside the dgrad op.
        ctx.w_is_t = w8t is not None
        ctx.save_for_backward(
            weight, w8t if w8t is not None else w8, w_inv_fwd if w_inv_dgrad is None else w_inv_dgrad
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
        return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)
    state.refresh(weight)
    # Tensorwise: one scale serves both GEMMs (Dynamo rejects the same tensor passed twice to apply).
    dgrad_scale = None if state.w_inv_dgrad is state.w_inv_fwd else state.w_inv_dgrad
    y = _Fp8FrozenLinear.apply(x, weight, state.w8, state.w8t, state.w_inv_fwd, dgrad_scale)
    if self.bias is not None:
        y = y + self.bias.to(y.dtype)
    return y


def eligible_base(module) -> Optional[str]:
    """None when `module` can take an fp8 frozen base, else why not."""
    if type(module) is not torch.nn.Linear:
        return f"{type(module).__name__} is not a plain nn.Linear"
    weight = module.weight
    if type(weight) not in (torch.Tensor, torch.nn.Parameter):
        # DTensor (FSDP2 / TP shards), tensor subclasses: the cached fp8 copy would be a local shard.
        return f"weight is a {type(weight).__name__}"
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return f"weight dtype {weight.dtype}"
    if weight.requires_grad:
        return "weight is trainable"
    if module.in_features % 16 or module.out_features % 16:
        return f"{module.in_features} x {module.out_features} is not a multiple of 16"
    if getattr(module, "weight_fake_quantizer", None) is not None:
        return "QAT fake-quantized"
    return None


def attach_fp8_lora_base(module) -> Fp8LoRABaseState:
    """Quantize the frozen base `module` (nn.Linear) once and route its GEMMs through fp8."""
    import types
    state = module.__dict__.get("_unsloth_fp8_lora")
    if state is None:
        state = Fp8LoRABaseState(module.weight)
        module.__dict__["_unsloth_fp8_lora"] = state
        module.forward = types.MethodType(_fp8_base_forward, module)
    return state


def detach_fp8_lora_base(module):
    module.__dict__.pop("_unsloth_fp8_lora", None)
    module.__dict__.pop("forward", None)

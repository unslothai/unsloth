# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Trains mxfp4-pack-quantized Linears in place; frozen weight dequantized per use, no 16-bit copy."""

import functools
import os
import types
from typing import Optional

import torch
from torch import nn

__all__ = [
    "Mxfp4PackedLinear",
    "Mxfp4PackedState",
    "mxfp4_packed_matmul",
    "mxfp4_keep_packed_enabled",
    "dequantize_mxfp4_packed",
    "is_mxfp4_scheme",
    "make_mxfp4_packed_linear",
    "adopt_compressed_mxfp4_modules",
    "stack_packed_expert_linears",
    "install_compressed_tensors_keep_packed",
    "patch_peft_merge_for_mxfp4_packed_linears",
    "patch_peft_inits_for_mxfp4_packed_linears",
]

MXFP4_KEEP_PACKED_ENV = "UNSLOTH_MXFP4_KEEP_PACKED"
MXFP4_FORMAT = "mxfp4-pack-quantized"

try:
    from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize as _zoo_mxfp4_dequantize
except Exception:
    _zoo_mxfp4_dequantize = None
try:
    from unsloth_zoo.mxfp4_gemm import mxfp4_gemm_available, mxfp4_matmul as _zoo_mxfp4_matmul
except Exception:
    _zoo_mxfp4_matmul = None
try:
    # Self-check only, no sm_80 floor (zoo #1899+): below sm_80 the tile kernel is the low-memory last resort.
    from unsloth_zoo.mxfp4_gemm import (
        mxfp4_grouped_gemv_available as _zoo_mxfp4_kernel_verified,
    )
except Exception:
    _zoo_mxfp4_kernel_verified = None

_FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip
_LUT_CACHE = {}
# torch < 2.12's ldexp forms 2^e in the input dtype: in fp16 2^16 is inf (0 * 2^16 -> NaN, 2^-10 * 2^20 -> inf)
# and 2^-25 is 0. Decode fp16 in fp32 there, then round once.
_LDEXP_IN_INPUT_DTYPE = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (
    2,
    12,
)


def mxfp4_keep_packed_enabled() -> bool:
    return os.environ.get(MXFP4_KEEP_PACKED_ENV, "1").strip() != "0"


def _dequantize_torch(blocks, scales, dtype):
    work = torch.float32 if dtype == torch.float16 and _LDEXP_IN_INPUT_DTYPE else dtype
    key = (blocks.device, work)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = _LUT_CACHE[key] = torch.tensor(_FP4_VALUES, dtype = work, device = blocks.device)
    *prefix, G, B = blocks.shape
    out = torch.empty(*prefix, G, B * 2, dtype = work, device = blocks.device)
    out[..., 0::2] = lut[(blocks & 0x0F).long()]
    out[..., 1::2] = lut[(blocks >> 4).long()]
    out = torch.ldexp(out, (scales.to(torch.int32) - 127).unsqueeze(-1)).to(dtype)
    return out.reshape(*prefix, G * B * 2)


def dequantize_mxfp4_packed(
    packed,
    scale,
    dtype = torch.bfloat16,
    out = None,
):
    """Packed ``[out, in / 2]`` + E8M0 scales ``[out, in / 32]`` -> ``[out, in]``; zoo kernel if self-verified, else torch.
    ``out``: a contiguous ``[out, in]`` buffer to decode into."""
    out_features, half = packed.shape
    blocks = packed.view(out_features, half // 16, 16)
    if _zoo_mxfp4_dequantize is not None and dtype in (torch.bfloat16, torch.float16):
        if out is None:
            return _zoo_mxfp4_dequantize(blocks, scale, dtype = dtype)
        return _zoo_mxfp4_dequantize(blocks, scale, dtype = dtype, out = out)
    weight = _dequantize_torch(blocks, scale, dtype)
    return weight if out is None else out.copy_(weight)


def _fused_max_rows():
    # Rows up to which the fused GEMM (bytes decoded in registers) beats decode + cuBLAS; B200 sweep.
    try:
        return int(os.environ.get("UNSLOTH_MXFP4_FUSED_MAX_ROWS", "64"))
    except ValueError:
        return 64


_FUSED_MAX_ROWS = _fused_max_rows()


def mxfp4_packed_matmul(
    x,
    packed,
    scale,
    transpose = False,
    out = None,
    bias = None,
):
    """``x @ W^T (+ bias)`` (``transpose=False``) or ``x @ W`` for the packed ``[out, in / 2]`` weight, in x's dtype.
    Skinny forward inputs (decode, small batches) never build the 16-bit weight on sm_80+; below sm_80 they decode it
    for cuBLAS only as far as memory allows (row chunks, else the tile kernel); the rest decode it once for cuBLAS.
    Differentiable in ``x`` / ``bias``: the kernels and the reused chunk buffer have no autograd of their own."""
    if torch.is_grad_enabled() and (
        x.requires_grad or (bias is not None and not transpose and bias.requires_grad)
    ):
        y = _Mxfp4PackedMatmulFunction.apply(x, packed, scale, transpose, bias)
        # As _linear: a used bias never writes out.
        if out is None or (bias is not None and not transpose):
            return y
        return out.copy_(y)
    return _mxfp4_packed_matmul(x, packed, scale, transpose, out, bias)


def _mxfp4_packed_matmul(x, packed, scale, transpose, out, bias):
    rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
    # dX (transpose) stays on decode + cuBLAS: the fused kernel only wins it for a few wide shapes.
    if _zoo_mxfp4_matmul is not None and rows <= _FUSED_MAX_ROWS and x.is_cuda:
        fused = mxfp4_gemm_available(x.device, x.dtype)
        if fused and not transpose:
            return _zoo_mxfp4_matmul(x, packed, scale, trans = transpose, out = out, bias = bias)
        if (
            not fused
            and _zoo_mxfp4_kernel_verified is not None
            and _zoo_mxfp4_kernel_verified(x.device, x.dtype)
        ):
            # Below sm_80 decode + cuBLAS beats the tile kernel, but only the tile kernel never builds the
            # 16-bit weight: decode in chunks of weight rows when the weight does not fit. dX (the backward of
            # such a forward) is bounded the same way.
            if transpose:
                return _bounded_dense_matmul_t(x, packed, scale, out)
            return _bounded_dense_matmul(x, packed, scale, out, bias)
    weight = dequantize_mxfp4_packed(packed, scale, x.dtype)
    if transpose:
        return torch.matmul(x, weight, out = out)
    return _linear(x, weight, out, bias)


class _Mxfp4PackedMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scale, transpose, bias):
        ctx.save_for_backward(packed, scale)
        ctx.transpose, ctx.x_dtype = transpose, x.dtype
        return _mxfp4_packed_matmul(x, packed, scale, transpose, None, None if transpose else bias)

    @staticmethod
    def backward(ctx, grad_output):
        packed, scale = ctx.saved_tensors
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = _mxfp4_packed_matmul(
                grad_output, packed, scale, not ctx.transpose, None, None
            ).to(ctx.x_dtype)
        if ctx.needs_input_grad[4] and not ctx.transpose:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_x, None, None, None, grad_bias


def _linear(x, weight, out, bias):
    if bias is not None or out is None:
        return torch.nn.functional.linear(x, weight, bias)
    return torch.matmul(x, weight.t(), out = out)


# Below sm_80 (`_bounded_dense_matmul`): per GPU, the bytes a decoded chunk may take once the whole 16-bit
# weight failed to allocate, and the calls left before the whole weight is tried again.
_DENSE_BUDGET = {}
_DENSE_RETRY_CALLS = 1024
# Fewer decoded rows than this per chunk: the tile kernel (output only) instead.
_DENSE_MIN_CHUNK_ROWS = 128


def _free_bytes(device):
    """What the caching allocator can still hand out: driver free (capped by a per-process memory fraction)
    plus its own cached free blocks. Only queried after an allocation failed: the queries cost ~100 us."""
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    try:
        cap = int(torch.cuda.get_per_process_memory_fraction(device) * total)
        free = min(free, cap - reserved)
    except Exception:
        pass
    return max(0, free) + max(0, reserved - torch.cuda.memory_allocated(device))


def _bounded_dense_matmul_eager(
    x,
    packed,
    scale,
    out = None,
    bias = None,
):
    """``x @ W^T (+ bias)`` without ever holding more of the 16-bit W than fits: the whole W when it allocates
    (one decode + one cuBLAS call), else row chunks of W written to ``y[..., r0:r1]`` (no cross-chunk sums),
    else the tile kernel."""
    N, half = packed.shape
    K = half * 2
    dtype, device = x.dtype, x.device
    row_bytes = K * x.element_size()
    key = device.index
    state = _DENSE_BUDGET.get(key)
    if state is not None:
        state[1] -= 1
        if state[1] <= 0:
            _DENSE_BUDGET.pop(key, None)
            state = None
    if state is None or N * row_bytes <= state[0]:
        # The weight, the decode's scratch, the output and the cuBLAS workspace: any of them may be the one
        # that does not fit, so the whole attempt is guarded and its tensors are dropped before the retry.
        try:
            weight = torch.empty((N, K), dtype = dtype, device = device)
            return _linear(x, dequantize_mxfp4_packed(packed, scale, dtype, out = weight), out, bias)
        except torch.cuda.OutOfMemoryError:
            weight = None
        state = _DENSE_BUDGET[key] = [_free_bytes(device) // 2, _DENSE_RETRY_CALLS]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    # Like _linear: ``out`` is only written when there is no bias, whichever path runs.
    if (
        bias is None
        and out is not None
        and out.dtype == dtype
        and out.is_contiguous()
        and out.numel() == M * N
    ):
        y = out.view(M, N)
    else:
        y = torch.empty((M, N), dtype = dtype, device = device)
    while True:
        step = min(N, state[0] // row_bytes)
        if step < _DENSE_MIN_CHUNK_ROWS:
            # Not even a small chunk fits: the tile kernel only needs the output.
            return _zoo_mxfp4_matmul(x, packed, scale, trans = False, out = y, bias = bias)
        try:
            buffer = torch.empty((step, K), dtype = dtype, device = device)
            for r0 in range(0, N, step):
                r1 = min(N, r0 + step)
                weight = dequantize_mxfp4_packed(
                    packed[r0:r1], scale[r0:r1], dtype, out = buffer[: r1 - r0]
                )
                y[:, r0:r1] = torch.nn.functional.linear(
                    x2, weight, None if bias is None else bias[r0:r1]
                )
            return y.view(*lead, N)
        except torch.cuda.OutOfMemoryError:
            # Chunks are rewritten whole, so a retry with half the rows restarts cleanly.
            buffer = weight = None
        state[0] = step * row_bytes // 2


def _bounded_dense_matmul_t_eager(
    x,
    packed,
    scale,
    out = None,
):
    """``x @ W`` (the dX of ``x @ W^T``) under the same budget as ``_bounded_dense_matmul_eager``: the whole W
    when it allocates, else row chunks of W whose products are summed in fp32, else the tile kernel."""
    N, half = packed.shape
    K = half * 2
    dtype, device = x.dtype, x.device
    row_bytes = K * x.element_size()
    key = device.index
    state = _DENSE_BUDGET.get(key)
    if state is not None:
        state[1] -= 1
        if state[1] <= 0:
            _DENSE_BUDGET.pop(key, None)
            state = None
    if state is None or N * row_bytes <= state[0]:
        try:
            weight = torch.empty((N, K), dtype = dtype, device = device)
            weight = dequantize_mxfp4_packed(packed, scale, dtype, out = weight)
            return torch.matmul(x, weight, out = out)
        except torch.cuda.OutOfMemoryError:
            weight = None
        state = _DENSE_BUDGET[key] = [_free_bytes(device) // 2, _DENSE_RETRY_CALLS]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, N)
    # Each chunk holds its 16-bit decode and the fp32 copy the fp32 product reads.
    chunk_row_bytes = row_bytes + K * 4
    while True:
        step = min(N, state[0] // chunk_row_bytes)
        if step < _DENSE_MIN_CHUNK_ROWS:
            return _zoo_mxfp4_matmul(x, packed, scale, trans = True, out = out)
        try:
            buffer = torch.empty((step, K), dtype = dtype, device = device)
            acc = torch.zeros((x2.shape[0], K), dtype = torch.float32, device = device)
            for r0 in range(0, N, step):
                r1 = min(N, r0 + step)
                weight = dequantize_mxfp4_packed(
                    packed[r0:r1], scale[r0:r1], dtype, out = buffer[: r1 - r0]
                )
                # fp32 products: a 16-bit chunk product can overflow even when the sum fits.
                acc.addmm_(x2[:, r0:r1].float(), weight.float())
            y = acc.to(dtype).view(*lead, K)
            return y if out is None else out.copy_(y)
        except torch.cuda.OutOfMemoryError:
            buffer = weight = acc = None
        state[0] = step * chunk_row_bytes // 2


_bounded_dense_matmul_op = None
_bounded_dense_matmul_t_op = None
try:

    @torch.library.custom_op("unsloth::mxfp4_bounded_dense_matmul", mutates_args = ())
    def _bounded_dense_matmul_op(
        x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return _bounded_dense_matmul_eager(x, packed, scale, None, bias)

    @_bounded_dense_matmul_op.register_fake
    def _(x, packed, scale, bias):
        return x.new_empty((*x.shape[:-1], packed.shape[0]))

except Exception:
    _bounded_dense_matmul_op = None
try:

    @torch.library.custom_op("unsloth::mxfp4_bounded_dense_matmul_t", mutates_args = ())
    def _bounded_dense_matmul_t_op(
        x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return _bounded_dense_matmul_t_eager(x, packed, scale, None)

    @_bounded_dense_matmul_t_op.register_fake
    def _(x, packed, scale):
        return x.new_empty((*x.shape[:-1], packed.shape[1] * 2))

except Exception:
    _bounded_dense_matmul_t_op = None


def _bounded_dense_matmul_t(x, packed, scale, out):
    if torch.compiler.is_compiling():
        if _bounded_dense_matmul_t_op is None:
            return torch.matmul(x, dequantize_mxfp4_packed(packed, scale, x.dtype), out = out)
        y = _bounded_dense_matmul_t_op(x, packed, scale)
        return y if out is None or out.shape != y.shape else out.copy_(y)
    return _bounded_dense_matmul_t_eager(x, packed, scale, out)


def _bounded_dense_matmul(x, packed, scale, out, bias):
    if torch.compiler.is_compiling():
        # Opaque to the trace: the allocation fallbacks must run at call time, not be traced once.
        if _bounded_dense_matmul_op is None:
            return _linear(x, dequantize_mxfp4_packed(packed, scale, x.dtype), out, bias)
        y = _bounded_dense_matmul_op(x, packed, scale, bias)
        if out is None or bias is not None or out.shape != y.shape:
            return y  # with a bias _linear never writes ``out`` either
        return out.copy_(y)
    return _bounded_dense_matmul_eager(x, packed, scale, out, bias)


def _compute_dtype(x):
    device_type = x.device.type
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    except Exception:
        pass
    return x.dtype


def _packed_linear(x, packed, scale, bias, dtype):
    return mxfp4_packed_matmul(
        x.to(dtype), packed, scale, bias = None if bias is None else bias.to(dtype)
    )


class _Mxfp4PackedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scale, bias):
        out = _packed_linear(x, packed, scale, bias, _compute_dtype(x))
        ctx.save_for_backward(packed, scale)
        ctx.x_dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        packed, scale = ctx.saved_tensors
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = mxfp4_packed_matmul(grad_output, packed, scale, transpose = True).to(ctx.x_dtype)
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_x, None, None, grad_bias


class Mxfp4PackedLinear(nn.Linear):
    _unsloth_mxfp4_packed_linear = True
    compute_dtype = torch.bfloat16

    def forward(self, x):
        bias = self.bias
        if torch.is_grad_enabled() and (
            x.requires_grad or (bias is not None and bias.requires_grad)
        ):
            return _Mxfp4PackedLinearFunction.apply(x, self.weight_packed, self.weight_scale, bias)
        # Inference / frozen input: no autograd node, less per-call overhead on the decode path.
        return _packed_linear(x, self.weight_packed, self.weight_scale, bias, _compute_dtype(x))

    def mxfp4_quant_state(self):
        """What kernels.utils passes as the quant state next to ``weight_packed``, so fused LoRA never holds a 16-bit weight."""
        state = self.__dict__.get("_unsloth_packed_state")
        packed, scale = self._parameters["weight_packed"], self._parameters["weight_scale"]
        if (
            state is None
            or state.packed is not packed
            or state.scale is not scale
            or state.dtype != self.compute_dtype
        ):
            state = self.__dict__["_unsloth_packed_state"] = Mxfp4PackedState(
                packed, scale, self.compute_dtype
            )
        return state

    def dequantize_weight(self, dtype = None):
        return dequantize_mxfp4_packed(
            self.weight_packed, self.weight_scale, dtype or self.compute_dtype
        )

    @property
    def weight(self):
        # Fresh exact decode per read, never kept: writes are lost (merge densifies first).
        return self.dequantize_weight()

    def reset_parameters(self):
        return

    def _apply(self, fn, *args, **kwargs):
        try:
            dtype = fn(torch.empty(0, dtype = self.compute_dtype)).dtype
        except Exception:
            dtype = self.compute_dtype
        result = super()._apply(fn, *args, **kwargs)
        if dtype.is_floating_point:
            self.compute_dtype = dtype
        return result

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, format={MXFP4_FORMAT}"
        )


def _is_transposed(W):
    # W is weight_packed [out, in / 2] or its .t() view; only the view has a unit leading stride.
    return W.dim() == 2 and W.stride(0) == 1 and W.stride(1) != 1


class Mxfp4PackedState:
    """Quant state for a packed base in Unsloth's LoRA kernels: ``W`` is ``weight_packed`` (or its ``.t()``)."""

    _unsloth_packed_weight_state = True

    def __init__(self, packed, scale, dtype):
        self.packed, self.scale, self.dtype = packed, scale, dtype

    def dequantize(
        self,
        W = None,
        dtype = None,
    ):
        weight = dequantize_mxfp4_packed(self.packed, self.scale, dtype or self.dtype)
        return weight.t() if W is not None and _is_transposed(W) else weight

    def matmul(
        self,
        X,
        W,
        out = None,
    ):
        """``X @ D(W).t()`` where D(W) is W's 16-bit value, as matmul_lora computes it."""
        return mxfp4_packed_matmul(X, self.packed, self.scale, transpose = _is_transposed(W), out = out)


def is_mxfp4_scheme(scheme, default_format = None) -> bool:
    if any(
        getattr(scheme, f, None) is not None for f in ("input_activations", "output_activations")
    ):
        return False
    fmt = getattr(scheme, "format", None) or default_format
    fmt = getattr(fmt, "value", fmt)
    if fmt != MXFP4_FORMAT:
        return False
    weights = getattr(scheme, "weights", None)
    if weights is None:
        return True
    kind = getattr(weights, "type", "float")
    kind = str(getattr(kind, "value", kind)).lower()
    return (
        kind == "float"
        and int(getattr(weights, "num_bits", 4) or 4) == 4
        and int(getattr(weights, "group_size", 32) or 32) == 32
    )


def _set_compressed_attributes(module, scheme):
    module.quantization_scheme = scheme
    try:
        from compressed_tensors.quantization import QuantizationStatus
        module.quantization_status = QuantizationStatus.COMPRESSED
    except Exception:
        module.quantization_status = "compressed"


def make_mxfp4_packed_linear(
    in_features: int,
    out_features: int,
    bias: Optional[torch.Tensor] = None,
    scheme = None,
    device = None,
    dtype = None,
) -> Mxfp4PackedLinear:
    if in_features % 32 != 0:
        raise ValueError(
            f"Unsloth: an MXFP4 Linear needs in_features divisible by 32, got {in_features}."
        )
    module = Mxfp4PackedLinear.__new__(Mxfp4PackedLinear)
    nn.Module.__init__(module)
    module.in_features = in_features
    module.out_features = out_features
    module.weight_packed = nn.Parameter(
        torch.empty(out_features, in_features // 2, dtype = torch.uint8, device = device),
        requires_grad = False,
    )
    module.weight_scale = nn.Parameter(
        torch.empty(out_features, in_features // 32, dtype = torch.uint8, device = device),
        requires_grad = False,
    )
    if isinstance(bias, nn.Parameter):
        module.bias = bias
    else:
        module.register_parameter("bias", None)
    if dtype is not None and dtype.is_floating_point:
        module.compute_dtype = dtype
    _set_compressed_attributes(module, scheme)
    return module


def _drop_compressed_tensors_forward(module):
    """Drop compressed-tensors' per-instance forward; other instance wrappers (offload) stay."""
    forward = module.__dict__.get("forward")
    code = getattr(getattr(forward, "__func__", forward), "__code__", None)
    if code is not None and code.co_name == "quantized_forward":
        del module.__dict__["forward"]


def _restore_dense_linear(module, dtype):
    device = module._parameters["weight_packed"].device
    for name in [n for n in module._parameters if n.startswith("weight_")]:
        del module._parameters[name]
    module.weight = nn.Parameter(
        torch.empty(module.out_features, module.in_features, dtype = dtype, device = device),
        requires_grad = False,
    )
    for name in ("quantization_scheme", "quantization_status"):
        module.__dict__.pop(name, None)
    _drop_compressed_tensors_forward(module)


def adopt_compressed_mxfp4_modules(
    model,
    default_format = None,
    packed_names = None,
    dtype = None,
) -> int:
    """Adopt every MXFP4 Linear as ``Mxfp4PackedLinear`` and drop the decompress hook; all or nothing (0).
    Linears not in ``packed_names`` go dense: Kimi-K3 ships its latent MoE projections in bf16."""
    if not mxfp4_keep_packed_enabled():
        return 0
    candidates, others, dense = [], 0, []
    for name, module in model.named_modules():
        if isinstance(module, Mxfp4PackedLinear):
            continue
        packed = getattr(module, "_parameters", {}).get("weight_packed")
        if packed is None:
            if getattr(module, "quantization_scheme", None) is not None:
                others += 1
            continue
        scale = module._parameters.get("weight_scale")
        scheme = getattr(module, "quantization_scheme", None)
        if not (
            type(module) is nn.Linear
            and "weight" not in module._parameters
            and packed.dtype == torch.uint8
            and scale is not None
            and scale.dtype == torch.uint8
            and is_mxfp4_scheme(scheme, default_format)
        ):
            others += 1
        elif packed_names is not None and name not in packed_names:
            dense.append(module)
        else:
            candidates.append(module)
    # The hook would decompress adopted modules too, so adopt only when it can be dropped.
    if others or not candidates:
        return 0
    for module in dense:
        _restore_dense_linear(module, dtype or torch.get_default_dtype())
    for module in candidates:
        module.__class__ = Mxfp4PackedLinear
        if dtype is not None and dtype.is_floating_point:
            module.compute_dtype = dtype
        _drop_compressed_tensors_forward(module)
    hook = getattr(model, "ct_decompress_hook", None)
    if hook is not None:
        hook.remove()
        try:
            delattr(model, "ct_decompress_hook")
        except AttributeError:
            pass
    return len(candidates)


def stack_packed_expert_linears(
    model,
    block_names,
    dtype = None,
) -> list:
    """Stack adopted per-expert Linears; gate_up is released before down is allocated (one-layer transient)."""
    from .compressed_tensors_bnb import _new_stacked_experts, _plain_expert_shape

    stacked = []
    for name in block_names:
        block = model.get_submodule(name)
        experts = block.experts
        if not isinstance(experts, nn.ModuleList) or not all(
            isinstance(getattr(e, w, None), Mxfp4PackedLinear)
            for e in experts
            for w in ("w1", "w2", "w3")
        ):
            continue
        dims = _plain_expert_shape(experts)
        if dims is None:
            continue
        hidden, inter = dims
        E, device = len(experts), experts[0].w1.weight_packed.device
        new = _new_stacked_experts(experts, dims, dtype or experts[0].w1.compute_dtype, "meta")
        empty = lambda *shape: torch.empty(shape, dtype = torch.uint8, device = device)  # noqa: E731
        for target, out_rows, in_features, projections in (
            (
                "gate_up",
                2 * inter,
                hidden,
                (("w1", slice(0, inter)), ("w3", slice(inter, 2 * inter))),
            ),
            ("down", hidden, inter, (("w2", slice(None)),)),
        ):
            stacked_blocks = empty(E, out_rows, in_features // 32, 16)
            stacked_scales = empty(E, out_rows, in_features // 32)
            with torch.no_grad():
                for e, expert in enumerate(experts):
                    for proj, rows in projections:
                        linear = getattr(expert, proj)
                        blocks = stacked_blocks[e, rows]
                        blocks.copy_(linear.weight_packed.view(blocks.shape))
                        stacked_scales[e, rows].copy_(linear.weight_scale)
                        del linear._parameters["weight_packed"], linear._parameters["weight_scale"]
            setattr(new, f"{target}_blocks", nn.Parameter(stacked_blocks, requires_grad = False))
            setattr(new, f"{target}_scales", nn.Parameter(stacked_scales, requires_grad = False))
        block.experts = new.finalize()
        stacked.append(name)
    return stacked


def _full_finetuning() -> bool:
    return os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1"


def _unwrap_offload_after_lazy_decompress(model) -> bool:
    """A plain transformers load (e.g. reloading a checkpoint Unsloth saved, in the same process) keeps
    the weights compressed and decompresses them in compressed-tensors' first-forward hook, which also
    wraps every module's parameters in an OffloadCache. Unsloth's compiled forwards cannot trace that
    cache ("Unsupported: Data pointer comparison"), so unwrap the same-device ones right after the
    hook, as Unsloth's own loads do. Real CPU / disk offload is left alone."""
    if not hasattr(model, "ct_decompress_hook") or hasattr(model, "_unsloth_ct_unwrap_hook"):
        return False

    def _unsloth_ct_unwrap_hook(module, args):
        # Pre-hooks run in registration order: still present means the decompress hook did not run.
        if hasattr(module, "ct_decompress_hook"):
            return None
        handle = module.__dict__.pop("_unsloth_ct_unwrap_hook", None)
        if handle is not None:
            handle.remove()
        try:
            from .loader_utils import _remove_same_device_compressed_tensors_offload
            _remove_same_device_compressed_tensors_offload(module)
        except Exception:
            pass
        return None

    model._unsloth_ct_unwrap_hook = model.register_forward_pre_hook(_unsloth_ct_unwrap_hook)
    return True


def install_compressed_tensors_keep_packed() -> bool:
    """Stock compressed-tensors loader keeps MXFP4 packed (adopt, then stack experts) under the same
    all-or-nothing plan; declined, full finetuning or explicit dequantize = stock. Idempotent."""
    try:
        from transformers.quantizers.quantizer_compressed_tensors import (
            CompressedTensorsHfQuantizer,
        )
    except Exception:
        return False
    if "_unsloth_keep_packed" in CompressedTensorsHfQuantizer.__dict__:
        return True
    original_before = CompressedTensorsHfQuantizer._process_model_before_weight_loading
    original_after = CompressedTensorsHfQuantizer._process_model_after_weight_loading

    def _process_model_before_weight_loading(self, model, **kwargs):
        result = original_before(self, model, **kwargs)
        self._unsloth_mxfp4_blocks = []
        config = getattr(self, "quantization_config", None)
        # `dequantize=True` (transformers 5.5+) or `run_compressed=False` (older).
        if (
            not mxfp4_keep_packed_enabled()
            or _full_finetuning()
            or getattr(config, "dequantize", False)
            or getattr(self, "run_compressed", True) is False
        ):
            return result
        fmt = getattr(config, "format", None) or getattr(
            getattr(config, "quantization_config", None), "format", None
        )
        try:
            from .compressed_tensors_bnb import (
                _checkpoint_keys,
                keep_mxfp4_experts_packed,
                plan_mxfp4_keep_packed,
            )

            # Skip planning non-MXFP4 checkpoints: key matching is slow on large ones.
            if not keep_mxfp4_experts_packed(config.to_dict()):
                return result
            keys = _checkpoint_keys(kwargs.get("checkpoint_files"))
            plan = plan_mxfp4_keep_packed(model, keys) if keys else None
        except Exception:
            plan = None
        if plan is None:
            return result
        packed_names = set(plan.linears) | {
            f"{name}.experts.{index}.{proj}"
            for name, block, _, _ in plan.blocks
            for index in range(len(block.experts))
            for proj in ("w1", "w2", "w3")
        }
        # transformers 5.x does not pass the load dtype here.
        dtype = kwargs.get("dtype")
        if not isinstance(dtype, torch.dtype):
            dtype = getattr(model, "dtype", None)
        adopted = adopt_compressed_mxfp4_modules(
            model,
            default_format = fmt,
            packed_names = packed_names,
            dtype = dtype if isinstance(dtype, torch.dtype) else None,
        )
        if adopted:
            self._unsloth_mxfp4_blocks = [name for name, _, _, _ in plan.blocks]
        return result

    def _process_model_after_weight_loading(self, model, **kwargs):
        result = original_after(self, model, **kwargs)
        _unwrap_offload_after_lazy_decompress(model)
        blocks = getattr(self, "_unsloth_mxfp4_blocks", None)
        if blocks:
            self._unsloth_mxfp4_blocks = []
            stacked = stack_packed_expert_linears(model, blocks)
            print(
                f"Unsloth: Kept the MXFP4 weights packed ({len(stacked)} expert stacks); "
                "they are dequantized on the fly."
            )
        return result

    CompressedTensorsHfQuantizer._process_model_before_weight_loading = (
        _process_model_before_weight_loading
    )
    CompressedTensorsHfQuantizer._process_model_after_weight_loading = (
        _process_model_after_weight_loading
    )
    CompressedTensorsHfQuantizer._unsloth_keep_packed = True
    return True


# PEFT merge: weight.data += delta silently no-ops on packed base; merge densifies, unmerge restores

_PACKED_STATE = "_unsloth_mxfp4_packed_state"


def _swap_class(module, cls):
    # accelerate keeps the wrapped forward as `_old_forward`; it must follow the class.
    old = module.__dict__.get("_old_forward")
    hooked = old is not None and getattr(old, "__func__", None) is type(module).forward
    module.__class__ = cls
    if hooked:
        module._old_forward = types.MethodType(cls.forward, module)


def _densify(module):
    weight = module.dequantize_weight()
    packed = module._parameters.pop("weight_packed")
    scale = module._parameters.pop("weight_scale")
    module.__dict__[_PACKED_STATE] = (packed, scale, module.compute_dtype)
    _swap_class(module, nn.Linear)
    module.weight = nn.Parameter(weight, requires_grad = False)


def _restore_packed(module):
    packed, scale, _ = module.__dict__.pop(_PACKED_STATE)
    device, dtype = module.weight.device, module.weight.dtype
    del module._parameters["weight"]
    _swap_class(module, Mxfp4PackedLinear)
    module.compute_dtype = dtype
    for name, param in (("weight_packed", packed), ("weight_scale", scale)):
        if param.device != device:
            param = nn.Parameter(param.data.to(device), requires_grad = False)
        module.register_parameter(name, param)


def patch_peft_merge_for_mxfp4_packed_linears() -> bool:
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except Exception:
        return False
    if getattr(LoraLinear.merge, "_unsloth_mxfp4_patched", False):
        return True
    original_merge = LoraLinear.merge
    original_unmerge = LoraLinear.unmerge

    def merge(self, *args, **kwargs):
        base = self.get_base_layer()
        if not isinstance(base, Mxfp4PackedLinear):
            return original_merge(self, *args, **kwargs)
        _densify(base)
        try:
            return original_merge(self, *args, **kwargs)
        finally:
            if not self.merged_adapters:
                _restore_packed(base)

    def unmerge(self, *args, **kwargs):
        result = original_unmerge(self, *args, **kwargs)
        base = self.get_base_layer()
        if _PACKED_STATE in base.__dict__ and not self.merged_adapters:
            _restore_packed(base)
        return result

    merge._unsloth_mxfp4_patched = True
    unmerge._unsloth_mxfp4_patched = True
    LoraLinear.merge = merge
    LoraLinear.unmerge = unmerge
    return True


# These inits subtract the adapter from `weight.data`, a throwaway decode on a packed base: silent model change.
_BASE_WEIGHT_INITS = ("pissa_init", "olora_init", "corda_init", "loftq_init", "lora_ga_init")


def _refuse_on_packed_base(original):
    @functools.wraps(original)
    def init(self, *args, **kwargs):
        if isinstance(self.get_base_layer(), Mxfp4PackedLinear):
            raise NotImplementedError(
                f"Unsloth: `{original.__name__}` rewrites the base weight, which stays packed in "
                "MXFP4 here and cannot be written. Use the default LoRA initialisation."
            )
        return original(self, *args, **kwargs)

    init._unsloth_mxfp4_patched = True
    return init


def patch_peft_inits_for_mxfp4_packed_linears() -> bool:
    try:
        from peft.tuners.lora.layer import LoraLayer
    except Exception:
        return False
    for name in _BASE_WEIGHT_INITS:
        original = getattr(LoraLayer, name, None)
        if original is not None and not getattr(original, "_unsloth_mxfp4_patched", False):
            setattr(LoraLayer, name, _refuse_on_packed_base(original))
    return True


patch_peft_merge_for_mxfp4_packed_linears()
patch_peft_inits_for_mxfp4_packed_linears()

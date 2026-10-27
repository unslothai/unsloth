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
import os
import functools
from contextlib import nullcontext
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn import functional as F
import math
from unsloth_zoo.utils import Version

from .indexing import long_indexing
from unsloth_zoo.log import logger
from unsloth_zoo.temporary_patches.common import torch_compile

torch_matmul = torch.matmul


# Read once for the decode GEMV: device_count() in a compiled region graph-breaks ("torch.* op returned non-Tensor").
_CUDA_MULTI_DEVICE = torch.cuda.is_available() and torch.cuda.device_count() > 1
_XPU_MULTI_DEVICE = (
    hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count() > 1
)


def _fp8_triton_device_context(tensor: torch.Tensor, static_device_count = False):
    # static_device_count only for forward-only kernels: torch 2.11 compiles FP8BlockQuantLinear's backward to zeros.
    if tensor.device.type == "cuda" and (
        _CUDA_MULTI_DEVICE if static_device_count else torch.cuda.device_count() > 1
    ):
        return torch.cuda.device(tensor.device)
    if tensor.device.type == "xpu" and (
        _XPU_MULTI_DEVICE
        if static_device_count
        else (hasattr(torch, "xpu") and torch.xpu.device_count() > 1)
    ):
        return torch.xpu.device(tensor.device)
    return nullcontext()


# Module level: a torch.library.Library deregisters its ops once collected.
_fp8_library = None


def _opaque_under_compile(name, fake):
    # torch.compile swaps the launcher for one registered op: tracing it breaks the graph (device_count) and recompiles
    # the Triton kernel under dynamic shapes 40-80x slower. Eager calls still run the plain function, at no cost.
    # torch.library.Library, not custom_op: compiled graphs dispatch it with ~3 us per call instead of ~23 us.
    def decorator(fn):
        global _fp8_library
        substitute_in_graph = getattr(torch._dynamo, "substitute_in_graph", None)
        if substitute_in_graph is None or not hasattr(torch.library, "infer_schema"):
            return fn
        try:
            if _fp8_library is None:
                _fp8_library = torch.library.Library("unsloth", "FRAGMENT")
            _fp8_library.define(name + torch.library.infer_schema(fn, mutates_args = ()))
            _fp8_library.impl(name, fn, "CompositeExplicitAutograd")
            torch.library.register_fake(f"unsloth::{name}", fake, lib = _fp8_library)
            substitute_in_graph(fn)(
                functools.wraps(fn)(
                    lambda *args, **kwargs: getattr(torch.ops.unsloth, name)(*args, **kwargs)
                )
            )
        except (
            RuntimeError,
            TypeError,
            ValueError,
        ):  # Already registered by an earlier import of this module.
            pass
        return fn

    return decorator


try:
    from transformers.integrations.finegrained_fp8 import FP8Linear
except:
    FP8Linear = None
    logger.info(
        "Unsloth: FP8 models need importing FP8Linear from `transformers.integrations.finegrained_fp8` but we don't see it."
    )

try:
    from transformers.integrations.finegrained_fp8 import FP8GroupedLinear
except:
    FP8GroupedLinear = None

try:
    from transformers.integrations.fbgemm_fp8 import FbgemmFp8Linear
except:
    FbgemmFp8Linear = None
    logger.info(
        "Unsloth: FP8 models need importing FbgemmFP8Linear from `transformers.integrations.fbgemm_fp8` but we don't see it."
    )

try:
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        triton_quantize_fp8_block,
    )
except:
    triton_quantize_fp8_block = None
    logger.info(
        "Unsloth: Could not find fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm.triton_quantize_fp8_block"
    )


def _import_torchao_blockwise_gemm():
    # torchao <= 0.17 ships blockwise_fp8_gemm under prototype; 0.18 moved it to torchao.kernel.
    try:
        from torchao.prototype.blockwise_fp8_inference.blockwise_quantization import (
            blockwise_fp8_gemm,
        )
        return blockwise_fp8_gemm
    except Exception:
        pass
    try:
        from torchao.kernel.blockwise_quantization import blockwise_fp8_gemm
        return blockwise_fp8_gemm
    except Exception:
        pass
    logger.info("Unsloth: Could not find torchao blockwise_fp8_gemm")
    return None


torchao_blockwise_gemm = _import_torchao_blockwise_gemm()


# ROCm Triton fuses fp32 multiply + fp16 cast into v_fma_mixlo_f16, which drops -0.0 and rounds
# differently from the torch loop, so the dequant kernels round to fp16 with integer ops there.
_FP8_EXACT_F16_ROUND = torch.version.hip is not None


def _fp8_exact_f16_round(dtype):
    flag = os.environ.get("UNSLOTH_FP8_EXACT_F16_ROUND", "")
    on = _FP8_EXACT_F16_ROUND if flag == "" else flag == "1"
    return on and dtype == torch.float16


@triton.jit
def _f32_to_f16_rne(x):
    # IEEE fp32 -> fp16, round to nearest even, subnormals kept, sign of zero kept, NaN quietened.
    u = x.to(tl.int32, bitcast = True)
    sign = (u >> 16) & 0x8000
    a = u & 0x7FFFFFFF
    c = tl.minimum(a, 0x477FF000)
    normal = (c + 0xFFF + ((c >> 13) & 1) - 0x38000000) >> 13
    # Below 2^-14, in units of 2^-24; clamped so normal lanes never shift by a negative count.
    shift = tl.minimum(tl.maximum(126 - (a >> 23), 1), 31)
    m = (a & 0x7FFFFF) | 0x800000
    q = m >> shift
    rem = m - (q << shift)
    half = 1 << (shift - 1)
    q = q + ((rem > half) | ((rem == half) & ((q & 1) == 1))).to(tl.int32)
    h = tl.where(a < 0x38800000, q, normal)
    h = tl.where(a >= 0x477FF000, 0x7C00, h)
    h = tl.where(a > 0x7F800000, 0x7E00 | ((a >> 13) & 0x3FF), h)
    return (h | sign).to(tl.uint16).to(tl.float16, bitcast = True)


@triton.jit
def weight_dequant_kernel(
    x_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    EXACT_F16: tl.constexpr = False,
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # tl.arange is int32, so offs_m * N overflows past 2**31 elements (flattened MoE expert stacks); index in int64.
    offs = offs_m[:, None].to(tl.int64) * N + offs_n[None, :].to(tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask = mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    if EXACT_F16:
        y = _f32_to_f16_rne(y)
    tl.store(y_ptr + offs, y, mask = mask)


@_opaque_under_compile(
    "fp8_weight_dequant_block",
    lambda x, s, block_size = 128, dtype = torch.bfloat16: torch.empty_like(
        x, dtype = dtype, memory_format = torch.contiguous_format
    ),
)
def weight_dequant_block(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype = dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    with _fp8_triton_device_context(x):
        weight_dequant_kernel[grid](
            x,
            s,
            y,
            M,
            N,
            BLOCK_SIZE = block_size,
            EXACT_F16 = _fp8_exact_f16_round(dtype),
            enable_fp_fusion = True,
        )
    return y


def _is_transposed_view(x):
    return (
        x.dim() == 2 and x.shape[0] > 1 and x.shape[1] > 1 and x.stride(0) == 1 and x.stride(1) != 1
    )


def _has_fbgemm_rowwise():
    try:
        return hasattr(torch.ops.fbgemm, "quantize_fp8_per_row") and hasattr(
            torch.ops.fbgemm, "f8f8bf16_rowwise"
        )
    except Exception:
        return False


def _scale_mul(x, s, dtype):
    # fp16: an fp16 scale goes subnormal below 6.1e-5 and overflows above 65504, zeroing or inf-ing
    # the weight, so multiply by the fp32 scale and round once. fp8 values are exact in fp16, and an
    # fp16 out= makes TensorIterator compute in fp32 per element without fp32 temporaries. Dynamo
    # refuses a strided out=; under compile Inductor fuses the fp32 expression instead (same bits).
    # Other dtypes keep their dtype-native expression.
    if _scale_dequant_one_pass_ok(x, s, dtype):
        return _scale_dequant_one_pass(x, s, dtype)
    if dtype != torch.float16:
        return x.to(dtype) * s.to(dtype)
    # out= has no autograd: a trainable weight or scale takes the fp32 expression (same bits).
    if torch.compiler.is_compiling() or (
        torch.is_grad_enabled() and (x.requires_grad or s.requires_grad)
    ):
        return (x.to(torch.float32) * s.to(torch.float32)).to(dtype)
    x16 = x.to(dtype)
    return torch.mul(x16, s.to(torch.float32), out = torch.empty_like(x16))


# At or below 4M elements the launch dominates on A100 and newer, and torch's two passes are no slower.
_SCALE_DEQUANT_MIN_NUMEL = 1 << 22


def _scale_dequant_one_pass_ok(x, s, dtype):
    # NVIDIA e4m3fn only; under compile only below sm89, where Inductor cannot emit fp8e4nv.
    if torch.compiler.is_compiling():
        # bf16 below sm80: Inductor never compiles the graph, so the torch expression runs eagerly as before.
        if not _fp8_e4m3fn_unsupported_in_trace(x) or _bf16_skipped_by_inductor(x, dtype):
            return False
    elif x.numel() <= _SCALE_DEQUANT_MIN_NUMEL:
        return False
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.float8_e4m3fn
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.dim() == 2
        and s.device == x.device
        and s.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and not (torch.is_grad_enabled() and (x.requires_grad or s.requires_grad))
        # The kernel indexes dense rows, so a transposed view must be a dense transpose.
        and (x.is_contiguous() or (_is_transposed_view(x) and x.t().is_contiguous()))
    )


@_opaque_under_compile(
    "fp8_scale_dequant",
    lambda x, s, dtype: torch.empty_like(x, dtype = dtype),
)
def _scale_dequant_one_pass(x: torch.Tensor, s: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    transposed = _is_transposed_view(x)
    if transposed:
        x = x.t()
    m, n = x.shape
    out = torch.empty((m, n), dtype = dtype, device = x.device)
    if out.numel() == 0:
        return out.t() if transposed else out
    xb = x.view(torch.uint8)
    BLOCK_M, BLOCK_N, num_warps = _SCALE_DEQUANT_TILE
    with _fp8_triton_device_context(x):
        _e4m3fn_scale_dequant_kernel[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))](
            xb,
            s,
            out,
            m,
            n,
            0 if s.numel() == 1 else s.stride(0),
            BLOCK_M = BLOCK_M,
            BLOCK_N = BLOCK_N,
            SCALE_ON_COLS = transposed,
            ROUND_SCALE = dtype != torch.float16,
            LONG_INDEXING = long_indexing(xb, out, block = BLOCK_M * n),
            num_warps = num_warps,
            enable_fp_fusion = True,
        )
    return out.t() if transposed else out


def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    dtype = torch.bfloat16,
):
    # Transposed views keep scales in storage order; square weights can't be told apart by shape.
    if _is_transposed_view(x):
        return weight_dequant(x.t(), s, dtype).t()
    # Per-tensor scale: single value for entire weight matrix
    if s.numel() == 1:
        return _scale_mul(x, s.view(1, 1), dtype)
    # Row quantized weight: scale shape is (m, 1) or (n, 1)
    elif s.ndim == 2 and s.shape[1] == 1:
        if x.shape[0] == s.shape[0]:
            y = _scale_mul(x, s, dtype)
        elif x.shape[1] == s.shape[0]:
            # Sometimes called with the transpose of the weight.
            y = _scale_mul(x.t(), s, dtype)
            y = y.t()
        else:
            raise ValueError(f"Incompatible shapes {x.shape = }, {s.shape = }")
        return y
    else:
        # Block scales: the any-shape helper also gives fast_dequantize callers the pre-sm89 fallback.
        return _blockwise_weight_dequant_any_shape(
            x, s, getattr(s, "block_size", None) or [128, 128], dtype
        )


# Copied from huggingface.co/deepseek-ai/DeepSeek-V3 inference/kernel.py
@triton.jit
def act_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr = False,
):
    pid = tl.program_id(axis = 0)
    if LONG_INDEXING:
        offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    else:
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    # All-zero row: keep scale at 1 so LoRA's zero dY does not become NaN (a deviation from the original
    # implementation).
    s = 1.0 if s == 0 else s
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


@_opaque_under_compile(
    "fp8_act_quant",
    lambda x, block_size = 128: (
        x.new_empty(x.shape, dtype = torch.float8_e4m3fn),
        x.new_empty(*x.shape[:-1], x.shape[-1] // block_size, dtype = torch.float32),
    ),
)
def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype = torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype = torch.float32)

    def grid(meta):
        return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

    with _fp8_triton_device_context(x):
        act_quant_kernel[grid](
            x, y, s, BLOCK_SIZE = block_size, LONG_INDEXING = long_indexing(x), enable_fp_fusion = True
        )
    return y, s


# Adapted from sgl-project/sglang python/sglang/srt/layers/quantization/fp8_kernel.py
@triton.jit
def _w8a8_block_fp8_matmul(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LONG_INDEXING: tl.constexpr = False,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """

    pid = tl.program_id(axis = 0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if LONG_INDEXING:
        offs_am = offs_am.to(tl.int64)
        offs_bn = offs_bn.to(tl.int64)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask = offs_k[None, :] < K - k * BLOCK_SIZE_K, other = 0.0)
        b = tl.load(b_ptrs, mask = offs_k[:, None] < K - k * BLOCK_SIZE_K, other = 0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if LONG_INDEXING:
        offs_cm = offs_cm.to(tl.int64)
        offs_cn = offs_cn.to(tl.int64)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask = c_mask)


try:
    from triton.tools.tensor_descriptor import TensorDescriptor as _TensorDescriptor
except Exception:
    _TensorDescriptor = None

# UNSLOTH_FP8_TMA_GEMM=0 keeps the pointer kernel above on every GPU.
_FP8_TMA_GEMM = os.environ.get("UNSLOTH_FP8_TMA_GEMM", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
_TL_OUT_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _w8a8_block_fp8_matmul_tma(
    a_desc,
    b_desc,
    c_desc,
    As,
    Bs,
    M,
    N,
    K,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # _w8a8_block_fp8_matmul with A, B and C moved by TMA descriptors (out-of-range rows and K tail
    # are zero filled on load, clipped on store) and a pipelined K loop; same per-tile math and order,
    # so outputs are bitwise identical. BLOCK_SIZE_N == group_n and BLOCK_SIZE_K == group_k.
    pid = tl.program_id(axis = 0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    As_ptrs = As + offs_am * stride_As_m
    Bs_ptrs = Bs + pid_n * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages = NUM_STAGES):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
        b = b_desc.load([pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K])
        a_s = tl.load(As_ptrs + k * stride_As_k)
        b_s = tl.load(Bs_ptrs + k * stride_Bs_k)
        accumulator += tl.dot(a, b.T) * a_s[:, None] * b_s
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator.to(OUT_DTYPE))


@functools.lru_cache(maxsize = None)
def _fp8_tma_gemm_arch_at(index):
    # Measured on B200 only (2.2-2.4x over the pointer kernel at 8192-token training shapes): 4 stages
    # of 128 x 128 fp8 A and B tiles need 128 KB of shared memory, more than sm_120 / sm_89 have.
    return torch.version.hip is None and torch.cuda.get_device_capability(index)[0] == 10


def _fp8_tma_gemm_disable(reason):
    global _FP8_TMA_GEMM
    if _FP8_TMA_GEMM:
        _FP8_TMA_GEMM = False
        logger.warning(
            f"Unsloth: TMA block FP8 GEMM disabled ({reason}); using the pointer kernel."
        )


def _fp8_tma_gemm_ok(A, B, C, BLOCK_SIZE_M, block_n, block_k, long_idx):
    if not _FP8_TMA_GEMM or _TensorDescriptor is None or torch.compiler.is_compiling():
        return False
    if A.device.type != "cuda" or C.dtype not in _TL_OUT_DTYPES or long_idx:
        return False
    if BLOCK_SIZE_M != 128 or block_n != 128 or block_k != 128:
        return False
    K = A.shape[-1]
    N = B.shape[0]
    # TMA: 16-byte aligned bases and row pitches.
    if K % 16 != 0 or (N * C.element_size()) % 16 != 0:
        return False
    if A.data_ptr() % 16 or B.data_ptr() % 16 or C.data_ptr() % 16:
        return False
    index = A.device.index if A.device.index is not None else torch.cuda.current_device()
    if not _fp8_tma_gemm_arch_at(index):
        return False
    checked = _FP8_TMA_SELF_CHECKED.get(index)
    if checked is None:
        if torch.cuda.is_current_stream_capturing():
            return False  # the self check allocates and syncs: run it outside graph capture
        checked = _FP8_TMA_SELF_CHECKED[index] = _fp8_tma_gemm_self_check(index)
    return checked


# from_tensor re-validates every field per call (~4 us each): validate once per (dtype, block shape).
_FP8_TMA_DESC_FIELDS = {}


def _fp8_tma_descriptor(tensor, rows, cols, block_shape):
    key = (tensor.dtype, block_shape)
    fields = _FP8_TMA_DESC_FIELDS.get(key)
    if fields is None:
        proto = _TensorDescriptor.from_tensor(tensor.view(rows, cols), list(block_shape))
        # Tensor-independent fields only: caching `base` would keep the first GEMM's tensor alive.
        fields = {
            k: v
            for k, v in (getattr(proto, "__dict__", None) or {}).items()
            if k not in ("base", "shape", "strides")
        }
        _FP8_TMA_DESC_FIELDS[key] = fields
    # A sliced weight's row pitch is not its width: keep from_tensor's real strides.
    if (
        not fields
        or rows <= 0
        or cols <= 0
        or tensor.numel() != rows * cols
        or not tensor.is_contiguous()
    ):
        return _TensorDescriptor.from_tensor(tensor.view(rows, cols), list(block_shape))
    desc = _TensorDescriptor.__new__(_TensorDescriptor)
    desc.__dict__.update(fields)
    desc.base = tensor
    desc.shape = [rows, cols]
    desc.strides = [cols, 1]
    desc.block_shape = list(block_shape)
    return desc


def _launch_fp8_tma_gemm(A, B, C, As, Bs, M, N, K, block_n, block_k):
    BLOCK_SIZE_M = 128
    a_desc = _fp8_tma_descriptor(A, M, K, (BLOCK_SIZE_M, block_k))
    b_desc = _fp8_tma_descriptor(B, N, K, (block_n, block_k))
    c_desc = _fp8_tma_descriptor(C, M, N, (BLOCK_SIZE_M, block_n))
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, block_n),)
    _w8a8_block_fp8_matmul_tma[grid](
        a_desc,
        b_desc,
        c_desc,
        As,
        Bs,
        M,
        N,
        K,
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = block_n,
        BLOCK_SIZE_K = block_k,
        GROUP_SIZE_M = 8,
        NUM_STAGES = 4,
        OUT_DTYPE = _TL_OUT_DTYPES[C.dtype],
        num_warps = 8,
        enable_fp_fusion = True,
    )


def _fp8_gemm_must_raise(error):
    return (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or type(error).__module__.startswith(("torch._dynamo", "torch._inductor"))
        or torch.compiler.is_compiling()
    )


_FP8_TMA_SELF_CHECKED = {}


def _fp8_tma_gemm_self_check(index):
    # Once per device: the TMA kernel must reproduce the pointer kernel bit for bit (ragged M, K tail).
    from torch.utils._python_dispatch import _disable_current_modes

    device = torch.device("cuda", index)
    try:
        with _disable_current_modes(), torch.no_grad():
            g = torch.Generator(device = device).manual_seed(3407)
            M, N, K = 200, 256, 400
            A = (torch.randn(M, K, device = device, generator = g) * 4).to(torch.float8_e4m3fn)
            B = (torch.randn(N, K, device = device, generator = g) * 4).to(torch.float8_e4m3fn)
            As = torch.rand(M, 4, device = device, generator = g) + 0.5
            Bs = torch.rand(2, 4, device = device, generator = g) + 0.5
            ref = torch.empty(M, N, device = device, dtype = torch.bfloat16)
            out = torch.empty_like(ref)
        with _disable_current_modes(), torch.no_grad(), _fp8_triton_device_context(A):
            _w8a8_block_fp8_matmul[(2 * 2,)](
                A,
                B,
                ref,
                As,
                Bs,
                M,
                N,
                K,
                128,
                128,
                A.stride(-2),
                A.stride(-1),
                B.stride(1),
                B.stride(0),
                ref.stride(-2),
                ref.stride(-1),
                As.stride(-2),
                As.stride(-1),
                Bs.stride(1),
                Bs.stride(0),
                BLOCK_SIZE_M = 128,
                BLOCK_SIZE_N = 128,
                BLOCK_SIZE_K = 128,
                GROUP_SIZE_M = 8,
                LONG_INDEXING = False,
                num_warps = 8,
                enable_fp_fusion = True,
            )
            _launch_fp8_tma_gemm(A, B, out, As, Bs, M, N, K, 128, 128)
            ok = torch.equal(out.view(torch.int16), ref.view(torch.int16))
    except Exception as error:
        if _fp8_gemm_must_raise(error):
            raise
        _fp8_tma_gemm_disable(f"{type(error).__name__}: {error}")
        return False
    if not ok:
        _fp8_tma_gemm_disable("self check mismatch")
    return ok


@_opaque_under_compile(
    "fp8_block_matmul_triton",
    lambda A, B, As, Bs, block_size, output_dtype = torch.float32: A.new_empty(
        A.shape[:-1] + (B.shape[0],), dtype = output_dtype
    ),
)
def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-wise FP8 matmul."""
    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    N, K = B.shape
    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    M = A.numel() // A.shape[-1]
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype = output_dtype)

    BLOCK_SIZE_M = 128
    if M < BLOCK_SIZE_M:
        BLOCK_SIZE_M = max(triton.next_power_of_2(M), 16)
    BLOCK_SIZE_K, BLOCK_SIZE_N = block_k, block_n

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    LONG_INDEXING = long_indexing(A, B, C)
    with _fp8_triton_device_context(A):
        if _fp8_tma_gemm_ok(A, B, C, BLOCK_SIZE_M, block_n, block_k, LONG_INDEXING):
            try:
                _launch_fp8_tma_gemm(A, B, C, As, Bs, M, N, K, block_n, block_k)
                return C
            except Exception as error:
                if _fp8_gemm_must_raise(error):
                    raise
                _fp8_tma_gemm_disable(f"{type(error).__name__}: {error}")
        _w8a8_block_fp8_matmul[grid](
            A,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            block_n,
            block_k,
            A.stride(-2),
            A.stride(-1),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            As.stride(-2),
            As.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
            BLOCK_SIZE_M = BLOCK_SIZE_M,
            BLOCK_SIZE_N = BLOCK_SIZE_N,
            BLOCK_SIZE_K = BLOCK_SIZE_K,
            GROUP_SIZE_M = 8,
            LONG_INDEXING = LONG_INDEXING,
            # Default 4 warps starve 128x128 tiles (8 is 1.3-2.1x faster from M >= 128); smaller decode tiles gain nothing.
            num_warps = 8 if BLOCK_SIZE_M == 128 else 4,
            enable_fp_fusion = True,
        )
    return C


# Decode-sized M: act_quant fused into a narrow-N w8a8 GEMM; the 128 x 128 tile above runs only N / 128 programs.
@triton.jit(do_not_specialize = ["M"])
def _w8a8_block_fp8_small_m(
    X,
    B,
    C,
    Bs,
    P,
    L,
    M,
    N,
    K,
    stride_xm,
    stride_bn,
    stride_cm,
    stride_Bs_n,
    stride_Bs_k,
    KB_PER_SPLIT,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(axis = 0)
    pid_k = tl.program_id(axis = 1)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] + offs_n[None, :] * stride_bn
    bs_ptrs = Bs + (offs_n // GROUP_N) * stride_Bs_n
    kb_start = pid_k * KB_PER_SPLIT
    kb_end = tl.minimum(kb_start + KB_PER_SPLIT, tl.cdiv(K, BLOCK_SIZE_K))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for kb in range(kb_start, kb_end):
        k0 = kb * BLOCK_SIZE_K
        # Same quantization as act_quant_kernel: per row and 128 block, scale max|x| / 448, 1 when all zero.
        x = tl.load(x_ptrs + k0, mask = m_mask[:, None], other = 0.0).to(tl.float32)
        a_s = tl.max(tl.abs(x), axis = 1) / 448.0
        a_s = tl.where(a_s == 0, 1.0, a_s)
        a = (x / a_s[:, None]).to(B.dtype.element_ty)
        b = tl.load(b_ptrs + k0, mask = n_mask[None, :], other = 0.0)
        b_s = tl.load(bs_ptrs + kb * stride_Bs_k, mask = n_mask, other = 0.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :]
    c_mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator.to(C.dtype.element_ty), mask = c_mask)
    else:
        # Deterministic split K: every split stores its partial, the last to arrive sums them in split order.
        p_ptrs = P + offs_m[:, None] * N + offs_n[None, :]
        tl.store(p_ptrs + pid_k * M * N, accumulator, mask = c_mask)
        tl.debug_barrier()
        arrived = tl.atomic_add(L + pid_n, 1, sem = "acq_rel")
        if arrived == SPLIT_K - 1:
            total = tl.load(p_ptrs, mask = c_mask, other = 0.0, cache_modifier = ".cg")
            for s in range(1, SPLIT_K):
                total += tl.load(p_ptrs + s * M * N, mask = c_mask, other = 0.0, cache_modifier = ".cg")
            tl.store(c_ptrs, total.to(C.dtype.element_ty), mask = c_mask)


_FP8_BLOCK_SMALL_M = os.environ.get("UNSLOTH_FP8_BLOCK_SMALL_M", "1") != "0"
_SMALL_M_MAX = 16


# Fitted on full config sweeps (L4, RTX PRO 6000; checked on B200): the widest N tile that still gives
# 2 tiles per SM, then split K while the grid stays within 4 programs per SM and each split keeps 8+ K blocks.
_SMALL_M_STAGES = {16: 5, 32: 4, 64: 3, 128: 4}


def _small_m_config(M, N, K, device):
    """(BLOCK_SIZE_N, SPLIT_K, num_warps, num_stages) for the fused small-M kernel; depends on N, K only."""
    sms = _sm_count(device)
    block_n = 16
    for candidate in (128, 64, 32):
        if triton.cdiv(N, candidate) >= 2 * sms:
            block_n = candidate
            break
    tiles = triton.cdiv(N, block_n)
    k_blocks = triton.cdiv(K, 128)
    split_k = 1
    while tiles * split_k * 2 <= 4 * sms and k_blocks // (split_k * 2) >= 8:
        split_k *= 2
    return block_n, split_k, 4, _SMALL_M_STAGES[block_n]


def can_use_fp8_block_small_m(X, weight, weight_scale, block_size):
    # Eager only: under torch.compile the M bound below would guard on the symbolic M and recompile.
    # torchao writes bf16 only (fp16 falls to act_quant + Triton GEMM), so fp16 keeps the fused kernel.
    if (
        not _FP8_BLOCK_SMALL_M
        or (fp8_block_matmul is not w8a8_block_fp8_matmul_triton and X.dtype != torch.float16)
        or torch.compiler.is_compiling()
    ):
        return False
    if not X.is_cuda or X.dtype not in (torch.bfloat16, torch.float16):
        return False
    if list(block_size) != [128, 128] or weight.dtype != torch.float8_e4m3fn:
        return False
    K = X.shape[-1]
    M = X.numel() // K if K else 0
    N = weight.shape[0]
    if not (1 <= M <= _SMALL_M_MAX) or weight.shape[1] != K or K % 128 or N % 16:
        return False
    # Checkpoints store weight_scale_inv in bf16 too; the kernel promotes it exactly as the w8a8 one does.
    if weight_scale.dtype not in (torch.float32, torch.bfloat16, torch.float16) or tuple(
        weight_scale.shape
    ) != (triton.cdiv(N, 128), K // 128):
        return False
    return not long_indexing(X, weight)


@_opaque_under_compile(
    "fp8_block_small_m_linear",
    lambda X, weight, weight_scale: X.new_empty(X.shape[:-1] + (weight.shape[0],)),
)
def fp8_block_small_m_linear(X, weight, weight_scale):
    """act_quant(X) then the w8a8 block matmul in one launch, for M <= _SMALL_M_MAX rows."""
    if not X.is_contiguous():
        X = X.contiguous()
    N, K = weight.shape
    M = X.numel() // K
    out = X.new_empty(X.shape[:-1] + (N,))
    block_n, split_k, warps, stages = _small_m_config(M, N, K, X.device)
    k_blocks = triton.cdiv(K, 128)
    kb_per_split = triton.cdiv(k_blocks, split_k)
    split_k = triton.cdiv(k_blocks, kb_per_split)
    tiles = triton.cdiv(N, block_n)
    if split_k > 1:
        partial = torch.empty((split_k, M, N), device = X.device, dtype = torch.float32)
        arrivals = torch.zeros(tiles, device = X.device, dtype = torch.int32)
    else:
        partial = arrivals = out
    with _fp8_triton_device_context(X, static_device_count = True):
        _w8a8_block_fp8_small_m[(tiles, split_k)](
            X,
            weight,
            out,
            weight_scale,
            partial,
            arrivals,
            M,
            N,
            K,
            K,
            weight.stride(0),
            N,
            weight_scale.stride(0),
            weight_scale.stride(1),
            kb_per_split,
            GROUP_N = 128,
            BLOCK_SIZE_M = 16,
            BLOCK_SIZE_N = block_n,
            BLOCK_SIZE_K = 128,
            SPLIT_K = split_k,
            num_warps = warps,
            num_stages = stages,
            enable_fp_fusion = True,
        )
    return out


@_opaque_under_compile(
    "fp8_block_matmul_torchao",
    lambda A, B, As, Bs, block_size, output_dtype = torch.bfloat16: A.new_empty(
        A.shape[:-1] + (B.shape[0],), dtype = output_dtype
    ),
)
def torchao_block_matmul(
    act_q: torch.Tensor,
    weight_q: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # torchao's kernel indexes in int32 too, so route launches past 2**31 elements to the int64 one.
    # It also reads the weight scale with block_size[1] along N, writes bf16 only (fp16 outputs would
    # round through bf16), and is slower on some GPUs (_prefer_triton_block_matmul_at).
    # Empty inputs (an MoE expert that got no tokens) too: torchao buckets M with log2(M).
    if (
        act_q.numel() == 0
        or output_dtype != torch.bfloat16
        or long_indexing(act_q, weight_q)
        or (act_q.numel() // act_q.shape[-1] * weight_q.shape[0] >= 2**31)
        or block_size[0] != block_size[1]
        or _prefer_triton_block_matmul_at(act_q.device, act_q.numel() // act_q.shape[-1])
    ):
        return w8a8_block_fp8_matmul_triton(
            act_q, weight_q, act_scale, weight_scale, block_size, output_dtype = output_dtype
        )
    with _fp8_triton_device_context(act_q):
        out = torchao_blockwise_gemm(
            act_q.contiguous(),
            act_scale.contiguous(),
            weight_q.contiguous(),
            weight_scale.contiguous(),
            block_size = block_size[1],
        )
    return out.to(output_dtype)


def _prefer_triton_block_matmul_at(device, M):
    # Outputs match bit for bit, so this is speed only (torchao 0.18, Llama-8B shapes):
    # sm_89 (L4): torchao is 1.73x slower than the pointer kernel at K = 14336, a net loss per layer.
    # sm_100: above 64 rows w8a8_block_fp8_matmul_triton takes BLOCK_SIZE_M = 128 and the TMA kernel
    # (~2x torchao); at 64 rows or fewer it runs the pointer kernel, which torchao beats by 1.1-2.3x.
    if device.type != "cuda" or torch.version.hip is not None:
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if tuple(torch.cuda.get_device_capability(index)) == (8, 9):
        return True
    return (
        M > 64 and _FP8_TMA_GEMM and _TensorDescriptor is not None and _fp8_tma_gemm_arch_at(index)
    )


def _select_fp8_block_matmul():
    # torchao_block_matmul hands each call to the Triton function where that is faster; pick Triton
    # outright only when every GPU prefers it at every M (sm_89), which also keeps the fused small-M
    # kernel there for bf16 (can_use_fp8_block_small_m), as torchao beats it on sm_90 / sm_100 / sm_120.
    if torchao_blockwise_gemm is None:
        return w8a8_block_fp8_matmul_triton
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count and all(
        _prefer_triton_block_matmul_at(torch.device("cuda", i), 1) for i in range(count)
    ):
        return w8a8_block_fp8_matmul_triton
    return torchao_block_matmul


# fbgemm <= 1.3.0 silently corrupts blockwise outputs for some shapes (fixed in 1.4.0), so never
# use it for block FP8. Preference: fbgemm (>=1.4.0) > torchao > triton (similar outputs/losses),
# except where the Triton kernels beat torchao (_prefer_triton_block_matmul_at).
fp8_block_matmul = _select_fp8_block_matmul()


def _rocm_fp8e4nv_kernel_supported(device):
    # AMD triton's supported_fp8_dtypes (third_party/amd/backend/compiler.py): no fp8e4nv before 3.3;
    # 3.3 adds gfx940-942 / gfx950, 3.4 gfx942 / gfx950 / gfx12*, and 3.5+ lists it on every arch
    # (software-emulated without OCP fp8 hardware; ran correctly on gfx1151 with triton 3.6).
    try:
        version = Version(triton.__version__).release[:2]
        if version >= (3, 5):
            return True
        if version < (3, 3):
            return False
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0].lower()
    except Exception:
        return False
    if version >= (3, 4):
        return arch in ("gfx942", "gfx950") or arch.startswith("gfx12")
    return arch in ("gfx940", "gfx941", "gfx942", "gfx950")


def _triton_release():
    # Compiled decode steps trace the gates below; torch 2.7 dynamo graph breaks inside packaging's Version.
    try:
        major, minor = triton.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return Version(triton.__version__).release[:2]


@functools.lru_cache(maxsize = None)
def _fp8_gemv_miscompiles(device):
    # Triton 3.3 miscompiles this GEMV on Blackwell (garbage rows at BLOCK_N 4/8); 3.4 is exact.
    if torch.version.hip is not None:
        return False
    try:
        return torch.cuda.get_device_capability(device) >= (10, 0) and _triton_release() < (3, 4)
    except Exception:
        return True


@functools.lru_cache(maxsize = None)
def _fp8_device_lacks_kernel(device):
    # A device's capability cannot change, so look it up once. Tests that simulate another
    # card must call _fp8_device_lacks_kernel.cache_clear().
    if torch.version.hip is not None:
        return not _rocm_fp8e4nv_kernel_supported(device)
    return torch.cuda.get_device_capability(device) < (8, 9)


# gfx11: byte decoder measured 7.5x faster than the torch loop on gfx1151, bit-identical for bf16 / fp32.
_HIP_FP8_BYTES_DEQUANT = os.environ.get("UNSLOTH_FP8_HIP_BYTES_DEQUANT", "1") != "0"


@functools.lru_cache(maxsize = None)
def _rocm_bytes_dequant_arch(device):
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0].lower()
    except Exception:
        return False
    return arch.startswith("gfx11")


def _hip_bytes_dequant(device, out_dtype):
    # fp16 excluded: on gfx1151 the byte decoder rounds fp16 differently from the torch loop.
    return (
        _HIP_FP8_BYTES_DEQUANT and out_dtype != torch.float16 and _rocm_bytes_dequant_arch(device)
    )


def _fp8_kernel_unsupported(tensor, kernel_dtype = None):
    """True when triton here cannot compile this fp8 dtype, so kernels taking it must be avoided.

    triton adds fp8e4nv only from sm89 on, so compare the full (major, minor): sm89 (4090,
    L40S, L4) does support it. ROCm's capability is gfx-derived, so it is gated on the gfx arch
    and triton version instead (_rocm_fp8e4nv_kernel_supported).

    kernel_dtype is the fp8 dtype the kernel really handles. It defaults to the tensor's own,
    which is right for a dequant, but a forward must pass e4m3fn: act_quant emits e4m3fn
    whatever the weight dtype is, so e5m2 weights hit the same unsupported kernel.
    """
    return (
        tensor.is_cuda
        and (tensor.dtype if kernel_dtype is None else kernel_dtype) == torch.float8_e4m3fn
        and _fp8_device_lacks_kernel(tensor.device)
    )


def _fp8_lacks_kernel_per_device():
    try:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return tuple(_fp8_device_lacks_kernel(torch.device("cuda", i)) for i in range(count))
    except Exception:
        return ()


# Read once: a capability query while tracing can graph-break a compiled region.
_FP8_DEVICE_LACKS_KERNEL = _fp8_lacks_kernel_per_device()


def _fp8_e4m3fn_unsupported_in_trace(tensor):
    if not tensor.is_cuda or tensor.dtype != torch.float8_e4m3fn:
        return False
    index = tensor.device.index if tensor.device.index is not None else torch.cuda.current_device()
    return index < len(_FP8_DEVICE_LACKS_KERNEL) and _FP8_DEVICE_LACKS_KERNEL[index]


def _native_bf16_per_device():
    try:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return tuple(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(count))
    except Exception:
        return ()


# Inductor skips (SkipFrame) any graph with a bf16 input or output on a card without native bf16 (sm_75):
# Dynamo then traces each nested frame alone, and an opaque op's body breaks fullgraph there.
_CUDA_NATIVE_BF16 = _native_bf16_per_device()


def _bf16_skipped_by_inductor(tensor, dtype):
    if dtype != torch.bfloat16 or not tensor.is_cuda or torch.version.hip is not None:
        return False
    index = tensor.device.index if tensor.device.index is not None else torch.cuda.current_device()
    return index < len(_CUDA_NATIVE_BF16) and not _CUDA_NATIVE_BF16[index]


@triton.jit
def _fp8_rowwise_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    N,
    K,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LONG_INDEXING: tl.constexpr = False,
):
    pid = tl.program_id(0)
    # Widen before the multiply: row * stride_wn overflows int32 past 2^31 weight elements.
    if LONG_INDEXING:
        pid = pid.to(tl.int64)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype = tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k, mask = offs_k < K, other = 0.0).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask = (offs_n[:, None] < N) & (offs_k[None, :] < K),
            other = 0.0,
        ).to(tl.float32)
        acc += w * x[None, :]
    s = tl.load(s_ptr + offs_n, mask = offs_n < N, other = 0.0).to(tl.float32)
    y = tl.sum(acc, axis = 1) * s
    tl.store(y_ptr + offs_n, y.to(y_ptr.dtype.element_ty), mask = offs_n < N)


# Batched decode (2..16 rows): each weight tile is read once for every row, no 16-bit weight copy.
# UNSLOTH_FP8_GEMV_ROWS=0 restores the dequantize path.
_FP8_GEMV_MAX_ROWS = 16
_FP8_GEMV_ROWS = os.environ.get("UNSLOTH_FP8_GEMV_ROWS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


@triton.jit(do_not_specialize = ["M"])
def _fp8_rowwise_gemv_rows_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    p_ptr,
    M,
    N,
    K,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    LONG_INDEXING: tl.constexpr = False,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    if LONG_INDEXING:
        pid = pid.to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    k_per = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
    k_lo = pid_k * k_per
    for k0 in range(k_lo, tl.minimum(k_lo + k_per, K), BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask = (offs_m[:, None] < M) & (offs_k[None, :] < K),
            other = 0.0,
        )
        # e4m3 values are exact in bf16 / fp16, so the dot sees the stored weights unrounded.
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask = (offs_n[:, None] < N) & (offs_k[None, :] < K),
            other = 0.0,
        ).to(x.dtype)
        acc = tl.dot(x, tl.trans(w), acc, input_precision = "ieee", out_dtype = tl.float32)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        s = tl.load(s_ptr + offs_n, mask = offs_n < N, other = 0.0).to(tl.float32)
        y = acc * s[None, :]
        tl.store(
            y_ptr + offs_m[:, None] * N + offs_n[None, :], y.to(y_ptr.dtype.element_ty), mask = mask
        )
    else:
        # fp32 partials, summed in a fixed order by the reduce kernel: deterministic, no atomics.
        tl.store(p_ptr + (pid_k * M + offs_m[:, None]) * N + offs_n[None, :], acc, mask = mask)


@triton.jit(do_not_specialize = ["M"])
def _fp8_rowwise_gemv_rows_reduce(
    p_ptr, s_ptr, y_ptr, M, N, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), dtype = tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(p_ptr + k * M * N + offs, mask = mask, other = 0.0)
    s = tl.load(s_ptr + offs % N, mask = mask, other = 0.0).to(tl.float32)
    tl.store(y_ptr + offs, (acc * s).to(y_ptr.dtype.element_ty), mask = mask)


def _fp8_rows_config(N, K, dtype):
    """(BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) from a sweep; split K only for long K or narrow N."""
    if _FP8_ROWS_CONFIG_OVERRIDE is not None:
        return _FP8_ROWS_CONFIG_OVERRIDE
    if K >= 8192:
        return (64 if dtype == torch.float16 else 32), 128, 8, 4, 4
    if N <= 1024 and K >= 4096:
        return 32, 128, 8, 4, 4
    block_n = 16 if N <= 2048 else (64 if N >= 12288 and dtype == torch.float16 else 32)
    return block_n, 256, 1, 4, 4


_FP8_ROWS_CONFIG_OVERRIDE = None


def _fp8_rowwise_gemv_rows(X, weight, scale, out, rows):
    N, K = weight.shape
    block_n, block_k, split_k, warps, stages = _fp8_rows_config(N, K, X.dtype)
    partials = (
        torch.empty((split_k, rows, N), device = X.device, dtype = torch.float32)
        if split_k > 1
        else out
    )
    with _fp8_triton_device_context(X, static_device_count = True):
        _fp8_rowwise_gemv_rows_kernel[(triton.cdiv(N, block_n), split_k)](
            X,
            weight,
            scale,
            out,
            partials,
            rows,
            N,
            K,
            weight.stride(0),
            BLOCK_M = _FP8_GEMV_MAX_ROWS,
            BLOCK_N = block_n,
            BLOCK_K = block_k,
            SPLIT_K = split_k,
            LONG_INDEXING = long_indexing(weight),
            num_warps = warps,
            num_stages = stages,
            enable_fp_fusion = True,
        )
        if split_k > 1:
            _fp8_rowwise_gemv_rows_reduce[(triton.cdiv(rows * N, 1024),)](
                partials, scale, out, rows, N, SPLIT_K = split_k, BLOCK = 1024, enable_fp_fusion = True
            )
    return out


def can_use_fp8_rowwise_gemv(X, weight, weight_scale):
    if not (X.is_cuda and weight.is_cuda and X.dtype in (torch.bfloat16, torch.float16)):
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2 or not weight.is_contiguous():
        return False
    if X.shape[-1] != weight.shape[1]:
        return False
    rows = X.numel() // X.shape[-1]
    if rows != 1 and not (2 <= rows <= _FP8_GEMV_MAX_ROWS and _FP8_GEMV_ROWS):
        return False
    # Exactly one scale per output row: a block grid can have N elements by coincidence.
    if (
        tuple(weight_scale.shape) not in ((weight.shape[0], 1), (weight.shape[0],))
        or weight.shape[0] == 1
    ):
        return False
    if torch.is_grad_enabled() and (X.requires_grad or weight.requires_grad):
        return False
    if _fp8_kernel_unsupported(weight):
        return False
    if not _fp8_gemv_miscompiles(weight.device):
        return True
    if rows != 1:
        return False
    # Triton 3.3 is exact at BLOCK_N 1 (N <= 4096); beats dequant + matmul when N >= 256 or K <= 4096 (RTX PRO 6000).
    N, K = weight.shape
    return N <= 4096 and (N >= 256 or K <= 4096) and _triton_builds_blackwell()


@functools.lru_cache(maxsize = None)
def _triton_builds_blackwell():
    # Triton 3.2 bundles a ptxas without sm_100 / sm_120.
    try:
        return _triton_release() >= (3, 3)
    except Exception:
        return False


def fp8_rowwise_gemv(X, weight, weight_scale):
    """X @ (weight * weight_scale).T for up to _FP8_GEMV_MAX_ROWS tokens without a 16-bit weight copy."""
    N, K = weight.shape
    out = torch.empty((*X.shape[:-1], N), device = X.device, dtype = X.dtype)
    if not X.is_contiguous():
        X = X.contiguous()
    scale = weight_scale if weight_scale.is_contiguous() else weight_scale.contiguous()
    rows = X.numel() // K
    if rows != 1:
        return _fp8_rowwise_gemv_rows(X, weight, scale, out, rows)
    block_n = 1 if N <= 4096 else (4 if N <= 32768 else 8)
    with _fp8_triton_device_context(X, static_device_count = True):
        _fp8_rowwise_gemv_kernel[(triton.cdiv(N, block_n),)](
            X,
            weight,
            scale,
            out,
            N,
            K,
            weight.stride(0),
            BLOCK_N = block_n,
            BLOCK_K = 512,
            LONG_INDEXING = long_indexing(weight),
            num_warps = 2,
            enable_fp_fusion = True,
        )
    return out


# Block-FP8 decode GEMV (1-16 tokens): reads the fp8 weight once, unquantized activations.
# UNSLOTH_FP8_BLOCK_GEMV=0 disables it.
_FP8_BLOCK_GEMV = os.environ.get("UNSLOTH_FP8_BLOCK_GEMV", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
_FP8_BLOCK_GEMV_MAX_ROWS = 16


# M stays unspecialized: one compiled variant for batch 1-16.
@triton.jit(do_not_specialize = ["M", "stride_ys"])
def _fp8_block_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    K,
    K_PER_SPLIT,
    stride_xm,
    stride_wn,
    stride_sn,
    stride_sk,
    stride_ys,
    stride_ym,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LONG_INDEXING: tl.constexpr = False,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    if LONG_INDEXING:
        pid_n = pid_n.to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    mask_m = offs_m < M
    s_rows = s_ptr + (offs_n // GROUP_N) * stride_sn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    k_begin = pid_k * K_PER_SPLIT
    k_end = tl.minimum(k_begin + K_PER_SPLIT, K)
    for k0 in range(k_begin, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_end
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask = mask_m[:, None] & mask_k[None, :],
            other = 0.0,
        )
        # e4m3 values are exact in bf16 and fp16, so the dot sees the stored weight.
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask = mask_n[:, None] & mask_k[None, :],
            other = 0.0,
        ).to(x.dtype)
        s = tl.load(s_rows + (k0 // GROUP_K) * stride_sk, mask = mask_n, other = 0.0).to(tl.float32)
        acc += tl.dot(x, tl.trans(w), out_dtype = tl.float32) * s[None, :]
    y = y_ptr + pid_k * stride_ys + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask = mask_m[:, None] & mask_n[None, :])


def _fp8_block_size(weight, weight_scale):
    return list(
        getattr(weight, "block_size", None)
        or getattr(weight_scale, "block_size", None)
        or [128, 128]
    )


def _fp8_block_gemv_weight_ok(weight, weight_scale):
    # Cached per (weight, scale): 0 = never, else max decode rows.
    if (
        not weight.is_cuda
        or weight.dtype != torch.float8_e4m3fn
        or weight.dim() != 2
        or not weight.is_contiguous()
    ):
        return 0
    N, K = weight.shape
    # Only where never slower than the block GEMM (L4, RTX PRO 6000; L2 flushed and hot). Small parts cap
    # rows at 8, and 2 over 8M elements (L4 2048 x 6144: 16 rows 1.98x the GEMM's time).
    small_gpu = _sm_count(weight.device) < 100
    if 2 * N > K or (small_gpu and N * K > _FP8_BLOCK_GEMV_SMALL_GPU_MAX_ELEMS):
        return 0
    max_rows = _FP8_BLOCK_GEMV_MAX_ROWS
    if small_gpu:
        max_rows = 2 if N * K > _FP8_BLOCK_GEMV_SMALL_GPU_MAX_ELEMS // 2 else 8
    if (
        weight_scale.ndim != 2
        or weight_scale.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        or weight_scale.device != weight.device
    ):
        return 0
    block_size = _fp8_block_size(weight, weight_scale)
    if len(block_size) != 2:
        return 0
    group_n, group_k = int(block_size[0]), int(block_size[1])
    if tuple(weight_scale.shape) != (triton.cdiv(N, group_n), triton.cdiv(K, group_k)):
        return 0
    if group_n < 1 or group_k < 16 or group_k & (group_k - 1):
        return 0
    if _fp8_kernel_unsupported(weight):
        return 0
    # Triton 3.3 miscompiles fp8 GEMVs on Blackwell; keep the GEMM there.
    return 0 if _fp8_gemv_miscompiles(weight.device) else max_rows


def can_use_fp8_block_gemv(X, weight, weight_scale):
    # Not validated on ROCm yet.
    if not _FP8_BLOCK_GEMV or torch.version.hip is not None:
        return False
    cached = getattr(weight, "_unsloth_fp8_block_gemv", None)
    if cached is not None and cached[0] is weight_scale:
        weight_ok = cached[1]  # max decode rows, 0 = never
    else:
        weight_ok = _fp8_block_gemv_weight_ok(weight, weight_scale)
        if not torch.compiler.is_compiling():
            weight._unsloth_fp8_block_gemv = (weight_scale, weight_ok)
    if not weight_ok:
        return False
    if not (X.is_cuda and X.dtype in (torch.bfloat16, torch.float16)) or X.device != weight.device:
        return False
    K = weight.shape[1]
    if X.shape[-1] != K:
        return False
    rows = X.numel() // K
    if not 1 <= rows <= weight_ok or (X.dim() >= 3 and X.shape[-2] != 1):
        return False
    return not (torch.is_grad_enabled() and (X.requires_grad or weight.requires_grad))


_FP8_BLOCK_GEMV_SMALL_GPU_MAX_ELEMS = 16 * 1024 * 1024
# Read once: a device-properties query inside a compiled region graph-breaks.
_SM_COUNTS = tuple(
    torch.cuda.get_device_properties(i).multi_processor_count
    for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
)


def _sm_count(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _SM_COUNTS[index] if index < len(_SM_COUNTS) else 0


@_opaque_under_compile(
    "fp8_block_gemv",
    lambda X, weight, weight_scale, group_n, group_k: X.new_empty(
        X.shape[:-1] + (weight.shape[0],)
    ),
)
def _fp8_block_gemv(
    X: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, group_n: int, group_k: int
) -> torch.Tensor:
    N, K = weight.shape
    x2 = X.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    sms = _sm_count(weight.device)
    block_n, num_warps, num_stages, programs_per_sm = (16, 2, 4, 1) if sms < 100 else (32, 4, 3, 4)
    block_k = min(128, group_k)
    n_tiles = triton.cdiv(N, block_n)
    k_blocks = triton.cdiv(K, group_k)
    split = max(1, min(k_blocks, (programs_per_sm * sms + n_tiles - 1) // n_tiles, 16))
    k_per_split = triton.cdiv(k_blocks, split) * group_k
    split = triton.cdiv(K, k_per_split)
    if split == 1:
        y = torch.empty((1, M, N), device = X.device, dtype = X.dtype)
    else:
        y = torch.empty((split, M, N), device = X.device, dtype = torch.float32)
    scale = weight_scale if weight_scale.stride(-1) == 1 else weight_scale.contiguous()
    with _fp8_triton_device_context(X, static_device_count = True):
        _fp8_block_gemv_kernel[(n_tiles, split)](
            x2,
            weight,
            scale,
            y,
            M,
            N,
            K,
            k_per_split,
            x2.stride(0),
            weight.stride(0),
            scale.stride(0),
            scale.stride(1),
            y.stride(0),
            y.stride(1),
            GROUP_N = group_n,
            GROUP_K = group_k,
            BLOCK_M = 16,
            BLOCK_N = block_n,
            BLOCK_K = block_k,
            LONG_INDEXING = long_indexing(weight),
            num_warps = num_warps,
            num_stages = num_stages,
            enable_fp_fusion = True,
        )
    out = y[0] if split == 1 else y.sum(0).to(X.dtype)
    return out.reshape(*X.shape[:-1], N)


def fp8_block_gemv(X, weight, weight_scale):
    """X @ dequant(weight, weight_scale).T for 1-16 decode tokens, activations unquantized."""
    group_n, group_k = (int(b) for b in _fp8_block_size(weight, weight_scale))
    return _fp8_block_gemv(X, weight, weight_scale, group_n, group_k)


def _with_fp8_block_gemv(block_forward):
    @functools.wraps(block_forward)
    def forward(X, weight, weight_scale):
        if can_use_fp8_block_gemv(X, weight, weight_scale):
            return fp8_block_gemv(X, weight, weight_scale)
        return block_forward(X, weight, weight_scale)

    return forward


def _fp8_block_small_m_first(X, weight, weight_scale):
    # 100+ SM parts: the fused small-M kernel is faster, so try it first (eager only; compiled uses the GEMV).
    if not (torch.is_grad_enabled() and X.requires_grad):
        # torchao selected: GEMV first (skips act_quant, beat the fused kernel at K = 14336 on sm_120).
        gemv_first = fp8_block_matmul is not w8a8_block_fp8_matmul_triton
        if gemv_first and can_use_fp8_block_gemv(X, weight, weight_scale):
            return fp8_block_gemv(X, weight, weight_scale)
        if (
            can_use_fp8_block_small_m(
                X, weight, weight_scale, _fp8_block_size(weight, weight_scale)
            )
            and weight.is_contiguous()
            and not _fp8_kernel_unsupported(weight, torch.float8_e4m3fn)
        ):
            return fp8_block_small_m_linear(X, weight, weight_scale)
        if not gemv_first and can_use_fp8_block_gemv(X, weight, weight_scale):
            return fp8_block_gemv(X, weight, weight_scale)
    return fp8_torch_block_quant_forward(X, weight, weight_scale)


# Expanding the scale over the whole weight needs two m*n float32 temporaries, ~6x the triton
# kernel's peak; chunk by block-rows so the scratch stays bounded. Values are identical.
_DEQUANT_CHUNK_ELEMS = 8 * 1024 * 1024


def _torch_blockwise_dequant(weight, weight_scale, block_size, out_dtype):
    # no_grad like the triton kernel; not compiled, since per-chunk graphs were slower than eager.
    with torch.no_grad():
        m, n = weight.shape
        out = torch.empty(m, n, dtype = out_dtype, device = weight.device)
        rows = max(
            block_size[0], (_DEQUANT_CHUNK_ELEMS // max(n, 1)) // block_size[0] * block_size[0]
        )
        for i in range(0, m, rows):
            j = min(i + rows, m)
            s = weight_scale[i // block_size[0] : -(-j // block_size[0])]
            s = s.repeat_interleave(block_size[0], 0)[: j - i]
            s = s.repeat_interleave(block_size[1], 1)[:, :n]
            out[i:j] = weight[i:j].to(torch.float32) * s
        return out


@triton.jit
def _e4m3fn_byte_to_float(b):
    # No fp8e4nv before sm_89: decode E4M3FN bytes exactly (bias 7, exp 0 subnormal, 0x7F/0xFF NaN).
    exp = (b >> 3) & 15
    man = b & 7
    normal = (((exp + 120) << 23) | (man << 20)).to(tl.float32, bitcast = True)
    value = tl.where(exp == 0, man.to(tl.float32) * 0.001953125, normal)
    value = tl.where((b & 127) == 127, float("nan"), value)
    # Sign as a bit, not a negation: -x can lower to 0 - x, which turns code 0x80 (-0.0) into +0.0.
    return (value.to(tl.int32, bitcast = True) | ((b & 128) << 24)).to(tl.float32, bitcast = True)


@triton.jit
def _e4m3fn_bytes_dequant_kernel(
    x_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_sm,
    stride_sn,
    BLOCK_M_Q: tl.constexpr,
    BLOCK_N_Q: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    EXACT_F16: tl.constexpr = False,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    if LONG_INDEXING:
        pid_r = pid_r.to(tl.int64)
        pid_c = pid_c.to(tl.int64)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r[:, None] < M) & (c[None, :] < N)
    b = tl.load(x_ptr + r[:, None] * stride_xm + c[None, :] * stride_xn, mask = mask, other = 0)
    # Per-element scale gather: a tile spans several quant blocks (also covers ragged / rectangular ones).
    s = tl.load(
        s_ptr + (r // BLOCK_M_Q)[:, None] * stride_sm + (c // BLOCK_N_Q)[None, :] * stride_sn,
        mask = mask,
        other = 0.0,
    ).to(tl.float32)
    # One fp32 multiply then one round to the output dtype, as the torch fallback does.
    y = _e4m3fn_byte_to_float(b.to(tl.int32)) * s
    if EXACT_F16:
        y = _f32_to_f16_rne(y)
    tl.store(y_ptr + r[:, None] * N + c[None, :], y.to(y_ptr.dtype.element_ty), mask = mask)


# (rows, cols, num_warps) per program, from a tile sweep on T4 and A100.
_BYTES_DEQUANT_TILE = (8, 512, 4)


@triton.jit
def _e4m3fn_scale_dequant_kernel(
    x_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    stride_s,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_ON_COLS: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    # Scale per tensor (stride_s 0), row or column. ROUND_SCALE mirrors x.to(dtype) * s.to(dtype) (fp16 keeps
    # the fp32 scale); e4m3 times a bf16 scale is exact in fp32, so one rounding gives torch's bits.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if LONG_INDEXING:
        pid_m = pid_m.to(tl.int64)
        pid_n = pid_n.to(tl.int64)
    r = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (r[:, None] < M) & (c[None, :] < N)
    offs = r[:, None] * N + c[None, :]
    x = _e4m3fn_byte_to_float(tl.load(x_ptr + offs, mask = mask, other = 0).to(tl.int32))
    if SCALE_ON_COLS:
        s = tl.load(s_ptr + c * stride_s, mask = c < N, other = 0.0)
    else:
        s = tl.load(s_ptr + r * stride_s, mask = r < M, other = 0.0)
    if ROUND_SCALE:
        s = s.to(y_ptr.dtype.element_ty)
    s = s.to(tl.float32)
    y = x * (s[None, :] if SCALE_ON_COLS else s[:, None])
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask = mask)


_SCALE_DEQUANT_TILE = (16, 256, 4)


@_opaque_under_compile(
    "fp8_e4m3fn_bytes_dequant",
    lambda weight, weight_scale, block_m, block_n, dtype: torch.empty(
        weight.shape, dtype = dtype, device = weight.device
    ),
)
def _e4m3fn_bytes_dequant(
    weight: torch.Tensor, weight_scale: torch.Tensor, block_m: int, block_n: int, dtype: torch.dtype
) -> torch.Tensor:
    # Raw bytes, as triton has no fp8e4nv below sm89; bit-identical to _torch_blockwise_dequant.
    m, n = weight.shape
    out = torch.empty((m, n), dtype = dtype, device = weight.device)
    if out.numel() == 0:
        return out
    x = weight.view(torch.uint8)
    BLOCK_R, BLOCK_C, num_warps = _BYTES_DEQUANT_TILE
    grid = (triton.cdiv(m, BLOCK_R), triton.cdiv(n, BLOCK_C))
    with _fp8_triton_device_context(weight):
        _e4m3fn_bytes_dequant_kernel[grid](
            x,
            weight_scale,
            out,
            m,
            n,
            x.stride(0),
            x.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            BLOCK_M_Q = block_m,
            BLOCK_N_Q = block_n,
            BLOCK_R = BLOCK_R,
            BLOCK_C = BLOCK_C,
            LONG_INDEXING = long_indexing(x, weight_scale, out),
            EXACT_F16 = _fp8_exact_f16_round(dtype),
            enable_fp_fusion = True,
            num_warps = num_warps,
        )
    return out


def _blockwise_weight_dequant_any_shape(weight, weight_scale, block_size, out_dtype):
    """Blockwise fp8 weight dequant for any shape: triton when the weight tiles
    evenly into block_size and this GPU can compile its fp8 dtype, else a
    torch-native per-block scale expansion."""
    m, n = weight.shape
    if weight_scale.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        weight_scale = weight_scale.to(torch.float32)  # e.g. float8_e8m0fnu scales break triton
    # Per-tensor quant: expand scalar to (ceil(m/128), ceil(n/128)) block shape
    if weight_scale.numel() == 1:
        # Per-tensor scale: the normal forward stashes the un-expanded scalar, which repeat_interleave
        # cannot grow to (m, n).
        return (weight.to(torch.float32) * weight_scale.float()).to(out_dtype)
    # An fp8 dtype this device cannot compile: fall back instead of raising from the fallback.
    kernel_fp8_unsupported = _fp8_kernel_unsupported(weight)
    if (
        m % block_size[0] != 0
        or n % block_size[1] != 0
        or block_size[0] != block_size[1]
        or kernel_fp8_unsupported
    ):
        # Uneven tiling, rectangular blocks, or an uncompilable fp8 dtype: the triton kernel uses one
        # BLOCK_SIZE for both axes, so it mis-indexes when block_size[0] != [1].
        if (
            weight.is_cuda
            and weight.dtype == torch.float8_e4m3fn
            and weight_scale.ndim == 2
            and (torch.version.hip is None or _hip_bytes_dequant(weight.device, out_dtype))
        ):
            return _e4m3fn_bytes_dequant(
                weight, weight_scale, int(block_size[0]), int(block_size[1]), out_dtype
            )
        return _torch_blockwise_dequant(weight, weight_scale, block_size, out_dtype)
    # Even tiling with square blocks: pass the real block size, since weight_dequant would silently
    # default to 128 and dequantize wrongly.
    return weight_dequant_block(weight, weight_scale, block_size = block_size[0], dtype = out_dtype)


class FP8BlockQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, weight_scale):
        m, n = weight.shape

        if weight_scale.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            # Upcast (e.g. e8m0) returns a fresh tensor and drops any Python attribute, so carry block_size
            # across the cast.
            _scale_block_size = getattr(weight_scale, "block_size", None)
            weight_scale = weight_scale.to(torch.float32)  # e8m0 scales break triton dtype mapping
            if _scale_block_size is not None:
                weight_scale.block_size = _scale_block_size

        # Original scale, saved for backward before any transformation
        original_weight_scale = weight_scale

        if weight_scale.numel() == 1:
            block_size = [128, 128]
            num_blocks_m = triton.cdiv(m, block_size[0])
            num_blocks_n = triton.cdiv(n, block_size[1])
            weight_scale = weight_scale.expand(num_blocks_m, num_blocks_n).contiguous()
        else:
            # Block quantization path
            p, q = weight_scale.shape
            block_size = getattr(weight, "block_size", None) or getattr(
                weight_scale, "block_size", [128, 128]
            )
            assert block_size is not None, "block_size is not set"
            if triton.cdiv(m, block_size[0]) != p or triton.cdiv(n, block_size[1]) != q:
                if triton.cdiv(m, block_size[0]) == q and triton.cdiv(n, block_size[1]) == p:
                    weight_scale = weight_scale.T
                    original_weight_scale = weight_scale
                else:
                    raise ValueError(
                        f"Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {block_size}"
                    )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        if X.shape[-1] % block_size[1] != 0 or _fp8_kernel_unsupported(weight, torch.float8_e4m3fn):
            # Odd hidden dim or pre-sm89 fp8 skips act_quant and gemm: dequant plus matmul, keeping the un-expanded scale
            W_deq = _blockwise_weight_dequant_any_shape(
                weight, original_weight_scale, block_size, X.dtype
            )
            ctx.weight = weight
            ctx.weight_scale = original_weight_scale
            ctx.block_size = block_size
            output = torch_matmul(X, W_deq.T)
            return output if output.dtype == X.dtype else output.to(X.dtype)

        if can_use_fp8_block_small_m(X, weight, weight_scale, block_size):
            output = fp8_block_small_m_linear(X, weight, weight_scale)
        else:
            qinput, scale = act_quant(X, block_size[1])
            output = fp8_block_matmul(
                qinput,
                weight,
                scale,
                weight_scale,
                block_size,
                output_dtype = X.dtype,
            )
        ctx.weight = weight
        ctx.weight_scale = original_weight_scale
        ctx.block_size = block_size
        # No no-op .to(): torch 2.11's compiled autograd.Function returns zero dX when the output aliases an intermediate.
        return output if output.dtype == X.dtype else output.to(X.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = _blockwise_weight_dequant_any_shape(
            ctx.weight, ctx.weight_scale, ctx.block_size, grad_output.dtype
        )
        grad_X = torch_matmul(grad_output, W_deq)
        del W_deq
        return grad_X, None, None


# Not torch.compiled: after a model's second weight shape, automatic dynamic shapes recompile the
# user Triton kernel into a version 40-80x slower than eager (the only work here is that kernel).
def fp8_torch_block_quant_forward(X, weight, weight_scale):
    if not (torch.is_grad_enabled() and X.requires_grad):
        # Nothing to differentiate (decode, no_grad): skip the autograd.Function's per-call host cost.
        return FP8BlockQuantLinear.forward(_NO_CTX(), X, weight, weight_scale)
    return FP8BlockQuantLinear.apply(X, weight, weight_scale)


class _NO_CTX:
    pass


def _quantize_fp8_per_row(x, scale_ub = None):
    # Same as FBGEMM quantize_fp8_per_row: scale = max(min(row absmax, scale_ub) / 448, 1 / (448 * 512)), saturating cast.
    row_max = x.abs().amax(dim = -1, keepdim = True).float()
    if scale_ub is not None:
        row_max = torch.minimum(row_max, scale_ub.float())
    scale = torch.clamp(row_max / 448.0, min = 1.0 / (448.0 * 512.0))
    return (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def _rowwise_gemm_works(gemm, device):
    x = torch.ones(128, 128, dtype = torch.float8_e4m3fn, device = device)
    try:
        return bool((gemm(x) == 128).all())
    except Exception:
        return False


@functools.lru_cache(maxsize = None)
def _probe_fp8_rowwise_backend(device):
    """Fastest rowwise FP8 GEMM that runs correctly on this device: FBGEMM ships no sm120 kernels ("cutlass
    cannot initialize" on RTX PRO 6000 / 5090), where torch._scaled_mm keeps FBGEMM's numerics; else dequant."""
    ones = lambda n: torch.ones(n, dtype = torch.float32, device = device)
    if _has_fbgemm_rowwise():
        from unsloth.import_fixes import suppress_cuda_printf
        with suppress_cuda_printf():
            if _rowwise_gemm_works(
                lambda x: torch.ops.fbgemm.f8f8bf16_rowwise(x, x, ones(128), ones(128)), device
            ):
                return "fbgemm"
    if _rowwise_gemm_works(
        lambda x: torch._scaled_mm(
            x, x.t(), scale_a = ones((128, 1)), scale_b = ones((1, 128)), out_dtype = torch.bfloat16
        ),
        device,
    ):
        return "scaled_mm"
    return "dequant"


def _fp8_rowwise_backend(device):
    return _probe_fp8_rowwise_backend(device)


if hasattr(torch._dynamo, "assume_constant_result"):
    # Compiled callers run the probe once at trace time and bake the answer in; an lru_cache wrapper would be traced.
    _fp8_rowwise_backend = torch._dynamo.assume_constant_result(_fp8_rowwise_backend)


class FbgemmFp8Linear_matmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        weight_scale,
        bias = None,
        backend = None,
    ):
        if backend is None:
            backend = _fp8_rowwise_backend(x.device)
        rowwise = weight.shape[0] == weight_scale.shape[0] and not _is_transposed_view(weight)
        if (
            backend == "fbgemm"
            and rowwise
            and (weight.shape[0] % 8 == 0 and weight.shape[1] % 8 == 0)
        ):
            # The kernel needs weight dims divisible by 8 (else "cutlass cannot implement"), and padding plus
            # f8f8bf16 is slower than dequant plus bf16 matmul.

            # quantize_fp8_per_row squashes leading dims; save the shape first.
            output_shape = (*x.shape[:-1], -1)
            # x_quantized/x_scale may land on a different device than x (FBGEMM quantize.cu#L1237); moving them
            # here produces gibberish, so move the output instead.
            x_quantized, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                x.view(-1, x.shape[-1]).contiguous(),
                scale_ub = getattr(weight, "input_scale_ub", None),
            )
            weight_scale_float32 = weight_scale.to(torch.float32)

            if not weight.is_contiguous():
                weight = weight.contiguous()
            if not weight_scale.is_contiguous():
                weight_scale = weight_scale.contiguous()

            output = torch.ops.fbgemm.f8f8bf16_rowwise(
                x_quantized, weight, x_scale, weight_scale_float32, use_fast_accum = True
            )
            output = output + bias if bias is not None else output
            # Move output back to x's device (the move-input path produced gibberish)
            output = output.to(x.device, x.dtype)
            output = output.reshape(output_shape)
            del x_quantized, x_scale
        elif (
            backend == "scaled_mm"
            and rowwise
            and (weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0)
            and x.shape[-1] == weight.shape[1]
            # Decode-sized calls: compiled dequant + GEMV beats quantizing the activation.
            and x.numel() > 64 * x.shape[-1]
        ):
            output_shape = (*x.shape[:-1], -1)
            x_quantized, x_scale = _quantize_fp8_per_row(
                x.reshape(-1, x.shape[-1]), getattr(weight, "input_scale_ub", None)
            )
            if not weight.is_contiguous():
                weight = weight.contiguous()
            output = torch._scaled_mm(
                x_quantized,
                weight.t(),
                scale_a = x_scale,
                scale_b = weight_scale.to(torch.float32).view(1, -1),
                out_dtype = x.dtype,
                use_fast_accum = True,
            )
            output = output + bias if bias is not None else output
            # FbgemmFp8Linear keeps its bias in float32; only cast when it promoted (a no-op .to() aliases).
            if output.dtype != x.dtype:
                output = output.to(x.dtype)
            output = output.reshape(output_shape)
            del x_quantized, x_scale
        elif weight_scale.shape[0] in (weight.shape[0], weight.shape[1]):
            # Transposed, non-divisible-by-8 (Qwen 2.5 VL 7B 3420x1280), or no working FP8 GEMM here: dequant.
            W_deq = weight_dequant(weight, weight_scale, x.dtype).T
            output = torch_matmul(x, W_deq)
            output = output + bias if bias is not None else output
            if output.dtype != x.dtype:
                output = output.to(x.dtype)
            del W_deq
        else:
            raise ValueError(
                f"Shapes are incompatible {weight.shape = }, {weight_scale.shape = }, {x.shape = }"
            )

        ctx.weight = weight
        ctx.weight_scale = weight_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = weight_dequant(ctx.weight, ctx.weight_scale, grad_output.dtype)
        grad_X = torch_matmul(grad_output, W_deq)
        del W_deq
        return grad_X, None, None, None, None


@torch_compile
def fbgemm_fp8_linear(
    X,
    weight,
    weight_scale,
    bias = None,
):
    return FbgemmFp8Linear_matmul.apply(
        X, weight, weight_scale, bias, _fp8_rowwise_backend(X.device)
    )


class FP8_fbgemm_block_linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        weight,
        weight_scale,
        bias = None,
    ):
        orig_shape = X.shape
        X = X.reshape(-1, X.shape[-1])  # reshape, not view: X may be strided

        bs_n, bs_k = getattr(weight, "block_size", None) or getattr(
            weight_scale, "block_size", [128, 128]
        )
        bs_m = bs_n

        if weight_scale.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            weight_scale = weight_scale.to(torch.float32)  # e8m0 scales break triton

        m, n = weight.shape
        p, q = weight_scale.shape if weight_scale.ndim == 2 else (0, 0)
        fits = triton.cdiv(m, bs_n) == p and triton.cdiv(n, bs_k) == q
        # numel() == 1 like fp8_linear and the dequant helper, but a (1, 1) that really is this weight's
        # grid still fits, so only a true per-tensor scale falls back.
        per_tensor = weight_scale.numel() == 1 and not fits
        if not per_tensor:
            fits_transposed = triton.cdiv(n, bs_n) == p and triton.cdiv(m, bs_k) == q
            # Backward passes W.t() (fast_lora's downW.t()), so its block axes swap too; at m == n only the
            # stride tells the two grids apart. Transpose the scale, not W, which X still has to matmul.
            if fits_transposed and (not fits or (bs_n != bs_k and weight.stride(0) == 1)):
                weight_scale = weight_scale.T
                bs_n, bs_k = bs_k, bs_n
            elif not fits:
                raise ValueError(
                    f"Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {bs_n, bs_k}"
                )

        # f8f8bf16_blockwise takes only 128x128x128 blocks, float32 scale grids and (measured on H800 /
        # fbgemm 1.4.0) in_features % 16 == 0, out_features % 8 == 0; anything else raises.
        kernel_supported = (
            not per_tensor
            # triton_quantize_fp8_block below emits fp8e4nv, so pre-sm89 needs the dequant path too.
            and not _fp8_kernel_unsupported(weight, torch.float8_e4m3fn)
            and weight_scale.dtype == torch.float32
            and (bs_m, bs_n, bs_k) == (128, 128, 128)
            and X.shape[-1] % 16 == 0
            and m % 8 == 0
        )
        if not kernel_supported:
            W_deq = _blockwise_weight_dequant_any_shape(weight, weight_scale, [bs_n, bs_k], X.dtype)
            output = torch_matmul(X, W_deq.T)
            output = output + bias if bias is not None else output
            output = output.view(*orig_shape[:-1], -1)
            del W_deq
            ctx.weight = weight
            ctx.weight_scale = weight_scale
            ctx.block_size = [bs_m, bs_n, bs_k]
            return output

        with _fp8_triton_device_context(X):
            # X is (tokens, K): its blocks are (bs_m, bs_k), not (bs_m, bs_n).
            xq, xs = triton_quantize_fp8_block(X, bs_m, bs_k, None)
        # f8f8bf16 always returns bf16; cast so both branches return X.dtype.
        output = torch.ops.fbgemm.f8f8bf16_blockwise(
            xq, weight.contiguous(), xs, weight_scale.contiguous(), bs_m, bs_n, bs_k
        ).to(X.dtype)
        output = output + bias if bias is not None else output

        output = output.view(*orig_shape[:-1], -1)

        del xq
        del xs

        ctx.weight = weight
        ctx.weight_scale = weight_scale
        ctx.block_size = [bs_m, bs_n, bs_k]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # weight_dequant assumes 128x128 and would mis-scale other block sizes.
        bs_m, bs_n, bs_k = ctx.block_size
        W_deq = _blockwise_weight_dequant_any_shape(
            ctx.weight, ctx.weight_scale, [bs_n, bs_k], grad_output.dtype
        )
        grad_X = torch_matmul(grad_output, W_deq)
        del W_deq
        return grad_X, None, None, None, None


@torch_compile
def fp8_fbgemm_block_linear(
    X,
    weight,
    weight_scale,
    bias = None,
):
    return FP8_fbgemm_block_linear.apply(X, weight, weight_scale, bias)


def test_has_fbgemm():
    # Probe whether the faster FBGEMM works on this GPU: RTX 4090/5090 and SM100 (Blackwell B200/B100)
    # fail with CUTLASS SM90 kernels.
    M, N, K = 128, 128, 128
    xq = torch.ones(M, K, dtype = torch.float8_e4m3fn, device = "cuda")
    wq = xq
    M, K = xq.shape
    N, _ = wq.shape
    block_scale = torch.ones(M // 128, K // 128, dtype = torch.float32, device = "cuda")
    has_fbgemm = False
    try:
        out = torch.ops.fbgemm.f8f8bf16_blockwise(xq, wq, block_scale, block_scale)
        assert torch.unique(out).item() == 128
        has_fbgemm = True
        del out
    except Exception as e:
        error_str = str(e).lower()
        # Disable FBGEMM on any CUTLASS/CUDA error (MMA, arch mismatch, launch, etc.)
        cutlass_cuda_errors = (
            "cutlass",
            "cuda error",
            "cuda runtime error",
            "no kernel image",
            "arch conditional",
            "mma instruction",
            "compute capability",
            "cute_invalid_control_path",
            "tma",
        )
        is_cutlass_cuda_error = any(err in error_str for err in cutlass_cuda_errors)

        if is_cutlass_cuda_error:
            print("Unsloth: FBGEMM on the current GPU cannot load - will switch to Triton kernels")
        # Row/channel FP8: 2D scale shaped (n, 1)
        else:
            print(
                f"Unsloth: FBGEMM on the current GPU cannot load with error = {e} - will switch to Triton kernels"
            )
        has_fbgemm = False
    del block_scale, xq
    torch.cuda.empty_cache()
    return has_fbgemm


fp8_block_quant_linear = fp8_torch_block_quant_forward
if "UNSLOTH_HAS_FBGEMM" not in os.environ:
    os.environ["UNSLOTH_HAS_FBGEMM"] = "0"
try:
    import fbgemm_gpu

    # >= 1.4.0 is fast and accurate (older versions corrupt some shapes whatever the input values) and
    # ~15% faster than torchao, but consumer GPUs fail, so probe it.
    if Version(fbgemm_gpu.__version__) >= Version("1.4.0"):
        # Suppress CUDA printf during the probe: on Blackwell (SM100) FBGEMM's SM90 CUTLASS kernel floods
        # stdout with "Arch conditional MMA" before aborting.
        from unsloth.import_fixes import suppress_cuda_printf
        with suppress_cuda_printf():
            _has_fbgemm = test_has_fbgemm()
        if _has_fbgemm:
            os.environ["UNSLOTH_HAS_FBGEMM"] = "1"
            logger.info(f"Using fbgemm_gpu block quantized FP8 matmul")
            fp8_block_quant_linear = fp8_fbgemm_block_linear
        else:
            os.environ["UNSLOTH_HAS_FBGEMM"] = "0"
except:
    pass
# Decode-sized calls take the GEMV whichever block GEMM was selected above.
if (
    fp8_block_quant_linear is fp8_torch_block_quant_forward
    and torch.version.hip is None
    and _SM_COUNTS
    and min(_SM_COUNTS) >= 100
):
    _fp8_block_small_m_first.__wrapped__ = fp8_torch_block_quant_forward
    fp8_block_quant_linear = _fp8_block_small_m_first
else:
    fp8_block_quant_linear = _with_fp8_block_gemv(fp8_block_quant_linear)


# Not torch.compiled, for the same reason as fp8_torch_block_quant_forward.
def fp8_linear(
    X,
    weight,
    weight_scale,
    bias = None,
):
    # Transposed block view (LoRA_MLP backward's downW.t()): the block kernel re-orients rectangular scale grids
    # from the shapes, but a square grid (or non-square blocks) is ambiguous, so dequantize the stored layout.
    if (
        _is_transposed_view(weight)
        and weight_scale.ndim == 2
        and weight_scale.shape[1] > 1
        and (
            weight_scale.shape[0] == weight_scale.shape[1]
            or len(set(getattr(weight_scale, "block_size", None) or [128, 128])) > 1
        )
    ):
        out = torch.matmul(X, weight_dequant(weight, weight_scale, X.dtype).t())
        return out if bias is None else out + bias
    # Per-tensor (scalar scale) or block FP8 (2D scale, multiple columns).
    if weight_scale.numel() == 1 or (weight_scale.ndim == 2 and weight_scale.shape[1] > 1):
        out = fp8_block_quant_linear(X, weight, weight_scale)
    # Row/channel FP8: 2D scale shaped (n, 1).
    else:
        out = fbgemm_fp8_linear(X, weight, weight_scale, bias)
    return out


_FP8_WEIGHT_DTYPES = tuple(
    getattr(torch, n)
    for n in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, n)
)


def module_forward_patch(forward_function, scale_attr = "weight_scale"):
    def patched_forward(self, X):
        weight = self.weight
        if weight.dtype not in _FP8_WEIGHT_DTYPES:
            # bf16 layer the skip list missed: its scale was never loaded.
            bias = self._parameters.get("bias")
            return torch.nn.functional.linear(
                X, weight.to(X.dtype), None if bias is None else bias.to(X.dtype)
            )
        weight_scale = getattr(self, scale_attr)
        # Kernels read block_size off weight/scale, else assume 128x128; tagging 128x128 would only cost a recompile.
        module_block_size = getattr(self, "block_size", None)
        if (
            module_block_size is not None
            and list(module_block_size) != [128, 128]
            and getattr(weight, "block_size", None) is None
            and getattr(weight_scale, "block_size", None) is None
        ):
            weight.block_size = list(module_block_size)
        out = forward_function(X, weight, weight_scale)
        # The kernels take no bias, so a biased Linear (Qwen2-style q/k/v) adds it here; fbgemm keeps it fp32.
        bias = self._parameters.get("bias")
        return out if bias is None else out + bias.to(out.dtype)

    return patched_forward


# Patch the forward functions of the layers (for compiled models)
if FbgemmFp8Linear is not None:
    FbgemmFp8Linear.forward = module_forward_patch(fbgemm_fp8_linear, "weight_scale")
if FP8Linear is not None:
    FP8Linear.forward = module_forward_patch(fp8_block_quant_linear, "weight_scale_inv")

# Fused grouped matmul has no backward: dequant for training, gate on self.training so
# the checkpoint no-grad forward and its recompute match.
if FP8GroupedLinear is not None:
    _fp8_grouped_forward_orig = FP8GroupedLinear.forward

    def _fp8_to_local(t):
        dt = getattr(getattr(torch, "distributed", None), "tensor", None)
        DTensor = getattr(dt, "DTensor", None) if dt is not None else None
        return t.to_local() if DTensor is not None and isinstance(t, DTensor) else t

    def _fp8_grouped_dequant(weight, scale_inv, block_size, dtype):
        # Honor the layer's block size; weight_dequant would assume 128 and mis-scale.
        if block_size is not None and len(block_size) == 2:
            return _blockwise_weight_dequant_any_shape(weight, scale_inv.float(), block_size, dtype)
        return weight_dequant(weight, scale_inv.float()).to(dtype)

    class _FP8GroupedMM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, scale_inv, n_groups, block_size, bias):
            weight, scale_inv = _fp8_to_local(weight), _fp8_to_local(scale_inv)
            hidden = x.shape[-1]
            W = _fp8_grouped_dequant(weight, scale_inv, block_size, x.dtype)
            out_per = W.shape[0] // n_groups
            xg = x.reshape(-1, n_groups, hidden).transpose(0, 1)
            y = torch.bmm(xg, W.view(n_groups, out_per, hidden).transpose(1, 2))
            y = y.transpose(0, 1).reshape(*x.shape[:-2], n_groups, out_per)
            if bias is not None:
                y = y + bias.view(n_groups, out_per)
            ctx.save_for_backward(weight, scale_inv)
            ctx.n_groups, ctx.out_per, ctx.x_shape = n_groups, out_per, x.shape
            ctx.dtype, ctx.has_bias, ctx.block_size = x.dtype, bias is not None, block_size
            return y

        @staticmethod
        def backward(ctx, grad_y):
            weight, scale_inv = ctx.saved_tensors
            ng, out_per, hidden = ctx.n_groups, ctx.out_per, ctx.x_shape[-1]
            W = _fp8_grouped_dequant(weight, scale_inv, ctx.block_size, ctx.dtype).view(
                ng, out_per, hidden
            )
            gy = grad_y.reshape(-1, ng, out_per).transpose(0, 1)
            grad_x = torch.bmm(gy, W).transpose(0, 1).reshape(ctx.x_shape)
            grad_bias = gy.sum(1).reshape(-1) if ctx.has_bias else None
            return grad_x, None, None, None, None, grad_bias

    def _fp8_grouped_forward(self, x):
        # The upstream eval path runs its own fp8e4nv kernels, so pre-sm89 has to use the
        # dequant path here too, not only while training.
        if self.weight.element_size() > 1 or (
            not self.training and not _fp8_kernel_unsupported(self.weight, torch.float8_e4m3fn)
        ):
            return _fp8_grouped_forward_orig(self, x)
        bias = self.bias if self.has_bias else None
        return _FP8GroupedMM.apply(
            x,
            self.weight,
            self.weight_scale_inv,
            self.n_groups,
            getattr(self, "block_size", None),
            bias,
        )

    FP8GroupedLinear.forward = _fp8_grouped_forward

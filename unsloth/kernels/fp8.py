# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from contextlib import nullcontext
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn import functional as F
import math
from unsloth_zoo.utils import Version
from unsloth_zoo.log import logger
from unsloth_zoo.temporary_patches.common import torch_compile

torch_matmul = torch.matmul


def _fp8_triton_device_context(tensor: torch.Tensor):
    if tensor.device.type == "cuda" and torch.cuda.device_count() > 1:
        return torch.cuda.device(tensor.device)
    if tensor.device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.device_count() > 1:
        return torch.xpu.device(tensor.device)
    return nullcontext()


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

try:
    from torchao.prototype.blockwise_fp8_inference.blockwise_quantization import (
        blockwise_fp8_gemm as torchao_blockwise_gemm,
    )
except:
    torchao_blockwise_gemm = None
    logger.info(
        "Unsloth: Could not find torchao.prototype.blockwise_fp8_inference.blockwise_quantization.blockwise_fp8_gemm"
    )


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # tl.arange is int32, so offs_m * N overflows for tensors with more than
    # 2**31 elements (e.g. flattened MoE expert stacks); index in int64.
    offs = offs_m[:, None].to(tl.int64) * N + offs_n[None, :].to(tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask = mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask = mask)


def weight_dequant_block(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    dtype = torch.bfloat16,
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
        weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE = block_size)
    return y


def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    dtype = torch.bfloat16,
):
    # Per-tensor scale: single value for entire weight matrix
    if s.numel() == 1:
        return x.to(dtype) * s.view(1, 1).to(dtype)
    # Row quantized weight: scale shape is (m, 1) or (n, 1)
    elif s.ndim == 2 and s.shape[1] == 1:
        if x.shape[0] == s.shape[0]:
            y = x.to(dtype) * s.to(dtype)
        elif x.shape[1] == s.shape[0]:
            # sometimes, this is called with the transpose of the weight. Adjust for that.
            y = x.t().to(dtype) * s.to(dtype)
            y = y.t()
        else:
            raise ValueError(f"Incompatible shapes {x.shape = }, {s.shape = }")
        return y
    # Block quantized weight: scale shape is (ceil(m/block_m), ceil(n/block_n))
    else:
        return weight_dequant_block(x, s, dtype = dtype)


# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    # All-zero row: keep scale at 1 so LoRA's zero dY doesn't become NaN
    # (a deviation from the original implementation).
    s = 1.0 if s == 0 else s
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype = torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype = torch.float32)

    def grid(meta):
        return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

    with _fp8_triton_device_context(x):
        act_quant_kernel[grid](x, y, s, BLOCK_SIZE = block_size)
    return y, s


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py
@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
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
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask = c_mask)


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

    with _fp8_triton_device_context(A):
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
        )
    return C


def torchao_block_matmul(
    act_q: torch.Tensor,
    weight_q: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int],
    output_dtype: torch.dtype = torch.bfloat16,
):
    with _fp8_triton_device_context(act_q):
        out = torchao_blockwise_gemm(
            act_q.contiguous(),
            act_scale.contiguous(),
            weight_q.contiguous(),
            weight_scale.contiguous(),
            block_size = block_size[1],
        )
    return out.to(output_dtype)


# fbgemm <=1.3.0 causes NaNs for high X values, so never use it for block FP8.
# Preference: fbgemm (>=1.4.0) > torchao > triton (similar outputs/losses).
# torchao is ~3x faster than the triton kernel but 15-30% slower than fbgemm (H100).
fp8_block_matmul = (
    torchao_block_matmul if torchao_blockwise_gemm is not None else w8a8_block_fp8_matmul_triton
)


def _blockwise_weight_dequant_any_shape(weight, weight_scale, block_size, out_dtype):
    """Blockwise fp8 weight dequant for any shape: triton when the weight tiles
    evenly into block_size, else a torch-native per-block scale expansion."""
    m, n = weight.shape
    if weight_scale.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        weight_scale = weight_scale.to(torch.float32)  # e.g. float8_e8m0fnu scales break triton
    if weight_scale.numel() == 1:
        # Per-tensor scale: the normal forward stashes the un-expanded scalar,
        # which repeat_interleave cannot grow to (m, n). Scale directly.
        return (weight.to(torch.float32) * weight_scale.float()).to(out_dtype)
    if m % block_size[0] != 0 or n % block_size[1] != 0 or block_size[0] != block_size[1]:
        # Uneven tiling, or rectangular blocks. The triton kernel uses a single
        # BLOCK_SIZE for both axes and derives the column scale stride from it, so
        # it mis-indexes the scale when block_size[0] != block_size[1]. Expand the
        # per-block scales in torch, which handles both dimensions independently.
        s_full = weight_scale.repeat_interleave(block_size[0], 0)[:m]
        s_full = s_full.repeat_interleave(block_size[1], 1)[:, :n]
        return (weight.to(torch.float32) * s_full).to(out_dtype)
    # Even tiling with square blocks: block-quant dequant with the real block size
    # (weight_dequant would silently default to 128 and dequantize wrongly).
    return weight_dequant_block(weight, weight_scale, block_size = block_size[0], dtype = out_dtype)


class FP8BlockQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, weight_scale):
        m, n = weight.shape

        if weight_scale.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            # Upcast (e.g. e8m0) returns a fresh tensor and drops any Python
            # attribute, so carry block_size across the cast for the lookup below.
            _scale_block_size = getattr(weight_scale, "block_size", None)
            weight_scale = weight_scale.to(torch.float32)  # e8m0 scales break triton dtype mapping
            if _scale_block_size is not None:
                weight_scale.block_size = _scale_block_size

        # Original scale, saved for backward before any transformation
        original_weight_scale = weight_scale

        # Per-tensor quant: expand scalar to (ceil(m/128), ceil(n/128)) block shape
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
                    original_weight_scale = weight_scale  # Update for transposed case
                else:
                    raise ValueError(
                        f"Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {block_size}"
                    )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        if X.shape[-1] % block_size[1] != 0:
            # Hidden dim not divisible by the activation block: dequant + plain matmul.
            # Use the original (un-expanded) scale so a scalar per-tensor scale keeps
            # the fast scalar path in both forward and backward.
            W_deq = _blockwise_weight_dequant_any_shape(
                weight, original_weight_scale, block_size, X.dtype
            )
            ctx.weight = weight
            ctx.weight_scale = original_weight_scale
            ctx.block_size = block_size
            return torch_matmul(X, W_deq.T).to(X.dtype)

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
        ctx.weight_scale = original_weight_scale  # Save original for backward
        ctx.block_size = block_size
        return output.to(X.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = _blockwise_weight_dequant_any_shape(
            ctx.weight, ctx.weight_scale, ctx.block_size, grad_output.dtype
        )
        grad_X = torch_matmul(grad_output, W_deq)
        del W_deq
        return grad_X, None, None


@torch_compile
def fp8_torch_block_quant_forward(X, weight, weight_scale):
    return FP8BlockQuantLinear.apply(X, weight, weight_scale)


class FbgemmFp8Linear_matmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        weight_scale,
        bias = None,
    ):
        if weight.shape[0] == weight_scale.shape[0] and (
            weight.shape[0] % 8 == 0 and weight.shape[1] % 8 == 0
        ):
            # The kernel needs weight dims divisible by 8 (else `cutlass cannot
            # implement`). Padding + f8f8bf16 is slower than dequant + bf16 matmul,
            # so f8f8bf16_rowwise runs only for proper, divisible-by-8 shapes.

            # quantize_fp8_per_row squashes leading dims; save the shape first
            output_shape = (*x.shape[:-1], -1)
            # x_quantized/x_scale may land on a different device than x (FBGEMM
            # quantize.cu#L1237). Moving them here produces gibberish; move the
            # output instead. Compute runs on weight's device regardless.
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
            weight.shape[0] != weight_scale.shape[0] and weight.shape[1] == weight_scale.shape[0]
        ) or (weight.shape[0] % 8 != 0 or weight.shape[1] % 8 != 0):
            # Transposed weight/scale (backward dY@W) or non-divisible-by-8 shape
            # (e.g. Qwen 2.5 VL 7B gate proj 3420x1280): dequant is preferred.
            W_deq = weight_dequant(weight, weight_scale).T
            output = torch_matmul(x, W_deq)
            output = output + bias if bias is not None else output
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
        W_deq = weight_dequant(ctx.weight, ctx.weight_scale)
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
    return FbgemmFp8Linear_matmul.apply(X, weight, weight_scale, bias)


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
        X = X.view(-1, X.shape[-1])

        bs_n, bs_k = getattr(weight, "block_size", None) or getattr(
            weight_scale, "block_size", [128, 128]
        )
        bs_m = bs_n

        m, n = weight.shape
        p, q = weight_scale.shape

        if triton.cdiv(m, bs_n) != p or triton.cdiv(n, bs_k) != q:
            if triton.cdiv(m, bs_n) == q and triton.cdiv(n, bs_k) == p:
                # Backward transposes the weight; transpose the scale to match
                # (transposing the weight itself would break matmul with X).
                weight_scale = weight_scale.T
            else:
                raise ValueError(
                    f"Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {bs_n, bs_k}"
                )

        with _fp8_triton_device_context(X):
            xq, xs = triton_quantize_fp8_block(X, bs_m, bs_n, None)
        # TODO: WARNING - diverges from baseline for high X values, producing
        # gibberish / high starting loss. Do not use until resolved; kept for a
        # future headstart.
        output = torch.ops.fbgemm.f8f8bf16_blockwise(
            xq, weight.contiguous(), xs, weight_scale.contiguous(), bs_m, bs_n, bs_k
        )
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
        W_deq = weight_dequant(ctx.weight, ctx.weight_scale)
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
    # Probe whether the faster FBGEMM works on this GPU. RTX 4090/5090 and
    # SM100 (Blackwell B200/B100) fail with CUTLASS SM90 kernels.
    # [TODO] Investigate with TorchAO why FBGEMM fails on consumer GPUs
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

    # >=1.4.0 is fast and accurate (older versions NaN on high X); ~15% faster
    # than torchao. Must probe blockwise FBGEMM since consumer GPUs fail.
    if Version(fbgemm_gpu.__version__) >= Version("1.4.0"):
        # Suppress CUDA printf during probe: on Blackwell (SM100), FBGEMM's
        # SM90 CUTLASS kernel floods stdout with "Arch conditional MMA" before aborting.
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


@torch_compile
def fp8_linear(
    X,
    weight,
    weight_scale,
    bias = None,
):
    # Per-tensor (scalar scale) or block FP8 (2D scale, multiple columns)
    if weight_scale.numel() == 1 or (weight_scale.ndim == 2 and weight_scale.shape[1] > 1):
        out = fp8_block_quant_linear(X, weight, weight_scale)
    # Row/channel FP8: 2D scale shaped (n, 1)
    else:
        out = fbgemm_fp8_linear(X, weight, weight_scale, bias)
    return out


def module_forward_patch(forward_function, scale_attr = "weight_scale"):
    def patched_forward(self, X):
        return forward_function(X, self.weight, getattr(self, scale_attr))

    return patched_forward


# Patch the forward functions of the layers (for compiled models)
if FbgemmFp8Linear is not None:
    FbgemmFp8Linear.forward = module_forward_patch(fbgemm_fp8_linear, "weight_scale")
if FP8Linear is not None:
    FP8Linear.forward = module_forward_patch(fp8_block_quant_linear, "weight_scale_inv")

# FP8GroupedLinear's fused grouped matmul has no autograd formula, so training
# backward fails. In training, use a custom autograd Function: dequant the frozen
# fp8 weight for a differentiable bmm, saving only the fp8 weight + scale and
# unwrapping TP shards; eval keeps the fused kernel. Gate on self.training (not
# is_grad_enabled) so the grad-checkpoint no-grad forward and its recompute match.
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
        if self.weight.element_size() > 1 or not self.training:
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

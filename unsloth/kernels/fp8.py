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

try:
    from transformers.integrations.finegrained_fp8 import FP8Linear
except:
    FP8Linear = None
    logger.info(
        "Unsloth: FP8 models need importing FP8Linear from `transformers.integrations.finegrained_fp8` but we don't see it."
    )

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
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask = mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask = mask)


def weight_dequant_block(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, dtype = torch.bfloat16
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
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE = block_size)
    return y


def weight_dequant(x: torch.Tensor, s: torch.Tensor, dtype = torch.bfloat16):
    if s.shape[1] == 1:
        # this is row quantized weight, just simple multiplication suffices
        if x.shape[0] == s.shape[0]:
            y = x.to(dtype) * s.to(dtype)
        elif x.shape[1] == s.shape[0]:
            # sometimes, this is called with the transpose of the weight. Adjust for that.
            y = x.t().to(dtype) * s.to(dtype)
            y = y.t()
        else:
            raise ValueError(f"Incompatible shapes {x.shape = }, {s.shape = }")
        return y
    else:
        # this is block quantized weight
        return weight_dequant_block(x, s, dtype = dtype)


# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    # For a row of all zeros, lets return zeros as is
    # for LoRA, there are cases where dY has 0 in it and we should not let it be NaN
    # this is a deviation from the original implementation.
    s = 1.0 if s == 0 else s
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype = torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype = torch.float32)

    def grid(meta):
        return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

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
    """This function performs matrix multiplication with block-wise
    quantization.
    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should
        be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.
    Returns:
        torch.Tensor: The result of matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype = output_dtype)

    BLOCK_SIZE_M = 128
    if M < BLOCK_SIZE_M:
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_M = max(BLOCK_SIZE_M, 16)
    BLOCK_SIZE_K = block_k
    assert block_k % BLOCK_SIZE_K == 0
    BLOCK_SIZE_N = block_n

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

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
    out = torchao_blockwise_gemm(
        act_q.contiguous(),
        act_scale.contiguous(),
        weight_q.contiguous(),
        weight_scale.contiguous(),
        block_size = block_size[1],
    )
    return out.to(output_dtype)


# Note that older versions of fbgemm (<=1.3.0) cause numerical imprecisions resulting in NaNs especially when X has high values in it.
# So our preference order is fbgemm (>=1.4.0) > torchao > triton. All of these have similar outputs/losses. Never use fbgemm (<=1.3.0) for block quantized FP8 matmul.
# This torchao FP8 matmul seems to be ~3x faster than the w8a8_block_fp8_matmul_triton. Though torchao is 15-30% slower than fbgemm implementation (on H100 GPUs).
fp8_block_matmul = (
    torchao_block_matmul
    if torchao_blockwise_gemm is not None
    else w8a8_block_fp8_matmul_triton
)


class FP8BlockQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, weight_scale):
        # block_size = getattr(weight, 'block_size', [128,128])
        m, n = weight.shape
        p, q = weight_scale.shape
        block_size = getattr(weight, "block_size", None) or getattr(
            weight_scale, "block_size", [128, 128]
        )
        assert block_size is not None, "block_size is not set"
        if triton.cdiv(m, block_size[0]) != p or triton.cdiv(n, block_size[1]) != q:
            if (
                triton.cdiv(m, block_size[0]) == q
                and triton.cdiv(n, block_size[1]) == p
            ):
                # weights are transposed during backward pass for training :)
                # We transpose weight scale to counter that. Note that transposing weight would cause issues with matmul with input X
                weight_scale = weight_scale.T
            else:
                raise ValueError(
                    f"Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {block_size}"
                )

        if not weight.is_contiguous():
            weight = weight.contiguous()
        # this is replica of https://github.com/huggingface/transformers/blob/01c9e1ba683b3e50d7c76bf92f2d470759fd5e81/src/transformers/integrations/finegrained_fp8.py#L331-L353
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
        ctx.weight_scale = weight_scale
        ctx.block_size = block_size
        return output.to(X.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = weight_dequant(ctx.weight, ctx.weight_scale)
        grad_X = torch_matmul(grad_output, W_deq)
        del W_deq
        return grad_X, None, None


@torch_compile
def fp8_torch_block_quant_forward(X, weight, weight_scale):
    return FP8BlockQuantLinear.apply(X, weight, weight_scale)


class FbgemmFp8Linear_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, weight_scale, bias = None):
        if weight.shape[0] == weight_scale.shape[0] and (
            weight.shape[0] % 8 == 0 and weight.shape[1] % 8 == 0
        ):
            # Edit: The kernel seems to expect that the weight has dimensions divisible by 8. Otherwise it throws `RuntimeError: cutlass cannot implement`
            # One thing we can do is to pad the weight and weight scale to multiple of 8 and perform a F8F8BF16 operation.
            # I tried benchmarking that for speed but observed that dequantize+bf16 matmul is significantly faster than padding+f8f8bf16 matmul. So we'll go that route.
            # So essentially, f8f8bf16_rowise only happens when shapes are proper (no transposes) and divisible by 8.

            # quantize_fp8_per_row will squash the leading dimensions, so save the desired shape here
            output_shape = (*x.shape[:-1], -1)
            # x_quantized and x_scale are not necessarily on the same device as x, this is an issue.
            # https://github.com/pytorch/FBGEMM/blob/e08af8539c391437f447173863df0f3f6f6f1855/fbgemm_gpu/experimental/gen_ai/src/quantize/quantize.cu#L1237C3-L1237C45
            x_quantized, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                x.view(-1, x.shape[-1]).contiguous(),
                scale_ub = getattr(weight, "input_scale_ub", None),
            )
            # moving x_quantized, x_scale here creates glibberish output ... However, if we move the output, it works
            # x_quantized, x_scale = x_quantized.to(x.device), x_scale.to(x.device)

            # The computation still happens on the device where self.weight is even if x_quantized is not on the same device as self.weight
            weight_scale_float32 = weight_scale.to(torch.float32)

            if not weight.is_contiguous():
                weight = weight.contiguous()
            if not weight_scale.is_contiguous():
                weight_scale = weight_scale.contiguous()

            output = torch.ops.fbgemm.f8f8bf16_rowwise(
                x_quantized, weight, x_scale, weight_scale_float32, use_fast_accum = True
            )
            output = output + bias if bias is not None else output
            # Hacky for now, we have the output to the device of x
            output = output.to(x.device, x.dtype)
            output = output.reshape(output_shape)
            del x_quantized, x_scale
        elif (
            weight.shape[0] != weight_scale.shape[0]
            and weight.shape[1] == weight_scale.shape[0]
        ) or (weight.shape[0] // 8 != 0 or weight.shape[1] // 8 != 0):
            # Either the weight/scale is transposed or its shape is not divisible by 8. Both cases, dequantizing is the preferred way.
            # The transpose case is generally noticed in backward pass when we do dY@W instead of @W.T as we do for forward.
            # The shape case, I noticed to happen in MLP of Qwen 2.5 VL 7B where the gate proj is of shape (3420, 1280) and 3420/8=427.5

            W_deq = weight_dequant(weight, weight_scale).T
            output = torch_matmul(x, W_deq)
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
def fbgemm_fp8_linear(X, weight, weight_scale, bias = None):
    return FbgemmFp8Linear_matmul.apply(X, weight, weight_scale, bias)


class FP8_fbgemm_block_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, weight_scale, bias = None):
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
                # weights are transposed during backward pass for training :)
                # We transpose weight scale to counter that. Note that transposing weight would cause issues with matmul with input X
                weight_scale = weight_scale.T
            else:
                raise ValueError(
                    f"Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {bs_n, bs_k}"
                )

        xq, xs = triton_quantize_fp8_block(X, bs_m, bs_n, None)
        ## TODO: Investigate and resolve the high divergence of this output from baseline
        # WARNING: This causes the outputs to diverge from expected when X has high values in it.
        # That results in the model producing gibberish, especially on longer sequences and training loss starting at high values like 8 instead of <1 ideally
        # Please refrain from using this till this issue is resolved. This exists here just for a future headstart.
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
def fp8_fbgemm_block_linear(X, weight, weight_scale, bias = None):
    return FP8_fbgemm_block_linear.apply(X, weight, weight_scale, bias)


def test_has_fbgemm():
    # We must manually check if the faster FBGEMM works on the specific GPU
    # For example RTX 5090 and RTX 4090 does not work
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
        e = str(e)
        if "cutlass cannot initialize" in e.lower():
            print(
                f"Unsloth: FBGEMM on the current GPU cannot load - will switch to Triton kernels"
            )
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

    # Older versions cause numerical imprecisions resulting in NaNs especially when X has high values in it.
    # This is both fast and accurate hence preferred.
    # This makes it 15% faster than the torchao implementation.
    if Version(fbgemm_gpu.__version__) >= Version("1.4.0"):
        # We must manually confirm if blockwise FBGEMM works!
        # This check is a must for consumer grade GPUs which fail
        if test_has_fbgemm():
            os.environ["UNSLOTH_HAS_FBGEMM"] = "1"
            logger.info(f"Using fbgemm_gpu block quantized FP8 matmul")
            fp8_block_quant_linear = fp8_fbgemm_block_linear
        else:
            os.environ["UNSLOTH_HAS_FBGEMM"] = "0"
except:
    pass


@torch_compile
def fp8_linear(X, weight, weight_scale, bias = None):
    if weight_scale.ndim == 2 and weight_scale.shape[1] > 1:
        # This is block quantized FP8 matmul
        out = fp8_block_quant_linear(X, weight, weight_scale)
    else:
        # Row quantized FP8
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

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
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn import functional as F

torch_matmul = torch.matmul

@torch.no_grad
def reconstruct_weight_fp8(
    W_fp8: torch.Tensor,
    W_scale: torch.Tensor,
    group_k: int,
    group_n: int,
    *,
    out_dtype=torch.bfloat16,
):
    K, N = W_fp8.shape
    num_k_groups = math.ceil(K / group_k)
    num_n_groups = math.ceil(N / group_n)

    # normalize scale to (num_k_groups, num_n_groups)
    if W_scale.numel() == 1:
        W_scale = W_scale.reshape(1, 1).expand(num_k_groups, num_n_groups)
    elif W_scale.dim() == 1 and W_scale.numel() == num_k_groups * num_n_groups:
        W_scale = W_scale.reshape(num_k_groups, num_n_groups)
    elif W_scale.dim() == 2 and W_scale.shape == (num_k_groups, num_n_groups):
        pass
    else:
        raise ValueError("Unsupported W_scale shape")

    W = W_fp8.to(dtype=W_scale.dtype).contiguous()
    W_scale = W_scale

    # If K or N not divisible by groups, handle last partial groups by padding
    Kpad = num_k_groups * group_k
    Npad = num_n_groups * group_n
    if Kpad != K or Npad != N:
        W_pad = W.new_zeros((Kpad, Npad))
        W_pad[:K, :N] = W
        W = W_pad

    Wg = W.view(num_k_groups, group_k, num_n_groups, group_n)
    Wg = Wg.permute(0, 2, 1, 3).contiguous()
    W_flat = Wg.view(num_k_groups * num_n_groups, group_k * group_n)

    ws_flat = W_scale.reshape(-1, 1)
    W_flat = W_flat * ws_flat

    # reshape back
    Wg = W_flat.view(num_k_groups, num_n_groups, group_k, group_n)
    Wg = Wg.permute(0, 2, 1, 3).to(out_dtype).contiguous()
    W_out = Wg.view(Kpad, Npad)[:K, :N]
    return W_out.T

# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)

def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
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

    pid = tl.program_id(axis=0)
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

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
    tl.store(c_ptrs, c, mask=c_mask)


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
    C = A.new_empty(C_shape, dtype=output_dtype)

    BLOCK_SIZE_M = 128
    if M < BLOCK_SIZE_M:
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_M = max(BLOCK_SIZE_M, 16)
    BLOCK_SIZE_K = block_k
    assert block_k % BLOCK_SIZE_K == 0
    BLOCK_SIZE_N = block_n

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

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
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
    )

    return C

class FP8_E4M3Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, weight_scale):
        # block_size = getattr(weight, 'block_size', [128,128])
        m,n = weight.shape
        p,q = weight_scale.shape
        block_size = getattr(weight, 'block_size', [m//p,n//q])
        # this is replica of https://github.com/huggingface/transformers/blob/01c9e1ba683b3e50d7c76bf92f2d470759fd5e81/src/transformers/integrations/finegrained_fp8.py#L331-L353
        qinput, scale = act_quant(X, block_size[1])
        output = w8a8_block_fp8_matmul_triton(
            qinput,
            weight,
            scale,
            weight_scale,
            block_size,
            output_dtype=X.dtype,
        )

        ctx.weight = weight
        ctx.weight_scale = weight_scale
        ctx.block_size = block_size

        return output.to(X.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = reconstruct_weight_fp8(ctx.weight, ctx.weight_scale, ctx.block_size[0], ctx.block_size[1])
        grad_X = torch_matmul(grad_output, W_deq.t())
        return grad_X, None, None

@torch.compile
def fp8_e4m3_forward(X, weight, weight_scale):
    return FP8_E4M3Linear.apply(X, weight, weight_scale)

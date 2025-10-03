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


class FP8_E4M3Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, weight_scale):
        # block_size = getattr(weight, 'block_size', [128,128])
        m,n = weight.shape
        p,q = weight_scale.shape
        assert m % p == 0 and n % q == 0, "FP8 Forward: weight and weight_scale shapes are not compatible"
        block_size = getattr(weight, 'block_size', [m//p,n//q])
        # this is replica of https://github.com/huggingface/transformers/blob/01c9e1ba683b3e50d7c76bf92f2d470759fd5e81/src/transformers/integrations/finegrained_fp8.py#L331-L353
        from transformers.integrations.finegrained_fp8 import act_quant, w8a8_block_fp8_matmul_triton
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

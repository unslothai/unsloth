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


class MPSRMSLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, eps: float, gemma: bool = False):
        # Save input shape to restore later
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim)

        # Use float32 for intermediate calculations for numerical stability
        # to match Triton kernels' behavior
        X_f32 = X.to(torch.float32)
        variance = X_f32.pow(2).mean(-1, keepdim = True)
        rms_inv = torch.rsqrt(variance + eps)

        # Casting back to X's dtype for the normalization
        X_norm = (X_f32 * rms_inv).to(X.dtype)

        if gemma:
            # Gemma variant uses W + 1
            Y = (W + 1.0).to(X.dtype) * X_norm
        else:
            Y = W.to(X.dtype) * X_norm

        ctx.save_for_backward(X, W, rms_inv)
        ctx.gemma = gemma
        ctx.eps = eps
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.reshape(-1, dim)
        X, W, rms_inv = ctx.saved_tensors
        gemma = ctx.gemma
        eps = ctx.eps

        # Intermediate in float32
        X_f32 = X.reshape(-1, dim).to(torch.float32)
        dY_f32 = dY.to(torch.float32)

        # Weights handling
        if gemma:
            W_effective = W.to(torch.float32) + 1.0
        else:
            W_effective = W.to(torch.float32)

        # dL/d(normalized_X)
        dX_norm = dY_f32 * W_effective

        # dL/dW sum over batch and sequence dimensions
        # X_norm = X_f32 * rms_inv
        # dW = sum(dY * X_norm)
        # Note: X_norm is in the original dtype in forward, but for gradient we use f32
        dW = (dY_f32 * (X_f32 * rms_inv)).sum(dim = 0)

        # dL/dX
        N = X_f32.shape[-1]
        X_norm_f32 = X_f32 * rms_inv

        # rowsum_dY_normed = sum(dX_norm * X_norm)
        rowsum_dY_normed = (dX_norm * X_norm_f32).sum(-1, keepdim = True)

        # dX = (rms_inv / N) * (N * dX_norm - X_norm * rowsum_dY_normed)
        dX = rms_inv * (dX_norm - (X_norm_f32 / N) * rowsum_dY_normed)

        return dX.reshape(*shape).to(X.dtype), dW.to(W.dtype), None, None


def mps_rms_layernorm(
    X: torch.Tensor, W: torch.Tensor, eps: float, gemma: bool = False
):
    return MPSRMSLayerNorm.apply(X, W, eps, gemma)

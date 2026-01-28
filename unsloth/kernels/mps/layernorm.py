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

class MPSLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, eps: float):
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim)
        
        # Intermediate in float32 for stability
        X_f32 = X.to(torch.float32)
        mean = X_f32.mean(-1, keepdim=True)
        # Using unbiased=False to match Triton/PyTorch LayerNorm defaults
        variance = X_f32.var(-1, keepdim=True, unbiased=False)
        inv_std = torch.rsqrt(variance + eps)
        
        X_norm = (X_f32 - mean) * inv_std
        
        # Cast to weights/X's dtype
        Y = W.to(X.dtype) * X_norm.to(X.dtype) + b.to(X.dtype)
        
        ctx.save_for_backward(X, W, b, mean, inv_std)
        ctx.eps = eps
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.reshape(-1, dim)
        X, W, b, mean, inv_std = ctx.saved_tensors
        eps = ctx.eps
        
        X_f32 = X.reshape(-1, dim).to(torch.float32)
        dY_f32 = dY.to(torch.float32)
        
        # Normalized input
        X_centered = X_f32 - mean
        X_norm = X_centered * inv_std
        
        # Gradients for weights and bias
        dW = (dY_f32 * X_norm).sum(0)
        db = dY_f32.sum(0)
        
        # dL/dX_norm
        dX_norm = dY_f32 * W.to(torch.float32)
        
        # dL/dX for LayerNorm
        N = dim
        sum_dX_norm = dX_norm.sum(-1, keepdim=True)
        sum_dX_norm_X_norm = (dX_norm * X_norm).sum(-1, keepdim=True)
        
        # dX = inv_std * (dX_norm - mean(dX_norm) - X_norm * mean(dX_norm * X_norm))
        dX = inv_std * (
            dX_norm - (sum_dX_norm / N) - X_norm * (sum_dX_norm_X_norm / N)
        )
        
        return dX.reshape(*shape).to(X.dtype), dW.to(W.dtype), db.to(b.dtype), None

def mps_layernorm(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, eps: float):
    return MPSLayerNorm.apply(X, W, b, eps)

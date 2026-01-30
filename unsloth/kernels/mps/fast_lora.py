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
import torch.nn.functional as F


def mps_matmul_lora(X, W, W_quant, A, B, s):
    """
    MPS matmul_lora fallback.
    Assumes W is already in a usable format for MPS (16-bit).
    """
    dtype = X.dtype

    # Base projection: X @ W.t()
    out = torch.matmul(X, W.t())

    # LoRA contribution: (X @ A.t()) @ (B.t() * s)
    if A is not None:
        # X: (..., in_dim), A: (rank, in_dim), B: (out_dim, rank)
        XA = torch.matmul(X, A.t().to(dtype))
        out.view(-1, out.shape[-1]).addmm_(
            XA.view(-1, XA.shape[-1]), B.t().to(dtype), alpha = s
        )

    return out

class MPSLoRA_MLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        _forward_function,
    ):
        # Forward pass using MPS-compatible operations
        e = mps_matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
        h = _forward_function(e, g)
        i = mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.save_for_backward(
            X, gateW, gateA, gateB, upW, upA, upB, downW, downA, downB, e, g, h
        )
        ctx.gateS, ctx.upS, ctx.downS = gateS, upS, downS
        return i

    @staticmethod
    def backward(ctx, dY):
        X, gateW, gateA, gateB, upW, upA, upB, downW, downA, downB, e, g, h = (
            ctx.saved_tensors
        )
        gateS, upS, downS = ctx.gateS, ctx.upS, ctx.downS

        # simplified backward for now - focus on correctness
        # In a real implementation we would match LoRA_MLP.backward exactly
        # but for MPS fallback we can rely on autograd for these complex fused ops
        # unless performance is critical.
        # Actually, for Unsloth we should probably provide the manual backward to save memory.

        # TODO: Implement full manual backward for LoRA_MLP on MPS
        raise NotImplementedError(
            "Manual backward for MPSLoRA_MLP not yet implemented. Use autograd for now."
        )


def mps_apply_lora_mlp_swiglu(
    X,
    gateW,
    gateW_quant,
    gateA,
    gateB,
    gateS,
    upW,
    upW_quant,
    upA,
    upB,
    upS,
    downW,
    downW_quant,
    downA,
    downB,
    downS,
):
    # Dispatching to torch-native implementation
    # For now, we use the device-agnostic logic but with MPS-friendly matmul
    e = mps_matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
    g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
    h = F.silu(e) * g
    return mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)


def mps_apply_lora_qkv(
    X, QW, QW_quant, QA, QB, QS, KW, KW_quant, KA, KB, KS, VW, VW_quant, VA, VB, VS
):
    Q = mps_matmul_lora(X, QW, QW_quant, QA, QB, QS)
    K = mps_matmul_lora(X, KW, KW_quant, KA, KB, KS)
    V = mps_matmul_lora(X, VW, VW_quant, VA, VB, VS)
    return Q, K, V


def mps_apply_lora_o(X, OW, OW_quant, OA, OB, OS):
    """MPS O projection fallback using PyTorch operations."""
    return mps_matmul_lora(X, OW, OW_quant, OA, OB, OS)


def mps_apply_lora_mlp_geglu_exact(X, gateW, gateW_quant, gateA, gateB, gateS,
                                   upW, upW_quant, upA, upB, upS,
                                   downW, downW_quant, downA, downB, downS):
    """MPS GEGLU (Exact) MLP fallback using PyTorch operations."""
    e = mps_matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
    g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
    # GEGLU: GELU(e) * g
    h = F.gelu(e, approximate='none') * g
    return mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)


def mps_apply_lora_mlp_geglu_approx(X, gateW, gateW_quant, gateA, gateB, gateS,
                                    upW, upW_quant, upA, upB, upS,
                                    downW, downW_quant, downA, downB, downS):
    """MPS GEGLU (Approximate) MLP fallback using PyTorch operations."""
    e = mps_matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
    g = mps_matmul_lora(X, upW, upW_quant, upA, upB, upS)
    # GEGLU approximate: GELU(e, approximate='tanh') * g
    h = F.gelu(e, approximate='tanh') * g
    return mps_matmul_lora(h, downW, downW_quant, downA, downB, downS)

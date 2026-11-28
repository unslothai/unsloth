# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""4-bit matmul for Unsloth's bitsandbytes stand-in. The weight stays packed: forward and backward
each dequantize it (Triton on GPUs), so only the packed bytes are kept for the backward pass."""

from typing import Optional

import torch
import torch.nn.functional as F_nn

from .. import functional as F


class MatMul4Bit(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A,
        B,
        out = None,
        bias = None,
        quant_state = None,
    ):
        ctx.state = quant_state
        ctx.B = B
        ctx.has_bias = bias is not None
        ctx.dtype_bias = None if bias is None else bias.dtype
        W = F.dequantize_4bit(B.view(-1, 1), quant_state).to(A.dtype)
        output = F_nn.linear(A, W, bias)
        if out is not None:
            out.copy_(output)
            output = out
        return output

    @staticmethod
    def backward(ctx, grad_output):
        req_A, _, _, req_bias, _ = ctx.needs_input_grad
        grad_A = grad_bias = None
        if req_bias:
            grad_bias = grad_output.sum(dim = tuple(range(grad_output.dim() - 1))).to(ctx.dtype_bias)
        if req_A:
            W = F.dequantize_4bit(ctx.B.view(-1, 1), ctx.state).to(grad_output.dtype)
            grad_A = torch.matmul(grad_output, W)
        return grad_A, None, None, grad_bias, None


def matmul_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    quant_state,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
    if quant_state is None:
        raise ValueError("quant_state is required")
    if A.numel() == 0:
        shape = quant_state.shape
        return torch.empty(A.shape[:-1] + shape[:1], dtype = A.dtype, device = A.device)
    if A.requires_grad or (bias is not None and bias.requires_grad):
        if torch.is_grad_enabled():
            return MatMul4Bit.apply(A, B, out, bias, quant_state)
    W = F.dequantize_4bit(B.view(-1, 1), quant_state).to(A.dtype)
    result = F_nn.linear(A, W, bias)
    if out is not None:
        out.copy_(result)
        return out
    return result

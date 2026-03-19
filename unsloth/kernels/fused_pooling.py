# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import triton
import triton.language as tl
import torch
from .utils import calculate_settings, torch_gpu_device


@triton.jit
def _fused_layernorm_mean_pool_backward(
    dY,  # grad w.r.t. X_flat, written in-place, (n_rows, hidden_dim)
    dY_row_stride,
    X,  # saved X_flat, (n_rows, hidden_dim)
    X_row_stride,
    W,  # LayerNorm weight, (hidden_dim,)
    r,  # saved inv_var, (n_rows,)
    mu,  # saved mean, (n_rows,)
    grad_pooled,  # upstream gradient, (batch_size, hidden_dim)
    grad_pooled_stride,
    seq_lengths,  # (batch_size,)
    padded_seq_len,  # int
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Which sentence does this row belong to?
    sentence_idx = row_idx // padded_seq_len
    tok_offset = row_idx % padded_seq_len
    seq_len = tl.load(seq_lengths + sentence_idx)

    # Skip padding rows
    if tok_offset >= seq_len:
        tl.store(
            dY + row_idx * dY_row_stride + col_offsets,
            tl.zeros([BLOCK_SIZE], dtype = tl.float32),
            mask = mask,
        )
        return

    # grad from mean pooling: grad_pooled[sentence] / seq_len
    dY_row = tl.load(
        grad_pooled + sentence_idx * grad_pooled_stride + col_offsets,
        mask = mask,
        other = 0,
    ).to(tl.float32) / seq_len.to(tl.float32)

    X_row = tl.load(X + row_idx * X_row_stride + col_offsets, mask = mask, other = 0).to(
        tl.float32
    )
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    inv_var = tl.load(r + row_idx).to(tl.float32)
    mean = tl.load(mu + row_idx).to(tl.float32)
    normed = (X_row - mean) * inv_var

    dY_W = dY_row * W_row
    dX_row = (
        dY_W
        - tl.sum(dY_W, axis = 0) / n_cols
        - normed * tl.sum(dY_W * normed, axis = 0) / n_cols
    )
    dX_row = dX_row * inv_var

    tl.store(dY + row_idx * dY_row_stride + col_offsets, dX_row, mask = mask)


@triton.jit
def _fused_layernorm_mean_pool_forward(
    Out,
    Out_row_stride,
    X,
    X_row_stride,
    W,
    b,
    inv_var,
    mu,
    seq_lengths,  # actual lengths per sentence, shape (batch_size,)
    padded_seq_len,  # padded sequence length (int)
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    sentence_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Actual number of real tokens in this sentence
    seq_len = tl.load(seq_lengths + sentence_idx)

    # Base row in the padded X_flat for this sentence
    base_row = sentence_idx * padded_seq_len

    # Load LayerNorm weights and bias once (float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask = mask, other = 0).to(tl.float32)

    # Accumulator for mean pooling (float32)
    pool_acc = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

    eps_f32 = tl.full((), eps, tl.float32)

    # Loop over real tokens in this sentence (in padded layout)
    for tok_offset in range(padded_seq_len):
        # Guard: skip padding tokens
        if tok_offset < seq_len:
            tok = base_row + tok_offset

            # Load this token's hidden states
            X_row = tl.load(
                X + tok * X_row_stride + col_offsets, mask = mask, other = 0
            ).to(tl.float32)

            # Compute mean
            mean_X = tl.sum(X_row, axis = 0) / n_cols
            XX = tl.where(mask, X_row - mean_X, 0)

            # Compute variance
            row_var = tl.sum(XX * XX, axis = 0) / n_cols
            inv_v = tl.math.rsqrt(row_var + eps_f32)

            # Save inv_var and mu for backward
            tl.store(inv_var + tok, inv_v)
            tl.store(mu + tok, mean_X)

            # Apply LayerNorm: (x - mean) * rsqrt(var + eps) * W + B
            normed = (XX * inv_v) * W_row + b_row

            # Accumulate for mean pooling
            pool_acc += normed

    # Divide by sequence length to get mean pool
    pool_acc = pool_acc / seq_len.to(tl.float32)

    # Store output for this sentence
    Out += sentence_idx * Out_row_stride
    tl.store(Out + col_offsets, pool_acc, mask = mask)


class FusedLayerNormMeanPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, bias, attention_mask, eps = 1e-12):
        batch_size, seq_len, hidden_dim = X.shape
        device = X.device

        # Flatten X to (batch*seq, hidden_dim)
        X_flat = X.view(-1, hidden_dim)
        n_rows = X_flat.shape[0]

        # Per-sentence real token counts
        seq_lengths = attention_mask.sum(dim = 1).to(torch.int32)  # (batch_size,)

        # Output: one pooled vector per sentence
        Out = torch.empty((batch_size, hidden_dim), dtype = X.dtype, device = device)

        # Allocate inv_var and mu for all tokens (needed for backward)
        inv_var = torch.empty(n_rows, dtype = torch.float32, device = device)
        mu_buf = torch.empty(n_rows, dtype = torch.float32, device = device)

        BLOCK_SIZE, num_warps = calculate_settings(hidden_dim)

        with torch_gpu_device(device):
            _fused_layernorm_mean_pool_forward[(batch_size,)](
                Out,
                Out.stride(0),
                X_flat,
                X_flat.stride(0),
                W,
                bias,
                inv_var,
                mu_buf,
                seq_lengths,
                seq_len,
                hidden_dim,
                eps,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps = num_warps,
            )

        ctx.save_for_backward(X_flat, W, bias, inv_var, mu_buf, seq_lengths)
        ctx.eps = eps
        ctx.shape = (batch_size, seq_len, hidden_dim)
        return Out

    @staticmethod
    def backward(ctx, grad_pooled):
        # grad_pooled: (batch_size, hidden_dim)
        X_flat, W, bias, inv_var, mu_buf, seq_lengths = ctx.saved_tensors
        batch_size, seq_len, hidden_dim = ctx.shape
        device = X_flat.device
        n_rows = X_flat.shape[0]

        BLOCK_SIZE, num_warps = calculate_settings(hidden_dim)

        # dX via Triton kernel (one thread-block per token row)
        dX_flat = torch.empty_like(X_flat)
        with torch_gpu_device(device):
            _fused_layernorm_mean_pool_backward[(n_rows,)](
                dX_flat,
                dX_flat.stride(0),
                X_flat,
                X_flat.stride(0),
                W,
                inv_var,
                mu_buf,
                grad_pooled,
                grad_pooled.stride(0),
                seq_lengths,
                seq_len,
                hidden_dim,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps = num_warps,
            )

        # dW and d_bias via vectorized PyTorch (no Python loop)
        # normed = (X - mu) * inv_var per token, then dY * normed / seq_len summed
        X_f32 = X_flat.float()
        normed = (X_f32 - mu_buf.unsqueeze(1)) * inv_var.unsqueeze(1)

        # Build per-token grad: grad_pooled[sentence_of_token] / seq_len_of_token
        # sentence_idx for each row in padded layout
        sentence_idx = torch.arange(batch_size, device = device).repeat_interleave(
            seq_len
        )
        per_token_grad = (
            grad_pooled[sentence_idx].float()
            / seq_lengths[sentence_idx].unsqueeze(1).float()
        )

        # Mask out padding rows
        tok_offsets = torch.arange(seq_len, device = device).repeat(batch_size)
        real_mask = tok_offsets < seq_lengths.repeat_interleave(seq_len)
        per_token_grad = per_token_grad * real_mask.unsqueeze(1)

        dW = (per_token_grad * normed).sum(dim = 0).to(W.dtype)
        d_bias = per_token_grad.sum(dim = 0).to(bias.dtype)

        dX = dX_flat.view(batch_size, seq_len, hidden_dim)
        return dX, dW, d_bias, None, None


def fused_layernorm_mean_pool(layernorm, X, attention_mask):
    """
    Convenience function that fuses LayerNorm + Mean Pooling into a single kernel.

    Args:
        layernorm: An nn.LayerNorm module with elementwise_affine=True.
        X: Input tensor of shape (batch_size, seq_len, hidden_dim).
        attention_mask: Tensor of shape (batch_size, seq_len), 1 for real tokens,
                        0 for padding.

    Returns:
        Pooled output of shape (batch_size, hidden_dim).
    """
    assert layernorm.elementwise_affine is True
    W = layernorm.weight
    bias = layernorm.bias
    eps = (
        layernorm.variance_epsilon
        if hasattr(layernorm, "variance_epsilon")
        else layernorm.eps
    )
    return FusedLayerNormMeanPool.apply(X, W, bias, attention_mask, eps)

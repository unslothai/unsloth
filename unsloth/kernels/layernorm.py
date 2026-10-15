# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Copyright 2024-present Andrej Karpathy & the llm.c team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import triton
import triton.language as tl
import torch
from .utils import calculate_settings, long_indexing, torch_gpu_device
from unsloth_zoo.patching_utils import (
    patch_layernorm,
)


@triton.jit
def layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,  # n_cols
    X,
    X_row_stride: tl.constexpr,  # n_cols
    W,
    b,
    r,
    mu,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr = True,
):
    row_idx = tl.program_id(0)
    if LONG_INDEXING:
        row_idx = row_idx.to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx

    # Per torchtune's Fp32LayerNorm, all modules are in float32.
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    mean_X = tl.sum(X_row, axis = 0) / n_cols
    # (X[0] - mean) == -mean, so mask it out.
    XX = tl.where(mask, X_row - mean_X, 0)
    row_var = tl.sum(XX * XX, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm.
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    tl.store(mu, mean_X)
    output = (XX * inv_var) * W_row
    if HAS_BIAS:
        output += tl.load(b + col_offsets, mask = mask, other = 0).to(tl.float32)
    tl.store(Y + col_offsets, output, mask = mask)


@triton.jit
def _layernorm_backward_row(
    dY,
    X,
    W_row,
    r,
    mu,
    row_idx,
    col_offsets,
    mask,
    n_cols: tl.constexpr,
):
    # Approximately follows karpathy/llm.c doc/layernorm/layernorm.md
    # Per torchtune's Fp32LayerNorm, all modules are in float32.
    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)

    inv_var = tl.load(r + row_idx).to(tl.float32)
    mean = tl.load(mu + row_idx).to(tl.float32)
    normed = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    dX_row = dY_W - tl.sum(dY_W, axis = 0) / n_cols - normed * tl.sum(dY_W * normed, axis = 0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dY + col_offsets, dX_row, mask = mask)
    return dY_row, normed


@triton.jit
def layernorm_backward(
    dY,
    dY_row_stride: tl.constexpr,  # n_cols
    X,
    X_row_stride: tl.constexpr,  # n_cols
    W,
    b,
    r,
    mu,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if LONG_INDEXING:
        row_idx = row_idx.to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)
    _layernorm_backward_row(
        dY + row_idx * dY_row_stride,
        X + row_idx * X_row_stride,
        W_row,
        r,
        mu,
        row_idx,
        col_offsets,
        mask,
        n_cols,
    )


# dW / dB follow flash-linear-attention's layer_norm_bwd_kernel_row + layer_norm_bwd
# (https://github.com/fla-org/flash-linear-attention, fla/modules/norm/layernorm.py, MIT licence,
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li): a few persistent programs each own a
# run of rows, add dY * xhat and dY into fp32 registers while computing dX, and store one partial
# row each; the partials are summed afterwards. dX per row is the frozen kernel's, same helper.
@triton.jit
def layernorm_backward_dw(
    dY,
    dY_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    r,
    mu,
    dW,
    dB,
    n_rows,
    rows_per_program,
    n_cols: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TRAIN_W: tl.constexpr,
    TRAIN_B: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)
    dW_acc = tl.zeros((BLOCK_SIZE,), dtype = tl.float32)
    dB_acc = tl.zeros((BLOCK_SIZE,), dtype = tl.float32)
    row_start = pid * rows_per_program
    for i in range(row_start, tl.minimum(row_start + rows_per_program, n_rows)):
        row_idx = i
        if LONG_INDEXING:
            row_idx = i.to(tl.int64)
        dY_row, normed = _layernorm_backward_row(
            dY + row_idx * dY_row_stride,
            X + row_idx * X_row_stride,
            W_row,
            r,
            mu,
            row_idx,
            col_offsets,
            mask,
            n_cols,
        )
        if TRAIN_W:
            dW_acc += dY_row * normed
        if TRAIN_B:
            dB_acc += dY_row
    if TRAIN_W:
        tl.store(dW + pid * n_cols + col_offsets, dW_acc, mask = mask)
    if TRAIN_B:
        tl.store(dB + pid * n_cols + col_offsets, dB_acc, mask = mask)


@triton.jit
def _layernorm_sum_partials(
    partials,
    out0,
    out1,
    n_programs,
    n_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    # Fixed tile order, so eager and torch.compile give the same bytes (a torch sum here is
    # re-lowered by Inductor in another order). Axis 1 picks the partial set: out0, then out1.
    which = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = cols < n_cols
    partials += which * n_programs * n_cols
    acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype = tl.float32)
    for start in tl.range(0, n_programs, BLOCK_ROWS, num_stages = 3):
        rows = start + tl.arange(0, BLOCK_ROWS)
        mask = (rows < n_programs)[:, None] & col_mask[None, :]
        acc += tl.load(partials + rows[:, None] * n_cols + cols[None, :], mask = mask, other = 0)
    total = tl.sum(acc, axis = 0)
    if which == 0:
        tl.store(out0 + cols, total.to(out0.dtype.element_ty), mask = col_mask)
    else:
        tl.store(out1 + cols, total.to(out1.dtype.element_ty), mask = col_mask)


# fla runs one program per SM, which at 4 to 8 warps leaves an A100 SM mostly idle; filling half of
# each SM's warps measured best on both A100 (1.1x to 1.2x at 1024 / 2048 columns) and T4.
@functools.lru_cache(maxsize = None)
def _dw_slots(device):
    try:
        props = torch.cuda.get_device_properties(device)
        return int(props.multi_processor_count), max(1, int(props.max_threads_per_multi_processor) // 64)
    except Exception:
        return 128, 32


class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim).contiguous()
        # The kernels read W and b at unit stride, and these are the ones saved for backward.
        W = W.contiguous()
        has_bias = b is not None
        b = b.contiguous() if has_bias else W
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device
        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device)
        r = torch.empty(n_rows, dtype = torch.float32, device = device)
        mu = torch.empty(n_rows, dtype = torch.float32, device = device)

        with torch_gpu_device(device):
            layernorm_forward[(n_rows,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                b,
                r,
                mu,
                n_cols,
                eps,
                LONG_INDEXING = long_indexing(X, Y, block = BLOCK_SIZE),
                BLOCK_SIZE = BLOCK_SIZE,
                HAS_BIAS = has_bias,
                num_warps = num_warps,
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, b, r, mu)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.reshape(-1, dim).contiguous()
        X, W, b, r, mu = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        train_w, train_b = ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        LONG_INDEXING = long_indexing(dY, X, block = ctx.BLOCK_SIZE)

        if not (train_w or train_b):
            with torch_gpu_device(dY.device):
                layernorm_backward[(n_rows,)](
                    dY,
                    dY.stride(0),
                    X,
                    X.stride(0),
                    W,
                    b,
                    r,
                    mu,
                    n_cols,
                    ctx.eps,
                    LONG_INDEXING = LONG_INDEXING,
                    BLOCK_SIZE = ctx.BLOCK_SIZE,
                    num_warps = ctx.num_warps,
                )
            return dY.view(*shape), None, None, None

        sms, warps_per_sm = _dw_slots(dY.device)
        programs = max(1, min(n_rows, sms * max(1, warps_per_sm // ctx.num_warps)))
        rows_per_program = max(1, triton.cdiv(n_rows, programs))
        programs = max(1, triton.cdiv(n_rows, rows_per_program))
        # One buffer for both partials: every program stores its rows, so empty is enough (zeros only
        # when nothing launches). At small shapes the backward is host bound, so fewer ops matter.
        alloc = torch.empty if n_rows > 0 else torch.zeros
        partials = alloc((train_w + train_b, programs, n_cols), dtype = torch.float32, device = dY.device)
        if n_rows > 0:
            with torch_gpu_device(dY.device):
                layernorm_backward_dw[(programs,)](
                    dY,
                    dY.stride(0),
                    X,
                    X.stride(0),
                    W,
                    r,
                    mu,
                    partials[0],
                    partials[-1],
                    n_rows,
                    rows_per_program,
                    n_cols,
                    LONG_INDEXING = LONG_INDEXING,
                    BLOCK_SIZE = ctx.BLOCK_SIZE,
                    TRAIN_W = train_w,
                    TRAIN_B = train_b,
                    num_warps = ctx.num_warps,
                )
        dW = torch.empty(n_cols, dtype = W.dtype, device = dY.device) if train_w else None
        dB = torch.empty(n_cols, dtype = b.dtype, device = dY.device) if train_b else None
        outs = [t for t in (dW, dB) if t is not None]
        with torch_gpu_device(dY.device):
            _layernorm_sum_partials[(triton.cdiv(n_cols, 64), len(outs))](
                partials,
                outs[0],
                outs[-1],
                programs,
                n_cols,
                BLOCK_ROWS = 64,
                BLOCK_COLS = 64,
            )
        return dY.view(*shape), dW, dB, None


def fast_layernorm(layernorm, X):
    assert layernorm.elementwise_affine is True
    W = layernorm.weight
    bias = layernorm.bias
    eps = layernorm.variance_epsilon if hasattr(layernorm, "variance_epsilon") else layernorm.eps
    out = Fast_Layernorm.apply(X, W, bias, eps)
    return out


def test_layernorm(
    dim = 1024,
    eps = 1e-5,
    dtype = torch.float16,
    bsz = 21,
    random_state = 3407,
    seqlen = 3341,
):
    from torch.nn import LayerNorm

    layernorm = LayerNorm((dim,), eps = eps, device = "cuda", dtype = dtype)
    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    torch.nn.init.uniform_(layernorm.weight)
    torch.nn.init.uniform_(layernorm.bias)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    XX = X.clone()
    X.requires_grad_(True)
    XX.requires_grad_(True)
    Y = layernorm(X)
    YY = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda", requires_grad = True)
    Y.backward(YY)
    correct_grad = X.grad.clone()
    Y = fast_layernorm(layernorm, XX)
    Y.backward(YY)
    assert torch.dist(correct_grad, XX.grad).item() <= 0.1


def testing_suite_layernorm():
    for dim in [512, 1024, 2048]:
        for dtype in [torch.float16, torch.bfloat16]:
            with torch.autocast(device_type = "cuda", dtype = dtype):
                for seqlen in [3341, 2048, 349]:
                    for random_state in [3407, 42]:
                        test_layernorm(
                            dim = dim,
                            eps = 1e-5,
                            dtype = dtype,
                            bsz = 21,
                            random_state = random_state,
                            seqlen = seqlen,
                        )

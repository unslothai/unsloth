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
from .utils import calculate_settings, long_indexing, norm_num_warps, torch_gpu_device
from . import triton_launch
from .triton_launch import launch
from unsloth_zoo.patching_utils import (
    patch_layernorm,
)


@triton.jit
def layernorm_forward(
    Y,
    X,
    W,
    b,
    r,
    mu,
    Y_row_stride: tl.constexpr,  # n_cols
    X_row_stride: tl.constexpr,  # n_cols
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
    # mask invalid lanes because subtraction turns loaded zeros into -mean.
    XX = tl.where(mask, X_row - mean_X, 0)
    row_var = tl.sum(XX * XX, axis = 0) / n_cols
    # explicit float32 scalar ensures correct type promotion on HIP/ROCm.
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
    dY, dX, X, W_row, r, mu, row_idx, col_offsets, mask, n_cols: tl.constexpr
):
    # based on karpathy/llm.c; float32 matches torchtune's Fp32LayerNorm.
    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)

    inv_var = tl.load(r + row_idx).to(tl.float32)
    mean = tl.load(mu + row_idx).to(tl.float32)
    normed = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    dX_row = dY_W - tl.sum(dY_W, axis = 0) / n_cols - normed * tl.sum(dY_W * normed, axis = 0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dX + col_offsets, dX_row, mask = mask)
    return dY_row, normed


@triton.jit
def layernorm_backward(
    dY,
    dX,
    X,
    W,
    b,
    r,
    mu,
    dY_row_stride: tl.constexpr,
    dX_row_stride: tl.constexpr,
    X_row_stride: tl.constexpr,
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
        dX + row_idx * dX_row_stride,
        X + row_idx * X_row_stride,
        W_row,
        r,
        mu,
        row_idx,
        col_offsets,
        mask,
        n_cols,
    )


# MIT 2023-2026 Songlin Yang et al.: dW/dB follow flash-linear-attention's layer_norm_bwd_kernel_row
@triton.jit
def layernorm_backward_dw(
    dY,
    dX,
    X,
    W,
    r,
    mu,
    dW,
    dB,
    n_rows,
    rows_per_program,
    dY_row_stride: tl.constexpr,
    dX_row_stride: tl.constexpr,
    X_row_stride: tl.constexpr,
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
            dX + row_idx * dX_row_stride,
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
    partials, out0, out1, n_programs, n_cols, BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr
):
    # Not torch.sum: Inductor reorders it, so eager and compiled bytes would differ.
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


# Half of each SM's warps (not fla's one program per SM) measured best on A100 and T4.
@functools.lru_cache(maxsize = None)
def _dw_slots(device):
    try:
        props = torch.cuda.get_device_properties(device)
        return int(props.multi_processor_count), max(
            1, int(props.max_threads_per_multi_processor) // 64
        )
    except Exception:
        return 128, 32


# num_warps per (compute capability, n_cols), replacing calculate_settings' where a sweep (fp16 + bf16 on
# sm_80+, fp16 on T4; 2k and 16k rows) found a width faster outside the run's A/A band in some cell and never
# slower in any, at least 3% on the geometric mean, with max error vs fp64 no worse than the default's; every
# entry then won again against main on torch 2.14.1 (same GPU tier, interleaved, outside A/A, never slower).
# _NUM_WARPS: forward + frozen-W backward; _DW_NUM_WARPS: the dW / dB backward.
_NUM_WARPS = {
    ((7, 5), 2560): 16,
    ((7, 5), 4096): 16,
    ((8, 0), 2560): 2,
    ((8, 9), 2048): 4,
    ((9, 0), 2560): 2,
    ((9, 0), 5120): 4,
    ((12, 0), 5120): 4,
}
_DW_NUM_WARPS = {
    ((7, 5), 4096): 16,
    ((8, 0), 5120): 32,
    ((9, 0), 5120): 32,
    ((12, 0), 4096): 16,
}


def _arch_settings(n_cols, device, dw):
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    table = _DW_NUM_WARPS if dw else _NUM_WARPS
    return BLOCK_SIZE, norm_num_warps(table, n_cols, num_warps, device)


_cached_arch_settings = functools.lru_cache(maxsize = None)(_arch_settings)


def _settings(
    n_cols,
    device = None,
    dw = False,
):
    # Cached per (width, device) for eager calls; tracing and symbolic (unhashable) widths skip it.
    if torch.compiler.is_compiling() or type(n_cols) is not int:
        return _arch_settings(n_cols, device, dw)
    return _cached_arch_settings(n_cols, device, dw)


def _launch(kernel, grid, args, n_tensors, constexprs, device, **options):
    # Eager CUDA skips Triton's arg binding (triton_launch.py); compiling calls keep kernel[grid].
    # is_compiling first: Dynamo must not read (and guard on) the mutable _ENABLED.
    if not torch.compiler.is_compiling() and triton_launch._ENABLED and device.type == "cuda":
        launch(kernel, grid, args, n_tensors, constexprs, device.index, **options)
    else:
        kernel[grid](*args, **constexprs, **options)


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
        device = X.device
        BLOCK_SIZE, num_warps = _settings(n_cols, device)
        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device)
        r = torch.empty(n_rows, dtype = torch.float32, device = device)
        mu = torch.empty(n_rows, dtype = torch.float32, device = device)

        with torch_gpu_device(device):
            _launch(
                layernorm_forward,
                (n_rows,),
                (Y, X, W, b, r, mu),
                6,
                dict(
                    Y_row_stride = Y.stride(0),
                    X_row_stride = X.stride(0),
                    n_cols = n_cols,
                    eps = eps,
                    LONG_INDEXING = long_indexing(X, Y, block = BLOCK_SIZE),
                    BLOCK_SIZE = BLOCK_SIZE,
                    HAS_BIAS = has_bias,
                ),
                device,
                num_warps = num_warps,
                enable_fp_fusion = True,
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
        device = dY.device
        train_w, train_b = ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        # do not write into dY because autograd may share it with other nodes or the caller
        dX = torch.empty_like(dY)
        LONG_INDEXING = long_indexing(dY, dX, X, block = ctx.BLOCK_SIZE)

        if not (train_w or train_b):
            with torch_gpu_device(device):
                _launch(
                    layernorm_backward,
                    (n_rows,),
                    (dY, dX, X, W, b, r, mu),
                    7,
                    dict(
                        dY_row_stride = dY.stride(0),
                        dX_row_stride = dX.stride(0),
                        X_row_stride = X.stride(0),
                        n_cols = n_cols,
                        eps = ctx.eps,
                        LONG_INDEXING = LONG_INDEXING,
                        BLOCK_SIZE = ctx.BLOCK_SIZE,
                    ),
                    device,
                    num_warps = ctx.num_warps,
                    enable_fp_fusion = True,
                )
            return dX.view(*shape), None, None, None

        dw_warps = _settings(n_cols, device, dw = True)[1]
        programs, rows_per_program = _dw_programs(device, n_rows, dw_warps)
        # every program stores its partial row, so empty works unless no program launches
        alloc = torch.empty if n_rows > 0 else torch.zeros
        partials = alloc((train_w + train_b, programs, n_cols), dtype = torch.float32, device = device)
        dW = torch.empty(n_cols, dtype = W.dtype, device = device) if train_w else None
        dB = torch.empty(n_cols, dtype = b.dtype, device = device) if train_b else None
        outs = [t for t in (dW, dB) if t is not None]
        # Unused slot must not alias the used one: torch.compile drops one of two writes to a buffer.
        spare = partials.new_empty(1) if len(outs) == 1 else None
        partial_w = partials[0] if train_w else spare
        partial_b = partials[-1] if train_b else spare
        out1 = outs[-1] if len(outs) == 2 else outs[0].new_empty(1)
        with torch_gpu_device(device):
            if n_rows > 0:
                _launch(
                    layernorm_backward_dw,
                    (programs,),
                    (dY, dX, X, W, r, mu, partial_w, partial_b, n_rows, rows_per_program),
                    8,
                    dict(
                        dY_row_stride = dY.stride(0),
                        dX_row_stride = dX.stride(0),
                        X_row_stride = X.stride(0),
                        n_cols = n_cols,
                        LONG_INDEXING = LONG_INDEXING,
                        BLOCK_SIZE = ctx.BLOCK_SIZE,
                        TRAIN_W = train_w,
                        TRAIN_B = train_b,
                    ),
                    device,
                    num_warps = dw_warps,
                    enable_fp_fusion = True,
                )
            _launch(
                _layernorm_sum_partials,
                ((n_cols + 63) // 64, len(outs)),
                (partials, outs[0], out1, programs, n_cols),
                3,
                dict(BLOCK_ROWS = 64, BLOCK_COLS = 64),
                device,
                enable_fp_fusion = True,
            )
        return dX.view(*shape), dW, dB, None


def _dw_programs(device, n_rows, num_warps):
    sms, warps_per_sm = _dw_slots(device)
    programs = max(1, min(n_rows, sms * max(1, warps_per_sm // num_warps)))
    rows_per_program = max(1, (n_rows + programs - 1) // programs)
    return max(1, (n_rows + rows_per_program - 1) // rows_per_program), rows_per_program


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

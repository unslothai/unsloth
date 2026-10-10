# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import warnings
import triton
import triton.language as tl
import torch
from typing import Tuple
from .utils import calculate_settings, long_indexing, torch_gpu_device


@triton.jit
def _rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    if LONG_INDEXING:
        row_idx = row_idx.to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm.
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)  # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)


def _rms_layernorm_backward(
    dY,
    dY_row_stride: tl.constexpr,
    dX,
    dX_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    GEMMA: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel for the backward pass
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    if LONG_INDEXING:
        row_idx = row_idx.to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    if GEMMA:
        dX += row_idx * dX_row_stride
    else:
        dX = dY

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA:
        dY_W = dY_row * (W_row + 1.0)
    else:
        dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dX + col_offsets, output, mask = mask)


_rms_layernorm_backward = triton.jit(_rms_layernorm_backward)
_rms_layernorm_backward = triton.heuristics(
    {
        "GEMMA": lambda args: bool(args["GEMMA"]),
        "LONG_INDEXING": lambda args: bool(args["LONG_INDEXING"]),
    }
)(_rms_layernorm_backward)


@triton.jit
def _gemma_rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Copies google-deepmind/gemma layers.py#L31 and keras-nlp gemma/rms_normalization.py#L33 exactly:
    # essentially all in float32.
    row_idx = tl.program_id(0)
    if LONG_INDEXING:
        row_idx = row_idx.to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm.
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask = mask)


@triton.jit
def _fold_lanes(x, ROWS: tl.constexpr, WARPS: tl.constexpr, HALF: tl.constexpr):
    return tl.sum(tl.reshape(x, (ROWS, WARPS, 2, HALF)), axis = 2)


@triton.jit
def _row_dot_in_row_kernel_order(a, b, ROWS: tl.constexpr, WARPS: tl.constexpr):
    # Bit-exact replay of the one-row kernels' tl.sum (4 warps, 64 <= BLOCK_SIZE <= 128):
    # fma(a_l, b_l, a_(l+16) * b_(l+16)), lane xor 8, 4, 2, 1, warp xor 2, 1; explicit steps only.
    a = tl.permute(tl.reshape(a, (ROWS, WARPS, 2, 16)), (0, 1, 3, 2))
    b = tl.permute(tl.reshape(b, (ROWS, WARPS, 2, 16)), (0, 1, 3, 2))
    a_lo, a_hi = tl.split(a)
    b_lo, b_hi = tl.split(b)
    acc = tl.fma(a_lo, b_lo, a_hi * b_hi)
    acc = _fold_lanes(acc, ROWS, WARPS, 8)
    acc = _fold_lanes(acc, ROWS, WARPS, 4)
    acc = _fold_lanes(acc, ROWS, WARPS, 2)
    acc = _fold_lanes(acc, ROWS, WARPS, 1)
    acc = tl.reshape(acc, (ROWS, WARPS))
    if WARPS == 4:
        acc = tl.sum(tl.reshape(acc, (ROWS, 2, 2)), axis = 1)
    return tl.sum(acc, axis = 1)


@triton.jit
def _rms_layernorm_forward_rows(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    r,
    n_rows,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    WARPS: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    pid = tl.program_id(0)
    if LONG_INDEXING:
        pid = pid.to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n_rows
    col_mask = col_offsets < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    X_rows = tl.load(
        X + rows[:, None] * X_row_stride + col_offsets[None, :], mask = mask, other = 0
    ).to(tl.float32)
    if GEMMA:
        W_row = tl.load(W + col_offsets, mask = col_mask, other = 0).to(tl.float32)
    else:
        W_row = tl.load(W + col_offsets, mask = col_mask, other = 0)

    row_var = _row_dot_in_row_kernel_order(X_rows, X_rows, ROWS, WARPS) / n_cols
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r + rows, inv_var, mask = row_mask)
    normed = X_rows * inv_var[:, None]
    if GEMMA:
        output = normed * (W_row[None, :] + 1.0)
    else:
        normed = normed.to(W_row.dtype)  # Exact copy from HF
        output = normed * W_row[None, :]
    tl.store(Y + rows[:, None] * Y_row_stride + col_offsets[None, :], output, mask = mask)


@triton.jit
def _rms_layernorm_backward_rows(
    dY,
    dY_row_stride: tl.constexpr,
    dX,
    dX_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    r,
    n_rows,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    WARPS: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    pid = tl.program_id(0)
    if LONG_INDEXING:
        pid = pid.to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n_rows
    col_mask = col_offsets < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    dY_rows = tl.load(
        dY + rows[:, None] * dY_row_stride + col_offsets[None, :], mask = mask, other = 0
    ).to(tl.float32)
    X_rows = tl.load(
        X + rows[:, None] * X_row_stride + col_offsets[None, :], mask = mask, other = 0
    ).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = col_mask, other = 0).to(tl.float32)

    inv_var = tl.load(r + rows, mask = row_mask, other = 0).to(tl.float32)
    normed = X_rows * inv_var[:, None]

    if GEMMA:
        dY_W = dY_rows * (W_row[None, :] + 1.0)
    else:
        dY_W = dY_rows * W_row[None, :]

    rowsum_dY_normed = _row_dot_in_row_kernel_order(dY_W, normed, ROWS, WARPS)
    # Spelled as in the one-row kernel: an explicit fma here turns -0 into +0 on sm75.
    output = (inv_var / n_cols)[:, None] * (n_cols * dY_W - normed * rowsum_dY_normed[:, None])
    if GEMMA:
        tl.store(dX + rows[:, None] * dX_row_stride + col_offsets[None, :], output, mask = mask)
    else:
        tl.store(dY + rows[:, None] * dY_row_stride + col_offsets[None, :], output, mask = mask)


# Off on ROCm (64-lane wavefronts reduce in another order) and before Triton 3.6 (slower backward).
def _triton_at_least(major, minor):
    try:
        return tuple(int(v) for v in triton.__version__.split(".")[:2]) >= (major, minor)
    except Exception:
        return False


_MULTIROW = (
    os.environ.get("UNSLOTH_RMSNORM_MULTIROW", "1") != "0"
    and not torch.version.hip
    and _triton_at_least(3, 6)
)
_MULTIROW_ELEMENTS = 4096
_MULTIROW_NUM_WARPS = 4


def _multirow_settings(n_cols):
    # Only widths 33..128 (one element per lane) can replay the one-row order; others stay one-row.
    if not _MULTIROW:
        return None
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if num_warps != 4 or not (64 <= BLOCK_SIZE <= 128):
        return None
    return BLOCK_SIZE, _MULTIROW_ELEMENTS // BLOCK_SIZE, BLOCK_SIZE // 32, _MULTIROW_NUM_WARPS


_MULTIROW_CHECKED = {}


def _multirow_disable(reason):
    global _MULTIROW
    if _MULTIROW:
        _MULTIROW = False
        warnings.warn(
            f"Unsloth: narrow-row RMSNorm kernels disabled ({reason}); using one row per program."
        )


def _multirow_must_raise(error, wrap):
    # Never fall back while torch.compile traces, nor on OOM or dynamo / inductor errors.
    return (
        wrap is not _eager_kernel
        or isinstance(error, torch.cuda.OutOfMemoryError)
        or type(error).__module__.startswith(("torch._dynamo", "torch._inductor"))
        or torch.compiler.is_compiling()
    )


def _bits(t):
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _multirow_self_check(device, dtype, W_dtype, n_cols, eps, gemma, multirow):
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes(), torch.no_grad(), torch_gpu_device(device):
        g = torch.Generator(device = device).manual_seed(3407)
        shape = (2 * multirow[1] + 3, n_cols)
        X = torch.randn(shape, device = device, generator = g)
        X = X * torch.exp2(torch.randint(-12, 13, shape, device = device, generator = g).float())
        dY = torch.randn(shape, device = device, generator = g)
        X[0], dY[0, ::2] = 0, 0  # all-zero row: signed zeros in dX
        X, dY = X.to(dtype), dY.to(dtype)
        W = torch.randn(n_cols, device = device, generator = g).to(W_dtype)
        results = []
        for rows in (False, multirow):
            Y, r = _rms_forward(X, W, eps, gemma, _eager_kernel, rows)
            dX = dY.clone()
            _rms_backward(dY.clone() if gemma else dX, dX, X, W, r, eps, gemma, _eager_kernel, rows)
            results.append((Y, r, dX))
        return all(torch.equal(_bits(a), _bits(b)) for a, b in zip(*results))


def _multirow_checked(X, W, eps, gemma):
    n_cols = X.shape[1]
    multirow = _multirow_settings(n_cols)
    if multirow is None:
        return None
    key = (X.device, X.dtype, W.dtype, int(n_cols), float(eps), gemma)
    verdict = _MULTIROW_CHECKED.get(key)
    if verdict is None:
        if X.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return None  # check on a later uncaptured call
        try:
            verdict = _multirow_self_check(X.device, X.dtype, W.dtype, n_cols, eps, gemma, multirow)
            reason = f"self-check mismatch for {key}"
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as error:
            # Its launches re-raise while torch.compile traces; this call then runs one row.
            verdict, reason = False, f"self-check failed for {key}: {error!r}"
        _MULTIROW_CHECKED[key] = verdict
        if not verdict:
            _multirow_disable(reason)
    return multirow if verdict and _MULTIROW else None


# Covers masked lanes past the last row: a block of columns, or ROWS rows in the multirow kernels.
_INDEX_MARGIN = 1 << 20


def _rms_forward(
    X,
    W,
    eps,
    gemma,
    wrap,
    multirow = None,
):
    # multirow: None picks (self-checked), False forces one row, settings force multi-row.
    n_rows, n_cols = X.shape
    Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = X.device)
    r = torch.empty(n_rows, dtype = torch.float32, device = X.device)
    if multirow is None:
        multirow = _multirow_checked(X, W, eps, gemma)
    long = long_indexing(X, Y, block = _INDEX_MARGIN)
    if multirow:
        BLOCK_SIZE, ROWS, WARPS, num_warps = multirow
        try:
            wrap(_rms_layernorm_forward_rows)[((n_rows + ROWS - 1) // ROWS,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                r,
                n_rows,
                n_cols,
                eps,
                GEMMA = gemma,
                BLOCK_SIZE = BLOCK_SIZE,
                ROWS = ROWS,
                WARPS = WARPS,
                # int32 rows measured slower only for the Gemma multi-row forward: keep main's int64.
                LONG_INDEXING = long or gemma,
                num_warps = num_warps,
            )
            return Y, r
        except Exception as error:
            if _multirow_must_raise(error, wrap):
                raise
            _multirow_disable(f"launch failed: {error!r}")
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
    wrap(fx)[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        r,
        r.stride(0),
        n_cols,
        eps,
        LONG_INDEXING = long,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = num_warps,
    )
    return Y, r


def _rms_backward(
    dY,
    dX,
    X,
    W,
    r,
    eps,
    gemma,
    wrap,
    multirow = None,
):
    # Non-Gemma writes dX over dY (the kernel ignores dX); Gemma writes into dX.
    n_rows, n_cols = dY.shape
    if multirow is None:
        multirow = _multirow_checked(X, W, eps, gemma)
    long = long_indexing(dY, dX, X, block = _INDEX_MARGIN)
    if multirow:
        BLOCK_SIZE, ROWS, WARPS, num_warps = multirow
        try:
            wrap(_rms_layernorm_backward_rows)[((n_rows + ROWS - 1) // ROWS,)](
                dY,
                dY.stride(0),
                dX,
                dX.stride(0),
                X,
                X.stride(0),
                W,
                r,
                n_rows,
                n_cols,
                eps,
                GEMMA = gemma,
                BLOCK_SIZE = BLOCK_SIZE,
                ROWS = ROWS,
                WARPS = WARPS,
                LONG_INDEXING = long,
                num_warps = num_warps,
            )
            return
        except Exception as error:
            if _multirow_must_raise(error, wrap):
                raise
            _multirow_disable(f"launch failed: {error!r}")
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    wrap(_rms_layernorm_backward)[(n_rows,)](
        dY,
        dY.stride(0),
        dX,
        dX.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        r,
        r.stride(0),
        n_cols,
        eps,
        GEMMA = gemma,
        LONG_INDEXING = long,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = num_warps,
    )


_eager_kernel = lambda kernel: kernel


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        W: torch.Tensor,
        eps: float,
        gemma: bool = False,
    ):
        shape = X.shape
        dim: int = shape[-1]
        X = X.reshape(-1, dim).contiguous()
        # kernels read W at unit stride, and this W is the one saved for backward.
        W = W.contiguous()
        with torch_gpu_device(X.device):
            Y, r = _rms_forward(X, W, eps, gemma, _eager_kernel)
        ctx.eps = eps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        shape = dY.shape
        dim: int = shape[-1]
        dY = dY.reshape(-1, dim).contiguous()
        X, W, r = ctx.saved_tensors
        dX = torch.empty_like(dY) if ctx.GEMMA else dY
        with torch_gpu_device(dY.device):
            _rms_backward(dY, dX, X, W, r, ctx.eps, ctx.GEMMA, _eager_kernel)
        dX = dX.view(*shape)
        return dX, None, None, None


# Compiled graphs take these ops: same kernels, fresh outputs (the Function writes dX over dY).
_TRACEABLE = hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton")


def _bf16_traceable():
    # Dynamo cannot trace bf16 Triton without native bf16 (T4); any visible GPU may hold a layer.
    try:
        if torch.version.hip:
            return True
        n = torch.cuda.device_count()
        return n > 0 and all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(n))
    except Exception:
        return False


_BF16_TRACEABLE = _TRACEABLE and _bf16_traceable()


def _tag_compile_cache(path):
    # Inductor's FX cache keys a triton_op without its source; key on the file so upgrades miss.
    config = getattr(torch.compiler, "config", None)
    if config is None or not hasattr(config, "cache_key_tag"):
        return
    with open(path, "rb") as file:
        tag = f"unsloth/{os.path.basename(path)}:{hashlib.sha256(file.read()).hexdigest()[:16]}"
    tags = [t for t in config.cache_key_tag.split(",") if t]
    if tag not in tags:
        config.cache_key_tag = ",".join(tags + [tag])


def _traced_kernel(kernel):
    # wrap_triton needs the JITFunction under triton.heuristics; callers pass its constexprs.
    if isinstance(kernel, triton.runtime.autotuner.Heuristics):
        kernel = kernel.fn
    return torch.library.wrap_triton(kernel)


if _TRACEABLE:

    @torch.library.triton_op("unsloth::rms_layernorm", mutates_args = ())
    def _rms_layernorm_op(
        X: torch.Tensor, W: torch.Tensor, eps: float, gemma: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Called eagerly (outside a compiled graph) the launch needs X's device made current.
        with torch_gpu_device(X.device):
            return _rms_forward(X, W, eps, gemma, _traced_kernel)

    @torch.library.triton_op("unsloth::rms_layernorm_backward", mutates_args = ())
    def _rms_layernorm_backward_op(
        dY: torch.Tensor, X: torch.Tensor, W: torch.Tensor, r: torch.Tensor, eps: float, gemma: bool
    ) -> torch.Tensor:
        dX = torch.empty_like(dY)
        # Non-Gemma rewrites its dY argument, so it gets a copy.
        with torch_gpu_device(dY.device):
            _rms_backward(dY if gemma else dX.copy_(dY), dX, X, W, r, eps, gemma, _traced_kernel)
        return dX

    def _rms_setup_context(ctx, inputs, output):
        X, W, eps, gemma = inputs
        ctx.eps, ctx.gemma = eps, gemma
        ctx.save_for_backward(X, W, output[1])

    def _rms_layernorm_op_backward(ctx, dY, dr):
        X, W, r = ctx.saved_tensors
        dX = torch.ops.unsloth.rms_layernorm_backward(dY.contiguous(), X, W, r, ctx.eps, ctx.gemma)
        return dX, None, None, None

    _rms_layernorm_op.register_autograd(
        _rms_layernorm_op_backward, setup_context = _rms_setup_context
    )
    _tag_compile_cache(__file__)


@torch.compiler.disable
def _fast_rms_layernorm_untraced(X, W, eps, gemma):
    return Fast_RMS_Layernorm.apply(X, W, eps, gemma)


def fast_rms_layernorm(
    layernorm,
    X: torch.Tensor,
    gemma: bool = False,
):
    W: torch.Tensor = layernorm.weight
    eps: float = (
        layernorm.variance_epsilon if hasattr(layernorm, "variance_epsilon") else layernorm.eps
    )
    if not torch.compiler.is_compiling():
        return Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    if (
        not _TRACEABLE
        or X.device.type != "cuda"
        or (X.dtype == torch.bfloat16 and not _BF16_TRACEABLE)
    ):
        return _fast_rms_layernorm_untraced(X, W, eps, gemma)
    shape = X.shape
    Y, _ = torch.ops.unsloth.rms_layernorm(
        X.reshape(-1, shape[-1]).contiguous(), W.contiguous(), eps, gemma
    )
    return Y.view(shape)


from transformers.models.llama.modeling_llama import LlamaRMSNorm


class Unsloth_LlamaRMSNorm(LlamaRMSNorm):
    def forward(self, X):
        return fast_rms_layernorm(self, X, gemma = False)


try:
    from transformers.models.mllama.modeling_mllama import MllamaTextRMSNorm
    class Unsloth_MllamaTextRMSNorm(MllamaTextRMSNorm):
        def forward(self, X):
            return fast_rms_layernorm(self, X, gemma = False)


except (ImportError, AttributeError):
    pass


def patch_rms_layernorm():
    import transformers.models.llama.modeling_llama

    transformers.models.llama.modeling_llama.LlamaRMSNorm = Unsloth_LlamaRMSNorm
    try:
        import transformers.models.mllama.modeling_mllama
        transformers.models.mllama.modeling_mllama.MllamaTextRMSNorm = Unsloth_MllamaTextRMSNorm
    except (ImportError, AttributeError, NameError):
        pass
    return


def unpatch_rms_layernorm():
    import transformers.models.llama.modeling_llama

    transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
    try:
        import transformers.models.mllama.modeling_mllama
        transformers.models.mllama.modeling_mllama.MllamaTextRMSNorm = MllamaTextRMSNorm
    except (ImportError, AttributeError, NameError):
        pass
    return


def test_rms_layernorm(
    dim = 1024,
    eps = 1e-5,
    dtype = torch.float16,
    bsz = 21,
    random_state = 3407,
    seqlen = 3341,
):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    torch.nn.init.uniform_(layernorm.weight)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    XX = X.clone()
    X.requires_grad_(True)
    XX.requires_grad_(True)
    Y = layernorm(X)
    YY = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda", requires_grad = True)
    Y.backward(YY)
    correct_grad = X.grad.clone()
    Y = fast_rms_layernorm(layernorm, XX)
    Y.backward(YY)
    assert torch.amax(correct_grad - XX.grad).item() <= 0.05


def testing_suite_layernorm():
    for dim in [512, 1024, 2048]:
        for dtype in [torch.float16, torch.bfloat16]:
            with torch.autocast(device_type = "cuda", dtype = dtype):
                for seqlen in [3341, 2048, 349]:
                    for random_state in [3407, 42]:
                        test_rms_layernorm(
                            dim = dim,
                            eps = 1e-5,
                            dtype = dtype,
                            bsz = 21,
                            random_state = random_state,
                            seqlen = seqlen,
                        )

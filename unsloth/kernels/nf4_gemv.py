# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Batch-1 NF4 GEMV for decoding, out[1, 1, N] = X[1, 1, K] @ dequant(W).T, in one Triton launch
# (nested absmax decode, NF4 lookup and dot product fused) on torch's current stream. Registered
# as unsloth::gemv_nf4 so torch.compile traces it and CUDA graphs capture it. Each quantization
# block is summed before scaling, so it matches bitsandbytes to fp32 accumulation noise, not bitwise.
# Every weight row must start a new quantization block (K % blocksize == 0); callers check.

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

from .nf4 import _HAS_MUL_RN, _mul
from .triton_launch import launch, tag_compile_cache

__all__ = [
    "gemv_nf4",
    "triton_gemv_eager",
]


@triton.jit
def _gemv_nf4_kernel(
    X,
    W,
    ABSMAX,
    CODE2,
    ABSMAX2,
    OFFSET,
    LUT,
    OUT,
    N,
    K,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE2: tl.constexpr,
    NESTED: tl.constexpr,
    USE_MUL_RN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # FSDP-QLoRA packs the 4bit weight into float storage; it is bytes either way.
    W = W.to(tl.pointer_type(tl.uint8))
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    HALF_K: tl.constexpr = BLOCK_K // 2
    NB: tl.constexpr = BLOCK_K // BLOCKSIZE
    half_k = tl.arange(0, HALF_K)
    ks = tl.arange(0, BLOCK_K)
    nbs = tl.arange(0, NB)
    blocks_per_row = K // BLOCKSIZE
    acc = tl.zeros([BLOCK_N], dtype = tl.float32)
    if NESTED:
        offset = tl.load(OFFSET)
    for k0 in range(0, K, BLOCK_K):
        kmask = (k0 + 2 * half_k) < K
        m = rmask[:, None] & kmask[None, :]
        b = tl.load(W + rows64[:, None] * (K // 2) + (k0 // 2 + half_k)[None, :], mask = m, other = 0)
        b = b.to(tl.int32)
        # High nibble is the earlier element, as bitsandbytes packs it.
        hi = tl.load(LUT + (b >> 4))
        lo = tl.load(LUT + (b & 15))
        w = tl.reshape(tl.join(hi, lo), (BLOCK_N, BLOCK_K))
        xk = k0 + ks
        x = tl.load(X + xk, mask = xk < K, other = 0.0).to(tl.float32)
        part = tl.sum(tl.reshape(w * x[None, :], (BLOCK_N, NB, BLOCKSIZE)), axis = 2)
        bmask = rmask[:, None] & ((k0 + nbs * BLOCKSIZE) < K)[None, :]
        blk = rows64[:, None] * blocks_per_row + (k0 // BLOCKSIZE + nbs)[None, :]
        if NESTED:
            q = tl.load(ABSMAX + blk, mask = bmask, other = 0).to(tl.int32)
            c2 = tl.load(CODE2 + q, mask = bmask, other = 0.0)
            s2 = tl.load(ABSMAX2 + blk // BLOCKSIZE2, mask = bmask, other = 0.0)
            # Two separate fp32 roundings as in fast_dequantize, even with fp fusion on.
            a = _mul(c2, s2, USE_MUL_RN)
            a = a + offset
        else:
            a = tl.load(ABSMAX + blk, mask = bmask, other = 0.0)
        acc += tl.sum(part * a, axis = 1)
    tl.store(OUT + rows, acc.to(OUT.dtype.element_ty), mask = rmask)


@triton.jit
def _gemv_nf4_words_kernel(
    X,
    W,
    ABSMAX,
    CODE2,
    ABSMAX2,
    OFFSET,
    LUT,
    OUT,
    N,
    K,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE2: tl.constexpr,
    NESTED: tl.constexpr,
    USE_MUL_RN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # _gemv_nf4_kernel reading int32 words of 8 nibbles; faster before Triton 3.7, slower on 3.7.
    W = W.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    WORDS: tl.constexpr = BLOCK_K // 8
    NB: tl.constexpr = BLOCK_K // BLOCKSIZE
    wk = tl.arange(0, WORDS)
    j = tl.arange(0, 8)
    # Little endian: byte b of a word holds values 2b (high nibble) and 2b + 1 (low nibble).
    shifts = (j // 2) * 8 + (1 - (j % 2)) * 4
    nbs = tl.arange(0, NB)
    words_per_row = K // 8
    blocks_per_row = K // BLOCKSIZE
    acc = tl.zeros([BLOCK_N], dtype = tl.float32)
    if NESTED:
        offset = tl.load(OFFSET)
    for k0 in range(0, K, BLOCK_K):
        kw = k0 // 8 + wk
        m = rmask[:, None] & (kw < words_per_row)[None, :]
        w = tl.load(W + rows64[:, None] * words_per_row + kw[None, :], mask = m, other = 0)
        v = tl.load(LUT + ((w[:, :, None] >> shifts[None, None, :]) & 15))
        xk = k0 + wk[:, None] * 8 + j[None, :]
        x = tl.load(X + xk, mask = xk < K, other = 0.0).to(tl.float32)
        part = tl.sum(tl.reshape(v * x[None, :, :], (BLOCK_N, NB, BLOCKSIZE)), axis = 2)
        bmask = rmask[:, None] & ((k0 + nbs * BLOCKSIZE) < K)[None, :]
        blk = rows64[:, None] * blocks_per_row + (k0 // BLOCKSIZE + nbs)[None, :]
        if NESTED:
            q = tl.load(ABSMAX + blk, mask = bmask, other = 0).to(tl.int32)
            c2 = tl.load(CODE2 + q, mask = bmask, other = 0.0)
            s2 = tl.load(ABSMAX2 + blk // BLOCKSIZE2, mask = bmask, other = 0.0)
            a = _mul(c2, s2, USE_MUL_RN)
            a = a + offset
        else:
            a = tl.load(ABSMAX + blk, mask = bmask, other = 0.0)
        acc += tl.sum(part * a, axis = 1)
    tl.store(OUT + rows, acc.to(OUT.dtype.element_ty), mask = rmask)


_TRITON_37 = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 7)
# None picks per GPU and Triton version below; "bytes" or "words" forces one kernel (tests, sweeps).
_FORCE_KERNEL = None


def triton_gemv_eager(device = None):
    """Whether eager (uncompiled) decode should take the Triton GEMV. On CUDA it does: with the
    direct launch (triton_launch.py) it beat bitsandbytes' GEMV in batch-1 decode on T4, L4, A100,
    RTX PRO 6000 and B200. HIP launches through Triton's own path, which only paid off from
    Triton 3.7."""
    return torch.version.hip is None or _TRITON_37


def _word_aligned(W):
    # Rows are K / 2 bytes (K % blocksize == 0), so only a view can break 4 byte alignment.
    return (W.storage_offset() * W.element_size()) % 4 == 0


@functools.lru_cache(maxsize = None)
def _gemv_config(N: int, K: int, blocksize: int, major: int, words_ok: bool, force):
    """(use the words kernel, grid, BLOCK_N, BLOCK_K, num_warps).

    Kernel time (CUDA graphs) over Llama 1B to 70B shapes: before Triton 3.7 the words kernel at
    one row per program and one warp beat bitsandbytes on T4, L4, A100, RTX PRO 6000 and B200
    (1.1x to 2.2x, lower error too); Triton 3.7 compiles the byte kernel better, best at 4 rows, BLOCK_K 2048 and
    4 warps (measured on B200)."""
    block_k = max(blocksize, min(2048, triton.next_power_of_2(K)))
    use_words = words_ok and (force == "words" or (force is None and not _TRITON_37))
    if use_words:
        if major == 7 and N * K >= (1 << 25):
            # T4: shorter K steps won on the 14336x4096, 4096x14336 and vocab shapes.
            block_k = max(blocksize, min(512, block_k))
        return True, (N,), 1, block_k, 1
    if force is None:
        return False, (-(-N // 4),), 4, block_k, 4
    # Forced byte kernel (tests): the pre-3.7 byte config.
    block_k = max(blocksize, min(1024, triton.next_power_of_2(K)))
    block_n, num_warps = (2, 4) if N <= 2048 else (4, 2)
    return False, (-(-N // block_n),), block_n, block_k, num_warps


@functools.lru_cache(maxsize = None)
def _major(index):
    return torch.cuda.get_device_capability(index)[0]


def _launch(kernels, X, W, absmax, code2, absmax2, offset, code, out, N, K, blocksize, blocksize2):
    nested = code2 is not None
    use_words, grid, block_n, block_k, num_warps = _gemv_config(
        N, K, blocksize, _major(X.device.index), _word_aligned(W), _FORCE_KERNEL
    )
    kernel = kernels[1] if use_words else kernels[0]
    args = (
        X,
        W,
        absmax,
        code2 if nested else absmax,
        absmax2 if nested else absmax,
        offset if nested else absmax,
        code,
        out,
        N,
        K,
    )
    constexprs = dict(
        BLOCKSIZE = blocksize,
        BLOCKSIZE2 = blocksize2 if nested else 1,
        NESTED = nested,
        USE_MUL_RN = _HAS_MUL_RN,
        BLOCK_N = block_n,
        BLOCK_K = block_k,
    )
    if kernel is _gemv_nf4_kernel or kernel is _gemv_nf4_words_kernel:
        launch(kernel, grid, args, 8, constexprs, X.device.index, num_warps = num_warps)
    else:
        kernel[grid](*args, **constexprs, num_warps = num_warps)
    return out


@torch.library.triton_op("unsloth::gemv_nf4", mutates_args = ())
def _gemv_nf4_op(
    X: torch.Tensor,
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: Optional[torch.Tensor],
    absmax2: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    code: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    out = torch.empty((1, 1, out_features), dtype = X.dtype, device = X.device)
    return _launch(
        (
            torch.library.wrap_triton(_gemv_nf4_kernel),
            torch.library.wrap_triton(_gemv_nf4_words_kernel),
        ),
        X,
        W,
        absmax,
        code2,
        absmax2,
        offset,
        code,
        out,
        out_features,
        in_features,
        blocksize,
        blocksize2,
    )


_KERNELS = (_gemv_nf4_kernel, _gemv_nf4_words_kernel)
tag_compile_cache(__file__)
_is_compiling = torch.compiler.is_compiling
_current_device = torch.cuda.current_device


def gemv_nf4(
    X,
    W_u8,
    absmax,
    code2,
    absmax2,
    offset,
    code,
    blocksize,
    blocksize2,
    shape,
    dtype,
    out = None,
):
    """X @ dequant(W).T for a single token; returns (1, 1, shape[0]) in X's dtype.

    ``code2/absmax2/offset`` are None for a quant state without double quantization, and
    ``offset`` is bitsandbytes' 0-dim device tensor (read in the kernel, so no host sync).
    Eager calls launch the kernel directly: the torch.library dispatcher costs about 20us
    per call, as much as the kernel itself at decode sizes. Compiled code goes through
    the registered op so Dynamo sees it.
    """
    out_features, in_features = int(shape[0]), int(shape[1])
    if not X.is_contiguous():
        X = X.contiguous()
    blocksize2 = int(blocksize2) if code2 is not None else 0
    if _is_compiling():
        result = _gemv_nf4_op(
            X,
            W_u8,
            absmax,
            code2,
            absmax2,
            offset,
            code,
            int(blocksize),
            blocksize2,
            out_features,
            in_features,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result
    if out is None:
        out = torch.empty((1, 1, out_features), dtype = X.dtype, device = X.device)
    args = (
        _KERNELS,
        X,
        W_u8,
        absmax,
        code2,
        absmax2,
        offset,
        code,
        out,
        out_features,
        in_features,
        int(blocksize),
        blocksize2,
    )
    # Triton launches on the current device's current stream, so only switch when X lives elsewhere.
    if X.device.index == _current_device():
        return _launch(*args)
    with torch.cuda.device(X.device):
        return _launch(*args)

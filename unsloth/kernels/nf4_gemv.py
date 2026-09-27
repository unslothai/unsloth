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

from .nf4 import _HAS_MUL_RN

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
            # Two separate fp32 roundings, as fast_dequantize does. PTX mul.rn is never contracted
            # into an FMA, so the scale is the same whether or not the compiled graph re-emits
            # this kernel with fp fusion on.
            if USE_MUL_RN:
                a = tl.inline_asm_elementwise(
                    "mul.rn.f32 $0, $1, $2;",
                    "=r,r,r",
                    [c2, s2],
                    dtype = tl.float32,
                    is_pure = True,
                    pack = 1,
                )
            else:
                a = c2 * s2
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
    # Same math as _gemv_nf4_kernel, but the weight is read as int32 words of 8 nibbles and
    # unpacked with one shift per value. Triton before 3.7 builds this form about 2x faster than
    # the byte tile above (and 3.7 the other way round), so the config picks per Triton version.
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
            if USE_MUL_RN:
                a = tl.inline_asm_elementwise(
                    "mul.rn.f32 $0, $1, $2;",
                    "=r,r,r",
                    [c2, s2],
                    dtype = tl.float32,
                    is_pure = True,
                    pack = 1,
                )
            else:
                a = c2 * s2
            a = a + offset
        else:
            a = tl.load(ABSMAX + blk, mask = bmask, other = 0.0)
        acc += tl.sum(part * a, axis = 1)
    tl.store(OUT + rows, acc.to(OUT.dtype.element_ty), mask = rmask)


_TRITON_37 = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 7)
# None picks per GPU and Triton version below; "bytes" or "words" forces one kernel (tests, sweeps).
_FORCE_KERNEL = None


# Compute capabilities (major, minor) where the Triton GEMV beats bitsandbytes' in eager decode even
# before Triton 3.7. Empty until measured there, so eager decode keeps bitsandbytes' GEMV.
EAGER_BEFORE_TRITON_37 = frozenset()


def triton_gemv_eager(device = None):
    """Whether eager (uncompiled) decode should take the Triton GEMV on this Triton and GPU."""
    if _TRITON_37:
        return True
    if not EAGER_BEFORE_TRITON_37:
        return False
    return tuple(torch.cuda.get_device_capability(device)) in EAGER_BEFORE_TRITON_37


def _word_aligned(W):
    # Each row is K / 2 bytes with K a multiple of the blocksize, so rows stay 4 byte aligned when
    # the tensor starts on one. The storage base always is; only a view can shift it.
    return (W.storage_offset() * W.element_size()) % 4 == 0


@functools.lru_cache(maxsize = None)
def _gemv_config(N: int, K: int, blocksize: int, major: int, words_ok: bool, force):
    """(use the words kernel, grid, BLOCK_N, BLOCK_K, num_warps).

    Timed with CUDA graphs (kernel time only) on a B200 over (4096,4096) (14336,4096)
    (4096,14336) (1024,4096): Triton 3.7 runs the byte kernel best at BLOCK_K=2048 and 4 warps,
    Triton 3.6 the words kernel at one row per program. Other GPUs keep the original configs
    until they are measured."""
    use_words = words_ok and (force == "words" or (force is None and major == 10 and not _TRITON_37))
    if use_words:
        block_k = max(blocksize, min(2048, triton.next_power_of_2(K)))
        block_n, num_warps = 1, 1
        return True, (triton.cdiv(N, block_n),), block_n, block_k, num_warps
    if major == 10 and force is None:
        block_k = max(blocksize, min(2048, triton.next_power_of_2(K)))
        block_n, num_warps = 4, 4
        return False, (triton.cdiv(N, block_n),), block_n, block_k, num_warps
    # Swept on a B200 over the shapes above: BLOCK_K=1024 won every shape; few rows per program
    # for small N, 4 rows and 2 warps for tall N.
    block_k = max(blocksize, min(1024, triton.next_power_of_2(K)))
    if N <= 2048:
        block_n, num_warps = 2, 4
    elif N <= 8192 and K <= 8192:
        block_n, num_warps = 8, 4
    else:
        block_n, num_warps = 4, 2
    return False, (triton.cdiv(N, block_n),), block_n, block_k, num_warps


@functools.lru_cache(maxsize = None)
def _major(index):
    return torch.cuda.get_device_capability(index)[0]


def _launch(kernels, X, W, absmax, code2, absmax2, offset, code, out, N, K, blocksize, blocksize2):
    nested = code2 is not None
    use_words, grid, block_n, block_k, num_warps = _gemv_config(
        N, K, blocksize, _major(X.device.index), _word_aligned(W), _FORCE_KERNEL
    )
    kernel = kernels[1] if use_words else kernels[0]
    kernel[grid](
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
        BLOCKSIZE = blocksize,
        BLOCKSIZE2 = blocksize2 if nested else 1,
        NESTED = nested,
        USE_MUL_RN = _HAS_MUL_RN,
        BLOCK_N = block_n,
        BLOCK_K = block_k,
        num_warps = num_warps,
    )
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

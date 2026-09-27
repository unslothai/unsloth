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

"""Batch-1 NF4 GEMV for decoding: out[1, 1, N] = X[1, 1, K] @ dequant(W).T

Two implementations share one signature:

gemv_nf4        Triton. One launch: the nested absmax decode, the NF4 lookup and the dot
                product are fused. Launches on torch's current stream, and is registered as
                ``unsloth::gemv_nf4`` through ``torch.library.triton_op`` so torch.compile
                traces it without a graph break and CUDA graphs can capture it.
gemv_nf4_bnb    bitsandbytes' cgemm_4bit_inference_naive through ctypes, byte-identical to the
                historical fast_gemv, but reading the live stream at call time. Registered as
                the opaque custom op ``unsloth::gemv_nf4_bnb`` so it also compiles with 0 breaks.

The Triton kernel sums each quantization block before scaling it, which is a different
reduction order from bitsandbytes, so the two agree to fp32 accumulation noise, not bitwise.
"""

import ctypes
import functools
from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = [
    "gemv_nf4",
    "gemv_nf4_bnb",
    "triton_gemv_supported",
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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
            # Two separate fp32 roundings, as fast_dequantize does (launched without FMA fusion).
            a = tl.load(CODE2 + q, mask = bmask, other = 0.0) * tl.load(
                ABSMAX2 + blk // BLOCKSIZE2, mask = bmask, other = 0.0
            )
            a = a + offset
        else:
            a = tl.load(ABSMAX + blk, mask = bmask, other = 0.0)
        acc += tl.sum(part * a, axis = 1)
    tl.store(OUT + rows, acc.to(OUT.dtype.element_ty), mask = rmask)


@functools.lru_cache(maxsize = None)
def _gemv_config(N: int, K: int, blocksize: int):
    # Swept on a B200 over (4096,4096) (14336,4096) (4096,14336) (1024,4096) (128256,4096):
    # BLOCK_K=1024 won every shape; few rows per program for small N, 4 rows and 2 warps for tall N.
    block_k = max(blocksize, min(1024, triton.next_power_of_2(K)))
    if N <= 2048:
        block_n, num_warps = 2, 4
    elif N <= 8192 and K <= 8192:
        block_n, num_warps = 8, 4
    else:
        block_n, num_warps = 4, 2
    return (triton.cdiv(N, block_n),), block_n, block_k, num_warps


def triton_gemv_supported(K: int, blocksize: int) -> bool:
    """The kernel indexes absmax per row, so every row must start a new quantization block."""
    return blocksize >= 2 and (blocksize & (blocksize - 1)) == 0 and K % blocksize == 0


def _launch(kernel, X, W, absmax, code2, absmax2, offset, code, out, N, K, blocksize, blocksize2):
    nested = code2 is not None
    grid, block_n, block_k, num_warps = _gemv_config(N, K, blocksize)
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
        BLOCK_N = block_n,
        BLOCK_K = block_k,
        num_warps = num_warps,
        enable_fp_fusion = False,
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
        torch.library.wrap_triton(_gemv_nf4_kernel),
        X, W, absmax, code2, absmax2, offset, code, out,
        out_features, in_features, blocksize, blocksize2,
    )


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
            X, W_u8, absmax, code2, absmax2, offset, code,
            int(blocksize), blocksize2, out_features, in_features,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result
    if out is None:
        out = torch.empty((1, 1, out_features), dtype = X.dtype, device = X.device)
    args = (
        _gemv_nf4_kernel, X, W_u8, absmax, code2, absmax2, offset, code, out,
        out_features, in_features, int(blocksize), blocksize2,
    )
    # Triton launches on the current device's current stream, so only switch when X lives elsewhere.
    if X.device.index == _current_device():
        return _launch(*args)
    with torch.cuda.device(X.device):
        return _launch(*args)


# bitsandbytes naive GEMV, live stream
_bnb_lib = None
_c_int = ctypes.c_int
_c_int32 = ctypes.c_int32
_c_void_p = ctypes.c_void_p


def _bnb():
    # Imported on first use so this module loads without bitsandbytes (the Triton path needs none).
    global _bnb_lib
    if _bnb_lib is None:
        import bitsandbytes.functional as bnb_functional

        _bnb_lib = bnb_functional.lib
    return _bnb_lib


def _ptr(t):
    return None if t is None else _c_void_p(t.data_ptr())


@torch.library.custom_op("unsloth::gemv_nf4_bnb", mutates_args = ())
def _gemv_nf4_bnb_op(
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
    lib = _bnb()
    device = X.device
    out = torch.empty((1, 1, out_features), dtype = X.dtype, device = device)
    with torch.cuda.device(device):
        stream = _c_void_p(torch._C._cuda_getCurrentRawStream(device.index))
        if code2 is not None:
            df = torch.empty(absmax.shape, dtype = torch.float32, device = device)
            lib.cdequantize_blockwise_fp32(
                _ptr(code2),
                _ptr(absmax),
                _ptr(absmax2),
                _ptr(df),
                _c_int(blocksize2),
                _c_int(df.numel()),
                stream,
            )
            df += offset
            absmax = df
        fx = (
            lib.cgemm_4bit_inference_naive_fp16
            if X.dtype == torch.float16
            else lib.cgemm_4bit_inference_naive_bf16
        )
        fx(
            _c_int32(out_features),
            _c_int32(1),
            _c_int32(in_features),
            _ptr(X),
            _ptr(W),
            _ptr(absmax),
            _ptr(code),
            _ptr(out),
            _c_int32(out_features),
            _c_int32((in_features + 1) // 2),
            _c_int32(out_features),
            _c_int32(blocksize),
            stream,
        )
    return out


@_gemv_nf4_bnb_op.register_fake
def _(X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, out_features, in_features):
    return X.new_empty((1, 1, out_features))


def gemv_nf4_bnb(
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
    """Same contract as gemv_nf4, computed by bitsandbytes' naive GEMV."""
    if X.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Unsloth: 4bit GEMV supports float16 and bfloat16, not {X.dtype}.")
    result = _gemv_nf4_bnb_op(
        X,
        W_u8,
        absmax,
        code2,
        absmax2,
        offset,
        code,
        int(blocksize),
        int(blocksize2) if blocksize2 is not None else 0,
        int(shape[0]),
        int(shape[1]),
    )
    if out is not None:
        out.copy_(result)
        return out
    return result

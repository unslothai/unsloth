# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Triton kernels for compressed-tensors ``pack-quantized`` integer weights kept packed.

Layout (compressed_tensors/compressors/pack_quantized/helpers.py): ``weight_packed`` is int32
``[N, ceil(K / pf)]`` with ``pf = 32 // bits`` values per word, element ``j`` in bits
``[j * bits, (j + 1) * bits)``, stored as ``q + 2 ** (bits - 1)``. ``weight_scale`` is ``[N, G]``;
an asymmetric ``weight_zero_point`` is packed the same way along the OUTPUT dim ``[ceil(N / pf), G]``;
act-order checkpoints carry ``weight_g_idx`` ``[K]`` (column -> group). The decode
``(q - zp) * scale`` is exact in fp32, then rounded like compressed-tensors (scale dtype, then output).
"""

import functools

import torch
import triton
import triton.language as tl

__all__ = [
    "Int4QuantState",
    "int4_dequantize",
    "int4_dequantize_weight",
    "int4_matmul",
    "int4_matmul_t",
]


class Int4QuantState:
    """Carried as ``weight.quant_state`` so Unsloth's LoRA kernels dispatch like bitsandbytes."""

    __slots__ = (
        "scale",
        "zero_point",
        "g_idx",
        "shape",
        "bits",
        "group_size",
        "dtype",
        "_launch",
        "_launchers",
    )

    def __init__(self, scale, zero_point, g_idx, shape, bits, group_size, dtype):
        self.scale = scale
        self.zero_point = zero_point
        self.g_idx = g_idx
        self.shape = torch.Size(shape)
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.dtype = dtype
        self._launch = None
        self._launchers = None


@triton.jit
def _as_float(v):
    # Exact int -> fp32 for |v| < 2**22 without the quarter-rate cvt: place v in the mantissa of 2**23.
    return ((v + 0x4B400000).to(tl.float32, bitcast = True)) - 12582912.0


@triton.jit
def _unpack_tile3(
    P,
    S,
    Z,
    G,
    rn,
    rkp,
    N,
    K,
    KP,
    stride_pn,
    stride_sn,
    stride_zn,
    BITS: tl.constexpr,
    PF: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_G: tl.constexpr,
):
    # fp32 decode of W[rn, words rkp] as [BLOCK_N, BLOCK_KP, PF] (column = word * PF + j).
    MASK: tl.constexpr = (1 << BITS) - 1
    n_ok = rn < N
    kp_ok = rkp < KP
    words = tl.load(
        P + rn[:, None] * stride_pn + rkp[None, :],
        mask = n_ok[:, None] & kp_ok[None, :],
        other = 0,
    )
    j = tl.arange(0, PF)
    q = (words[:, :, None] >> (j * BITS)[None, None, :]) & MASK
    if HAS_G:
        # act-order: every column has its own group.
        k3 = rkp[:, None] * PF + j[None, :]
        k3_ok = k3 < K
        g3 = tl.load(G + k3, mask = k3_ok, other = 0)
        m3 = n_ok[:, None, None] & k3_ok[None, :, :]
        s = tl.load(S + rn[:, None, None] * stride_sn + g3[None, :, :], mask = m3, other = 0.0)
        if HAS_Z:
            zw = tl.load(
                Z + (rn // PF)[:, None, None] * stride_zn + g3[None, :, :], mask = m3, other = 0
            )
            z = (zw >> ((rn % PF) * BITS)[:, None, None]) & MASK
        else:
            z = 1 << (BITS - 1)
        w = _as_float(q - z) * s.to(tl.float32)
    else:
        # GROUP % PF == 0 (checked on the host): one group per packed word.
        g = (rkp * PF) // GROUP
        m2 = n_ok[:, None] & kp_ok[None, :]
        s = tl.load(S + rn[:, None] * stride_sn + g[None, :], mask = m2, other = 0.0)
        if HAS_Z:
            zw = tl.load(Z + (rn // PF)[:, None] * stride_zn + g[None, :], mask = m2, other = 0)
            z = (zw >> ((rn % PF) * BITS)[:, None]) & MASK
            zf = _as_float(z)
            w = (_as_float(q) - zf[:, :, None]) * s.to(tl.float32)[:, :, None]
        else:
            w = (_as_float(q) - (1 << (BITS - 1))) * s.to(tl.float32)[:, :, None]
    return w


@triton.jit
def _unpack_tile(
    P,
    S,
    Z,
    G,
    rn,
    rkp,
    rk,
    N,
    K,
    KP,
    stride_pn,
    stride_sn,
    stride_zn,
    BITS: tl.constexpr,
    PF: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KP: tl.constexpr,
):
    w = _unpack_tile3(
        P, S, Z, G, rn, rkp, N, K, KP, stride_pn, stride_sn, stride_zn,
        BITS, PF, GROUP, HAS_Z, HAS_G,
    )  # fmt: skip
    return tl.reshape(w, (BLOCK_N, BLOCK_KP * PF))


@triton.jit
def _dequant_kernel(
    P,
    S,
    Z,
    G,
    Out,
    N,
    K,
    KP,
    stride_pn,
    stride_sn,
    stride_zn,
    stride_on,
    BITS: tl.constexpr,
    PF: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rkp = pid_k * BLOCK_KP + tl.arange(0, BLOCK_KP)
    rk = pid_k * (BLOCK_KP * PF) + tl.arange(0, BLOCK_KP * PF)
    w = _unpack_tile(
        P, S, Z, G, rn, rkp, rk, N, K, KP, stride_pn, stride_sn, stride_zn,
        BITS, PF, GROUP, HAS_Z, HAS_G, BLOCK_N, BLOCK_KP,
    )  # fmt: skip
    # compressed-tensors multiplies in the scale dtype, then casts: round the same way twice.
    tl.store(
        Out + rn[:, None] * stride_on + rk[None, :],
        w.to(S.dtype.element_ty).to(Out.dtype.element_ty),
        mask = (rn < N)[:, None] & (rk < K)[None, :],
    )


@triton.jit
def _gemv_kernel(
    X,
    P,
    S,
    Z,
    G,
    Y,
    M,
    N,
    K,
    KP,
    stride_xm,
    stride_pn,
    stride_sn,
    stride_zn,
    stride_ym,
    BITS: tl.constexpr,
    PF: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KP: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Decode: one activation row per program axis 2 (M tiny), CUDA-core reduction.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    m = tl.program_id(2)
    BLOCK_K: tl.constexpr = BLOCK_KP * PF
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype = tl.float32)
    n_k = tl.cdiv(KP, BLOCK_KP)
    j = tl.arange(0, PF)
    for kb in range(pid_k, n_k, SPLIT_K):
        rkp = kb * BLOCK_KP + tl.arange(0, BLOCK_KP)
        k2 = rkp[:, None] * PF + j[None, :]
        x = tl.load(X + m * stride_xm + k2, mask = k2 < K, other = 0.0).to(tl.float32)
        w = _unpack_tile3(
            P, S, Z, G, rn, rkp, N, K, KP, stride_pn, stride_sn, stride_zn,
            BITS, PF, GROUP, HAS_Z, HAS_G,
        )  # fmt: skip
        acc += tl.sum(tl.sum(w * x[None, :, :], axis = 2), axis = 1)
    ptrs = Y + m * stride_ym + rn
    if SPLIT_K == 1:
        tl.store(ptrs, acc.to(Y.dtype.element_ty), mask = rn < N)
    else:
        tl.atomic_add(ptrs, acc, mask = rn < N, sem = "relaxed")


def _launch_args(packed, qs):
    """``((P, S, Z, G), (N, K, KP), strides, constexprs)``, cached on the quant state (decode is launch bound)."""
    cached = qs._launch
    if cached is not None and cached[0] == packed.data_ptr():
        return cached[1]
    N, K = qs.shape
    bits = qs.bits
    pf = 32 // bits
    group = qs.group_size if qs.group_size > 0 else K
    if qs.g_idx is None and group % pf != 0:
        # A word would straddle two groups: address the groups per column instead.
        qs.g_idx = torch.arange(K, device = packed.device, dtype = torch.int32) // group
    zp, g_idx, scale = qs.zero_point, qs.g_idx, qs.scale
    Z = zp if zp is not None else packed
    G = g_idx if g_idx is not None else packed
    args = (
        (packed, scale, Z, G),
        (N, K, packed.shape[1]),
        (packed.stride(0), scale.stride(0), Z.stride(0) if zp is not None else 0),
        dict(BITS = bits, PF = pf, GROUP = group, HAS_Z = zp is not None, HAS_G = g_idx is not None),
    )
    qs._launch = (packed.data_ptr(), args)
    qs._launchers = None
    return args


_args_tail = _launch_args


class _on_device:
    """``torch.cuda.device`` only when ``device`` is not already current (the context costs a few us)."""

    __slots__ = ("ctx",)

    def __init__(self, device):
        self.ctx = None
        index = device.index
        if index is not None and index != torch.cuda.current_device():
            self.ctx = torch.cuda.device(index)

    def __enter__(self):
        if self.ctx is not None:
            self.ctx.__enter__()

    def __exit__(self, *exc):
        if self.ctx is not None:
            self.ctx.__exit__(*exc)


def int4_dequantize(packed, qs, dtype = None, out = None):
    """``[N, K]`` dense decode of a packed weight (exact)."""
    dtype = dtype or qs.dtype or torch.bfloat16
    (P, S, Z, G), (N, K, KP), (sp, ss, sz), meta = _launch_args(packed, qs)
    if out is None:
        out = torch.empty((N, K), dtype = dtype, device = packed.device)
    BLOCK_N, BLOCK_KP = 16, max(1, 512 // meta["PF"])
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(KP, BLOCK_KP), 1)
    args = (P, S, Z, G, out, N, K, KP, sp, ss, sz, out.stride(0))
    key = (packed.device.index, "dq", out.dtype, out.data_ptr() % 16 == 0, out.stride(0) % 16 == 0)
    launchers = qs._launchers
    if launchers is None:
        launchers = qs._launchers = {}
    with _on_device(packed.device):
        compiled = launchers.get(key)
        if compiled is None:
            compiled = _dequant_kernel.warmup(
                *args, grid = grid, BLOCK_N = BLOCK_N, BLOCK_KP = BLOCK_KP, num_warps = 4, **meta
            )
            launchers[key] = compiled
        compiled[grid](*args)
    return out


def int4_dequantize_weight(W, qs, dtype = None):
    """``fast_dequantize`` contract: ``W`` is the packed weight or its ``.t()``; returns the matching dense view."""
    if W.stride(-1) != 1 and W.dim() == 2 and W.shape[1] == qs.shape[0]:
        return int4_dequantize(W.t(), qs, dtype).t()
    return int4_dequantize(W, qs, dtype)


@functools.lru_cache(maxsize = None)
def _sm_count(index):
    return torch.cuda.get_device_properties(index).multi_processor_count


# Measured on B200 against dequantize + cuBLAS and bitsandbytes' gemv (bench_kernels.py): the fused
# CUDA-core kernel only pays off for a few rows; past that cuBLAS on the exact decode wins.
GEMV_MAX_ROWS = 4


def int4_matmul(x, packed, qs, out = None):
    """``x @ W.T`` for packed ``W``; ``x`` is ``[..., K]``."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    M = x2.shape[0]
    N = qs.shape[0]
    if M > GEMV_MAX_ROWS:
        W = int4_dequantize(packed, qs, x.dtype)
        y = torch.matmul(x2, W.t(), out = None if out is None else out.view(M, N))
        return y.view(*shape[:-1], N)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    (P, S, Z, G), (N, K, KP), (sp, ss, sz), meta = _launch_args(packed, qs)
    BLOCK_N, BLOCK_KP = 8, max(1, 512 // meta["PF"])
    n_blocks = triton.cdiv(N, BLOCK_N)
    # Split K only when the rows alone cannot fill the GPU (the fp32 atomics cost two extra launches).
    sms = _sm_count(x.device.index or 0)
    split = 1 if n_blocks * M >= sms else min(4, triton.cdiv(sms, n_blocks * M))
    if split > 1:
        y = torch.zeros((M, N), dtype = torch.float32, device = x.device)
    elif out is not None and out.is_contiguous() and out.dtype == x.dtype:
        y = out.view(M, N)
    else:
        y = torch.empty((M, N), dtype = x.dtype, device = x.device)
    grid = (n_blocks, split, M)
    args = (x2, P, S, Z, G, y, M, N, K, KP, x2.stride(0), sp, ss, sz, y.stride(0))
    # Reuse the compiled kernel directly (half the launch cost of the JIT path) under the same
    # specialization: Triton keys on value 1 and 16-divisibility of ints and pointers.
    key = (
        x.device.index,
        M,
        split,
        x2.dtype,
        x2.data_ptr() % 16 == 0,
        y.data_ptr() % 16 == 0,
        x2.stride(0) % 16 == 0,
        y.stride(0) % 16 == 0,
    )
    launchers = qs._launchers
    if launchers is None:
        launchers = qs._launchers = {}
    with _on_device(x.device):
        compiled = launchers.get(key)
        if compiled is None:
            compiled = _gemv_kernel.warmup(
                *args, grid = grid, BLOCK_N = BLOCK_N, BLOCK_KP = BLOCK_KP, SPLIT_K = split,
                num_warps = 4, **meta,
            )  # fmt: skip
            launchers[key] = compiled
        compiled[grid](*args)
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    if out is not None and y.data_ptr() != out.data_ptr():
        out.view(M, N).copy_(y)
        y = out
    return y.view(*shape[:-1], N)


def int4_matmul_t(dy, packed, qs):
    """``dy @ W`` (the dX of ``x @ W.T``) for packed ``W``; ``dy`` is ``[..., N]``."""
    W = int4_dequantize(packed, qs, dy.dtype)
    return torch.matmul(dy, W)

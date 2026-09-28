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
import inspect
import os

import torch
import triton
import triton.language as tl

__all__ = [
    "Int4QuantState",
    "int4_dequantize",
    "int4_dequantize_weight",
    "int4_matmul",
    "int4_matmul_t",
    "int4_repack_",
    "int4_unpack",
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
        "layout",
        "_fast",
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
        self.layout = None
        self._fast = None


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


# Triton < 3.7 CompiledKernel launchers also want constexprs (TypeError): fall back to the JIT launch.
_FAST_LAUNCH = None


def _launch(kernel, launchers, key, grid, args, **constexprs):
    global _FAST_LAUNCH
    if _FAST_LAUNCH is not False:
        compiled = launchers.get(key)
        if compiled is None:
            compiled = launchers[key] = kernel.warmup(*args, grid = grid, **constexprs)
        try:
            compiled[grid](*args)
            _FAST_LAUNCH = True
            return
        except TypeError:
            if _FAST_LAUNCH:
                raise
            _FAST_LAUNCH = False
    kernel[grid](*args, **constexprs)


def int4_dequantize(
    packed,
    qs,
    dtype = None,
    out = None,
):
    dtype = dtype or qs.dtype or torch.bfloat16
    if qs.layout is not None:
        return _dequant_repacked(packed, qs, dtype, out)
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
        _launch(
            _dequant_kernel,
            launchers,
            key,
            grid,
            args,
            BLOCK_N = BLOCK_N,
            BLOCK_KP = BLOCK_KP,
            num_warps = 4,
            **meta,
        )
    return out


def int4_dequantize_weight(
    W,
    qs,
    dtype = None,
):
    """``fast_dequantize`` contract: ``W`` is the packed weight or its ``.t()``; returns the matching dense view."""
    if W.stride(-1) != 1 and W.dim() == 2 and W.shape[1] == qs.shape[0]:
        return int4_dequantize(W.t(), qs, dtype).t()
    return int4_dequantize(W, qs, dtype)


@functools.lru_cache(maxsize = None)
def _sm_count(index):
    return torch.cuda.get_device_properties(index).multi_processor_count


# The fused GEMV decodes the weight once per row: only one row beats decode + cuBLAS.
GEMV_MAX_ROWS = 1


# Weights are repacked once at load (in place, no second copy) into vLLM Marlin's layout when vLLM is installed, else
# tinygemm's (bf16 only; UNSLOTH_INT4_LAYOUT=tinygemm forces it). Rows where the fused kernel beats dequantize + cuBLAS,
# by CC major (training always dequantizes, so its numerics never depend on the layout):
MARLIN_MAX_ROWS = {8: 512, 10: 32, 11: 32, 12: 1024}
TINYGEMM_MAX_ROWS = {8: 16, 10: 4, 11: 4, 12: 16}
# tile rows, tile columns, tiles advance along N first, destination bit of each source nibble-index bit
_LAYOUTS = {
    "tinygemm": (8, 128, False, (3, 4, 0, 5, 6, 1, 2, 7, 8, 9)),
    "marlin": (64, 16, True, (3, 7, 0, 8, 9, 1, 2, 4, 5, 6)),
}
_LAYOUT_OK = {}
_FAST_CHECKED = {}


@functools.lru_cache(maxsize = None)
def _fast_rows(layout, index):
    table = MARLIN_MAX_ROWS if layout == "marlin" else TINYGEMM_MAX_ROWS
    return table.get(torch.cuda.get_device_capability(index)[0], 32 if layout == "marlin" else 4)


_MARLIN_API = None


def _marlin_api():
    global _MARLIN_API
    if _MARLIN_API is None:
        _MARLIN_API = False
        if os.environ.get("UNSLOTH_INT4_MARLIN", "1") != "0" and torch.version.hip is None:
            try:
                from vllm import _custom_ops as ops
                from vllm.scalar_type import scalar_types

                new = hasattr(torch.ops._C, "marlin_gemm")
                if not new and not hasattr(torch.ops._C, "gptq_marlin_gemm"):
                    return _MARLIN_API
                repack_perm = "perm" in inspect.signature(ops.gptq_marlin_repack).parameters
                types = {False: scalar_types.uint4b8, True: scalar_types.uint4}
                _MARLIN_API = (ops, new, repack_perm, types, {})
            except Exception:
                pass
    return _MARLIN_API


def _marlin_scale_perm():
    return [i + 8 * j for i in range(8) for j in range(8)]


def _repack_words(layout, packed, N, K):
    """The checkpoint's ``[N, K / 8]`` words in ``layout``, viewed as ``[N, K / 8]`` again."""
    if layout == "marlin":
        ops, _, repack_perm, _, _ = _marlin_api()
        gptq = packed.t().contiguous()  # GPTQ packs along K too, transposed
        if repack_perm:
            words = ops.gptq_marlin_repack(
                gptq, torch.empty(0, dtype = torch.int32, device = packed.device), K, N, 4
            )
        else:
            words = ops.gptq_marlin_repack(gptq, K, N, 4)
    else:
        # Nibble i of word w is column 8w + i; tinygemm wants bytes (q[2j] << 4) | q[2j + 1].
        b = packed.unsqueeze(-1) >> torch.arange(0, 32, 8, device = packed.device, dtype = torch.int32)
        pairs = (((b & 15) << 4) | ((b >> 4) & 15)).to(torch.uint8).reshape(N, K // 2)
        words = torch._convert_weight_to_int4pack(pairs, 8)
    return words.reshape(N, K // 8)


def _layout_verified(layout, device):
    """Probe the repack with base-16 digits of each position; it must be the tile bit permutation hard-coded above."""
    key = (layout, device)
    ok = _LAYOUT_OK.get(key)
    if ok is None:
        ok = False
        try:
            TN, TK, nfast, bits = _LAYOUTS[layout]
            Np, Kp = (128, 256) if layout == "marlin" else (16, 256)
            pos = torch.arange(Np * Kp, device = device, dtype = torch.int32).view(Np, Kp)
            idx = torch.zeros(Np * Kp, dtype = torch.int32, device = device)
            nib = torch.arange(0, 32, 4, device = device, dtype = torch.int32)
            for d in range(4):
                q = ((pos >> (4 * d)) & 15).view(Np, -1, 8)
                probe = torch.zeros((Np, Kp // 8), dtype = torch.int32, device = device)
                for i in range(8):
                    probe |= q[:, :, i] << (4 * i)
                words = _repack_words(layout, probe, Np, Kp).reshape(-1)
                idx |= ((words.unsqueeze(-1) >> nib) & 15).reshape(-1) << (4 * d)
            e = torch.arange(Np * Kp, device = device, dtype = torch.int32)
            tile, local = e // 1024, e % 1024
            NT_N, NT_K = Np // TN, Kp // TK
            tn, tk = (tile % NT_N, tile // NT_N) if nfast else (tile // NT_K, tile % NT_K)
            dest = torch.zeros_like(local)
            for b, d in enumerate(bits):
                dest |= ((local >> b) & 1) << d
            want = (tn * TN + dest // TK) * Kp + tk * TK + dest % TK
            ok = bool(torch.equal(idx, want))
        except Exception:
            ok = False
        _LAYOUT_OK[key] = ok
    return ok


def _pick_layout(packed, qs):
    if (
        os.environ.get("UNSLOTH_INT4_REPACK", "1") == "0"
        or not packed.is_cuda
        or torch.version.hip is not None
        or qs.bits != 4
        or qs.g_idx is not None
        or packed.dtype != torch.int32
        or torch.cuda.get_device_capability(packed.device) < (8, 0)
    ):
        return None
    N, K = qs.shape
    group, zp, dtype = qs.group_size, qs.zero_point, qs.dtype
    if (
        packed.shape != (N, K // 8)
        or group <= 0
        or K % max(group, 128)
        or qs.scale.dtype != dtype
        or qs.scale.shape != (N, K // max(group, 1))
        or (zp is not None and zp.shape != ((N + 7) // 8, K // max(group, 1)))
    ):
        return None
    tinygemm = (
        dtype == torch.bfloat16
        and group in (32, 64, 128, 256)
        and N % 8 == 0
        and hasattr(torch, "_weight_int4pack_mm")
        and os.environ.get("UNSLOTH_INT4_TINYGEMM", "1") != "0"
    )
    marlin = (
        dtype in (torch.float16, torch.bfloat16)
        and group in (32, 64, 128)
        and N % 64 == 0
        and bool(_marlin_api())
    )
    if marlin and (not tinygemm or os.environ.get("UNSLOTH_INT4_LAYOUT") != "tinygemm"):
        return "marlin"
    return "tinygemm" if tinygemm else None


def int4_repack_(packed, qs):
    """Repack ``packed`` in place into a fused-kernel layout; returns it, or None (left untouched)."""
    layout = _pick_layout(packed, qs)
    if layout is None or not _layout_verified(layout, packed.device):
        return None
    N, K = qs.shape
    with torch.no_grad(), _on_device(packed.device):
        words = _repack_words(layout, packed, N, K)
        before = int4_dequantize(packed, qs, qs.dtype)
        qs.layout = layout
        same = torch.equal(_dequant_repacked(words, qs, qs.dtype), before)
        del before
        if not same:
            qs.layout = None
            return None
        packed.copy_(words)
    qs._launch = qs._launchers = qs._fast = None
    return layout


@triton.jit
def _dequant_repacked_kernel(
    P,
    S,
    Z,
    Out,
    NT_N,
    NT_K,
    stride_sn,
    stride_zn,
    stride_on,
    MARLIN: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    R: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_Z: tl.constexpr,
    RAW: tl.constexpr,
):
    # One program: tile row tn, R tiles along K, written as [TN, R * TK] contiguous rows.
    pid = tl.program_id(0)
    tn = pid // (NT_K // R)
    tk0 = (pid % (NT_K // R)) * R
    r = tl.arange(0, R)
    if MARLIN:
        tile = (tk0 + r) * NT_N + tn
    else:
        tile = tn * NT_K + tk0 + r
    words = tl.load(P + tile[:, None] * 128 + tl.arange(0, 128)[None, :])
    q = (words[:, :, None] >> (tl.arange(0, 8) * 4)[None, None, :]) & 15
    # The tile's nibble-index bits, grouped by where they land (_LAYOUTS), permuted into row-major [TN, R * TK].
    if MARLIN:  # E(9:7) D(6:5) C(4:3) B(2) A2(1) A1(0) -> rows C A2 E, columns A1 D B
        q = tl.permute(tl.reshape(q, (R, 8, 4, 4, 2, 2, 2)), (3, 5, 1, 0, 6, 2, 4))
    else:  # E(9:7) D(6:5) C(4:3) B(2) A(1:0) -> rows E, columns C A D B
        q = tl.permute(tl.reshape(q, (R, 8, 4, 4, 2, 4)), (1, 0, 3, 5, 2, 4))
    q = tl.reshape(q, (TN, R * TK))
    n = tn * TN + tl.arange(0, TN)
    offs = n[:, None] * stride_on + tk0 * TK + tl.arange(0, R * TK)[None, :]
    if RAW:
        tl.store(Out + offs, q.to(tl.int8))
    else:
        NG: tl.constexpr = (R * TK) // GROUP
        g = (tk0 * TK) // GROUP + tl.arange(0, NG)
        s = tl.load(S + n[:, None] * stride_sn + g[None, :]).to(tl.float32)
        if HAS_Z:
            zw = tl.load(Z + (n // 8)[:, None] * stride_zn + g[None, :])
            z = _as_float((zw >> ((n % 8) * 4)[:, None]) & 15)
        else:
            z = tl.full((TN, NG), 8.0, tl.float32)
        s = tl.reshape(tl.broadcast_to(s[:, :, None], (TN, NG, GROUP)), (TN, R * TK))
        z = tl.reshape(tl.broadcast_to(z[:, :, None], (TN, NG, GROUP)), (TN, R * TK))
        # Same fp32 math and double rounding as _dequant_kernel: bit-identical weights.
        w = (_as_float(q) - z) * s
        tl.store(Out + offs, w.to(S.dtype.element_ty).to(Out.dtype.element_ty))


def _dequant_repacked(
    words,
    qs,
    dtype,
    out = None,
    raw = False,
):
    N, K = qs.shape
    TN, TK, _, _ = _LAYOUTS[qs.layout]
    NT_N, NT_K = N // TN, K // TK
    group = qs.group_size
    R = max(8, group // TK)
    while NT_K % R:
        R //= 2
    if out is None:
        out = torch.empty((N, K), dtype = torch.int8 if raw else dtype, device = words.device)
    zp = qs.zero_point
    with _on_device(words.device):
        _dequant_repacked_kernel[(NT_N * (NT_K // R),)](
            words, qs.scale, qs.scale if zp is None else zp, out, NT_N, NT_K, qs.scale.stride(0),
            0 if zp is None else zp.stride(0), out.stride(0), MARLIN = qs.layout == "marlin", TN = TN, TK = TK,
            R = R, GROUP = group, HAS_Z = zp is not None, RAW = raw, num_warps = 4,
        )  # fmt: skip
    return out


def int4_unpack(packed, qs):
    """The checkpoint-layout ``[N, K / 8]`` words of a repacked weight (what gets saved)."""
    if qs.layout is None:
        return packed
    N, K = qs.shape
    q = _dequant_repacked(packed, qs, None, raw = True).to(torch.int32).view(N, K // 8, 8)
    out = torch.zeros((N, K // 8), dtype = torch.int32, device = packed.device)
    for i in range(8):
        out |= q[:, :, i] << (4 * i)
    return out


def _fast_args(packed, qs):
    """Cached fused-kernel call for a repacked weight, self-checked once per (layout, device, zero points, dtype)."""
    fast = qs._fast
    if fast is not None and fast[0] == packed.data_ptr():
        return fast[1]
    N, K = qs.shape
    group, zp, scale = qs.group_size, qs.zero_point, qs.scale
    if qs.layout == "tinygemm":
        s = scale.reshape(N, -1).t()
        if zp is None:
            zero = torch.zeros_like(s)
        else:
            # (q - z) * s == (q - 8) * s + (8 - z) * s, tinygemm's float zero.
            z = (
                zp.t().unsqueeze(-1) >> torch.arange(0, 32, 4, device = zp.device, dtype = torch.int32)
            ) & 15
            zero = ((8 - z.reshape(zp.shape[1], -1)[:, :N]).float() * s.float()).to(s.dtype)
        call = (
            torch.ops.aten._weight_int4pack_mm.default,
            packed.view(N // 8, K // 128, 32, 4),
            group,
            torch.stack((s, zero), -1).contiguous(),
        )
    else:
        _, new, _, types, workspaces = _marlin_api()
        perm = _marlin_scale_perm()
        ms = scale.t().contiguous().reshape(-1, 64)[:, perm].reshape(-1, N).contiguous()
        empty = torch.empty(0, dtype = torch.int32, device = packed.device)
        mz = empty
        if zp is not None:
            G = zp.shape[1]
            z = (
                (
                    zp.t().unsqueeze(-1)
                    >> torch.arange(0, 32, 4, device = zp.device, dtype = torch.int32)
                )
                & 15
            ).reshape(G, N)
            z = (
                z.reshape(-1, 64)[:, perm]
                .reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]]
                .reshape(G, N // 8, 8)
            )
            mz = torch.zeros((G, N // 8), dtype = torch.int32, device = zp.device)
            for i in range(8):
                mz |= z[:, :, i] << (4 * i)
        ws = workspaces.get(packed.device)
        if ws is None:
            ws = workspaces[packed.device] = torch.zeros(
                _sm_count(packed.device.index or 0), dtype = torch.int32, device = packed.device
            )
        mq, sid = packed.view(K // 16, 2 * N), types[zp is not None].id
        # vLLM's Python wrapper costs ~2 us per call (decode is launch bound): call the op overload.
        if new:
            call = (
                torch.ops._C.marlin_gemm.default,
                (None, mq, None, ms, None, None, mz, ws, sid),
                (N, K, False, True, False),
            )
        else:
            call = (
                torch.ops._C.gptq_marlin_gemm.default,
                (None, mq, None, ms, None, mz, empty, empty, ws, sid),
                (N, K, True, False, True, False),
            )
    key = (qs.layout, packed.device, zp is not None, qs.dtype)
    if key not in _FAST_CHECKED:
        g = torch.Generator(device = packed.device).manual_seed(0)
        probe = torch.randn(4, K, device = packed.device, dtype = qs.dtype, generator = g)
        ref = probe.float() @ int4_dequantize(packed, qs, qs.dtype).float().t()
        got = _fast_call(call, probe).float()
        _FAST_CHECKED[key] = bool(((got - ref).abs().max() <= 1e-2 * ref.abs().max() + 1e-3).item())
    call = call if _FAST_CHECKED[key] else None
    qs._fast = (packed.data_ptr(), call)
    return call


def _fast_call(call, x2):
    if len(call) == 4:
        return call[0](x2, call[1], call[2], call[3])
    op, pre, post = call
    return op(x2, *pre, x2.shape[0], *post)


def int4_matmul(
    x,
    packed,
    qs,
    out = None,
    fast = True,
):
    """``x @ W.T`` for packed ``W``; ``x`` is ``[..., K]``. ``fast = False`` (training) keeps the exact dequantize + matmul."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    M = x2.shape[0]
    N = qs.shape[0]
    if (
        M == 0
    ):  # Nothing to launch; the kernels reject a zero grid and the split below divides by M.
        return x.new_empty((*shape[:-1], N)) if out is None else out
    if qs.layout is not None:
        call = None
        if (
            fast
            and not torch.is_grad_enabled()
            and M <= _fast_rows(qs.layout, x2.device.index or 0)
        ):
            call = _fast_args(packed, qs)
        if call is not None:
            y = _fast_call(call, x2 if x2.is_contiguous() else x2.contiguous())
            if out is not None:
                out.view(M, N).copy_(y)
                y = out
            return y.view(*shape[:-1], N)
        W = int4_dequantize(packed, qs, x.dtype)
        y = torch.matmul(x2, W.t(), out = None if out is None else out.view(M, N))
        return y.view(*shape[:-1], N)
    if M > GEMV_MAX_ROWS or not fast:
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
    # Reuse the compiled kernel (half the JIT launch cost); Triton specializes on ==1 and 16-divisibility.
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
        _launch(
            _gemv_kernel, launchers, key, grid, args, BLOCK_N = BLOCK_N, BLOCK_KP = BLOCK_KP, SPLIT_K = split,
            num_warps = 4, **meta,
        )  # fmt: skip
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

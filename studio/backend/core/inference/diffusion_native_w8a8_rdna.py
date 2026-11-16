# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton W8A8 linear for the native int8 layers on ROCm RDNA3 / RDNA4 (gfx11xx / gfx12xx): one kernel rotates
(ConvRot) and row-quantises the activation, one int8 GEMM applies both row scales and the bias in its epilogue.
Opaque custom ops, so a compiled model keeps them whole and calls sharing an input (Q / K / V) share one quant.
Each card runs a parity probe first; a failure there keeps the stock path. UNSLOTH_NATIVE_INT8_RDNA_TRITON=0 disables."""

from __future__ import annotations

import os
import re
import threading
import types
from functools import lru_cache
from typing import Any, Optional

RDNA_TRITON_ENV = "UNSLOTH_NATIVE_INT8_RDNA_TRITON"
_NS = "unsloth_studio"
_ROTQ_OP = "native_rdna_rotq_rows"
_MM_OP = "native_rdna_w8a8_mm"
_ARCH = re.compile(r"^gfx1[12][0-9a-f]{2}$")
_MIN_TRITON = (3, 2)
_ROTQ_MAX_GROUPS = 128
_QUANT_CHUNK = 256
# rotq: (group-rows per program BLOCK_M, K step, warps, stages); BLOCK_M grows to hold one whole activation row.
# mm: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warps, stages). gfx1151 sweep at 4096 rows: K 12288 fastest at
# 128x128x128 / 8 warps (8.41 ms vs 12.96 stock); K <= 4096 at 64x128x128 / 4 warps (1.92 vs 2.04 ms for the
# 128-row tile, 2.45 stock; N 3072 and 12288 alike). Under 1024 rows the 64-wide tile.
_CFG = {
    "rotq": (64, 32, 4, 1),
    "mm_large": (128, 128, 128, 8, 8, 2),
    "mm_large_short_k": (64, 128, 128, 8, 4, 2),
    "mm_small": (64, 64, 64, 8, 4, 2),
}
_SHORT_K = 4096
RDNA_TILES_ENV = "UNSLOTH_NATIVE_INT8_RDNA_TILES"


def _tile_overrides(raw: Optional[str]) -> dict:
    """``{"rotq": [bm, bk, warps, stages], "mm_large": [...6], "mm_large_short_k": [...6], "mm_small": [...6]}`` from
    JSON; bad input -> {}."""
    import json
    try:
        got = json.loads(raw) if raw else {}
        out = {}
        for key, size in (("rotq", 4), ("mm_large", 6), ("mm_large_short_k", 6), ("mm_small", 6)):
            val = got.get(key)
            if val is not None:
                val = tuple(int(v) for v in val)
                if len(val) != size or any(v <= 0 for v in val):
                    return {}
                out[key] = val
        return out
    except Exception:  # noqa: BLE001
        return {}


_CFG.update(_tile_overrides(os.environ.get(RDNA_TILES_ENV)))
# The per-row reduction tile is [BLOCK_M, ROWS_P2]; AMD WMMA layouts need both dims >= 16.
_MIN_ROWS_P2 = 16

# "devices": device key -> None (runs) or the reason it does not; "on": card index -> True, read by compiled graphs.
# Every key exists up front: a compiled graph guards on this dict's keys, so later writes must not add any.
_STATE: dict = {"devices": {}, "on": {}, "broken": None, "rotq": None, "mm": None}
_LOCK = threading.Lock()


def rdna_triton_disabled() -> bool:
    return os.environ.get(RDNA_TRITON_ENV, "").strip().lower() in ("0", "false", "no", "off")


def _triton_version_ok() -> bool:
    try:
        import triton
        match = re.match(r"(\d+)\.(\d+)", str(triton.__version__))
        return bool(match) and (int(match.group(1)), int(match.group(2))) >= _MIN_TRITON
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize = 1)
def _kernels() -> Optional[types.SimpleNamespace]:
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001 - no Triton means the stock path
        return None

    @triton.jit(do_not_specialize = ["M"])
    def rotq_rows(
        x_ptr,
        h_ptr,
        q_ptr,
        s_ptr,
        M,
        NG,
        stride_xm,
        G: tl.constexpr,  # rotation group width (also the quant chunk without a rotation)
        BLOCK_M: tl.constexpr,  # group rows per program
        BLOCK_K: tl.constexpr,  # K step inside a group
        ROWS: tl.constexpr,  # activation rows per program
        ROWS_P2: tl.constexpr,  # ROWS next pow2
        ROT: tl.constexpr,  # apply the Hadamard
    ):
        # One program = ROWS whole rows laid out as (row, group) lanes, so each row's amax closes in the tile.
        pid = tl.program_id(0)
        t = tl.arange(0, BLOCK_M)
        lr = t // NG
        gi = t % NG
        row = pid.to(tl.int64) * ROWS + lr
        live = (lr < ROWS) & (row < M)
        row_c = tl.where(live, row, 0)
        offs_n = tl.arange(0, G)
        if ROT:
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, G), dtype = tl.float32)
            a_ptrs = x_ptr + row_c[:, None] * stride_xm + gi[:, None] * G + offs_k[None, :]
            h_ptrs = h_ptr + offs_k[:, None] * G + offs_n[None, :]
            for _k in range(0, G // BLOCK_K):
                a = tl.load(a_ptrs, mask = live[:, None], other = 0.0)
                acc = tl.dot(a, tl.load(h_ptrs), acc, out_dtype = tl.float32)
                a_ptrs += BLOCK_K
                h_ptrs += BLOCK_K * G
            # The stock rotation is a bf16 / fp16 matmul: its output rounds to the activation dtype.
            z = acc.to(x_ptr.dtype.element_ty).to(tl.float32)
        else:
            z = tl.load(
                x_ptr + row_c[:, None] * stride_xm + gi[:, None] * G + offs_n[None, :],
                mask = live[:, None],
                other = 0.0,
            ).to(tl.float32)
        j = tl.arange(0, ROWS_P2)
        sel = lr[:, None] == j[None, :]
        amax = tl.max(tl.where(sel, tl.max(tl.abs(z), axis = 1)[:, None], 0.0), axis = 0)
        # As _w8a8_rows: s = max(amax, 1e-12) * fp32(1 / 127), reciprocal in fp64 rounded to fp32.
        s = tl.maximum(amax, 1e-12) * 0.007874015748031496
        inv = (1.0 / s.to(tl.float64)).to(tl.float32)
        inv_t = tl.max(tl.where(sel, inv[None, :], 0.0), axis = 1)
        # Round half to even for |p| < 2**22 without libdevice (no shared spelling on HIP); fp fusion is off.
        p = (z * inv_t[:, None] + 12582912.0) - 12582912.0
        p = tl.minimum(tl.maximum(p, -127.0), 127.0)
        tl.store(
            q_ptr + row_c[:, None] * (NG * G) + gi[:, None] * G + offs_n[None, :],
            p.to(tl.int8),
            mask = live[:, None],
        )
        srow = pid.to(tl.int64) * ROWS + j
        tl.store(s_ptr + srow, s, mask = (j < ROWS) & (srow < M))

    @triton.jit(do_not_specialize = ["M"])
    def w8a8_mm(
        a_ptr,
        w_ptr,
        xs_ptr,
        ws_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        HAS_BIAS: tl.constexpr,
        EVEN_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rm[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + rn[None, :] * K + offs_k[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.int32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(w_ptrs)
            else:
                kmask = offs_k < K - k * BLOCK_K
                a = tl.load(a_ptrs, mask = kmask[None, :], other = 0)
                b = tl.load(w_ptrs, mask = kmask[:, None], other = 0)
            acc = tl.dot(a, b, acc, out_dtype = tl.int32)
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        # As _w8a8_rows: (int32 * xs) * ws (+ bias), all fp32, one rounding to the output dtype.
        xs = tl.load(xs_ptr + rm)
        ws = tl.load(ws_ptr + rn)
        y = acc.to(tl.float32) * xs[:, None]
        y = y * ws[None, :]
        if HAS_BIAS:
            y = y + tl.load(b_ptr + rn).to(tl.float32)[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, y.to(c_ptr.dtype.element_ty), mask = mask)

    return types.SimpleNamespace(rotq_rows = rotq_rows, w8a8_mm = w8a8_mm, triton = triton)


def _rotq_tile(k: int, group: int) -> Optional[tuple]:
    """(group width, groups per row, BLOCK_M, rows per program), or None if K does not fit."""
    g = int(group) if group else _QUANT_CHUNK
    # Only the group the per-card probe checks: a group below BLOCK_K skips the rotation loop and quantises zeros.
    if k <= 0 or g != _QUANT_CHUNK or k % g:
        return None
    ng = k // g
    if ng > _ROTQ_MAX_GROUPS:
        return None
    bm = _CFG["rotq"][0]
    while bm < ng:
        bm *= 2
    return g, ng, bm, bm // ng


def shape_supported(in_features: int, out_features: int, group: int) -> bool:
    return (
        _rotq_tile(int(in_features), int(group)) is not None
        and int(in_features) % 16 == 0
        and int(out_features) % 16 == 0
    )


def _rotq_launch(x2: Any, group: int) -> tuple:
    import torch

    from .diffusion_convrot import build_convrot_hadamard

    m, k = x2.shape
    q = torch.empty((m, k), device = x2.device, dtype = torch.int8)
    s = torch.empty((m, 1), device = x2.device, dtype = torch.float32)
    if m == 0:
        return q, s
    tile = _rotq_tile(k, group)
    if tile is None or x2.stride(-1) != 1:
        return reference_rotq(x2, group)
    if _STATE.get("broken"):
        return reference_rotq(x2, group)
    g, ng, bm, rows = tile
    _, bk, warps, stages = _CFG["rotq"]
    kern = _kernels()
    triton = kern.triton
    h = build_convrot_hadamard(g, device = x2.device, dtype = x2.dtype) if group else x2
    try:
        _launch_rotq(kern, x2, h, q, s, m, ng, g, bm, bk, rows, group, warps, stages)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - a kernel that cannot build here: stock maths from now on
        _mark_broken("rotate+quant", exc)
        return reference_rotq(x2, group)
    return q, s


def _launch_rotq(kern, x2, h, q, s, m, ng, g, bm, bk, rows, group, warps, stages) -> None:
    import torch
    triton = kern.triton
    with torch.cuda.device(x2.device):
        kern.rotq_rows[(triton.cdiv(m, rows),)](
            x2,
            h,
            q,
            s,
            m,
            ng,
            x2.stride(0),
            G = g,
            BLOCK_M = bm,
            BLOCK_K = bk,
            ROWS = rows,
            ROWS_P2 = max(triton.next_power_of_2(rows), _MIN_ROWS_P2),
            ROT = bool(group),
            num_warps = warps,
            num_stages = stages,
            enable_fp_fusion = False,
        )


def _mark_broken(what: str, exc: BaseException) -> None:
    if _STATE.get("broken"):
        return
    _STATE["broken"] = f"{what}: {type(exc).__name__}: {str(exc)[:200]}"
    import logging

    logging.getLogger(__name__).warning(
        "diffusion.native_quant: RDNA Triton %s failed (%s); stock int8 maths from now on (set %s=0 to skip it)",
        what,
        exc,
        RDNA_TRITON_ENV,
    )


def _mm_launch(xq: Any, xs: Any, wq: Any, ws: Any, bias: Any, out_dtype: Any) -> Any:
    import torch

    xq, xs, wq, ws = xq.contiguous(), xs.contiguous(), wq.contiguous(), ws.contiguous()
    bias = bias.contiguous() if bias is not None else None
    m, k = xq.shape
    n = wq.shape[0]
    out = torch.empty((m, n), device = xq.device, dtype = out_dtype)
    if m == 0:
        return out
    if _STATE.get("broken"):
        return reference_mm_stock(xq, xs, wq, ws, bias, out_dtype)
    try:
        _launch_mm(xq, xs, wq, ws, bias, out, m, n, k)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001
        _mark_broken("int8 GEMM", exc)
        return reference_mm_stock(xq, xs, wq, ws, bias, out_dtype)
    return out


def _mm_cfg(m: int, k: int) -> tuple:
    if m < 1024:
        return _CFG["mm_small"]
    return _CFG["mm_large_short_k"] if k <= _SHORT_K else _CFG["mm_large"]


def _launch_mm(xq, xs, wq, ws, bias, out, m, n, k) -> None:
    import torch

    kern = _kernels()
    triton = kern.triton
    bm, bn, bk, gm, warps, stages = _mm_cfg(m, k)
    with torch.cuda.device(xq.device):
        kern.w8a8_mm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            xq,
            wq,
            xs,
            ws,
            bias if bias is not None else ws,
            out,
            m,
            n,
            k,
            HAS_BIAS = bias is not None,
            EVEN_K = k % bk == 0,
            BLOCK_M = bm,
            BLOCK_N = bn,
            BLOCK_K = bk,
            GROUP_M = gm,
            num_warps = warps,
            num_stages = stages,
            enable_fp_fusion = False,
        )


def reference_rotq(x2: Any, group: int) -> tuple:
    """Eager ``_w8a8_rows`` prologue: (int8 codes [M, K], fp32 row scales [M, 1])."""
    import torch

    from .diffusion_convrot import build_convrot_hadamard

    if group:
        h = build_convrot_hadamard(int(group), device = x2.device, dtype = x2.dtype)
        x2 = (x2.reshape(x2.shape[0], -1, int(group)) @ h).reshape(x2.shape)
    s = torch.linalg.vector_norm(x2, ord = float("inf"), dim = 1, keepdim = True, dtype = torch.float32)
    s = s.clamp(min = 1e-12) * (1.0 / 127.0)
    inv = (1.0 / s.double()).float()
    q = torch.mul(x2, inv).round().clamp(-127, 127).to(torch.int8)
    return q, s


def reference_mm_stock(xq: Any, xs: Any, wq: Any, ws: Any, bias: Any, out_dtype: Any) -> Any:
    """The stock ``_w8a8_rows`` GEMM + epilogue (torch._int_mm needs > 16 rows; fewer go through fp32)."""
    import torch

    if xq.shape[0] > 16 and xq.is_cuda:
        acc = torch._int_mm(xq, wq.t())
    else:
        acc = (xq.float() @ wq.float().t()).round().to(torch.int32)
    out = torch.mul(acc, xs.reshape(-1, 1))
    w = ws.reshape(-1)
    out = torch.addcmul(bias.float(), out, w) if bias is not None else out * w
    return out.to(out_dtype)


def reference_mm(xq: Any, xs: Any, wq: Any, ws: Any, bias: Any, out_dtype: Any) -> Any:
    import torch

    acc = torch.matmul(xq.to(torch.float64), wq.t().to(torch.float64)).to(torch.float32)
    out = acc * xs.reshape(-1, 1) * ws.reshape(1, -1)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


@lru_cache(maxsize = 1)
def _ops() -> Optional[types.SimpleNamespace]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _NS, None)
    if ns is not None and hasattr(ns, _ROTQ_OP) and hasattr(ns, _MM_OP):
        return types.SimpleNamespace(rotq = getattr(ns, _ROTQ_OP), mm = getattr(ns, _MM_OP))
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_NS}::{_ROTQ_OP}",
            mutates_args = (),
            schema = "(Tensor x, int group) -> (Tensor, Tensor)",
        )
        def _rotq(x, group):
            return _rotq_launch(x, group)

        @_rotq.register_fake
        def _(x, group):
            return (
                x.new_empty(x.shape, dtype = torch.int8),
                x.new_empty((x.shape[0], 1), dtype = torch.float32),
            )

        @custom_op(
            f"{_NS}::{_MM_OP}",
            mutates_args = (),
            schema = "(Tensor xq, Tensor xs, Tensor wq, Tensor ws, Tensor? bias, ScalarType out_dtype) -> Tensor",
        )
        def _mm(xq, xs, wq, ws, bias, out_dtype):
            return _mm_launch(xq, xs, wq, ws, bias, out_dtype)

        @_mm.register_fake
        def _(xq, xs, wq, ws, bias, out_dtype):
            return xq.new_empty((xq.shape[0], wq.shape[0]), dtype = out_dtype)
    except Exception:  # noqa: BLE001 - a registration failure keeps the stock path
        return None
    ns = getattr(torch.ops, _NS)
    return types.SimpleNamespace(rotq = getattr(ns, _ROTQ_OP), mm = getattr(ns, _MM_OP))


def _key(device: Any) -> Any:
    import torch

    if device is None:
        return torch.device("cuda", int(torch.cuda.current_device()))
    device = torch.device(device)
    if device.index is None:
        return torch.device(device.type, int(torch.cuda.current_device()))
    return device


def _probe(device: Any) -> Optional[str]:
    """None when both kernels match the eager reference on ``device``, else why not."""
    import torch

    gen = torch.Generator(device = "cpu").manual_seed(0)
    ops = _ops()
    for dtype in (torch.bfloat16, torch.float16):
        # 1040 rows: the large GEMM tiles (short K and long K) with a partial last block; 96 / 64: the small tile.
        # K covers several whole-row tilings (3 / 12 / 48 groups of 256) and the unrotated quant.
        for group, k, rows in (
            (256, 768, 1040),
            (256, 3072, 1040),
            (256, 12288, 1040),
            (256, 12288, 64),
            (0, 3072, 96),
        ):
            x = (torch.randn(rows, k, generator = gen) * 3).to(dtype).to(device)
            x[3] = 0  # an all-zero row keeps the 1e-12 floor
            wq = torch.randint(-127, 128, (48, k), generator = gen, dtype = torch.int8).to(device)
            _STATE["broken"] = None
            ws = (torch.rand(48, generator = gen) * 1e-2 + 1e-4).to(device)
            bias = torch.randn(48, generator = gen).to(dtype).to(device)
            q, s = ops.rotq(x, group)
            rq, rs = reference_rotq(x, group)
            diff = (q.int() - rq.int()).abs()
            # The rotation's fp32 accumulation order may move a bf16 tie: at most one code, rarely.
            if int(diff.max()) > 1 or float((diff > 0).float().mean()) > 1e-3:
                return f"rotate+quant codes differ ({dtype}, group {group})"
            if not torch.allclose(s, rs, rtol = 1e-2, atol = 0):
                return f"rotate+quant scales differ ({dtype}, group {group})"
            out = ops.mm(rq, rs, wq, ws, bias, dtype)
            ref = reference_mm(rq, rs, wq, ws, bias, dtype)
            if not torch.allclose(out.float(), ref.float(), rtol = 2e-2, atol = 1e-2):
                return f"int8 GEMM differs ({dtype})"
            out = ops.mm(rq, rs, wq, ws, None, dtype)
            ref = reference_mm(rq, rs, wq, ws, None, dtype)
            if not torch.allclose(out.float(), ref.float(), rtol = 2e-2, atol = 1e-2):
                return f"int8 GEMM differs ({dtype}, no bias)"
            if _STATE.get("broken"):  # a launch failed and fell back: the kernels do not build here
                why, _STATE["broken"] = _STATE["broken"], None
                return why
    torch.cuda.synchronize(device)
    return None


def prepare(device: Any = None) -> bool:
    if rdna_triton_disabled():
        return False
    try:
        import torch

        from .rocm_bf16 import _device_gfx_arch, is_rocm_torch

        if not is_rocm_torch(torch) or not torch.cuda.is_available():
            return False
        key = _key(device)
        if key.type != "cuda":
            return False
    except Exception:  # noqa: BLE001
        return False
    devices = _STATE["devices"]
    with _LOCK:
        if key in devices:
            return devices[key] is None
        why: Optional[str] = None
        arch = _device_gfx_arch(torch, key.index)
        if not _ARCH.match(arch):
            why = f"ROCm {arch or 'unknown arch'}"
        elif not _triton_version_ok() or _kernels() is None:
            why = "no Triton >= 3.2"
        elif _ops() is None:
            why = "no torch.library.custom_op"
        else:
            try:
                why = _probe(key)
            except Exception as exc:  # noqa: BLE001 - a kernel that cannot build keeps the stock path
                why = f"{type(exc).__name__}: {exc}"
        devices[key] = why
        if why is None:
            ops = _ops()
            # Plain dict entries: a compiled graph reads the op handles without tracing the lru_cache.
            _STATE["rotq"], _STATE["mm"] = ops.rotq, ops.mm
            _STATE["on"][int(key.index)] = True
    try:
        from loggers import get_logger
        log = get_logger(__name__)
    except Exception:  # noqa: BLE001
        import logging
        log = logging.getLogger(__name__)
    if why is None:
        log.info("diffusion.native_quant: int8 W8A8 on %s runs the RDNA Triton kernels", key)
    else:
        log.info("diffusion.native_quant: int8 W8A8 on %s keeps the stock kernels (%s)", key, why)
    return why is None


def enabled_for(index: Any) -> bool:
    """Traceable lookup of an earlier ``prepare`` for card ``index`` (never probes; a compiled graph reads one dict)."""
    return bool(_STATE["on"].get(index, False))


def state() -> dict:
    return {str(k): why for k, why in _STATE["devices"].items()}


def rotq(x2: Any, group: int) -> tuple:
    return _STATE["rotq"](x2, group)


def w8a8_mm(xq: Any, xs: Any, wq: Any, ws: Any, bias: Any, out_dtype: Any) -> Any:
    return _STATE["mm"](xq, xs, wq, ws, bias, out_dtype)

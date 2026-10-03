# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""int8 x int8 GEMM with torchao's dequant epilogue in the kernel: bf16 out instead of an int32 ``_int_mm`` output.

The stock torchao W8A8 linear writes the int32 accumulator to HBM (4 bytes per output element) and the next compiled
kernel reads it back to apply ``(c * x_scale) * w_scale``. Here one Triton kernel does the GEMM and the epilogue and
writes bf16, so every int8 Linear writes half the bytes and its consumer (norm, RoPE, SwiGLU, act quant) reads half.

NUMERICS: the Linear output is bit-identical to torchao's epilogue, eager and compiled under Studio's
``emulate_precision_casts``: ``bf16(bf16(r(c) * xs) * ws)`` (+ bias as a bf16 add), where ``r`` rounds the int32
accumulator to bf16 when the activation scale is bf16 (int32 * bf16 promotes to bf16, via fp32: two roundings) and to
fp32 when it is fp32.
The activation quant is torchao's own math, so Inductor still fuses it into the producer. A compiled block is NOT
bit-identical end to end: the consumer reductions (LayerNorm / RMSNorm Welford) that used to fuse the epilogue now
read bf16 and Inductor re-tiles them, a rounding-order change smaller than compiled-vs-eager (LPIPS-gated).

Per-arch gate: sm80 / sm89 / sm120 on (measured), everything else stock. sm75: Triton cannot lower the int8 dot;
sm100: Triton int8 is slower than cuBLAS. ROCm (weight-only int8 there) and CPU never reach it.

Kill switch: ``UNSLOTH_DIFFUSION_INT8_GEMM=0``; ``=1`` also enables it on an unmeasured arch (still probe-gated).

ConvRot Linears (MiniMax-H3) run ``ConvRotLinear.forward``'s own rotation first, so eager output stays bit-identical;
kill switch ``UNSLOTH_DIFFUSION_INT8_GEMM_CONVROT=0``.

Rotated Linears on sm120 / sm80 (group 256) run the rotation and the act quant as ONE kernel (``rotq_i8``): the same
rotation GEMM in the same K order, then torchao's per-row quant with every bf16 rounding kept, so codes and scales are
bit-identical (probed per device). Kill switch ``UNSLOTH_DIFFUSION_INT8_ROTQUANT=0``.

A block-streamed denoiser installs against its onload device (``install(..., device = ...)``): group offload's
``swap_tensors`` keeps each Parameter's identity and the forward reads the payload off the live Parameter, so every
call sees the onloaded copy. Kill switch ``UNSLOTH_DIFFUSION_INT8_GEMM_STREAMED=0``.
"""

from __future__ import annotations

import os
import threading
import types
from functools import lru_cache
from typing import Any, Optional

INT8_GEMM_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM"
INT8_GEMM_CONVROT_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM_CONVROT"
INT8_GEMM_STREAMED_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM_STREAMED"
INT8_ROTQUANT_ENV = "UNSLOTH_DIFFUSION_INT8_ROTQUANT"
_MIN_TRITON = (3, 2)
# torch._int_mm needs M > 16; below that the stock path (safe_int_mm padding) stays in charge.
_MIN_ROWS = 17
_K_ALIGN = 64
_N_ALIGN = 16
_OP_NAMESPACE = "unsloth_studio"
_OP_NAME = "int8_mm_dequant"
_ROTQ_OP_NAME = "convrot_act_quant_int8"
_REC = "_unsloth_i8_gemm"
_MARK = "_unsloth_i8_gemm_prev"
_NO_PREV = object()
_LOCK = threading.Lock()
_OP_HANDLE: Any = None
# Engagement census; never read inside a traced region.
_CALLS = [0]

# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) per (major, minor), tuned on DiT shapes. Absent = off.
_ARCH_CONFIG = {
    (8, 0): (
        128,
        128,
        128,
        8,
        4,
        3,
    ),  # A100
    (8, 9): (256, 128, 128, 8, 8, 3),  # L4
    (12, 0): (128, 128, 64, 8, 4, 4),  # RTX PRO 6000
}
# When the arch tile does not fit this part's shared memory.
_FALLBACK_CONFIG = (128, 128, 64, 8, 4, 4)

# rotq_i8 (BLOCK_M group rows, BLOCK_K, num_warps, num_stages); measured end to end on these archs only.
_ROTQ_CONFIG = {
    (8, 0): (128, 32, 8, 3),
    (12, 0): (128, 32, 8, 4),
}
_ROTQ_FALLBACK = (128, 32, 8, 3)
_ROTQ_GROUPS = (256,)


def int8_gemm_mode() -> str:
    """'off', 'force' or 'auto' from the environment."""
    raw = (os.environ.get(INT8_GEMM_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


def convrot_enabled() -> bool:
    raw = (os.environ.get(INT8_GEMM_CONVROT_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def streamed_enabled() -> bool:
    raw = (os.environ.get(INT8_GEMM_STREAMED_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def rotquant_enabled() -> bool:
    """ConvRot Linears quantize their activation with the fused rotation kernel unless
    ``UNSLOTH_DIFFUSION_INT8_ROTQUANT=0``."""
    raw = (os.environ.get(INT8_ROTQUANT_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def rotquant_config(capability: Optional[tuple], mode: Optional[str] = None) -> Optional[tuple]:
    """The fused rotation tile for this compute capability, or None (stock rotation)."""
    mode = int8_gemm_mode() if mode is None else mode
    if mode == "off" or capability is None or not rotquant_enabled():
        return None
    cap = (int(capability[0]), int(capability[1]))
    cfg = _ROTQ_CONFIG.get(cap)
    if cfg is None and mode == "force" and cap >= (8, 0):
        return _ROTQ_FALLBACK
    return cfg


def _triton_version_ok(version: Optional[str] = None) -> bool:
    if version is None:
        try:
            import triton
            version = str(triton.__version__)
        except Exception:  # noqa: BLE001
            return False
    import re

    match = re.match(r"(\d+)\.(\d+)", version)
    return bool(match) and (int(match.group(1)), int(match.group(2))) >= _MIN_TRITON


def arch_config(capability: Optional[tuple], mode: Optional[str] = None) -> Optional[tuple]:
    """The tile for this compute capability, or None where the lever stays off."""
    mode = int8_gemm_mode() if mode is None else mode
    if mode == "off" or capability is None:
        return None
    cap = (int(capability[0]), int(capability[1]))
    cfg = _ARCH_CONFIG.get(cap)
    if cfg is None and mode == "force" and cap >= (8, 0):
        return _FALLBACK_CONFIG
    return cfg


@lru_cache(maxsize = 1)
def _kernels() -> Optional[types.SimpleNamespace]:
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001 - no Triton means the stock path
        return None
    globals().update(triton = triton, tl = tl)

    @triton.jit
    def _rbf16(x):
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def i8mm_dq(
        a_ptr,
        w_ptr,
        xs_ptr,
        ws_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_wn,
        stride_cm,
        HAS_BIAS: tl.constexpr,
        XS_FP32: tl.constexpr,
        WS_FP32: tl.constexpr,
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
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # int64 row / column offsets: M * K overflows int32 on large video token grids.
        offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rm[:, None] * stride_am + offs_k[None, :]
        w_ptrs = w_ptr + rn[None, :] * stride_wn + offs_k[:, None]
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

        # torchao epilogue, every rounding kept: (int32 * xs) -> xs dtype, * ws -> ws dtype, -> bf16, + bias.
        xs = tl.load(xs_ptr + rm).to(tl.float32)
        ws = tl.load(ws_ptr + rn).to(tl.float32)
        if XS_FP32:
            y = acc.to(tl.float32)
        else:
            # int32 -> fp32 -> bf16 rounds twice; Triton folds ``acc.to(fp32).to(bf16)`` into one rounding.
            y = _rbf16(tl.extra.cuda.libdevice.int2float_rn(acc))
        y = _rbf16(y * xs[:, None])
        # *_rn: ptxas fuses packed f32x2 mul + add into an FMA on sm_100 even with fp fusion off.
        y = tl.extra.cuda.libdevice.mul_rn(y, ws[None, :])
        if not WS_FP32:
            y = _rbf16(y)
        if HAS_BIAS:
            y = tl.extra.cuda.libdevice.add_rn(y, tl.load(b_ptr + rn).to(tl.float32)[None, :])
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, y.to(tl.bfloat16), mask = mask)

    from triton.language.extra import libdevice

    @triton.jit
    def rotq_i8(
        x_ptr,
        h_ptr,
        q_ptr,
        s_ptr,
        M,
        NG,
        G: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ROWS: tl.constexpr,
        ROWS_P2: tl.constexpr,
        QMIN: tl.constexpr,
        QMAX: tl.constexpr,
        DIV: tl.constexpr,
        EPS: tl.constexpr,
        FP32_SCALE: tl.constexpr,
    ):
        # one program = ROWS whole activation rows, so the per-row amax closes inside the tile
        pid = tl.program_id(0)
        t = tl.arange(0, BLOCK_M)
        lr = t // NG
        row = pid.to(tl.int64) * ROWS + lr
        live = (lr < ROWS) & (row < M)
        rg = tl.where(live, row * NG + (t % NG), 0)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, G)
        acc = tl.zeros((BLOCK_M, G), dtype = tl.float32)
        a_ptrs = x_ptr + rg[:, None] * G + offs_k[None, :]
        h_ptrs = h_ptr + offs_k[:, None] * G + offs_n[None, :]
        for _k in range(0, G // BLOCK_K):
            a = tl.load(a_ptrs, mask = live[:, None], other = 0.0)
            acc = tl.dot(a, tl.load(h_ptrs), acc, out_dtype = tl.float32)
            a_ptrs += BLOCK_K
            h_ptrs += BLOCK_K * G
        z = _rbf16(acc)
        # s = bf16(max(bf16(amax / DIV), EPS)), q = clamp(rint(bf16(z * bf16(1 / s)))); FP32_SCALE (torchao >= 0.18
        # Int8Tensor) keeps the reciprocal and the product in fp32.
        j = tl.arange(0, ROWS_P2)
        sel = lr[:, None] == j[None, :]
        amax = tl.max(tl.where(sel, tl.max(tl.abs(z), axis = 1)[:, None], 0.0), axis = 0)
        s = _rbf16(tl.maximum(_rbf16(libdevice.div_rn(amax, DIV)), EPS))
        inv = libdevice.div_rn(tl.full((ROWS_P2,), 1.0, tl.float32), s)
        if not FP32_SCALE:
            inv = _rbf16(inv)
        inv_t = tl.max(tl.where(sel, inv[None, :], 0.0), axis = 1)
        p = z * inv_t[:, None]
        if not FP32_SCALE:
            p = _rbf16(p)
        p = libdevice.rint(p)
        p = tl.minimum(tl.maximum(p, QMIN), QMAX)
        tl.store(q_ptr + rg[:, None] * G + offs_n[None, :], p.to(tl.int8), mask = live[:, None])
        srow = pid.to(tl.int64) * ROWS + j
        tl.store(s_ptr + srow, s.to(s_ptr.dtype.element_ty), mask = (j < ROWS) & (srow < M))

    return types.SimpleNamespace(i8mm_dq = i8mm_dq, rotq_i8 = rotq_i8)


# device index -> probed tile (None = stock).
_DEVICE_CFG: dict = {}


def reference(a: Any, w: Any, xs: Any, ws: Any, bias: Any) -> Any:
    """Eager torchao epilogue, same ops and order as the stock linear."""
    import torch

    m = a.shape[0]
    if m < _MIN_ROWS:  # _int_mm's row floor: pad, then drop the pad rows (exact)
        a = torch.cat([a, a.new_zeros((_MIN_ROWS - m, a.shape[1]))])
    c = torch._int_mm(a, w.t())[:m]
    y = (c * xs.reshape(-1, 1)).to(torch.bfloat16)
    y = y * ws.reshape(
        -1
    )  # bf16 scales: a bf16 product; fp32 (prequant) scales: fp32 until after the bias (v2)
    if bias is not None:
        y = y + bias
    return y.to(torch.bfloat16)


def _launch(a: Any, w: Any, xs: Any, ws: Any, bias: Any, cfg: tuple) -> Any:
    import torch

    k = _kernels()
    m, kk = a.shape
    n = w.shape[0]
    bm, bn, bk, gm, warps, stages = cfg
    out = torch.empty((m, n), device = a.device, dtype = torch.bfloat16)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    with torch.cuda.device(a.device):
        k.i8mm_dq[grid](
            a,
            w,
            xs,
            ws,
            bias if bias is not None else ws,
            out,
            m,
            n,
            kk,
            a.stride(0),
            w.stride(0),
            out.stride(0),
            HAS_BIAS = bias is not None,
            XS_FP32 = xs.dtype == torch.float32,
            WS_FP32 = ws.dtype == torch.float32,
            EVEN_K = kk % bk == 0,
            BLOCK_M = bm,
            BLOCK_N = bn,
            BLOCK_K = bk,
            GROUP_M = gm,
            num_warps = warps,
            num_stages = stages,
            enable_fp_fusion = False,
        )
    return out


def _aligned(a: Any, w: Any) -> bool:
    """K, N, both row strides and both int8 base pointers on 16 bytes. Triton specialises on these, and an off-16 variant
    spills; the driver then reserves that local memory device-wide for the process, outside PyTorch's allocator."""
    return (
        a.shape[1] % 16 == 0
        and w.shape[0] % 16 == 0
        and a.stride(0) % 16 == 0
        and w.stride(0) % 16 == 0
        and a.data_ptr() % 16 == 0
        and w.data_ptr() % 16 == 0
    )


def _run(a: Any, w: Any, xs: Any, ws: Any, bias: Any) -> Any:
    """Op body: the probed tile for this device, else (launch failure, ``_aligned`` false) the stock epilogue."""
    _CALLS[0] += 1
    a = a if a.stride(-1) == 1 else a.contiguous()
    cfg = _DEVICE_CFG.get(a.device.index)
    if cfg is not None and w.stride(-1) == 1 and w.device == a.device and _aligned(a, w):
        try:
            return _launch(a, w, xs, ws, bias, cfg)
        except Exception:  # noqa: BLE001 - a failed launch keeps the stock math
            _DEVICE_CFG[a.device.index] = None
    return reference(a, w, xs, ws, bias)


@lru_cache(maxsize = 1)
def _op() -> Any:
    """``unsloth_studio::int8_mm_dequant`` (opaque to Inductor; no host sync, allocation only), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _OP_NAMESPACE, None)
    if ns is not None and hasattr(ns, _OP_NAME):
        return getattr(ns, _OP_NAME)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:  # torch < 2.4
        return None
    try:

        @custom_op(
            f"{_OP_NAMESPACE}::{_OP_NAME}",
            mutates_args = (),
            schema = "(Tensor a, Tensor w, Tensor xs, Tensor ws, Tensor? bias) -> Tensor",
        )
        def _int8_mm_dequant(a, w, xs, ws, bias):
            return _run(a, w, xs, ws, bias)

        @_int8_mm_dequant.register_fake
        def _(a, w, xs, ws, bias):
            return a.new_empty((a.shape[0], w.shape[0]), dtype = torch.bfloat16)
    except Exception:  # noqa: BLE001 - a registration failure keeps the stock path
        return None
    return getattr(torch.ops, _OP_NAMESPACE).int8_mm_dequant


def device_config(index: int) -> Optional[tuple]:
    """Probe once per device: arch gate, Triton, op registration, then a launch that must match the eager torchao
    epilogue bit for bit (bf16 and fp32 activation scales, bias, ragged M / N / K). None keeps the stock path."""
    if index in _DEVICE_CFG:
        return _DEVICE_CFG[index]
    cfg = None
    try:
        import torch
        if (
            torch.cuda.is_available()
            and not getattr(torch.version, "hip", None)
            and _triton_version_ok()
            and _kernels() is not None
            and _op() is not None
        ):
            want = arch_config(torch.cuda.get_device_capability(index))
            for cand in (want, _FALLBACK_CONFIG) if want is not None else ():
                if cand is not None and _probe(index, cand):
                    cfg = cand
                    break
    except Exception:  # noqa: BLE001
        cfg = None
    _DEVICE_CFG[index] = cfg
    return cfg


# (M, N, K, bias, fp32 scales). Ragged N / K stay on 16: an off-16 probe compiles the spilling variant (see ``_aligned``).
_PROBE_SHAPES = (
    (257, 384, 512, False, False),
    (33, 208, 144, True, True),
    (300, 528, 1040, True, False),
)


def _probe(index: int, cfg: tuple) -> bool:
    import torch

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(0)
    try:
        for m, n, k, bias, xs32 in _PROBE_SHAPES:
            a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).to(dev)
            w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).to(dev)
            xs = (torch.rand(m, generator = g) * 0.02 + 1e-4).to(torch.bfloat16)
            xs = (xs.float() if xs32 else xs).to(dev)
            ws = (torch.rand(n, generator = g) * 0.002 + 1e-5).to(torch.bfloat16).to(dev)
            ws = (
                ws.float() if xs32 else ws
            )  # fp32 weight scales + bias: the prequant (v2) rounding order
            b = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16).to(dev) if bias else None
            if not torch.equal(_launch(a, w, xs, ws, b, cfg), reference(a, w, xs, ws, b)):
                return False
        a, w = tie_operands(dev)
        xs = torch.ones(a.shape[0], device = dev, dtype = torch.bfloat16)
        ws = torch.ones(w.shape[0], device = dev, dtype = torch.bfloat16)
        if not torch.equal(_launch(a, w, xs, ws, None, cfg), reference(a, w, xs, ws, None)):
            return False
        torch.cuda.synchronize(dev)
        return True
    except Exception:  # noqa: BLE001 - out of shared memory, compile failure, ...
        return False


# (QMIN, QMAX, DIV, EPS): v1 _int8_symm_per_token_reduced_range_quant, v2 Int8Tensor.from_hp(PerRow, SYMMETRIC).
_ROTQ_QPARAMS = {
    "v1": (-127.0, 127.0, 127.0, 1e-5),
    "v2": (-128.0, 127.0, 127.5, 1.1920928955078125e-07),
}
# device index -> probed rotq tile (None = stock).
_ROTQ_DEVICE: dict = {}
_ROTQ_CALLS = [0]
_ROTQ_HANDLE: Any = None


@lru_cache(maxsize = 1)
def _v2_act_scale_fp32() -> bool:
    """torchao 0.18+ ``Int8Tensor.from_hp`` returns an fp32 activation scale (and quantizes in fp32)."""
    try:
        import torch
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

        x = torch.ones(1, 16, dtype = torch.bfloat16)
        return Int8Tensor.from_hp(x, PerRow()).scale.dtype == torch.float32
    except Exception:  # noqa: BLE001 - unknown: the probe below still has to match bit for bit
        return False


def _rotq_scale_fp32(kind: str) -> bool:
    return kind == "v2" and _v2_act_scale_fp32()


def rotquant_reference(x2d: Any, group: int, kind: str) -> tuple:
    """The stock path: ConvRotLinear.forward's rotation (bf16 GEMM), then torchao's activation quant. (int8, scale)."""
    import torch
    from .diffusion_convrot import build_convrot_hadamard, rotate_convrot_activation

    xr = rotate_convrot_activation(
        x2d, build_convrot_hadamard(group, device = x2d.device, dtype = x2d.dtype), group
    )
    if kind == "v1":
        q, scale = _act_quant_v1(xr)
    else:
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

        t = Int8Tensor.from_hp(xr, PerRow())
        q, scale = t.qdata, t.scale
    if _rotq_scale_fp32(kind):
        return q, scale.reshape(-1).to(torch.float32)
    return q, scale.reshape(-1).to(torch.bfloat16)


def _rotq_rows(k: int, group: int, cfg: tuple) -> int:
    """Whole activation rows per program (0: this K does not fit one tile)."""
    return cfg[0] // (k // group)


def _rotq_launch(x2d: Any, group: int, kind: str, cfg: tuple) -> tuple:
    import torch
    import triton
    from .diffusion_convrot import build_convrot_hadamard

    kern = _kernels()
    m, k = x2d.shape
    bm, bk, warps, stages = cfg
    rows = _rotq_rows(k, group, cfg)
    qmin, qmax, div, eps = _ROTQ_QPARAMS[kind]
    h = build_convrot_hadamard(group, device = x2d.device, dtype = torch.bfloat16)
    q = torch.empty((m, k), device = x2d.device, dtype = torch.int8)
    fp32_scale = _rotq_scale_fp32(kind)
    s = torch.empty((m,), device = x2d.device, dtype = torch.float32 if fp32_scale else torch.bfloat16)
    with torch.cuda.device(x2d.device):
        kern.rotq_i8[(triton.cdiv(m, rows),)](
            x2d,
            h,
            q,
            s,
            m,
            k // group,
            G = group,
            BLOCK_M = bm,
            BLOCK_K = bk,
            ROWS = rows,
            ROWS_P2 = max(triton.next_power_of_2(rows), 2),
            QMIN = qmin,
            QMAX = qmax,
            DIV = div,
            EPS = eps,
            FP32_SCALE = fp32_scale,
            num_warps = warps,
            num_stages = stages,
            enable_fp_fusion = False,
        )
    return q, s


def rotquant_supported(x2d: Any, group: int, cfg: Optional[tuple]) -> bool:
    """Shape / dtype / device / group gate of the fused kernel (anything else keeps the stock rotation + quant)."""
    import torch

    if (
        cfg is None
        or group not in _ROTQ_GROUPS
        or x2d.dim() != 2
        or x2d.dtype != torch.bfloat16
        or not x2d.is_cuda
    ):
        return False
    k = x2d.shape[1]
    return k % group == 0 and k >= group and _rotq_rows(k, group, cfg) >= 1


def _rotq_run(x2d: Any, group: int, v2: bool) -> tuple:
    """Op body: the probed tile for this device, the stock rotation + quant if anything does not fit."""
    _ROTQ_CALLS[0] += 1
    kind = "v2" if v2 else "v1"
    x2d = x2d if x2d.is_contiguous() else x2d.contiguous()
    cfg = _ROTQ_DEVICE.get(x2d.device.index)
    if rotquant_supported(x2d, group, cfg):
        try:
            return _rotq_launch(x2d, group, kind, cfg)
        except Exception:  # noqa: BLE001 - a failed launch keeps the stock math
            _ROTQ_DEVICE[x2d.device.index] = None
    return rotquant_reference(x2d, group, kind)


@lru_cache(maxsize = 1)
def _rotq_op() -> Any:
    """``unsloth_studio::convrot_act_quant_int8`` (opaque to Inductor, allocation only), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _OP_NAMESPACE, None)
    if ns is not None and hasattr(ns, _ROTQ_OP_NAME):
        return getattr(ns, _ROTQ_OP_NAME)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_OP_NAMESPACE}::{_ROTQ_OP_NAME}",
            mutates_args = (),
            schema = "(Tensor x, int group, bool v2) -> (Tensor, Tensor)",
        )
        def _convrot_act_quant_int8(x, group, v2):
            return _rotq_run(x, group, v2)

        @_convrot_act_quant_int8.register_fake
        def _(x, group, v2):
            return (
                x.new_empty((x.shape[0], x.shape[1]), dtype = torch.int8),
                x.new_empty(
                    (x.shape[0],),
                    dtype = torch.float32
                    if _rotq_scale_fp32("v2" if v2 else "v1")
                    else torch.bfloat16,
                ),
            )
    except Exception:  # noqa: BLE001
        return None
    return getattr(getattr(torch.ops, _OP_NAMESPACE), _ROTQ_OP_NAME)


def rotquant_device_config(index: int) -> Optional[tuple]:
    """Once per device: arch gate, then bit-exact codes and scales vs the stock path (v1, v2, outliers, zero row, ragged
    M). None = stock."""
    if index in _ROTQ_DEVICE:
        return _ROTQ_DEVICE[index]
    cfg = None
    try:
        import torch
        if (
            torch.cuda.is_available()
            and not getattr(torch.version, "hip", None)
            and _triton_version_ok()
            and _kernels() is not None
            and _rotq_op() is not None
        ):
            want = rotquant_config(torch.cuda.get_device_capability(index))
            for cand in (want, _ROTQ_FALLBACK) if want is not None else ():
                if _rotq_probe(index, cand):
                    cfg = cand
                    break
    except Exception:  # noqa: BLE001
        cfg = None
    _ROTQ_DEVICE[index] = cfg
    return cfg


def _rotq_probe(index: int, cfg: tuple) -> bool:
    import torch

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(2)
    try:
        for m, k in ((257, 256 * 3), (33, 256 * 21), (130, 256 * 56), (17, 256 * 128)):
            if _rotq_rows(k, 256, cfg) < 1:
                continue
            x = torch.randn(m, k, generator = g) * (torch.rand(1, k, generator = g) * 4)
            x[:, :3] *= 60
            x[1] = 0
            x[2, 11] = 3e4
            x = x.to(torch.bfloat16).to(dev)
            for kind in ("v1", "v2"):
                q, s = _rotq_launch(x, 256, kind, cfg)
                rq, rs = rotquant_reference(x, 256, kind)
                if not (torch.equal(q, rq) and torch.equal(s, rs)):
                    return False
        torch.cuda.synchronize(dev)
        return True
    except Exception:  # noqa: BLE001
        return False


def rotquant_call_count() -> int:
    return _ROTQ_CALLS[0]


def tie_operands(
    device: Any,
    rows: int = 32,
    k: int = 4096,
) -> tuple:
    """int8 (a, w) whose int32 products sit one or two units off a bf16 midpoint in [2^24, 2^26), both signs."""
    import torch

    k2 = 128
    k1 = k - k2
    targets = []
    for e in (24, 25):
        for mult in (0, 3, 50, 100):
            mid = (1 << e) + mult * (1 << (e - 7)) + (1 << (e - 8))
            for d in (-1, 1, -2, 2):
                targets += [mid + d, -(mid + d)]
    w = torch.zeros(len(targets), k, dtype = torch.int8)
    for j, t in enumerate(targets):
        s1, s2 = divmod(abs(t), 127)
        full, rem = divmod(s1, 127)
        sign = 1 if t >= 0 else -1
        w[j, :full] = 127 * sign
        if rem:
            w[j, full] = rem * sign
        w[j, k1] = s2 * sign
    a = torch.ones(rows, k, dtype = torch.int8)
    a[:, :k1] = 127
    return a.to(device), w.to(device)


def _v1_parts(w: Any) -> Optional[tuple]:
    """(int8 [N, K], scale [N]) of a torchao v1 ``LinearActivationQuantizedTensor`` over a plain-layout symmetric
    per-channel ``AffineQuantizedTensor`` with the per-token reduced-range activation quant, else None."""
    try:
        if type(w).__name__ != "LinearActivationQuantizedTensor":
            return None
        fn = getattr(w, "input_quant_func", None)
        if getattr(fn, "__name__", "") != "_int8_symm_per_token_reduced_range_quant":
            return None
        if getattr(w, "quant_kwargs", None):
            return None
        aqt = w.original_weight_tensor
        if type(aqt).__name__ != "AffineQuantizedTensor":
            return None
        impl = aqt.tensor_impl
        if (
            type(impl).__name__ != "PlainAQTTensorImpl"
            or type(getattr(impl, "_layout", None)).__name__ != "PlainLayout"
        ):
            return None
        data, scale, zp = impl.int_data, impl.scale, getattr(impl, "zero_point", None)
        import torch

        if data.dtype != torch.int8 or data.dim() != 2 or scale.numel() != data.shape[0]:
            return None
        if zp is not None and bool((zp != 0).any()):
            return None
        if tuple(getattr(aqt, "block_size", ())) != (1, data.shape[1]):
            return None
        return data, scale.reshape(-1)
    except Exception:  # noqa: BLE001
        return None


def _v2_parts(w: Any, check_zero_point: bool = True) -> Optional[tuple]:
    """(int8 [N, K], scale [N]) of a plain dynamic symmetric per-row torchao ``Int8Tensor``, else None.
    ``check_zero_point=False`` inside a traced forward: the check is a host sync and a graph break."""
    try:
        from .diffusion_int8_fused import _plain_int8_weight

        if not _plain_int8_weight(w):
            return None
        if (
            check_zero_point
            and getattr(w, "zero_point", None) is not None
            and bool((w.zero_point != 0).any())
        ):
            return None
        return w.qdata, w.scale.reshape(-1)
    except Exception:  # noqa: BLE001
        return None


def _act_quant_v1(x2d: Any) -> tuple:
    """torchao 0.17's ``_int8_symm_per_token_reduced_range_quant`` minus the tensor-subclass wrapper."""
    import torch
    from torchao.quantization.quant_primitives import (
        MappingType,
        choose_qparams_affine,
        quantize_affine,
    )

    block = (1, x2d.shape[-1])
    scale, zero_point = choose_qparams_affine(
        x2d,
        MappingType.SYMMETRIC,
        block,
        torch.int8,
        -127,
        127,
        1e-5,
        torch.float32 if x2d.dtype == torch.float16 else None,
        None,
    )
    q = quantize_affine(x2d, block, scale, zero_point, torch.int8, -127, 127)
    return q, scale


def _act_quant_v2(x2d: Any, weight: Any) -> tuple:
    from .diffusion_int8_fused import _act_quant
    return _act_quant(x2d, weight)


@lru_cache(maxsize = 4)
def _v1_act_quant_matches(index: int) -> bool:
    """The subclass-free v1 activation quant must equal torchao's own, bit for bit, on this torchao."""
    try:
        import torch
        from torchao.quantization.quant_api import _int8_symm_per_token_reduced_range_quant

        g = torch.Generator(device = "cpu").manual_seed(1)
        x = (torch.randn(67, 256, generator = g) * 3).to(torch.bfloat16)
        x[5] = 0
        x[9, :7] *= 300
        x = x.to(torch.device("cuda", index))
        ref = _int8_symm_per_token_reduced_range_quant(x)
        q, s = _act_quant_v1(x)
        impl = ref.tensor_impl
        return bool(
            torch.equal(q, impl.int_data) and torch.equal(s.reshape(-1), impl.scale.reshape(-1))
        )
    except Exception:  # noqa: BLE001
        return False


def _linear_forward(self: Any, x: Any) -> Any:
    """``nn.Linear.forward`` for a torchao int8 dynamic weight: torchao's activation quant, then the fused GEMM."""
    import torch

    rec = self.__dict__.get(_REC)
    if rec is None or x.dtype != torch.bfloat16 or not x.is_cuda:
        return type(self).forward(self, x)
    kind, group, rotq, weight = rec
    if self.weight is not weight:  # weight replaced since install (reload / LoRA bake): stock
        return type(self).forward(self, x)
    # Payload off the live parameter, never a cached alias: a moved weight must not leave a stale device or pin a copy.
    if kind == "v1":
        impl = weight.original_weight_tensor.tensor_impl
        wq, ws = impl.int_data, impl.scale.reshape(-1)
    else:
        wq, ws = weight.qdata, weight.scale.reshape(-1)
    if wq.device != x.device:
        return type(self).forward(self, x)
    lead = x.shape[:-1]
    if x.numel() < _MIN_ROWS * x.shape[-1]:
        return type(self).forward(self, x)
    if (
        rotq
        and _ROTQ_HANDLE is not None
        and rotquant_supported(x.reshape(-1, x.shape[-1]), group, _ROTQ_DEVICE.get(x.device.index))
    ):
        xq, xs = _ROTQ_HANDLE(x.reshape(-1, x.shape[-1]), group, kind == "v2")
    else:
        if group is not None:
            from .diffusion_convrot import build_convrot_hadamard, rotate_convrot_activation
            x = rotate_convrot_activation(
                x, build_convrot_hadamard(group, device = x.device, dtype = x.dtype), group
            )
        x2d = x.reshape(-1, x.shape[-1])
        if kind == "v1":
            xq, xs = _act_quant_v1(x2d)
        else:
            xq, xs = _act_quant_v2(x2d, weight)
    y = _OP_HANDLE(xq, wq, xs.reshape(-1), ws, self.bias)
    return y.reshape(*lead, y.shape[-1])


def linear_from_q(q: Any, xs: Any, weight: Any, bias: Any) -> Optional[Any]:
    """For ``diffusion_int8_fused``: the down projection on an already-quantized activation, or None (stock)."""
    import torch

    if _OP_HANDLE is None or int8_gemm_mode() == "off" or q.device.index is None:
        return None
    if _DEVICE_CFG.get(q.device.index) is None:
        return None
    # The fused MLP already admitted this weight (symmetric, and its stock epilogue ignores the zero point too).
    parts = _v2_parts(weight, check_zero_point = False)
    if (
        parts is None
        or q.shape[0] < _MIN_ROWS
        or parts[1].dtype not in (torch.bfloat16, torch.float32)
    ):
        return None
    if parts[0].shape[1] % _K_ALIGN or parts[0].shape[0] % _N_ALIGN:
        return None
    if bias is not None and bias.dtype != torch.bfloat16:
        return None
    return _OP_HANDLE(q, parts[0], xs.reshape(-1), parts[1], bias)


def _rotation_group(module: Any) -> Optional[int]:
    """The ConvRot group the forward above can reproduce, else None."""
    try:
        from .diffusion_convrot import convrot_linear_class, is_power_of_four

        if type(module) is not convrot_linear_class():
            return None
        group = module.convrot_groupsize
        if not is_power_of_four(group) or module.in_features % group:
            return None
        return int(group)
    except Exception:  # noqa: BLE001
        return None


def _eligible(module: Any) -> Optional[tuple]:
    """(kind, rotation group or None, scale, weight) for a Linear the fused GEMM can run, else None."""
    from torch import nn

    group = None
    if type(module) is not nn.Linear:
        # any other subclass transforms its input in a way this forward would skip
        group = _rotation_group(module) if convrot_enabled() else None
        if group is None:
            return None
    w = module.weight
    bias = module.bias
    import torch

    if bias is not None and bias.dtype != torch.bfloat16:
        return None
    # v1: bf16 scales only; v2 also fp32 (prequant checkpoints). Off-grid K runs masked loads, far slower than cuBLAS.
    out_f, in_f = getattr(module, "out_features", 0), getattr(module, "in_features", 0)
    if in_f % _K_ALIGN or out_f % _N_ALIGN:
        return None
    parts = _v1_parts(w)
    if parts is not None and parts[1].dtype == torch.bfloat16:
        return ("v1", group, parts[1], w)
    parts = _v2_parts(w)
    if parts is not None and parts[1].dtype in (torch.bfloat16, torch.float32):
        return ("v2", group, parts[1], w)
    return None


def candidates(transformer: Any) -> int:
    try:
        return sum(1 for m in transformer.modules() if _eligible(m) is not None)
    except Exception:  # noqa: BLE001
        return 0


def install(
    transformer: Any,
    logger: Any = None,
    offload_active: bool = False,
    device: Any = None,
) -> int:
    """Idempotent; returns the (candidate) count. Must run before the first compiled forward.

    ``device``: the onload device of a block-streamed denoiser, whose weights sit on the host between blocks; the
    probe runs there and the swap happens now. Without it an offloaded denoiser keeps the stock path."""
    if int8_gemm_mode() == "off" or transformer is None:
        return 0
    if device is not None:
        if not streamed_enabled():
            return 0
        return _finalize(transformer, logger, device = device)
    if offload_active:
        return 0
    from .diffusion_int8_fused import resident_cuda_device, run_on_first_call

    if resident_cuda_device(transformer) is not None:
        return _finalize(transformer, logger)
    # Weights not on the GPU yet: arch gate now (status never claims a stock arch), probe + swap at the first forward.
    try:
        import torch

        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return 0
        if not _triton_version_ok() or _kernels() is None:
            return 0
        if arch_config(torch.cuda.get_device_capability(torch.cuda.current_device())) is None:
            return 0
    except Exception:  # noqa: BLE001
        return 0
    n = candidates(transformer)
    if n:
        run_on_first_call(transformer, "int8_gemm", lambda t: _finalize(t, logger))
    return n


def _finalize(
    transformer: Any,
    logger: Any = None,
    device: Any = None,
) -> int:
    count = _swap(transformer, logger, device = device)
    try:
        transformer._unsloth_int8_gemm = count  # the deferred install recorded the candidate count
    except Exception:  # noqa: BLE001
        pass
    return count


def _swap(
    transformer: Any,
    logger: Any = None,
    device: Any = None,
) -> int:
    global _OP_HANDLE, _ROTQ_HANDLE
    from .diffusion_int8_fused import resident_cuda_device

    import torch

    if device is not None:
        try:
            dev = torch.device(device)
        except Exception:  # noqa: BLE001
            return 0
        if (
            dev.type != "cuda"
            or getattr(torch.version, "hip", None)
            or not torch.cuda.is_available()
        ):
            return 0
    else:
        dev = resident_cuda_device(transformer)
    if dev is None:
        return 0

    index = dev.index if dev.index is not None else torch.cuda.current_device()
    recs = [(m, _eligible(m)) for m in transformer.modules()]
    recs = [(m, r) for m, r in recs if r is not None]
    if not recs or device_config(index) is None:
        return 0
    if any(r[0] == "v1" for _, r in recs) and not _v1_act_quant_matches(index):
        recs = [(m, r) for m, r in recs if r[0] != "v1"]
    rotq = any(r[1] is not None for _, r in recs) and rotquant_device_config(index) is not None
    count = 0
    with _LOCK:
        _OP_HANDLE = _op()
        if _OP_HANDLE is None:
            return 0
        if rotq:
            _ROTQ_HANDLE = _rotq_op()
        for module, rec in recs:
            if _MARK in module.__dict__:
                count += 1
                continue
            # kind, rotation group, fused rotation + act quant on, the Parameter; no payload alias
            module.__dict__[_REC] = (rec[0], rec[1], bool(rotq and rec[1] is not None), rec[3])
            module.__dict__[_MARK] = module.__dict__.get("forward", _NO_PREV)
            module.forward = types.MethodType(_linear_forward, module)
            count += 1
    if logger is not None and count:
        logger.info(
            "diffusion.int8_gemm: %d int8 Linear(s) run the fused-dequant GEMM (bf16 out) on sm_%d%d%s",
            count,
            *torch.cuda.get_device_capability(index),
            ", ConvRot rotation fused into the act quant" if rotq else "",
        )
    return count


def uninstall(transformer: Any = None) -> None:
    if transformer is None:
        return
    from .diffusion_int8_fused import cancel_first_call

    cancel_first_call(transformer, "int8_gemm")
    with _LOCK:
        for module in transformer.modules():
            prev = module.__dict__.pop(_MARK, None)
            if prev is None:
                continue
            module.__dict__.pop(_REC, None)
            if prev is _NO_PREV:
                module.__dict__.pop("forward", None)
            else:
                module.forward = prev


def is_installed(module: Any) -> bool:
    return _MARK in getattr(module, "__dict__", {})


def call_count() -> int:
    return _CALLS[0]

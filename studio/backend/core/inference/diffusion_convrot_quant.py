# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused ConvRot rotation and int8 per-row quant (rotq_i8), bit-identical to stock; each device is
probed first and a miss keeps the stock path; H16 variant not bit-exact;
UNSLOTH_DIFFUSION_INT8_ROTQUANT=0 disables"""

from __future__ import annotations

import os
import types
from functools import lru_cache
from typing import Any, Optional
from .rocm_bf16 import is_rocm_torch

INT8_ROTQUANT_ENV = "UNSLOTH_DIFFUSION_INT8_ROTQUANT"
INT8_ROTQUANT_H16_ENV = "UNSLOTH_DIFFUSION_INT8_ROTQUANT_H16"
INT8_ROTQUANT_SWIGLU_ENV = "UNSLOTH_DIFFUSION_INT8_ROTQUANT_SWIGLU"
_OP_NAMESPACE = "unsloth_studio"
_ROTQ_OP_NAME = "convrot_act_quant_int8"
_SWIGLU_OP_NAME = "convrot_swiglu_act_quant_int8"

# (BLOCK_M group rows, BLOCK_K, num_warps, num_stages) per (major, minor); measured end to end. Absent = stock.
_ROTQ_CONFIG = {
    (8, 0): (128, 32, 8, 3),  # A100
    (8, 9): (128, 32, 8, 3),  # L4
    (12, 0): (128, 32, 8, 4),  # RTX PRO 6000
}
_ROTQ_FALLBACK = (128, 32, 8, 3)
_ROTQ_GROUPS = (256,)
# Per-K tiles: first match wins, bounds inclusive; tiles whose BLOCK_M can't hold a row skipped.
_ROTQ_NARROW = (64, 32, 4, 3)
_ROTQ_K_TILES: dict = {
    cap: ((256, 8192, _ROTQ_NARROW), (11264, 16384, _ROTQ_NARROW))
    for cap in ((8, 0), (8, 9), (12, 0))
}

# (QMIN, QMAX, DIV, EPS): v1 _int8_symm_per_token_reduced_range_quant, v2 Int8Tensor.from_hp(PerRow, SYMMETRIC).
_ROTQ_QPARAMS = {
    "v1": (-127.0, 127.0, 127.0, 1e-5),
    "v2": (-128.0, 127.0, 127.5, 1.1920928955078125e-07),
}
# device index -> probed rotq tile (None = stock), and the per-K rules whose tile passed the probe there.
_ROTQ_DEVICE: dict = {}
_ROTQ_DEVICE_K: dict = {}
_ROTQ_CALLS = [0]
# Set once a device passed the probe; read by traced forwards (dynamo must not trace into the lru_cache'd registration).
_ROTQ_HANDLE: Any = None
# device index -> passing rotq_h16 mode ("ieee" / "tf32x3"; falsy = dense rotq_i8) / SwiGLU probe result
_H16_DEVICE: dict = {}
_SWIGLU_DEVICE: dict = {}
_H16_CALLS = [0]
_SWIGLU_CALLS = [0]
_SWIGLU_HANDLE: Any = None
# 16 group rows per program: fastest (within 2%) on L4; 128 rows overflow 100 KB smem. One masked row per program at
# K 12288 was 1.2x slower on A100 and spilled on B200.
_H16_TILE_ROWS = 16
_H16_WARPS = 4
_H16_WARPS_TWO_PASS = 8
# Second stage: tf32x3 (fp32-level error) where fp32 FMA rate is a fraction of tf32 (A100: 1.2x faster); sm89 / sm86 /
# sm120 run fp32 FMAs at their tf32 rate, so they keep IEEE FMAs.
_H16_TF32X3_ARCHS = ((8, 0), (9, 0), (10, 0), (10, 3))
# max chunks per row to unroll (a longer unrolled row overflows sm100 TMEM)
_H16_UNROLL_MAX = 4


def _env_mode(name: str) -> str:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


def rotquant_enabled() -> bool:
    """False only under ``UNSLOTH_DIFFUSION_INT8_ROTQUANT=0``."""
    return _env_mode(INT8_ROTQUANT_ENV) != "off"


def rotquant_h16_enabled() -> bool:
    """False only under ``UNSLOTH_DIFFUSION_INT8_ROTQUANT_H16=0`` (the dense, bit-identical ``rotq_i8``)."""
    return _env_mode(INT8_ROTQUANT_H16_ENV) != "off"


def rotquant_swiglu_enabled() -> bool:
    """False only under ``UNSLOTH_DIFFUSION_INT8_ROTQUANT_SWIGLU=0``."""
    return _env_mode(INT8_ROTQUANT_SWIGLU_ENV) != "off"


def rotquant_mode() -> str:
    """'off', 'force' or 'auto'. The int8 GEMM switch is the master: its ``0`` turns the whole swap off and its ``1``
    forces this lever too."""
    gemm = _env_mode("UNSLOTH_DIFFUSION_INT8_GEMM")
    own = _env_mode(INT8_ROTQUANT_ENV)
    if gemm == "off" or own == "off":
        return "off"
    if gemm == "force" or own == "force":
        return "force"
    return "auto"


def rotquant_config(capability: Optional[tuple], mode: Optional[str] = None) -> Optional[tuple]:
    """The fused rotation tile for this compute capability, or None (stock rotation)."""
    mode = rotquant_mode() if mode is None else mode
    if mode == "off" or capability is None or not rotquant_enabled():
        return None
    cap = (int(capability[0]), int(capability[1]))
    cfg = _ROTQ_CONFIG.get(cap)
    if cfg is None and mode == "force" and cap >= (8, 0):
        return _ROTQ_FALLBACK
    return cfg


def rotq_k_tiles(capability: Optional[tuple]) -> tuple:
    """This arch's per-K rules, or () (``UNSLOTH_DIFFUSION_INT8_GEMM_TILES=0`` keeps every K on the default tile)."""
    if capability is None or _env_mode("UNSLOTH_DIFFUSION_INT8_GEMM_TILES") == "off":
        return ()
    return tuple(_ROTQ_K_TILES.get((int(capability[0]), int(capability[1])), ()))


def _triton_ok() -> bool:
    try:
        import re

        import triton

        match = re.match(r"(\d+)\.(\d+)", str(triton.__version__))
        return bool(match) and (int(match.group(1)), int(match.group(2))) >= (3, 2)
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize = 1)
def _kernels() -> Optional[types.SimpleNamespace]:
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001 - no Triton means the stock path
        return None

    @triton.jit
    def _rbf16(x):
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _swiglu_bf16(g, v):
        # Inductor's silu(g) * v rounding. ATen's IEEE-divide form matched eager bitwise but was 1.5-2.3x slower on B200.
        gf = g.to(tl.float32)
        silu = _rbf16(gf * tl.sigmoid(gf))
        return (silu * v.to(tl.float32)).to(tl.bfloat16)

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
        # s = bf16(max(bf16(amax / DIV), EPS)), q = clamp(rint(bf16(z * bf16(1 / s))))
        # FP32_SCALE (torchao >= 0.18 Int8Tensor) keeps the reciprocal and the product in fp32.
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

    @triton.jit
    def _h16_tile(
        x_ptr,
        v_ptr,
        gidx,
        live,
        h,
        h32,
        TM: tl.constexpr,
        SWIGLU: tl.constexpr,
        TF32X3: tl.constexpr,
    ):
        # X[a, b] = x[16 a + b]; Y = X . H16 on tensor cores (H16 entries +-1/4: exact products), Z = H16 . Y in fp32.
        offs = tl.arange(0, 256)
        ptrs = gidx[:, None] * 256 + offs[None, :]
        x = tl.load(x_ptr + ptrs, mask = live[:, None], other = 0.0)
        if SWIGLU:
            x = _swiglu_bf16(x, tl.load(v_ptr + ptrs, mask = live[:, None], other = 0.0))
        y = tl.dot(tl.reshape(x, (TM * 16, 16)), h, out_dtype = tl.float32)
        y = tl.reshape(tl.permute(tl.reshape(y, (TM, 16, 16)), (0, 2, 1)), (TM * 16, 16))
        if TF32X3:
            z = tl.dot(y, h32, input_precision = "tf32x3", out_dtype = tl.float32)
        else:
            z = tl.dot(y, h32, input_precision = "ieee", out_dtype = tl.float32)
        z = tl.reshape(tl.permute(tl.reshape(z, (TM, 16, 16)), (0, 2, 1)), (TM, 256))
        return _rbf16(z)

    @triton.jit
    def _h16_quant_store(
        z,
        inv_t,
        q_ptr,
        gidx,
        live,
        QMIN: tl.constexpr,
        QMAX: tl.constexpr,
        FP32_SCALE: tl.constexpr,
    ):
        offs = tl.arange(0, 256)
        p = z * inv_t[:, None]
        if not FP32_SCALE:
            p = _rbf16(p)
        p = libdevice.rint(p)
        p = tl.minimum(tl.maximum(p, QMIN), QMAX)
        tl.store(q_ptr + gidx[:, None] * 256 + offs[None, :], p.to(tl.int8), mask = live[:, None])

    @triton.jit
    def _h16_amax_chunk(
        c,
        amax,
        x_ptr,
        v_ptr,
        row,
        gl,
        M,
        NG,
        h,
        h32,
        CG: tl.constexpr,
        ROWS: tl.constexpr,
        SWIGLU: tl.constexpr,
        TF32X3: tl.constexpr,
    ):
        live = row < M
        gidx = tl.where(live, row * NG + c * CG + gl, 0)
        z = _h16_tile(x_ptr, v_ptr, gidx, live, h, h32, ROWS * CG, SWIGLU, TF32X3)
        m = tl.where(live, tl.max(tl.abs(z), axis = 1), 0.0)
        return tl.maximum(amax, tl.max(tl.reshape(m, (ROWS, CG)), axis = 1))

    @triton.jit
    def _h16_quant_chunk(
        c,
        inv_t,
        x_ptr,
        v_ptr,
        q_ptr,
        row,
        gl,
        M,
        NG,
        h,
        h32,
        CG: tl.constexpr,
        ROWS: tl.constexpr,
        SWIGLU: tl.constexpr,
        TF32X3: tl.constexpr,
        QMIN: tl.constexpr,
        QMAX: tl.constexpr,
        FP32_SCALE: tl.constexpr,
    ):
        live = row < M
        gidx = tl.where(live, row * NG + c * CG + gl, 0)
        z = _h16_tile(x_ptr, v_ptr, gidx, live, h, h32, ROWS * CG, SWIGLU, TF32X3)
        _h16_quant_store(z, inv_t, q_ptr, gidx, live, QMIN, QMAX, FP32_SCALE)

    @triton.jit
    def rotq_h16(
        x_ptr,
        v_ptr,
        h_ptr,
        h32_ptr,
        q_ptr,
        s_ptr,
        M,
        NG,
        NCH: tl.constexpr,
        CG: tl.constexpr,
        ROWS: tl.constexpr,
        ONE_PASS: tl.constexpr,
        UNROLL: tl.constexpr,
        SWIGLU: tl.constexpr,
        TF32X3: tl.constexpr,
        QMIN: tl.constexpr,
        QMAX: tl.constexpr,
        DIV: tl.constexpr,
        EPS: tl.constexpr,
        FP32_SCALE: tl.constexpr,
    ):
        # ONE_PASS: rotate, amax, quantize from registers. Else an amax pass, then recompute + quant (row re-read from L2).
        TM: tl.constexpr = ROWS * CG
        pid = tl.program_id(0)
        t = tl.arange(0, TM)
        row = pid.to(tl.int64) * ROWS + t // CG
        gl = t % CG
        o16 = tl.arange(0, 16)
        h = tl.load(h_ptr + o16[:, None] * 16 + o16[None, :])
        h32 = tl.load(h32_ptr + o16[:, None] * 16 + o16[None, :])
        if ONE_PASS:
            live = row < M
            gidx = tl.where(live, row * NG + gl, 0)
            z = _h16_tile(x_ptr, v_ptr, gidx, live, h, h32, TM, SWIGLU, TF32X3)
            m = tl.where(live, tl.max(tl.abs(z), axis = 1), 0.0)
            amax = tl.max(tl.reshape(m, (ROWS, CG)), axis = 1)
        else:
            amax = tl.zeros((ROWS,), dtype = tl.float32)
            if UNROLL:
                for c in tl.static_range(NCH):
                    amax = _h16_amax_chunk(
                        c, amax, x_ptr, v_ptr, row, gl, M, NG, h, h32, CG, ROWS, SWIGLU, TF32X3
                    )
            else:
                for c in range(0, NCH):
                    amax = _h16_amax_chunk(
                        c, amax, x_ptr, v_ptr, row, gl, M, NG, h, h32, CG, ROWS, SWIGLU, TF32X3
                    )
        # s = bf16(max(bf16(amax / DIV), EPS)), q = clamp(rint(bf16(z * bf16(1 / s)))), as rotq_i8
        s = _rbf16(tl.maximum(_rbf16(libdevice.div_rn(amax, DIV)), EPS))
        inv = libdevice.div_rn(tl.full((ROWS,), 1.0, tl.float32), s)
        if not FP32_SCALE:
            inv = _rbf16(inv)
        inv_t = tl.reshape(tl.broadcast_to(inv[:, None], (ROWS, CG)), (TM,))
        if ONE_PASS:
            _h16_quant_store(z, inv_t, q_ptr, gidx, live, QMIN, QMAX, FP32_SCALE)
        elif UNROLL:
            for c in tl.static_range(NCH):
                _h16_quant_chunk(
                    c,
                    inv_t,
                    x_ptr,
                    v_ptr,
                    q_ptr,
                    row,
                    gl,
                    M,
                    NG,
                    h,
                    h32,
                    CG,
                    ROWS,
                    SWIGLU,
                    TF32X3,
                    QMIN,
                    QMAX,
                    FP32_SCALE,
                )
        else:
            for c in range(0, NCH):
                _h16_quant_chunk(
                    c,
                    inv_t,
                    x_ptr,
                    v_ptr,
                    q_ptr,
                    row,
                    gl,
                    M,
                    NG,
                    h,
                    h32,
                    CG,
                    ROWS,
                    SWIGLU,
                    TF32X3,
                    QMIN,
                    QMAX,
                    FP32_SCALE,
                )
        srow = pid.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        tl.store(s_ptr + srow, s.to(s_ptr.dtype.element_ty), mask = srow < M)

    return types.SimpleNamespace(rotq_i8 = rotq_i8, rotq_h16 = rotq_h16, triton = triton)


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


def act_quant_v1(x2d: Any) -> tuple:
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


def swiglu_reference(g2d: Any, v2d: Any) -> Any:
    """The stock MLP's down-projection input: ``F.silu(gate) * value`` on bf16 (ATen, eager)."""
    import torch.nn.functional as F
    return F.silu(g2d) * v2d


def rotquant_reference(
    x2d: Any,
    group: int,
    kind: str,
    v2d: Any = None,
) -> tuple:
    """The stock path: ConvRotLinear.forward's rotation (bf16 GEMM), then torchao's activation quant. (int8, scale).
    ``v2d``: ``x2d`` is a SwiGLU gate and this the value; the rotated activation is ``swiglu_reference``'s product."""
    import torch
    from .diffusion_convrot import build_convrot_hadamard, rotate_convrot_activation

    if v2d is not None:
        x2d = swiglu_reference(x2d, v2d)
    xr = rotate_convrot_activation(
        x2d, build_convrot_hadamard(group, device = x2d.device, dtype = x2d.dtype), group
    )
    if kind == "v1":
        q, scale = act_quant_v1(xr)
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
    from .diffusion_convrot import build_convrot_hadamard

    kern = _kernels()
    triton = kern.triton
    m, k = x2d.shape
    bm, bk, warps, stages = cfg
    rows = _rotq_rows(k, group, cfg)
    qmin, qmax, div, eps = _ROTQ_QPARAMS[kind]
    h = build_convrot_hadamard(group, device = x2d.device, dtype = torch.bfloat16)
    q = torch.empty((m, k), device = x2d.device, dtype = torch.int8)
    fp32_scale = _rotq_scale_fp32(kind)
    s = torch.empty((m,), device = x2d.device, dtype = torch.float32 if fp32_scale else torch.bfloat16)
    if m == 0:
        return q, s
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


def _h16_plan(ng: int, tile_rows: int = _H16_TILE_ROWS) -> tuple:
    """(groups per chunk CG, rows per program, chunks, one pass) for a row of ``ng`` 256-wide groups: CG the largest
    power of two dividing ng (at most tile_rows), so every chunk is full and no lane is masked off on a live row; one
    pass when CG == ng, else two (amax, then a recompute + quant)."""
    cg = 1
    while cg * 2 <= tile_rows and ng % (cg * 2) == 0:
        cg *= 2
    rows = max(1, tile_rows // cg)
    return cg, rows, ng // cg, cg == ng


def _h16_launch(
    x2d: Any,
    kind: str,
    v2d: Any = None,
    tile_rows: int = _H16_TILE_ROWS,
    warps: Optional[int] = None,
    tf32x3: Optional[bool] = None,
) -> tuple:
    """``rotq_h16`` on a contiguous bf16 ``[M, K]`` (K a multiple of 256): (int8 [M, K], scale [M]). ``v2d``: the
    SwiGLU value half (``x2d`` the gate)."""
    import torch
    from .diffusion_convrot import build_convrot_hadamard

    kern = _kernels()
    m, k = x2d.shape
    ng = k // 256
    cg, rows, nch, one = _h16_plan(ng, tile_rows)
    if tf32x3 is None:
        tf32x3 = _H16_DEVICE.get(x2d.device.index) == "tf32x3"
    if warps is None:
        warps = _H16_WARPS if one else _H16_WARPS_TWO_PASS
    qmin, qmax, div, eps = _ROTQ_QPARAMS[kind]
    h = build_convrot_hadamard(16, device = x2d.device, dtype = torch.bfloat16)
    h32 = build_convrot_hadamard(16, device = x2d.device, dtype = torch.float32)
    fp32_scale = _rotq_scale_fp32(kind)
    q = torch.empty((m, k), device = x2d.device, dtype = torch.int8)
    s = torch.empty((m,), device = x2d.device, dtype = torch.float32 if fp32_scale else torch.bfloat16)
    if m == 0:
        return q, s
    with torch.cuda.device(x2d.device):
        kern.rotq_h16[(kern.triton.cdiv(m, rows),)](
            x2d,
            x2d if v2d is None else v2d,
            h,
            h32,
            q,
            s,
            m,
            ng,
            nch,
            CG = cg,
            ROWS = rows,
            ONE_PASS = one,
            UNROLL = nch <= _H16_UNROLL_MAX,
            SWIGLU = v2d is not None,
            TF32X3 = bool(tf32x3),
            QMIN = qmin,
            QMAX = qmax,
            DIV = div,
            EPS = eps,
            FP32_SCALE = fp32_scale,
            num_warps = warps,
            num_stages = 1,
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


def rotq_tile_for(
    index: Any,
    k: int,
    group: int = 256,
) -> Optional[tuple]:
    """The probed tile this device quantizes a K-wide activation with: a per-K rule, else the arch default."""
    cfg = _ROTQ_DEVICE.get(index)
    if cfg is None:
        return None
    for k_lo, k_hi, tile in _ROTQ_DEVICE_K.get(index, ()):
        if k_lo <= k <= k_hi and k % group == 0 and _rotq_rows(k, group, tile) >= 1:
            return tile
    return cfg


def _h16_ok(
    x2d: Any,
    group: int,
    v2d: Any = None,
) -> bool:
    """``rotq_h16`` covers this call: its probe passed on the device (and the SwiGLU one for a gate / value pair)."""
    import torch

    index = x2d.device.index
    return (
        bool(_H16_DEVICE.get(index))
        and (v2d is None or bool(_SWIGLU_DEVICE.get(index)))
        and group == 256
        and x2d.dim() == 2
        and x2d.dtype == torch.bfloat16
        and x2d.is_cuda
        and x2d.shape[1] % 256 == 0
        and x2d.shape[1] >= 256
    )


def _rotq_run(
    x2d: Any,
    group: int,
    v2: bool,
    v2d: Any = None,
) -> tuple:
    """Op body: ``rotq_h16`` where it passed its probe, else the probed dense tile for this device and K, else the stock
    rotation + quant. ``v2d``: ``x2d`` is a SwiGLU gate and this its value half."""
    _ROTQ_CALLS[0] += 1
    kind = "v2" if v2 else "v1"
    x2d = x2d if x2d.is_contiguous() else x2d.contiguous()
    if v2d is not None:
        v2d = v2d if v2d.is_contiguous() else v2d.contiguous()
        _SWIGLU_CALLS[0] += 1
    index = x2d.device.index
    if _h16_ok(x2d, group, v2d):
        try:
            _H16_CALLS[0] += 1
            return _h16_launch(x2d, kind, v2d)
        except Exception:  # noqa: BLE001 - a failed launch keeps the dense kernel / stock math
            _H16_DEVICE[index] = False
    if v2d is not None:  # no dense SwiGLU kernel: the stock product, rotation and quant
        return rotquant_reference(x2d, group, kind, v2d)
    cfg = rotq_tile_for(index, x2d.shape[-1], group) if x2d.dim() == 2 else None
    if rotquant_supported(x2d, group, cfg):
        try:
            return _rotq_launch(x2d, group, kind, cfg)
        except Exception:  # noqa: BLE001 - a failed launch keeps the stock math
            if cfg != _ROTQ_DEVICE.get(index) and _ROTQ_DEVICE_K.get(index):
                _ROTQ_DEVICE_K[index] = ()  # drop the per-K tiles, keep the probed default
                return _rotq_run(x2d, group, v2)
            _ROTQ_DEVICE[index] = None
    return rotquant_reference(x2d, group, kind, v2d)


def _fake_quant(x: Any, v2: bool) -> tuple:
    import torch
    return (
        x.new_empty((x.shape[0], x.shape[1]), dtype = torch.int8),
        x.new_empty(
            (x.shape[0],),
            dtype = torch.float32 if _rotq_scale_fp32("v2" if v2 else "v1") else torch.bfloat16,
        ),
    )


@lru_cache(maxsize = 1)
def _swiglu_op() -> Any:
    """``unsloth_studio::convrot_swiglu_act_quant_int8(gate, value, group, v2)``: the rotated, quantized
    ``silu(gate) * value`` (opaque to Inductor), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _OP_NAMESPACE, None)
    if ns is not None and hasattr(ns, _SWIGLU_OP_NAME):
        return getattr(ns, _SWIGLU_OP_NAME)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_OP_NAMESPACE}::{_SWIGLU_OP_NAME}",
            mutates_args = (),
            schema = "(Tensor gate, Tensor value, int group, bool v2) -> (Tensor, Tensor)",
        )
        def _convrot_swiglu_act_quant_int8(gate, value, group, v2):
            return _rotq_run(gate, group, v2, value)

        @_convrot_swiglu_act_quant_int8.register_fake
        def _(gate, value, group, v2):
            return _fake_quant(gate, v2)
    except Exception:  # noqa: BLE001
        return None
    return getattr(getattr(torch.ops, _OP_NAMESPACE), _SWIGLU_OP_NAME)


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
            return _fake_quant(x, v2)
    except Exception:  # noqa: BLE001
        return None
    return getattr(getattr(torch.ops, _OP_NAMESPACE), _ROTQ_OP_NAME)


def rotquant_device_config(index: int) -> Optional[tuple]:
    """Once per device: arch gate, then bit-exact codes and scales vs the stock path (v1, v2, outliers, zero row, ragged
    M). None = stock. A passing device also publishes the op handle the traced forwards call, and probes ``rotq_h16``
    (tolerance, ``_h16_probe``) and the SwiGLU prologue on top of whichever rotation kernel will run."""
    global _ROTQ_HANDLE, _SWIGLU_HANDLE
    if index in _ROTQ_DEVICE:
        return _ROTQ_DEVICE[index]
    cfg = None
    rules: tuple = ()
    try:
        import torch
        if (
            torch.cuda.is_available()
            and not is_rocm_torch(torch)
            and _triton_ok()
            and _kernels() is not None
            and _rotq_op() is not None
        ):
            cap = torch.cuda.get_device_capability(index)
            want = rotquant_config(cap)
            for cand in (want, _ROTQ_FALLBACK) if want is not None else ():
                if _rotq_probe(index, cand):
                    cfg = cand
                    break
            if cfg is not None:
                ok: dict = {}
                for rule in rotq_k_tiles(cap):
                    if rule[2] not in ok:
                        ok[rule[2]] = rule[2] == cfg or _rotq_probe(index, rule[2])
                rules = tuple(rule for rule in rotq_k_tiles(cap) if ok[rule[2]])
    except Exception:  # noqa: BLE001
        cfg = None
    _ROTQ_DEVICE_K[index] = rules if cfg is not None else ()
    _ROTQ_DEVICE[index] = cfg
    h16 = sw = False
    if cfg is not None:
        try:
            h16 = _h16_probe(index) if rotquant_h16_enabled() else False
            if h16:
                _H16_DEVICE[index] = h16  # the SwiGLU probe launches in this mode
            if h16 and rotquant_swiglu_enabled() and _swiglu_op() is not None:
                sw = _swiglu_probe(index)
        except Exception:  # noqa: BLE001 - the dense kernel stays, no SwiGLU prologue
            h16 = sw = False
    elif _UNLISTED_DEVICE.get((index, unlisted_mode())):
        h16 = (
            _UNLISTED_PROBE.get(index) or False
        )  # no dense tile: keep the verdict unlisted_device_ok armed
    _H16_DEVICE[index] = h16
    _SWIGLU_DEVICE[index] = sw
    if cfg is not None:
        _ROTQ_HANDLE = _rotq_op()
        if sw:
            _SWIGLU_HANDLE = _swiglu_op()
    return cfg


# (M, K): ragged M, one-group rows, 21 groups (a ragged row count per tile), 56 and 128 groups (two rows / one row).
_ROTQ_PROBE_SHAPES = ((257, 256 * 3), (33, 256 * 21), (130, 256 * 56), (17, 256 * 128))
# rotq_h16: the dense shapes and long rows (two-pass) plus 16 and 4 groups (K 4096 / 1024: the one-pass branch)
_H16_PROBE_SHAPES = _ROTQ_PROBE_SHAPES + (
    (19, 256 * 48),
    (5, 256 * 160),
    (65, 256 * 16),
    (9, 256 * 4),
)
_SWIGLU_PROBE_SHAPES = ((257, 256 * 3), (130, 256 * 48), (33, 256 * 21), (65, 256 * 16))


def _rotq_probe(index: int, cfg: tuple) -> bool:
    import torch

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(2)
    try:
        for m, k in _ROTQ_PROBE_SHAPES:
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
        from .diffusion_cuda_graph import synchronize_outside_capture

        synchronize_outside_capture(dev)
        return True
    except Exception:  # noqa: BLE001
        return False


def _probe_input(m: int, k: int, g: Any) -> Any:
    import torch

    x = torch.randn(m, k, generator = g) * (torch.rand(1, k, generator = g) * 4)
    x[:, :3] *= 60
    x[1] = 0
    x[2, 11] = 3e4
    return x.to(torch.bfloat16)


def codes_close(q: Any, s: Any, rq: Any, rs: Any) -> bool:
    """``rotq_h16`` vs stock codes: every code within two steps (v1's bf16 ``z / s`` has step 0.5 near 127); <= 1 in
    10^4 codes (and 8) off on scale-matched rows; <= 1% of rows (and 1) one bf16 step off in scale."""
    import torch

    if q.shape != rq.shape or q.dtype != rq.dtype or s.shape != rs.shape or s.dtype != rs.dtype:
        return False
    if not (bool(torch.isfinite(s).all()) and bool(torch.isfinite(rs).all())):
        return False  # NaN compares false against every tolerance below
    dq = (q.to(torch.int16) - rq.to(torch.int16)).abs()
    if q.numel() and int(dq.max()) > 2:
        return False
    same = s == rs
    rows_off = int((~same).sum())
    if rows_off > max(1, s.numel() // 100):
        return False
    if rows_off:
        sf, rf = s.float(), rs.float()
        if bool(((sf - rf).abs() > rf.abs() * 2.0**-7)[~same].any()):
            return False
    off = int(dq[same].ne(0).sum())
    return off <= max(8, int(same.sum()) * q.shape[-1] // 10000)


def _h16_probe(index: int) -> Any:
    """``rotq_h16`` vs the stock rotation + quant on the dense probe's shapes and a long row (``codes_close``): the
    second-stage mode that passed ("tf32x3" first on ``_H16_TF32X3_ARCHS``, else "ieee"), or False."""
    import torch

    dev = torch.device("cuda", index)
    cap = tuple(torch.cuda.get_device_capability(index))
    modes = ("tf32x3", "ieee") if cap in _H16_TF32X3_ARCHS else ("ieee",)
    for mode in modes:
        g = torch.Generator(device = "cpu").manual_seed(3)
        try:
            ok = True
            for m, k in _H16_PROBE_SHAPES:
                x = _probe_input(m, k, g).to(dev)
                for kind in ("v1", "v2"):
                    q, s = _h16_launch(x, kind, tf32x3 = mode == "tf32x3")
                    rq, rs = rotquant_reference(x, 256, kind)
                    if not codes_close(q, s, rq, rs):
                        ok = False
                        break
                if not ok:
                    break
            from .diffusion_cuda_graph import synchronize_outside_capture

            synchronize_outside_capture(dev)
            if ok:
                return mode
        except Exception:  # noqa: BLE001
            continue
    return False


def _swiglu_probe(index: int) -> bool:
    """``rotq_h16``'s SwiGLU prologue vs ``silu(gate) * value`` (ATen, eager) then the stock rotation + quant:
    ``codes_close``. Gates span SiLU's saturated tails, zero and tiny magnitudes."""
    import torch

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(4)
    try:
        for m, k in _SWIGLU_PROBE_SHAPES:
            gate = (_probe_input(m, k, g).float() * 0.05).to(torch.bfloat16).to(dev)
            gate[0, :7] = torch.tensor([-90.0, -20.0, -1e-3, 0.0, 1e-3, 20.0, 90.0])
            value = _probe_input(m, k, g).to(dev)
            for kind in ("v1", "v2"):
                rq, rs = rotquant_reference(gate, 256, kind, value)
                q, s = _h16_launch(gate, kind, value)
                if not codes_close(q, s, rq, rs):
                    return False
        from .diffusion_cuda_graph import synchronize_outside_capture

        synchronize_outside_capture(dev)
        return True
    except Exception:  # noqa: BLE001
        return False


def rotquant_call_count() -> int:
    return _ROTQ_CALLS[0]


def h16_call_count() -> int:
    return _H16_CALLS[0]


def swiglu_call_count() -> int:
    return _SWIGLU_CALLS[0]


def convrot_act_quant(x2d: Any, group: int, v2: bool) -> Optional[tuple]:
    """``(int8 [M, K], scale [M])`` of ``x2d @ blockdiag(H_group)`` quantized like torchao (v2 = ``Int8Tensor``, else
    the v1 per-token quant), from the fused kernel; None when this device / shape / group is not covered, so the caller
    keeps its own rotation + quant. Safe inside a traced forward: no host sync, the probe ran at install time."""
    import torch

    if _ROTQ_HANDLE is None or x2d.dim() != 2 or x2d.dtype != torch.bfloat16 or not x2d.is_cuda:
        return None
    if not (
        _h16_ok(x2d, group)
        or rotquant_supported(x2d, group, rotq_tile_for(x2d.device.index, x2d.shape[-1], group))
    ):
        return None
    return _ROTQ_HANDLE(x2d, group, v2)


def convrot_swiglu_act_quant(g2d: Any, v2d: Any, group: int, v2: bool) -> Optional[tuple]:
    """``convrot_act_quant`` of ``silu(g2d) * v2d`` without the product ever reaching HBM: the down projection of a
    rotated SwiGLU MLP. None when this device / shape / group is not covered (the caller keeps the stock product and
    its own rotation + quant). Safe inside a traced forward: no host sync, the probe ran at install time."""
    import torch

    if (
        _SWIGLU_HANDLE is None
        or g2d.dim() != 2
        or g2d.shape != v2d.shape
        or g2d.dtype != torch.bfloat16
        or v2d.dtype != torch.bfloat16
        or not g2d.is_cuda
        or g2d.device != v2d.device
        or not _SWIGLU_DEVICE.get(g2d.device.index)
    ):
        return None
    if not _h16_ok(g2d, group, v2d):
        return None
    return _SWIGLU_HANDLE(g2d, v2d, group, v2)


# Unlisted archs: main runs the rotation as a dense bf16 GEMM (1/8 the int8 rate on GA102) + a separate act quant.
# rotq_h16 must pass the probe and be no slower than both, timed from CUDA-graph replays (the compiled denoiser replays
# kernels; host launch cost must not count). ``..._UNLISTED=1`` skips the latency check, not the probe.
INT8_ROTQUANT_UNLISTED_ENV = "UNSLOTH_DIFFUSION_INT8_ROTQUANT_UNLISTED"
_UNLISTED_DEVICE: dict = {}
_UNLISTED_TIMING: dict = {}
_UNLISTED_PROBE: dict = {}  # index -> _h16_probe result (mode independent)
_UNLISTED_LATENCY_SHAPES = ((2048, 4096), (1024, 12288))


def unlisted_mode() -> str:
    if rotquant_mode() == "off" or not rotquant_h16_enabled():
        return "off"
    return _env_mode(INT8_ROTQUANT_UNLISTED_ENV)


def unlisted_wanted(capability: Optional[tuple], mode: Optional[str] = None) -> bool:
    """True on a CUDA sm80+ arch with no dense ``rotq_i8`` tile (the listed archs keep their probed path)."""
    mode = unlisted_mode() if mode is None else mode
    if mode == "off" or capability is None:
        return False
    cap = (int(capability[0]), int(capability[1]))
    return cap >= (8, 0) and rotquant_config(cap) is None


def _unlisted_latency(
    index: int,
    tf32x3: bool,
    stock_quant: Any = None,
) -> dict:
    """{"h16": us, "stock": us} over ``_UNLISTED_LATENCY_SHAPES``: each arm captured in a CUDA graph (two passes)
    and replayed, best of seven interleaved rounds. Stock = the rotation GEMM, then ``stock_quant`` of its output when
    given, else the GEMM alone (a lower bound)."""
    import torch
    from .diffusion_convrot import build_convrot_hadamard, rotate_convrot_activation
    from .diffusion_cuda_graph import CAPTURE_MODE, _capturing, synchronize_outside_capture

    dev = torch.device("cuda", index)
    # An explicit CPU generator: a default-generator draw raises while another thread records (up to torch 2.11).
    gen = torch.Generator().manual_seed(0)
    xs = [
        torch.randn(m, k, generator = gen).to(dev, torch.bfloat16)
        for m, k in _UNLISTED_LATENCY_SHAPES
    ]
    h = build_convrot_hadamard(256, device = dev, dtype = torch.bfloat16)

    def stock(x):
        r = rotate_convrot_activation(x, h, 256)
        return r if stock_quant is None else stock_quant(r)

    fns = {"h16": lambda x: _h16_launch(x, "v2", tf32x3 = tf32x3), "stock": stock}
    reps = 2
    graphs = {}
    with torch.cuda.device(dev):
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for fn in fns.values():
                for x in xs:
                    fn(x)  # compile / autotune outside the capture
        torch.cuda.current_stream(dev).wait_stream(side)
        synchronize_outside_capture(dev)
        for name, fn in fns.items():
            graph = torch.cuda.CUDAGraph()
            # Serialised with Studio's captures and recorded thread_local, like every other capture.
            with _capturing(), torch.cuda.graph(graph, capture_error_mode = CAPTURE_MODE):
                for _ in range(reps):
                    for x in xs:
                        fn(x)
            graphs[name] = graph
        for graph in graphs.values():
            graph.replay()
        synchronize_outside_capture(dev)
        start, end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        best = {name: float("inf") for name in graphs}
        for _ in range(7):
            for name, graph in graphs.items():
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                best[name] = min(best[name], start.elapsed_time(end) * 1000.0 / reps)
        del graphs
    return best


def unlisted_device_ok(index: int, stock_quant: Any = None) -> bool:
    """Once per (device, mode), outside any trace / capture: arch gate, ``_h16_probe`` (once per device), latency
    check (auto only). Keyed by mode so switching ``..._UNLISTED`` between loads takes effect without a restart."""
    global _ROTQ_HANDLE
    mode = unlisted_mode()
    if mode == "off":  # read per load: the switches act without a restart
        return False
    key = (index, mode)
    if key not in _UNLISTED_DEVICE:
        ok = False
        try:
            import torch
            if (
                torch.cuda.is_available()
                and not is_rocm_torch(torch)
                and _triton_ok()
                and unlisted_wanted(torch.cuda.get_device_capability(index), mode)
                and _kernels() is not None
                and _rotq_op() is not None
            ):
                if index not in _UNLISTED_PROBE:
                    _UNLISTED_PROBE[index] = _h16_probe(index)
                h16 = _UNLISTED_PROBE[index]
                if h16:
                    ok = True
                    if mode != "force":
                        timing = _unlisted_latency(index, h16 == "tf32x3", stock_quant)
                        _UNLISTED_TIMING[index] = {k: round(v, 2) for k, v in timing.items()}
                        ok = timing["h16"] <= timing["stock"]
        except Exception:  # noqa: BLE001 - main's path stays
            ok = False
        _UNLISTED_DEVICE[key] = ok
    ok = _UNLISTED_DEVICE[key]
    h16 = _UNLISTED_PROBE.get(index)
    if ok:
        _H16_DEVICE[index] = h16
        _ROTQ_HANDLE = _rotq_op()
    elif h16:  # probed unlisted card the latency check now rejects: drop a verdict a forced load left behind
        _H16_DEVICE.pop(index, None)
    return ok

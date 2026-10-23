# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ConvRot rotation and the int8 per-row activation quant as ONE kernel (``rotq_i8``), for every int8 ConvRot Linear.

The stock path is two kernels: ``ConvRotLinear.forward``'s block-Hadamard rotation as a bf16 GEMM (writes the rotated
activation to HBM), then torchao's per-row act quant (reads it back, writes int8 + one scale per row). ``rotq_i8`` holds
whole activation rows of the ``[M * K / G, G]`` group view per tile, runs the rotation GEMM in the stock K order with
the stock bf16 rounding, closes the per-row amax in the epilogue and writes the int8 codes and the row scale: one read
of x, no rotated intermediate.

NUMERICS: codes and scales are bit-identical to the stock rotation + torchao quant (v1
``_int8_symm_per_token_reduced_range_quant``; v2 ``Int8Tensor.from_hp(PerRow)``: bf16 scale, reciprocal and product on
torchao <= 0.17, fp32 on >= 0.18). Every device is probed against that path before use (outliers, a zero row, ragged
M, every K the tile admits); a miss keeps the stock path on that device.

One home for the kernel: ``diffusion_int8_gemm`` uses it for every ConvRot Linear it swaps (fused-dequant GEMM on
sm80 / sm89 / sm120, cuBLAS ``_int_mm`` + torchao's epilogue on the other measured archs), and ``convrot_act_quant``
gives any other int8 consumer of a rotated activation the same codes.

Per-arch gate (``_ROTQ_CONFIG``): measured archs only; ``UNSLOTH_DIFFUSION_INT8_ROTQUANT=1`` (or
``UNSLOTH_DIFFUSION_INT8_GEMM=1``) also enables it on an unmeasured sm80+ part, still probe-gated. Kill switch
``UNSLOTH_DIFFUSION_INT8_ROTQUANT=0``; ``UNSLOTH_DIFFUSION_INT8_GEMM=0`` turns off the whole int8 Linear swap.

Every other sm80+ arch (sm86, sm87, sm90, sm100, sm103, sm121, ...) gets ``rotq_fht_i8`` instead (bottom of this file):
the same fused rotation + quant without a per-arch tile, accuracy-probed and latency-gated per device.
"""

from __future__ import annotations

import os
import types
from functools import lru_cache
from typing import Any, Optional
from .rocm_bf16 import is_rocm_torch

INT8_ROTQUANT_ENV = "UNSLOTH_DIFFUSION_INT8_ROTQUANT"
_OP_NAMESPACE = "unsloth_studio"
_ROTQ_OP_NAME = "convrot_act_quant_int8"

# (BLOCK_M group rows, BLOCK_K, num_warps, num_stages) per (major, minor); measured end to end. Absent = stock.
_ROTQ_CONFIG = {
    (8, 0): (128, 32, 8, 3),  # A100
    (8, 9): (128, 32, 8, 3),  # L4
    (12, 0): (128, 32, 8, 4),  # RTX PRO 6000
}
_ROTQ_FALLBACK = (128, 32, 8, 3)
_ROTQ_GROUPS = (256,)
# Per-K tiles: (major, minor) -> ((K_lo, K_hi, tile), ...), first match wins, bounds inclusive; a tile whose BLOCK_M
# cannot hold one whole row at that K is skipped. K 10240 stays on the default (a 64-row tile wastes 38% of it there).
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
# rotq_fht_i8 (groups per chunk, warps) by variant: True = tensor-core Kronecker (sm80+), False = butterflies.
_FHT_TILE = {True: (16, 4), False: (16, 2)}
# Set once a device passed the probe; read by traced forwards (dynamo must not trace into the lru_cache'd registration).
_ROTQ_HANDLE: Any = None


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

    @triton.jit
    def _h4_last(v, N: tl.constexpr):
        # 0.5 * H4 along the last axis of ``v`` [N, 4]: row d of H4 is the all-ones row with operand 3 - d negated
        a, b = tl.split(tl.reshape(v, (N, 2, 2)))
        x0, x2 = tl.split(a)
        x1, x3 = tl.split(b)
        h = 0.5 * (((x0 + x1) + x2) + x3)
        return tl.reshape(tl.join(tl.join(h - x3, h - x1), tl.join(h - x2, h - x0)), (N, 4))

    @triton.jit
    def _fht256(v, CH: tl.constexpr):
        # x @ H256 for each of the CH rows of ``v`` [CH, 256], H256 = kron(H4, H4, H4, H4) / 16: four radix-4 stages,
        # each on the last axis, then a cyclic permute brings the next base-4 digit last (four permutes: identity).
        for _s in tl.static_range(4):
            v = tl.reshape(_h4_last(tl.reshape(v, (CH * 64, 4)), CH * 64), (CH, 4, 4, 4, 4))
            v = tl.reshape(tl.permute(v, (0, 4, 1, 2, 3)), (CH, 256))
        return v

    @triton.jit
    def _kron16(v, h_ptr, CH: tl.constexpr):
        # x @ H256 = H16 @ X @ H16 per group (X = the group as [16, 16], H16 symmetric, H256 = kron(H16, H16)), as
        # batched 16x16 tensor-core dots. Every product is exact (H16 = +-1/4); the second factor's operand is the fp32
        # first product split into three bf16 parts (24 significand bits), so both sums accumulate in fp32.
        o16 = tl.arange(0, 16)
        h = tl.load(h_ptr + o16[:, None] * 16 + o16[None, :])
        hb = tl.broadcast_to(h[None, :, :], (CH, 16, 16))
        y = tl.dot(tl.reshape(v, (CH, 16, 16)), hb, out_dtype = tl.float32)
        a = y.to(tl.bfloat16)
        r = y - a.to(tl.float32)
        b = r.to(tl.bfloat16)
        c = (r - b.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(hb, c, out_dtype = tl.float32)
        acc = tl.dot(hb, b, acc, out_dtype = tl.float32)
        acc = tl.dot(hb, a, acc, out_dtype = tl.float32)
        return tl.reshape(acc, (CH, 256))

    @triton.jit
    def _rot_group(x_ptr, h_ptr, offs_g, offs_e, live, CH: tl.constexpr, KRON: tl.constexpr):
        ptrs = x_ptr + offs_g[:, None] * 256 + offs_e[None, :]
        v = tl.load(ptrs, mask = live[:, None], other = 0.0)
        if KRON:
            return _rbf16(_kron16(v, h_ptr, CH))
        return _rbf16(_fht256(v.to(tl.float32), CH))

    @triton.jit
    def _quant_store(
        z,
        inv,
        q_ptr,
        offs_g,
        offs_e,
        live,
        QMIN: tl.constexpr,
        QMAX: tl.constexpr,
        FP32_SCALE: tl.constexpr,
    ):
        p = z * inv
        if not FP32_SCALE:
            p = _rbf16(p)
        p = tl.minimum(tl.maximum(libdevice.rint(p), QMIN), QMAX)
        tl.store(q_ptr + offs_g[:, None] * 256 + offs_e[None, :], p.to(tl.int8), mask = live[:, None])

    @triton.jit
    def rotq_fht_i8(
        x_ptr,
        h_ptr,
        q_ptr,
        s_ptr,
        NG,
        CH: tl.constexpr,
        ONE_PASS: tl.constexpr,
        KRON: tl.constexpr,
        QMIN: tl.constexpr,
        QMAX: tl.constexpr,
        DIV: tl.constexpr,
        EPS: tl.constexpr,
        FP32_SCALE: tl.constexpr,
    ):
        # one program = one activation row of NG 256-wide groups, CH groups per chunk. KRON: two 16-point factors on
        # bf16 tensor cores (sm80+); else four radix-4 butterfly stages in fp32 (any arch). Two chunks or more: amax
        # pass, then a recompute + quant pass.
        row = tl.program_id(0).to(tl.int64)
        offs_g = tl.arange(0, CH)
        offs_e = tl.arange(0, 256)
        base = x_ptr + row * NG * 256
        qbase = q_ptr + row * NG * 256
        if ONE_PASS:
            z = _rot_group(base, h_ptr, offs_g, offs_e, offs_g < NG, CH, KRON)
            a = tl.max(tl.max(tl.abs(z), axis = 1), axis = 0)
        else:
            amax = tl.zeros((CH,), dtype = tl.float32)
            for c in range(0, NG, CH):
                zc = _rot_group(base + c * 256, h_ptr, offs_g, offs_e, (c + offs_g) < NG, CH, KRON)
                amax = tl.maximum(amax, tl.max(tl.abs(zc), axis = 1))
            a = tl.max(amax, axis = 0)
        s = _rbf16(tl.maximum(_rbf16(libdevice.div_rn(a, DIV)), EPS))
        inv = libdevice.div_rn(s * 0.0 + 1.0, s)
        if not FP32_SCALE:
            inv = _rbf16(inv)
        if ONE_PASS:
            _quant_store(z, inv, qbase, offs_g, offs_e, offs_g < NG, QMIN, QMAX, FP32_SCALE)
        else:
            for c in range(0, NG, CH):
                live = (c + offs_g) < NG
                zc = _rot_group(base + c * 256, h_ptr, offs_g, offs_e, live, CH, KRON)
                _quant_store(zc, inv, qbase + c * 256, offs_g, offs_e, live, QMIN, QMAX, FP32_SCALE)
        tl.store(s_ptr + row, s.to(s_ptr.dtype.element_ty))

    return types.SimpleNamespace(rotq_i8 = rotq_i8, rotq_fht_i8 = rotq_fht_i8, triton = triton)


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


def rotquant_reference(x2d: Any, group: int, kind: str) -> tuple:
    """The stock path: ConvRotLinear.forward's rotation (bf16 GEMM), then torchao's activation quant. (int8, scale)."""
    import torch
    from .diffusion_convrot import build_convrot_hadamard, rotate_convrot_activation

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


def _fht_launch(
    x2d: Any,
    kind: str,
    ch: Optional[int] = None,
    warps: Optional[int] = None,
    kron: Optional[bool] = None,
) -> tuple:
    """``rotq_fht_i8`` on a contiguous bf16 ``[M, K]``, K a multiple of 256: (int8 [M, K], scale [M])."""
    import torch
    from .diffusion_convrot import build_convrot_hadamard

    kern = _kernels()
    m, k = x2d.shape
    ng = k // 256
    if kron is None:
        kron = _FHT_DEVICE.get(x2d.device.index)
        if kron is None:
            kron = torch.cuda.get_device_capability(x2d.device) >= (8, 0)
    ch_d, warps_d = _FHT_TILE[bool(kron)]
    ch = ch_d if ch is None else ch
    warps = warps_d if warps is None else warps
    ch = max(16 if kron else 1, min(ch, kern.triton.next_power_of_2(ng)))
    qmin, qmax, div, eps = _ROTQ_QPARAMS[kind]
    fp32_scale = _rotq_scale_fp32(kind)
    h16 = build_convrot_hadamard(16, device = x2d.device, dtype = torch.bfloat16)
    q = torch.empty((m, k), device = x2d.device, dtype = torch.int8)
    s = torch.empty((m,), device = x2d.device, dtype = torch.float32 if fp32_scale else torch.bfloat16)
    if m == 0:
        return q, s
    with torch.cuda.device(x2d.device):
        kern.rotq_fht_i8[(m,)](
            x2d,
            h16,
            q,
            s,
            ng,
            CH = ch,
            ONE_PASS = ng <= ch,
            KRON = bool(kron),
            QMIN = qmin,
            QMAX = qmax,
            DIV = div,
            EPS = eps,
            FP32_SCALE = fp32_scale,
            num_warps = warps,
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


def _rotq_run(x2d: Any, group: int, v2: bool) -> tuple:
    """Op body: the probed tile for this device and K, the stock rotation + quant if anything does not fit."""
    _ROTQ_CALLS[0] += 1
    kind = "v2" if v2 else "v1"
    x2d = x2d if x2d.is_contiguous() else x2d.contiguous()
    index = x2d.device.index
    cfg = rotq_tile_for(index, x2d.shape[-1], group) if x2d.dim() == 2 else None
    if rotquant_supported(x2d, group, cfg):
        try:
            return _rotq_launch(x2d, group, kind, cfg)
        except Exception:  # noqa: BLE001 - a failed launch keeps the stock math
            if cfg != _ROTQ_DEVICE.get(index) and _ROTQ_DEVICE_K.get(index):
                _ROTQ_DEVICE_K[index] = ()  # drop the per-K tiles, keep the probed default
                return _rotq_run(x2d, group, v2)
            _ROTQ_DEVICE[index] = None
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
    M). None = stock. A passing device also publishes the op handle the traced forwards call."""
    global _ROTQ_HANDLE
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
    if cfg is not None:
        _ROTQ_HANDLE = _rotq_op()
    return cfg


# (M, K): ragged M, one-group rows, 21 groups (a ragged row count per tile), 56 and 128 groups (two rows / one row).
_ROTQ_PROBE_SHAPES = ((257, 256 * 3), (33, 256 * 21), (130, 256 * 56), (17, 256 * 128))


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
        torch.cuda.synchronize(dev)
        return True
    except Exception:  # noqa: BLE001
        return False


def rotquant_call_count() -> int:
    return _ROTQ_CALLS[0]


def convrot_act_quant(x2d: Any, group: int, v2: bool) -> Optional[tuple]:
    """``(int8 [M, K], scale [M])`` of ``x2d @ blockdiag(H_group)`` quantized like torchao (v2 = ``Int8Tensor``, else
    the v1 per-token quant), from the fused kernel; None when this device / shape / group is not covered, so the caller
    keeps its own rotation + quant. Safe inside a traced forward: no host sync, the probe ran at install time."""
    import torch

    if (
        _ROTQ_HANDLE is None
        or x2d.dim() != 2
        or x2d.dtype != torch.bfloat16
        or not x2d.is_cuda
        or not rotquant_supported(x2d, group, rotq_tile_for(x2d.device.index, x2d.shape[-1], group))
    ):
        return None
    return _ROTQ_HANDLE(x2d, group, v2)


# ---- rotq_fht_i8: the fused rotation + quant for every arch the tiled kernel does not cover ----
#
# ``rotq_i8`` runs the rotation as a dense 256x256 bf16 GEMM (512 flops per element) and is tuned and probed bit-exact
# per arch; elsewhere (sm75, sm86, sm87, sm90, sm100, sm121, ...) the stock path ran that same dense GEMM through
# cuBLAS and wrote the rotated activation to HBM before the act quant read it back. H256 = kron(H16, H16), so
# ``rotq_fht_i8`` rotates a group as H16 @ X @ H16 (X the group as [16, 16]) on bf16 tensor cores (sm80+: 128 flops
# per element, exact products, fp32 sums), or as four radix-4 butterfly stages in fp32 (any arch, sm75's only
# option), and quantizes in the same pass: no rotated intermediate, no tile table. The sum order differs from the
# GEMM's, so a rotated value can land on the other bf16 neighbour of a tie and move its code by one; measured on the
# real Qwen-Image-2.1 shapes: 0 to 2 codes in 18M, scales equal.
#
# Gate, per device, at install: an arch with no ``_ROTQ_CONFIG`` entry, an accuracy probe of each variant against
# the stock rotation + torchao quant (codes within one, scales within one bf16 step, mismatches rare), then a latency
# check: the faster passing variant runs only if it is not slower than the two kernels it replaces, timed on the same
# card (the rotation GEMM, then the caller's act quant kernel; the GEMM alone when the caller has none). The variant
# is internal to the op, so the traced graph only depends on on / off.
# ``UNSLOTH_DIFFUSION_INT8_ROTQUANT_FHT=0`` keeps the stock path; ``=1`` skips the latency check (accuracy still probed);
# ``UNSLOTH_DIFFUSION_INT8_ROTQUANT=0`` / ``UNSLOTH_DIFFUSION_INT8_GEMM=0`` turn it off too.
INT8_ROTQUANT_FHT_ENV = "UNSLOTH_DIFFUSION_INT8_ROTQUANT_FHT"
_FHT_OP_NAME = "convrot_fht_act_quant_int8"
_FHT_MIN_CAP = (7, 5)
# device index -> the variant that serves it (True = tensor-core Kronecker, False = butterflies) or None (stock);
# filled once per device by ``fht_device_ok``.
_FHT_DEVICE: dict = {}
# device index -> {variant: kernel us, "gemm": rotation GEMM us} of the latency check, for the log.
_FHT_TIMING: dict = {}
_FHT_CALLS = [0]
# Set once a device passed; read by traced forwards.
_FHT_HANDLE: Any = None


def fht_mode() -> str:
    """'off', 'force' or 'auto': off under its own ``=0`` or when the rotation lever / int8 swap is off."""
    if rotquant_mode() == "off":
        return "off"
    return _env_mode(INT8_ROTQUANT_FHT_ENV)


def fht_wanted(capability: Optional[tuple], mode: Optional[str] = None) -> bool:
    """True where ``rotq_fht_i8`` replaces the stock rotation: CUDA sm75+ that the tiled ``rotq_i8`` does not cover
    (those archs keep their probed bit-exact kernel)."""
    mode = fht_mode() if mode is None else mode
    if mode == "off" or capability is None:
        return False
    cap = (int(capability[0]), int(capability[1]))
    return cap >= _FHT_MIN_CAP and rotquant_config(cap) is None


def _fht_run(x2d: Any, v2: bool) -> tuple:
    """Op body: the FHT kernel, or the stock rotation + quant for a shape it does not take."""
    import torch

    _FHT_CALLS[0] += 1
    kind = "v2" if v2 else "v1"
    x2d = x2d if x2d.is_contiguous() else x2d.contiguous()
    if (
        x2d.dim() == 2
        and x2d.dtype == torch.bfloat16
        and x2d.is_cuda
        and x2d.shape[1] % 256 == 0
        and x2d.shape[1] >= 256
        and _FHT_DEVICE.get(x2d.device.index) is not None
    ):
        try:
            return _fht_launch(x2d, kind)
        except Exception:  # noqa: BLE001 - a failed launch keeps the stock math
            _FHT_DEVICE[x2d.device.index] = None
    return rotquant_reference(x2d, 256, kind)


@lru_cache(maxsize = 1)
def _fht_op() -> Any:
    """``unsloth_studio::convrot_fht_act_quant_int8`` (opaque to Inductor, allocation only), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _OP_NAMESPACE, None)
    if ns is not None and hasattr(ns, _FHT_OP_NAME):
        return getattr(ns, _FHT_OP_NAME)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_OP_NAMESPACE}::{_FHT_OP_NAME}",
            mutates_args = (),
            schema = "(Tensor x, bool v2) -> (Tensor, Tensor)",
        )
        def _convrot_fht_act_quant_int8(x, v2):
            return _fht_run(x, v2)

        @_convrot_fht_act_quant_int8.register_fake
        def _(x, v2):
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
    return getattr(getattr(torch.ops, _OP_NAMESPACE), _FHT_OP_NAME)


# (M, K): ragged M, one group, 21 groups (a ragged last chunk), 48 groups (three chunks), 128 groups.
_FHT_PROBE_SHAPES = ((257, 256), (33, 256 * 21), (130, 256 * 48), (17, 256 * 128), (64, 4096))
# Allowed fraction of codes that differ (by one) from the stock path: the fp32 sum-order ties. Measured ~1e-6.
_FHT_MAX_CODE_MISMATCH = 1e-3


def _fht_close(q: Any, s: Any, rq: Any, rs: Any) -> bool:
    import torch

    if q.dtype != rq.dtype or s.dtype != rs.dtype or q.shape != rq.shape or s.shape != rs.shape:
        return False
    dq = (q.to(torch.int16) - rq.to(torch.int16)).abs()
    if int(dq.max()) > 1 or float((dq != 0).float().mean()) > _FHT_MAX_CODE_MISMATCH:
        return False
    # scales are bf16 values (fp32-typed on torchao >= 0.18): one bf16 step apart at most
    rel = ((s.float() - rs.float()).abs() / rs.float().abs().clamp_min(1e-30)).max()
    return bool(rel <= 2.0**-7)


def _fht_probe(index: int, kron: bool) -> bool:
    import torch

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(3)
    try:
        for m, k in _FHT_PROBE_SHAPES:
            x = torch.randn(m, k, generator = g) * (torch.rand(1, k, generator = g) * 4)
            x[:, :3] *= 60
            x[1] = 0
            x[2, 11] = 3e4
            x = x.to(torch.bfloat16).to(dev)
            for kind in ("v1", "v2"):
                q, s = _fht_launch(x, kind, kron = kron)
                rq, rs = rotquant_reference(x, 256, kind)
                if not _fht_close(q, s, rq, rs):
                    return False
        torch.cuda.synchronize(dev)
        return True
    except Exception:  # noqa: BLE001
        return False


_FHT_LATENCY_SHAPES = ((2048, 4096), (1024, 12288))


def _fht_latency(
    index: int,
    variants: tuple,
    stock_quant: Any = None,
) -> dict:
    """{variant: kernel us, "gemm": stock us} over ``_FHT_LATENCY_SHAPES``. Each arm is captured in a CUDA graph and
    replayed (the compiled denoiser replays its kernels the same way, so host launch cost must not count), best of
    five interleaved rounds. Stock = the rotation GEMM, then ``stock_quant`` of its output when given (the act quant
    kernel the caller runs after it), else the GEMM alone (a lower bound)."""
    import torch
    from .diffusion_convrot import build_convrot_hadamard, rotate_convrot_activation

    dev = torch.device("cuda", index)
    # one chunk per row (K 4096) and the multi-chunk recompute path (K 12288)
    xs = [torch.randn(m, k, device = dev).to(torch.bfloat16) for m, k in _FHT_LATENCY_SHAPES]
    h = build_convrot_hadamard(256, device = dev, dtype = torch.bfloat16)

    def stock(x):
        r = rotate_convrot_activation(x, h, 256)
        return r if stock_quant is None else stock_quant(r)

    fns = {v: (lambda x, v = v: _fht_launch(x, "v2", kron = v)) for v in variants}
    fns["gemm"] = stock
    reps = 4
    graphs = {}
    with torch.cuda.device(dev):
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for fn in fns.values():
                for x in xs:
                    fn(x)  # compile / autotune outside the capture
        torch.cuda.current_stream(dev).wait_stream(side)
        torch.cuda.synchronize(dev)
        for name, fn in fns.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(reps):
                    for x in xs:
                        fn(x)
            graphs[name] = graph
        start, end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        best = {name: float("inf") for name in graphs}
        for name, graph in graphs.items():
            graph.replay()
        torch.cuda.synchronize(dev)
        for _ in range(5):
            for name, graph in graphs.items():
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                best[name] = min(
                    best[name], start.elapsed_time(end) * 1000.0 / reps
                )  # us per pass over the shapes
        del graphs
    return best


def fht_device_ok(index: int, stock_quant: Any = None) -> bool:
    """Once per device: arch gate, accuracy probe of each variant, latency check (the fastest passing variant must
    not be slower than the stock rotation GEMM + ``stock_quant``, or the GEMM alone; skipped under ``=1``). Publishes
    the op handle on success. Call outside any trace / graph capture."""
    global _FHT_HANDLE
    if (
        fht_mode() == "off"
    ):  # read per load: the switches act without a restart, the cached probe stays
        return False
    if index in _FHT_DEVICE:
        return _FHT_DEVICE[index] is not None
    choice = None
    try:
        import torch
        if (
            torch.cuda.is_available()
            and not is_rocm_torch(torch)
            and _triton_ok()
            and fht_wanted(torch.cuda.get_device_capability(index))
            and _kernels() is not None
            and _fht_op() is not None
        ):
            # Kronecker needs bf16 tensor cores (sm80+); the butterflies run anywhere
            tried = (True, False) if torch.cuda.get_device_capability(index) >= (8, 0) else (False,)
            variants = tuple(v for v in tried if _fht_probe(index, v))
            if variants and fht_mode() == "force":
                choice = variants[0]
            elif variants:
                timing = _fht_latency(index, variants, stock_quant)
                _FHT_TIMING[index] = {str(k): round(v, 2) for k, v in timing.items()}
                fastest = min(variants, key = lambda v: timing[v])
                if timing[fastest] <= timing["gemm"]:
                    choice = fastest
    except Exception:  # noqa: BLE001
        choice = None
    _FHT_DEVICE[index] = choice
    if choice is not None:
        _FHT_HANDLE = _fht_op()
    return choice is not None


def fht_call_count() -> int:
    return _FHT_CALLS[0]

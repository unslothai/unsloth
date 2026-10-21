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
"""

from __future__ import annotations

import os
import types
from functools import lru_cache
from typing import Any, Optional

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

    return types.SimpleNamespace(rotq_i8 = rotq_i8, triton = triton)


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
            and not getattr(torch.version, "hip", None)
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

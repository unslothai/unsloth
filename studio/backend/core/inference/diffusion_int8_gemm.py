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
Tiles: one default per arch plus a per-shape table (``_SHAPE_TILES``) for the shapes where another tile measured
faster; every tile is probed bit-exact per device, and a tile that fails falls back to the arch default. All tiles
accumulate in int32, so the output does not depend on the tile.

Kill switch: ``UNSLOTH_DIFFUSION_INT8_GEMM=0``; ``=1`` also enables it on an unmeasured arch (still probe-gated).
``UNSLOTH_DIFFUSION_INT8_GEMM_TILES=0`` keeps the arch default tile for every shape.

LATENCY GATE: a static tile table goes stale with the toolchain (A100 on Triton 3.8: the shipped 128x128x128 tile runs
the DiT shapes at ~59 TOPS vs ~430 on Triton 3.6 / 3.7, while cuBLAS ``_int_mm`` holds ~420). At install, every swapped
(N, K) is timed once per M bucket on this device: the probed tile, the per-shape tile and a few arch candidates against
cuBLAS ``_int_mm`` plus one int32 -> bf16 pass (the bytes the stock epilogue moves once Inductor fuses it into the
consumer). The fastest fused tile wins unless cuBLAS beats it (``_CUBLAS_MARGIN``); a cuBLAS bucket keeps the stock
torchao forward for that call (traced, so Inductor still fuses its epilogue). A chosen alternative tile still has to
pass the bit-exact probe. Results persist per toolchain (GPU name, capability, torch, Triton, CUDA, kernel source) as
JSON under the Studio cache (``UNSLOTH_DIFFUSION_INT8_GEMM_TUNE_DIR`` overrides), so a later load only reads them.
``UNSLOTH_DIFFUSION_INT8_GEMM_AUTOTUNE=0`` skips the gate (static table, the previous behaviour).

ConvRot Linears (MiniMax-H3, Qwen-Image-2.1 ConvRot) run ``ConvRotLinear.forward``'s own rotation first, so eager
output stays bit-identical; kill switch ``UNSLOTH_DIFFUSION_INT8_GEMM_CONVROT=0``.

Rotated Linears on the archs ``diffusion_convrot_quant`` covers run the rotation and the act quant as ONE kernel
(``rotq_i8``, codes and scales bit-identical, probed per device). Where the fused GEMM is off but that kernel is on
(sm100), only the rotated Linears are swapped and their GEMM is cuBLAS ``_int_mm`` with torchao's epilogue, op for
op. Kill switch ``UNSLOTH_DIFFUSION_INT8_ROTQUANT=0``.

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
INT8_GEMM_TILES_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM_TILES"
INT8_GEMM_AUTOTUNE_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM_AUTOTUNE"
INT8_GEMM_TUNE_DIR_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM_TUNE_DIR"
_MIN_TRITON = (3, 2)
# torch._int_mm needs M > 16; below that the stock path (safe_int_mm padding) stays in charge.
_MIN_ROWS = 17
_K_ALIGN = 64
_N_ALIGN = 16
_OP_NAMESPACE = "unsloth_studio"
_OP_NAME = "int8_mm_dequant"
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

# Per-shape tiles: (major, minor) -> ((N_lo, N_hi, K_lo, K_hi, tile), ...), first match wins, bounds inclusive,
# M >= _SHAPE_MIN_M only. No L4 rule on purpose: per-GEMM wins there cost s/step end to end (power-capped clocks).
_SHAPE_TILES: dict = {
    (12, 0): (
        # MiniMax-H3 fused QKV / FFN in
        (16384, 32768, 4096, 6144, (128, 128, 64, 32, 4, 4)),
        # deep K: H3 out / FFN down, Qwen-Image-2.1 MLP out, Z-Image w2
        (1, 8192, 7168, 16384, (256, 128, 128, 8, 8, 3)),
    ),
}
_SHAPE_MIN_M = 1024

# Latency gate. Extra fused candidates per arch, timed beside the probed default and the per-shape tile. Triton 3.8 on
# sm80 / sm89 slows every tile holding 128 int32 accumulators per thread (A100 128x128x128 / 4 warps 430 -> 59 TOPS,
# L4 256x128x128 / 8 warps 96 -> 25); the 64-per-thread tiles below stay fast there (A100 ~290, L4 ~110 TOPS).
_ALT_TILES: dict = {
    (8, 0): ((64, 128, 128, 8, 4, 4), (128, 64, 128, 8, 4, 4), (128, 128, 128, 8, 8, 3)),
    (8, 9): ((128, 128, 128, 8, 8, 3), (128, 128, 64, 8, 8, 4), (128, 128, 64, 8, 4, 4)),
    (12, 0): ((128, 128, 64, 8, 4, 3), (128, 128, 64, 8, 8, 4)),
}
# M buckets: (upper bound exclusive, M timed). Text-stream / small images, then image and video token grids.
_M_BUCKETS = ((2048, 1024), (None, 4096))
# cuBLAS must beat the best fused tile by this factor. Its arm already pays a full int32 -> bf16 pass (6 B / element),
# more than the stock path's real extra traffic once Inductor fuses the epilogue (4 B), so a tie stays fused. 0.92
# kept A100 / Triton 3.8 klein on fused tiles 1.7% slower end to end than the kill switch.
_CUBLAS_MARGIN = 1.0
# Alternative tiles must beat the probed default by this factor (timing noise must not churn the tile).
_ALT_MARGIN = 0.97
# Timing scratch cap (int32 accumulator of the cuBLAS arm); a bigger bucket is timed at fewer rows.
_TUNE_MAX_BYTES = 384 << 20
_TUNE_VERSION = 1
_CUBLAS = "cublas"
# device index -> {(n, k, bucket): tile tuple or _CUBLAS}
_DEVICE_PLAN: dict = {}
# device index -> {tile: probe passed}
_PROBED: dict = {}


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


def tiles_enabled() -> bool:
    raw = (os.environ.get(INT8_GEMM_TILES_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def autotune_enabled() -> bool:
    raw = (os.environ.get(INT8_GEMM_AUTOTUNE_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def m_bucket(m: int) -> int:
    for i, (hi, _) in enumerate(_M_BUCKETS):
        if hi is None or m < hi:
            return i
    return len(_M_BUCKETS) - 1


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

    return types.SimpleNamespace(i8mm_dq = i8mm_dq)


# device index -> probed tile (None = stock).
_DEVICE_CFG: dict = {}
# device index -> the per-shape rules whose tile passed the probe on that device.
_DEVICE_TILES: dict = {}


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


def tile_for(index: Any, m: int, n: int, k: int) -> Optional[tuple]:
    """The probed tile this device runs an ``[m, k] x [n, k]`` GEMM with: a per-shape rule, else the arch default."""
    cfg = _DEVICE_CFG.get(index)
    if cfg is None or m < _SHAPE_MIN_M:
        return cfg
    for n_lo, n_hi, k_lo, k_hi, tile in _DEVICE_TILES.get(index, ()):
        if n_lo <= n <= n_hi and k_lo <= k <= k_hi:
            return tile
    return cfg


def planned(index: Any, m: int, n: int, k: int) -> Any:
    """The latency gate's choice for this call: a tile, ``_CUBLAS``, or None (not timed: ``tile_for`` decides)."""
    plan = _DEVICE_PLAN.get(index)
    if not plan:
        return None
    return plan.get((int(n), int(k), m_bucket(m)))


def fused_for(index: Any, m: int, n: int, k: int) -> bool:
    """False where the latency gate measured cuBLAS faster for this shape (the caller keeps the stock forward)."""
    return planned(index, m, n, k) != _CUBLAS


def _run(a: Any, w: Any, xs: Any, ws: Any, bias: Any) -> Any:
    """Op body: the probed tile for this device and shape, else (launch failure, ``_aligned`` false) the stock
    epilogue."""
    _CALLS[0] += 1
    a = a if a.stride(-1) == 1 else a.contiguous()
    index = a.device.index
    choice = planned(index, a.shape[0], w.shape[0], a.shape[1]) if _DEVICE_CFG.get(index) else None
    if choice == _CUBLAS:
        return reference(a, w, xs, ws, bias)
    cfg = choice if choice is not None else tile_for(index, a.shape[0], w.shape[0], a.shape[1])
    if cfg is not None and w.stride(-1) == 1 and w.device == a.device and _aligned(a, w):
        try:
            return _launch(a, w, xs, ws, bias, cfg)
        except Exception:  # noqa: BLE001 - a failed launch keeps the stock math
            if choice is not None and cfg != _DEVICE_CFG.get(index):
                _DEVICE_PLAN[index] = {}  # drop the timed tiles, keep the probed default
                return _run(a, w, xs, ws, bias)
            if cfg != _DEVICE_CFG.get(index) and _DEVICE_TILES.get(index):
                _DEVICE_TILES[index] = ()  # drop the shape tiles, keep the probed default
                return _run(a, w, xs, ws, bias)
            _DEVICE_CFG[index] = None
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


def shape_tiles(capability: Optional[tuple]) -> tuple:
    """This arch's per-shape rules, or () (``UNSLOTH_DIFFUSION_INT8_GEMM_TILES=0``, unmeasured arch)."""
    if capability is None or not tiles_enabled():
        return ()
    return tuple(_SHAPE_TILES.get((int(capability[0]), int(capability[1])), ()))


def device_config(index: int) -> Optional[tuple]:
    """Probe once per device: arch gate, Triton, op registration, then a launch that must match the eager torchao
    epilogue bit for bit (bf16 and fp32 activation scales, bias, ragged M / N / K). None keeps the stock path. Each
    per-shape tile is probed the same way; one that fails leaves its shapes on the default tile."""
    if index in _DEVICE_CFG:
        return _DEVICE_CFG[index]
    cfg = None
    rules: tuple = ()
    try:
        import torch
        if (
            torch.cuda.is_available()
            and not getattr(torch.version, "hip", None)
            and _triton_version_ok()
            and _kernels() is not None
            and _op() is not None
        ):
            cap = torch.cuda.get_device_capability(index)
            want = arch_config(cap)
            for cand in (want, _FALLBACK_CONFIG) if want is not None else ():
                if cand is not None and _probe(index, cand):
                    cfg = cand
                    break
            if cfg is not None:
                ok: dict = {}
                for rule in shape_tiles(cap):
                    tile = rule[4]
                    if tile not in ok:
                        ok[tile] = tile == cfg or _probe(index, tile)
                rules = tuple(rule for rule in shape_tiles(cap) if ok[rule[4]])
    except Exception:  # noqa: BLE001
        cfg = None
    _DEVICE_TILES[index] = rules if cfg is not None else ()
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
    from .diffusion_convrot_quant import act_quant_v1
    return act_quant_v1(x2d)


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


def stock_gemm(xq: Any, wq: Any, xs: Any, ws: Any, bias: Any) -> Any:
    """torchao's int8 Linear after its act quant, op for op: cuBLAS ``_int_mm``, ``* x_scale`` in the scale's dtype,
    bf16, ``* w_scale``, ``+ bias``, bf16 (``reference`` minus the small-M padding; callers keep M >= 17). torchao 0.17's
    zero-point pass subtracts an exact zero for this symmetric quant and is skipped."""
    import torch

    y = (torch._int_mm(xq, wq.t()) * xs.reshape(-1, 1)).to(torch.bfloat16)
    y = y * ws.reshape(-1)
    if bias is not None:
        y = y + bias
    return y.to(torch.bfloat16)


def _linear_forward(self: Any, x: Any) -> Any:
    """``nn.Linear.forward`` for a torchao int8 dynamic weight: torchao's activation quant (fused with the ConvRot
    rotation where that kernel is on), then the fused-dequant GEMM, or cuBLAS + torchao's epilogue where it is not."""
    import torch

    rec = self.__dict__.get(_REC)
    if rec is None or x.dtype != torch.bfloat16 or not x.is_cuda:
        return type(self).forward(self, x)
    kind, group, rotq, weight, fused = rec
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
    if fused and not fused_for(x.device.index, x.numel() // x.shape[-1], wq.shape[0], wq.shape[1]):
        fused = False  # the latency gate measured cuBLAS faster here: torchao's own epilogue, fused by Inductor
    q = None
    if rotq:
        from .diffusion_convrot_quant import convrot_act_quant
        q = convrot_act_quant(x.reshape(-1, x.shape[-1]), group, kind == "v2")
    if q is not None:
        xq, xs = q
    elif not fused:
        # rotation kernel not covering this call and no fused GEMM: the module's own stock forward
        return type(self).forward(self, x)
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
    if fused:
        y = _OP_HANDLE(xq, wq, xs.reshape(-1), ws, self.bias)
    else:
        y = stock_gemm(xq, wq, xs, ws, self.bias)
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
    if not fused_for(q.device.index, q.shape[0], parts[0].shape[0], parts[0].shape[1]):
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


def candidates(transformer: Any, rotated_only: bool = False) -> int:
    try:
        recs = (_eligible(m) for m in transformer.modules())
        return sum(1 for r in recs if r is not None and (r[1] is not None or not rotated_only))
    except Exception:  # noqa: BLE001
        return 0


def _gates(capability: Optional[tuple]) -> tuple:
    """(fused GEMM on this arch, rotation kernel on this arch), before any probe."""
    from .diffusion_convrot_quant import rotquant_config

    gemm = arch_config(capability) is not None
    rotq = convrot_enabled() and rotquant_config(capability) is not None
    return gemm, rotq


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
        gemm, rotq = _gates(torch.cuda.get_device_capability(torch.cuda.current_device()))
        if not (gemm or rotq):
            return 0
    except Exception:  # noqa: BLE001
        return 0
    n = candidates(transformer, rotated_only = not gemm)
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
    global _OP_HANDLE
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
    if not recs:
        return 0
    from .diffusion_convrot_quant import rotquant_device_config

    gemm = device_config(index) is not None
    rotq = any(r[1] is not None for _, r in recs) and rotquant_device_config(index) is not None
    if not gemm:
        # no fused GEMM here: only rotated Linears gain (one kernel instead of rotation GEMM + act quant)
        recs = [(m, r) for m, r in recs if r[1] is not None] if rotq else []
    if any(r[0] == "v1" for _, r in recs) and not _v1_act_quant_matches(index):
        recs = [(m, r) for m, r in recs if r[0] != "v1"]
    if not recs:
        return 0
    if gemm and autotune_enabled():
        import time

        t0 = time.perf_counter()
        try:
            plan = tune(index, {(int(m.out_features), int(m.in_features)) for m, _ in recs})
        except Exception:  # noqa: BLE001 - an untimed shape keeps the static tile
            plan = _DEVICE_PLAN.get(index, {})
        if logger is not None and plan:
            n_cublas = sum(1 for c in plan.values() if c == _CUBLAS)
            logger.info(
                "diffusion.int8_gemm: latency gate %d shape bucket(s) fused, %d cuBLAS (%.1f s)",
                len(plan) - n_cublas,
                n_cublas,
                time.perf_counter() - t0,
            )
    count = 0
    with _LOCK:
        if gemm:
            _OP_HANDLE = _op()
            if _OP_HANDLE is None:
                return 0
        for module, rec in recs:
            if _MARK in module.__dict__:
                count += 1
                continue
            # kind, rotation group, fused rotation + act quant on, the Parameter (no payload alias), fused GEMM on
            module.__dict__[_REC] = (
                rec[0],
                rec[1],
                bool(rotq and rec[1] is not None),
                rec[3],
                gemm,
            )
            module.__dict__[_MARK] = module.__dict__.get("forward", _NO_PREV)
            module.forward = types.MethodType(_linear_forward, module)
            count += 1
    if logger is not None and count:
        logger.info(
            "diffusion.int8_gemm: %d int8 Linear(s) %s on sm_%d%d%s",
            count,
            "run the fused-dequant GEMM (bf16 out)" if gemm else "keep cuBLAS",
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


def toolchain_key(index: int) -> str:
    """What a timing depends on: GPU, torch (cuBLAS), Triton, CUDA and the kernel source."""
    import hashlib
    import inspect

    import torch
    import triton

    src = inspect.getsource(_kernels)
    cap = torch.cuda.get_device_capability(index)
    return "|".join(
        (
            f"v{_TUNE_VERSION}",
            torch.cuda.get_device_name(index),
            f"sm{cap[0]}{cap[1]}",
            f"torch{torch.__version__}",
            f"triton{triton.__version__}",
            f"cuda{torch.version.cuda}",
            hashlib.sha1(src.encode("utf-8")).hexdigest()[:12],
        )
    )


def _tune_path(key: str) -> Optional[str]:
    import hashlib

    root = (os.environ.get(INT8_GEMM_TUNE_DIR_ENV) or "").strip()
    if not root:
        try:
            from utils.paths.storage_roots import cache_root
        except Exception:  # noqa: BLE001 - outside Studio: in-memory only
            return None
        root = str(cache_root() / "diffusion_int8_gemm")
    return os.path.join(root, "tune_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".json")


def _load_plan(path: Optional[str], key: str) -> dict:
    import json

    if path is None or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding = "utf-8") as f:
            data = json.load(f)
        if data.get("key") != key:
            return {}
        out = {}
        for name, choice in data.get("plan", {}).items():
            n, k, b = (int(v) for v in name.split(","))
            out[(n, k, b)] = _CUBLAS if choice == _CUBLAS else tuple(int(v) for v in choice)
        return out
    except Exception:  # noqa: BLE001 - a corrupt file is re-timed
        return {}


def _save_plan(path: Optional[str], key: str, plan: dict, timings: dict) -> None:
    import json

    if path is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok = True)
        data = {
            "key": key,
            "plan": {
                f"{n},{k},{b}": (c if c == _CUBLAS else list(c)) for (n, k, b), c in sorted(plan.items())
            },
            "ms": timings,
        }
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding = "utf-8") as f:
            json.dump(data, f, indent = 1, sort_keys = True)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - read-only cache: timed again next load
        pass


def _time_ms(fn: Any, iters: int = 3, rounds: int = 3) -> float:
    import torch

    fn()
    torch.cuda.synchronize()
    best = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        best.append(start.elapsed_time(end) / iters)
    best.sort()
    return best[len(best) // 2]


def _measure(index: int, n: int, k: int, m: int, tiles: tuple) -> dict:
    """ms per arm at [m, k] x [n, k]: each candidate tile and ``_CUBLAS`` (``_int_mm`` + one int32 -> bf16 pass)."""
    import torch

    dev = torch.device("cuda", index)
    m = max(_MIN_ROWS, min(m, _TUNE_MAX_BYTES // (4 * n)))
    with torch.cuda.device(dev):
        a = torch.randint(-127, 128, (m, k), device = dev, dtype = torch.int8)
        w = torch.randint(-127, 128, (n, k), device = dev, dtype = torch.int8)
        xs = torch.full((m,), 1e-3, device = dev, dtype = torch.bfloat16)
        ws = torch.full((n,), 1e-3, device = dev, dtype = torch.bfloat16)
        out: dict = {}

        def cublas():
            return torch._int_mm(a, w.t()).to(torch.bfloat16)

        out[_CUBLAS] = _time_ms(cublas)
        for tile in tiles:
            try:
                out[tile] = _time_ms(lambda tile = tile: _launch(a, w, xs, ws, None, tile))
            except Exception:  # noqa: BLE001 - out of shared memory on this part, ...
                continue
        del a, w, xs, ws
    return out


def _probed(index: int, tile: tuple) -> bool:
    seen = _PROBED.setdefault(index, {})
    if tile not in seen:
        seen[tile] = tile == _DEVICE_CFG.get(index) or _probe(index, tile)
    return seen[tile]


def tune(index: int, shapes: Any) -> dict:
    """Latency gate for ``(n, k)`` shapes on device ``index`` (fused GEMM already probed on): fills ``_DEVICE_PLAN``
    from the toolchain's file, times what is missing, writes it back. Must run eagerly (not under capture / trace)."""
    import torch

    default = _DEVICE_CFG.get(index)
    if default is None:
        return {}
    cap = tuple(torch.cuda.get_device_capability(index))
    key = toolchain_key(index)
    path = _tune_path(key)
    plan = _DEVICE_PLAN.setdefault(index, {})
    stored = _load_plan(path, key)
    timings: dict = {}
    dirty = False
    for n, k in sorted(shapes):
        if k % _K_ALIGN or n % _N_ALIGN:
            continue
        for b, (_, rep_m) in enumerate(_M_BUCKETS):
            slot = (n, k, b)
            choice = stored.get(slot)
            if choice is None:
                static = tile_for(index, rep_m, n, k)
                tiles = tuple(dict.fromkeys((static, default, *_ALT_TILES.get(cap, ()))))
                ms = _measure(index, n, k, rep_m, tiles)
                timings[f"{n},{k},{b}"] = {str(t): round(v, 4) for t, v in ms.items()}
                fused_ms = {t: v for t, v in ms.items() if t != _CUBLAS}
                if not fused_ms:
                    continue
                choice = static if static in fused_ms else min(fused_ms, key = fused_ms.get)
                fastest = min(fused_ms, key = fused_ms.get)
                if fused_ms[fastest] < _ALT_MARGIN * fused_ms[choice]:
                    choice = fastest
                if ms[_CUBLAS] < _CUBLAS_MARGIN * fused_ms[choice]:
                    choice = _CUBLAS
                stored[slot] = choice
                dirty = True
            if choice != _CUBLAS and not _probed(index, choice):
                choice = default  # an alternative tile that is not bit-exact here: the probed default
            plan[slot] = choice
    if dirty:
        _save_plan(path, key, stored, timings)
    return plan

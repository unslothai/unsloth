# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""int8 GEMM with torchao dequant epilogue in Triton, bf16 out; sm80/sm89/sm120 measured, other sm80+ behind a
bit-exact probe and a latency check (UNSLOTH_DIFFUSION_INT8_GEMM_UNLISTED=0 keeps them stock), sm100 / sm103 stock
(ROCm and CPU never reach it); Linear output bit-identical, compiled block not end to end;
UNSLOTH_DIFFUSION_INT8_GEMM=0 disables."""

from __future__ import annotations

import os
import threading
import types
from functools import lru_cache
from typing import Any, Optional
from .rocm_bf16 import is_rocm_torch

INT8_GEMM_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM"
INT8_GEMM_CONVROT_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM_CONVROT"
INT8_GEMM_STREAMED_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM_STREAMED"
INT8_GEMM_TILES_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM_TILES"
INT8_SWIGLU_STREAMED_ENV = "UNSLOTH_DIFFUSION_INT8_SWIGLU_STREAMED"
INT8_GEMM_UNLISTED_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM_UNLISTED"
_MIN_TRITON = (3, 2)
# torch._int_mm needs M > 16; below that the stock path (safe_int_mm padding) stays in charge.
_MIN_ROWS = 17
_K_ALIGN = 64
_N_ALIGN = 16
_OP_NAMESPACE = "unsloth_studio"
_OP_NAME = "int8_mm_dequant"
_REC = "_unsloth_i8_gemm"
_MARK = "_unsloth_i8_gemm_prev"
# on a SwiGLU MLP whose down projection takes its codes from the gate / value halves
_FF_MARK = "_unsloth_i8_gemm_swiglu_prev"
_SWIGLU_FF_CLASSES = ("QwenImage21SwiGLUFeedForward",)
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
# When the arch tile does not fit this part's shared memory, and the unlisted archs' tile (64 KB of stages: fits 99 KB).
_FALLBACK_CONFIG = (128, 128, 64, 8, 4, 4)
# Measured slower than cuBLAS: not probed unless forced.
_UNLISTED_SKIP = ((10, 0), (10, 3))

# First match wins, bounds inclusive. No L4 rule on purpose: power-capped clocks lose end to end.
# (major, minor) -> ((N_lo, N_hi, K_lo, K_hi, tile), ...), M >= _SHAPE_MIN_M only.
_SHAPE_TILES: dict = {
    (12, 0): (
        # MiniMax-H3 fused QKV / FFN in
        (16384, 32768, 4096, 6144, (128, 128, 64, 32, 4, 4)),
        # deep K: H3 out / FFN down, Qwen-Image-2.1 MLP out, Z-Image w2
        (1, 8192, 7168, 16384, (256, 128, 128, 8, 8, 3)),
    ),
}
_SHAPE_MIN_M = 1024


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


def swiglu_streamed_enabled() -> bool:
    raw = (os.environ.get(INT8_SWIGLU_STREAMED_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def tiles_enabled() -> bool:
    raw = (os.environ.get(INT8_GEMM_TILES_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


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


def unlisted_mode() -> str:
    """'off', 'force' (no latency check) or 'auto' for the sm80+ archs without an ``_ARCH_CONFIG`` tile."""
    gemm = int8_gemm_mode()
    if gemm != "auto":
        return gemm
    raw = (os.environ.get(INT8_GEMM_UNLISTED_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


def unlisted_arch(capability: Optional[tuple]) -> bool:
    if capability is None:
        return False
    cap = (int(capability[0]), int(capability[1]))
    return cap >= (8, 0) and cap not in _ARCH_CONFIG


def unlisted_wanted(capability: Optional[tuple], mode: Optional[str] = None) -> bool:
    mode = unlisted_mode() if mode is None else mode
    if mode == "off" or not unlisted_arch(capability):
        return False
    return mode == "force" or (int(capability[0]), int(capability[1])) not in _UNLISTED_SKIP


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
    y = y * ws.reshape(-1)
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


def _run(a: Any, w: Any, xs: Any, ws: Any, bias: Any) -> Any:
    """Op body: the probed tile for this device and shape, else (launch failure, ``_aligned`` false) the stock
    epilogue."""
    _CALLS[0] += 1
    a = a if a.stride(-1) == 1 else a.contiguous()
    index = a.device.index
    cfg = tile_for(index, a.shape[0], w.shape[0], a.shape[1])
    if cfg is not None and w.stride(-1) == 1 and w.device == a.device and _aligned(a, w):
        try:
            return _launch(a, w, xs, ws, bias, cfg)
        except Exception:  # noqa: BLE001 - a failed launch keeps the stock math
            if cfg != _DEVICE_CFG.get(index) and _DEVICE_TILES.get(index):
                _DEVICE_TILES[index] = ()  # drop the shape tiles, keep the probed default
                return _run(a, w, xs, ws, bias)
            _DEVICE_CFG[index] = None
            if index in _UNLISTED_CARDS:
                _UNLISTED_PROBE[index] = False
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
    per-shape tile is probed the same way; one that fails leaves its shapes on the default tile. Unlisted archs:
    ``unlisted_config``, re-read per call so its switch acts between loads."""
    if index in _UNLISTED_CARDS:
        return unlisted_config(index)
    if index in _DEVICE_CFG:
        return _DEVICE_CFG[index]
    cfg = None
    rules: tuple = ()
    try:
        import torch
        if (
            torch.cuda.is_available()
            and not is_rocm_torch(torch)
            and _triton_version_ok()
            and _kernels() is not None
            and _op() is not None
        ):
            cap = torch.cuda.get_device_capability(index)
            if unlisted_arch(cap):
                _UNLISTED_CARDS.add(index)
                return unlisted_config(index)
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


_UNLISTED_CARDS: set = set()
_UNLISTED_PROBE: dict = {}
_UNLISTED_OK: dict = {}
_UNLISTED_TIMING: dict = {}
# (M, N, K): Qwen-Image-2.1 attention projection, MLP in, MLP down
_UNLISTED_LATENCY_SHAPES = ((2048, 4096, 4096), (2048, 12288, 4096), (2048, 4096, 12288))


def _unlisted_latency(index: int, cfg: tuple) -> dict:
    """{"fused": us, "stock": us} summed over ``_UNLISTED_LATENCY_SHAPES``, each arm captured in a CUDA graph and
    replayed, best of seven interleaved rounds. Stock = cuBLAS ``_int_mm``; its epilogue fuses into the consumer, so
    each arm ends in one read of its output (int32 for stock, bf16 here) standing in for that consumer."""
    import torch
    from .diffusion_cuda_graph import CAPTURE_MODE, _capturing, synchronize_outside_capture

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(5)
    ops = []
    for m, n, k in _UNLISTED_LATENCY_SHAPES:
        a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).to(dev)
        w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).to(dev)
        xs = torch.full((m,), 1e-3, device = dev, dtype = torch.bfloat16)
        ws = torch.full((n,), 1e-3, device = dev, dtype = torch.bfloat16)
        ops.append((a, w, xs, ws))
    fns = {
        "fused": lambda a, w, xs, ws: _launch(a, w, xs, ws, None, cfg).amax(),
        "stock": lambda a, w, xs, ws: torch._int_mm(a, w.t()).amax(),
    }
    reps = 2
    graphs = {}
    with torch.cuda.device(dev):
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for fn in fns.values():
                for op in ops:
                    fn(*op)  # compile / autotune outside the capture
        torch.cuda.current_stream(dev).wait_stream(side)
        synchronize_outside_capture(dev)
        for name, fn in fns.items():
            graph = torch.cuda.CUDAGraph()
            with _capturing(), torch.cuda.graph(graph, capture_error_mode = CAPTURE_MODE):
                for _ in range(reps):
                    for op in ops:
                        fn(*op)
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


def unlisted_config(index: int) -> Optional[tuple]:
    """Unlisted sm80+ card: ``_FALLBACK_CONFIG`` when it passed the bit-exact probe (once per device) and the latency
    check (once per (device, mode); skipped under force), else None. Outside any trace / capture."""
    import torch

    mode = unlisted_mode()
    cfg = None
    try:
        if unlisted_wanted(torch.cuda.get_device_capability(index), mode):
            if index not in _UNLISTED_PROBE:
                _UNLISTED_PROBE[index] = _probe(index, _FALLBACK_CONFIG)
            if _UNLISTED_PROBE[index]:
                key = (index, mode)
                if key not in _UNLISTED_OK:
                    ok = True
                    if mode != "force":
                        try:
                            timing = _unlisted_latency(index, _FALLBACK_CONFIG)
                            _UNLISTED_TIMING[index] = {k: round(v, 1) for k, v in timing.items()}
                            ok = timing["fused"] <= timing["stock"]
                        except Exception:  # noqa: BLE001 - main's path stays
                            ok = False
                    _UNLISTED_OK[key] = ok
                cfg = _FALLBACK_CONFIG if _UNLISTED_OK[key] else None
    except Exception:  # noqa: BLE001
        cfg = None
    _DEVICE_TILES[index] = ()
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
            ws = ws.float() if xs32 else ws
            b = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16).to(dev) if bias else None
            if not torch.equal(_launch(a, w, xs, ws, b, cfg), reference(a, w, xs, ws, b)):
                return False
        a, w = tie_operands(dev)
        xs = torch.ones(a.shape[0], device = dev, dtype = torch.bfloat16)
        ws = torch.ones(w.shape[0], device = dev, dtype = torch.bfloat16)
        if not torch.equal(_launch(a, w, xs, ws, None, cfg), reference(a, w, xs, ws, None)):
            return False
        from .diffusion_cuda_graph import synchronize_outside_capture

        synchronize_outside_capture(dev)
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


def _payload(kind: str, weight: Any) -> tuple:
    """(int8 [N, K], scale [N]) off the live parameter, never a cached alias (a moved weight must not go stale)."""
    if kind == "v1":
        impl = weight.original_weight_tensor.tensor_impl
        return impl.int_data, impl.scale.reshape(-1)
    return weight.qdata, weight.scale.reshape(-1)


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
    wq, ws = _payload(kind, weight)
    if wq.device != x.device:
        return type(self).forward(self, x)
    lead = x.shape[:-1]
    if x.numel() < _MIN_ROWS * x.shape[-1]:
        return type(self).forward(self, x)
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


def _hooked(module: Any) -> bool:
    """A forward (pre-)hook, own or global, or an instance ``forward``: a fused path skipping ``module`` would bypass it."""
    import torch

    d = module.__dict__
    if "forward" in d or d.get("_forward_hooks") or d.get("_forward_pre_hooks"):
        return True
    nn_module = torch.nn.modules.module
    return bool(nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks)


def _swapped_rec(module: Any) -> Optional[tuple]:
    """Record of a Linear we swapped with its install-time weight and no offload / forward hook or override, else None:
    the shared paths run the GEMM without calling the module, so any of those would be bypassed."""
    import torch

    d = module.__dict__
    rec = d.get(_REC)
    if rec is None or _MARK not in d or "_diffusers_hook" in d or "_hf_hook" in d:
        return None
    if getattr(d.get("forward"), "__func__", None) is not _linear_forward:
        return None
    if d.get("_forward_hooks") or d.get("_forward_pre_hooks"):
        return None
    nn_module = torch.nn.modules.module
    if nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks:
        return None
    if module.weight is not rec[3]:
        return None
    return rec


def shared_rotq_linears(x: Any, linears: tuple) -> Optional[list]:
    """``[m(x) for m in linears]`` with one shared rotation + act quant; None when any is not covered."""
    import torch

    if x.dtype != torch.bfloat16 or not x.is_cuda or x.numel() < _MIN_ROWS * x.shape[-1]:
        return None
    recs = [_swapped_rec(m) for m in linears]
    if any(r is None or not r[2] for r in recs):
        return None
    kind, group = recs[0][0], recs[0][1]
    if any(r[0] != kind or r[1] != group for r in recs):
        return None
    parts = [_payload(r[0], r[3]) for r in recs]
    if any(wq.device != x.device for wq, _ in parts):
        return None
    from .diffusion_convrot_quant import convrot_act_quant

    q = convrot_act_quant(x.reshape(-1, x.shape[-1]), group, kind == "v2")
    if q is None:
        return None
    xq, xs = q
    lead = x.shape[:-1]
    outs = []
    for m, r, (wq, ws) in zip(linears, recs, parts):
        if r[4]:
            y = _OP_HANDLE(xq, wq, xs.reshape(-1), ws, m.bias)
        else:
            y = stock_gemm(xq, wq, xs, ws, m.bias)
        outs.append(y.reshape(*lead, y.shape[-1]))
    return outs


def _swiglu_down(down: Any, gate: Any, value: Any) -> Optional[Any]:
    """``down(silu(gate) * value)`` via the fused SwiGLU rotation + quant, else None."""
    import torch

    rec = _swapped_rec(down)
    if rec is None:
        return None
    kind, group, rotq, weight, fused = rec
    if (
        not rotq
        or gate.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
        or not gate.is_cuda
        or gate.shape != value.shape
        or gate.shape[-1] != down.in_features
        or gate.numel() < _MIN_ROWS * gate.shape[-1]
    ):
        return None
    wq, ws = _payload(kind, weight)
    if wq.device != gate.device:
        return None
    from .diffusion_convrot_quant import convrot_swiglu_act_quant

    k = gate.shape[-1]
    q = convrot_swiglu_act_quant(gate.reshape(-1, k), value.reshape(-1, k), group, kind == "v2")
    if q is None:
        return None
    xq, xs = q
    if fused:
        y = _OP_HANDLE(xq, wq, xs.reshape(-1), ws, down.bias)
    else:
        y = stock_gemm(xq, wq, xs, ws, down.bias)
    return y.reshape(*gate.shape[:-1], y.shape[-1])


def _plain_silu(act: Any) -> bool:
    """``act`` is an exact, unhooked ``nn.SiLU`` (the fused prologue hard-codes SiLU); ``type() is`` avoids a graph break."""
    from torch import nn
    return type(act) is nn.SiLU and not _hooked(act)


def _swiglu_ff_forward(self: Any, hidden_states: Any) -> Any:
    """``QwenImage21SwiGLUFeedForward.forward`` on the shared and SwiGLU-fused quants."""
    pair = shared_rotq_linears(hidden_states, (self.gate_layer, self.proj))
    if pair is None:
        gate, value = self.gate_layer(hidden_states), self.proj(hidden_states)
    else:
        gate, value = pair
    y = _swiglu_down(self.out, gate, value) if _plain_silu(self.activation_fn) else None
    if y is None:
        return self.out(self.activation_fn(gate) * value)
    return y


def _swiglu_on(index: int) -> bool:
    from .diffusion_convrot_quant import _SWIGLU_DEVICE, rotquant_swiglu_enabled
    return rotquant_swiglu_enabled() and bool(_SWIGLU_DEVICE.get(index))


def _swiglu_ff_eligible(module: Any) -> bool:
    from torch import nn

    if type(module).__name__ not in _SWIGLU_FF_CLASSES or "forward" in module.__dict__:
        return False
    parts = [getattr(module, n, None) for n in ("gate_layer", "proj", "out")]
    if any(p is None or _REC not in p.__dict__ for p in parts):
        return False
    act = getattr(module, "activation_fn", None)
    if type(act) is not nn.SiLU or "forward" in act.__dict__:
        return False
    # only a rotated down projection on the fused rotation kernel gains (its input is the product)
    return bool(parts[2].__dict__[_REC][2])


def _install_swiglu_ff(
    transformer: Any,
    index: int,
    streamed: bool = False,
) -> int:
    """Patch the eligible SwiGLU MLPs on card ``index`` (``streamed``: also host-resident ones)."""
    count = 0
    for module in transformer.modules():
        if _FF_MARK in module.__dict__:
            count += 1
            continue
        if not _swiglu_ff_eligible(module):
            continue
        card = _weight_card(module.out)
        if card != index and not (streamed and card is None):
            continue
        module.__dict__[_FF_MARK] = True
        module.forward = types.MethodType(_swiglu_ff_forward, module)
        count += 1
    return count


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


def _unlisted_rotated(index: int) -> bool:
    """Unlisted card: ``rotq_h16`` armed for it (probe + its own latency check). Takes no ``diffusion_int8_fused``
    lock: its SwiGLU fusion reaches this through ``rotated_ff`` while holding it."""
    if not convrot_enabled():
        return False
    try:
        from . import diffusion_convrot_quant as cq
        from .diffusion_int8_fused import _act_quant_device_ok, _act_quant_op

        actq = None
        if (index, cq.unlisted_mode()) not in cq._UNLISTED_DEVICE:
            actq = _act_quant_op() if _act_quant_device_ok(index) else None
        return bool(cq.unlisted_device_ok(index, actq))
    except Exception:  # noqa: BLE001
        return False


def rotated_swapped(index: int) -> Optional[bool]:
    """On an unlisted card (outside any trace): True when rotated Linears go through this swap (fused GEMM and the
    ``rotq_h16`` rotation + act quant both on), so the Qwen-Image-2.1 / Z-Image stock fusions stand down. False keeps
    them, exactly as on main. None on a listed arch (``_gates`` decides)."""
    import torch

    cap = torch.cuda.get_device_capability(index)
    if not unlisted_arch(cap) or _gates(cap)[1] or int8_gemm_mode() == "off":
        return None
    return _unlisted_rotated(index) and device_config(index) is not None


def install(
    transformer: Any,
    logger: Any = None,
    offload_active: bool = False,
    device: Any = None,
) -> int:
    """Idempotent, returns the candidate count; must run before the first compiled forward; block-streamed denoisers need device or keep stock path."""
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

        if not torch.cuda.is_available() or is_rocm_torch(torch):
            return 0
        if not _triton_version_ok() or _kernels() is None:
            return 0
        cap = torch.cuda.get_device_capability(torch.cuda.current_device())
        gemm, rotq = _gates(cap)
        gemm = gemm or unlisted_wanted(cap)
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
    from .diffusion_int8_fused import resident_cuda_device

    import torch

    if device is not None:
        try:
            dev = torch.device(device)
        except Exception:  # noqa: BLE001
            return 0
        if dev.type != "cuda" or is_rocm_torch(torch) or not torch.cuda.is_available():
            return 0
    else:
        dev = resident_cuda_device(transformer)
        if dev is None:
            cards = _split_cards(transformer)
            if cards:
                count = sum(_swap_on(transformer, logger, index) for index in cards)
                return count
    if dev is None:
        return 0

    index = dev.index if dev.index is not None else torch.cuda.current_device()
    return _swap_on(transformer, logger, index, streamed = device is not None)


def _split_cards(transformer: Any) -> tuple:
    """The CUDA ordinals of a layer-split denoiser whose every weight sits on a card (NVIDIA), else ()."""
    try:
        import torch

        from .diffusion_layer_split import is_layer_split

        if is_rocm_torch(torch) or not is_layer_split(transformer):
            return ()
        devices = {p.device for p in transformer.parameters()}
    except Exception:  # noqa: BLE001
        return ()
    if not devices or any(d.type != "cuda" or d.index is None for d in devices):
        return ()
    return tuple(sorted(d.index for d in devices))


def _weight_card(module: Any) -> Optional[int]:
    weight = getattr(module, "weight", None)
    dev = getattr(weight, "device", None)
    return dev.index if dev is not None and dev.type == "cuda" else None


def _swap_on(
    transformer: Any,
    logger: Any,
    index: int,
    streamed: bool = False,
) -> int:
    """Swap the eligible int8 Linears of ``transformer`` against card ``index``'s probed kernels. On a layer split
    (several cards) only the Linears whose weight sits on ``index``."""
    global _OP_HANDLE
    import torch

    multi = len(_split_cards(transformer)) > 1
    recs = [(m, _eligible(m)) for m in transformer.modules()]
    recs = [(m, r) for m, r in recs if r is not None and (not multi or _weight_card(m) == index)]
    if not recs:
        return 0
    from .diffusion_convrot_quant import rotquant_device_config

    gemm = device_config(index) is not None
    rotated = any(r[1] is not None for _, r in recs)
    rotq = rotated and rotquant_device_config(index) is not None
    if gemm and rotated and not rotq and index in _UNLISTED_CARDS:
        # unlisted card: rotated Linears only with rotq_h16; else the stock fusions keep them, as on main
        rotq = _unlisted_rotated(index)
        if not rotq:
            recs = [(m, r) for m, r in recs if r[1] is None]
            if logger is not None:
                logger.info(
                    "diffusion.int8_gemm: rotated Linears keep the stock path on cuda:%d (rotq_h16 off)",
                    index,
                )
    if not gemm and logger is not None and _UNLISTED_TIMING.get(index):
        t = _UNLISTED_TIMING[index]
        logger.info(
            "diffusion.int8_gemm: fused-dequant GEMM kept off on cuda:%d (latency check: %.0f vs %.0f us cuBLAS)",
            index,
            t["fused"],
            t["stock"],
        )
    if not gemm:
        # no fused GEMM here: only rotated Linears gain (one kernel instead of rotation GEMM + act quant)
        recs = [(m, r) for m, r in recs if r[1] is not None] if rotq else []
    if any(r[0] == "v1" for _, r in recs) and not _v1_act_quant_matches(index):
        recs = [(m, r) for m, r in recs if r[0] != "v1"]
    if not recs:
        return 0
    count = 0
    with _LOCK:
        if gemm:
            _OP_HANDLE = _op()
            if _OP_HANDLE is None:
                return 0
        for module, rec in recs:
            if _MARK in module.__dict__:
                if multi:
                    module.__dict__[_REC] = (
                        rec[0],
                        rec[1],
                        bool(rotq and rec[1] is not None),
                        rec[3],
                        gemm,
                    )
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
        ff = (
            _install_swiglu_ff(
                transformer,
                index,
                streamed = streamed and not multi and swiglu_streamed_enabled(),
            )
            if rotq
            else 0
        )
    if logger is not None and count:
        from .diffusion_convrot_quant import _H16_DEVICE
        timing = _UNLISTED_TIMING.get(index) if gemm else None
        logger.info(
            "diffusion.int8_gemm: %d int8 Linear(s) %s on sm_%d%d%s%s%s",
            count,
            "run the fused-dequant GEMM (bf16 out)" if gemm else "keep cuBLAS",
            *torch.cuda.get_device_capability(index),
            f" (latency check: {timing['fused']:.0f} vs {timing['stock']:.0f} us cuBLAS)"
            if timing
            else "",
            (
                ", ConvRot rotation fused into the act quant ("
                + ("H16 x H16 kernel" if _H16_DEVICE.get(index) else "dense rotation kernel")
                + ")"
            )
            if rotq
            else "",
            (
                f", one act quant for gate / value on {ff} SwiGLU MLP(s)"
                + (
                    " + the SiLU product fused into the down projection's act quant"
                    if _swiglu_on(index)
                    else ""
                )
            )
            if ff
            else "",
        )
    return count


def after_layer_split(transformer: Any, logger: Any = None) -> dict:
    """Run once a layer split placed ``transformer``'s blocks, before its first forward: re-home the fused int8 / QKV
    records built while it sat on one card (``diffusion_int8_fused.relocate_after_layer_split``), and probe each card's
    GEMM tiles so the int8 Linears swapped before the split run their own card's kernel; each RoPE patch's probe map
    covers every card. Idempotent; a no-op on a denoiser that is not split."""
    out: dict = {"cards": (), "gemm": 0}
    cards = _split_cards(transformer)
    if len(cards) < 2:
        return out
    out["cards"] = cards
    try:
        from .diffusion_int8_fused import relocate_after_layer_split
        out.update(relocate_after_layer_split(transformer, logger))
    except Exception as exc:  # noqa: BLE001 - the stock paths stay correct
        if logger is not None:
            logger.warning(
                "diffusion.int8_gemm: fused record re-home after the layer split failed (%s)", exc
            )
    try:
        from .diffusion_qwenimage21_rope import probe_split_cards
        out["rope_cards"] = probe_split_cards(cards)
    except Exception:  # noqa: BLE001 - an unprobed card keeps the complex RoPE
        pass
    try:
        from .diffusion_zimage_fused import probe_split_cards as zimage_probe
        out["zimage_rope_cards"] = zimage_probe(cards)
    except Exception:  # noqa: BLE001 - an unprobed card keeps the complex RoPE
        pass
    if int8_gemm_mode() != "off" and any(_MARK in m.__dict__ for m in transformer.modules()):
        try:
            out["gemm"] = sum(_swap_on(transformer, logger, index) for index in cards)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning(
                    "diffusion.int8_gemm: per-card swap after the layer split failed (%s)", exc
                )
    return out


def uninstall(transformer: Any = None) -> None:
    if transformer is None:
        return
    from .diffusion_int8_fused import cancel_first_call

    cancel_first_call(transformer, "int8_gemm")
    with _LOCK:
        for module in transformer.modules():
            if module.__dict__.pop(_FF_MARK, None) is not None:
                fwd = module.__dict__.get("forward")
                if getattr(fwd, "__func__", None) is _swiglu_ff_forward:
                    module.__dict__.pop("forward", None)
            prev = module.__dict__.pop(_MARK, None)
            if prev is None:
                continue
            module.__dict__.pop(_REC, None)
            if prev is _NO_PREV:
                module.__dict__.pop("forward", None)
            else:
                module.forward = prev


def is_installed(module: Any) -> bool:
    d = getattr(module, "__dict__", {})
    return _MARK in d or _FF_MARK in d


def call_count() -> int:
    return _CALLS[0]

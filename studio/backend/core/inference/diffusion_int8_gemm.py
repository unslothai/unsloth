# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""int8 x int8 GEMM with torchao's dequant epilogue in the kernel: bf16 out instead of an int32 ``_int_mm`` output.

The stock torchao W8A8 linear writes the int32 accumulator to HBM (4 bytes per output element) and the next compiled
kernel reads it back to apply ``(c * x_scale) * w_scale``. Here one Triton kernel does the GEMM and the epilogue and
writes bf16, so every int8 Linear writes half the bytes and its consumer (norm, RoPE, SwiGLU, act quant) reads half.

NUMERICS: the Linear output is bit-identical to torchao's epilogue, eager and compiled under Studio's
``emulate_precision_casts``: ``bf16(bf16(r(c) * xs) * ws)`` (+ bias as a bf16 add), where ``r`` rounds the int32
accumulator to bf16 when the activation scale is bf16 (int32 * bf16 promotes to bf16) and is exact when it is fp32.
The activation quant is torchao's own math, so Inductor still fuses it into the producer. A compiled block is NOT
bit-identical end to end: the consumer reductions (LayerNorm / RMSNorm Welford) that used to fuse the epilogue now
read bf16 and Inductor re-tiles them, a rounding-order change smaller than compiled-vs-eager (LPIPS-gated).

Per-arch gate: sm80 / sm89 / sm120 on (measured), everything else stock. sm75: Triton cannot lower the int8 dot;
sm100: Triton int8 is slower than cuBLAS. ROCm (weight-only int8 there) and CPU never reach it.

Kill switch: ``UNSLOTH_DIFFUSION_INT8_GEMM=0``; ``=1`` also enables it on an unmeasured arch (still probe-gated).
"""

from __future__ import annotations

import os
import threading
import types
from functools import lru_cache
from typing import Any, Optional

INT8_GEMM_ENV = "UNSLOTH_DIFFUSION_INT8_GEMM"
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


def int8_gemm_mode() -> str:
    """'off', 'force' or 'auto' from the environment."""
    raw = (os.environ.get(INT8_GEMM_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


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
        y = acc.to(tl.float32)
        if not XS_FP32:
            y = _rbf16(y)
        y = _rbf16(y * xs[:, None])
        y = y * ws[None, :]
        if not WS_FP32:
            y = _rbf16(y)
        if HAS_BIAS:
            y = y + tl.load(b_ptr + rn).to(tl.float32)[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, y.to(tl.bfloat16), mask = mask)

    return types.SimpleNamespace(i8mm_dq = i8mm_dq)


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


def _run(a: Any, w: Any, xs: Any, ws: Any, bias: Any) -> Any:
    """Op body: the probed tile for this device, the stock epilogue if the launch fails."""
    _CALLS[0] += 1
    a = a if a.stride(-1) == 1 else a.contiguous()
    cfg = _DEVICE_CFG.get(a.device.index)
    if cfg is not None:
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


def _probe(index: int, cfg: tuple) -> bool:
    import torch

    dev = torch.device("cuda", index)
    g = torch.Generator(device = "cpu").manual_seed(0)
    try:
        for m, n, k, bias, xs32 in (
            (257, 384, 512, False, False),
            (33, 200, 136, True, True),
            (300, 520, 1000, True, False),
        ):
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
        torch.cuda.synchronize(dev)
        return True
    except Exception:  # noqa: BLE001 - out of shared memory, compile failure, ...
        return False


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
    kind, _wq, _ws, weight = rec
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
    x2d = x.reshape(-1, x.shape[-1])
    if x2d.shape[0] < _MIN_ROWS:
        return type(self).forward(self, x)
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


def _eligible(module: Any) -> Optional[tuple]:
    from torch import nn

    if type(module) is not nn.Linear:  # a subclass (ConvRotLinear) transforms the input first
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
        return ("v1",) + parts + (w,)
    parts = _v2_parts(w)
    if parts is not None and parts[1].dtype in (torch.bfloat16, torch.float32):
        return ("v2",) + parts + (w,)
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
) -> int:
    """Idempotent; returns the (candidate) count. Must run before the first compiled forward."""
    if int8_gemm_mode() == "off" or transformer is None or offload_active:
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


def _finalize(transformer: Any, logger: Any = None) -> int:
    count = _swap(transformer, logger)
    try:
        transformer._unsloth_int8_gemm = count  # the deferred install recorded the candidate count
    except Exception:  # noqa: BLE001
        pass
    return count


def _swap(transformer: Any, logger: Any = None) -> int:
    global _OP_HANDLE
    from .diffusion_int8_fused import resident_cuda_device

    dev = resident_cuda_device(transformer)
    if dev is None:
        return 0
    import torch

    index = dev.index if dev.index is not None else torch.cuda.current_device()
    recs = [(m, _eligible(m)) for m in transformer.modules()]
    recs = [(m, r) for m, r in recs if r is not None]
    if not recs or device_config(index) is None:
        return 0
    if any(r[0] == "v1" for _, r in recs) and not _v1_act_quant_matches(index):
        recs = [(m, r) for m, r in recs if r[0] != "v1"]
    count = 0
    with _LOCK:
        _OP_HANDLE = _op()
        if _OP_HANDLE is None:
            return 0
        for module, rec in recs:
            if _MARK in module.__dict__:
                count += 1
                continue
            module.__dict__[_REC] = (
                rec[0],
                None,
                None,
                rec[3],
            )  # kind + the Parameter, no payload alias
            module.__dict__[_MARK] = module.__dict__.get("forward", _NO_PREV)
            module.forward = types.MethodType(_linear_forward, module)
            count += 1
    if logger is not None and count:
        logger.info(
            "diffusion.int8_gemm: %d int8 Linear(s) run the fused-dequant GEMM (bf16 out) on sm_%d%d",
            count,
            *torch.cuda.get_device_capability(index),
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

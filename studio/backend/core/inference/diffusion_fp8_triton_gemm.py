# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused Triton row-wise fp8 GEMM for cards whose torch build has no row-wise ``_scaled_mm`` (the Windows wheels).

Epilogue = ``_scale_epilogue``'s fp32 ops in the same order, each ``mul.rn.f32`` / ``add.rn.f32`` inline PTX (round to
nearest, denormals kept, never contracted to an FMA), rounded once to the output dtype. The accumulator is not cuBLAS's:
a request kind (output dtype x fast / precise accumulate) takes the kernel only where its error vs fp64 is within
1.05x of the tensorwise path it replaces, or under a floor far below the output rounding (``verify_accuracy``).
sm_89 is eager only (up to 1.24x slower than the compiled fallback). ``=0`` keeps the fallback. Triton >= 3.2."""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
from typing import Any, Optional

ENV = "UNSLOTH_DIFFUSION_FP8_TRITON_GEMM"
_OFF_WORDS = ("0", "off", "false", "no", "none")
# 32-bit offsets in the kernel: larger operands keep the tensorwise fallback.
_MAX_ELEMS = (1 << 31) - 1
# Autotune key: rows rounded up to a power of two (capped), so a new resolution reuses a tuned config.
_MAX_BUCKET = 8192
# Row bucket of a traced call whose row count is symbolic: one fixed key, no per-M tuning, no guard.
_TRACED_BUCKET = 0
# Older Triton: Triton 3.0 compile failures can kill the process; 3.2 is the oldest the kernel was built against.
_MIN_TRITON = (3, 2)
# An autotune that hit an OOM retries the shape after this many seconds, doubling up to the cap.
_SKIP_RETRY_S = 30.0
_SKIP_RETRY_MAX_S = 960.0

_LOCK = threading.RLock()
_STATE: dict = {
    "error": None,
    "verified": {},
    "failed": {},
    "tuned": set(),
    "skip": {},
    "unfit": {},
    "check": {},
    "accuracy": {},
    # (card, check) verdicts adopted from the on-disk cache instead of measured in this process
    "persisted": set(),
    "op": None,
    "op_error": None,
}

try:
    import triton
    import triton.language as tl
except Exception as _exc:  # noqa: BLE001 - no triton (or triton-windows): the tensorwise fallback stays
    triton = None
    tl = None
    _STATE["error"] = f"{type(_exc).__name__}: {_exc}"

_INTERPRET = os.environ.get("TRITON_INTERPRET", "") == "1"


def _version_tuple(version: Any) -> tuple:
    import re
    found = re.match(r"(\d+)\.(\d+)", str(version))
    return (int(found.group(1)), int(found.group(2))) if found else ()


if triton is not None:
    _ver = _version_tuple(getattr(triton, "__version__", ""))
    if not _ver or _ver < _MIN_TRITON:
        _STATE["error"] = (
            f"triton {getattr(triton, '__version__', '?')} < {'.'.join(map(str, _MIN_TRITON))}"
        )
        triton = None
    elif not hasattr(tl, "inline_asm_elementwise"):
        # Without inline PTX the epilogue would round with .ftz (libdevice) or contract to an FMA.
        _STATE["error"] = "triton has no inline_asm_elementwise"
        triton = None

if triton is not None:

    def _configs() -> list:
        # Stages sized to fit the ~100 KB shared memory of sm_89 / sm_12x consumer cards.
        return [
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                num_warps = w,
                num_stages = s,
            )
            for bm, bn, bk, w, s in (
                (128, 128, 128, 8, 3),
                (128, 256, 64, 8, 3),
                (128, 128, 64, 4, 4),
                (64, 128, 128, 4, 4),
                (128, 64, 128, 4, 4),
                (64, 64, 128, 4, 4),
            )
        ]

    def _autotune_kwargs() -> dict:
        import inspect
        try:
            params = inspect.signature(triton.autotune).parameters
        except Exception:  # noqa: BLE001
            return {}
        return {"cache_results": True} if "cache_results" in params else {}

    @triton.jit
    def _mul_rn(x, y):
        # libdevice mul_rn lowers to mul.rn.ftz.f32 (denormals flushed); plain * may contract into an FMA.
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [x, y], dtype = tl.float32, is_pure = True, pack = 1
        )

    @triton.jit
    def _add_rn(x, y):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=f,f,f", [x, y], dtype = tl.float32, is_pure = True, pack = 1
        )

    @triton.jit(do_not_specialize = ["M", "M_BUCKET", "ARCH"])
    def _fp8_rowwise_gemm_kernel(
        a_ptr,
        b_ptr,
        sa_ptr,
        sb_ptr,
        bias_ptr,
        c_ptr,
        M,
        N,
        K,
        M_BUCKET,
        ARCH,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_sa,
        stride_sb,
        stride_cm,
        stride_cn,
        HAS_BIAS: tl.constexpr,
        PIN_RN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BLOCK_M)
        num_n = tl.cdiv(N, BLOCK_N)
        group = GROUP_M * num_n
        first_m = (pid // group) * GROUP_M
        group_rows = min(num_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % group) % group_rows)
        pid_n = (pid % group) // group_rows
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        # Out-of-range rows / columns wrap onto valid ones for the loads; the store masks them.
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        a_ptrs = a_ptr + ram[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = b_ptr + rk[:, None] * stride_bk + rbn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_left = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask = rk[None, :] < k_left, other = 0.0)
            b = tl.load(b_ptrs, mask = rk[:, None] < k_left, other = 0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        sa = tl.load(sa_ptr + rm * stride_sa, mask = rm < M, other = 0.0).to(tl.float32)
        sb = tl.load(sb_ptr + rn * stride_sb, mask = rn < N, other = 0.0).to(tl.float32)
        # _scale_epilogue's order: (acc * a_scale) * b_scale + bias, each rounded to nearest in fp32.
        if PIN_RN:
            out = _mul_rn(_mul_rn(acc, sa[:, None]), sb[None, :])
        else:
            out = acc * sa[:, None] * sb[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + rn, mask = rn < N, other = 0.0).to(tl.float32)
            if PIN_RN:
                out = _add_rn(out, bias[None, :])
            else:
                out = out + bias[None, :]
        c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask = (rm[:, None] < M) & (rn[None, :] < N))

    try:
        from triton.runtime import errors as _runtime_errors
    except Exception:  # noqa: BLE001
        _runtime_errors = None
    try:
        from triton.compiler import errors as _compiler_errors
    except Exception:  # noqa: BLE001
        _compiler_errors = None
    _UNFIT: tuple = ()
    for _m, _names in (
        (_runtime_errors, ("OutOfResources", "PTXASError")),
        (_compiler_errors, ("CompileTimeAssertionFailure",)),
    ):
        _UNFIT += tuple(getattr(_m, n) for n in _names if isinstance(getattr(_m, n, None), type))

    # ARCH (compute capability): Triton's in-memory tuning cache keys on the key args only, not on the device.
    _fp8_rowwise_gemm = triton.autotune(
        configs = _configs(), key = ["M_BUCKET", "N", "K", "ARCH"], **_autotune_kwargs()
    )(_fp8_rowwise_gemm_kernel)


def enabled() -> bool:
    return (os.environ.get(ENV) or "").strip().lower() not in _OFF_WORDS


def available() -> bool:
    return triton is not None


def card_paths(cap: Any) -> tuple:
    """(eager, compiled) for a card of capability ``cap``: the arches it was measured on, nothing else."""
    try:
        cap = tuple(int(c) for c in cap)
    except Exception:  # noqa: BLE001
        return (False, False)
    if cap == (8, 9):
        return (True, False)
    if len(cap) >= 1 and cap[0] == 12:
        return (True, True)
    return (False, False)


def source_digest() -> str:
    """Short sha256 of this file: torch's FX / AOT caches key a triton_op without its kernel source."""
    try:
        with open(__file__, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return "unknown"


def _bucket(m: Any) -> int:
    # A symbolic row count (dynamic-shape trace) gets one fixed key: Inductor tunes a user kernel once per compiled
    # graph anyway, and a SymInt key would re-tune per M wherever Triton's own autotuner runs it. No guard either way.
    if isinstance(m, int):
        b = 1
        while b < m and b < _MAX_BUCKET:
            b *= 2
        return b
    return _TRACED_BUCKET


_ARCH: dict = {}


def _arch(device: Any) -> int:
    """Compute capability as ``major * 10 + minor`` (0 off CUDA), cached per card: primed by the eager self-check, so
    a trace only reads it."""
    if getattr(device, "type", None) != "cuda":
        return 0
    import torch

    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _ARCH:
        major, minor = torch.cuda.get_device_capability(index)
        _ARCH[index] = int(major) * 10 + int(minor)
    return _ARCH[index]


def eligible(
    a: Any, a_scale: Any, b: Any, b_scale: Any, bias: Optional[Any], output_dtype: Any
) -> bool:
    """Operands the kernel handles: 2-D e4m3 a [M, K] row-major, b [K, N] column-major (torchao's weight.t()),
    a_scale [M, 1], b_scale [1, N], optional 1-D bias [N], bf16 / fp16 out, 32-bit offsets."""
    import torch
    try:
        if output_dtype not in (torch.bfloat16, torch.float16):
            return False
        if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
            return False
        if a.dim() != 2 or b.dim() != 2 or a_scale.dim() != 2 or b_scale.dim() != 2:
            return False
        m, k = a.shape
        k2, n = b.shape
        if k != k2 or a.stride(1) != 1 or b.stride(0) != 1:
            return False
        if tuple(a_scale.shape) != (m, 1) or tuple(b_scale.shape) != (1, n):
            return False
        if bias is not None and (bias.dim() != 1 or bias.shape[0] != n or bias.stride(0) != 1):
            return False
        if a_scale.device != a.device or b.device != a.device or b_scale.device != a.device:
            return False
        if bias is not None and bias.device != a.device:
            return False
        if not (m > 0 and n > 0 and k > 0):
            return False
        return (
            (m - 1) * a.stride(0) + k <= _MAX_ELEMS
            and (n - 1) * b.stride(1) + k <= _MAX_ELEMS
            and (m - 1) * a_scale.stride(0) < _MAX_ELEMS
            and (n - 1) * b_scale.stride(1) < _MAX_ELEMS
            and m * n <= _MAX_ELEMS
        )
    except Exception:  # noqa: BLE001 - anything unexpected keeps the tensorwise fallback
        return False


def _launch(
    kernel: Any, a: Any, a_scale: Any, b: Any, b_scale: Any, bias: Optional[Any], output_dtype: Any
) -> Any:
    import torch

    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device = a.device, dtype = output_dtype)

    def grid(meta: dict) -> tuple:
        # -(-x // y): SymInt-safe ceil division
        return (-(-m // meta["BLOCK_M"]) * -(-n // meta["BLOCK_N"]),)

    kernel[grid](
        a,
        b,
        a_scale,
        b_scale,
        bias if bias is not None else a_scale,
        c,
        m,
        n,
        k,
        _bucket(m),
        _arch(a.device),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        a_scale.stride(0),
        b_scale.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS = bias is not None,
        PIN_RN = not _INTERPRET,
    )
    return c


def _key(
    a: Any, a_scale: Any, b: Any, b_scale: Any, bias: Optional[Any], output_dtype: Any
) -> tuple:
    """What Triton's autotuner (sizes in its key + every tensor argument's dtype) and JIT specialization (strides
    equal to 1 / divisible by 16) tell apart: a launch matching a recorded key neither tunes nor compiles."""
    return (
        a.device.index,
        _bucket(int(a.shape[0])),
        int(b.shape[1]),
        int(a.shape[1]),
        output_dtype,
        a.dtype,
        b.dtype,
        a_scale.dtype,
        b_scale.dtype,
        None if bias is None else bias.dtype,
        tuple(
            int(x) == 1 or int(x) % 16 == 0
            for x in (a.stride(0), b.stride(1), a_scale.stride(0), b_scale.stride(1))
        ),
    )


def _capturing(device: Any) -> bool:
    try:
        from .diffusion_fp8_act_quant import capturing
        return bool(capturing(device))
    except Exception:  # noqa: BLE001
        return True


def _is_oom(exc: BaseException) -> bool:
    try:
        from .diffusion_fp8_unit_scale import _is_oom as is_oom
        return bool(is_oom(exc))
    except Exception:  # noqa: BLE001
        return "out of memory" in str(exc).lower()


def _is_fatal(exc: BaseException) -> bool:
    """Errors the fallback must not swallow: OOM (the fallback needs more memory, not less) and Dynamo's own."""
    return type(exc).__module__.startswith("torch._dynamo") or _is_oom(exc)


def eager_gemm(
    a: Any, a_scale: Any, b: Any, b_scale: Any, bias: Optional[Any], output_dtype: Any
) -> Optional[Any]:
    """The fused GEMM for an eager call, or None (tensorwise fallback). A launch failure turns it off for the card."""
    import torch

    index = a.device.index
    if triton is None or _STATE["failed"].get(index):
        return None
    key = _key(a, a_scale, b, b_scale, bias, output_dtype)
    skip = _STATE["skip"].get(key)
    if skip is not None and time.monotonic() < skip[0]:
        return None
    # Autotuning and the self-check sync the host: never first inside a CUDA graph capture, nothing recorded.
    if not verified(index, output_dtype) and (
        _capturing(a.device) or not verify(index, output_dtype)
    ):
        return None
    if key not in _STATE["tuned"] and _capturing(a.device):
        return None
    other_card = index != torch.cuda.current_device()
    try:
        # The autotuner keeps per-call state on itself: one untuned (benchmarking) launch at a time.
        with _LOCK if key not in _STATE["tuned"] else contextlib.nullcontext():
            with torch.cuda.device(a.device) if other_card else contextlib.nullcontext():
                out = _launch(_fp8_rowwise_gemm, a, a_scale, b, b_scale, bias, output_dtype)
    except Exception as exc:  # noqa: BLE001
        if key not in _STATE["tuned"] and _is_oom(exc):
            # Autotuning's benchmark buffers did not fit: this shape keeps the fallback for a while, then tunes again
            # (memory frees between renders), backing off 30 s, 60 s, ... up to 16 min.
            delay = _SKIP_RETRY_S if skip is None else min(2 * skip[1], _SKIP_RETRY_MAX_S)
            _STATE["skip"][key] = (time.monotonic() + delay, delay)
            return None
        if _is_fatal(exc):
            raise
        _STATE["failed"][index] = f"{type(exc).__name__}: {exc}"[:2000]
        return None
    _STATE["tuned"].add(key)
    _STATE["skip"].pop(key, None)
    return out


def _build_op() -> Any:
    if _STATE["op"] is not None or _STATE["op_error"] is not None:
        return _STATE["op"]
    try:
        import torch
        from torch.library import triton_op, wrap_triton

        name = "unsloth_studio::fp8_rowwise_gemm"

        def fp8_rowwise_gemm(a, a_scale, b, b_scale, bias, output_dtype):
            return _launch(
                wrap_triton(_fp8_rowwise_gemm), a, a_scale, b, b_scale, bias, output_dtype
            )

        # Real types, not this module's postponed (string) annotations: the schema is inferred from them.
        fp8_rowwise_gemm.__annotations__ = {
            "a": torch.Tensor,
            "a_scale": torch.Tensor,
            "b": torch.Tensor,
            "b_scale": torch.Tensor,
            "bias": Optional[torch.Tensor],
            "output_dtype": torch.dtype,
            "return": torch.Tensor,
        }
        triton_op(name, mutates_args = {})(fp8_rowwise_gemm)
        _STATE["op"] = torch.ops.unsloth_studio.fp8_rowwise_gemm.default
    except Exception as exc:  # noqa: BLE001 - torch < 2.6 (no triton_op) or a re-registration: compiled keeps the fallback
        _STATE["op_error"] = f"{type(exc).__name__}: {exc}"[:2000]
    return _STATE["op"]


def traced_available() -> bool:
    return triton is not None and _build_op() is not None


def traced_gemm(
    a: Any, a_scale: Any, b: Any, b_scale: Any, bias: Optional[Any], output_dtype: Any
) -> Any:
    """The fused GEMM inside a Dynamo / AOTAutograd trace: a triton_op Inductor inlines (no graph break). Inductor
    benchmarks the configs once, at the compiled graph's first launch, and keeps that config for every row count; an
    OOM there raises out of the compiled graph like any other OOM in it (no fallback can run inside a graph)."""
    return _STATE["op"](a, a_scale, b, b_scale, bias, output_dtype)


# Self-check K: 7 / 13 iterations of BLOCK_K 128 / 64, past every config's num_stages (3-4) prologue, so the
# pipelined loop runs in steady state; not a multiple of 64 (a masked tail tile).
_CHECK_K = 784


def _check_operands(device: str) -> tuple:
    """Integer fp8 operands, power-of-two scales: integer sums (|sum| <= 36 * K < 2**24) are exact in fp32, so a
    correct kernel matches bit for bit. Odd M; K, N multiples of 16 (the production specialization) but of no
    BLOCK_K / BLOCK_N."""
    import torch

    g = torch.Generator(device = "cpu").manual_seed(0)
    m, k, n = 77, _CHECK_K, 144
    a = torch.randint(-6, 7, (m, k), generator = g).to(torch.float32)
    w = torch.randint(-6, 7, (n, k), generator = g).to(torch.float32)
    sa = 2.0 ** torch.randint(-4, 3, (m, 1), generator = g).to(torch.float32)
    sb = 2.0 ** torch.randint(-4, 3, (1, n), generator = g).to(torch.float32)
    bias = torch.randint(-64, 65, (n,), generator = g).to(torch.float32) / 4
    return (
        a.to(torch.float8_e4m3fn).to(device),
        sa.to(device),
        w.to(torch.float8_e4m3fn).to(device).t(),
        sb.to(device),
        bias.to(device),
    )


def verified(index: Any, output_dtype: Any) -> bool:
    """The self-check passed for this card and dtype, and no launch on the card has failed since."""
    return bool(_STATE["verified"].get((index, output_dtype))) and not _STATE["failed"].get(index)


def card_failed(index: Any) -> bool:
    return bool(_STATE["failed"].get(index))


def check_pending(index: Any, output_dtype: Any) -> bool:
    """The self-check has no verdict for this card and dtype yet (it hit an OOM): a later call may still pass it."""
    return (
        triton is not None
        and (index, output_dtype) not in _STATE["verified"]
        and not _STATE["failed"].get(index)
    )


def accuracy_pending(index: Any) -> bool:
    """The accuracy check has no verdict for this bf16-verified card yet (it hit an OOM)."""
    import torch
    return (
        triton is not None and index not in _STATE["accuracy"] and verified(index, torch.bfloat16)
    )


def _on_card(index: int) -> Any:
    import torch
    return torch.cuda.device(index)


def _device(index: int) -> str:
    return f"cuda:{index}"


def verify(index: int, output_dtype: Any) -> bool:
    """Bit-exact self-check of every config, with and without bias, cached per (card, dtype); a pass is persisted
    across processes (``_namespace``). OOM is no verdict: False now, retried next call. Never raises."""
    import torch

    key = (index, output_dtype)
    if key in _STATE["verified"]:
        return verified(index, output_dtype)
    ok = False
    slot = f"verify|{_dtype_name(output_dtype)}"
    if triton is not None and not _STATE["failed"].get(index) and _persisted(index, slot) is True:
        try:
            _arch(torch.device(_device(index)))  # primed as the measured check does
        except Exception:  # noqa: BLE001 - unreadable card: measure instead
            pass
        else:
            _STATE["verified"][key] = True
            _STATE["persisted"].add((index, slot))
            return verified(index, output_dtype)
    if triton is not None and not _STATE["failed"].get(index):
        try:
            with _LOCK, _on_card(index):
                a, sa, b, sb, bias = _check_operands(_device(index))
                _arch(a.device)  # primed here, so a trace only reads it
                ref = (a.float() * sa) @ (b.float() * sb)
                bias = bias.to(torch.bfloat16)
                ok = True
                ran = 0
                for cfg in _fp8_rowwise_gemm.configs:
                    for bb in (None, bias):
                        want = (ref + bb.float() if bb is not None else ref).to(output_dtype)
                        try:
                            got = _launch(_ConfigLaunch(cfg), a, sa, b, sb, bb, output_dtype)
                        except _UNFIT as exc:
                            _STATE["unfit"].setdefault(index, []).append(
                                f"{cfg}: {type(exc).__name__}"
                            )
                            break
                        ran += 1
                        if not torch.equal(got, want):
                            ok = False
                            _STATE["check"][key] = (
                                f"self-check mismatch: {cfg} out={output_dtype} bias={bb is not None}"
                            )
                            break
                    if not ok:
                        break
                if ok and ran == 0:
                    ok = False
                    _STATE["check"][key] = "no config fits this card"
        except Exception as exc:  # noqa: BLE001 - a compile / launch failure keeps the tensorwise fallback
            if _is_oom(exc):
                _STATE["check"][key] = f"OOM during the self-check, retried later: {exc}"[:500]
                return False
            ok = False
            _STATE["check"][key] = f"{type(exc).__name__}: {exc}"[:2000]
    _STATE["verified"][key] = ok
    if ok:  # only a pass is persisted: a failure is measured again by the next process
        _persist(index, slot, True)
    return ok


# Accuracy check shapes (M, K, N): Flux-class token rows at 512^2 / 1024^2, attention / FF-out K.
_ACC_SHAPES = ((512, 3072, 3072), (512, 9216, 3072), (4096, 3072, 3072), (4096, 9216, 3072))
_ACC_RATIO = 1.05
# Absolute floor on the mean relative error vs fp64: 1/64 of half an ulp of the output dtype (bf16 2**-8, fp16
# 2**-11). An fp32-level difference that small changes about 1 in 64 output roundings or fewer, by one ulp; a reduced
# precision (Hopper-style 14-bit) accumulator sits around 2**-12 and fails it.
_ACC_FLOOR = {"bfloat16": 2.0**-14, "float16": 2.0**-17}
_ACC_KINDS = tuple((dt, fast) for dt in ("bfloat16", "float16") for fast in (True, False))


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _kind(dtype: Any, fast: Any) -> str:
    return f"{_dtype_name(dtype)}:{'fast' if fast else 'precise'}"


def _accuracy_operands(device: str, m: int, k: int, n: int, seed: int) -> tuple:
    """Production-like operands: activation rows spread over ~e^+-2 in magnitude, small weights, each quantised to
    e4m3 with its own amax / 448 scale (row-wise a [M, 1], column-wise b [1, N]), bf16 bias."""
    import torch

    g = torch.Generator(device = device).manual_seed(seed)
    x = torch.randn(m, k, generator = g, device = device)
    x.mul_(torch.exp(torch.randn(m, 1, generator = g, device = device)))
    sa = x.abs().amax(1, keepdim = True).clamp_min(1e-12) / 448
    a = (x / sa).to(torch.float8_e4m3fn)
    del x
    w = torch.randn(n, k, generator = g, device = device) * 0.02
    sw = w.abs().amax(1, keepdim = True).clamp_min(1e-12) / 448
    b = (w / sw).to(torch.float8_e4m3fn).t()
    del w
    bias = (torch.randn(n, generator = g, device = device) * 0.5).to(torch.bfloat16)
    return a, sa, b, sw.t().contiguous(), bias


def _fp64_reference(a: Any, sa: Any, b: Any, sb: Any, bias: Any) -> Any:
    import torch

    k = a.shape[1]
    ref = None
    for k0 in range(0, k, 1024):
        part = a[:, k0 : k0 + 1024].double() @ b[k0 : k0 + 1024].double()
        ref = part if ref is None else ref.add_(part)
        del part
    ref.mul_(sa.double()).mul_(sb.double()).add_(bias.double())
    return ref


def _rel_error(out: Any, ref: Any, denom: float) -> float:
    """sum |out - ref| / sum |ref|, in fp64, 512 rows at a time."""
    total = 0.0
    for r0 in range(0, ref.shape[0], 512):
        total += float((out[r0 : r0 + 512].double() - ref[r0 : r0 + 512]).abs().sum())
    return total / denom


def verify_accuracy(index: int) -> bool:
    """Per-op error of the kernel vs the tensorwise path it replaces (``_scaled_mm`` at unit scale, fp32 out where the
    card has it, then ``_scale_epilogue``), both against fp64, on production-like operands at ``_ACC_SHAPES``; every
    shipped config, the worst one counts. Verdict per request kind (output dtype x fast / precise accumulate), each
    needing, at every shape:

    * fp32 level (accumulator + epilogue, before the output rounding; the bit-exact self-check proves the rounding
      itself): kernel <= max(1.05 x the existing path at that accumulate mode, ``_ACC_FLOOR[dtype]``);
    * bf16 output (bf16 kinds): kernel <= 1.05 x the existing path's bf16 output.

    Needs the bf16 self-check; cached per card, and a measured verdict persisted across processes (``_namespace``;
    an error is not); an OOM is no verdict (retried). True when any kind passed."""
    import torch

    if index in _STATE["accuracy"]:
        return any(_STATE["accuracy"][index].get("ok", {}).values())
    if triton is None or not verified(index, torch.bfloat16):
        return False
    info: dict = {"ok": {k: False for k in (_kind(d, f) for d, f in _ACC_KINDS)}}
    try:
        from .diffusion_fp8_unit_scale import _fp32_out_runs, _scale_epilogue
        with _LOCK, _on_card(index):
            dev = _device(index)
            acc_dtype = torch.float32 if _fp32_out_runs(index) else torch.bfloat16
            info["existing_acc"] = _dtype_name(acc_dtype)
            stored = _stored_accuracy(index, info["existing_acc"])
            if stored is not None:
                _STATE["accuracy"][index] = stored
                _STATE["persisted"].add((index, "accuracy"))
                return any(stored["ok"].values())
            one = torch.ones((), device = dev, dtype = torch.float32)
            rows: list = []
            for seed, (m, k, n) in enumerate(_ACC_SHAPES):
                a, sa, b, sb, bias = _accuracy_operands(dev, m, k, n, seed)
                ref = _fp64_reference(a, sa, b, sb, bias)
                denom = float(ref.abs().sum()) or 1.0
                row = {"shape": [m, k, n]}
                for fast in (True, False):
                    acc = torch._scaled_mm(
                        a, b, scale_a = one, scale_b = one, out_dtype = acc_dtype, use_fast_accum = fast
                    )
                    acc = acc[0] if isinstance(acc, tuple) else acc
                    name = "fast" if fast else "precise"
                    row[f"{name}_fp32"] = _rel_error(
                        _scale_epilogue(acc.clone(), sa, sb, bias, torch.float32), ref, denom
                    )
                    row[f"{name}_bf16"] = _rel_error(
                        _scale_epilogue(acc, sa, sb, bias, torch.bfloat16), ref, denom
                    )
                    del acc
                k32 = kbf = 0.0
                for cfg in _fp8_rowwise_gemm.configs:
                    try:
                        got = _launch(_ConfigLaunch(cfg), a, sa, b, sb, bias, torch.float32)
                    except _UNFIT:
                        try:
                            # The served (bf16) variant does not fit either: the autotuner never picks it.
                            _launch(_ConfigLaunch(cfg), a, sa, b, sb, bias, torch.bfloat16)
                        except _UNFIT:
                            continue
                        # Served but unmeasured: 100% error, no verdict can pass (finite, for JSON).
                        k32 = kbf = 1.0
                        row["unmeasured"] = str(cfg)
                        continue
                    k32 = max(k32, _rel_error(got, ref, denom))
                    del got
                    got = _launch(_ConfigLaunch(cfg), a, sa, b, sb, bias, torch.bfloat16)
                    kbf = max(kbf, _rel_error(got, ref, denom))
                    del got
                row["kernel_fp32"], row["kernel_bf16"] = k32, kbf
                rows.append(row)
                del a, b, ref
            info["shapes"] = rows
            for dt, fast in _ACC_KINDS:
                name = "fast" if fast else "precise"
                ok = all(
                    r["kernel_fp32"] <= max(_ACC_RATIO * r[f"{name}_fp32"], _ACC_FLOOR[dt])
                    and (dt != "bfloat16" or r["kernel_bf16"] <= _ACC_RATIO * r[f"{name}_bf16"])
                    for r in rows
                )
                info["ok"][_kind(dt, fast)] = bool(rows) and ok
    except Exception as exc:  # noqa: BLE001 - unknown accuracy: every request kind keeps the tensorwise path
        if _is_oom(exc):
            return False
        info["ok"] = {k: False for k in info["ok"]}
        info["error"] = f"{type(exc).__name__}: {exc}"[:500]
    _STATE["accuracy"][index] = info
    if "error" not in info and info.get(
        "shapes"
    ):  # a measured verdict; an error is measured again next process
        _persist(index, "accuracy", info)
    return any(info["ok"].values())


def _stored_accuracy(index: int, existing_acc: str) -> Optional[dict]:
    """The persisted accuracy verdict for this card, or None (missing, stale, malformed, or measured against another
    existing-path accumulator). Never raises."""
    try:
        got = _persisted(index, "accuracy")
        kinds = {_kind(d, f) for d, f in _ACC_KINDS}
        if (
            not isinstance(got, dict)
            or "error" in got
            or got.get("existing_acc") != existing_acc
            or not isinstance(got.get("ok"), dict)
            or set(got["ok"]) != kinds
            or not all(isinstance(v, bool) for v in got["ok"].values())
            or not isinstance(got.get("shapes"), list)
            or len(got["shapes"]) != len(_ACC_SHAPES)
        ):
            return None
        return dict(got, persisted = True)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- verdicts persisted across processes

# Bump when the self-check / accuracy rule changes in a way the source digest does not see.
_PERSIST_VERSION = 1
_PERSIST: list = []  # [Namespace or None], built once


def _persist_extra() -> dict:
    """Key components beyond the probe fingerprint (card name + UUID + capability, driver, torch, CUDA, Triton,
    UNSLOTH_DIFFUSION_* env) and the source digest of this file and the tensorwise path it is measured against."""
    return {
        "version": _PERSIST_VERSION,
        "check_k": _CHECK_K,
        "acc_shapes": [list(s) for s in _ACC_SHAPES],
        "acc_ratio": _ACC_RATIO,
        "acc_floor": dict(_ACC_FLOOR),
        "configs": [str(c) for c in _fp8_rowwise_gemm.configs] if triton is not None else None,
        "interpret": _INTERPRET,
    }


def _namespace() -> Any:
    if not _PERSIST:
        try:
            from .diffusion_probe_cache import Namespace
            _PERSIST.append(
                Namespace(
                    "fp8_triton_gemm",
                    ("diffusion_fp8_triton_gemm.py", "diffusion_fp8_unit_scale.py"),
                    extra = _persist_extra,
                )
            )
        except Exception:  # noqa: BLE001 - no cache: measured per process
            _PERSIST.append(None)
    return _PERSIST[0]


def _persisted(index: Any, slot: str) -> Any:
    try:
        ns = _namespace()
        return ns.get(f"cuda:{int(index)}", slot) if ns is not None else None
    except Exception:  # noqa: BLE001 - a miss, never an error
        return None


def _persist(index: Any, slot: str, value: Any) -> None:
    try:
        ns = _namespace()
        if ns is not None:
            ns.put(f"cuda:{int(index)}", slot, value)
    except Exception:  # noqa: BLE001 - a cache write never fails a load
        pass


def accurate(index: Any, output_dtype: Any, fast: Any) -> bool:
    """The accuracy check passed for this card and request kind."""
    info = _STATE["accuracy"].get(index)
    return bool(info and info.get("ok", {}).get(_kind(output_dtype, bool(fast))))


class _ConfigLaunch:
    """``kernel[grid](...)`` with one fixed autotune config."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def __getitem__(self, grid: Any) -> Any:
        cfg = self.cfg

        def run(*args: Any, **kwargs: Any) -> Any:
            return _fp8_rowwise_gemm_kernel[grid](
                *args, **kwargs, **cfg.kwargs, num_warps = cfg.num_warps, num_stages = cfg.num_stages
            )

        return run


def describe() -> dict:
    return {
        "available": triton is not None,
        "error": _STATE["error"],
        "verified": {
            f"{i}:{str(d).replace('torch.', '')}": v for (i, d), v in _STATE["verified"].items()
        },
        "failed": dict(_STATE["failed"]),
        "check": {
            f"{i}:{str(d).replace('torch.', '')}": v for (i, d), v in _STATE["check"].items()
        },
        "unfit": {i: list(v) for i, v in _STATE["unfit"].items()},
        "accuracy": {i: dict(v) for i, v in _STATE["accuracy"].items()},
        "persisted": sorted(f"{i}:{slot}" for i, slot in _STATE["persisted"]),
        "skipped_shapes": len(_STATE["skip"]),
        "op_error": _STATE["op_error"],
    }

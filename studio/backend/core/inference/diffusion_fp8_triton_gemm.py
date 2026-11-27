# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused Triton row-wise fp8 GEMM for cards whose torch build has no row-wise ``_scaled_mm`` (the Windows wheels).

Epilogue = ``_scale_epilogue``'s fp32 ops in the same order, each round-to-nearest (no FMA contraction), rounded once
to the output dtype. sm_89 is eager only (up to 1.24x slower than the compiled fallback). ``=0`` keeps the fallback."""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
from typing import Any, Optional

ENV = "UNSLOTH_DIFFUSION_FP8_TRITON_GEMM"
_OFF_WORDS = ("0", "off", "false", "no", "none")
# 32-bit offsets in the kernel: larger operands keep the tensorwise fallback.
_MAX_ELEMS = (1 << 31) - 1
# Autotune key: rows rounded up to a power of two (capped), so a new resolution reuses a tuned config.
_MAX_BUCKET = 8192

_LOCK = threading.RLock()
_STATE: dict = {
    "error": None,
    "verified": {},
    "failed": {},
    "tuned": set(),
    "skip": set(),
    "unfit": {},
    "check": {},
    "precise": {},
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
libdevice = None
if triton is not None:
    try:
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001
        try:
            from triton.language.extra.cuda import libdevice
        except Exception:  # noqa: BLE001
            libdevice = None
    if libdevice is None and not _INTERPRET:
        # Without the round-to-nearest intrinsics the epilogue could contract to an FMA: keep the fallback.
        _STATE["error"] = "triton has no libdevice"
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

    @triton.jit(do_not_specialize = ["M", "M_BUCKET"])
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
            out = libdevice.mul_rn(libdevice.mul_rn(acc, sa[:, None]), sb[None, :])
        else:
            out = acc * sa[:, None] * sb[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + rn, mask = rn < N, other = 0.0).to(tl.float32)
            if PIN_RN:
                out = libdevice.add_rn(out, bias[None, :])
            else:
                out = out + bias[None, :]
        c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask = (rm[:, None] < M) & (rn[None, :] < N))

    _UNFIT: tuple = ()
    for _mod, _names in (
        ("triton.runtime.errors", ("OutOfResources", "PTXASError")),
        ("triton.compiler.errors", ("CompileTimeAssertionFailure",)),
    ):
        try:
            _m = __import__(_mod, fromlist = list(_names))
            _UNFIT += tuple(
                getattr(_m, n) for n in _names if isinstance(getattr(_m, n, None), type)
            )
        except Exception:  # noqa: BLE001
            pass

    _fp8_rowwise_gemm = triton.autotune(
        configs = _configs(), key = ["M_BUCKET", "N", "K"], **_autotune_kwargs()
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


def _bucket(m: Any) -> Any:
    # Plain ints (eager) and SymInts (traced) alike: no math.log2, no comparisons that guard.
    if isinstance(m, int):
        b = 1
        while b < m and b < _MAX_BUCKET:
            b *= 2
        return b
    return m


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
    if key in _STATE["skip"]:
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
            # Autotuning's benchmark buffers did not fit: this shape keeps the fallback, nothing else changes.
            _STATE["skip"].add(key)
            return None
        if _is_fatal(exc):
            raise
        _STATE["failed"][index] = f"{type(exc).__name__}: {exc}"[:2000]
        return None
    _STATE["tuned"].add(key)
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
    """The fused GEMM inside a Dynamo / AOTAutograd trace: a triton_op Inductor inlines (no graph break)."""
    return _STATE["op"](a, a_scale, b, b_scale, bias, output_dtype)


def _check_operands(device: str) -> tuple:
    """Integer fp8 operands, power-of-two scales: exact in fp32, so a correct kernel matches bit for bit. Odd M; K, N
    multiples of 16 (the production specialization) but of no BLOCK_K / BLOCK_N."""
    import torch

    g = torch.Generator(device = "cpu").manual_seed(0)
    m, k, n = 77, 208, 144
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


def check_pending(index: Any, output_dtype: Any) -> bool:
    """The self-check has no verdict for this card and dtype yet (it hit an OOM): a later call may still pass it."""
    return (
        triton is not None
        and (index, output_dtype) not in _STATE["verified"]
        and not _STATE["failed"].get(index)
    )


def precise_pending(index: Any) -> bool:
    """The accumulator check has no verdict for this bf16-verified card yet (it hit an OOM)."""
    import torch
    return triton is not None and index not in _STATE["precise"] and verified(index, torch.bfloat16)


def verify(index: int, output_dtype: Any) -> bool:
    """Bit-exact self-check of every config, with and without bias, cached per (card, dtype). OOM is no verdict:
    False now, retried next call. Never raises."""
    import torch

    key = (index, output_dtype)
    if key in _STATE["verified"]:
        return verified(index, output_dtype)
    ok = False
    if triton is not None and not _STATE["failed"].get(index):
        try:
            with _LOCK, torch.cuda.device(index):
                a, sa, b, sb, bias = _check_operands(f"cuda:{index}")
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
    return ok


def _precise_operands(device: str) -> tuple:
    """Rows whose exact product is 0: K/2 random e4m3 products, then the same products negated. What survives is the
    accumulator's rounding residue: about 2**-22 of the largest partial sum for an fp32 accumulator, 2**-11 for a
    14-bit one (Hopper-style fast accumulate), emulated on CPU."""
    import torch

    g = torch.Generator(device = "cpu").manual_seed(1)
    m, half, n = 64, 2048, 128
    x = (torch.randn(m, half, generator = g) * 64).to(torch.float8_e4m3fn)
    w = (torch.randn(n, half, generator = g) * 64).to(torch.float8_e4m3fn)
    a = torch.cat([x, x], 1)
    wb = torch.cat([w, (-w.float()).to(torch.float8_e4m3fn)], 1)
    partial = float((x.float() @ w.float().t()).abs().max())
    return a.to(device), wb.to(device).t(), partial


# Residue bound at K = 4096: 2**6 above an fp32 accumulator's, 2**5 below a 14-bit one's.
_PRECISE_RESIDUE = 2.0**-16


def verify_precise(index: int) -> bool:
    """Whether card ``index``'s fp8 tensor-core accumulation is as precise as cuBLAS's precise accumulate, so a
    precise-accumulate GEMM may take the kernel: every shipped config, on cancelling rows, against the tensorwise
    ``_scaled_mm`` at ``use_fast_accum=False``. Needs the bf16 self-check; cached per card; an OOM is no verdict."""
    import torch

    if index in _STATE["precise"]:
        return bool(_STATE["precise"][index].get("ok"))
    if triton is None or not verified(index, torch.bfloat16):
        return False
    info: dict = {"ok": False}
    try:
        with _LOCK, torch.cuda.device(index):
            dev = f"cuda:{index}"
            a, b, partial = _precise_operands(dev)
            sa = torch.ones((a.shape[0], 1), device = dev)
            sb = torch.ones((1, b.shape[1]), device = dev)
            one = torch.ones((), device = dev)
            for fast in (False, True):
                acc = torch._scaled_mm(
                    a, b, scale_a = one, scale_b = one, out_dtype = torch.float32, use_fast_accum = fast
                )
                acc = acc[0] if isinstance(acc, tuple) else acc
                info["cublas_fast" if fast else "cublas_precise"] = float(acc.abs().max()) / partial
            worst = 0.0
            for cfg in _fp8_rowwise_gemm.configs:
                try:
                    got = _launch(_ConfigLaunch(cfg), a, sa, b, sb, None, torch.bfloat16)
                except _UNFIT:
                    continue
                worst = max(worst, float(got.float().abs().max()) / partial)
            info["kernel"] = worst
            info["ok"] = worst <= max(4 * info["cublas_precise"], _PRECISE_RESIDUE)
    except Exception as exc:  # noqa: BLE001 - unknown accumulator: precise requests keep cuBLAS
        if _is_oom(exc):
            return False
        info["error"] = f"{type(exc).__name__}: {exc}"[:500]
    _STATE["precise"][index] = info
    return bool(info["ok"])


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
        "precise": {i: dict(v) for i, v in _STATE["precise"].items()},
        "op_error": _STATE["op_error"],
    }

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FlashInfer's per-call dispatch work, hoisted into a cache. Every private import is HERE.

Fenced three ways: an EXACT version allowlist (a symbol that moves in 0.6.7 is not a bug in 0.6.7),
every private import in ONE try so there is no partial fast path, and a runtime per-device
``verify()``, because a symbol that still exists but means something else passes a version check.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

# The only versions whose private layout was read and verified. Exact, never a minimum.
_SUPPORTED = ("0.6.6",)

NVFP4_FAST_DISPATCH_ENV = "UNSLOTH_NVFP4_FAST_DISPATCH"

# Weights only: an activation view would be unbounded.
_TRANSPOSE_CACHE_MAX = 4096

_LOCK = threading.Lock()
_AVAILABLE: Optional[tuple] = None
_VERIFIED: dict[int, tuple] = {}
_QUANT_FN: dict[int, tuple] = {}
_GEMM_PLAN: dict = {}
_TRANSPOSED: dict = {}


def fast_dispatch_env() -> str:
    """``auto`` (default), ``0`` (never), ``1`` (skip the version check, nothing else)."""
    raw = os.environ.get(NVFP4_FAST_DISPATCH_ENV, "").strip().lower()
    return raw if raw in ("auto", "0", "1") else "auto"


def _probe() -> tuple:
    """``(ok, reason)``: is this FlashInfer one whose private dispatch layout is known here."""
    env = fast_dispatch_env()
    if env == "0":
        return False, f"{NVFP4_FAST_DISPATCH_ENV}=0"
    try:
        import flashinfer

        version = str(getattr(flashinfer, "__version__", "unknown"))
        if env != "1" and version not in _SUPPORTED:
            return False, (
                f"flashinfer {version} is not in the verified set "
                f"{', '.join(_SUPPORTED)}; using the public API"
            )
        from flashinfer.autotuner import AutoTuner  # noqa: F401
        from flashinfer.fp4_quantization import get_fp4_quantization_module  # noqa: F401
        from flashinfer.gemm.gemm_base import (  # noqa: F401
            DEFAULT_WORKSPACE_SIZE,
            _MM_FP4_TUNING_CONFIG_128x4,
            _get_cache_buf,
            get_cutlass_fp4_gemm_module,
        )
        from flashinfer.utils import device_support_pdl, get_compute_capability  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any failure means the public API, never a raise
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"
    return True, f"flashinfer {version} private dispatch symbols present"


def available() -> tuple:
    """``(ok, reason)``, memoised. ``reset()`` re-reads the env and the symbols."""
    global _AVAILABLE
    cached = _AVAILABLE
    if cached is not None:
        return cached
    result = _probe()
    with _LOCK:
        _AVAILABLE = result
    return result


def enabled(device: Any) -> bool:
    """Whether the fast path may be used on ``device``: false until ``verify(device)`` passed."""
    from .diffusion_nvfp4_ops import _device_index

    record = _VERIFIED.get(_device_index(device))
    return bool(record and record[0])


def verify(device: Any) -> tuple:
    """Run both paths on ``device`` and require bit identity. ``(ok, reason)``, once per device."""
    from .diffusion_nvfp4_ops import _device_index

    index = _device_index(device)
    cached = _VERIFIED.get(index)
    if cached is not None:
        return cached
    ok, reason = available()
    record = _run_verify(device) if ok else (False, reason)
    with _LOCK:
        _VERIFIED[index] = record
    return record


def _run_verify(device: Any) -> tuple:
    import torch
    try:
        import flashinfer
        with torch.cuda.device(device):
            x = torch.randn(256, 512, device = device, dtype = torch.bfloat16) * 0.05
            w = torch.randn(128, 512, device = device, dtype = torch.bfloat16) * 0.02
            a_gsf = _global_scale(x)
            w_gsf = _global_scale(w)
            alpha = (1.0 / (a_gsf * w_gsf)).float()

            want_q, want_sf = flashinfer.nvfp4_quantize(x, a_gsf, do_shuffle = False)
            wq, w_sf = flashinfer.nvfp4_quantize(w, w_gsf, do_shuffle = False)
            got_q, got_sf = _fast_quantize(x, a_gsf, force = True)
            if got_q is None:
                return False, "the cached quantiser refused this device"
            if not (torch.equal(want_q, got_q) and torch.equal(want_sf, got_sf)):
                return False, "the cached quantiser is not bit-identical to nvfp4_quantize"

            want = torch.empty(256, 128, device = device, dtype = torch.bfloat16)
            flashinfer.mm_fp4(
                want_q,
                wq.T,
                want_sf,
                w_sf.T,
                alpha,
                torch.bfloat16,
                out = want,
                backend = "cutlass",
            )
            got = torch.empty(256, 128, device = device, dtype = torch.bfloat16)
            plan = gemm_plan(want_q, wq.T, want_sf, w_sf.T, alpha, got, 128, "cutlass", force = True)
            if plan is None:
                return False, "no cached GEMM plan could be built"
            runner, tactic, workspace = plan
            runner(
                inputs = [
                    want_q,
                    wq.T,
                    want_sf,
                    w_sf.T,
                    alpha,
                    torch.bfloat16,
                    got,
                    16,
                    True,
                    workspace,
                ],
                tactic = tactic,
            )
            torch.cuda.synchronize(device)
            if not torch.equal(want, got):
                return False, "the cached GEMM is not bit-identical to mm_fp4"
    except Exception as exc:  # noqa: BLE001 - a verify that raises is a verify that failed
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"
    return True, "verified bit-identical against the public API"


def _global_scale(t: Any):
    return (6.0 * 448.0 / t.float().abs().amax().clamp(min = 1e-8)).reshape(1).to(t.device)


def quant_fn(device: Any, *, force: bool = False):
    """The bound pybind quantiser and its ``enable_pdl`` flag for ``device``, or None."""
    from .diffusion_nvfp4_ops import _device_index

    index = _device_index(device)
    got = _QUANT_FN.get(index)
    if got is not None:
        return got
    if not (force or enabled(device)):
        return None
    try:
        return _build_quant_fn(device, index)
    except Exception:  # noqa: BLE001 - a quantiser that will not bind means the public API
        return None


def _build_quant_fn(device: Any, index: int):
    import torch

    from flashinfer.fp4_quantization import get_fp4_quantization_module
    from flashinfer.utils import device_support_pdl, get_compute_capability

    with torch.cuda.device(device):
        major, minor = get_compute_capability(device)
        got = (
            get_fp4_quantization_module(f"{major}{minor}").fp4_quantize_sm100,
            device_support_pdl(device),
        )
    with _LOCK:
        return _QUANT_FN.setdefault(index, got)


def _fast_quantize(
    x: Any,
    global_sf: Any,
    *,
    force: bool = False,
):
    """``nvfp4_quantize(do_shuffle = False)`` minus the per-call device queries."""
    if not x.is_contiguous():
        return None, None
    got = quant_fn(x.device, force = force)
    if got is None:
        return None, None
    fn, pdl = got
    # Exactly the arguments nvfp4_quantize(do_shuffle = False, sfLayout = 128x4) passes through.
    xq, sf = fn(x, global_sf, 16, False, True, False, pdl)
    return xq, sf.reshape((-1, x.shape[-1] // 16))


def transposed(t: Any):
    """A kept ``.T`` of a weight buffer, keyed on ``(data_ptr, shape)`` against a stale view."""
    key = (t.data_ptr(), tuple(t.shape))
    view = _TRANSPOSED.get(key)
    if view is not None:
        return view
    with _LOCK:
        if len(_TRANSPOSED) >= _TRANSPOSE_CACHE_MAX:
            _TRANSPOSED.clear()
        return _TRANSPOSED.setdefault(key, t.T)


def gemm_plan(
    xq: Any,
    wq_t: Any,
    x_sf: Any,
    w_sf_t: Any,
    alpha: Any,
    out: Any,
    n: int,
    backend: str,
    *,
    force: bool = False,
):
    """``(runner, tactic, workspace)``, or None (use ``mm_fp4``) including for a COLD key under
    capture, where ``choose_one`` may profile and bake tactics into the graph."""
    from .diffusion_nvfp4_ops import _device_index, _is_capturing

    device = xq.device
    key = (xq.shape[0], xq.shape[1], n, backend, _device_index(device))
    got = _GEMM_PLAN.get(key)
    if got is not None:
        return got
    if backend != "cutlass" or not (force or enabled(device)) or _is_capturing():
        return None
    try:
        return _build_plan(key, device, [xq, wq_t, x_sf, w_sf_t, alpha, out])
    except Exception:  # noqa: BLE001 - a plan that will not build means the public API
        return None


def _build_plan(key: tuple, device: Any, operands: list):
    import torch

    from flashinfer.autotuner import AutoTuner
    from flashinfer.gemm.gemm_base import (
        DEFAULT_WORKSPACE_SIZE,
        _MM_FP4_TUNING_CONFIG_128x4,
        _get_cache_buf,
        get_cutlass_fp4_gemm_module,
    )
    from flashinfer.utils import get_compute_capability

    # The guard spans the whole builder: both allocation and profiling hit the CURRENT device.
    with torch.cuda.device(device):
        workspace = _get_cache_buf("mm_fp4_workspace", DEFAULT_WORKSPACE_SIZE, device)
        major, minor = get_compute_capability(device)
        runner = get_cutlass_fp4_gemm_module(major, minor).cutlass_fp4_gemm_runner()
        xq, wq_t, x_sf, w_sf_t, alpha, out = operands
        inputs = [xq, wq_t, x_sf, w_sf_t, alpha, torch.bfloat16, out, 16, True, workspace]
        # Probe FlashInfer's OWN tactic cache, so this runs the tactic the public path would.
        _, tactic = AutoTuner.get().choose_one(
            "fp4_gemm", [runner], _MM_FP4_TUNING_CONFIG_128x4, inputs
        )
    plan = (runner, tactic, workspace)
    with _LOCK:
        return _GEMM_PLAN.setdefault(key, plan)


def reset() -> None:
    """Forget everything. Called on unload: the transposed views would pin a freed model."""
    global _AVAILABLE
    with _LOCK:
        _AVAILABLE = None
        _VERIFIED.clear()
        _QUANT_FN.clear()
        _GEMM_PLAN.clear()
        _TRANSPOSED.clear()


def describe() -> dict:
    """What the caches hold right now. For the preflight record and for tests."""
    return {
        "available": available()[0],
        "reason": available()[1],
        "verified_devices": sorted(index for index, rec in _VERIFIED.items() if rec[0]),
        "plans": len(_GEMM_PLAN),
        "transposed": len(_TRANSPOSED),
    }

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FlashInfer's per-call dispatch work, hoisted into a cache. Every private import is HERE.

Why this exists. On a clean B200, block-compiled z-image 512px at 100 percent 4-bit spends 0.618 s
of wall clock on 0.125 s of GPU work: 20 percent busy, against 51 percent for the same model in
fp8 with the same 0.126 s of GPU work. Identical device work, 2.5x the wall clock. The GEMM is not
slow; the host cannot issue it. Per call ``flashinfer.mm_fp4`` costs 61 to 73 us of HOST time to
launch a kernel that runs in 5 to 65 us, and a render makes 2151 of those calls.

The cost is work that is CONSTANT for a given shape and is rebuilt every call:
``mm_fp4`` constructs a four-entry dict of runner-factory lambdas, builds a fresh cutlass runner and
re-hashes its inputs for an ``AutoTuner.choose_one`` cache probe; ``fp4_quantize`` re-runs
``device_support_pdl`` and ``get_compute_capability`` and looks up its module by an f-string. So
this caches ``(runner, tactic, workspace)`` per (M, K, N, backend, device) and the quantiser's bound
pybind function per device, and calls straight through: 61.8 -> 18.1 us of host time per call,
bit-identical, and 0.6309 -> 0.4074 s (1.55x) end to end at z-image 512, 0.6984 -> 0.4802 (1.45x) at
1024, un-graphed.

**This reaches into FlashInfer's private internals, so it is fenced three ways.**

1. An EXACT version allowlist. Not a minimum: a private symbol that moves in 0.6.7 is not a bug in
   0.6.7. ``UNSLOTH_NVFP4_FAST_DISPATCH=1`` skips the version check for someone deliberately
   testing a new release; it does not skip anything else.
2. Every private import in ONE try, in ONE file. A missing or moved symbol returns "unavailable"
   with the exception in the reason, and the public API runs instead. There is no partial fast path.
3. A runtime ``verify()``: quantise and GEMM both ways on this exact device and require
   ``torch.equal``. It runs once per device from the preflight, off the request path, and no fast
   call is made anywhere until it has passed. A symbol that still exists but means something else
   is exactly the failure a version check cannot see.

The fast path is off until all three pass, and every failure is silent to the render and readable
in the preflight record.

Capture safety: ``gemm_plan`` returns None for a cold key while the stream is capturing, because
``choose_one`` may PROFILE, and profiling launches inside a capture bake candidate tactics into the
graph. The prewarm makes every key warm before any capture, so this is a fallback rather than a
path. Nothing here caches a result tensor -- only a runner, a tactic, a workspace and two transposed
VIEWS of weight buffers that live for the model's lifetime.

Retire this file when FlashInfer exposes a plan/run split of its own; the cache is worth upstreaming
rather than keeping.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

# The only versions whose private layout was read and verified. Exact, never a minimum.
_SUPPORTED = ("0.6.6",)

NVFP4_FAST_DISPATCH_ENV = "UNSLOTH_NVFP4_FAST_DISPATCH"

# ``.T`` is a fresh view object every time it is evaluated, and the two of them measured about 12 us
# of the public path's host cost. Weight buffers are built once and live for the model's lifetime,
# so their transposes can simply be kept. Keyed on data_ptr AND shape, so a reallocated buffer
# cannot silently return a stale view. Weights only: an activation view would be unbounded.
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
        # One try for the whole private surface. A partial fast path is not a thing that exists.
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
    """Whether the fast path may be used on ``device``. A dict lookup, on the request path.

    False until ``verify(device)`` has passed there, which the preflight does once per device.
    """
    from .diffusion_nvfp4_ops import _device_index

    record = _VERIFIED.get(_device_index(device))
    return bool(record and record[0])


def verify(device: Any) -> tuple:
    """Run both paths on ``device`` and require bit identity. ``(ok, reason)``, once per device.

    Off the request path (the preflight calls it), because it quantises, GEMMs twice and compares.
    A version allowlist answers "is this the layout I read"; only this answers "does it still mean
    what it meant".
    """
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


# ── the cached pieces ─────────────────────────────────────────────────────────────────────────


def quant_fn(device: Any, *, force: bool = False):
    """The bound pybind quantiser plus its ``enable_pdl`` flag for ``device``, or None.

    What is cached is what ``fp4_quantize`` recomputes per call: the compute capability, the PDL
    support query and the module lookup keyed by an f-string. ``force`` is for ``verify`` alone,
    which has to run the fast path BEFORE the fast path is allowed to be used.
    """
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
    """``flashinfer.nvfp4_quantize(do_shuffle = False)`` without the per-call device queries.

    Returns ``(None, None)`` when the fast path does not apply, so the caller runs the public one.
    A column-major input is refused rather than transposed: ``fp4_quantize`` handles that case with
    a transpose dance this deliberately does not reimplement.
    """
    if not x.is_contiguous():
        return None, None
    got = quant_fn(x.device, force = force)
    if got is None:
        return None, None
    fn, pdl = got
    # Exactly the arguments nvfp4_quantize(do_shuffle = False, sfLayout = 128x4) passes through:
    # sf_vec_size 16, sf_use_ue8m0 False, is_sf_swizzled_layout True, is_sf_8x4_layout False.
    xq, sf = fn(x, global_sf, 16, False, True, False, pdl)
    return xq, sf.reshape((-1, x.shape[-1] // 16))


def transposed(t: Any):
    """A kept ``.T`` of a weight buffer, keyed on ``(data_ptr, shape)``.

    Bounded and cleared wholesale rather than evicted one at a time: the population is the model's
    weight buffers, so it is a few hundred entries and a clear is a warm-up, not a stall. Holding
    the view holds a reference to the base tensor, which is a weight that is alive anyway.
    """
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
    """``(runner, tactic, workspace)`` for this shape, or None to use the public ``mm_fp4``.

    None when the fast path is off, when the plan cannot be built, and -- the one that matters --
    when the key is COLD and the stream is capturing: ``choose_one`` may profile, and a profiling
    launch inside a capture is baked into the graph forever.
    """
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

    # The guard spans the whole builder, not just the probe: _get_cache_buf allocates the workspace
    # on the CURRENT device, and choose_one launches every candidate tactic while profiling.
    with torch.cuda.device(device):
        workspace = _get_cache_buf("mm_fp4_workspace", DEFAULT_WORKSPACE_SIZE, device)
        major, minor = get_compute_capability(device)
        runner = get_cutlass_fp4_gemm_module(major, minor).cutlass_fp4_gemm_runner()
        xq, wq_t, x_sf, w_sf_t, alpha, out = operands
        inputs = [xq, wq_t, x_sf, w_sf_t, alpha, torch.bfloat16, out, 16, True, workspace]
        # One probe of FlashInfer's OWN tactic cache, so the fast path runs exactly the tactic the
        # public path would have chosen for this shape rather than a guess.
        _, tactic = AutoTuner.get().choose_one(
            "fp4_gemm", [runner], _MM_FP4_TUNING_CONFIG_128x4, inputs
        )
    plan = (runner, tactic, workspace)
    with _LOCK:
        return _GEMM_PLAN.setdefault(key, plan)


def reset() -> None:
    """Forget everything: the availability probe, the per-device verdicts and the three caches.

    Called on unload with the CUDA graph pool and the barriers. The transposed views hold weight
    buffers, so keeping them past an unload would pin a freed model's memory.
    """
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

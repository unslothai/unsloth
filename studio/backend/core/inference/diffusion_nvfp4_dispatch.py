# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FlashInfer dispatch: every private import lives here, behind an exact-version allowlist in one
all-or-nothing try, plus per-device verify()"""

from __future__ import annotations

import dataclasses
import os
import threading
import weakref
from functools import partial
from typing import Any, Optional

# Exact versions whose private layout was read, never a minimum.
_SUPPORTED = ("0.6.6", "0.7.3")

NVFP4_FAST_DISPATCH_ENV = "UNSLOTH_NVFP4_FAST_DISPATCH"

# Weights only: an activation view would be unbounded.
_TRANSPOSE_CACHE_MAX = 4096

_LOCK = threading.Lock()
_AVAILABLE: Optional[tuple] = None
_VERIFIED: dict[int, tuple] = {}
_QUANT_FN: dict[int, tuple] = {}
_GEMM_PLAN: dict = {}
_TRANSPOSED: dict = {}
# device index -> (runner, workspace, tactic count): everything in a plan except the tactic.
_RUNNERS: dict[int, tuple] = {}
# One tune at a time: profiling launches must not overlap, and 0.6.6's tuning mode is process-global.
_TUNE_LOCK = threading.Lock()
_BUCKET_CONFIGS: dict = {}
# Own op name: a stock mm_fp4 entry (measured at some other M) must never answer a bucket.
BUCKET_OP = "fp4_gemm.studio_m_bucket"


def fast_dispatch_env() -> str:
    """``auto`` (default), ``0`` (never), ``1`` (skip the version check, nothing else)."""
    raw = os.environ.get(NVFP4_FAST_DISPATCH_ENV, "").strip().lower()
    return raw if raw in ("auto", "0", "1") else "auto"


def _probe() -> tuple:
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
    global _AVAILABLE
    cached = _AVAILABLE
    if cached is not None:
        return cached
    result = _probe()
    with _LOCK:
        _AVAILABLE = result
    return result


def enabled(device: Any) -> bool:
    from .diffusion_nvfp4_ops import _device_index
    record = _VERIFIED.get(_device_index(device))
    return bool(record and record[0])


def verify(device: Any) -> tuple:
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
            # An explicit CPU generator: up to torch 2.11 a default-generator draw raises while another thread
            # records, and a CPU draw moved to the device needs no CUDA generator (absent on a spoofed / CPU-only torch).
            gen = torch.Generator().manual_seed(0)
            x = torch.randn(256, 512, dtype = torch.bfloat16, generator = gen).to(device) * 0.05
            w = torch.randn(128, 512, dtype = torch.bfloat16, generator = gen).to(device) * 0.02
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
            from .diffusion_cuda_graph import synchronize_outside_capture

            synchronize_outside_capture(device)
            if not torch.equal(want, got):
                return False, "the cached GEMM is not bit-identical to mm_fp4"
    except Exception as exc:  # noqa: BLE001 - a raise is a failed verify
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"
    return True, "verified bit-identical against the public API"


def _global_scale(t: Any):
    return (6.0 * 448.0 / t.float().abs().amax().clamp(min = 1e-8)).reshape(1).to(t.device)


def quant_fn(device: Any, *, force: bool = False):
    from .diffusion_nvfp4_ops import _device_index

    # Gate before the cache read: verify() forces this entry BEFORE it knows it is bit-identical.
    if not (force or enabled(device)):
        return None
    index = _device_index(device)
    got = _QUANT_FN.get(index)
    if got is not None:
        return got
    try:
        return _build_quant_fn(device, index)
    except Exception:  # noqa: BLE001 - no quantiser means the public API
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
    if not x.is_contiguous():
        return None, None
    got = quant_fn(x.device, force = force)
    if got is None:
        return None, None
    fn, pdl = got
    # By keyword: 0.7 put ``is_global_scale_inversed`` before ``enable_pdl``; positional pdl inverts the scale.
    xq, sf = fn(
        x,
        global_sf,
        sf_vec_size = 16,
        sf_use_ue8m0 = False,
        is_sf_swizzled_layout = True,
        is_sf_8x4_layout = False,
        enable_pdl = pdl,
    )
    return xq, sf.reshape((-1, x.shape[-1] // 16))


def _drop_transposed(key: int, ref: Any) -> None:
    """Weakref callback; lock-free because a collection can land on a thread already holding the lock."""
    entry = _TRANSPOSED.get(key)
    if entry is not None and entry[0] is ref:
        _TRANSPOSED.pop(key, None)


def transposed(t: Any):
    """A kept ``.T`` released with the weight; views ``t.detach()`` since ``t.T`` would pin ``t`` via ``_base``."""
    key = id(t)
    stamp = (t.data_ptr(), tuple(t.shape))
    entry = _TRANSPOSED.get(key)
    if entry is not None and entry[0]() is t and entry[1] == stamp:
        return entry[2]
    with _LOCK:
        entry = _TRANSPOSED.get(key)
        if entry is not None and entry[0]() is t and entry[1] == stamp:
            return entry[2]
        if len(_TRANSPOSED) >= _TRANSPOSE_CACHE_MAX:
            _TRANSPOSED.clear()
        view = t.detach().T
        # partial, never a closure over ``t``: a callback holding the weight would pin it forever.
        _TRANSPOSED[key] = (weakref.ref(t, partial(_drop_transposed, key)), stamp, view)
        return view


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
    """``(runner, tactic, workspace)``, or None (use ``mm_fp4``), also for a COLD key under capture.
    Keyed by M bucket; ``force`` (verify) keeps the exact-M AutoTuner lookup and caches nothing."""
    from . import diffusion_nvfp4_tactics as tactics
    from .diffusion_nvfp4_ops import _device_index, _is_capturing

    device = xq.device
    bucketed = tactics.bucketed_enabled() and not force
    m = int(xq.shape[0])
    key = (tactics.m_bucket(m) if bucketed else m, xq.shape[1], n, backend, _device_index(device))
    got = _GEMM_PLAN.get(key)
    if got is not None:
        return got
    if backend != "cutlass" or not (force or enabled(device)):
        return None
    if _is_capturing():
        # Never a build or a tune here; a bucket already tuned (this process or on disk) is fine.
        return _warm_plan(key, device) if bucketed else None
    operands = [xq, wq_t, x_sf, w_sf_t, alpha, out]
    try:
        if bucketed:
            return _build_bucket_plan(key, device, operands)
        return _build_plan(key, device, operands, store = not force)
    except Exception:  # noqa: BLE001 - no plan means the public API
        return None


def _runner(device: Any, index: int) -> tuple:
    got = _RUNNERS.get(index)
    if got is not None:
        return got
    import torch

    from flashinfer.gemm.gemm_base import (
        DEFAULT_WORKSPACE_SIZE,
        _get_cache_buf,
        get_cutlass_fp4_gemm_module,
    )
    from flashinfer.utils import get_compute_capability

    with torch.cuda.device(device):
        workspace = _get_cache_buf("mm_fp4_workspace", DEFAULT_WORKSPACE_SIZE, device)
        major, minor = get_compute_capability(device)
        runner = get_cutlass_fp4_gemm_module(major, minor).cutlass_fp4_gemm_runner()
        try:
            # Static per module (an index into the cutlass config table); it ignores both arguments.
            count = len(runner.get_valid_tactics(None, None))
        except Exception:  # noqa: BLE001 - unknown table size: tactics stay in memory only
            count = None
    with _LOCK:
        return _RUNNERS.setdefault(index, (runner, workspace, count))


def _warm_plan(key: tuple, device: Any):
    """A plan from cached parts only (no allocation, no profiling), for use inside a capture."""
    from . import diffusion_nvfp4_tactics as tactics

    bucket, half_k, n, backend, index = key
    parts = _RUNNERS.get(index)
    if parts is None:
        return None
    runner, workspace, count = parts
    try:
        env = tactics.device_environment(device, count, backend)
        tactic = tactics.lookup(env, half_k * 2, n, bucket)
    except Exception:  # noqa: BLE001 - a cold bucket under capture takes the public API
        return None
    if tactic is None:
        return None
    plan = (runner, tactic, workspace)
    with _LOCK:
        return _GEMM_PLAN.setdefault(key, plan)


def _build_bucket_plan(key: tuple, device: Any, operands: list):
    from . import diffusion_nvfp4_tactics as tactics

    bucket, half_k, n, backend, index = key
    runner, workspace, count = _runner(device, index)
    env = tactics.device_environment(device, count, backend)
    k = half_k * 2
    tactic = tactics.lookup(env, k, n, bucket)
    if tactic is None:
        with _TUNE_LOCK:
            tactic = tactics.lookup(env, k, n, bucket)
            if tactic is None:
                try:
                    tuned = _tune_bucket(device, runner, workspace, operands, bucket)
                except Exception:  # noqa: BLE001 - the default tactic, this process only
                    tuned = None
                # -1 is FlashInfer's "nothing ran" (every tactic failed, or an OOM): never persisted.
                persist = tuned is not None and tuned >= 0
                tactic = tactics.record(
                    env, k, n, bucket, -1 if tuned is None else tuned, persist = persist
                )
    plan = (runner, tactic, workspace)
    with _LOCK:
        return _GEMM_PLAN.setdefault(key, plan)


def _constant_bucket(bucket: int, _m: int) -> int:
    return bucket


def bucket_config(base: Any, bucket: int):
    """``base`` profiling exactly one M (``bucket``) and keying every M to it; memoised per bucket so
    FlashInfer's nearest-profile cache sees a stable spec."""
    cache_key = (id(base), int(bucket))
    got = _BUCKET_CONFIGS.get(cache_key)
    if got is not None and got[0] is base:
        return got[1]
    spec = base.dynamic_tensor_specs[0]
    pinned = dataclasses.replace(
        spec,
        gen_tuning_buckets = (int(bucket),),
        map_to_tuning_buckets = partial(_constant_bucket, int(bucket)),
    )
    config = dataclasses.replace(base, dynamic_tensor_specs = (pinned,))
    with _LOCK:
        return _BUCKET_CONFIGS.setdefault(cache_key, (base, config))[1]


def _tune_bucket(device: Any, runner: Any, workspace: Any, operands: list, bucket: int) -> int:
    """FlashInfer's own profiler (its timing, failure handling and blocklist), at M = ``bucket`` only."""
    import time

    import flashinfer
    import torch

    from flashinfer.autotuner import AutoTuner
    from flashinfer.gemm.gemm_base import _MM_FP4_TUNING_CONFIG_128x4

    config = bucket_config(_MM_FP4_TUNING_CONFIG_128x4, bucket)
    xq, wq_t, x_sf, w_sf_t, alpha, out = operands
    inputs = [xq, wq_t, x_sf, w_sf_t, alpha, torch.bfloat16, out, 16, True, workspace]
    start = time.perf_counter()
    with torch.cuda.device(device):
        with flashinfer.autotune(True):
            _, tactic = AutoTuner.get().choose_one(BUCKET_OP, [runner], config, inputs)
    from .diffusion_nvfp4_tactics import logger

    logger.info(
        "nvfp4.tactics: tuned mm_fp4 K=%d N=%d at M bucket %d -> tactic %s (%.2f s)",
        int(xq.shape[1]) * 2,
        int(out.shape[1]),
        int(bucket),
        tactic,
        time.perf_counter() - start,
    )
    return int(tactic)


def _build_plan(
    key: tuple,
    device: Any,
    operands: list,
    *,
    store: bool = True,
):
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
        _, tactic = AutoTuner.get().choose_one(
            "fp4_gemm", [runner], _MM_FP4_TUNING_CONFIG_128x4, inputs
        )
    plan = (runner, tactic, workspace)
    if not store:
        return plan
    with _LOCK:
        return _GEMM_PLAN.setdefault(key, plan)


def forget_availability() -> None:
    """Re-probe on the next ``available()``."""
    global _AVAILABLE
    with _LOCK:
        _AVAILABLE = None


def reset() -> None:
    """Called on unload: the transposed views would pin a freed model."""
    global _AVAILABLE
    with _LOCK:
        _AVAILABLE = None
        _VERIFIED.clear()
        _QUANT_FN.clear()
        _GEMM_PLAN.clear()
        _TRANSPOSED.clear()
        _RUNNERS.clear()


def describe() -> dict:
    from . import diffusion_nvfp4_tactics as tactics
    return {
        "available": available()[0],
        "reason": available()[1],
        "verified_devices": sorted(index for index, rec in _VERIFIED.items() if rec[0]),
        "plans": len(_GEMM_PLAN),
        "transposed": len(_TRANSPOSED),
        "tactics": tactics.describe(),
    }

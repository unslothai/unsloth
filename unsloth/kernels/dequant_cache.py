# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""VRAM-budgeted cache of dequantized frozen NF4 LoRA base weights.

Under Unsloth gradient checkpointing a frozen NF4 linear is dequantized three times per micro-batch
(no-grad forward, checkpoint replay, backward). The decode is deterministic, so keeping the 16-bit copy
of as many weights as fit in otherwise idle VRAM is exact and removes those decodes.

UNSLOTH_NF4_DEQUANT_CACHE:
  auto (default) / 1   on, sized from free memory at every optimizer step (below); off while vLLM is loaded
                       or several ranks share a GPU.
  0 / off              off: every call site runs its original code.
  <float>              on, at most that many GiB per device (also with vLLM loaded).
Budget (the zoo moe_expert_cache design), measured at the safe point after each optimizer step, when the
step's activations are freed: room = free + reserved-unused - 2 * transient - margin, with transient the
largest step peak above the resident memory seen so far and margin max(1 GiB, 5% of the card). Nothing is
cached before the first optimizer step, so inference and evaluation-only runs are untouched.
Safety: weights are only admitted while room is left; every 8th hit re-reads free memory and drops the oldest
entries while it is under the margin (and admits nothing more until the next safe point); an out-of-memory
error while admitting drops the cache for the rest of the process. Entries die with their packed weight
(storage weak reference, also swept after every gc collection), on any in-place change to it (version
counter), at the end of SFTTrainer.train(), in for_inference() and before saving, and are never served on
another stream or during CUDA graph capture.
Only the LoRA kernels' own matmul sites read the cache, so the dense tensor never reaches other code. Eager
they take the cached tensor; traced (per-block compile) they call an opaque op that does the lookup and the
same GEMM, so the compiled graph does not depend on what is cached and nothing recompiles."""

import collections
import os
import sys
import threading
import time
from typing import List, Optional

import torch

__all__ = [
    "STATS",
    "cache_mode",
    "clear_dequant_cache",
    "cached_count",
    "dequantize",
    "traced_route",
    "nf4_matmul",
    "nf4_addmm_",
    "nf4_lora_first",
]

STATS = collections.Counter()

_OFF = ("0", "off", "false", "no", "none")
_ON = ("", "1", "auto", "on", "true", "yes")
MARGIN_MIN_BYTES = 1 << 30
MARGIN_FRAC = 0.05
TRANSIENT_SLACK = 2.0
_CHECK_EVERY = 8


def parse_mode(raw):
    """UNSLOTH_NF4_DEQUANT_CACHE -> None (off), "auto", or a byte cap (float)."""
    raw = (raw if raw is not None else "auto").strip().lower()
    if raw in _OFF:
        return None
    if raw in _ON:
        return "auto"
    try:
        gib = float(raw)
    except ValueError:
        return None
    return gib * 2**30 if gib > 0 else None


_MODE = parse_mode(os.environ.get("UNSLOTH_NF4_DEQUANT_CACHE"))


def cache_mode():
    return _MODE


_lock = threading.RLock()
# key (packed storage pointer, dtype) -> [storage weakref, version, absmax pointer, stream, dense [N, K]]
_CACHE = collections.OrderedDict()
_BUDGET = {}  # device index -> bytes the cache may hold (set at safe points only)
_TRANSIENT = {}  # device index -> largest step peak above the resident memory seen so far
_SEEN = set()  # device indices with eligible lookups
_CALLS = collections.Counter()  # device index -> lookups since the last memory check
_HELD = collections.Counter()  # device index -> bytes cached
_DISABLED = False  # set for good after an out-of-memory error while anything was cached


def _storage_ref(W):
    try:
        from torch.multiprocessing.reductions import StorageWeakRef
        return StorageWeakRef(W.untyped_storage())
    except Exception:
        return None


def _raw_stream(index):
    try:
        return torch._C._cuda_getCurrentRawStream(index)
    except Exception:
        return None  # not a CUDA device (the bookkeeping unit tests run on CPU tensors)


def _nbytes(dense):
    return dense.numel() * dense.element_size()


def _cached_bytes(index = None):
    if index is not None:
        return _HELD[index]
    return sum(_HELD.values())


def _put(key, entry):
    old = _CACHE.pop(key, None)
    if old is not None:
        _HELD[old[4].device.index] -= _nbytes(old[4])
    _CACHE[key] = entry
    _HELD[entry[4].device.index] += _nbytes(entry[4])


def _pop(key):
    entry = _CACHE.pop(key, None)
    if entry is not None:
        _HELD[entry[4].device.index] -= _nbytes(entry[4])
    return entry


def cached_count():
    return len(_CACHE)


def clear_dequant_cache(oom = False):
    """Drop every cached weight (and the budgets: they are recomputed at the next optimizer step)."""
    with _lock:
        _CACHE.clear()
        _HELD.clear()
        _BUDGET.clear()
        if oom:
            STATS["oom_clears"] += 1
        STATS["bytes"] = 0


def _margin(total):
    return max(MARGIN_MIN_BYTES, int(MARGIN_FRAC * total))


def _usable(index):
    """(memory that can still be allocated on the device, total) or None."""
    try:
        free, total = torch.cuda.mem_get_info(index)
        stats = torch.cuda.memory_stats(index)
        unused = stats.get("reserved_bytes.all.current", 0) - stats.get(
            "allocated_bytes.all.current", 0
        )
        return free + max(unused, 0), total, stats
    except Exception:
        return None


def _auto_allowed():
    if "vllm" in sys.modules or os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "1":
        return False
    try:
        local = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if local > max(torch.cuda.device_count(), 1):
            return False
    except Exception:
        pass
    return True


def _evict(index, target_bytes):
    """Drop the oldest entries of `index` until it holds at most target_bytes."""
    held = _cached_bytes(index)
    for key in [k for k, e in _CACHE.items() if e[4].device.index == index]:
        if held <= target_bytes:
            break
        held -= _nbytes(_pop(key)[4])
        STATS["evictions"] += 1
    STATS["bytes"] = _cached_bytes()


def _on_optimizer_step(optimizer, args, kwargs):
    """Safe point: the step's activations are freed; size every device's budget from what is free now."""
    if _MODE is None or _DISABLED or not _SEEN:
        return
    if _MODE == "auto" and not _auto_allowed():
        if _CACHE:
            clear_dequant_cache()
        return
    with _lock:
        for key in [k for k, e in _CACHE.items() if e[0].expired()]:
            _pop(key)
            STATS["stale"] += 1
        for index in list(_SEEN):
            measured = _usable(index)
            if measured is None:
                continue
            usable, total, stats = measured
            allocated = stats.get("allocated_bytes.all.current", 0)
            peak = stats.get("allocated_bytes.all.peak", 0)
            transient = max(_TRANSIENT.get(index, 0), peak - allocated)
            _TRANSIENT[index] = transient
            held = _cached_bytes(index)
            room = usable - int(TRANSIENT_SLACK * transient) - _margin(total)
            budget = max(held + room, 0)
            if _MODE != "auto":
                budget = min(budget, int(_MODE))
            if held > budget:
                _evict(index, budget)
            _BUDGET[index] = budget
            STATS["safe_points"] += 1


try:
    from torch.optim.optimizer import register_optimizer_step_post_hook
    register_optimizer_step_post_hook(_on_optimizer_step)
except Exception:
    pass


def _sweep_after_gc(phase, info):
    # `del model; gc.collect(); torch.cuda.empty_cache()` must also free the dense copies of its weights.
    if phase != "stop" or not _CACHE:
        return
    with _lock:
        for key in [k for k, e in _CACHE.items() if e[0].expired()]:
            _pop(key)
            STATS["stale"] += 1
        STATS["bytes"] = _cached_bytes()


import gc as _gc  # noqa: E402

_gc.callbacks.append(_sweep_after_gc)


def _guard(index):
    """Every _CHECK_EVERY hits: drop the oldest cached weights while free memory is under the margin."""
    _CALLS[index] += 1
    if _CALLS[index] < _CHECK_EVERY:
        return
    _CALLS[index] = 0
    try:
        # The driver's free memory alone is cheap; only when it is short count the allocator's unused cache too.
        free, total = torch.cuda.mem_get_info(index)
        if free >= _margin(total):
            return
    except Exception:
        pass
    measured = _usable(index)
    if measured is None:
        return
    usable, total, _ = measured
    margin = _margin(total)
    if usable >= margin:
        return
    with _lock:
        held = _cached_bytes(index)
        _evict(index, max(held - (margin - usable), 0))
        # Do not refill what was just dropped before the next safe point re-measures.
        _BUDGET[index] = min(_BUDGET.get(index, 0), _cached_bytes(index))


def _eligible(W, quant_state_dtype):
    return (
        _MODE is not None
        and not _DISABLED
        and W.is_cuda
        and not W.requires_grad
        and quant_state_dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not torch.is_inference_mode_enabled()
        and not torch.cuda.is_current_stream_capturing()
    )


def _lookup(W, absmax, decode, dtype):
    """Dense [N, K] for the packed W (or its .t() view), from the cache or decode(); decode(fresh) returns the
    oriented result of fast_dequantize. Returns the oriented tensor like fast_dequantize."""
    index = W.device.index
    if not _CACHE and _BUDGET.get(index, 0) <= 0:
        _SEEN.add(index)
        return decode(False)  # nothing cached and no budget (before the first optimizer step)
    # Hit path kept short: it runs for every frozen linear, three times per micro-batch.
    entry = _CACHE.get((W.data_ptr(), dtype))
    if entry is not None:
        _guard(index)
        if entry[1] == W._version and entry[3] == _raw_stream(index) and not entry[0].expired():
            STATS["hits"] += 1
            dense = entry[4]
            return dense.t() if W.shape[0] == 1 else dense
    return _miss(W, absmax, decode, dtype, index)


def _miss(W, absmax, decode, dtype, index):
    global _DISABLED
    _SEEN.add(index)
    key = (W.data_ptr(), dtype)
    stream = _raw_stream(index)
    with _lock:
        entry = _CACHE.get(key)
        if entry is not None and (
            entry[0].expired() or entry[1] != W._version or entry[2] != absmax.data_ptr()
        ):
            _pop(key)  # reloaded / requantized / edited in place: decode again
            STATS["stale"] += 1
        STATS["misses"] += 1
        budget = _BUDGET.get(index, 0)
        if budget <= 0 or (entry is not None and key in _CACHE):
            return decode(False)  # no budget, or cached for another stream
        need = W.numel() * 2 * (torch.finfo(dtype).bits // 8)
        if _cached_bytes(index) + need > budget:
            STATS["declined"] += 1
            return decode(False)
        measured = _usable(index)
        ref = _storage_ref(W)
        if ref is None or measured is None or measured[0] - need < _margin(measured[1]):
            STATS["declined"] += 1
            return decode(False)
        try:
            out = decode(True)
        except torch.cuda.OutOfMemoryError:
            out = None
        if out is None:
            _DISABLED = True
            clear_dequant_cache(oom = True)
            torch.cuda.empty_cache()
            return decode(False)
        _put(key, [ref, W._version, absmax.data_ptr(), stream, out.t() if W.shape[0] == 1 else out])
        STATS["admits"] += 1
        STATS["bytes"] = _cached_bytes()
        return out


def _is_bnb_nf4(quant_state):
    # bitsandbytes' QuantState class (not the legacy list form, not FP8 / INT4 / packed states), NF4 only.
    cls = type(quant_state)
    return (
        cls.__name__ == "QuantState"
        and cls.__module__.startswith("bitsandbytes")
        and getattr(quant_state, "quant_type", None) == "nf4"
    )


def dequantize(
    W,
    quant_state,
    use_global_buffer = False,
    dtype = None,
):
    """fast_dequantize(W, quant_state, ...) for the LoRA kernels' matmul sites, from the cache when it
    holds W. Eager only; callers must not write into the result."""
    from .utils import fast_dequantize

    if (
        _MODE is None
        or quant_state is None
        or torch.compiler.is_compiling()
        or not _is_bnb_nf4(quant_state)
        or not _eligible(W, quant_state.dtype)
    ):
        return fast_dequantize(W, quant_state, use_global_buffer = use_global_buffer, dtype = dtype)

    def decode(fresh):
        if fresh:
            return fast_dequantize(W, quant_state, use_global_buffer = False, dtype = dtype)
        return fast_dequantize(W, quant_state, use_global_buffer = use_global_buffer, dtype = dtype)

    return _lookup(W, quant_state.absmax, decode, quant_state.dtype)


# ----------------------------------------------------------------------------------------------------------
# Traced callers: opaque ops doing the lookup and the GEMM, so the dense tensor never enters the graph (Inductor
# may reuse a dead buffer, which must never be a cached weight) and the graph is the same whatever is cached.
# Registered as plain CUDA kernels (no autograd wrapper: they are only called with grad off): their dispatch costs
# less than the custom_op they replace.


def _parts(quant_state):
    from .utils import _unpack_quant_state
    absmax, shape, dtype, blocksize, code, code2, absmax2, offset, blocksize2 = _unpack_quant_state(
        quant_state
    )
    return absmax, code2, absmax2, offset, code, int(blocksize), int(blocksize2), list(shape), dtype


def _dense_from_parts(W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype):
    def decode(fresh):
        from .nf4 import dequantize_nf4
        out = dequantize_nf4(
            W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype
        )
        return out.t() if W.shape[0] == 1 else out

    if (
        _MODE is None
        or _DISABLED
        or torch.is_inference_mode_enabled()
        or torch.cuda.is_current_stream_capturing()
    ):
        return decode(False)
    return _lookup(W, absmax, decode, dtype)


def traced_route(W, quant_state):
    """True when a traced (Dynamo) caller should use the nf4_* ops below for this weight. Decided from
    the module constant and the quant state type only, so it adds no guards that change during a run."""
    if not _HAS_OPS or _MODE is None or quant_state is None or not _is_bnb_nf4(quant_state):
        return False
    from . import utils as _U

    if not getattr(_U, "_USE_NF4_KERNELS", False):
        return False
    if quant_state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if getattr(quant_state, "state2", None) is not None and not hasattr(
        quant_state.state2, "absmax"
    ):
        return False
    return (
        torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
        and isinstance(W, torch.Tensor)
        and W.dtype == torch.uint8
    )


def _out_features(W, shape):
    # X @ dense(W).t(): dense(W) is [N, K] for the packed weight and [K, N] for its .t() view.
    return shape[1] if W.shape[0] == 1 else shape[0]


def _matmul_impl(
    X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype, XA, Bt, alpha
):
    Wd = _dense_from_parts(
        W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype
    )
    if XA is None:
        return torch.matmul(X, Wd.t())
    if _lora_first_addmm is None:
        _bind_lora_first()
    return _lora_first_addmm(X, Wd.t(), XA, Bt, alpha)


def _addmm_impl(
    out, X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype
):
    Wd = _dense_from_parts(
        W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype
    )
    out.addmm_(X, Wd.t())


_HAS_OPS = False
_lora_first_addmm = None
try:
    if hasattr(torch.ops.unsloth, "nf4_cached_matmul") and hasattr(
        torch.ops.unsloth, "nf4_cached_addmm_"
    ):
        _HAS_OPS = (
            True  # a re-executed module: the first definition (and its library) stays registered
        )
    else:
        _LIB = torch.library.Library("unsloth", "FRAGMENT")
        _PARTS_SCHEMA = (
            "Tensor W, Tensor absmax, Tensor? code2, Tensor? absmax2, Tensor? offset, Tensor code, "
            "int blocksize, int blocksize2, int[] shape, ScalarType dtype"
        )
        _LIB.define(
            f"nf4_cached_matmul(Tensor X, {_PARTS_SCHEMA}, Tensor? XA, Tensor? Bt, float alpha) -> Tensor"
        )
        _LIB.define(f"nf4_cached_addmm_(Tensor(a!) out, Tensor X, {_PARTS_SCHEMA}) -> ()")
        _LIB.impl("nf4_cached_matmul", _matmul_impl, "CUDA")
        _LIB.impl("nf4_cached_addmm_", _addmm_impl, "CUDA")

        @torch.library.register_fake("unsloth::nf4_cached_matmul", lib = _LIB)
        def _(
            X,
            W,
            absmax,
            code2,
            absmax2,
            offset,
            code,
            blocksize,
            blocksize2,
            shape,
            dtype,
            XA,
            Bt,
            alpha,
        ):
            return X.new_empty((X.shape[0], _out_features(W, shape)))

        @torch.library.register_fake("unsloth::nf4_cached_addmm_", lib = _LIB)
        def _(out, X, W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype):
            return None

        _HAS_OPS = True
except Exception:
    _HAS_OPS = False


def _bind_lora_first():
    global _lora_first_addmm
    if _lora_first_addmm is None:
        from .utils import _lora_first_addmm as fn
        _lora_first_addmm = fn


def nf4_matmul(
    X,
    W,
    quant_state,
    XA = None,
    Bt = None,
    alpha = 1.0,
):
    """Traced X @ fast_dequantize(W).t() (XA None) or _lora_first_addmm(X, dense.t(), XA, Bt, alpha)."""
    if X.dim() != 2:
        shape = X.shape
        out = nf4_matmul(X.reshape(-1, shape[-1]), W, quant_state, XA, Bt, alpha)
        return out.view(*shape[:-1], out.shape[-1])
    return torch.ops.unsloth.nf4_cached_matmul(X, W, *_parts(quant_state), XA, Bt, float(alpha))


def nf4_addmm_(out, X, W, quant_state):
    """Traced out.addmm_(X, fast_dequantize(W).t()); returns out."""
    torch.ops.unsloth.nf4_cached_addmm_(out, X, W, *_parts(quant_state))
    return out


def nf4_lora_first(X, W, quant_state, XA, Bt, alpha):
    return nf4_matmul(X, W, quant_state, XA, Bt, alpha)

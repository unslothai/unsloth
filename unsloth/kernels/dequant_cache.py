# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""VRAM-budgeted cache of dequantized frozen NF4 LoRA base weights.

Under Unsloth gradient checkpointing a frozen NF4 linear is dequantized three times per micro-batch
(no-grad forward, checkpoint replay, backward). The decode is deterministic, so keeping the 16-bit copy
of as many weights as fit in otherwise idle VRAM is exact and removes those decodes.

UNSLOTH_NF4_DEQUANT_CACHE:
  auto (default) / 1   on, sized from free memory at every optimizer step (below), holding weights for one
                       optimizer step at a time; off while vLLM is loaded, several ranks share a GPU or
                       another process uses it.
  0 / off              off: every call site runs its original code.
  <float>              on, at most that many GiB per device, kept across optimizer steps (also with vLLM
                       loaded or another process on the GPU).
Budget (the zoo moe_expert_cache design), measured at the safe point after each optimizer step, when the
step's activations are freed: room = free + reserved-unused - 2 * transient - margin, with transient the
largest step peak above the resident memory seen so far and margin max(1 GiB, 5% of the card). Free memory is
also capped by torch.cuda.set_per_process_memory_fraction. Nothing is cached before the first optimizer step, and
nothing outside transformers' Trainer.train(), so inference, evaluation-only runs and custom loops are untouched.
Safety: the budget only covers steps no larger than those already measured, so the cache is dropped (until the
next safe point) when it is needed for anything else:
  * a matmul with more rows (tokens) than any measured before the last safe point (a longer batch, a larger
    evaluation or generation batch): its transient was never measured;
  * Trainer.evaluate() / predict(), for_inference() and saving; the end of Trainer.train() also returns it to
    the driver;
  * auto mode only: another process holding more than the margin on the device, or growing by more than it.
Weights are only admitted while room is left; every 8th hit re-reads free memory and drops the oldest entries
while it is under the margin (and admits nothing more until the next safe point); an out-of-memory error while
admitting drops the cache for the rest of the process. Entries die with their packed weight
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
_ROWS_SEEN = {}  # device index -> most matmul rows of any lookup before the last safe point (budget covers these)
_ROWS_NEW = {}  # device index -> most matmul rows since the last safe point
_FOREIGN = {}  # device index -> memory other processes held at the first safe point
_FRACTIONS = {}  # device index -> set_per_process_memory_fraction (torch without the getter)
_ACTIVE = 0  # Trainer.train() calls running: nothing is cached outside them


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


def clear_dequant_cache(oom = False, release = False):
    """Drop every cached weight (and the budgets: they are recomputed at the next optimizer step).
    release=True also returns the freed blocks to the driver, for other processes."""
    with _lock:
        held = bool(_CACHE)
        _CACHE.clear()
        _HELD.clear()
        _BUDGET.clear()
        if oom:
            STATS["oom_clears"] += 1
        STATS["bytes"] = 0
    if release and held:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _drop_device(index, reason):
    """Drop index's entries and admit nothing more on it until the next safe point."""
    with _lock:
        if _CACHE:
            _evict(index, 0)
        _BUDGET[index] = 0
        STATS[reason] += 1


def _fraction(index):
    """The caching allocator's set_per_process_memory_fraction for the device (1.0 when unset)."""
    get = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if get is not None:
        try:
            return float(get(index))
        except Exception:
            pass
    return _FRACTIONS.get(index, 1.0)


def _record_fraction_setter():
    # torch without get_per_process_memory_fraction: remember what is set from now on.
    if hasattr(torch.cuda, "get_per_process_memory_fraction"):
        return
    setter = getattr(torch.cuda, "set_per_process_memory_fraction", None)
    if setter is None or getattr(setter, "_unsloth_dq_recorded", False):
        return

    def set_per_process_memory_fraction(fraction, device = None):
        out = setter(fraction, device)
        try:
            index = torch.cuda._utils._get_device_index(device, optional = True)
            _FRACTIONS[index] = float(fraction)
        except Exception:
            pass
        return out

    set_per_process_memory_fraction._unsloth_dq_recorded = True
    torch.cuda.set_per_process_memory_fraction = set_per_process_memory_fraction
    try:
        torch.cuda.memory.set_per_process_memory_fraction = set_per_process_memory_fraction
    except Exception:
        pass


_record_fraction_setter()


def _margin(total):
    return max(MARGIN_MIN_BYTES, int(MARGIN_FRAC * total))


def _usable(index):
    """(memory that can still be allocated on the device, total) or None."""
    try:
        free, total = torch.cuda.mem_get_info(index)
        stats = torch.cuda.memory_stats(index)
        allocated = stats.get("allocated_bytes.all.current", 0)
        unused = stats.get("reserved_bytes.all.current", 0) - allocated
        usable = free + max(unused, 0)
        fraction = _fraction(index)
        if fraction < 1.0:
            # The allocator refuses to reserve past fraction * total whatever the driver has free.
            usable = min(usable, int(fraction * total) - allocated)
        return usable, total, stats
    except Exception:
        return None


def _foreign_bytes(index):
    """Device memory held outside this process's caching allocator (other processes, CUDA contexts)."""
    try:
        free, total = torch.cuda.mem_get_info(index)
        return max(total - free - torch.cuda.memory_reserved(index), 0)
    except Exception:
        return 0


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


def _evict(
    index,
    target_bytes,
    stat = "evictions",
):
    """Drop the oldest entries of `index` until it holds at most target_bytes."""
    held = _cached_bytes(index)
    for key in [k for k, e in _CACHE.items() if e[4].device.index == index]:
        if held <= target_bytes:
            break
        held -= _nbytes(_pop(key)[4])
        STATS[stat] += 1
    STATS["bytes"] = _cached_bytes()


def _on_optimizer_step(optimizer, args, kwargs):
    """Safe point: the step's activations are freed; size every device's budget from what is free now."""
    if _MODE is None or _DISABLED or not _SEEN:
        return
    if (_MODE == "auto" and not _auto_allowed()) or not _ACTIVE:
        if _CACHE:
            clear_dequant_cache(release = True)
        return
    with _lock:
        for key in [k for k, e in _CACHE.items() if e[0].expired()]:
            _pop(key)
            STATS["stale"] += 1
        for index in list(_SEEN):
            # The budget below covers steps up to the largest matmul seen so far, no larger.
            _ROWS_SEEN[index] = max(_ROWS_SEEN.get(index, 0), _ROWS_NEW.pop(index, 0))
            measured = _usable(index)
            if measured is None:
                continue
            usable, total, stats = measured
            if _MODE == "auto":
                # Another process on this GPU (a kept chat model, a server) may grow into what the cache
                # takes: stay off while one holds more than the margin, or once one grew by more than it.
                foreign = _foreign_bytes(index)
                first = _FOREIGN.setdefault(index, foreign)
                if foreign > _margin(total) or foreign - first > _margin(total):
                    if _cached_bytes(index):
                        _evict(index, 0)
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    _BUDGET[index] = 0
                    STATS["foreign_off"] += 1
                    continue
            allocated = stats.get("allocated_bytes.all.current", 0)
            peak = stats.get("allocated_bytes.all.peak", 0)
            transient = max(_TRANSIENT.get(index, 0), peak - allocated)
            _TRANSIENT[index] = transient
            held = _cached_bytes(index)
            if _MODE == "auto" and held:
                # Auto keeps weights for one optimizer step only: callbacks, evaluation, saving and any other
                # code between steps see the memory the base code leaves free. The next step's first
                # micro-batch decodes and admits again; its other passes and micro-batches hit.
                _evict(index, 0, "step_drops")
                measured = _usable(index)
                if measured is None:
                    continue
                usable, total, _ = measured
                held = 0
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
        # Under a per-process memory fraction the driver's free memory does not bound the allocator.
        free, total = torch.cuda.mem_get_info(index)
        if free >= _margin(total) and _fraction(index) >= 1.0:
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


def _lookup(
    W,
    absmax,
    decode,
    dtype,
    rows = 0,
):
    """Dense [N, K] for the packed W (or its .t() view), from the cache or decode(); decode(fresh) returns the
    oriented result of fast_dequantize. Returns the oriented tensor like fast_dequantize. rows: the matmul's
    row count (tokens), which sizes the step's activations."""
    index = W.device.index
    if rows > _ROWS_NEW.get(index, 0):
        _ROWS_NEW[index] = rows
    if not _CACHE and _BUDGET.get(index, 0) <= 0:
        _SEEN.add(index)
        return decode(False)  # nothing cached and no budget (before the first optimizer step)
    if rows > _ROWS_SEEN.get(index, 0):
        # More tokens than any step the budget was measured on: its activations may need what the cache holds.
        _drop_device(index, "rows_drops")
        return decode(False)
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


def rows_of(X):
    """Rows of the matmul X @ W.t() (tokens): what the activations of the step scale with."""
    return X.numel() // X.shape[-1] if X.dim() > 1 and X.shape[-1] else X.numel()


def dequantize(
    W,
    quant_state,
    use_global_buffer = False,
    dtype = None,
    rows = 0,
):
    """fast_dequantize(W, quant_state, ...) for the LoRA kernels' matmul sites, from the cache when it
    holds W. Eager only; callers must not write into the result. rows: rows_of(X) of the matmul."""
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

    return _lookup(W, quant_state.absmax, decode, quant_state.dtype, rows)


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


def _dense_from_parts(
    W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype, rows
):
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
    return _lookup(W, absmax, decode, dtype, rows)


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
        W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype, X.shape[0]
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
        W, absmax, code2, absmax2, offset, code, blocksize, blocksize2, shape, dtype, X.shape[0]
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


# ----------------------------------------------------------------------------------------------------------
# Trainer scope: the budget is only set between on_train_begin and the end of Trainer.train(); evaluation and
# prediction run without the cache (their batches were never measured), and the end of training returns the
# cache's memory to the driver. A default callback carries the scope (wrappers other code puts around
# Trainer.train cannot bypass it); the train() wrapper also ends it when training raises.


def _end_train():
    global _ACTIVE
    _ACTIVE = 0
    clear_dequant_cache(release = True)


def _make_callback():
    from transformers.trainer_callback import TrainerCallback
    class UnslothDequantCacheScope(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            global _ACTIVE
            _ACTIVE = 1

        def on_step_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                clear_dequant_cache()  # evaluation follows this step: no budget until the next safe point

        def on_train_end(self, args, state, control, **kwargs):
            _end_train()

    return UnslothDequantCacheScope


def install_trainer_scope():
    try:
        import transformers.trainer as T
    except Exception:
        return False
    import functools

    Trainer = T.Trainer
    callbacks = getattr(T, "DEFAULT_CALLBACKS", None)
    if isinstance(callbacks, list) and not any(
        getattr(c, "__name__", "") == "UnslothDequantCacheScope" for c in callbacks
    ):
        callbacks.append(_make_callback())

    train = Trainer.train
    if not getattr(train, "_unsloth_dq_scope", False):

        @functools.wraps(train)
        def scoped_train(self, *args, **kwargs):
            try:
                return train(self, *args, **kwargs)
            finally:
                _end_train()

        scoped_train._unsloth_dq_scope = True
        Trainer.train = scoped_train

    def uncached(method):
        @functools.wraps(method)
        def run(self, *args, **kwargs):
            clear_dequant_cache()
            return method(self, *args, **kwargs)

        run._unsloth_dq_scope = True
        return run

    for name in ("evaluate", "predict"):
        method = getattr(Trainer, name, None)
        if method is not None and not getattr(method, "_unsloth_dq_scope", False):
            setattr(Trainer, name, uncached(method))
    return True


if _MODE is not None:
    try:
        install_trainer_scope()
    except Exception:
        pass

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static-buffer CUDA-graph capture of ONE denoiser forward, for the diffusion backends.

Each denoiser module gets a ``GraphedForward`` in its INSTANCE ``__dict__`` under ``forward``.
Not a proxy: the pipelines read ``.dtype`` / ``.config`` / ``.cache_context`` off the transformer.
Not a hook: hooks move ``nn.Module._call_impl`` off its fast path and offload owns the hook list.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import os
import threading
import traceback
import weakref
import types
from functools import update_wrapper
from typing import Any, Optional

from .diffusion_bg_compile import capture_suppressed as _bg_capture_suppressed
from .diffusion_bg_compile import eager_forced as _bg_eager_forced

CUDA_GRAPH_DISABLE_ENV = "UNSLOTH_DISABLE_CUDA_GRAPH"
# Master switch for every diffusion CUDA graph (whole-forward and per-block): "0" turns them all off.
CUDA_GRAPHS_ENV = "UNSLOTH_DIFFUSION_CUDA_GRAPHS"
# =0: an offloaded denoiser never records its whole step with the copies inside (per-block graphs still may).
OFFLOAD_CUDA_GRAPH_ENV = "UNSLOTH_DIFFUSION_OFFLOAD_CUDA_GRAPH"
# =0: a block-streamed whole-step graph records its copies into the graph pool instead of the prefetcher's slot ring.
STEP_SLOTS_ENV = "UNSLOTH_DIFFUSION_STEP_GRAPH_SLOTS"

# Per module. Legitimate second keys exist, but each costs a pool-sized slice of VRAM.
MAX_GRAPHS_PER_MODULE = 4

# Three per the PyTorch docs. The warmup forces lazy allocations OUTSIDE the capture.
WARMUP_ITERS = 3

_TRUE_TOKENS = ("1", "true", "yes", "on")


def _torch():
    """The torch module, imported lazily so this file imports on a torch-free host."""
    import torch
    return torch


# Comma list of family names (or "all") armed whatever the family declares: for measuring a family before it opts in.
FORCE_FAMILIES_ENV = "UNSLOTH_DIFFUSION_CUDA_GRAPH_FAMILIES"
# A module attribute: zero-arg callable returning hashable per-call state that the forward branches on.
GRAPH_KEY_EXTRA_ATTR = "_unsloth_graph_key_extra"


def _family_forced(family: Any) -> bool:
    raw = (os.environ.get(FORCE_FAMILIES_ENV) or "").strip().lower()
    if not raw:
        return False
    names = {n.strip() for n in raw.split(",") if n.strip()}
    return "all" in names or str(getattr(family, "name", "")).lower() in names


def offload_graphs_enabled() -> bool:
    return (os.environ.get(OFFLOAD_CUDA_GRAPH_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def cuda_graph_disabled() -> bool:
    """Whether an env kill switch is set. Read before any torch or pipe inspection."""
    if os.environ.get(CUDA_GRAPHS_ENV, "").strip().lower() in ("0", "off", "false", "no"):
        return True
    return os.environ.get(CUDA_GRAPH_DISABLE_ENV, "").strip().lower() in _TRUE_TOKENS


def _disabled_reason() -> str:
    if os.environ.get(CUDA_GRAPHS_ENV, "").strip().lower() in ("0", "off", "false", "no"):
        return f"disabled by {CUDA_GRAPHS_ENV}=0"
    return f"disabled by {CUDA_GRAPH_DISABLE_ENV}"


# ``torch.cuda.graph`` begins its capture in CUDA's default global mode, which prohibits the
# "potentially unsafe" calls -- cudaEventQuery among them -- from EVERY thread for as long as it
# records. The denoise progress poller queries a CUDA event ten times a second and the denoiser
# captures its graphs during the first steps of a render, so the two really do overlap.
#
# A flag read before the query would be a TOCTOU: the reader can see "no capture", the capture can
# begin, and the query then lands inside it and invalidates it. So the lock has to span the CUDA
# call itself, not just the depth. Capture raises the depth UNDER the lock before it enters
# ``torch.cuda.graph``, and a querier holds the same lock ACROSS its query, which leaves only two
# orderings and both are safe: either the querier holds the lock and capture entry waits the few
# microseconds a query takes, or capture got there first and the querier sees a non-zero depth and
# skips. The depth is not a flag because nothing says two pipelines cannot be capturing at once,
# and neither holds the lock while it records, so captures do not serialise against each other.
_CAPTURE_LOCK = threading.Lock()
_CAPTURE_DEPTH = 0


@contextlib.contextmanager
def _capturing():
    """Mark a capture as recording for the duration of the block.

    The depth MUST be raised before ``torch.cuda.graph`` is entered, which is what makes the
    ordering above hold; keep this the outermost of the two context managers.
    """
    global _CAPTURE_DEPTH
    with _CAPTURE_LOCK:
        _CAPTURE_DEPTH += 1
    try:
        yield
    finally:
        with _CAPTURE_LOCK:
            _CAPTURE_DEPTH -= 1


@contextlib.contextmanager
def hold_off_capture():
    """Yield True while it is safe to make a CUDA call a recording capture would prohibit.

    While a True is held, no capture can ENTER ``torch.cuda.graph``: capture entry takes the same
    lock. Yields False, having taken nothing, when a capture is already recording or is entering
    right now -- the caller must then skip its CUDA call entirely.

    The acquire is non-blocking on purpose. This is polled at 10 Hz for progress reporting, and a
    render must never wait on a progress tick; a skipped poll costs a tenth of a second of bar.
    """
    if not _CAPTURE_LOCK.acquire(blocking = False):
        yield False
        return
    try:
        yield _CAPTURE_DEPTH == 0
    finally:
        _CAPTURE_LOCK.release()


# Raised while a whole step warms up or records: per-block graphs then run their compute, not nested replays.
_STEP_RECORDING = [0]


@contextlib.contextmanager
def _recording_step():
    _STEP_RECORDING[0] += 1
    try:
        yield
    finally:
        _STEP_RECORDING[0] -= 1


def step_recording() -> bool:
    return _STEP_RECORDING[0] > 0


def capture_in_progress() -> bool:
    """Whether a capture is recording, as a snapshot for reporting.

    NOT safe to gate a CUDA call on: by the time the caller acts the answer can have changed.
    Use ``hold_off_capture`` for that.
    """
    with _CAPTURE_LOCK:
        return _CAPTURE_DEPTH > 0


@contextlib.contextmanager
def _graph_without_flush(graph: Any, pool: Any = None):
    """``torch.cuda.graph`` minus its ``empty_cache`` / host ``emptyCache``: same capture stream discipline, same
    global capture mode, same pool."""
    torch = _torch()
    torch.cuda.synchronize()
    # torch.cuda.graph's own capture stream: pool blocks are reused only from the stream that recorded them.
    stream = getattr(torch.cuda.graph, "default_capture_stream", None)
    if stream is None or getattr(stream, "device", None) != torch.device(
        "cuda", torch.cuda.current_device()
    ):
        stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph.capture_begin(pool = pool, capture_error_mode = "global")
        try:
            yield
        finally:
            graph.capture_end()


# Not ``torch.utils._pytree``: it makes an unregistered object a LEAF, which must stay visible.


def _flatten(obj: Any, out: list) -> tuple:
    """Append every tensor in ``obj`` to ``out``; return a spec that rebuilds ``obj``."""
    if _torch().is_tensor(obj):
        out.append(obj)
        return ("t", len(out) - 1)
    if isinstance(obj, (list, tuple)):
        return ("l", isinstance(obj, tuple), [_flatten(o, out) for o in obj])
    if isinstance(obj, dict):
        return ("d", [(k, _flatten(v, out)) for k, v in obj.items()])
    if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
        return ("v", obj)
    raise TypeError(f"cannot make a static buffer for a {type(obj).__name__} in the call tree")


def _rebuild(spec: tuple, tensors: list) -> Any:
    kind = spec[0]
    if kind == "t":
        return tensors[spec[1]]
    if kind == "l":
        values = [_rebuild(s, tensors) for s in spec[2]]
        return tuple(values) if spec[1] else values
    if kind == "d":
        return {k: _rebuild(s, tensors) for k, s in spec[1]}
    return spec[1]


def graph_key(obj: Any) -> tuple:
    """Hashable identity of a call: shape AND stride AND dtype AND device, because a graph replays
    into the exact buffers it recorded against. Scalars key by VALUE, constants at capture time."""
    if _torch().is_tensor(obj):
        return (
            "t",
            tuple(obj.shape),
            tuple(obj.stride()),
            obj.dtype,
            obj.device.type,
            obj.device.index,
        )
    if isinstance(obj, (list, tuple)):
        return ("l", isinstance(obj, tuple), tuple(graph_key(o) for o in obj))
    if isinstance(obj, dict):
        return ("d", tuple((k, graph_key(v)) for k, v in obj.items()))
    if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
        return ("v", obj)
    return ("o", type(obj).__name__, id(obj))


def _copy_mode(entry: Any) -> Any:
    if getattr(entry, "inference_copy", False):
        return _torch().inference_mode()
    return contextlib.nullcontext()


def _uncapturable(key: tuple) -> bool:
    kind = key[0]
    if kind == "o":
        return True
    if kind == "l":
        return any(_uncapturable(k) for k in key[2])
    if kind == "d":
        return any(_uncapturable(k) for _, k in key[1])
    return False


def _has_float(key: tuple) -> bool:
    """True when the key holds a float, ie a per-step timestep meaning one graph per step."""
    kind = key[0]
    if kind == "v":
        return isinstance(key[1], float)
    if kind == "l":
        return any(_has_float(k) for k in key[2])
    if kind == "d":
        return any(_has_float(k) for _, k in key[1])
    return False


# Private pools per graph OOM a dual-DiT family; the graphs never replay concurrently, so share.
_POOL_BOX: list = [None]

# A pool id handed to a capture after the last graph recorded into it died is a dangling pointer.
_LIVE_WRAPPERS: "weakref.WeakSet" = weakref.WeakSet()


def _drop_pool_if_unused() -> None:
    """Forget the shared pool id once no live graph records into it: capturing into a pool whose graphs are all gone
    dies on the allocator's use_count assert. Checked per graph, not per wrapper: a wrapper still holding graphs in an
    EARLIER pool (one dropped before) does not keep the current id alive."""
    try:
        pool = _POOL_BOX[0]
        if pool is None:
            return
        for wrapper in tuple(_LIVE_WRAPPERS):
            for entry in tuple(wrapper.cache.values()):
                if getattr(entry, "pool_token", pool) == pool:  # handles are fresh tuples per call
                    return
        _POOL_BOX[0] = None
    except Exception:  # noqa: BLE001
        pass


# A failed capture leaves the pinned host allocator (torch >= 2.11) recording to its pool for good, with a filter
# holding a raw pointer to the failed CUDAGraph: so that graph is never freed and its pool never recorded into again.
MAX_FAILED_CAPTURES = 16
_FAILED_GRAPHS: list = []
_COLLIDED_GRAPHS: list = []
_FAILED_LOCK = threading.Lock()


def captures_exhausted() -> bool:
    """True once MAX_FAILED_CAPTURES recordings failed in this process; every later capture then runs eager."""
    return len(_FAILED_GRAPHS) >= MAX_FAILED_CAPTURES


def refuse_if_exhausted(logger: Any = None) -> None:
    """Raise (the caller's failed-capture path then runs eager) once the failure budget is spent; log that once."""
    global _EXHAUSTED_LOGGED
    if not captures_exhausted():
        return
    with _FAILED_LOCK:
        first = not _EXHAUSTED_LOGGED
        _EXHAUSTED_LOGGED = True
    if first and logger is not None:
        logger.warning(
            "diffusion.cuda_graph: %d graph captures failed in this process; no further captures are attempted",
            len(_FAILED_GRAPHS),
        )
    raise RuntimeError(
        f"{len(_FAILED_GRAPHS)} graph captures failed in this process; capture is off"
    )


_EXHAUSTED_LOGGED = False


def retire_failed_capture(
    graph: Any,
    pool: Any,
    exc: Optional[BaseException] = None,
) -> None:
    """Clean up after a capture into ``pool`` that raised. The caller must stop handing ``pool`` to captures.

    "already recording" comes from ``capture_begin`` finding the pool in another capture: that recording is not ours
    to end. A graph whose recording did end (the forward raised, instantiate failed) is reset, which releases its pool
    reference and graph, and kept as an empty husk like the rest."""
    begun_elsewhere = exc is not None and "already recording" in str(exc)
    ended_here = False if begun_elsewhere else _abandon_capture_pool(pool)
    if not begun_elsewhere:  # the generators then belong to the other thread's live capture
        _heal_generators()
    if not ended_here and not begun_elsewhere:
        try:
            graph.reset()
        except Exception:  # noqa: BLE001
            pass
    with _FAILED_LOCK:
        # A collision with another recording left nothing behind: kept alive, but not a failure the budget counts.
        (_COLLIDED_GRAPHS if begun_elsewhere else _FAILED_GRAPHS).append(graph)
    if pool is not None and _POOL_BOX[0] == pool:
        _POOL_BOX[0] = None


def _nvfp4_flashinfer_linears(module: Any) -> list:
    try:
        from .diffusion_nvfp4_linear import is_nvfp4_flashinfer_linear
    except Exception:  # noqa: BLE001 - no backend module, no NVFP4 layers to find
        return []
    found: list = []
    try:
        for name, sub in module.named_modules():
            if is_nvfp4_flashinfer_linear(sub):
                found.append((name, sub))
    except Exception:  # noqa: BLE001 - an exotic module tree is simply not an NVFP4 one
        return []
    return found


def _protect_keyed(module: Any) -> bool:
    """Whether the graph key needs the precision branch: lever armed AND the module holds NVFP4 layers."""
    try:
        from .diffusion_nvfp4_protect import module_controller
        if not module_controller(module).armed:
            return False
        return bool(_nvfp4_flashinfer_linears(module))
    except Exception:  # noqa: BLE001 - a tree we cannot walk simply keys the way it always did
        return False


def protect_graph_key(controller: Any = None) -> tuple:
    from .diffusion_nvfp4_protect import protect_graph_key as _key
    return _key(controller = controller)


def _unbaked_nvfp4_layers(layers: list) -> list:
    return [name for name, layer in layers if not getattr(layer, "activation_scales_baked", False)]


def _prewarm_token_counts(live: list) -> tuple:
    """Candidate GEMM row counts (M) from the warm-up's shapes, smallest first; bounded."""
    counts = {1}
    for tensor in live:
        try:
            shape = tuple(int(dim) for dim in tensor.shape)
        except Exception:  # noqa: BLE001 - not a shaped tensor, nothing to read
            continue
        if len(shape) < 2:
            continue
        rows = 1
        for dim in shape[:-1]:
            rows *= dim
        if rows > 0:
            counts.add(rows)
    return tuple(sorted(counts)[:8])


def _pool_bytes(graph: Any) -> int:
    """Device bytes the graph's private pool holds (shared by every graph of the load), 0 if unreadable."""
    try:
        pool = tuple(graph.pool())
        total = 0
        for seg in _torch().cuda.memory_snapshot():
            if tuple(seg.get("segment_pool_id") or ()) == pool:
                total += int(seg.get("total_size", 0))
        return total
    except Exception:  # noqa: BLE001
        return 0


def _current_stream() -> Any:
    try:
        return _torch().cuda.current_stream()
    except Exception:  # noqa: BLE001
        return None


def _restore_stream(stream: Any) -> None:
    if stream is None:
        return
    try:
        torch = _torch()
        if torch.cuda.current_stream() != stream:
            torch.cuda.set_stream(stream)
    except Exception:  # noqa: BLE001
        pass


def _abandon_capture_pool(pool: Any) -> bool:
    """After a capture that raised: take the allocator off the capture's pool if ``capture_end`` never did.

    ``CUDAGraph::capture_end`` checks ``cudaStreamEndCapture`` before ``endAllocateToPool``, so an invalidated capture
    can leave the pool in the allocator's ``captures_underway``: on torch 2.6 every later ``empty_cache`` then trips
    ``INTERNAL ASSERT captures_underway.empty()`` and no later capture in the process records; later torch skips the
    global release instead, so ``empty_cache`` frees nothing. That failed graph never releases the reference its
    ``capture_begin`` took (its reset releases only once ``capture_end`` got past the pool), so it is released here,
    and ONLY when this call ended the capture: had ``capture_end`` got that far (an error raised in the forward that
    did not invalidate the stream, a failed instantiate) the graph owns the reference and releases it itself, and a
    second release here would drop a shared pool under live graphs or abort the process. True when this call ended it."""
    torch = _torch()
    end = getattr(torch._C, "_cuda_endAllocateToPool", None) or getattr(
        torch._C, "_cuda_endAllocateCurrentStreamToPool", None
    )
    if end is None or pool is None:
        return False
    try:
        device = torch.cuda.current_device()
        end(device, pool)
    except Exception:  # noqa: BLE001 - capture_end already took it off, or capture_begin never put it on
        return False
    try:
        torch._C._cuda_releasePool(device, pool)
    except Exception:  # noqa: BLE001
        pass
    return True


def _heal_generators() -> None:
    """Take every CUDA default generator out of graph-capture mode after a FAILED capture.

    ``CUDAGraph.capture_end`` ends the generators' capture only after ``cudaStreamEndCapture`` succeeds. When the
    capture was invalidated (a host sync inside it, a kernel that may not be recorded) it raises first, and every later
    eager draw from the generator (``torch.randn(device = "cuda")``, a pipeline's noise) fails with "Offset increment
    outside graph capture encountered unexpectedly" for the rest of the process. A clone of the state keeps the seed
    and the eager offset, so the eager sequence continues as if the capture had never run."""
    try:
        torch = _torch()
        for gen in getattr(torch.cuda, "default_generators", ()) or ():
            clone = getattr(gen, "clone_state", None)
            restore = getattr(gen, "graphsafe_set_state", None)
            if callable(clone) and callable(restore):
                restore(clone())
    except Exception:  # noqa: BLE001
        pass


# Offloaded: each key's first SPEED_SAMPLES replays are judged against its own eager steps (_judge_keys).
SPEED_SAMPLES = 3
# Eager steps timed before a key's capture (after one untimed warm-up); never the capture's own warm-ups, whose
# copies follow the recorded schedule and so measure the replay.
SPEED_EAGER_SAMPLES = 2
# A key's replay this much slower than its eager step (and by at least SPEED_MIN_MS) drops that key's graph.
SPEED_MARGIN = 0.03
SPEED_MIN_MS = 2.0
# Streamed: the recording holds a ring and a pool the eager step does not, so it must be at least this much faster.
SPEED_GAIN = 0.01
SPEED_CHECK_ENV = "UNSLOTH_DIFFUSION_OFFLOAD_GRAPH_SPEED_CHECK"


def speed_check_enabled() -> bool:
    return (os.environ.get(SPEED_CHECK_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _timing_events() -> tuple:
    torch = _torch()
    return torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)


def live_pool_bytes() -> int:
    """Bytes the shared graph pool holds while any graph lives (the status number)."""
    try:
        return max(
            [int(w.stats.get("pool_bytes", 0)) for w in tuple(_LIVE_WRAPPERS) if w.cache] or [0]
        )
    except Exception:  # noqa: BLE001
        return 0


def live_pool_free_bytes() -> int:
    """Bytes of the live shared pool that are reserved but NOT allocated: what ``memory_reserved() -
    memory_allocated()`` would wrongly credit as reusable. The pool's allocated part (the graphs' static outputs) is
    already outside that difference. 0 when no graph lives or the snapshot is unreadable."""
    # Every pool a live graph recorded into: a failed capture retires the box's pool while earlier graphs keep it.
    pools = {
        tuple(entry.pool_token)
        for w in tuple(_LIVE_WRAPPERS)
        for entry in tuple(w.cache.values())
        if getattr(entry, "pool_token", None) is not None
    }
    if not pools:
        return 0
    try:
        free = 0
        cuda = _torch().cuda
        device = cuda.current_device()  # the snapshot spans every card
        for seg in cuda.memory_snapshot():
            if seg.get("device", device) != device:
                continue
            if tuple(seg.get("segment_pool_id") or ()) in pools:
                free += int(seg.get("total_size", 0)) - int(seg.get("allocated_size", 0))
        return max(0, free)
    except Exception:  # noqa: BLE001 - over-counting only makes the guard stricter
        return live_pool_bytes()


def _warn(logger: Any, what: str, exc: Any) -> None:
    if logger is not None:
        logger.warning("diffusion.cuda_graph: %s failed: %s", what, exc)


def _outer_layer(module: Any) -> Any:
    """The ``_unsloth_outer_forward`` layer in ``module``'s forward slot (its ``inner`` None = class forward)."""
    try:
        slot = module.__dict__.get("forward")
    except Exception:  # noqa: BLE001 - no instance dict, nothing layered
        return None
    return slot if getattr(slot, "_unsloth_outer_forward", False) is True else None


class _Entry:
    """One captured graph plus the static buffers it replays into."""

    __slots__ = (
        "graph",
        "static",
        "in_spec",
        "out_spec",
        "out_tensors",
        "sticky",
        "last",
        "inference_copy",
        "pool_token",
    )


class GraphedForward:
    """A callable that replaces one denoiser module's ``forward`` with a CUDA-graph replay.

    It walks the ``(args, kwargs)`` tree rather than naming positions, so every family goes through
    the identical object. A capture that raises poisons the wrapper rather than half-graphing."""

    def __init__(
        self,
        module: Any,
        *,
        warmup: int = WARMUP_ITERS,
        max_graphs: int = MAX_GRAPHS_PER_MODULE,
        logger: Any = None,
        call: Any = None,
        placement: Optional["OffloadPlacement"] = None,
    ) -> None:
        self.module = module
        # How the weights reach the GPU (None = resident); see OffloadPlacement.
        self.placement = placement
        # The CLASS forward: the eager callable, whatever is already in the instance slot. A class
        # whose stock forward syncs the host gets its capture-safe rewrite (bit-identical) instead.
        # ``call`` overrides it (GraphedCompiledCall: the module's whole-module compiled callable).
        safe = None
        if call is None or placement is not None:
            try:
                from .diffusion_capture_safe import resolve as _capture_safe  # noqa: PLC0415
                safe, _ = _capture_safe(type(module))
            except Exception:  # noqa: BLE001 - the stock forward is still a correct eager callable
                safe = None
        self.capture_safe = safe is not None
        # A forward whose call carries a non-tensor object (Qwen-Image-2.1's KV cache) names a planner that turns a
        # call into a tensors-only one, or None for "run this call eager"; ``sticky`` kwargs only change between
        # renders, so a replay copies them when the caller passes a different tensor object.
        self.plan = getattr(safe, "__unsloth_graph_plan__", None) if safe is not None else None
        self.sticky = frozenset(getattr(safe, "__unsloth_graph_sticky__", ()) or ())
        # How a planned step enters through the offload hooks (offload_placement refuses a placement without it).
        self.placed = getattr(safe, "__unsloth_graph_placed__", None) if safe is not None else None
        if placement is not None:
            # Model offload calls the slot with the weights already on the GPU, so a rewrite can sit there; a block
            # offload slot holds the hook chain, which then reaches the planned step through ``_placed_step``.
            rewrite = safe is not None and self.plan is None and placement.mode == "model"
            call = safe.__get__(module) if rewrite else placement.eager
        if call is not None:
            self.orig = call
        else:
            self.orig = (safe if safe is not None else type(module).forward).__get__(module)
        try:
            update_wrapper(self, self.orig, updated = ())
            # The wrapped callable's attributes, but never over this wrapper's own: under block offload ``orig`` can
            # be the static step skip, whose ``plan`` / ``stats`` would otherwise replace the graph's.
            for name, value in dict(getattr(self.orig, "__dict__", None) or {}).items():
                self.__dict__.setdefault(name, value)
        except Exception:  # noqa: BLE001
            pass
        try:
            # LOAD-BEARING: H3's modular pipeline filters kwargs by ``inspect.signature(
            # transformer.forward).parameters``, so a bare ``**kwargs`` signature drops tensors.
            self.__signature__ = inspect.signature(self.orig)
        except Exception:  # noqa: BLE001
            pass
        self.warmup = int(warmup)
        self.max_graphs = int(max_graphs)
        self.logger = logger
        self.enabled = False
        self.bypassed = False
        self.poisoned = False
        self.capture_error: Optional[dict] = None
        self.cache: dict = {}
        self.cap_hit = False
        self._warmed: dict = {}
        # Keys recorded before: a placement change re-records them with no warm-up (same shapes, new addresses).
        self._recorded: dict = {}
        # Offloaded only, per graph key: {"seen": eager calls, "eager": [(start, end)], "graph": [(start, end)],
        # "verdict": None until judged, then "" (kept) or the reason it was dropped}.
        self._judge: dict = {}
        self._dropped: set = set()  # keys that run eager for the load
        self._slower: Optional[str] = None
        # Per-block fallback for the calls this graph leaves eager: its arming callable and the handle it returned.
        self.fallback: Any = None
        self.fallback_handle: Any = None
        self.protect_keyed: Optional[bool] = None
        self.protect_ctl: Any = None
        self.stats = {
            "captures": 0,
            "replays": 0,
            "eager_calls": 0,
            "fallbacks": 0,
            "refused_float": 0,
            "refused_host_tensor": 0,
            "refused_object": 0,
            "cap_skips": 0,
            "invalidations": 0,
            "pool_bytes": 0,
            "planned_eager": 0,
            "shape_warmups": 0,
            "evictions": 0,
            "speed_eager": 0,
        }
        self.valid_token: Any = None
        _LIVE_WRAPPERS.add(self)

    def install(self) -> "GraphedForward":
        """Set ``forward`` via ``__dict__`` (``__setattr__`` would inspect it), under any outer step-skip layer."""
        if self.placement is not None:
            self.placement.install(self)
            return self
        outer = _outer_layer(self.module)
        if outer is not None:
            outer.inner = self
            return self
        self.module.__dict__["forward"] = self
        return self

    def uninstall(self) -> "GraphedForward":
        if self.placement is not None:
            self.placement.uninstall(self)
            return self
        outer = _outer_layer(self.module)
        if outer is not None:
            if outer.inner is self:
                outer.inner = None
            return self
        if self.module.__dict__.get("forward") is self:
            self.module.__dict__.pop("forward", None)
            return self
        # Offload hooks installed after this wrapper wrap it: unwrap it inside their chain, keeping the hooks.
        registry = getattr(self.module, "_diffusers_hook", None)
        for ref in list(getattr(registry, "_fn_refs", None) or ()):
            if getattr(ref, "forward", None) is self:
                ref.forward = type(self.module).forward.__get__(self.module)
        return self

    def enable(self) -> "GraphedForward":
        self.install()
        self.enabled = True
        return self

    def disable(self) -> "GraphedForward":
        """Disarm AND uninstall, so the call path matches a load that never built a wrapper."""
        self.enabled = False
        self.uninstall()
        return self

    def set_bypass(self, on: bool) -> "GraphedForward":
        """Run eager without dropping the captured graphs (a chunk under a step cache)."""
        self.bypassed = bool(on)
        if getattr(self, "fallback_handle", None) is not None:
            self.fallback_handle.set_bypass(on)
        return self

    def _release(self) -> None:
        """Return the dropped graphs' pool segments to the device; ``gc.collect`` alone leaves them in the private pool."""
        try:
            gc.collect()
            torch = _torch()
            cuda = getattr(torch, "cuda", None)
            if getattr(self, "placement", None) is not None:
                # the recordings ran on their own capture stream, whose cuBLAS workspace outlives them
                clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
                if callable(clear):
                    clear()
            if cuda is not None and hasattr(cuda, "empty_cache"):
                cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def reset(self) -> "GraphedForward":
        """Drop every captured graph (weights changed under us: LoRA load, unload, adapter switch)."""
        self.cache.clear()
        self._recorded.clear()
        if getattr(self, "fallback_handle", None) is not None:
            try:
                self.fallback_handle.reset()
            except Exception:  # noqa: BLE001
                pass
        self._release()
        return self

    def poison(self, exc: BaseException) -> "GraphedForward":
        """Record why capture failed and run eager for the rest of this wrapper's life."""
        self.capture_error = {
            "type": type(exc).__name__,
            "msg": str(exc)[:4000],
            "traceback": traceback.format_exc()[-6000:],
        }
        self.poisoned = True
        # ``__call__`` short-circuits to eager on ``poisoned`` before it reads the cache, so every
        # entry is now unreachable and only pins its statics, outputs and slice of the pool for the
        # life of the load. Freeing is left to the caller's ``_release``, which runs once the failed
        # capture's own frames are gone, so both go back in one pass.
        self.cache.clear()
        _drop_pool_if_unused()
        return self

    def free(self) -> "GraphedForward":
        """Drop the graphs and the slot."""
        self.reset()
        self.fallback = None
        if getattr(self, "fallback_handle", None) is not None:
            try:
                self.fallback_handle.free()
            except Exception:  # noqa: BLE001
                pass
            self.fallback_handle = None
        self.uninstall()
        if self.placement is not None:
            self.placement.release_slots()
        self.enabled = False
        _LIVE_WRAPPERS.discard(self)
        return self

    def describe(self) -> dict:
        """A JSON-safe summary for the status payload, without the capture traceback."""
        error = self.capture_error
        return {
            "module": type(self.module).__name__,
            "enabled": bool(self.enabled),
            "bypassed": bool(self.bypassed),
            "poisoned": bool(self.poisoned),
            "graphs": len(self.cache),
            "warmup": int(self.warmup),
            "max_graphs": int(self.max_graphs),
            "cap_hit": bool(self.cap_hit),
            "placement": None if self.placement is None else self.placement.mode,
            "stats": dict(self.stats),
            "fallback": None
            if getattr(self, "fallback_handle", None) is None
            else self.fallback_handle.describe(),
            "capture_error": None
            if not error
            else {
                "type": str(error.get("type")),
                "msg": str(error.get("msg")),
            },
        }

    def _judge_keys(self) -> None:
        """Drop the graph of every key whose replays measured slower than its own eager steps.

        A replay records the onload copies too, and how fast a recorded copy schedule runs depends on the machine and
        on how it was recorded (an A100 VM: a Qwen-Image-2.1 prompt length recorded after the first replayed 16%
        slower than eager while the first matched it; an L4 ran replayed copies 8% faster). Each key's first replays
        are timed against its own eager steps with CUDA events, read only once they have completed (no host wait)."""
        for key, state in list(self._judge.items()):
            if state["verdict"] is not None or len(state["graph"]) < SPEED_SAMPLES:
                continue
            # Event queries and the drop's syncs / frees are prohibited while another thread records: judge later.
            with hold_off_capture() as safe:
                if not safe or not all(end.query() for _, end in state["graph"]):
                    continue
                try:
                    # Best of each, so a stall on a shared card in either window cannot tip the verdict alone.
                    eager = min(start.elapsed_time(end) for start, end in state["eager"])
                    graph = min(start.elapsed_time(end) for start, end in state["graph"])
                except Exception:  # noqa: BLE001
                    state["verdict"] = ""
                    continue
                self.stats["eager_ms"] = round(eager, 3)
                self.stats["replay_ms"] = round(graph, 3)
                placement = getattr(self, "placement", None)
                streams = placement is not None and placement.streams()
                if streams:
                    kept = graph < eager * (1.0 - SPEED_GAIN)
                else:
                    kept = graph <= eager * (1.0 + SPEED_MARGIN) or graph - eager < SPEED_MIN_MS
                if kept:
                    state["verdict"] = ""
                    continue
                if graph > eager:
                    reason = f"replay measured slower than the eager step on this device ({graph:.1f} vs {eager:.1f} ms)"
                else:
                    reason = (
                        f"replay measured within {SPEED_GAIN:.0%} of the eager step on this device ({graph:.1f} vs "
                        f"{eager:.1f} ms), not worth the memory a streamed recording holds"
                    )
                state["verdict"] = reason
                ring = False
                if streams:
                    # One placement, one verdict: its other keys stream the same copies.
                    ring = bool(placement.decline(reason))
                    for other in list(self.cache):
                        if other != key:
                            self.cache.pop(other, None)
                            self._dropped.add(other)
                self._slower = reason
                self._dropped.add(key)
                if self.logger is not None:
                    self.logger.info(
                        "diffusion.cuda_graph: %s drops a graph: %s",
                        type(self.module).__name__,
                        reason,
                    )
                if self.cache.pop(key, None) is not None:
                    try:
                        _torch().cuda.current_stream().synchronize()  # no replay still running on what is freed
                    except Exception:  # noqa: BLE001
                        pass
                    _drop_pool_if_unused()
                if not self.cache:
                    self.capture_error = {"type": "Refused", "msg": reason}
                    if not ring:
                        # (with a ring, its drop at the end of this forward flushes once: a mid-forward flush fragments the cache)
                        self._release()

    def _timed_eager(self, call: Any, args: tuple, kwargs: dict, key: Any) -> Any:
        """One of the caller's eager steps before ``key`` records (its eager reference).

        ``call`` is the step the graph would record (the planned step under a plan), on fresh copies of the caller's
        tensors. The first call of a key may still compile or autotune and is not timed (for a planned module it is
        also the new prompt length's warm-up); the next SPEED_EAGER_SAMPLES are."""
        self.stats["eager_calls"] += 1
        self.stats["speed_eager"] += 1
        # On copies laid out like the capture's statics, as the shape warm-up does: compiled code guards on strides and
        # storage offsets, so steps on the caller's views compile a variant the capture then compiles again.
        live: list = []
        spec = _flatten((args, kwargs), live)
        try:
            statics = self._statics(live)
        except _torch().cuda.OutOfMemoryError:
            # No room for the probe's copies: decline the key and run the caller's own step.
            self._dropped.add(key)
            self._slower = "no memory for the speed probe's input copies"
            return call(*args, **kwargs)
        args, kwargs = _rebuild(spec, statics)
        state = self._judge_state(key)
        state["seen"] += 1
        if state["seen"] == 1:
            self._warmed[key] = None
            return call(*args, **kwargs)
        # Event records are prohibited while another thread records a graph: such a step goes untimed.
        start = end = None
        with hold_off_capture() as safe:
            if safe:
                start, end = _timing_events()
                start.record()
        out = call(*args, **kwargs)
        if start is not None:
            with hold_off_capture() as safe:
                if safe:
                    end.record()
                    state["eager"].append((start, end))
        return out

    def _judge_state(self, key: Any) -> dict:
        state = self._judge.get(key)
        if state is None:
            state = self._judge[key] = {"seen": 0, "eager": [], "graph": [], "verdict": None}
            while len(self._judge) > 4 * self.max_graphs:
                self._judge.pop(next(iter(self._judge)))
        return state

    def _placed_step(self, **step: Any) -> Any:
        return self.placed(self.orig, step)

    def _eager(self, args: tuple, kwargs: dict) -> Any:
        self.stats["eager_calls"] += 1
        return self.orig(*args, **kwargs)

    def _engage_fallback(self) -> None:
        """Arm the per-block graphs once this whole-step graph dropped a key or failed: the calls it now leaves eager
        run through them. Between forwards' blocks (the step has not started), so the slot ring can change."""
        fallback, self.fallback = self.fallback, None
        try:
            self.fallback_handle = fallback()
        except Exception as exc:  # noqa: BLE001
            _warn(self.logger, "per-block fallback", exc)
            self.fallback_handle = None
        if self.fallback_handle is not None and self.logger is not None:
            self.logger.info(
                "diffusion.cuda_graph: %s records per block for the steps its whole-step graph leaves eager",
                type(self.module).__name__,
            )

    def __call__(self, *args, **kwargs):
        if _bg_capture_suppressed():
            # Background compile thread: never capture off the render thread.
            return self.orig(*args, **kwargs)
        if getattr(self, "fallback", None) is not None and (self.poisoned or self._dropped):
            self._engage_fallback()
        if (
            not self.enabled
            or _bg_eager_forced()
            or self.bypassed
            or self.poisoned
            # Belt to the caller's per-chunk bypass: one step's cond and uncond share a key.
            or getattr(self.module, "_unsloth_step_cache", None)
        ):
            return self._eager(args, kwargs)

        # True or absent builds an output dataclass, which a replay hands back frozen.
        if kwargs.get("return_dict", True) is not False:
            return self._eager(args, kwargs)

        if self.placement is not None:
            # The recorded graphs replay the weight addresses of the placement they saw; a new placement records anew.
            token = self.placement.token()
            if token != self.valid_token:
                if self.cache:
                    self.stats["invalidations"] += 1
                    # Not reset(): its empty_cache on every model-offload onload makes the next onload pay the
                    # cudaMallocs again. The dropped pool's blocks stay with the allocator.
                    self.cache.clear()
                    _drop_pool_if_unused()  # the next capture must not record into a pool no graph holds
                self._warmed.clear()
                self.valid_token = token
            refusal = self.placement.refusal(self.stats)
            if self._judge and speed_check_enabled():
                self._judge_keys()
                if self._dropped and getattr(self, "fallback", None) is not None:
                    self._engage_fallback()
                refusal = self.placement.refusal(self.stats)
            skip = getattr(self.placement, "skip", None)
            if refusal is None and skip is not None and _skips_planned(skip):
                # This render skips steps: the skip layer must see every call, so they all run eager (per-block
                # graphs, where armed, still record below the hooks).
                self.stats["skip_eager"] = self.stats.get("skip_eager", 0) + 1
                if getattr(self, "fallback", None) is not None:
                    self._engage_fallback()
                return self._eager(args, kwargs)
            if refusal is not None:
                if self.logger is not None and getattr(self, "_refusal_logged", None) != refusal:
                    self._refusal_logged = refusal
                    self.logger.info(
                        "diffusion.cuda_graph: %s runs eager: %s",
                        type(self.module).__name__,
                        refusal,
                    )
                self.capture_error = {"type": "Refused", "msg": refusal}
                return self._eager(args, kwargs)
            if (self.capture_error or {}).get("type") == "Refused" and (
                self.cache or not self._slower
            ):
                self.capture_error = None

        call = self.orig
        # Every eager fallback below runs the caller's own call, never the planned one.
        eager_args, eager_kwargs = args, kwargs
        if self.plan is not None:
            planned = self.plan(self.module, args, kwargs)
            if planned is None:
                self.stats["planned_eager"] += 1
                return self._eager(args, kwargs)
            call, kwargs = planned
            args = ()
            if self.placement is not None:
                # The offload hooks onload the top-level weights: the step must enter through the hooked forward.
                call = self._placed_step

        try:
            key = graph_key((args, kwargs))
            if self.protect_keyed is None:
                self.protect_keyed = _protect_keyed(self.module)
                if self.protect_keyed:
                    from .diffusion_nvfp4_protect import module_controller
                    self.protect_ctl = module_controller(self.module)
                if self.protect_keyed:
                    self.max_graphs *= 2
                    if self.logger is not None:
                        self.logger.info(
                            "diffusion.cuda_graph: NVFP4 per-step precision is armed on %s; graph "
                            "cap raised to %d (one graph per branch per input shape)",
                            type(self.module).__name__,
                            self.max_graphs,
                        )
            if self.protect_keyed:
                # One graph per branch: a W4A4 replay at a W4A16 step means the lever never fired.
                key = key + protect_graph_key(self.protect_ctl)
            extra = getattr(self.module, GRAPH_KEY_EXTRA_ATTR, None)
            if callable(extra):
                # Python state a pre-hook set for THIS call that picks a branch inside the forward (HunyuanVideo-1.5's
                # null-mask flag): a graph recorded on one branch must never replay for the other.
                key = key + (("x", extra()),)
            entry = self.cache.get(key)
        except Exception:  # noqa: BLE001 - an unhashable tree is simply not capturable
            self.stats["refused_object"] += 1
            return self._eager(eager_args, eager_kwargs)

        if entry is None:
            if _uncapturable(key):
                self.stats["refused_object"] += 1
                return self._eager(eager_args, eager_kwargs)
            if _has_float(key):
                self.stats["refused_float"] += 1
                return self._eager(eager_args, eager_kwargs)
            if self.placement is not None and speed_check_enabled():
                if key in self._dropped:
                    return self._eager(eager_args, eager_kwargs)
                state = self._judge.get(key)
                if state is None or (
                    state["verdict"] is None and len(state["eager"]) < SPEED_EAGER_SAMPLES
                ):
                    # The eager reference for _judge_keys, timed with CUDA events and never waited on.
                    return self._timed_eager(call, args, kwargs, key)
            if self.plan is not None and self.stats["captures"] and key not in self._warmed:
                # A new prompt length: this real step is the new shape's warm-up; the next one records.
                self._warmed[key] = None
                while len(self._warmed) > 4 * self.max_graphs:
                    self._warmed.pop(next(iter(self._warmed)))
                self.stats["shape_warmups"] += 1
                # On copies laid out like the capture's statics (fresh, offset 0): compiled code guards on strides and
                # storage offsets, so a step on the caller's views would warm a variant the capture then misses and
                # compiles inside the capture, which fails it.
                live: list = []
                spec = _flatten((args, kwargs), live)
                warm_args, warm_kwargs = _rebuild(spec, self._statics(live))
                return call(*warm_args, **warm_kwargs)
            if len(self.cache) >= self.max_graphs and self.plan is not None:
                # A planned call keys on the prompt length, so evict the least recently replayed graph.
                self.cache.pop(next(iter(self.cache)))
                self.stats["evictions"] += 1
                # The last graph on the shared pool takes the pool with it; never hand its id to the next capture.
                _drop_pool_if_unused()
            if len(self.cache) >= self.max_graphs:
                # The recorded graphs keep replaying: one rare new key is no reason to drop them.
                self.stats["cap_skips"] += 1
                if not self.cap_hit:
                    self.cap_hit = True
                    if self.logger is not None:
                        self.logger.warning(
                            "diffusion.cuda_graph: graph cap of %d reached on %s; further input "
                            "shapes run eager",
                            self.max_graphs,
                            type(self.module).__name__,
                        )
                return self._eager(eager_args, eager_kwargs)
            captured = None
            stream_before = _current_stream()
            try:
                captured = self._capture(args, kwargs, call, key)
            except Exception as exc:  # noqa: BLE001 - a failed capture must never fail the render
                # torch.cuda.graph's __exit__ raises in capture_end before it leaves its capture stream, so the thread
                # would run everything after (the eager retry, the rest of the process) on the capture stream.
                _restore_stream(stream_before)
                if "already recording" not in str(
                    exc
                ):  # a collision: the generators are the live capture's
                    _heal_generators()  # before anything eager draws from the CUDA RNG
                if self.placement is not None:
                    self.placement.recover()  # the eager retry must not wait on the dead capture's copies
                    self.placement.release_slots()  # nothing will replay into the ring for this load
                self.poison(exc)
                self.stats["fallbacks"] += 1
                if self.logger is not None:
                    self.logger.warning(
                        "diffusion.cuda_graph: capture failed on %s (%s: %s); this load runs eager",
                        type(self.module).__name__,
                        type(exc).__name__,
                        exc,
                    )
                # The handled exception keeps _capture's frame, and with it the statics and the pool slice,
                # alive through the eager retry, which then OOMs on a tight card. capture_error has the text.
                exc.__traceback__ = None
            if captured is None:
                self._release()
                return self._eager(eager_args, eager_kwargs)
            entry = captured
            self.cache[key] = entry
            if self.logger is not None:
                self.logger.debug(
                    "diffusion.cuda_graph: captured %s graph %d/%d over %d input tensor(s)",
                    type(self.module).__name__,
                    len(self.cache),
                    self.max_graphs,
                    len(entry.static),
                )

        elif self.plan is not None:
            self.cache[key] = self.cache.pop(key)

        live: list = []
        _flatten((args, kwargs), live)
        last = entry.last
        state = self._judge.get(key) if self.placement is not None else None
        timing = (
            state is not None and state["verdict"] is None and len(state["graph"]) < SPEED_SAMPLES
        )
        start = None
        if timing:
            # Through the input copies and output clones: the eager step pays neither. Not while another thread
            # records, whose capture prohibits event records.
            with hold_off_capture() as safe:
                if safe:
                    start, end = _timing_events()
                    start.record()
        with _copy_mode(entry):
            for index, (dst, src) in enumerate(zip(entry.static, live)):
                if last is not None and entry.sticky[index]:
                    ref = last[index]
                    if ref is not None and ref() is src:
                        continue
                    dst.copy_(src)
                    last[index] = weakref.ref(src)
                    continue
                dst.copy_(src)
        entry.graph.replay()
        # Cloned: every replay writes the SAME buffers, which the pipeline holds across steps.
        outs = [t.clone() for t in entry.out_tensors]
        if start is not None:
            with hold_off_capture() as safe:
                if safe:
                    end.record()
                    state["graph"].append((start, end))
        self.stats["replays"] += 1
        skip = getattr(self.placement, "skip", None) if self.placement is not None else None
        if skip is not None and getattr(skip, "counting", False):
            # The replay passed the (pass-through) skip layer by: count the computed step as it would have.
            skip.stats["calls"] += 1
            skip.stats["computed"] += 1
        return _rebuild(entry.out_spec, outs)

    def _statics(self, live: list) -> list:
        """Fresh copies of ``live``, as a capture records from (and the shape warm-up step runs on)."""
        torch = _torch()
        if self.plan is not None:
            # Inference tensors in, inference tensors as statics: compiled code guards on the dispatch keys, so a
            # mismatch makes the graph record its own (separately autotuned, so not bit-equal) compiled variant.
            static = []
            for t in live:
                with torch.inference_mode(bool(t.is_inference())):
                    static.append(torch.empty_like(t))
        else:
            # A tensor made under ``torch.inference_mode()``, which renders run in, refuses ``copy_``.
            with torch.inference_mode(False):
                static = [torch.empty_like(t) for t in live]
        # Writing into an inference tensor is only allowed inside inference mode.
        copying = types.SimpleNamespace(
            inference_copy = self.plan is not None and any(t.is_inference() for t in static)
        )
        with _copy_mode(copying):
            for dst, src in zip(static, live):
                dst.copy_(src)
        return static

    def _capture(
        self,
        args: tuple,
        kwargs: dict,
        call: Any = None,
        key: Any = None,
    ) -> _Entry:
        torch = _torch()
        call = self.orig if call is None else call
        entry = _Entry()
        live: list = []
        entry.in_spec = _flatten((args, kwargs), live)
        entry.sticky, entry.last = None, None
        if self.sticky:
            # Flat positions of the sticky kwargs, in ``_flatten``'s order (args, then kwargs by insertion).
            marks: list = []
            _flatten(args, marks)
            sticky = [False] * len(marks)
            for name, value in kwargs.items():
                part: list = []
                _flatten(value, part)
                sticky.extend([name in self.sticky] * len(part))
            if any(sticky):
                entry.sticky = tuple(sticky)
                entry.last = [weakref.ref(t) if s else None for t, s in zip(live, sticky)]

        bad = [(i, str(t.device)) for i, t in enumerate(live) if t.device.type != "cuda"]
        if bad:
            self.stats["refused_host_tensor"] += 1
            raise RuntimeError(
                f"{len(bad)} input tensor(s) are not on cuda ({bad[:4]}); a host tensor read "
                f"inside a captured region is baked in at its recorded value"
            )

        nvfp4_layers = _nvfp4_flashinfer_linears(self.module)
        unbaked = _unbaked_nvfp4_layers(nvfp4_layers)
        if unbaked:
            raise RuntimeError(
                f"{len(unbaked)} NVFP4 linear(s) report unbaked activation scales "
                f"(first: {unbaked[0]}); a scale still being calibrated would be frozen into the "
                f"graph at whatever value this capture saw, so this load runs eager"
            )

        entry.static = self._statics(live)
        # Writing into an inference tensor is only allowed inside inference mode.
        entry.inference_copy = self.plan is not None and any(t.is_inference() for t in entry.static)
        static_args, static_kwargs = _rebuild(entry.in_spec, entry.static)

        if nvfp4_layers:
            # Before the warm-up: a captured tuning launch would bake in the default tactic.
            from .diffusion_nvfp4_linear import nvfp4_prewarm
            nvfp4_prewarm(self.module, _prewarm_token_counts(live), logger = self.logger)

        refuse_if_exhausted(
            self.logger
        )  # before the warm-ups, and before the step skip's counters are saved
        # Under an offload placement the capture also records the copy-stream onloads (OffloadPlacement.record).
        record = call if self.placement is None else self.placement.recorder(call)
        # Side stream: a workspace first created DURING capture is only valid while recording.
        # A planned module's later captures were warmed by a real step (``__call__``): no warm-up, no cache flushes.
        recapture = self.plan is not None and self.stats["captures"] > 0
        # An offloaded key re-recorded after its weights moved is already warm: it records like a planned re-capture.
        recapture = recapture or (
            self.placement is not None and key is not None and key in self._recorded
        )
        if self.placement is not None:
            self.placement.before_capture()  # anything the recorded copies land in exists before the capture
            self.valid_token = (
                self.placement.token()
            )  # the placement this capture records (the ring included)
        # A pass-through step skip under the hooks counts every call; the warm-ups and the recording are not steps.
        skip = getattr(self.placement, "skip", None) if self.placement is not None else None
        skip_stats = (
            dict(skip.stats)
            if skip is not None and isinstance(getattr(skip, "stats", None), dict)
            else None
        )
        # Per-block graphs under the hooks run their compute while the whole step warms up and records.
        # A key warmed by its timed eager steps records straight away: a side-stream warm-up would cache a second set
        # of activations (the cold render's peak).
        warm = (
            self.placement is not None
            and key is not None
            and int((getattr(self, "_judge", {}).get(key) or {}).get("seen", 0)) >= 1
        )
        with _recording_step():
            if not recapture and not warm:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    # A re-record under a new offload placement only needs the forward to have run once on it.
                    again = self.placement is not None and self.stats["captures"] > 0
                    for _ in range(min(self.warmup, 1) if again else self.warmup):
                        record(*static_args, **static_kwargs)
                torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            # An explicit pool id, never None: a capture that fails must hand its pool back (retire_failed_capture).
            pool = _POOL_BOX[0]
            if pool is None and callable(getattr(torch.cuda, "graph_pool_handle", None)):
                pool = torch.cuda.graph_pool_handle()
            recorder = _graph_without_flush if recapture else torch.cuda.graph
            try:
                with _capturing(), recorder(graph, pool = pool):
                    out = record(*static_args, **static_kwargs)
            except BaseException as exc:
                retire_failed_capture(graph, pool, exc)
                raise
            finally:
                if skip_stats is not None:
                    skip.stats = skip_stats
        self.stats["pool_bytes"] = _pool_bytes(graph)
        if key is not None:
            self._recorded[key] = None
            while len(self._recorded) > 4 * self.max_graphs:
                self._recorded.pop(next(iter(self._recorded)))
        if _POOL_BOX[0] is None:
            try:
                _POOL_BOX[0] = graph.pool()
            except Exception:  # noqa: BLE001 - a private pool per graph still works
                pass

        entry.graph = graph
        # The pool this graph recorded into (see _drop_pool_if_unused), not whatever the box holds by now.
        entry.pool_token = pool if pool is not None else _POOL_BOX[0]
        out_tensors: list = []
        entry.out_spec = _flatten(out, out_tensors)
        entry.out_tensors = out_tensors
        self.stats["captures"] += 1
        return entry


class GraphedCompiledCall(GraphedForward):
    """``GraphedForward`` for a whole-module compiled denoiser (SDXL U-Net): ``Module.__call__`` serves it from
    ``_compiled_call_impl`` and never reads ``forward``, so the replay sits there and captures the compiled callable."""

    def __init__(self, module: Any, **kwargs: Any) -> None:
        compiled = getattr(module, "_compiled_call_impl", None)
        if compiled is None:
            raise RuntimeError(f"{type(module).__name__} is not whole-module compiled")
        self.compiled = compiled
        super().__init__(module, call = compiled, **kwargs)

    def install(self) -> "GraphedCompiledCall":
        self.module._compiled_call_impl = self
        return self

    def uninstall(self) -> "GraphedCompiledCall":
        if getattr(self.module, "_compiled_call_impl", None) is self:
            self.module._compiled_call_impl = self.compiled
        return self


# Offloaded denoisers.


def _leaf_ptrs(t: Any, out: list) -> None:
    """Data pointers of ``t``'s storage, through tensor subclasses (torchao int8 / fp8 weights) to their inner tensors."""
    flatten = getattr(type(t), "__tensor_flatten__", None)
    if flatten is not None and type(t).__name__ not in ("Tensor", "Parameter"):
        try:
            names, _ctx = t.__tensor_flatten__()
            for name in names:
                _leaf_ptrs(getattr(t, name), out)
            return
        except Exception:  # noqa: BLE001 - fall through to the plain pointer
            pass
    try:
        out.append((str(t.device), int(t.data_ptr())))
    except Exception:  # noqa: BLE001
        out.append(("?", id(t)))


def _module_tensors(module: Any) -> list:
    seen: set = set()
    out: list = []
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


class OffloadPlacement:
    """Where an offloaded denoiser's replay sits and when its recorded addresses stop holding.

    ``group``: diffusers block offload driven by the event-fenced prefetch (diffusion_offload_prefetch). The replay
    replaces the module's hooked forward, so ONE graph records the whole step: the copy-stream copies of every
    streamed group from its pinned host tensor into the graph pool, the compute stream waiting on each copy's event,
    and the resident groups read in place. The pool holds the in-flight window at fixed addresses, so it plays the
    prefetch ring. Recorded again when a residency change, a release / restore or a host-copy swap moves an address.

    ``model``: whole-module CPU offload (accelerate). The hook moves the module on the eager path first and the replay
    sits under it, in ``_old_forward``. Recorded again when the onload placed the weights at new addresses; after
    ``MODEL_MAX_INVALIDATIONS`` re-records the module runs eager (each re-record costs a forward)."""

    MODEL_MAX_INVALIDATIONS = 2
    # Past MODEL_MAX_INVALIDATIONS, re-recording continues while each recording serves at least this many replays.
    MODEL_MIN_REPLAYS_PER_RECORD = 4

    def __init__(self, module: Any, mode: str):
        self.module = module
        self.mode = mode
        self.prev: Any = None
        self._epoch: Any = None
        self._fp: Any = None
        self._checked: Any = object()
        self._why: Optional[str] = None
        self._tensors: Optional[list] = None
        # Set when a key's replay did not pay on a streaming placement: nothing records again for this load.
        self._declined: Optional[str] = None
        self._slots = False
        if mode == "group":
            self.prev = module.__dict__.get("forward")
        else:
            self.prev = getattr(module, "_old_forward", None)
        if self.prev is None:
            raise RuntimeError(f"no {mode} offload forward to wrap")
        self.eager = self.prev
        # A static step skip (diffusion_step_skip) the replay would bypass: see GraphedForward.__call__.
        self.skip: Any = self.prev if getattr(self.prev, "_unsloth_outer_forward", False) else None
        # A capture-safe rewrite (diffusion_capture_safe) under block offload: the hook chain ends in the bound class
        # forward, which the rewrite replaces while the graph is installed (bit-identical, so eager calls may take it).
        self._inner: Any = None
        self._inner_stock: Any = None
        self._inner_safe: Any = None
        if mode == "group":
            safe = _plain_rewrite(module)
            if safe is not None:
                ref = _innermost_class_forward(module)
                if ref is None:
                    raise RuntimeError(
                        "the block-offload hook chain does not end in the class forward"
                    )
                self._inner, self._inner_stock = ref, ref.forward
                self._inner_safe = safe.__get__(module)

    def install(self, handle: Any) -> None:
        if self.mode == "group":
            if self._inner is not None:
                self._inner.forward = self._inner_safe
            self.module.__dict__["forward"] = handle
            self._enable_slots()
        else:
            self.module._old_forward = handle

    def _prefetcher(self) -> Any:
        from .diffusion_offload_prefetch import module_prefetcher
        return module_prefetcher(self.module)

    def _enable_slots(self) -> None:
        """Streamed groups copy into the prefetcher's slot ring (diffusion_offload_prefetch), so the recorded copies
        land at fixed addresses outside the graph pool: the pool then holds one step's activations, not a second
        prefetch window beside the eager one. Off with UNSLOTH_DIFFUSION_STEP_GRAPH_SLOTS=0."""
        if (os.environ.get(STEP_SLOTS_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
            return
        if not self.streams():
            return
        try:
            pf = self._prefetcher()
            if pf is not None and callable(getattr(pf, "enable_slots", None)):
                pf.enable_slots()
                self._slots = bool(getattr(pf, "slot_of", None))
        except Exception:  # noqa: BLE001
            self._slots = False

    def streams(self) -> bool:
        """Whether this placement moves weights every step (model offload, or a block group not pinned resident):
        the recording then holds memory beside the eager step's."""
        if self.mode != "group":
            return True
        try:
            from .diffusion_memory import _offload_groups
            pf = self._prefetcher()
            for group in _offload_groups(self.module):
                if getattr(group, "_unsloth_resident", False) or getattr(
                    group, "_unsloth_pinned_top", False
                ):
                    continue
                if pf is None or pf.owns(group):
                    return True
        except Exception:  # noqa: BLE001 - unknown layout: treat as streaming (the stricter speed check)
            return True
        return False

    def before_capture(self) -> None:
        """Allocate the slot ring before a capture, so no slot is carved out of the graph pool."""
        if not self._slots:
            return
        try:
            pf = self._prefetcher()
            if pf is not None and callable(getattr(pf, "materialize_slots", None)):
                pf.materialize_slots()
        except Exception:  # noqa: BLE001
            pass

    def release_slots(self) -> None:
        """Hand the slot ring back once no graph of this placement can replay into it."""
        if not self._slots:
            return
        self._slots = False
        try:
            pf = self._prefetcher()
            if pf is not None and callable(getattr(pf, "drop_slots_after_forward", None)):
                pf.drop_slots_after_forward()
        except Exception:  # noqa: BLE001
            pass

    def decline(self, reason: str) -> bool:
        """No more recording on this placement for the load; hand the slot ring back after the current forward.
        True when that drop is scheduled (it then also releases the dropped graphs' pool)."""
        self._declined = reason
        if not self._slots:
            return False
        self._slots = False
        try:
            pf = self._prefetcher()
            if pf is not None and callable(getattr(pf, "drop_slots_after_forward", None)):
                pf.drop_slots_after_forward()
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def uninstall(self, handle: Any) -> None:
        if self._inner is not None and self._inner.forward is self._inner_safe:
            self._inner.forward = self._inner_stock
        if self.mode == "group":
            if self.module.__dict__.get("forward") is handle:
                self.module.__dict__["forward"] = self.prev
            return
        if getattr(self.module, "_old_forward", None) is handle:
            self.module._old_forward = self.prev
        if (
            self.module.__dict__.get("forward") is handle
        ):  # hooks removed since (diffusers re-enables them per call)
            self.module.__dict__.pop("forward", None)

    def recorder(self, call: Any) -> Any:
        """What a capture records for ``call``: under block offload the prefetcher's copies join the capture."""
        if self.mode != "group":
            return call

        def record(*args: Any, **kwargs: Any) -> Any:
            from .diffusion_offload_prefetch import module_prefetcher

            pf = module_prefetcher(self.module)
            pf.begin()  # inside the capture: forks the copy stream in, and end() joins it back
            try:
                return call(*args, **kwargs)
            finally:
                pf.end()

        return record

    def record(self, *args: Any, **kwargs: Any) -> Any:
        return self.recorder(self.eager)(*args, **kwargs)

    def recover(self) -> None:
        """After a failed capture: drop what the prefetcher queued inside it (see GroupPrefetcher.abandon)."""
        if self.mode != "group":
            return
        try:
            from .diffusion_offload_prefetch import module_prefetcher
            pf = module_prefetcher(self.module)
            if pf is not None:
                pf.abandon()
        except Exception:  # noqa: BLE001
            pass

    def _fingerprint(self) -> tuple:
        out: list = []
        if self.mode == "group":
            from .diffusion_memory import _offload_groups
            from .diffusion_offload_prefetch import _group_tensors, module_prefetcher

            pf = module_prefetcher(self.module)
            for group in _offload_groups(self.module):
                resident = bool(getattr(group, "_unsloth_resident", False))
                out.append((id(group), resident, bool(pf is not None and pf.owns(group))))
                tensors = _group_tensors(group)
                if resident:
                    for t in tensors:
                        _leaf_ptrs(t, out)
                else:
                    cpu = getattr(group, "cpu_param_dict", None) or {}
                    for t in tensors:
                        _leaf_ptrs(cpu.get(t, t), out)
            # Streamed copies land in the slot ring when it is on: a new ring is a new placement.
            raws = getattr(pf, "slot_raw", None) or {}
            out.append(("slots", tuple((i, int(raws[i].data_ptr())) for i in sorted(raws))))
            return tuple(out)
        # Read from the module's registry on every call: Module.to() replaces buffers with new tensor objects on each
        # offload / onload, so a list taken once would keep fingerprinting the obsolete ones.
        for t in _module_tensors(self.module):
            _leaf_ptrs(t, out)
        return tuple(out)

    def token(self) -> Any:
        if self.mode == "group":
            from .diffusion_offload_prefetch import placement_epoch

            epoch = placement_epoch()
            if epoch != self._epoch:
                self._epoch = epoch
                self._fp = self._fingerprint()
            return self._fp
        self._fp = self._fingerprint()
        return self._fp

    def refusal(self, stats: dict) -> Optional[str]:
        if getattr(self, "_declined", None) is not None:
            return self._declined
        if self.mode == "model":
            # Judged once per placement, when the weights land (the fingerprint is rebuilt every call).
            if self._checked != self._fp:
                self._checked = self._fp
                moves = int(stats.get("invalidations", 0))
                records = max(1, int(stats.get("captures", 0)))
                replays = int(stats.get("replays", 0))
                self._why = None
                if (
                    moves >= self.MODEL_MAX_INVALIDATIONS
                    and replays < self.MODEL_MIN_REPLAYS_PER_RECORD * records
                ):
                    self._why = (
                        f"model offload placed the weights at new addresses on {moves} onloads, {replays} replays "
                        f"over {records} recordings; each re-record costs a step"
                    )
            return self._why
        if self._checked is not self._fp:
            from .diffusion_offload_prefetch import capture_refusal
            self._checked = self._fp
            self._why = capture_refusal(self.module)
        return self._why


def _skips_planned(skip: Any) -> bool:
    plan = getattr(skip, "plan", None) or ()
    return bool(plan) and not all(plan)


def offload_placement(module: Any) -> tuple:
    """``(placement, refusal)`` for ``module`` as it is placed now: ``(None, None)`` when nothing moves it."""
    try:
        subs = [m for m in module.modules() if m is not module]
    except Exception:  # noqa: BLE001
        subs = []
    if any(getattr(m, "_hf_hook", None) is not None for m in subs):
        return None, (
            "leaf-level (sequential) offload uploads every layer synchronously from host memory inside the forward"
        )
    hf_hook = getattr(module, "_hf_hook", None)
    compiled = getattr(module, "_compiled_call_impl", None) is not None
    if hf_hook is not None:
        if compiled:
            return (
                None,
                "whole-module compiled denoiser: the CPU-offload hook runs inside its compiled call",
            )
        if "CpuOffload" not in type(hf_hook).__name__:
            return (
                None,
                f"offload hook {type(hf_hook).__name__} moves the weights inside the forward",
            )
        why = _forward_refusal(module, "model")
        if why is not None:
            return None, why
        try:
            return OffloadPlacement(module, "model"), None
        except Exception as exc:  # noqa: BLE001
            return None, f"model offload hook unreadable ({exc})"
    registry = getattr(module, "_diffusers_hook", None)
    hooks = getattr(registry, "hooks", None) or {}
    if not any("offload" in str(key) for key in hooks):
        return None, None
    if compiled:
        return (
            None,
            "whole-module compiled denoiser: the offload hooks run inside its compiled call",
        )
    try:
        from diffusers.hooks import group_offloading as go
        name = getattr(go, "_GROUP_OFFLOADING", "group_offloading")
    except Exception:  # noqa: BLE001
        name = "group_offloading"
    if name not in hooks:
        return None, "an offload hook other than block offloading moves the weights"
    skip = None
    for fn_ref in getattr(registry, "_fn_refs", None) or ():
        if getattr(getattr(fn_ref, "forward", None), "_unsloth_outer_forward", False):
            # The static step skip under the offload hook: a render that plans skips runs every call through it
            # (GraphedForward.__call__); one that plans none passes straight through it, so the graph may replay.
            skip = fn_ref.forward
    why = _forward_refusal(module, "group")
    if why is not None:
        return None, why
    from .diffusion_offload_prefetch import capture_refusal

    why = capture_refusal(module)
    if why is not None and "background" not in why:
        return None, why
    try:
        placement = OffloadPlacement(module, "group")
    except Exception as exc:  # noqa: BLE001
        return None, f"block offload hook unreadable ({exc})"
    placement.skip = skip
    return placement, None


def _forward_refusal(module: Any, mode: str) -> Optional[str]:
    """Why ``module``'s own forward cannot record under ``mode`` offload, else None (see GraphedForward.__init__)."""
    try:
        from .diffusion_capture_safe import resolve as _capture_safe  # noqa: PLC0415
        safe, why = _capture_safe(type(module))
    except Exception:  # noqa: BLE001 - the stock forward is then what records
        return None
    if why is not None:
        return why
    if safe is None:
        return None
    if getattr(safe, "__unsloth_graph_plan__", None) is not None:
        if getattr(safe, "__unsloth_graph_placed__", None) is None:
            return f"{type(module).__name__} plans its graph steps but has no entry through the offload hooks"
        return None
    if mode == "group" and _innermost_class_forward(module) is None:
        return (
            f"{type(module).__name__} records only through its capture-safe forward, and the block-offload hook "
            "chain does not end in the class forward it would replace"
        )
    return None


def _plain_rewrite(module: Any) -> Any:
    """``module``'s capture-safe class-forward rewrite when it is a plain one (not a planned step), else None."""
    try:
        from .diffusion_capture_safe import resolve as _capture_safe  # noqa: PLC0415
        safe, _ = _capture_safe(type(module))
    except Exception:  # noqa: BLE001
        return None
    if safe is None or getattr(safe, "__unsloth_graph_plan__", None) is not None:
        return None
    return safe


def _innermost_class_forward(module: Any) -> Any:
    """The diffusers hook-chain reference whose ``forward`` is ``module``'s bound class forward, else None."""
    registry = getattr(module, "_diffusers_hook", None)
    refs = getattr(registry, "_fn_refs", None) or ()
    if not refs:
        return None
    ref = refs[0]
    if getattr(ref, "original_forward", None) is not None:
        return None  # a hook with its own new_forward owns the innermost call
    current = getattr(ref, "forward", None)
    owner = next((k for k in type(module).__mro__ if "forward" in vars(k)), None)
    if owner is None or getattr(current, "__self__", None) is not module:
        return None
    if getattr(current, "__func__", None) is not vars(owner)["forward"]:
        return None
    return ref


def arm_after_placement(
    pipe: Any,
    *,
    logger: Any = None,
    max_graphs: int = MAX_GRAPHS_PER_MODULE,
) -> tuple:
    """Arm the graphs once placement is final, in the slot each denoiser's offload leaves free. Returns
    ``(handles, reason)``: the reason names what kept a denoiser eager, else what engaged."""
    uninstall_all(getattr(pipe, "_unsloth_cuda_graphs", ()) or (), logger = logger)
    handles: list = []
    refusals: list = []
    modes: list = []
    for module in _denoiser_modules(pipe):
        placement, why = offload_placement(module)
        if why is not None:
            refusals.append(f"{type(module).__name__}: {why}")
            continue
        if placement is None and getattr(pipe, "_unsloth_cuda_graph_offload_only", False):
            refusals.append(
                f"{type(module).__name__}: the family records offloaded steps only, and it stays resident"
            )
            continue
        try:
            if placement is None:
                kind = (
                    GraphedCompiledCall
                    if getattr(module, "_compiled_call_impl", None) is not None
                    else GraphedForward
                )
                handles.append(kind(module, max_graphs = max_graphs, logger = logger).enable())
                modes.append("resident")
            else:
                handles.append(
                    GraphedForward(
                        module, max_graphs = max_graphs, logger = logger, placement = placement
                    ).enable()
                )
                modes.append(placement.mode)
        except Exception as exc:  # noqa: BLE001
            refusals.append(f"{type(module).__name__}: {exc}")
    installed = tuple(handles)
    try:
        pipe._unsloth_cuda_graphs = installed
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "handle stash", exc)
    if refusals:
        reason = "; ".join(refusals)
    else:
        labels = {
            "resident": "resident",
            "group": "block-streamed (copies recorded in the graph)",
            "model": "model offload",
        }
        reason = "captured per input shape: " + ", ".join(sorted({labels.get(m, m) for m in modes}))
    if logger is not None:
        logger.info(
            "diffusion.cuda_graph: armed on %d denoiser module(s) after placement (%s)",
            len(installed),
            reason,
        )
    return installed, reason


def _denoiser_modules(pipe: Any) -> list:
    """What the graph layer arms: every denoiser DiT, else a whole-compile-list U-Net."""
    from .diffusion_speed import _denoiser_dits, _denoiser_unet

    dits = _denoiser_dits(pipe)
    if dits:
        return dits
    unet = _denoiser_unet(pipe)
    return [unet] if unet is not None else []


def graph_eligible(
    target: Any,
    *,
    family: Any,
    pipe: Any,
    offload_active: bool,
    cache_active: bool,
    speed_mode: str,
    family_default: bool = True,
    logger: Any = None,
    offloaded: bool = False,
) -> tuple[bool, str]:
    """Whether this load may be graphed, and a short reason when it may not. Cheapest first. ``offloaded``: the
    denoiser moves (graphs armed after placement), which a family can opt into on its own (``offload_cuda_graph``)."""
    if cuda_graph_disabled():
        return False, _disabled_reason()

    device = getattr(target, "device", None)
    if device != "cuda":
        return False, f"device is {device or 'unknown'}"

    # ROCm reports device "cuda" and has no usable graph capture here; XPU / MPS / CPU never do.
    backend = getattr(target, "backend", "cuda")
    if backend != "cuda":
        return False, f"backend is {backend}"

    if offload_active:
        # Onload hooks move weights host to card, so a graph's baked pointers go stale.
        return False, "offload active"

    if cache_active:
        return False, "step cache active"

    mode = str(speed_mode or "").strip().lower()
    if mode not in ("default", "max"):
        return False, f"speed tier {mode or 'off'}"

    # Lazy: ``diffusion_speed`` imports this module, so a module-level import here is a cycle.
    from .diffusion_speed import _denoiser_dits, _denoiser_unet

    # A whole-compiled U-Net serves from ``_compiled_call_impl``: GraphedCompiledCall captures there.
    if not _denoiser_dits(pipe) and _denoiser_unet(pipe) is None:
        return False, "no denoiser transformer"

    # Sage (pip or hub build) under a replayed graph renders noise (FLUX.1-schnell, A100); ungraphed is correct.
    if any(
        getattr(m, "_unsloth_attention_backend", None) in ("sage", "sage_hub")
        for m in _denoiser_dits(pipe)
    ):
        return False, "SageAttention is not CUDA-graph safe"

    try:
        cuda = getattr(_torch(), "cuda", None)
        if cuda is None or not hasattr(cuda, "CUDAGraph") or not cuda.is_available():
            return False, "torch.cuda unavailable"
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "availability probe", exc)
        return False, "torch.cuda unavailable"

    wanted = bool(getattr(family, "supports_cuda_graph", family_default))
    wanted = wanted or (offloaded and bool(getattr(family, "offload_cuda_graph", False)))
    if not wanted and not _family_forced(family):
        return False, str(getattr(family, "cuda_graph_decline", None) or "family opts out")

    # A denoiser whose forward syncs the host would only invalidate its capture and run eager; decline
    # up front with the reason when its capture-safe rewrite did not apply.
    try:
        from .diffusion_capture_safe import resolve as _capture_safe  # noqa: PLC0415
        for module in _denoiser_dits(pipe):
            _, why = _capture_safe(type(module))
            if why:
                return False, why
    except Exception as exc:  # noqa: BLE001 - the capture itself still poisons safely
        _warn(logger, "capture-safe probe", exc)

    return True, "eligible"


def install_cuda_graphs(
    pipe: Any,
    *,
    logger: Any = None,
    max_graphs: int = MAX_GRAPHS_PER_MODULE,
) -> tuple:
    """Arm one ``GraphedForward`` per denoiser module; the first denoising step captures."""
    handles: list = []
    for module in _denoiser_modules(pipe):
        try:
            kind = (
                GraphedCompiledCall
                if getattr(module, "_compiled_call_impl", None) is not None
                else GraphedForward
            )
            handles.append(kind(module, max_graphs = max_graphs, logger = logger).enable())
        except Exception as exc:  # noqa: BLE001 - a second expert may fail without failing the load
            _warn(logger, f"install on {type(module).__name__}", exc)

    installed = tuple(handles)
    try:
        pipe._unsloth_cuda_graphs = installed
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "handle stash", exc)
    if installed and logger is not None:
        logger.info("diffusion.cuda_graph: armed on %d denoiser module(s)", len(installed))
    return installed


def set_bypass(handles: Any, on: bool) -> None:
    """Run eager without dropping graphs, for every handle."""
    for handle in handles or ():
        try:
            handle.set_bypass(on)
        except Exception:  # noqa: BLE001 - optimisation only
            pass


def reset_all(handles: Any) -> None:
    """Drop every captured graph, for every handle (the weights changed).

    Drops the pool token with the last graph: a stale token dies on the allocator's
    "use_count > 0 INTERNAL ASSERT FAILED", raised after ``torch.cuda.graph`` entered its side
    stream, so the thread is left off the default stream too."""
    for handle in handles or ():
        try:
            handle.reset()
        except Exception:  # noqa: BLE001 - optimisation only
            pass
    _drop_pool_if_unused()


def uninstall_all(handles: Any, *, logger: Any = None) -> None:
    """Free every handle and forget the shared pool if nothing else holds a graph. Idempotent."""
    for handle in handles or ():
        try:
            if logger is not None:
                logger.debug("diffusion.cuda_graph: %s", handle.describe())
            handle.free()
        except Exception:  # noqa: BLE001
            pass
    _drop_pool_if_unused()


_REFUSALS = ("refused_float", "refused_host_tensor", "refused_object")


def stats(handles: Any) -> dict:
    """JSON-safe aggregate over the handles, for the status payload."""
    out = {
        "graphs": 0,
        "captures": 0,
        "replays": 0,
        "eager_calls": 0,
        "fallbacks": 0,
        "cap_skips": 0,
        "invalidations": 0,
        "pool_bytes": 0,
        "planned_eager": 0,
        "shape_warmups": 0,
        "evictions": 0,
        "speed_eager": 0,
        **{field: 0 for field in _REFUSALS},
        "placements": [],
        "poisoned": False,
        "capture_error": None,
        # Offloaded only: the measured eager step and replay the speed check judged (ms), once judged.
        "eager_ms": None,
        "replay_ms": None,
    }
    for handle in handles or ():
        try:
            out["graphs"] += len(handle.cache)
            for field in (
                "captures",
                "replays",
                "eager_calls",
                "fallbacks",
                "cap_skips",
                "invalidations",
                "planned_eager",
                "shape_warmups",
                "evictions",
                "speed_eager",
                *_REFUSALS,
            ):
                out[field] += int(handle.stats.get(field, 0))
            out["pool_bytes"] = max(out["pool_bytes"], int(handle.stats.get("pool_bytes", 0)))
            if out["eager_ms"] is None and handle.stats.get("eager_ms") is not None:
                out["eager_ms"] = float(handle.stats["eager_ms"])
                out["replay_ms"] = float(handle.stats["replay_ms"])
            placement = getattr(handle, "placement", None)
            out["placements"].append("resident" if placement is None else placement.mode)
            if handle.poisoned:
                out["poisoned"] = True
            error = handle.capture_error
            if error and out["capture_error"] is None:
                out["capture_error"] = {
                    "type": str(error.get("type")),
                    "msg": str(error.get("msg")),
                }
        except Exception:  # noqa: BLE001
            pass
    return out


def never_engaged(handles: Any) -> Optional[str]:
    """Why the armed graphs never replayed a step; None before the first call or once any engaged."""
    if not handles:
        return None
    reasons = [h.why_off() for h in handles if callable(getattr(h, "why_off", None))]
    if reasons and len(reasons) == len(handles):
        return None if any(r is None for r in reasons) else "; ".join(reasons)
    s = stats(handles)
    refused = [
        str((h.capture_error or {}).get("msg"))
        for h in handles
        if (getattr(h, "capture_error", None) or {}).get("type") == "Refused"
    ]
    if refused and not s["captures"] and not s["replays"]:
        return f"armed, but every denoiser call so far ran eager: {refused[0]}"
    if refused and not s["graphs"]:
        # Recorded, then dropped for the load: the steps run eager from here on.
        return f"captured, then dropped: {refused[0]}; denoiser steps now run eager"
    if all(getattr(h, "poisoned", False) for h in handles):
        error = s["capture_error"] or {}
        return f"capture failed ({error.get('type') or 'error'}); every denoiser step runs eager"
    if s["captures"] or s["replays"] or not s["eager_calls"]:
        return None
    if s["eager_calls"] == s["speed_eager"]:
        # Every eager call so far was an offloaded step timed for the speed check, before its first capture.
        return None
    parts = [
        f"{s[field]} {label}"
        for field, label in (
            ("refused_object", "with a non-tensor argument"),
            ("refused_float", "with a float argument"),
            ("refused_host_tensor", "with a host tensor"),
            ("cap_skips", "past the graph cap"),
        )
        if s[field]
    ]
    detail = ", ".join(parts) if parts else "bypassed"
    return f"armed, but all {s['eager_calls']} denoiser call(s) so far ran eager ({detail})"


def live_status(resolved: Any, speed_optims: Any, handles: Any) -> tuple:
    """Copies, never mutates: the load-time record stays as recorded."""
    why = never_engaged(handles)
    optims = list(speed_optims or ())
    if why is None:
        return resolved, optims
    optims = [o for o in optims if o != "cuda_graph"]
    if isinstance(resolved, dict) and isinstance(resolved.get("cuda_graph"), dict):
        resolved = {
            **resolved,
            "cuda_graph": {**resolved["cuda_graph"], "value": "off", "reason": why},
        }
    return resolved, optims


# Per-block graphs (diffusion_block_graph) for the loads the whole-forward recording cannot hold.

WHOLE_REASON = "denoiser step captured per input shape, replayed bit-identically"


def _block_basics(
    target: Any, family: Any, cache_engaged: bool, speed_mode: str, family_default: bool
) -> Optional[str]:
    """The whole-forward checks that also bind per-block recording (everything except offload and capture-safety)."""
    if cuda_graph_disabled():
        return _disabled_reason()
    if getattr(target, "device", None) != "cuda":
        return f"device is {getattr(target, 'device', None) or 'unknown'}"
    backend = getattr(target, "backend", "cuda")
    if backend != "cuda":
        return f"backend is {backend}"
    if cache_engaged:
        return "step cache active"
    mode = str(speed_mode or "").strip().lower()
    if mode not in ("default", "max"):
        return f"speed tier {mode or 'off'}"
    if not bool(getattr(family, "supports_cuda_graph", family_default)):
        return "family opts out"
    try:
        torch = _torch()
        if not torch.cuda.is_available() or not hasattr(torch.cuda, "graph_pool_handle"):
            return "torch.cuda unavailable"
    except Exception:  # noqa: BLE001
        return "torch.cuda unavailable"
    return None


def arm_block_graphs(
    pipe: Any,
    applied: dict,
    *,
    target: Any,
    family: Any,
    hooked: bool,
    pinned: bool = False,
    cache_engaged: bool = False,
    speed_mode: str = "default",
    family_default: bool = True,
    logger: Any = None,
) -> tuple:
    """After placement: keep a whole-forward recording where it holds, else record per block.

    Per-block recording is the default only where every block stays on the device (``not hooked``, or ``pinned``
    resident under its hooks); streamed and model-offloaded denoisers need ``UNSLOTH_DIFFUSION_BLOCK_GRAPHS=1``.

    A whole forward cannot be recorded once an offload hook moves the denoiser (``hooked``) or when the forward is
    not capture-safe (Qwen-Image-2.1's prefix K/V object); its repeated blocks still can, keyed by where their weights
    sit. Updates ``applied["cuda_graph"]`` and the pipe's reason / handles; returns the handles now armed."""
    handles = tuple(getattr(pipe, "_unsloth_cuda_graphs", ()) or ())
    whole = [h for h in handles if isinstance(h, GraphedForward)]
    prior = str(getattr(pipe, "_unsloth_cuda_graph_reason", None) or "")
    step = [h for h in whole if h.placement is not None or h.plan is not None]
    if whole and len(step) == len(whole):
        # The whole step stays primary; where per-block graphs are the default they arm for the calls it leaves eager.
        pipe._unsloth_cuda_graph_mode = "step"
        why = _block_basics(target, family, cache_engaged, speed_mode, family_default)
        if why is None:
            _attach_block_fallback(pipe, step, target, hooked, pinned, logger)
        return handles
    if not hooked and (whole or "not capture-safe" not in prior):
        # Nothing moves the denoiser: the speed layer's decision stands, except a forward that is not capture-safe,
        # whose blocks still record.
        return handles
    why = _block_basics(target, family, cache_engaged, speed_mode, family_default)
    if whole:
        uninstall_all(whole, logger = logger)
    pipe._unsloth_cuda_graph_mode = None
    if why is not None:
        _set_reason(pipe, why)
        applied["cuda_graph"] = False
        pipe._unsloth_cuda_graphs = ()
        return ()
    try:
        from .diffusion_block_graph import (
            MODEL_OFFLOAD_REASON,
            OPT_IN_REASON,
            block_graphs_disabled,
            block_graphs_requested,
            install_block_graphs,
        )
        from .diffusion_speed import _denoiser_dits
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "block graph import", exc)
        applied["cuda_graph"] = False
        pipe._unsloth_cuda_graphs = ()
        _set_reason(pipe, "offload active" if hooked else prior or "block graphs unavailable")
        return ()
    if any(
        getattr(t, "_unsloth_attention_backend", None) in ("sage", "sage_hub")
        for t in _denoiser_dits(pipe)
    ):
        applied["cuda_graph"] = False
        pipe._unsloth_cuda_graphs = ()
        _set_reason(pipe, "SageAttention is not CUDA-graph safe")
        return ()
    model_offload = any(getattr(t, "_hf_hook", None) is not None for t in _denoiser_dits(pipe))
    stays = (not hooked or bool(pinned)) and not model_offload
    if block_graphs_disabled() or not (stays or block_graphs_requested()):
        applied["cuda_graph"] = False
        pipe._unsloth_cuda_graphs = ()
        if block_graphs_disabled():
            _set_reason(pipe, "disabled by UNSLOTH_DIFFUSION_BLOCK_GRAPHS=0")
        else:
            _set_reason(pipe, MODEL_OFFLOAD_REASON if model_offload else OPT_IN_REASON)
        return ()
    armed: list = []
    reasons: list = []
    for transformer in _denoiser_dits(pipe):
        try:
            handle, reason = install_block_graphs(
                transformer, device = getattr(target, "torch_device", None), logger = logger
            )
        except Exception as exc:  # noqa: BLE001
            _warn(logger, "block graph install", exc)
            handle, reason = None, f"install failed ({type(exc).__name__})"
        if handle is not None:
            armed.append(handle)
        else:
            reasons.append(reason)
    pipe._unsloth_cuda_graphs = tuple(armed)
    applied["cuda_graph"] = bool(armed)
    if armed:
        where = (
            "pinned denoiser"
            if hooked and pinned
            else "offloaded denoiser"
            if hooked
            else "denoiser"
        )
        slots = sum(int(getattr(h, "slots_mib", 0) or 0) for h in armed)
        _set_reason(
            pipe,
            f"{where} recorded per block, keyed by input shape and weight placement"
            + (f"; streamed blocks read a {slots} MiB slot ring" if slots else ""),
        )
        pipe._unsloth_cuda_graph_mode = "blocks"
    else:
        _set_reason(
            pipe, "; ".join(r for r in reasons if r) or "no denoiser block could be recorded"
        )
    return tuple(armed)


def _attach_block_fallback(
    pipe: Any, step: list, target: Any, hooked: bool, pinned: bool, logger: Any
) -> None:
    try:
        from .diffusion_block_graph import (
            block_graphs_disabled,
            block_graphs_requested,
            install_block_graphs,
        )
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "block graph import", exc)
        return
    if block_graphs_disabled():
        return
    for handle in step:
        model_offload = getattr(handle.placement, "mode", None) == "model"
        stays = (not hooked or bool(pinned)) and not model_offload
        if not (stays or block_graphs_requested()):
            continue

        def arm(module: Any = handle.module) -> Any:
            armed, _why = install_block_graphs(
                module, device = getattr(target, "torch_device", None), logger = logger
            )
            if armed is not None:
                _set_reason(
                    pipe,
                    str(getattr(pipe, "_unsloth_cuda_graph_reason", None) or "")
                    + "; the steps it leaves eager record per block",
                )
            return armed

        handle.fallback = arm


def _set_reason(pipe: Any, reason: str) -> None:
    try:
        pipe._unsloth_cuda_graph_reason = reason
    except Exception:  # noqa: BLE001
        pass


def status_reason(pipe: Any, on: bool) -> str:
    """The cuda_graph badge text: the block layer's own sentence when it armed, else why graphs are off."""
    if on:
        mode = getattr(pipe, "_unsloth_cuda_graph_mode", None)
        reason = str(getattr(pipe, "_unsloth_cuda_graph_reason", None) or "")
        if mode == "blocks":
            return reason or WHOLE_REASON
        head, _, tail = reason.partition("; ")
        if mode == "step" and head.startswith("captured") and "resident" not in head:
            # "denoiser step captured per input shape: block-streamed (copies recorded in the graph), replayed ..."
            return f"denoiser step {head}, replayed bit-identically" + (f"; {tail}" if tail else "")
        return WHOLE_REASON + (f"; {tail}" if mode == "step" and tail else "")
    return str(getattr(pipe, "_unsloth_cuda_graph_reason", None) or "speed tier does not capture")


def held_bytes(handles: Any) -> int:
    """Device bytes the armed graphs hold outside the caching allocator's reach (pools, static buffers, slot ring)."""
    total = 0
    for handle in handles or ():
        fn = getattr(handle, "held_bytes", None)
        if callable(fn):
            try:
                total += int(fn())
            except Exception:  # noqa: BLE001
                pass
    return total

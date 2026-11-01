# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-block CUDA graphs keyed by inputs and weight addresses; moved weights re-record once; offload
hooks stay eager; static inputs shared since replays never overlap"""

from __future__ import annotations

import os
import traceback
from collections import OrderedDict
from typing import Any, Callable, Optional

from .diffusion_bg_compile import capture_suppressed as _bg_capture_suppressed
from .diffusion_bg_compile import eager_forced as _bg_eager_forced


def _step_recording() -> bool:
    try:
        from .diffusion_cuda_graph import step_recording
    except Exception:  # noqa: BLE001
        return False
    return step_recording()


BLOCK_GRAPHS_ENV = "UNSLOTH_DIFFUSION_BLOCK_GRAPHS"

# Recordings kept per block (input layouts x weight placements); the least recently replayed goes first.
MAX_GRAPHS_PER_BLOCK = 6

# Distinct weight placements a block may see without one replay before it stops trying (address churn: a streaming
# path without the slot ring, or disk offload). Each attempt costs one eager call, never a recording.
MAX_UNREPLAYED_PLACEMENTS = 6

_OFF = ("0", "off", "false", "no")


def _torch():
    import torch
    return torch


def block_graphs_disabled() -> bool:
    return (os.environ.get(BLOCK_GRAPHS_ENV) or "").strip().lower() in _OFF


def block_graphs_requested() -> bool:
    """Forced on everywhere, streamed and model-offloaded denoisers included."""
    return (os.environ.get(BLOCK_GRAPHS_ENV) or "").strip().lower() in ("1", "on", "true", "yes")


# Streamed steps are copy-bound; model offload re-records every render.
OPT_IN_REASON = (
    "offloaded denoiser streams its blocks: per-block CUDA graphs measured no faster (streamed steps are "
    "copy-bound) and hold extra VRAM; set " + BLOCK_GRAPHS_ENV + "=1 to record per block"
)
MODEL_OFFLOAD_REASON = (
    "model offload re-uploads the denoiser every render, so every block would record again; set "
    + BLOCK_GRAPHS_ENV
    + "=1 to record per block"
)


# Auto engages FBCache only at 20+ steps, since its win scales with step count: "dev" schedules qualify, distilled turbo never does


def _kv_layer_cache(obj: Any) -> bool:
    name = type(obj).__name__
    return name.endswith("KVLayerCache") and hasattr(obj, "k") and hasattr(obj, "v")


def _flatten(obj: Any, out: list) -> tuple:
    torch = _torch()
    if torch.is_tensor(obj):
        out.append(obj)
        return ("t", len(out) - 1)
    if isinstance(obj, (list, tuple)):
        return ("l", isinstance(obj, tuple), [_flatten(o, out) for o in obj])
    if isinstance(obj, dict):
        return ("d", [(k, _flatten(v, out)) for k, v in obj.items()])
    if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
        return ("v", obj)
    if _kv_layer_cache(obj) and torch.is_tensor(obj.k) and torch.is_tensor(obj.v):
        return ("kv", type(obj), _flatten(obj.k, out), _flatten(obj.v, out))
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
    if kind == "kv":
        cache = spec[1].__new__(spec[1])
        cache.k = _rebuild(spec[2], tensors)
        cache.v = _rebuild(spec[3], tensors)
        return cache
    return spec[1]


def _walk(obj: Any, live: list) -> tuple:
    """Hashable key of a call tree, appending its tensors to ``live`` (compiled blocks guard on inference mode)."""
    torch = _torch()
    if torch.is_tensor(obj):
        live.append(obj)
        return (
            "t",
            tuple(obj.shape),
            tuple(obj.stride()),
            obj.dtype,
            obj.device.type,
            obj.device.index,
            bool(obj.is_inference()),
        )
    if isinstance(obj, (list, tuple)):
        return ("l", isinstance(obj, tuple), tuple(_walk(o, live) for o in obj))
    if isinstance(obj, dict):
        return ("d", tuple((k, _walk(v, live)) for k, v in obj.items()))
    if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
        return ("v", obj)
    if _kv_layer_cache(obj) and torch.is_tensor(obj.k) and torch.is_tensor(obj.v):
        return ("kv", type(obj), _walk(obj.k, live), _walk(obj.v, live))
    return ("o", type(obj).__name__, id(obj))


def _unwalk(key: tuple, tensors: Any) -> Any:
    kind = key[0]
    if kind == "t":
        return next(tensors)
    if kind == "l":
        values = [_unwalk(k, tensors) for k in key[2]]
        return tuple(values) if key[1] else values
    if kind == "d":
        return {k: _unwalk(v, tensors) for k, v in key[1]}
    if kind == "kv":
        cache = key[1].__new__(key[1])
        cache.k = _unwalk(key[2], tensors)
        cache.v = _unwalk(key[3], tensors)
        return cache
    return key[1]


def graph_key(obj: Any) -> tuple:
    return _walk(obj, [])


def _refusal(key: tuple, kwargs: dict) -> Optional[str]:
    def walk(k: tuple) -> Optional[str]:
        kind = k[0]
        if kind == "o":
            return "object"
        if kind == "v" and isinstance(k[1], float):
            return "float"
        if kind == "l":
            for sub in k[2]:
                why = walk(sub)
                if why:
                    return why
        if kind == "d":
            for _, sub in k[1]:
                why = walk(sub)
                if why:
                    return why
        return None

    why = walk(key)
    if why:
        return why
    # A prefix K/V cache is only read on a cached step; the prefill writes it from Python.
    if _has_kv(key) and kwargs.get("kv_cache_mode") != "cached":
        return "kv_write"
    return None


def _has_kv(key: tuple) -> bool:
    kind = key[0]
    if kind == "kv":
        return True
    if kind == "l":
        return any(_has_kv(k) for k in key[2])
    if kind == "d":
        return any(_has_kv(k) for _, k in key[1])
    return False


def _is_wrapper_subclass(t: Any) -> bool:
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass
        return bool(is_traceable_wrapper_subclass(t))
    except Exception:  # noqa: BLE001
        return False


class _WeightView:
    """The block's weights as device addresses, through every level of torchao inner tensors (a wrapper's own
    ``data_ptr`` is 0; torchao 0.17's int8 nests two)."""

    __slots__ = ("tensors",)

    def __init__(self, module: Any) -> None:
        seen: set = set()
        tensors: list = []
        for t in list(module.parameters()) + list(module.buffers()):
            if id(t) not in seen:
                seen.add(id(t))
                tensors.append(t)
        self.tensors = tensors

    def placement(self, device_index: Optional[int]) -> Optional[tuple]:
        """Data pointers of every plain tensor under the weights, or None when one is off ``device_index``."""
        ptrs: list = []
        for t in self.tensors:
            for p in _leaves(t):
                dev = p.device
                if dev.type != "cuda" or (device_index is not None and dev.index != device_index):
                    return None
                ptrs.append(p.data_ptr())
        return tuple(ptrs)


def _leaves(t: Any) -> list:
    if not _is_wrapper_subclass(t):
        return [t]
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:  # noqa: BLE001 - unflattenable: key on what it reports
        return [t]
    out: list = []
    for n in names:
        x = getattr(t, n, None)
        if x is not None:
            out.extend(_leaves(x))
    return out


_UNSEEN = object()


class _Entry:
    __slots__ = ("graph", "static_in", "static_out", "out_spec", "replays", "slots", "pool")


import weakref

_LIVE: "weakref.WeakSet" = weakref.WeakSet()


def pool_bytes(device: Optional[int] = None) -> int:
    """Device bytes held in every live block-graph pool (on CUDA ``device`` when given): reserved by the allocator yet
    never free for other work, so the memory guard must not credit them back as reclaimable."""
    total = 0
    for shared in tuple(_LIVE):
        index = getattr(shared, "device_index", None)
        if device is not None and index is not None and index != device:
            continue
        total += int(getattr(shared, "pool_bytes", 0) or 0)
    return total


class _Shared:
    """Pool, capture stream and static buffers per (block class, layout), shared by one denoiser's blocks."""

    def __init__(
        self,
        device_index: Optional[int],
        logger: Any = None,
    ) -> None:
        self.device_index = device_index
        self.logger = logger
        self.root: Any = None  # the denoiser, for its NVFP4 per-step precision branch
        self.pool = None
        self.stream = None
        self.static_in: dict = {}
        self.static_out: dict = {}
        self.static_refs: dict = {}  # slot -> recordings reading its buffers
        self.warmed: set = set()
        self.pool_bytes = 0
        self.static_bytes = 0
        self.strict_device = False
        _LIVE.add(self)

    def capture_stream(self) -> Any:
        if self.stream is None:
            torch = _torch()
            self.stream = torch.cuda.Stream(device = self.device_index)
        return self.stream

    def graph_pool(self) -> Any:
        if self.pool is None:
            torch = _torch()
            self.pool = torch.cuda.graph_pool_handle()
        return self.pool

    def statics_for(self, slot: tuple, live: list) -> list:
        buffers = self.static_in.get(slot)
        if buffers is None:
            buffers = [_static_like(t) for t in live]
            self.static_in[slot] = buffers
            self.static_bytes += sum(_nbytes(b) for b in buffers)
        return buffers

    def outputs_for(self, slot: tuple, metas: list) -> list:
        buffers = self.static_out.get(slot)
        if buffers is None:
            torch = _torch()
            buffers = []
            for shape, stride, dtype, device, inference in metas:
                with torch.inference_mode(inference):
                    buffers.append(torch.empty_strided(shape, stride, dtype = dtype, device = device))
            self.static_out[slot] = buffers
            self.static_bytes += sum(_nbytes(b) for b in buffers)
        return buffers

    def hold(self, slots: tuple) -> None:
        for slot in slots:
            self.static_refs[slot] = self.static_refs.get(slot, 0) + 1

    def drop(self, slots: tuple) -> None:
        """Free the static buffers of ``slots`` no recording reads any more (an evicted layout)."""
        for slot in slots:
            left = self.static_refs.get(slot, 0) - 1
            if left > 0:
                self.static_refs[slot] = left
                continue
            self.static_refs.pop(slot, None)
            buffers = self.static_in.pop(slot, None) or self.static_out.pop(slot, None) or ()
            self.static_bytes = max(0, self.static_bytes - sum(_nbytes(b) for b in buffers))
            self.warmed.discard(slot)

    def release(self) -> None:
        self.static_in.clear()
        self.static_out.clear()
        self.static_refs.clear()
        self.warmed.clear()
        self.pool = None
        self.pool_bytes = 0
        self.static_bytes = 0


def _static_like(t: Any) -> Any:
    """Buffer must match t in shape, stride, dtype, inference mode and offset, or the block recompiles"""
    torch = _torch()
    shape, stride, offset = tuple(t.shape), tuple(t.stride()), int(t.storage_offset())
    extent = 0 if t.numel() == 0 else 1 + sum((n - 1) * st for n, st in zip(shape, stride) if n > 0)
    with torch.inference_mode(bool(t.is_inference())):
        base = torch.empty(offset + extent, dtype = t.dtype, device = t.device)
        return base.as_strided(shape, stride, offset)


def _nbytes(t: Any) -> int:
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:  # noqa: BLE001
        return 0


def _meta(t: Any) -> tuple:
    return (tuple(t.shape), tuple(t.stride()), t.dtype, t.device, bool(t.is_inference()))


_LIVE_BLOCKS: "weakref.WeakSet" = weakref.WeakSet()


def live_block_graph_count(device_index: Optional[int]) -> int:
    """Per-block graphs alive on CUDA device ``device_index`` (``diffusion_cuda_graph.live_graph_count``)."""
    return sum(
        len(b.cache)
        for b in tuple(_LIVE_BLOCKS)
        if getattr(b, "_unsloth_capture_device", None) == device_index
    )


class BlockGraph:
    """Replaces one block's compute callable (below its offload hooks) with a recording keyed by inputs and weight
    placement. Any refusal or failure runs ``compute`` itself, so the block is never worse than ungraphed."""

    def __init__(
        self,
        block: Any,
        compute: Callable,
        shared: _Shared,
        *,
        max_graphs: int = MAX_GRAPHS_PER_BLOCK,
    ) -> None:
        self.block = block
        self.compute = compute
        self.shared = shared
        self.cls = type(block).__name__
        self.weights = _WeightView(block)
        self.max_graphs = int(max_graphs)
        self.cache: "OrderedDict[tuple, _Entry]" = OrderedDict()
        self.enabled = True
        self.bypassed = False
        self.poisoned = False
        self.capture_error: Optional[dict] = None
        # (layout, placement) -> output meta of its first call; several, since CFG alternates two text lengths
        self.seen: "OrderedDict[tuple, tuple]" = OrderedDict()
        self.unreplayed_placements = 0
        self.churned = False
        self.protect: Any = None  # None until resolved, then False or the NVFP4 step controller
        self.segment: Any = None
        self.seg_index = 0
        self.owner: Any = None
        self.placements: set = set()
        self.refusals: dict = {}
        self.stats = {
            "captures": 0,
            "recaptures": 0,
            "replays": 0,
            "eager_calls": 0,
            "warmups": 0,
            "fallbacks": 0,
            "evictions": 0,
            "refused_float": 0,
            "refused_object": 0,
            "refused_host_input": 0,
            "refused_kv_write": 0,
            "refused_host_weight": 0,
            "refused_grad": 0,
            "refused_output": 0,
            "address_churn": 0,
        }
        try:
            from functools import update_wrapper
            update_wrapper(self, compute)
        except Exception:  # noqa: BLE001
            pass

    def set_bypass(self, on: bool) -> "BlockGraph":
        self.bypassed = bool(on)
        return self

    def reset(self) -> "BlockGraph":
        """Drop every recording (the weights changed: a LoRA load or scale, an unload) and re-read which tensors the
        block owns, since an adapter adds parameters the placement key must cover."""
        self._forget_all()
        try:
            self.weights = _WeightView(self.block)
        except Exception:  # noqa: BLE001
            pass
        self.seen.clear()
        self.unreplayed_placements = 0
        self.placements.clear()
        self.refusals.clear()
        return self

    def _eager(self, args: tuple, kwargs: dict) -> Any:
        self.stats["eager_calls"] += 1
        return self.compute(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _bg_capture_suppressed() or _step_recording():
            # a whole-step graph above records this block's kernels (diffusion_cuda_graph), not a nested replay
            return self.compute(*args, **kwargs)
        if (
            not self.enabled
            or self.bypassed
            or self.poisoned
            or self.churned
            or _bg_eager_forced()
            or _step_cache_live(self.shared.root)
        ):
            return self._eager(args, kwargs)
        segment = self.segment
        if segment is not None:
            out = segment.call(self.seg_index, args, kwargs)
            if out is not _NOT_HANDLED:
                return out
        torch = _torch()
        if torch.is_grad_enabled():
            self.stats["refused_grad"] += 1
            return self._eager(args, kwargs)
        live: list = []
        try:
            key = _walk((args, kwargs), live)
            why = self.refusals.get(key, _UNSEEN)
            if why is _UNSEEN:
                why = _refusal(key, kwargs)
                if len(self.refusals) < 64:
                    self.refusals[key] = why
        except Exception:  # noqa: BLE001
            why = "object"
            key = None
        if why:
            self.stats["refused_" + why] += 1
            return self._eager(args, kwargs)
        placement = self.weights.placement(self.shared.device_index)
        if placement is None:
            self.stats["refused_host_weight"] += 1
            return self._eager(args, kwargs)
        if any(t.device.type != "cuda" for t in live):
            # A capture never records a host read, so a replay would keep the recorded value.
            self.stats["refused_host_input"] += 1
            return self._eager(args, kwargs)
        if self.shared.strict_device and (
            torch.cuda.current_device() != self.shared.device_index
            or any(t.device.index != self.shared.device_index for t in live)
        ):
            self.stats["refused_host_input"] += 1
            return self._eager(args, kwargs)
        if self.protect is None:
            self.protect = self._protect_controller()
        if self.protect:
            # One recording per precision branch: a W4A4 replay at a W4A16 step would skip the protection.
            from .diffusion_cuda_graph import protect_graph_key
            placement = (placement, protect_graph_key(self.protect))
        try:
            extra = _root_key_extra(self.shared.root)
        except Exception:  # noqa: BLE001 - a branch flag that cannot be read: nothing safe to replay
            self.stats["refused_object"] += 1
            return self._eager(args, kwargs)
        if extra is not None:
            placement = (placement, extra)
        full = (key, placement)
        entry = self.cache.get(full)
        if entry is None:
            if self.owner is not None and self.owner.scope_sealed():
                self.stats["length_eager"] = self.stats.get("length_eager", 0) + 1
                return self._eager(args, kwargs)
            if full not in self.seen:
                # First call at this layout and placement: run it for real (the warm-up), record on the next one.
                return self._first_sighting(full, args, kwargs)
            entry = self._record(full, live)
            if entry is None:
                return self._eager(args, kwargs)
        else:
            self.cache.move_to_end(full)
        return self._replay(entry, live)

    def _protect_controller(self) -> Any:
        root = self.shared.root
        try:
            from .diffusion_cuda_graph import _protect_keyed

            if root is None or not _protect_keyed(root):
                return False
            from .diffusion_nvfp4_protect import module_controller

            return module_controller(root)
        except Exception:  # noqa: BLE001
            return False

    def _first_sighting(self, full: tuple, args: tuple, kwargs: dict) -> Any:
        placement = full[1]
        if placement not in self.placements:
            self.placements.add(placement)
            self.unreplayed_placements += 1
            if self.unreplayed_placements > MAX_UNREPLAYED_PLACEMENTS:
                self.churned = True
                self.stats["address_churn"] += 1
                return self._eager(args, kwargs)
        out = self._eager(args, kwargs)
        try:
            flat: list = []
            spec = _flatten(out, flat)
            self.seen[full] = (spec, [_meta(t) for t in flat])
            while len(self.seen) > 4:
                self.seen.popitem(last = False)
        except Exception:  # noqa: BLE001 - an output the graph cannot hand back (a dataclass, an object)
            self.stats["refused_output"] += 1
        return out

    def _record(self, full: tuple, live: list) -> Optional[_Entry]:
        from .diffusion_cuda_graph import (
            CaptureRefused,
            _capturing,
            refuse_if_exhausted,
            retire_failed_capture,
        )

        torch = _torch()
        key = full[0]
        slot = (self.cls, key)
        out_spec, metas = self.seen.pop(full)
        entry = _Entry()
        out_slot = (self.cls, key, "out", tuple(metas))
        try:
            entry.static_in = self.shared.statics_for(slot, live)
            entry.static_out = self.shared.outputs_for(out_slot, metas)
            for dst, src in zip(entry.static_in, live):
                dst.copy_(src)
            static_args, static_kwargs = _unwalk(key, iter(entry.static_in))
            if slot not in self.shared.warmed:
                # Any guard the static buffers trip recompiles here, never inside the recording.
                self.compute(*static_args, **static_kwargs)
                self.shared.warmed.add(slot)
                self.stats["warmups"] += 1
            current = torch.cuda.current_stream()
            stream = self.shared.capture_stream()
            stream.wait_stream(current)
            refuse_if_exhausted(self.shared.logger)
            graph = torch.cuda.CUDAGraph()
            pool = self.shared.graph_pool()
            with torch.cuda.stream(stream):
                with _capturing():
                    # read after the entry, which first hands back any retired graph pool
                    before = torch.cuda.memory_reserved()
                    try:
                        # Inside the try: a capture_begin that raised after taking the pool is abandoned too.
                        graph.capture_begin(pool = pool, capture_error_mode = "thread_local")
                        try:
                            out = self.compute(*static_args, **static_kwargs)
                            flat: list = []
                            spec = _flatten(out, flat)
                            if spec != out_spec or len(flat) != len(entry.static_out):
                                raise RuntimeError("block output layout changed between calls")
                            for dst, src in zip(entry.static_out, flat):
                                dst.copy_(src)
                        finally:
                            graph.capture_end()
                    except BaseException as exc:
                        # Later blocks record into a fresh pool; this one stays with the graphs already in it.
                        retire_failed_capture(graph, pool, exc)
                        if self.shared.pool == pool:
                            self.shared.pool = None
                        raise
                del out, flat
            current.wait_stream(stream)
            self.shared.pool_bytes += max(0, torch.cuda.memory_reserved() - before)
        except CaptureRefused:
            # Refused before capture_begin: nothing recorded, the pool untouched. Eager now, record on a later call.
            self.seen[full] = (out_spec, metas)
            self.stats["capture_refusals"] = self.stats.get("capture_refusals", 0) + 1
            return None
        except Exception as exc:  # noqa: BLE001 - a block that cannot record runs its compute for the load's life
            self._poison(exc)
            return None
        entry.slots = (slot, out_slot)
        self.shared.hold(entry.slots)
        entry.graph = graph
        entry.pool = pool
        entry.out_spec = out_spec
        entry.replays = 0
        if any(k[0] == full[0] for k in self.cache):
            self.stats["recaptures"] += 1
        self.cache[full] = entry
        self.stats["captures"] += 1
        if self.owner is not None:
            self.owner.note_capture()
        from .diffusion_cuda_graph import capture_device, note_capture

        self._unsloth_capture_device = capture_device()
        # counted live before the generation moves, so a reader seeing an unchanged generation never misses it
        _LIVE_BLOCKS.add(self)
        note_capture(self._unsloth_capture_device)
        while len(self.cache) > self.max_graphs:
            _drop_entries(self, [self.cache.popitem(last = False)[1]], evicting = True)
            self.stats["evictions"] += 1
        return entry

    def _forget(self, entry: _Entry) -> None:
        entry.graph = None
        self.shared.drop(getattr(entry, "slots", ()))

    def _forget_all(self) -> None:
        entries = list(self.cache.values())
        self.cache.clear()
        _drop_entries(self, entries)

    def _replay(self, entry: _Entry, live: list) -> Any:
        for dst, src in zip(entry.static_in, live):
            if dst.data_ptr() != src.data_ptr():
                dst.copy_(src)
        entry.graph.replay()
        entry.replays += 1
        self.stats["replays"] += 1
        self.unreplayed_placements = 0
        self.placements.clear()
        return _rebuild(entry.out_spec, [t.clone() for t in entry.static_out])

    def _poison(self, exc: BaseException) -> None:
        self.capture_error = {
            "type": type(exc).__name__,
            "msg": str(exc)[:4000],
            "traceback": traceback.format_exc()[-6000:],
        }
        self.poisoned = True
        self._forget_all()
        self.stats["fallbacks"] += 1
        logger = self.shared.logger
        if logger is not None:
            logger.warning(
                "diffusion.block_graph: recording %s failed (%s: %s); this block runs ungraphed",
                self.cls,
                type(exc).__name__,
                exc,
            )
        exc.__traceback__ = None


SEGMENT_GRAPHS_ENV = "UNSLOTH_DIFFUSION_SEGMENT_GRAPHS"
_NOT_HANDLED = object()


def segment_graphs_disabled() -> bool:
    return (os.environ.get(SEGMENT_GRAPHS_ENV) or "").strip().lower() in _OFF


class _SegRecord:
    """One eager pass over a segment: each block's call layout, where each of its input tensors came from (an earlier
    block's output, or one of the first block's inputs), and its output layout."""

    __slots__ = ("keys", "prov", "out_specs")

    def __init__(self) -> None:
        self.keys: list = []
        self.prov: list = []
        self.out_specs: list = []


class _SegEntry:
    __slots__ = ("graph", "static_in", "outputs", "record", "slots", "replays", "pool")


def _root_key_extra(root: Any) -> Any:
    """Branch state (GRAPH_KEY_EXTRA_ATTR) keys block graphs, so branches never replay each other's."""
    from .diffusion_cuda_graph import GRAPH_KEY_EXTRA_ATTR

    extra = getattr(root, GRAPH_KEY_EXTRA_ATTR, None) if root is not None else None
    return ("x", extra()) if callable(extra) else None


def _step_cache_live(root: Any) -> bool:
    """Whether a step cache (First-Block-Cache) is engaged on the denoiser: its hooks decide inside the forward, per
    step and per cond / uncond context, which a recording cannot see. Belt to the caller's per-chunk ``set_bypass``,
    as ``GraphedForward`` does: an auto cache that turns on at generate time keeps every block eager."""
    try:
        return bool(getattr(root, "_unsloth_step_cache", None)) if root is not None else False
    except Exception:  # noqa: BLE001
        return False


def _lazy_chain(root: Any) -> bool:
    """Whether this forward of a split ``root`` routes lazily: its class is one the split verified to pass each block's
    outputs straight to the next block, and the call carries no residual input. Only such a chain may replay as one
    segment, since a segment cannot see an op between blocks that rewrites a tensor in place."""
    try:
        from .diffusion_layer_split import layer_split_of
        split = layer_split_of(root)
    except Exception:  # noqa: BLE001
        return False
    return bool(split is not None and split.lazy_now)


def _version(t: Any) -> Any:
    # an inference tensor carries no version counter
    try:
        return None if t.is_inference() else t._version
    except Exception:  # noqa: BLE001
        return None


class SegmentGraph:
    """One CUDA graph over a run of blocks a layer split keeps on one card, back to back in one block list: the first
    block's call replays the whole run, and each later block's call checks that it got exactly the tensors the run
    handed on (the previous block's outputs, the first block's other inputs) and returns its recorded outputs. Any
    other call (an op between blocks rewrote a tensor, an eager boundary copied it) leaves the segment to the per-block
    path for the rest of that forward, which computes the block itself.

    Lifecycle per (layout of the first block's inputs, weight placement): the first forward records the run eagerly,
    the second warms it up once and captures it, every later one replays. All on the card's own stream and pool."""

    def __init__(
        self,
        graphs: list,
        shared: _Shared,
        max_graphs: int = MAX_GRAPHS_PER_BLOCK,
    ) -> None:
        self.graphs = list(graphs)
        self.shared = shared
        self.max_graphs = int(max_graphs)
        self.cache: "OrderedDict[tuple, _SegEntry]" = OrderedDict()
        self.seen: "OrderedDict[tuple, _SegRecord]" = OrderedDict()
        self.refused: set = set()
        self.poisoned = False
        self.enabled = True  # off: every block takes its own graph's path; recordings are kept
        self.capture_error: Optional[dict] = None
        self.owner: Any = None
        self._reset_forward()
        self.stats = {
            "segment_records": 0,
            "segment_captures": 0,
            "segment_replays": 0,
            "segment_warmups": 0,
            "segment_breaks": 0,
            "segment_refusals": 0,
            "segment_fallbacks": 0,
        }
        for i, g in enumerate(self.graphs):
            g.segment = self
            g.seg_index = i
        _LIVE_BLOCKS.add(self)

    def _reset_forward(self) -> None:
        self.pos: Optional[int] = None
        self.mode: Optional[str] = None
        self.cur: Any = None
        self.ext: list = []
        self.returned: list = []
        self.versions: list = []

    def reset(self) -> None:
        entries = list(self.cache.values())
        self.cache.clear()
        _drop_entries(self, entries)
        self.seen.clear()
        self.refused.clear()
        self._reset_forward()

    def detach(self) -> None:
        self.reset()
        for g in self.graphs:
            if g.segment is self:
                g.segment = None

    def _forget(self, entry: _SegEntry) -> None:
        entry.graph = None
        entry.outputs = None
        self.shared.drop(getattr(entry, "slots", ()))

    def _placement(self) -> Optional[tuple]:
        out = []
        for g in self.graphs:
            p = g.weights.placement(self.shared.device_index)
            if p is None:
                return None
            out.append(p)
        return tuple(out)

    def call(self, i: int, args: tuple, kwargs: dict) -> Any:
        if self.poisoned or not self.enabled:
            return _NOT_HANDLED
        if i == 0:
            return self._start(args, kwargs)
        if self.pos != i:
            return _NOT_HANDLED
        return self._next(i, args, kwargs)

    def _start(self, args: tuple, kwargs: dict) -> Any:
        self._reset_forward()
        torch = _torch()
        if (
            torch.is_grad_enabled()
            or not _lazy_chain(self.shared.root)
            or _step_cache_live(self.shared.root)
        ):
            return _NOT_HANDLED
        live: list = []
        try:
            key = _walk((args, kwargs), live)
            extra = _root_key_extra(self.shared.root)
        except Exception:  # noqa: BLE001
            return _NOT_HANDLED
        if _refusal(key, kwargs) or any(t.device.type != "cuda" for t in live):
            return _NOT_HANDLED
        if torch.cuda.current_device() != self.shared.device_index or any(
            t.device.index != self.shared.device_index for t in live
        ):
            return _NOT_HANDLED
        for g in self.graphs:
            if g.protect is None:
                g.protect = g._protect_controller()
            if g.poisoned or g.churned or g.bypassed or g.protect:
                return _NOT_HANDLED
        placement = self._placement()
        if placement is None:
            return _NOT_HANDLED
        full = (key, placement) if extra is None else (key, placement, extra)
        if full in self.refused:
            return _NOT_HANDLED
        entry = self.cache.get(full)
        if entry is None and self.owner is not None and self.owner.scope_sealed():
            return _NOT_HANDLED  # its blocks run ungraphed too (BlockGraph.owner)
        if entry is None and full in self.seen:
            entry = self._capture(full, live)
            if entry is None:
                return _NOT_HANDLED
        if entry is not None:
            self.cache.move_to_end(full)
            for dst, src in zip(entry.static_in, live):
                if dst.data_ptr() != src.data_ptr():
                    dst.copy_(src)
            entry.graph.replay()
            entry.replays += 1
            self.stats["segment_replays"] += 1
            self.mode, self.cur, self.ext = "replay", entry, live
            return self._hand_out(0, entry)
        rec = _SegRecord()
        out = self.graphs[0].compute(*args, **kwargs)
        flat: list = []
        try:
            spec = _flatten(out, flat)
        except Exception:  # noqa: BLE001
            self.refused.add(full)
            self.stats["segment_refusals"] += 1
            return out
        rec.keys.append(key)
        rec.prov.append(tuple(("e", e) for e in range(len(live))))
        rec.out_specs.append(spec)
        self.mode, self.cur, self.ext = "record", (full, rec), live
        self.returned = [flat]
        self.versions = [[_version(t) for t in flat]]
        self.pos = 1
        self.graphs[0].stats["eager_calls"] += 1
        return out

    def _provenance(self, live: list) -> Optional[tuple]:
        out = []
        for t in live:
            found = None
            for j in range(len(self.returned) - 1, -1, -1):
                for m, r in enumerate(self.returned[j]):
                    if t is r:
                        if _version(t) != self.versions[j][m]:
                            return None  # rewritten in place since its block handed it on
                        found = ("o", j, m)
                        break
                if found is not None:
                    break
            if found is None:
                for e, x in enumerate(self.ext):
                    if t is x:
                        found = ("e", e)
                        break
            if found is None:
                return None
            out.append(found)
        return tuple(out)

    def _next(self, i: int, args: tuple, kwargs: dict) -> Any:
        live: list = []
        try:
            key = _walk((args, kwargs), live)
        except Exception:  # noqa: BLE001
            self._abandon()
            return _NOT_HANDLED
        prov = self._provenance(live)
        last = i == len(self.graphs) - 1
        if self.mode == "record":
            full, rec = self.cur
            if prov is None or _refusal(key, kwargs):
                self.refused.add(full)
                self.stats["segment_refusals"] += 1
                self._abandon()
                return _NOT_HANDLED
            out = self.graphs[i].compute(*args, **kwargs)
            flat: list = []
            try:
                spec = _flatten(out, flat)
            except Exception:  # noqa: BLE001
                self.refused.add(full)
                self.stats["segment_refusals"] += 1
                self._abandon()
                return out
            rec.keys.append(key)
            rec.prov.append(prov)
            rec.out_specs.append(spec)
            self.returned.append(flat)
            self.versions.append([_version(t) for t in flat])
            self.graphs[i].stats["eager_calls"] += 1
            if last:
                self.seen[full] = rec
                while len(self.seen) > 4:
                    self.seen.popitem(last = False)
                self.stats["segment_records"] += 1
                self._reset_forward()
            else:
                self.pos = i + 1
            return out
        entry = self.cur
        if prov is None or key != entry.record.keys[i] or prov != entry.record.prov[i]:
            self.stats["segment_breaks"] += 1
            self._abandon()
            return _NOT_HANDLED
        return self._hand_out(i, entry)

    def _hand_out(self, i: int, entry: _SegEntry) -> Any:
        tensors = entry.outputs[i]
        last = i == len(self.graphs) - 1
        if last:
            tensors = [t.clone() for t in tensors]
            self._reset_forward()
        else:
            if i == 0:
                self.returned = [tensors]
                self.versions = [[_version(t) for t in tensors]]
            else:
                self.returned.append(tensors)
                self.versions.append([_version(t) for t in tensors])
            self.pos = i + 1
        self.graphs[i].stats["segment_hits"] = self.graphs[i].stats.get("segment_hits", 0) + 1
        return _rebuild(entry.record.out_specs[i], tensors)

    def _abandon(self) -> None:
        self._reset_forward()

    def _chain(self, rec: _SegRecord, ext: list) -> list:
        results: list = []
        for i, g in enumerate(self.graphs):
            tensors = [ext[p[1]] if p[0] == "e" else results[p[1]][p[2]] for p in rec.prov[i]]
            call_args, call_kwargs = _unwalk(rec.keys[i], iter(tensors))
            out = g.compute(*call_args, **call_kwargs)
            flat: list = []
            if _flatten(out, flat) != rec.out_specs[i]:
                raise RuntimeError("segment block output layout changed between calls")
            results.append(flat)
        return results

    def _capture(self, full: tuple, live: list) -> Optional[_SegEntry]:
        from .diffusion_cuda_graph import (
            CaptureRefused,
            _capturing,
            capture_device,
            note_capture,
            refuse_if_exhausted,
            retire_failed_capture,
        )

        torch = _torch()
        rec = self.seen.pop(full)
        entry = _SegEntry()
        slot = ("segment", id(self), full[0])
        try:
            entry.static_in = self.shared.statics_for(slot, live)
            for dst, src in zip(entry.static_in, live):
                dst.copy_(src)
            # once, on the compute stream: a guard the static buffers trip recompiles here, never inside the capture
            self._chain(rec, entry.static_in)
            self.stats["segment_warmups"] += 1
            current = torch.cuda.current_stream()
            stream = self.shared.capture_stream()
            stream.wait_stream(current)
            refuse_if_exhausted(self.shared.logger)
            graph = torch.cuda.CUDAGraph()
            pool = self.shared.graph_pool()
            with torch.cuda.stream(stream):
                with _capturing():
                    # read after the entry, which first hands back any retired graph pool
                    before = torch.cuda.memory_reserved()
                    try:
                        graph.capture_begin(pool = pool, capture_error_mode = "thread_local")
                        try:
                            outputs = self._chain(rec, entry.static_in)
                        finally:
                            graph.capture_end()
                    except BaseException as exc:
                        retire_failed_capture(graph, pool, exc)
                        if self.shared.pool == pool:
                            self.shared.pool = None
                        raise
            current.wait_stream(stream)
            self.shared.pool_bytes += max(0, torch.cuda.memory_reserved() - before)
        except CaptureRefused:
            self.seen[full] = rec
            self.stats["segment_refusals"] = self.stats.get("segment_refusals", 0) + 1
            return None
        except Exception as exc:  # noqa: BLE001 - this layout stays per block for the load's life
            self.capture_error = {"type": type(exc).__name__, "msg": str(exc)[:4000]}
            self.refused.add(full)
            self.stats["segment_fallbacks"] += 1
            if self.shared.logger is not None:
                self.shared.logger.warning(
                    "diffusion.block_graph: recording a %d-block segment failed (%s: %s); its blocks record one by one",
                    len(self.graphs),
                    type(exc).__name__,
                    exc,
                )
            return None
        entry.graph = graph
        entry.pool = pool
        entry.outputs = outputs
        entry.record = rec
        entry.slots = (slot,)
        entry.replays = 0
        self.shared.hold(entry.slots)
        self.cache[full] = entry
        self.stats["segment_captures"] += 1
        if self.owner is not None:
            self.owner.note_capture()
        self._unsloth_capture_device = capture_device()
        note_capture(self._unsloth_capture_device)
        while len(self.cache) > self.max_graphs:
            _drop_entries(self, [self.cache.popitem(last = False)[1]], evicting = True)
        return entry


def _segments(transformer: Any, graphs: list) -> list:
    """Runs of two or more armed blocks on one card, in forward order, as (graphs, shared) pairs."""
    try:
        from .diffusion_layer_split import find_blocks
    except Exception:  # noqa: BLE001
        return []
    by_block = {id(g.block): g for g in graphs}
    runs: list = []
    current: list = []
    current_key = None
    for _qual, group, block in find_blocks(transformer):
        g = by_block.get(id(block))
        key = (group, g.shared.device_index, id(g.shared)) if g is not None else None
        if g is not None and key == current_key:
            current.append(g)
            continue
        if len(current) > 1:
            runs.append(current)
        current = [g] if g is not None else []
        current_key = key
    if len(current) > 1:
        runs.append(current)
    return [(run, run[0].shared) for run in runs]


def _group_offload_hook(module: Any) -> Any:
    try:
        from diffusers.hooks import group_offloading as go

        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            return None
        return registry.get_hook(getattr(go, "_GROUP_OFFLOADING", "group_offloading"))
    except Exception:  # noqa: BLE001
        return None


def _is_original_forward(fn: Any, module: Any) -> bool:
    return getattr(fn, "__self__", None) is module and getattr(fn, "__func__", None) is getattr(
        type(module), "forward", None
    )


def _own_forward(fn: Any, module: Any) -> Any:
    """Live class forward for a hook that captured the original before a class-level patch, else None."""
    if getattr(fn, "__self__", None) is not module:
        return None
    func = getattr(fn, "__func__", None)
    cls = type(module)
    live = getattr(cls, "forward", None)
    if func is None or live is None:
        return None
    if func is live:
        return fn
    qualname = getattr(func, "__qualname__", None)
    owner = getattr(func, "__module__", None)
    for base in cls.__mro__:
        if qualname == f"{base.__qualname__}.forward" and owner == base.__module__:
            try:
                return live.__get__(module, cls)
            except Exception:  # noqa: BLE001 - not a descriptor: leave the hook's forward alone
                return None
    return None


def _inner_hook_reason(block: Any) -> Optional[str]:
    """Why a block's own subtree moves weights mid-forward (per-layer offload), which no recording can hold."""
    for name, sub in block.named_modules():
        if sub is block:
            continue
        if getattr(sub, "_hf_hook", None) is not None:
            return "per-layer offload hooks inside the block"
        if _group_offload_hook(sub) is not None:
            return "per-layer group offload inside the block"
    hook = getattr(block, "_hf_hook", None)
    if hook is not None:
        return "an accelerate hook on the block"
    return None


def repeated_blocks(transformer: Any) -> list:
    names = set(getattr(transformer, "_repeated_blocks", None) or ())
    if not names:
        return []
    return [m for m in transformer.modules() if type(m).__name__ in names and m is not transformer]


class BlockGraphSet:
    """Every ``BlockGraph`` of one denoiser, behind the handle interface of ``diffusion_cuda_graph``."""

    def __init__(
        self,
        transformer: Any,
        shared: _Shared,
        logger: Any = None,
    ) -> None:
        self.module = transformer
        self.shared = shared
        self.card_shared: dict = {}
        self.segments: list = []
        self.logger = logger
        self.graphs: list = []
        self.restores: list = []
        self.enabled = True
        self.bypassed = False
        self.slots_mib = 0
        self._prefetcher = None
        self.mode = "blocks"
        self.max_graphs = MAX_GRAPHS_PER_BLOCK
        self.length_varies = False
        self.scope: Any = None
        self._sealed_scopes: set = set()
        self._scope_captures: dict = {}
        self._render_scopes: set = set()

    def scope_sealed(self) -> bool:
        return self.length_varies and self.scope in self._sealed_scopes

    def note_capture(self) -> None:
        if self.length_varies:
            self._scope_captures[self.scope] = self._scope_captures.get(self.scope, 0) + 1

    def track_scopes(self) -> None:
        """Read each denoiser call's image scope (resolution, batch, references) before its blocks run."""
        from .diffusion_cuda_graph import _image_scope

        def _pre(module: Any, args: Any, kwargs: Any) -> None:
            try:
                self.scope = _image_scope(args, kwargs)
            except Exception:  # noqa: BLE001 - an unkeyable call: no scope, nothing seals
                self.scope = None
            self._render_scopes.add(self.scope)

        self.length_varies = True
        handle = self.module.register_forward_pre_hook(_pre, with_kwargs = True)
        self.restores.append(handle.remove)

    def shared_for(self, index: int) -> _Shared:
        """The pool, stream and static buffers of card ``index``: the primary's own, else one per other card."""
        if self.shared.device_index is None:
            self.shared.device_index = index
        if self.shared.device_index == index:
            self.shared.strict_device = True
            return self.shared
        state = self.card_shared.get(index)
        if state is None:
            state = _Shared(index, self.logger)
            state.root = self.shared.root
            state.strict_device = True
            self.card_shared[index] = state
        return state

    def cards(self) -> set:
        return {s.device_index for s in self._all_shared()}

    def _all_shared(self) -> list:
        return [self.shared, *self.card_shared.values()]

    @property
    def cache(self) -> dict:
        out: dict = {}
        for i, g in enumerate(self.graphs):
            for k, v in g.cache.items():
                out[(i, k)] = v
        return out

    @property
    def stats(self) -> dict:
        total: dict = {}
        for g in self.graphs:
            for k, v in g.stats.items():
                total[k] = total.get(k, 0) + int(v)
        for seg in self.segments:
            for k, v in seg.stats.items():
                total[k] = total.get(k, 0) + int(v)
        total["captures"] = total.get("captures", 0) + total.get("segment_captures", 0)
        total["warmups"] = total.get("warmups", 0) + total.get("segment_warmups", 0)
        total["replays"] = total.get("replays", 0) + total.get("segment_replays", 0)
        # The whole-forward wrapper's names, so diffusion_cuda_graph.stats aggregates both kinds alike.
        total["refused_host_tensor"] = total.get("refused_host_weight", 0)
        total["cap_skips"] = total.get("evictions", 0)
        return total

    @property
    def poisoned(self) -> bool:
        return bool(self.graphs) and all(g.poisoned or g.churned for g in self.graphs)

    @property
    def capture_error(self) -> Optional[dict]:
        for g in self.graphs:
            if g.capture_error:
                return g.capture_error
        return None

    def set_bypass(self, on: bool) -> "BlockGraphSet":
        self.bypassed = bool(on)
        for g in self.graphs:
            g.set_bypass(on)
        return self

    def reset(self) -> "BlockGraphSet":
        self._sealed_scopes.clear()
        self._scope_captures.clear()
        self._render_scopes.clear()
        for seg in self.segments:
            seg.reset()
        for g in self.graphs:
            g.reset()
        for state in self._all_shared():
            state.release()
        _release_cached()
        return self

    def free(self) -> "BlockGraphSet":
        for seg in self.segments:
            seg.detach()
        self.segments.clear()
        for g in self.graphs:
            g.reset()
            g.enabled = False
        for restore in reversed(self.restores):
            try:
                restore()
            except Exception:  # noqa: BLE001
                pass
        self.restores.clear()
        self.graphs.clear()
        for state in self._all_shared():
            state.release()
        self.enabled = False
        try:
            from .diffusion_offload_prefetch import module_prefetcher
            pf = module_prefetcher(self.module)
            if pf is not None:
                pf.disable_slots()
        except Exception:  # noqa: BLE001
            pass
        _release_cached()
        return self

    def describe(self) -> dict:
        s = self.stats
        return {
            "module": type(self.module).__name__,
            "mode": self.mode,
            "blocks": len(self.graphs),
            "enabled": bool(self.enabled),
            "bypassed": bool(self.bypassed),
            "poisoned": bool(self.poisoned),
            "graphs": sum(len(g.cache) for g in self.graphs),
            "max_graphs": self.max_graphs,
            "stats": s,
            "slot_ring_mib": int(self.slots_mib),
            "slot_ring_allocated_mib": int(
                getattr(getattr(self, "_prefetcher", None), "slot_bytes", 0) or 0
            )
            >> 20,
            "pool_mib": int(sum(st.pool_bytes for st in self._all_shared()) >> 20),
            "static_mib": int(sum(st.static_bytes for st in self._all_shared()) >> 20),
            **({"cards": sorted(self.cards())} if self.card_shared else {}),
            **(
                {
                    "segments": len(self.segments),
                    "segment_blocks": sum(len(seg.graphs) for seg in self.segments),
                    "segment_graphs": sum(len(seg.cache) for seg in self.segments),
                }
                if self.segments
                else {}
            ),
            "capture_error": None
            if not self.capture_error
            else {
                "type": str(self.capture_error.get("type")),
                "msg": str(self.capture_error.get("msg")),
            },
        }

    def why_off(self) -> Optional[str]:
        """Why no block has replayed, once a render has run; None while any replays or before the first call."""
        s = self.stats
        if s.get("replays") or not (s.get("eager_calls") or s.get("captures")):
            return None
        churned = sum(1 for g in self.graphs if g.churned)
        poisoned = sum(1 for g in self.graphs if g.poisoned)
        parts = []
        if churned:
            parts.append(f"{churned} block(s) find their weights at a new address every call")
        if poisoned:
            err = self.capture_error or {}
            parts.append(f"{poisoned} block(s) failed to record ({err.get('type') or 'error'})")
        for field, label in (
            ("refused_host_weight", "weights not on the GPU"),
            ("refused_kv_write", "prefix K/V prefill steps"),
            ("refused_float", "a float argument"),
            ("refused_object", "a non-tensor argument"),
            ("refused_output", "an output the graph cannot hand back"),
        ):
            if s.get(field):
                parts.append(f"{s[field]} call(s) with {label}")
        detail = "; ".join(parts) or "every call so far was a first sighting"
        return f"armed per block, but all {s.get('eager_calls', 0)} block call(s) ran their compute ({detail})"

    def held_bytes(self) -> int:
        """Device memory this layer holds that the caching allocator cannot hand to anything else."""
        pf = getattr(self, "_prefetcher", None)
        slots = int(getattr(pf, "slot_bytes", 0) or 0) if pf is not None else 0
        return int(sum(st.pool_bytes + st.static_bytes for st in self._all_shared()) + slots)


def _drop_entries(
    owner: Any,
    entries: list,
    evicting: bool = False,
) -> None:
    """The one way a block or segment graph drops recordings (eviction, poison, reset): forget them, then count their
    pool as retired, so no capture records beside its cached segments before a flush. An eviction from the current
    pool is not counted (the entry just recorded into it keeps it alive); any other drop is."""
    if not entries:
        return
    for entry in entries:
        owner._forget(entry)
    current = getattr(getattr(owner, "shared", None), "pool", None)
    if (
        evicting
        and current is not None
        and all(getattr(e, "pool", None) == current for e in entries)
    ):
        return
    try:
        from .diffusion_cuda_graph import note_pool_retired
        note_pool_retired()
    except Exception:  # noqa: BLE001
        pass


def _release_cached() -> None:
    try:
        from .diffusion_cuda_graph import release_cached_outside_capture
        if _torch().cuda.is_available():
            release_cached_outside_capture()
    except Exception:  # noqa: BLE001
        pass


def _device_index(transformer: Any, fallback: Any = None) -> Optional[int]:
    torch = _torch()
    for cand in (getattr(transformer, "_unsloth_resident_device", None), fallback):
        if cand is None:
            continue
        try:
            dev = torch.device(cand)
        except Exception:  # noqa: BLE001
            continue
        if dev.type == "cuda":
            return dev.index  # None: whichever CUDA device is current when the block runs
    return None


def _block_card(block: Any) -> Optional[int]:
    """The CUDA ordinal every weight of ``block`` sits on (a layer split places each block whole), else None."""
    index: Any = _UNSEEN
    for t in _WeightView(block).tensors:
        for leaf in _leaves(t):
            dev = leaf.device
            if dev.type != "cuda" or dev.index is None:
                return None
            if index is _UNSEEN:
                index = dev.index
            elif dev.index != index:
                return None
    return None if index is _UNSEEN else int(index)


def install_block_graphs(
    transformer: Any,
    *,
    device: Any = None,
    logger: Any = None,
    slots: bool = True,
) -> tuple[Optional[BlockGraphSet], str]:
    """Arm one ``BlockGraph`` per repeated block of ``transformer``. Returns (handle or None, reason)."""
    if block_graphs_disabled():
        return None, f"block graphs disabled by {BLOCK_GRAPHS_ENV}"
    blocks = repeated_blocks(transformer)
    if not blocks:
        return None, "no repeated blocks to record"
    from .diffusion_layer_split import is_layer_split

    torch = _torch()
    split = is_layer_split(transformer)
    shared = _Shared(_device_index(transformer, device), logger)
    shared.root = transformer
    handle = BlockGraphSet(transformer, shared, logger)
    compile_below_offload_hooks(transformer, logger)
    refused: dict = {}
    for block in blocks:
        why = _inner_hook_reason(block)
        if why:
            refused[why] = refused.get(why, 0) + 1
            continue
        compiled = getattr(block, "_compiled_call_impl", None)
        hook = _group_offload_hook(block)
        block_shared = shared
        if split:
            index = _block_card(block)
            if index is None:
                refused["weights off the GPU"] = refused.get("weights off the GPU", 0) + 1
                continue
            block_shared = handle.shared_for(index)
        if hook is not None and compiled is not None:
            # compiled through its hooks (UNSLOTH_DIFFUSION_COMPILE_BELOW_HOOKS=0): the compute is inside a dynamo frame
            why = "compiled through its offload hooks"
            refused[why] = refused.get(why, 0) + 1
            continue
        try:
            if hook is not None:
                # Below the hook: the onload / offload stay eager Python and the recording holds only the compute.
                registry = getattr(block, "_diffusers_hook", None)
                refs = list(getattr(registry, "_fn_refs", None) or ())
                target = getattr(block, "_unsloth_below_hook_ref", None)
                compute = None
                if target in refs:
                    compute = target.forward
                else:
                    target = None
                    for r in refs:
                        compute = _own_forward(getattr(r, "forward", None), block)
                        if compute is not None:
                            target = r
                            break
                if target is None:
                    refused["a stacked hook chain"] = refused.get("a stacked hook chain", 0) + 1
                    continue
                original_fwd = target.forward
                graph = BlockGraph(block, compute, block_shared)
                target.forward = graph

                def restore(ref: Any = target, fn: Any = original_fwd) -> None:
                    if ref.forward is not fn:
                        ref.forward = fn
            elif compiled is not None:
                graph = BlockGraph(block, compiled, block_shared)
                block._compiled_call_impl = graph

                def restore(
                    m: Any = block,
                    c: Any = compiled,
                    g: Any = graph,
                ) -> None:
                    if getattr(m, "_compiled_call_impl", None) is g:
                        m._compiled_call_impl = c
            else:
                slot = block.__dict__.get("forward")
                fwd = slot if slot is not None else type(block).forward.__get__(block)
                graph = BlockGraph(block, fwd, block_shared)
                block.__dict__["forward"] = graph

                def restore(
                    m: Any = block,
                    s: Any = slot,
                    g: Any = graph,
                ) -> None:
                    if m.__dict__.get("forward") is g:
                        if s is None:
                            m.__dict__.pop("forward", None)
                        else:
                            m.__dict__["forward"] = s
        except Exception as exc:  # noqa: BLE001
            refused[f"install failed ({type(exc).__name__})"] = (
                refused.get(f"install failed ({type(exc).__name__})", 0) + 1
            )
            continue
        handle.graphs.append(graph)
        handle.restores.append(restore)
    if not handle.graphs:
        reason = (
            ", ".join(f"{n} block(s): {w}" for w, n in refused.items()) or "no block could be armed"
        )
        return None, reason
    if split and not segment_graphs_disabled():
        handle.segments = [
            SegmentGraph(run, run_shared)
            for run, run_shared in _segments(transformer, handle.graphs)
        ]
        handle.mode = "segments" if handle.segments else handle.mode
    if split and _prompt_length_varies(transformer):
        handle.track_scopes()
        for unit in (*handle.graphs, *handle.segments):
            unit.owner = handle
    if slots and torch.cuda.is_available():
        try:
            from .diffusion_offload_prefetch import module_prefetcher
            pf = module_prefetcher(transformer)
            if pf is not None:
                planned = pf.enable_slots(logger = logger)
                handle._prefetcher = pf
                handle.slots_mib = planned if getattr(pf, "slot_streamed", 1) else 0
        except Exception as exc:  # noqa: BLE001 - streamed blocks then simply churn and run their compute
            if logger is not None:
                logger.warning("diffusion.block_graph: slot ring unavailable (%s)", exc)
    if logger is not None:
        logger.info(
            "diffusion.block_graph: armed %d of %d %s block(s)%s%s%s",
            len(handle.graphs),
            len(blocks),
            type(transformer).__name__,
            (
                f" over cards {sorted(handle.cards())}"
                + (
                    f" as {len(handle.segments)} per-card segment graph(s)"
                    if handle.segments
                    else ""
                )
            )
            if split
            else "",
            f"; streamed blocks read a {handle.slots_mib} MiB slot ring"
            if handle.slots_mib
            else "",
            ("; refused " + ", ".join(f"{n}: {w}" for w, n in refused.items())) if refused else "",
        )
    return handle, "armed"


def _prompt_length_varies(transformer: Any) -> bool:
    try:
        from .diffusion_cuda_graph import _length_varies
        return _length_varies(transformer)
    except Exception:  # noqa: BLE001
        return False


COMPILE_BELOW_HOOKS_ENV = "UNSLOTH_DIFFUSION_COMPILE_BELOW_HOOKS"


def compile_below_hooks_enabled() -> bool:
    return (os.environ.get(COMPILE_BELOW_HOOKS_ENV) or "").strip().lower() not in _OFF


def compile_below_offload_hooks(transformer: Any, logger: Any = None) -> int:
    """Compiles each block's own forward, keeping hooks eager; traced hooks recompile on every release."""
    if not compile_below_hooks_enabled():
        return 0
    kwargs = getattr(transformer, "_unsloth_regional_compile_kwargs", None)
    if not isinstance(kwargs, dict):
        return 0
    torch = _torch()
    guard = getattr(transformer, "_unsloth_compile_guard", None)
    moved = 0
    for block in repeated_blocks(transformer):
        compiled = getattr(block, "_compiled_call_impl", None)
        if compiled is None or _group_offload_hook(block) is None:
            continue
        registry = getattr(block, "_diffusers_hook", None)
        target = live = None
        for r in list(getattr(registry, "_fn_refs", None) or ()):
            live = _own_forward(getattr(r, "forward", None), block)
            if live is not None:
                target = r
                break
        if target is None:
            continue
        # compiles the live class forward: a per-arch patch installed after the hook captured the original runs too
        fn = torch.compile(live, **dict(kwargs))
        if guard is not None and callable(getattr(guard, "wrap", None)):
            fn = guard.wrap(fn, live, transformer)
            # A failed compile falls back to the live forward, so a class patch keeps running eager below the hook.
            guard.restores.append(lambda ref = target, f = live: setattr(ref, "forward", f))
        from . import diffusion_block_restride

        if diffusion_block_restride.is_wrapped(compiled):
            # FLUX.1's block 0 keeps the slice layout of the others, so all of them share one compiled graph.
            fn = diffusion_block_restride.wrap(fn)
        rewrap = getattr(compiled, "_unsloth_rewrap", None)
        if callable(rewrap):
            # A family adapter around the compiled call (LTX-2's STG flag) must wrap the moved compile too.
            fn = rewrap(fn)
        target.forward = fn
        block._compiled_call_impl = None
        block._unsloth_below_hook_ref = target
        moved += 1
    if moved and logger is not None:
        logger.info(
            "diffusion.speed: %d offloaded %s blocks compile below their offload hooks",
            moved,
            type(transformer).__name__,
        )
    return moved


def compile_pipe_below_offload_hooks(pipe: Any, logger: Any = None) -> int:
    try:
        from .diffusion_speed import _denoiser_dits
    except Exception:  # noqa: BLE001
        return 0
    moved = 0
    for transformer in _denoiser_dits(pipe):
        try:
            moved += compile_below_offload_hooks(transformer, logger)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning("diffusion.speed: compile below offload hooks failed (%s)", exc)
    return moved

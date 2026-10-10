# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CUDA graphs per denoiser block, for denoisers whose weights an offload tier moves.

A whole-forward graph only holds while every weight stays at one address. Here each repeated block records on its
own, keyed by its inputs AND the device addresses of its weights: moved weights run eagerly once, then record again.
Streamed blocks get fixed addresses from the prefetcher's slot ring (``GroupPrefetcher.enable_slots``). The offload
hooks stay eager around the recorded compute. Static inputs are shared per (block class, layout) since replays never
overlap, and all recordings share one pool per denoiser.
"""

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

MAX_GRAPHS_PER_BLOCK = 6

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
    __slots__ = ("graph", "static_in", "static_out", "out_spec", "replays", "slots")


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
        self.root: Any = None
        self.pool = None
        self.stream = None
        self.static_in: dict = {}
        self.static_out: dict = {}
        self.static_refs: dict = {}
        self.warmed: set = set()
        self.pool_bytes = 0
        self.static_bytes = 0
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
    """A buffer the compiled block cannot tell from ``t``: same shape, stride, dtype, inference mode AND storage
    offset (a view such as Qwen-Image-2.1's ``full[:, prefix_len:]`` is guarded on its offset; a fresh tensor at
    offset 0 recompiles the block)."""
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
        self.seen: "OrderedDict[tuple, tuple]" = OrderedDict()
        self.unreplayed_placements = 0
        self.churned = False
        self.protect: Any = None
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
            return self.compute(*args, **kwargs)
        if not self.enabled or self.bypassed or self.poisoned or self.churned or _bg_eager_forced():
            return self._eager(args, kwargs)
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
            # A capture never records a host read.
            self.stats["refused_host_input"] += 1
            return self._eager(args, kwargs)
        if self.protect is None:
            self.protect = self._protect_controller()
        if self.protect:
            # A W4A4 replay at a W4A16 step would skip the protection.
            from .diffusion_cuda_graph import protect_graph_key
            placement = (placement, protect_graph_key(self.protect))
        full = (key, placement)
        entry = self.cache.get(full)
        if entry is None:
            if full not in self.seen:
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
        from .diffusion_cuda_graph import _capturing, refuse_if_exhausted, retire_failed_capture

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
                before = torch.cuda.memory_reserved()
                with _capturing():
                    try:
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
                        retire_failed_capture(graph, pool, exc)
                        if self.shared.pool == pool:
                            self.shared.pool = None
                        raise
                del out, flat
            current.wait_stream(stream)
            self.shared.pool_bytes += max(0, torch.cuda.memory_reserved() - before)
        except Exception as exc:  # noqa: BLE001 - a block that cannot record runs its compute for the load's life
            self._poison(exc)
            return None
        entry.slots = (slot, out_slot)
        self.shared.hold(entry.slots)
        entry.graph = graph
        entry.out_spec = out_spec
        entry.replays = 0
        if any(k[0] == full[0] for k in self.cache):
            self.stats["recaptures"] += 1
        self.cache[full] = entry
        self.stats["captures"] += 1
        while len(self.cache) > self.max_graphs:
            _, evicted = self.cache.popitem(last = False)
            self._forget(evicted)
            self.stats["evictions"] += 1
        return entry

    def _forget(self, entry: _Entry) -> None:
        entry.graph = None
        self.shared.drop(getattr(entry, "slots", ()))

    def _forget_all(self) -> None:
        for entry in self.cache.values():
            self._forget(entry)
        self.cache.clear()

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
        self.logger = logger
        self.graphs: list = []
        self.restores: list = []
        self.enabled = True
        self.bypassed = False
        self.slots_mib = 0
        self._prefetcher = None
        self.mode = "blocks"
        self.max_graphs = MAX_GRAPHS_PER_BLOCK

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
        for g in self.graphs:
            g.reset()
        self.shared.release()
        _release_cached()
        return self

    def free(self) -> "BlockGraphSet":
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
        self.shared.release()
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
            "pool_mib": int(self.shared.pool_bytes >> 20),
            "static_mib": int(self.shared.static_bytes >> 20),
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
        return int(self.shared.pool_bytes + self.shared.static_bytes + slots)


def _release_cached() -> None:
    try:
        import gc

        gc.collect()
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
            return dev.index
    return None


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
    torch = _torch()
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
        if hook is not None and compiled is not None:
            why = "compiled through its offload hooks"
            refused[why] = refused.get(why, 0) + 1
            continue
        try:
            if hook is not None:
                registry = getattr(block, "_diffusers_hook", None)
                refs = list(getattr(registry, "_fn_refs", None) or ())
                target = getattr(block, "_unsloth_below_hook_ref", None)
                if target not in refs:
                    target = next(
                        (
                            r
                            for r in refs
                            if _is_original_forward(getattr(r, "forward", None), block)
                        ),
                        None,
                    )
                if target is None:
                    refused["a stacked hook chain"] = refused.get("a stacked hook chain", 0) + 1
                    continue
                compute = target.forward
                graph = BlockGraph(block, compute, shared)
                target.forward = graph

                def restore(ref: Any = target, fn: Any = compute) -> None:
                    if ref.forward is not fn:
                        ref.forward = fn
            elif compiled is not None:
                graph = BlockGraph(block, compiled, shared)
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
                graph = BlockGraph(block, fwd, shared)
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
            "diffusion.block_graph: armed %d of %d %s block(s)%s%s",
            len(handle.graphs),
            len(blocks),
            type(transformer).__name__,
            f"; streamed blocks read a {handle.slots_mib} MiB slot ring"
            if handle.slots_mib
            else "",
            ("; refused " + ", ".join(f"{n}: {w}" for w, n in refused.items())) if refused else "",
        )
    return handle, "armed"


COMPILE_BELOW_HOOKS_ENV = "UNSLOTH_DIFFUSION_COMPILE_BELOW_HOOKS"


def compile_below_hooks_enabled() -> bool:
    return (os.environ.get(COMPILE_BELOW_HOOKS_ENV) or "").strip().lower() not in _OFF


def compile_below_offload_hooks(transformer: Any, logger: Any = None) -> int:
    """Compile each offload-hooked block's own ``forward`` so the hooks stay eager outside the compiled region.

    Traced through, the hooks graph-break the block and put residency state into the guards, so every release /
    re-pin recompiles. Idempotent; returns the blocks moved."""
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
        target = next(
            (
                r
                for r in list(getattr(registry, "_fn_refs", None) or ())
                if _is_original_forward(getattr(r, "forward", None), block)
            ),
            None,
        )
        if target is None:
            continue
        original = target.forward
        fn = torch.compile(original, **dict(kwargs))
        if guard is not None and callable(getattr(guard, "wrap", None)):
            fn = guard.wrap(fn, original, transformer)
            guard.restores.append(lambda ref = target, f = original: setattr(ref, "forward", f))
        from . import diffusion_block_restride

        if diffusion_block_restride.is_wrapped(compiled):
            fn = diffusion_block_restride.wrap(fn)
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

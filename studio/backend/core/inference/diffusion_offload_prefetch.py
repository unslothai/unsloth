# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Event-fenced prefetch for a block-streamed denoiser's diffusers offload groups.

diffusers fences its stream prefetch on the host (``stream.synchronize()`` per onload) and copies one group ahead.
Here the compute stream waits on a per-group CUDA event instead, and up to ``depth`` streamed groups are copied ahead
in the order the first forward recorded.

A forward's first copies are queued at the first block's onload (behind a compute-stream event when the top-level
group uploads on the compute stream): queued earlier they hold that upload back on the copy engine.

Memory stays bounded without a host wait: each offload records a compute-stream event the copy stream waits on, so a
freed block goes to the next prefetch only after the compute that read it. Bytes in flight never pass ``window``.
Hooks are per-group / per-module instance attributes; diffusers' classes are untouched.
"""

from __future__ import annotations

import os
from typing import Any, Optional

ASYNC_PREFETCH_ENV = "UNSLOTH_DIFFUSION_ASYNC_PREFETCH"
PREFETCH_DEPTH_ENV = "UNSLOTH_DIFFUSION_PREFETCH_DEPTH"
DEFAULT_PREFETCH_DEPTH = 2
MAX_PREFETCH_DEPTH = 8
PREFETCHER_ATTR = "_unsloth_group_prefetcher"
_BG_PIN_ATTR = "_unsloth_background_pin"


# Bumped on any denoiser placement change; CUDA graphs recorded under an older epoch are dropped.
_PLACEMENT_EPOCH = [0]


def bump_placement_epoch() -> None:
    _PLACEMENT_EPOCH[0] += 1


def placement_epoch() -> int:
    return _PLACEMENT_EPOCH[0]


def async_prefetch_enabled() -> bool:
    return (os.environ.get(ASYNC_PREFETCH_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def prefetch_depth() -> int:
    raw = (os.environ.get(PREFETCH_DEPTH_ENV) or "").strip()
    try:
        depth = int(raw) if raw else DEFAULT_PREFETCH_DEPTH
    except ValueError:
        depth = DEFAULT_PREFETCH_DEPTH
    return max(1, min(MAX_PREFETCH_DEPTH, depth))


def _go() -> Any:
    from diffusers.hooks import group_offloading as go
    return go


def _group_tensors(group: Any) -> list:
    """The tensors diffusers' stream onload moves, in its order (module params, module buffers, then the group's)."""
    out: list = []
    seen: set = set()
    for module in getattr(group, "modules", None) or ():
        for t in module.parameters():
            if id(t) not in seen:
                seen.add(id(t))
                out.append(t)
        for t in module.buffers():
            if id(t) not in seen:
                seen.add(id(t))
                out.append(t)
    for t in list(getattr(group, "parameters", None) or ()) + list(
        getattr(group, "buffers", None) or ()
    ):
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


_GENERIC_TO_COPY_DROPS_NON_BLOCKING: dict = {}


def _generic_to_copy_drops_non_blocking(cls: Any) -> bool:
    """torchao < 0.18's common ``aten._to_copy`` (every tensor class declaring ``tensor_data_names``, e.g. ``Int8Tensor``)
    moves the inner tensors with ``.to(device)``, dropping ``non_blocking``: one host wait per inner tensor. 0.18 passes
    it through. True only for that generic handler on such a torchao, never for a class with its own ``_to_copy``."""
    hit = _GENERIC_TO_COPY_DROPS_NON_BLOCKING.get(cls)
    if hit is not None:
        return hit
    hit = False
    try:
        import torch
        import torchao

        major, minor = (int(x) for x in str(torchao.__version__).split("+")[0].split(".")[:2])
        if (
            (major, minor) < (0, 18)
            and hasattr(cls, "tensor_data_names")
            and hasattr(cls, "tensor_attribute_names")
        ):
            fn = (
                (getattr(cls, "_ATEN_OP_TABLE", {}) or {})
                .get(cls, {})
                .get(torch.ops.aten._to_copy.default)
            )
            inner = getattr(fn, "__wrapped__", None)
            hit = bool(
                inner is not None
                and getattr(inner, "__module__", "") == "torchao.utils"
                and "_implements_common_tensor_ops" in getattr(inner, "__qualname__", "")
            )
    except Exception:  # noqa: BLE001 - unknown layout: keep the stock to()
        hit = False
    _GENERIC_TO_COPY_DROPS_NON_BLOCKING[cls] = hit
    return hit


def _generic_to_device(src: Any, device: Any) -> Any:
    """torchao 0.18's common ``_to_copy`` (same constructor call, inner tensors moved with ``non_blocking=True``)."""

    def move(t: Any) -> Any:
        return None if t is None else t.to(device, non_blocking = True)

    tensors = [move(getattr(src, n)) for n in src.tensor_data_names]
    optional = [move(getattr(src, n)) for n in getattr(src, "optional_tensor_data_names", ())]
    attrs = [getattr(src, a) if a != "device" else device for a in src.tensor_attribute_names]
    optional_attrs = [
        getattr(src, a) if a != "device" else device
        for a in getattr(src, "optional_tensor_attribute_names", ())
    ]
    return type(src)(*tensors, *attrs, *optional, *optional_attrs)


def _to_device(src: Any, device: Any) -> Any:
    """``src.to(device, non_blocking=True)``; torchao <= 0.17 int8 payloads (whose ``to`` drops it) are moved directly."""
    if _generic_to_copy_drops_non_blocking(type(src)):
        return _generic_to_device(src, device)
    name = type(src).__name__
    if name == "LinearActivationQuantizedTensor":
        inner = getattr(src, "original_weight_tensor", None)
        if type(inner).__name__ == "AffineQuantizedTensor" and type(
            getattr(inner, "tensor_impl", None)
        ).__name__ == ("PlainAQTTensorImpl"):
            return src._apply_fn_to_data(lambda w: _to_device(w, device))
    elif name == "AffineQuantizedTensor":
        if type(getattr(src, "tensor_impl", None)).__name__ == "PlainAQTTensorImpl":
            return src._apply_fn_to_data(
                lambda impl: impl._apply_fn_to_data(lambda x: x.to(device, non_blocking = True))
            )
    return src.to(device, non_blocking = True)


def _inner(t: Any) -> list:
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass
        if is_traceable_wrapper_subclass(t):
            names, _ = t.__tensor_flatten__()
            return [(n, getattr(t, n)) for n in names]
    except Exception:  # noqa: BLE001
        pass
    return []


_SLOT_ALIGN = 512


def _plain_leaves(t: Any) -> list:
    """The plain tensors holding ``t``'s data, depth first (a wrapper subclass's inner tensors)."""
    inner = _inner(t)
    if not inner:
        return [t]
    out: list = []
    for _, x in inner:
        out.extend(_plain_leaves(x))
    return out


def _packed_bytes(sources: list) -> Optional[int]:
    """Bytes a group's tensors take packed into one slot (each leaf aligned); None if a leaf cannot be viewed in."""
    total = 0
    for src in sources:
        for leaf in _plain_leaves(src):
            if not leaf.is_contiguous():
                return None
            total = (total + _SLOT_ALIGN - 1) // _SLOT_ALIGN * _SLOT_ALIGN
            total += leaf.numel() * leaf.element_size()
    return total


def _carve(raw: Any, offset: int, like: Any) -> tuple:
    """A view of ``raw`` (uint8) shaped like ``like`` at the next aligned offset; returns (view, end offset)."""
    offset = (offset + _SLOT_ALIGN - 1) // _SLOT_ALIGN * _SLOT_ALIGN
    n = like.numel() * like.element_size()
    view = raw[offset : offset + n].view(like.dtype).view(like.shape)
    return view, offset + n


def _slot_views(raw: Any, sources: list, device: Any) -> list:
    """Device tensors laid out in ``raw`` mirroring ``sources`` (plain views, or wrapper subclasses whose inner
    tensors are views), so a fill is an in-place copy and every fill of this group lands at the same addresses."""
    offset = 0
    views: list = []

    def build(src: Any) -> Any:
        nonlocal offset
        inner = _inner(src)
        if not inner:
            view, offset = _carve(raw, offset, src)
            return view
        wrapper = _to_device(src, device)
        for name, x in inner:
            setattr(wrapper, name, build(x))
        return wrapper

    for src in sources:
        views.append(build(src))
    return views


def _copy_into(dst: Any, src: Any) -> None:
    inner = _inner(dst)
    if inner:
        for name, d in inner:
            _copy_into(d, getattr(src, name))
        return
    dst.copy_(src, non_blocking = True)


def _rewrap(buf: Any) -> Any:
    """A new wrapper around ``buf``'s inner tensors (no copy), for ``swap_tensors`` to consume."""
    names, ctx = buf.__tensor_flatten__()
    return type(buf).__tensor_unflatten__(
        {n: getattr(buf, n) for n in names}, ctx, buf.size(), buf.stride()
    )


def _point_at(t: Any, buf: Any, torchao: bool, swap: Any) -> None:
    """Make parameter / buffer ``t`` read ``buf``'s storage, leaving ``buf`` itself untouched."""
    if not torchao:
        t.data = buf
        return
    inner = _inner(t)
    if (
        inner
        and all(x.device.type == buf.device.type for _, x in _inner(buf))
        and t.device == buf.device
    ):
        for name, _ in _inner(buf):
            setattr(t, name, getattr(buf, name))
        return
    if swap is not None:
        swap(t, _rewrap(buf))
    else:
        for name, _ in _inner(buf):
            setattr(t, name, getattr(buf, name))


def _detach_from_slot(group: Any, buffers: list) -> None:
    ptrs = set()
    for b in buffers:
        for x in _plain_leaves(b):
            ptrs.add(x.data_ptr())

    def detach(obj: Any) -> None:
        for name, x in _inner(obj):
            if _inner(x):
                detach(x)
            elif x.data_ptr() in ptrs:
                setattr(obj, name, x.clone())

    for t in _group_tensors(group):
        if _inner(t):
            detach(t)
        elif t.data_ptr() in ptrs:
            t.data = t.data.clone()


def _group_nbytes(group: Any) -> int:
    from .diffusion_memory import _storage_nbytes

    cpu = getattr(group, "cpu_param_dict", None) or {}
    total = 0
    for t in _group_tensors(group):
        total += sum(_storage_nbytes(cpu.get(t, t)))
    return int(total)


class GroupPrefetcher:
    """``ready``: group id -> CUDA event of its queued copy (None once waited on); a group is on the device exactly
    while its id is in ``ready``."""

    def __init__(
        self,
        module: Any,
        groups: list,
        device: Any,
        depth: int,
        window_bytes: Optional[int] = None,
    ):
        import torch

        self.module = module
        self.device = torch.device(device)
        self.depth = int(depth)
        self.groups = list(groups)
        self.by_id = {id(g): g for g in self.groups}
        self.nbytes = {id(g): _group_nbytes(g) for g in self.groups}
        largest = max(self.nbytes.values()) if self.nbytes else 0
        self.window = int(window_bytes) if window_bytes else (self.depth + 1) * largest
        self.ready: dict = {}
        self.inflight_bytes = 0
        self.peak_inflight_bytes = 0
        self.order: list = []
        self.seen: list = []
        self.pos = 0
        self.active = False
        self.on_order = False
        self.pending = False
        streams = [g.stream for g in self.groups if getattr(g, "stream", None) is not None]
        self.stream = streams[0] if streams else None
        self.fence_first = False
        self.capturing = False
        self.stats = {"forwards": 0, "copies": 0, "prefetched": 0, "missed": 0, "dropped": 0}
        # Slot ring: a group's copy lands at fixed offsets so recorded CUDA graphs read stable addresses.
        self.slot_of: dict = {}
        self.slot_buffers: dict = {}
        self.slot_raw: dict = {}
        self.slot_owner: dict = {}
        self.slot_size = 0
        self.slot_bytes = 0
        self.slot_bytes_planned = 0
        self.slot_streamed = 0

    def owns(self, group: Any) -> bool:
        return getattr(group, "__dict__", {}).get("onload_") is getattr(
            group, "_unsloth_prefetch_onload", None
        )

    def _compute(self) -> Any:
        import torch
        return torch.cuda.current_stream(self.device)

    def _slot_free(self, group: Any, must: bool) -> Optional[bool]:
        """Whether ``group``'s slot can take its copy now: True, False (wait: the occupant is on the device), or None
        (no slot / a forced onload whose slot is busy: copy to fresh memory instead)."""
        where = self.slot_of.get(id(group))
        if where is None:
            return None
        owner = self.slot_owner.get(where)
        if owner is None or owner == id(group) or owner not in self.ready:
            return True
        if not must:
            return False
        occupant = self.by_id.get(owner)
        if occupant is not None and self.ready.get(owner) is not None:
            self._release(occupant, self.ready.pop(owner))
            return True
        self.stats["slot_fallbacks"] = self.stats.get("slot_fallbacks", 0) + 1
        return None

    def _issue(
        self,
        group: Any,
        must: bool = True,
    ) -> bool:
        """Queue ``group``'s host-to-device copy on its copy stream and point its tensors at the destinations.
        False (nothing queued) only when ``must`` is False and the group's slot is still occupied."""
        import torch

        slot = self._slot_free(group, must) if getattr(self, "slot_of", None) else None
        if slot is False:
            return False
        pinner = getattr(group, _BG_PIN_ATTR, None)
        if pinner is not None:
            if getattr(self, "capturing", False) and not _pinned_done(pinner, group):
                raise RuntimeError("a background pin is still swapping this group's host copies")
            pinner.wait(group)
        go = _go()
        is_torchao = getattr(go, "_is_torchao_tensor", lambda t: False)
        swap = getattr(go, "_swap_torchao_tensor", None)
        cpu = group.cpu_param_dict
        stream = group.stream
        where = self.slot_of.get(id(group)) if slot else None
        with torch.cuda.stream(stream):
            # Made on the copy stream: the allocator orders block reuse per stream.
            buffers = self._views_for(group, where) if where is not None else None
            for i, t in enumerate(_group_tensors(group)):
                src = cpu[t]
                if not src.is_pinned():
                    if getattr(self, "capturing", False):
                        # a graph would replay this copy from a temporary pinned buffer reused after the call
                        raise RuntimeError("a streamed group's host copy is not pinned")
                    src = src.pin_memory()
                if buffers is not None:
                    _copy_into(buffers[i], src)
                    _point_at(t, buffers[i], is_torchao(t), swap)
                    continue
                moved = _to_device(src, self.device)
                if is_torchao(t) and swap is not None:
                    swap(t, moved)
                else:
                    t.data = moved
        if buffers is not None:
            self.slot_owner[where] = id(group)
            self.stats["slot_fills"] = self.stats.get("slot_fills", 0) + 1
        ready = torch.cuda.Event()
        ready.record(stream)
        gid = id(group)
        self.ready[gid] = ready
        self.inflight_bytes += self.nbytes.get(gid, 0)
        self.peak_inflight_bytes = max(self.peak_inflight_bytes, self.inflight_bytes)
        self.stats["copies"] += 1
        return True

    def onload(self, group: Any) -> None:
        gid = id(group)
        self.kick()
        if self.active:
            self.seen.append(gid)
            if self.on_order and self.pos < len(self.order) and self.order[self.pos] == gid:
                self.pos += 1
            else:
                self.on_order = False
        if gid not in self.ready:
            self._issue(group)
            if self.active and self.order:
                self.stats["missed"] += 1
        elif self.ready[gid] is not None:
            self.stats["prefetched"] += 1
        event = self.ready[gid]
        if event is not None:
            self._compute().wait_event(event)
            self.ready[gid] = None
        self._fill()

    def offload(self, group: Any) -> None:
        gid = id(group)
        if gid in self.ready:
            self._release(group, self.ready.pop(gid))
        else:
            self._release(group, None, counted = False)
        self._fill()

    def _release(
        self,
        group: Any,
        event: Any,
        counted: bool = True,
    ) -> None:
        import torch

        compute = self._compute()
        if event is not None:
            compute.wait_event(event)
            self.stats["dropped"] += 1
        done = torch.cuda.Event()
        done.record(compute)
        group.stream.wait_event(done)
        if counted:
            self.inflight_bytes -= self.nbytes.get(id(group), 0)
        type(group).offload_(group)

    def _forget_disowned(self) -> None:
        """Groups made resident while on the device (their onload_ replaced) leave the window without a release."""
        import torch

        compute = None
        for gid in list(self.ready):
            group = self.by_id.get(gid)
            if group is not None and self.owns(group):
                continue
            event = self.ready.pop(gid)
            if event is not None:
                compute = compute or self._compute()
                compute.wait_event(event)
            self.inflight_bytes -= self.nbytes.get(gid, 0)
            if group is not None and getattr(self, "slot_of", None) and self._holds_slot(group):
                compute = compute or self._compute()
                with torch.cuda.stream(compute):
                    _detach_from_slot(group, self.slot_buffers[gid])
                done = torch.cuda.Event()
                done.record(compute)
                group.stream.wait_event(done)
                self.slot_owner.pop(self.slot_of[gid], None)

    def _fill(self) -> None:
        if not (self.active and self.on_order):
            return
        ahead = 0
        for gid in self.order[self.pos :]:
            if ahead >= self.depth:
                break
            group = self.by_id.get(gid)
            if group is None or not self.owns(group):
                continue
            if gid in self.ready:
                ahead += 1
                continue
            if self.inflight_bytes + self.nbytes.get(gid, 0) > self.window and self.ready:
                break
            if getattr(self, "slot_of", None):
                if not self._issue(group, must = False):
                    break
            else:
                self._issue(group)
            ahead += 1

    def _holds_slot(self, group: Any) -> bool:
        where = self.slot_of.get(id(group))
        return where is not None and self.slot_owner.get(where) == id(group)

    def enable_slots(self, logger: Any = None) -> int:
        """Give every streamed block group one of ``depth + 1`` byte slots, round-robin in block order (the prefetch
        never holds more groups than that), each slot as large as the largest packed group: the ring is the prefetch
        window. A group's copies always land at the same offsets of its slot. Returns the ring's MiB (allocated on
        first use). The top-level group keeps fresh copies: no graph reads it."""
        if self.slot_of:
            return self.slot_bytes_planned >> 20
        members: list = []
        size = 0
        for group in self.groups:
            if getattr(group, "offload_leader", None) is self.module:
                continue
            cpu = getattr(group, "cpu_param_dict", None) or {}
            try:
                need = _packed_bytes([cpu.get(t, t) for t in _group_tensors(group)])
            except Exception:  # noqa: BLE001
                need = None
            if need is None:
                continue
            members.append(group)
            size = max(size, need)
        if not members:
            return 0
        count = min(self.depth + 1, len(members))
        self.slot_streamed = sum(1 for g in members if not getattr(g, "_unsloth_resident", False))
        for i, group in enumerate(members):
            self.slot_of[id(group)] = i % count
        self.slot_size = size
        self.slot_bytes_planned = count * size
        if logger is not None:
            logger.info(
                "diffusion.memory: %s streams %d block groups through %d slots of %d MiB (the prefetch window)",
                type(self.module).__name__,
                len(members),
                count,
                size >> 20,
            )
        return self.slot_bytes_planned >> 20

    def _views_for(self, group: Any, where: int) -> list:
        gid = id(group)
        views = self.slot_buffers.get(gid)
        if views is not None:
            return views
        import torch

        raw = self.slot_raw.get(where)
        if raw is None:
            raw = self._alloc_slot(where)
        cpu = group.cpu_param_dict
        views = _slot_views(raw, [cpu[t] for t in _group_tensors(group)], self.device)
        self.slot_buffers[gid] = views
        return views

    def _alloc_slot(self, where: int) -> Any:
        import torch

        raw = torch.empty(self.slot_size, dtype = torch.uint8, device = self.device)
        self.slot_raw[where] = raw
        self.slot_bytes += self.slot_size
        bump_placement_epoch()
        return raw

    def materialize_slots(self) -> None:
        """Allocate every slot of the ring now (a whole-step capture must not carve one out of its graph pool)."""
        if not self.slot_of or not self.slot_size:
            return
        streamed = {
            self.slot_of[id(g)]
            for g in self.groups
            if id(g) in self.slot_of and not getattr(g, "_unsloth_resident", False)
        }
        for where in sorted(streamed):
            if where not in self.slot_raw:
                self._alloc_slot(where)

    def drop_slots_after_forward(self) -> None:
        """Drop the ring once the current forward ends (``end``), or now between forwards."""
        if self.active:
            self._drop_slots_at_end = True
            return
        self._drop_ring()

    def _drop_ring(self) -> None:
        import torch

        from .diffusion_cuda_graph import hold_off_capture
        with hold_off_capture() as safe:
            if not safe:
                # Another thread is recording: its capture prohibits these syncs and frees. Retry after a later forward.
                self._drop_slots_at_end = True
                return
            self._drop_slots_at_end = False
            self.disable_slots()
            try:
                import gc

                gc.collect()
                clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
                if callable(clear):
                    clear()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def disable_slots(self) -> None:
        """Drop the ring. Only between forwards: every streamed group is back on the host by then."""
        import torch

        if self.active:
            return
        if self.slot_raw:
            torch.cuda.synchronize(self.device)
            bump_placement_epoch()
        self.slot_of = {}
        self.slot_buffers = {}
        self.slot_raw = {}
        self.slot_owner = {}
        self.slot_size = 0
        self.slot_bytes = 0
        self.slot_bytes_planned = 0
        self.slot_streamed = 0

    def begin(self) -> None:
        self.stats["forwards"] += 1
        self._forget_disowned()
        self.seen = []
        self.pos = 0
        self.active = True
        self.on_order = bool(self.order)
        self.pending = True
        self.capturing = False
        if self.stream is not None:
            import torch
            if torch.cuda.is_current_stream_capturing():
                # fork the copy stream into the capture before anything is queued on it
                self.capturing = True
                self.stream.wait_stream(self._compute())

    def kick(self) -> None:
        """First block onload of a forward (streamed or resident): queue the first ``depth`` streamed groups."""
        if self.pending:
            self.pending = False
            if self.fence_first and self.stream is not None and self.on_order:
                import torch

                after_top = torch.cuda.Event()
                after_top.record(self._compute())
                self.stream.wait_event(after_top)
            self._fill()

    def end(self) -> None:
        if not self.active:
            return
        self.active = False
        self.on_order = False
        self.pending = False
        self._forget_disowned()
        for gid in list(self.ready):
            group = self.by_id.get(gid)
            if group is not None:
                self._release(group, self.ready.pop(gid))
        self.ready.clear()
        self.inflight_bytes = 0
        if getattr(self, "capturing", False):
            self.capturing = False
            self._compute().wait_stream(self.stream)
        if self.seen:
            self.order = list(self.seen)
        if getattr(self, "_drop_slots_at_end", False):
            self._drop_ring()

    def abandon(self) -> None:
        """A capture failed mid-forward: the events and device copies it queued died with it. Forget them (never wait
        on a captured event outside its graph) and put every streamed group back on its host copy. Outside capture."""
        import torch

        self.active = self.on_order = self.pending = self.capturing = False
        self.ready.clear()
        self.inflight_bytes = 0
        try:
            torch.cuda.synchronize(self.device)
        except Exception:  # noqa: BLE001 - the copies below are what matters
            pass
        for group in self.groups:
            if self.owns(group):
                try:
                    type(group).offload_(group)
                except Exception:  # noqa: BLE001 - a group the capture never reached is already on the host
                    pass


def _adopt_top_group(
    module: Any,
    stream: Any,
    logger: Any = None,
) -> Optional[Any]:
    """Give a torchao top-level group (no stream in diffusers: synchronous upload, copy-back every forward) pinned host
    copies and the module's copy stream, so the prefetcher drives it like a block. None if left as it was."""
    import torch

    from .diffusion_memory import (
        GROUP_OFFLOAD_PIN_ENV,
        PIN_TOP_GROUP_ENV,
        _module_host_mib,
        _pin_budget_mib,
        _pinned_memory_capped,
    )

    off = ("0", "off", "false", "no")
    if (os.environ.get(PIN_TOP_GROUP_ENV) or "").strip().lower() in off:
        return None
    if str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower() in off:
        return None
    go = _go()
    registry = getattr(module, "_diffusers_hook", None)
    get_hook = getattr(registry, "get_hook", None)
    hook = (
        get_hook(getattr(go, "_GROUP_OFFLOADING", "group_offloading"))
        if callable(get_hook)
        else None
    )
    group = getattr(hook, "group", None)
    if (
        group is None
        or getattr(group, "stream", None) is not None
        or getattr(group, "offload_to_disk_path", None)
        or getattr(getattr(group, "onload_device", None), "type", None) != "cuda"
        or "onload_" in getattr(group, "__dict__", {})
        or getattr(group, "_unsloth_resident", False)
    ):
        return None
    tensors = _group_tensors(group)
    is_torchao = getattr(go, "_is_torchao_tensor", lambda t: False)
    if not tensors or not any(is_torchao(t) for t in tensors):
        return None
    holder = _tensor_holder(tensors)
    need = _module_host_mib(holder)
    budget = None if _pinned_memory_capped() else _pin_budget_mib()
    if budget is None or need > budget:
        return None
    to_cpu = getattr(type(group), "_to_cpu", None)
    if to_cpu is None:
        return None
    host = {}
    for t in tensors:
        src = t if t.device.type == "cpu" else t.cpu()
        copy = to_cpu(src, False)
        if not copy.is_pinned():
            return None
        host[t] = copy
    group.cpu_param_dict = host
    group.stream = stream
    group.record_stream = True
    group.non_blocking = True
    group.low_cpu_mem_usage = False
    type(group).offload_(group)
    if logger is not None:
        logger.info(
            "diffusion.memory: %s torchao top-level weights (%d MiB) stream from a pinned copy on the copy stream",
            type(module).__name__,
            need,
        )
    return group


def _tensor_holder(tensors: list) -> Any:
    class _Holder:
        def parameters(self, recurse: bool = True):
            return iter(tensors)

        def buffers(self, recurse: bool = True):
            return iter(())

    return _Holder()


def install_group_prefetch(
    module: Any,
    device: Any,
    logger: Any = None,
    *,
    depth: Optional[int] = None,
) -> int:
    """Drive ``module``'s streamed diffusers offload groups with a ``GroupPrefetcher``. Returns the number of groups
    covered (0: unchanged, e.g. kill switch, no CUDA, disk offload, or an older diffusers without stream groups)."""
    if not async_prefetch_enabled():
        return 0
    top = None
    try:
        import torch

        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            return 0
        if getattr(module, PREFETCHER_ATTR, None) is not None:
            return 0
        from .diffusion_memory import _offload_groups

        go = _go()
        hook_name = getattr(go, "_GROUP_OFFLOADING", "group_offloading")
        hooks = []
        for sub in module.modules():
            registry = getattr(sub, "_diffusers_hook", None)
            get_hook = getattr(registry, "get_hook", None)
            hook = get_hook(hook_name) if callable(get_hook) else None
            if hook is not None and getattr(hook, "group", None) is not None:
                hooks.append(hook)
        groups = [g for g in _offload_groups(module) if getattr(g, "stream", None) is not None]

        def _refusal(g: Any) -> Optional[str]:
            if getattr(g, "offload_to_disk_path", None):
                return "disk offload"
            if getattr(g, "_unsloth_resident", False) or "onload_" in getattr(g, "__dict__", {}):
                return "onload already replaced"
            cpu = getattr(g, "cpu_param_dict", None)
            if not isinstance(cpu, dict) or any(t not in cpu for t in _group_tensors(g)):
                return "no host copies"
            if not getattr(g, "record_stream", False):
                return "record_stream off"
            return None

        # All or nothing: a stream group left to diffusers would onload itself with no fence once the chain is cut.
        refused = (
            next((r for r in map(_refusal, groups) if r), None) if groups else "no stream groups"
        )
        if refused:
            if logger is not None:
                logger.info(
                    "diffusion.memory: %s keeps diffusers' stream prefetch (%s)",
                    type(module).__name__,
                    refused,
                )
            return 0
        streams = {id(g.stream) for g in groups}
        if len(streams) != 1:
            return 0
        top = _adopt_top_group(module, groups[0].stream, logger)
        if top is not None:
            groups = [top] + groups
        pf = GroupPrefetcher(module, groups, device, prefetch_depth() if depth is None else depth)
        pf.fence_first = top is None
        disable = getattr(getattr(torch, "compiler", None), "disable", None)

        def _eager(fn: Any) -> Any:
            return disable(fn) if callable(disable) else fn

        owned = {id(g) for g in groups}
        registry = getattr(module, "_diffusers_hook", None)
        for name in ("_LAZY_PREFETCH_GROUP_OFFLOADING", "_LAYER_EXECUTION_TRACKER"):
            key = getattr(go, name, None)
            if isinstance(key, str) and registry is not None:
                try:
                    registry.remove_hook(key, recurse = True)
                except Exception:  # noqa: BLE001 - not registered on this module
                    pass
        for hook in hooks:
            group = hook.group
            nxt = getattr(hook, "next_group", None)
            if id(group) in owned or (nxt is not None and id(nxt) in owned):
                hook.next_group = None
            if id(group) in owned:
                group.onload_self = True
                group.non_blocking = True
        for group in groups:

            def onload_(
                *_a: Any,
                _g: Any = group,
                **_k: Any,
            ) -> None:
                pf.onload(_g)

            def offload_(
                *_a: Any,
                _g: Any = group,
                **_k: Any,
            ) -> None:
                pf.offload(_g)

            group.onload_ = _eager(onload_)
            group.offload_ = _eager(offload_)
            group._unsloth_prefetch_onload = group.onload_
            group._unsloth_prefetcher = pf
        module.register_forward_pre_hook(_eager(lambda *_a, **_k: pf.begin()))
        module.register_forward_hook(_eager(lambda *_a, **_k: pf.end()), always_call = True)
        setattr(module, PREFETCHER_ATTR, pf)
        bump_placement_epoch()
        state = getattr(module, "_unsloth_stream_state", None)
        if not isinstance(state, dict):
            state = {"streamed": 1}
            try:
                module._unsloth_stream_state = state
            except AttributeError:
                pass
        state["fenced"] = True
        state["kick"] = pf.kick
        if logger is not None:
            logger.info(
                "diffusion.memory: %s streams %d offload groups with event-fenced prefetch (%d ahead, %d MiB window)",
                type(module).__name__,
                len(groups),
                pf.depth,
                pf.window >> 20,
            )
        return len(groups)
    except Exception as exc:  # noqa: BLE001 - diffusers' own onload stays
        if top is not None and "onload_" not in getattr(top, "__dict__", {}):
            top.stream = None
            top.record_stream = False
        if logger is not None:
            logger.warning("diffusion.memory: event-fenced prefetch unavailable (%s)", exc)
        return 0


def module_prefetcher(module: Any) -> Optional[GroupPrefetcher]:
    return getattr(module, PREFETCHER_ATTR, None)


def _pinned_done(pinner: Any, group: Any) -> bool:
    event = getattr(pinner, "_done", {}).get(id(group))
    return event is None or event.is_set()


def capture_refusal(module: Any) -> Optional[str]:
    """Why a CUDA graph cannot record ``module``'s block-streamed forward with its copies inside, else None.

    A replay repeats every copy from the host address it recorded into the device buffer it recorded, so every
    streamed group must copy from a pinned host tensor no background pin is still replacing, and every group must be
    driven by the prefetcher, pinned resident, or the dense top-level group's pinned upload."""
    pf = module_prefetcher(module)
    if pf is None:
        return "block streaming without the event-fenced prefetch (its stream waits are host synchronizations)"
    try:
        from .diffusion_memory import _offload_groups
        for group in _offload_groups(module):
            if getattr(group, "offload_to_disk_path", None):
                return "a group streams from disk"
            if getattr(group, "_unsloth_resident", False) or getattr(
                group, "_unsloth_pinned_top", False
            ):
                continue
            if not pf.owns(group):
                return "a streamed group is not driven by the prefetcher"
            pinner = getattr(group, _BG_PIN_ATTR, None)
            if pinner is not None and not _pinned_done(pinner, group):
                return "host copies are still being pinned in the background"
            cpu = getattr(group, "cpu_param_dict", None) or {}
            for t in _group_tensors(group):
                src = cpu.get(t)
                if src is None or not src.is_pinned():
                    return "a streamed group's host copy is not pinned"
    except Exception as exc:  # noqa: BLE001 - unknown layout: never capture it
        return f"offload layout unreadable ({type(exc).__name__})"
    return None

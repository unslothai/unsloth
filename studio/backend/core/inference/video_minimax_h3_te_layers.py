# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-granular upload of MiniMax-H3's streamed conditioner.

Leaf-level group offload moves the 50 decoder layers as 550 groups. From pageable host memory each group makes a
temporary pinned copy of every tensor on one thread and the side stream is synchronised before the next group, so the
host copy and the transfer never overlap (L4: 24.4 GB in about 24 s). Here each layer is one transfer into one of two
device slots: pageable bytes go through a small reused pinned ring (filled on a few threads, each chunk sent as soon as
it is full), pinned bytes are sent directly, and the compute stream waits on an event instead of the host. The next
layer is sent while the current one computes. Weights, kernels and dtypes are unchanged, so the hidden states are the
same bits. The diffusers groups stay registered (their onload / offload are no-ops for these layers), so the resident
host share, its yield and its top-up keep working on ``cpu_param_dict``. ``UNSLOTH_H3_TE_LAYER_STREAM=0`` keeps the
leaf-level upload."""

from __future__ import annotations

import threading
from typing import Any, Optional

H3_TE_LAYER_STREAM_ENV = "UNSLOTH_H3_TE_LAYER_STREAM"
_ATTR = "_unsloth_h3_te_layer_stream"
_ALIGN = 512
# Pinned ring: 4 x 64 MiB. Small next to the reserve; large enough that each transfer runs at full PCIe rate.
_CHUNK_BYTES = 64 << 20
_CHUNKS = 4
_COPY_THREADS = 4


def h3_te_layer_stream_enabled() -> bool:
    import os
    return str(os.environ.get(H3_TE_LAYER_STREAM_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _align(value: int) -> int:
    return (value + _ALIGN - 1) // _ALIGN * _ALIGN


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _has_tensors(module: Any) -> bool:
    return (
        next(iter(module.parameters()), None) is not None
        or next(iter(module.buffers()), None) is not None
    )


def _group_tensors(group: Any) -> list:
    return (
        [t for m in getattr(group, "modules", None) or () for t in m.parameters()]
        + [t for m in getattr(group, "modules", None) or () for t in m.buffers()]
        + list(getattr(group, "parameters", None) or ())
        + list(getattr(group, "buffers", None) or ())
    )


def _layer_plans(layers: list, groups: list) -> Optional[list]:
    """Per layer, ``[(tensor, group, offset, nbytes)]`` packed at ``_ALIGN``, plus the layer's groups; None when a
    tensor under a layer has no stream group holding its host copy, or a group spans two layers."""
    import torch

    owner: dict = {}
    for group in groups:
        cpu = getattr(group, "cpu_param_dict", None)
        if not cpu:
            continue
        for tensor in _group_tensors(group):
            if tensor in cpu:
                if owner.setdefault(id(tensor), group) is not group:
                    return None  # one tensor, two host copies
    plans = []
    for layer in layers:
        entries = []
        layer_groups: dict = {}
        offset = 0
        seen: set = set()
        for tensor in list(layer.parameters()) + list(layer.buffers()):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            group = owner.get(id(tensor))
            if group is None or type(tensor.data) is not torch.Tensor or tensor.numel() == 0:
                return None
            nbytes = tensor.numel() * tensor.element_size()
            entries.append((tensor, group, offset, nbytes))
            layer_groups[id(group)] = group
            offset = _align(offset + nbytes)
        if not entries:
            return None
        plans.append((entries, list(layer_groups.values()), offset))
    # A group whose tensors live in two layers would be released by one layer while the other still uses it.
    seen_groups: set = set()
    for _entries, layer_groups, _size in plans:
        for group in layer_groups:
            if id(group) in seen_groups:
                return None
            seen_groups.add(id(group))
        for group in layer_groups:
            covered = {id(e[0]) for e in _entries}
            if any(id(t) not in covered for t in _group_tensors(group)):
                return None
    return plans


class H3TELayerStream:
    """Uploads decoder layer ``i`` into device slot ``i % 2`` while layer ``i - 1`` computes.

    ``device.type == "cpu"`` runs every copy synchronously with no streams (tests)."""

    def __init__(
        self,
        layers: list,
        plans: list,
        device: Any,
        *,
        staging: Optional[list] = None,
        threads: int = _COPY_THREADS,
    ) -> None:
        import torch

        self.layers = layers
        self.plans = plans
        self.device = torch.device(device)
        self.slot_bytes = max(size for _e, _g, size in plans)
        self.staging = staging or []
        self.threads = max(1, int(threads))
        self._cuda = self.device.type == "cuda"
        self.copy_stream = torch.cuda.Stream(self.device) if self._cuda else None
        self._chunk_events: list = [None] * len(self.staging)
        self._next_chunk = 0
        self._pool = None
        self._lock = threading.Lock()
        self._reset()
        # engagement counters (tests and logs)
        self.sessions = 0
        self.layers_streamed = 0
        self.bytes_staged = 0
        self.bytes_direct = 0
        self.fallbacks = 0

    # -- session ---------------------------------------------------------------------------------------------------
    def _reset(self) -> None:
        self.slots: list = []
        self.slot_owner: list = [None, None]
        self.ready: dict = {}
        self.done: dict = {}
        self.issued: set = set()
        self.swapped: dict = {}  # layer index -> [(tensor, host)]
        self.active = False

    def _start(self) -> None:
        import torch

        self.slots = [
            torch.empty(self.slot_bytes, dtype = torch.uint8, device = self.device) for _ in range(2)
        ]
        self.active = True
        self.sessions += 1

    def finish(self) -> None:
        """Restore every layer to its host copy and drop the slots. Idempotent."""
        if not self.active and not self.swapped:
            return
        try:
            if self.copy_stream is not None:
                self.copy_stream.synchronize()
        finally:
            for index in list(self.swapped):
                self._restore(index)
            self._reset()

    # -- copies ----------------------------------------------------------------------------------------------------
    def _executor(self) -> Any:
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(
                max_workers = self.threads, thread_name_prefix = "unsloth-h3-te-layer"
            )
        return self._pool

    def _host(self, tensor: Any, group: Any) -> Any:
        cpu = getattr(group, "cpu_param_dict", None) or {}
        host = cpu.get(tensor)
        return host if host is not None else tensor.data

    def _issue(self, index: int) -> None:
        """Queue layer ``index``'s bytes into slot ``index % 2`` on the copy stream."""
        import torch

        if index in self.issued or not (0 <= index < len(self.plans)):
            return
        slot_index = index % 2
        previous = self.slot_owner[slot_index]
        if previous is not None and previous in self.swapped:
            # Out of order (the slot's layer is still swapped in): release it first.
            self._release(previous)
        slot = self.slots[slot_index]
        entries, _groups, _size = self.plans[index]
        stream_ctx = torch.cuda.stream(self.copy_stream) if self._cuda else _nullcontext()
        with stream_ctx:
            if self._cuda and previous is not None and previous in self.done:
                self.copy_stream.wait_event(self.done.pop(previous))
            pageable = []
            for tensor, group, offset, nbytes in entries:
                host = self._host(tensor, group)
                src = host.detach().reshape(-1).view(torch.uint8)
                if not self._cuda:
                    slot[offset : offset + nbytes].copy_(src)
                    self.bytes_direct += nbytes
                elif not self.staging or host.is_pinned():
                    # Pinned: async DMA. No ring: the driver stages the pageable bytes itself.
                    slot[offset : offset + nbytes].copy_(src, non_blocking = True)
                    self.bytes_direct += nbytes
                else:
                    pageable.append((src, offset, nbytes))
            if pageable:
                self._stage(slot, pageable)
            if self._cuda:
                event = torch.cuda.Event()
                event.record(self.copy_stream)
                self.ready[index] = event
        self.slot_owner[slot_index] = index
        self.issued.add(index)

    def _stage(self, slot: Any, pageable: list) -> None:
        """Pageable bytes through the pinned ring: fill a chunk on the pool, send it, move to the next chunk."""
        chunk_bytes = self.staging[0].numel()
        # (src, src offset, slot offset, length) pieces, cut at chunk boundaries of the packed stream.
        pieces: list = []
        fill = 0
        for src, offset, nbytes in pageable:
            done = 0
            while done < nbytes:
                take = min(nbytes - done, chunk_bytes - fill)
                pieces.append((src, done, offset + done, take, fill))
                done += take
                fill += take
                if fill == chunk_bytes:
                    self._flush(slot, pieces)
                    pieces, fill = [], 0
        if pieces:
            self._flush(slot, pieces)

    def _flush(self, slot: Any, pieces: list) -> None:
        import torch

        k = self._next_chunk
        self._next_chunk = (k + 1) % len(self.staging)
        chunk = self.staging[k]
        event = self._chunk_events[k]
        if event is not None:
            event.synchronize()  # its last transfer has left the ring
        jobs = []
        for src, src_off, _slot_off, take, fill in pieces:
            # Split big pieces so the pool shares one tensor's page faults.
            step = max(1 << 20, -(-take // self.threads))
            for start in range(0, take, step):
                n = min(step, take - start)
                jobs.append(
                    (
                        chunk[fill + start : fill + start + n],
                        src[src_off + start : src_off + start + n],
                    )
                )
        if len(jobs) > 1 and self.threads > 1:
            list(self._executor().map(lambda job: job[0].copy_(job[1]), jobs))
        else:
            for dst, src in jobs:
                dst.copy_(src)
        for _src, _src_off, slot_off, take, fill in pieces:
            slot[slot_off : slot_off + take].copy_(chunk[fill : fill + take], non_blocking = True)
            self.bytes_staged += take
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        self._chunk_events[k] = event

    # -- swap ------------------------------------------------------------------------------------------------------
    def _swap_in(self, index: int) -> None:
        import torch

        slot = self.slots[index % 2]
        if self._cuda:
            torch.cuda.current_stream(self.device).wait_event(self.ready.pop(index))
        held = []
        for tensor, group, offset, nbytes in self.plans[index][0]:
            host = self._host(tensor, group)
            view = slot[offset : offset + nbytes].view(tensor.dtype).view(tensor.shape)
            held.append((tensor, host))
            tensor.data = view
        self.swapped[index] = held

    def _restore(self, index: int) -> None:
        for tensor, host in self.swapped.pop(index, ()):
            tensor.data = host

    def _release(self, index: int) -> None:
        import torch

        if index not in self.swapped:
            return
        if self._cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.done[index] = event
        self._restore(index)

    # -- hooks -----------------------------------------------------------------------------------------------------
    def pre(self, index: int) -> None:
        with self._lock:
            if index in self.swapped:
                return
            try:
                if not self.active:
                    self._start()
                self._issue(index)
                self._swap_in(index)
            except Exception:  # noqa: BLE001 -- e.g. no room for the slots: this layer moves on its own
                self.finish()
                self._onload_alone(index)

    def _onload_alone(self, index: int) -> None:
        """Synchronous per-tensor upload of one layer, no slots (what the leaf groups would have moved)."""
        held = []
        try:
            for tensor, group, _offset, _nbytes in self.plans[index][0]:
                host = self._host(tensor, group)
                held.append((tensor, host))
                tensor.data = host.to(self.device)
        finally:
            self.swapped[index] = held
        self.fallbacks += 1

    def post(self, index: int, ok: bool) -> None:
        with self._lock:
            if not self.active:
                self._restore(index)  # a layer uploaded on its own, or nothing
                return
            if not ok:
                self.finish()
                return
            self._release(index)
            self.layers_streamed += 1
            if index + 1 < len(self.plans):
                self._issue(index + 1)
            else:
                # The last layer: drop the slots (the denoiser runs next) once its kernels are queued.
                self.done.clear()
                self.issued.clear()
                self.slots = []
                self.slot_owner = [None, None]
                self.active = False


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: Any) -> bool:
        return False


def _staging_ring(device: Any) -> Optional[list]:
    """The pinned ring, or None when pinning is off, capped (Windows / WSL) or refused."""
    try:
        import os

        import torch

        from .diffusion_memory import (
            GROUP_OFFLOAD_PIN_ENV,
            _pin_budget_mib,
            _pinned_memory_capped,
            registered_pinned_empty,
        )
        from .host_pinning import host_pinning_allowed

        forced = str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower()
        if forced in ("0", "off", "false", "no") or not host_pinning_allowed(device):
            return None
        if _pinned_memory_capped():
            return None
        budget = _pin_budget_mib()
        if budget is None or budget < (_CHUNK_BYTES * _CHUNKS) >> 20:
            return None
        ring = []
        for _ in range(_CHUNKS):
            chunk = registered_pinned_empty(_CHUNK_BYTES)
            if chunk is None:
                chunk = torch.empty(_CHUNK_BYTES, dtype = torch.uint8, pin_memory = True)
            ring.append(chunk)
        return ring
    except Exception:  # noqa: BLE001 -- no ring: driver-staged pageable copies
        return None


def attach_h3_te_layer_stream(
    target: Any, layers: list, plans: list, streamer: H3TELayerStream
) -> None:
    """Hook ``streamer`` onto ``layers`` and quiet their diffusers groups; all or nothing (raises with nothing left)."""
    import torch

    disable = getattr(getattr(torch, "compiler", None), "disable", None)

    def _wrap(fn: Any) -> Any:
        return disable(fn) if callable(disable) else fn

    handles = []
    quieted = []
    try:
        for index, layer in enumerate(layers):

            def _pre(
                _module: Any,
                _args: Any,
                _index: int = index,
            ) -> None:
                streamer.pre(_index)

            def _post(
                _module: Any,
                _args: Any,
                output: Any,
                _index: int = index,
            ) -> None:
                # An exception in the layer leaves ``output`` None: restore everything.
                streamer.post(_index, output is not None)

            handles.append(layer.register_forward_pre_hook(_wrap(_pre)))
            handles.append(layer.register_forward_hook(_wrap(_post), always_call = True))

        def _end(_module: Any, _args: Any, _output: Any) -> None:
            streamer.finish()

        handles.append(target.register_forward_hook(_wrap(_end), always_call = True))
        # The diffusers groups of these layers stay registered (host copies, resident share) but move nothing.
        for _entries, layer_groups, _size in plans:
            for group in layer_groups:
                group.onload_ = _noop
                group.offload_ = _noop
                quieted.append(group)
        setattr(target, _ATTR, streamer)
    except Exception:
        for handle in handles:
            handle.remove()
        for group in quieted:
            group.__dict__.pop("onload_", None)
            group.__dict__.pop("offload_", None)
        raise
    streamer.handles = handles


def install_h3_te_layer_stream(
    text_encoder: Any,
    device: Any,
    *,
    pinned: bool,
    logger: Any = None,
) -> Optional[H3TELayerStream]:
    """Install on a conditioner that ``stream_h3_text_encoder`` put under leaf-level stream offload. ``pinned``: the
    host copy is pinned in place (no ring needed). Returns the streamer, or None with nothing changed. Never raises."""
    try:
        import torch

        if not h3_te_layer_stream_enabled():
            return None
        target = getattr(text_encoder, "model", None)
        language = getattr(target, "language_model", None)
        layers_all = list(getattr(language, "layers", None) or ())
        onload = torch.device(device)
        if target is None or not layers_all or onload.type != "cuda":
            return None
        if getattr(target, _ATTR, None) is not None:
            return getattr(target, _ATTR)
        from .diffusion_memory import _offload_groups

        # Layers with weights only: the terminator slot past the read layer has none.
        layers = [layer for layer in layers_all if _has_tensors(layer)]
        if not layers or layers != layers_all[: len(layers)]:
            return None
        plans = _layer_plans(layers, _offload_groups(target))
        if plans is None:
            if logger is not None:
                logger.info(
                    "video.h3_te_layers: layout not recognised, keeping the leaf-level upload"
                )
            return None
        staging = None if pinned else _staging_ring(onload)
        if not pinned and staging is None:
            # Pinning off or capped (Windows / WSL): the leaf-level upload, as before.
            if logger is not None:
                logger.info(
                    "video.h3_te_layers: no pinned ring available, keeping the leaf-level upload"
                )
            return None
        streamer = H3TELayerStream(layers, plans, onload, staging = staging)
        attach_h3_te_layer_stream(target, layers, plans, streamer)
        if logger is not None:
            logger.info(
                "video.h3_te_layers: %d decoder layers upload as one transfer each (%.0f MB slots x 2, %s)",
                len(layers),
                streamer.slot_bytes / 1e6,
                "pinned host copy"
                if pinned
                else f"pageable host copy through a {(_CHUNK_BYTES * _CHUNKS) >> 20} MiB pinned ring",
            )
        return streamer
    except Exception as exc:  # noqa: BLE001 -- the leaf-level upload still works
        if logger is not None:
            logger.warning("video.h3_te_layers: keeping the leaf-level upload: %s", exc)
        return None

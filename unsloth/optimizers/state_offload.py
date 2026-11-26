# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_OFFLOAD_OPTIMIZER_STATE=1: state on host, chunked to GPU in step; parameter-local only.

On CUDA the chunk copies overlap the step through a fixed staging ring no larger than one serial
chunk (UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP=0 keeps them serial)."""

import contextlib
import copy
import os

import torch

__all__ = [
    "optimizer_state_offload_enabled",
    "offload_optimizer_state",
    "install_optimizer_state_offload",
]

_ENV = "UNSLOTH_OFFLOAD_OPTIMIZER_STATE"

try:
    from torch.distributed.tensor import DTensor
except Exception:
    DTensor = None


def _chunk_bytes():
    try:
        mb = float(os.environ.get(_ENV + "_CHUNK_MB", "1024"))
    except ValueError:
        mb = 1024.0
    return int(mb * 2**20) if mb > 0 else float("inf")


def optimizer_state_offload_enabled():
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _overlap_enabled():
    return os.environ.get(_ENV + "_OVERLAP", "1").strip().lower() not in ("0", "false", "no", "off")


_RING_ALIGN = 512
_RING_PARTS = 4
_RING_SEGMENT_BYTES = 64 << 20


def _innermost(optimizer):
    # accelerate's AcceleratedOptimizer is itself an Optimizer subclass wrapping .optimizer.
    while isinstance(getattr(optimizer, "optimizer", None), torch.optim.Optimizer):
        optimizer = optimizer.optimizer
    return optimizer


# Classes whose step only touches parameters that have a gradient, so stepping one chunk at a time
# is the same update. Others (schedulefree advances group["k"] on every call) step unchunked.
_CHUNK_SAFE_MODULES = (
    "torch.optim.",
    "bitsandbytes.",
    "transformers.optimization",
    "unsloth.optimizers.",
)


def _grad_only_step(optimizer):
    for cls in type(optimizer).__mro__:
        if "step" in vars(cls):
            return cls.__module__.startswith(_CHUNK_SAFE_MODULES)
    return False


def _chunk_safe(optimizer):
    # Q-GaLore weight_quant swaps p.data for a one element placeholder inside step, so the full
    # size gradients hidden between chunks can no longer be put back on it.
    if any(group.get("weight_quant", False) for group in optimizer.param_groups):
        return False
    return _grad_only_step(optimizer)


def _tensor_attrs(value):
    # Tensors held by state objects such as GaLoreProjector (__slots__, no __dict__).
    if isinstance(value, (torch.Tensor, dict, list, tuple, str, int, float, bool)) or value is None:
        return ()
    names = set(getattr(value, "__dict__", ()))
    for cls in type(value).__mro__:
        slots = vars(cls).get("__slots__", ())
        names.update((slots,) if isinstance(slots, str) else slots)
    return [
        (name, getattr(value, name))
        for name in sorted(names)
        if isinstance(getattr(value, name, None), torch.Tensor)
    ]


def _held_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    # LBFGS keeps its history (old_dirs, old_stps, ro) as lists of tensors.
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, torch.Tensor)]
    return [t for _, t in _tensor_attrs(value)]


def _local(tensor):
    """This rank's shard of a DTensor (FSDP2), whose numel() is the global one; else the tensor."""
    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


def _with_local(value, local):
    """`local` in the place of `value`: a DTensor keeps its wrapper (DCP sees the stock layout)."""
    if DTensor is not None and isinstance(value, DTensor):
        value = copy.copy(value)
        value._local_tensor = local
        return value
    return local


def _local_nbytes(tensor):
    local = _local(tensor)
    return local.numel() * local.element_size()


def _source_device(local, param, key, device_of):
    if local.device.type != "cpu":
        return local.device
    return device_of(param, key, local) if device_of is not None else None


def _synchronize(device):
    module = getattr(torch, device.type, None)
    sync = getattr(module, "synchronize", None)
    if sync is not None:
        # torch.mps.synchronize takes no device argument.
        sync() if device.type == "mps" else sync(device)


def _plain_copy_source(tensor, host):
    # A tensor the side stream can copy straight into `host` (no temporary on that stream).
    return (
        type(tensor) is torch.Tensor
        and tensor.device.type == "cuda"
        and tensor.dtype == host.dtype
        and tensor.shape == host.shape
        and tensor.is_contiguous()
    )


class _Ring:
    """Device staging buffer: copies in on one side stream, out on another, events order reuse.
    Nothing is allocated on a side stream (record_stream only on state a step replaced)."""

    def __init__(self, device, capacity, streams):
        self.device, self.capacity = device, capacity
        self.compute = torch.cuda.current_stream(device)
        self.h2d, self.d2h = streams
        self.buffer = torch.empty(capacity, dtype = torch.uint8, device = device)
        # The buffer may reuse memory the compute stream is still working on.
        self.h2d.wait_stream(self.compute)
        self.d2h.wait_stream(self.compute)
        self.head = 0
        self.live = []
        self.done = []  # (start, end, event) of copied out ranges, in copy order

    def place(self, tensors):
        """Copy [(param, key, host, padded bytes)] in: (views, ready event, range), or None if full."""
        nbytes = sum(n for *_, n in tensors)
        start = 0 if self.head + nbytes > self.capacity else self.head
        end = start + nbytes
        if any(s < end and start < e for s, e in self.live):
            return None
        last = -1
        for i, (s, e, _) in enumerate(self.done):
            if s < end and start < e:
                last = i
        if last >= 0:
            # Copies out finish in order, so the newest overlapping one covers the older ones.
            self.h2d.wait_event(self.done[last][2])
            self.done = self.done[last + 1 :]
        views, offset = [], start
        with torch.cuda.stream(self.h2d):
            for p, key, host, n in tensors:
                raw = self.buffer[offset : offset + host.numel() * host.element_size()]
                view = raw.view(host.dtype).view(host.shape)
                view.copy_(host, non_blocking = True)
                views.append((p, key, host, view, offset, n))
                offset += n
        ready = torch.cuda.Event()
        ready.record(self.h2d)
        self.live.append((start, end))
        self.head = end
        return views, ready, (start, end)

    def use(self, placed, state):
        views, ready, _ = placed
        self.compute.wait_event(ready)
        for p, key, _, view, _, _ in views:
            state[p][key] = _with_local(state[p][key], view)

    def copy_out(self, placed, state):
        views, _, (start, end) = placed
        stepped = torch.cuda.Event()
        stepped.record(self.compute)
        self.d2h.wait_event(stepped)
        segment, copied = start, 0
        with torch.cuda.stream(self.d2h):
            for p, key, host, view, offset, n in views:
                current = state[p].get(key)
                local = _local(current)
                if local is view:
                    host.copy_(view, non_blocking = True)
                elif _plain_copy_source(local, host):
                    # Replaced by the step (bfloat16 stochastic rounding stores new moments).
                    host.copy_(local, non_blocking = True)
                    local.record_stream(self.d2h)
                else:
                    continue
                state[p][key] = _with_local(current, host)
                copied += n
                if copied >= _RING_SEGMENT_BYTES:
                    event = torch.cuda.Event()
                    event.record(self.d2h)
                    self.done.append((segment, offset + n, event))
                    segment, copied = offset + n, 0
        event = torch.cuda.Event()
        event.record(self.d2h)
        self.done.append((segment, end, event))
        self.live.remove((start, end))

    def close(self):
        # The buffer goes back to the compute stream's pool only after every copy.
        self.compute.wait_stream(self.h2d)
        self.compute.wait_stream(self.d2h)
        self.buffer = None


class _Sequence:
    __slots__ = ("ring", "parts", "placed", "next", "current", "stopped")

    def __init__(self, ring, parts):
        self.ring, self.parts = ring, parts
        self.placed = []
        self.next = 0
        self.current = None
        self.stopped = False


class _StateOffloader:
    def __init__(
        self,
        optimizer,
        pin_memory = True,
        chunk_bytes = None,
    ):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self.chunk_bytes = chunk_bytes if chunk_bytes is not None else _chunk_bytes()
        self.buffers = {}
        self.offloaded = {}
        self.offloaded_bytes = 0
        self.steps = 0
        self.inside = False
        self.hidden = None
        self.pending = set()
        self.overlapped_steps = 0
        self.streams = None
        self.sequence = None
        self.ring_oom_fallbacks = 0

    def _host_buffer(self, param, key, src):
        buffer = self.buffers.get((param, key))
        if buffer is None or buffer.shape != src.shape or buffer.dtype != src.dtype:
            try:
                buffer = torch.empty(
                    src.shape, dtype = src.dtype, device = "cpu", pin_memory = self.pin_memory
                )
            except RuntimeError:
                # No pinnable host memory (some WSL / container setups): pageable still works.
                self.pin_memory = False
                buffer = torch.empty(src.shape, dtype = src.dtype, device = "cpu")
            self.buffers[(param, key)] = buffer
        buffer.copy_(src, non_blocking = self.pin_memory)
        return buffer

    def _states(self, params):
        if params is None:
            return list(self.optimizer.state.items())
        state = self.optimizer.state
        return [(p, state[p]) for p in params if p in state]

    def offload(
        self,
        params = None,
        device_of = None,
    ):
        # device_of returns the device for host-resident state, or None to leave it host-only.
        devices = set()
        for param, state in self._states(params):
            for key, value in list(state.items()):
                if isinstance(value, (list, tuple)):
                    items = list(value)
                    for i, local in enumerate(items):
                        if not isinstance(local, torch.Tensor):
                            continue
                        device = _source_device(local, param, (key, i), device_of)
                        if device is not None:
                            devices.add(local.device)
                            items[i] = self._host_buffer(param, (key, i), local)
                            self.offloaded[(param, (key, i))] = device
                    if isinstance(value, list):
                        value[:] = items
                    else:
                        state[key] = type(value)(items)
                    continue
                for name, local in _tensor_attrs(value):
                    device = _source_device(local, param, (key, name), device_of)
                    if device is None:
                        continue
                    devices.add(local.device)
                    setattr(value, name, self._host_buffer(param, (key, name), local))
                    self.offloaded[(param, (key, name))] = device
                is_dtensor = DTensor is not None and isinstance(value, DTensor)
                local = value._local_tensor if is_dtensor else value
                if not isinstance(local, torch.Tensor):
                    continue
                device = _source_device(local, param, key, device_of)
                if device is None:
                    continue
                devices.add(local.device)
                host = self._host_buffer(param, key, local)
                self.offloaded[(param, key)] = device
                if is_dtensor:
                    value = copy.copy(value)
                    value._local_tensor = host
                    state[key] = value
                else:
                    state[key] = host
        return devices

    def finish(self, devices):
        # The host copies must be complete before anything reads them (torch.save, the next step).
        for device in devices:
            _synchronize(device)
        self.offloaded_bytes = sum(b.numel() * b.element_size() for b in self.buffers.values())

    def restore(self, params = None):
        for param, state in self._states(params):
            for key, value in list(state.items()):
                if isinstance(value, (list, tuple)):
                    items = list(value)
                    for i, local in enumerate(items):
                        device = self.offloaded.get((param, (key, i)))
                        if (
                            device is not None
                            and isinstance(local, torch.Tensor)
                            and local.device != device
                        ):
                            items[i] = local.to(device, non_blocking = True)
                    if isinstance(value, list):
                        value[:] = items
                    else:
                        state[key] = type(value)(items)
                    continue
                for name, local in _tensor_attrs(value):
                    device = self.offloaded.get((param, (key, name)))
                    if device is not None and local.device != device:
                        setattr(value, name, local.to(device, non_blocking = True))
                device = self.offloaded.get((param, key))
                if device is None:
                    continue
                is_dtensor = DTensor is not None and isinstance(value, DTensor)
                local = value._local_tensor if is_dtensor else value
                if not isinstance(local, torch.Tensor) or local.device == device:
                    continue
                local = local.to(device, non_blocking = True)
                if is_dtensor:
                    value = copy.copy(value)
                    value._local_tensor = local
                    state[key] = value
                else:
                    state[key] = local

    def _chunks(self, params):
        # Optimizers update only parameters that have a gradient, so stepping one chunk at a time
        # (other gradients hidden) bounds the on-device state to chunk_bytes instead of all of it.
        chunks, chunk, size = [], [], 0
        state = self.optimizer.state
        groups = {p: g for g in self.optimizer.param_groups for p in g["params"]}
        for p in params:
            held = [t for v in state.get(p, {}).values() for t in _held_tensors(v)]
            if held:
                nbytes = sum(_local_nbytes(v) for v in held)
            else:
                # Not stepped yet: two moments, a third with amsgrad or centered RMSprop with
                # momentum (square_avg, momentum_buffer, grad_avg), at least fp32 each.
                group = groups.get(p, {})
                # bitsandbytes AdEMAMix keeps a double-sized state1 next to state2.
                third = (
                    group.get("amsgrad")
                    or (group.get("centered") and group.get("momentum", 0) > 0)
                    or "ademamix" in type(self.optimizer).__name__.lower()
                )
                moments = 3 if third else 2
                nbytes = moments * max(p.element_size(), 4) * _local(p).numel()
            if chunk and size + nbytes > self.chunk_bytes:
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append(p)
            size += nbytes
        if chunk:
            chunks.append(chunk)
        return chunks

    def _pre_step(self, optimizer, args, kwargs):
        if self.inside:
            return
        # torch passes the step's own args to hooks, self included.
        has_self = bool(args) and args[0] is optimizer
        rest = args[1:] if has_self else args
        closure = rest[0] if rest else kwargs.get("closure")
        # A closure recomputes the loss on every call, so it must run once over all parameters.
        if closure is not None or not _chunk_safe(optimizer):
            if self.sequence is not None:
                self._sequence_end(self.sequence)
            if closure is not None or not _grad_only_step(optimizer):
                self.restore()
                return

            # Unchunked weight_quant Q-GaLore still skips parameters without a gradient. Select
            # them from the step's closure, after later pre hooks that may create gradients.
            def _restore_active():
                self.restore(
                    [
                        p
                        for group in optimizer.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                )
                return None

            if rest:
                rest = (_restore_active,) + tuple(rest[1:])
            else:
                kwargs = dict(kwargs, closure = _restore_active)
            return ((optimizer,) + tuple(rest) if has_self else tuple(rest)), kwargs

        # Chunk-safe steps call a closure first, before any update, so the chunked updates run
        # after every step pre hook (registered later ones too), as a plain step would.
        def _chunked_updates():
            self._step_chunks(optimizer, rest[1:], dict(kwargs))
            return None

        if rest:
            rest = (_chunked_updates,) + tuple(rest[1:])
        else:
            kwargs = dict(kwargs, closure = _chunked_updates)
        return ((optimizer,) + tuple(rest) if has_self else tuple(rest)), kwargs

    def _ring_tensors(self, params):
        """(device, {param: [(param, key, host, padded bytes)]}) for one CUDA device, or None to stay
        serial (ROCm, CPU, first step, nested state, pageable memory, kill switch). FSDP2 DTensor
        state is staged by its local shard, the only part this rank holds."""
        if not _overlap_enabled() or not self.pin_memory:
            return None
        if getattr(torch.version, "hip", None) or not torch.cuda.is_available():
            return None
        state = self.optimizer.state
        device = None
        sizes = {}
        for p in params:
            entries = state.get(p)
            if not entries:
                return None
            tensors = []
            for key, value in entries.items():
                if not isinstance(value, torch.Tensor):
                    if _held_tensors(value):
                        return None
                    continue
                value = _local(value)
                target = self.offloaded.get((p, key))
                if target is None:
                    if value.device.type != "cpu":
                        return None
                    continue
                if value is not self.buffers.get((p, key)) or target.type != "cuda":
                    return None
                if device is None:
                    device = target
                elif target != device:
                    return None
                nbytes = value.numel() * value.element_size()
                tensors.append((p, key, value, -(-nbytes // _RING_ALIGN) * _RING_ALIGN))
            sizes[p] = tensors
        if device is None:
            return None
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device, sizes

    def _streams(self, device):
        if self.streams is None or self.streams[0].device != device:
            self.streams = (torch.cuda.Stream(device), torch.cuda.Stream(device))
        return self.streams

    def _new_ring(self, device, capacity):
        """The staging ring, or None (serial copies) if its one contiguous block hits OOM (fragmentation)."""
        try:
            return _Ring(device, capacity, self._streams(device))
        except torch.cuda.OutOfMemoryError:
            self.ring_oom_fallbacks += 1
            return None

    def _ring_plan(self, params, chunks):
        found = self._ring_tensors(params)
        if found is None:
            return None
        device, sizes = found
        nbytes = {p: sum(n for *_, n in sizes[p]) for p in params}
        # As large as the biggest serial chunk, so device staging never grows.
        capacity = max(sum(nbytes[p] for p in chunk) for chunk in chunks)
        target = max(capacity // _RING_PARTS, 1)
        parts, part, size = [], [], 0
        for p in params:
            if part and size + nbytes[p] > target:
                parts.append(part)
                part, size = [], 0
            part.append(p)
            size += nbytes[p]
        if part:
            parts.append(part)
        if len(parts) <= 1 or capacity <= 0:
            return None
        return device, capacity, [(part, [t for p in part for t in sizes[p]]) for part in parts]

    def _step_ring(self, optimizer, step, args, kwargs, params, grads, plan, ring, devices):
        """Chunked step with neighbouring parts' copies overlapping each part's step; same updates."""
        device, capacity, parts = plan
        state = self.optimizer.state
        grad_of = dict(zip(params, grads))
        devices.add(device)
        for p in params:
            p.grad = None
        placed = []
        try:
            for index, (members, _) in enumerate(parts):
                while len(placed) < len(parts):
                    slot = ring.place(parts[len(placed)][1])
                    if slot is None:
                        break
                    placed.append(slot)
                ring.use(placed[index], state)
                for p in members:
                    p.grad = grad_of[p]
                step(optimizer, *args, **kwargs)
                for p in members:
                    p.grad = None
                ring.copy_out(placed[index], state)
                placed[index] = None
                # State the step added, or could not copy on the side stream.
                devices |= self.offload(members)
        finally:
            ring.close()
            for p, g in grad_of.items():
                p.grad = g
        self.overlapped_steps += 1

    @contextlib.contextmanager
    def pipelined(self, batches):
        """Step calls for these batches follow in order (bf16 stochastic rounding: one weight per call):
        prefetch the next batches' state while each steps, one sync at the end instead of per call."""
        found = None
        flat = [p for batch in batches for p in batch]
        if self.sequence is None and len(batches) > 1 and len({id(p) for p in flat}) == len(flat):
            found = self._ring_tensors(flat)
        if found is None:
            yield
            return
        device, sizes = found
        parts = [(list(batch), [t for p in batch for t in sizes[p]]) for batch in batches]
        # One batch's staging at most, as each serial call held.
        capacity = max(sum(n for *_, n in tensors) for _, tensors in parts)
        if capacity <= 0:
            yield
            return
        ring = self._new_ring(device, capacity)
        if ring is None:
            yield
            return
        sequence = _Sequence(ring, parts)
        self.sequence = sequence
        try:
            yield
            complete = (
                not sequence.stopped and sequence.current is None and sequence.next == len(parts)
            )
        finally:
            self._sequence_end(sequence)
        if complete:
            self.overlapped_steps += 1

    def _sequence_end(self, sequence):
        if sequence.stopped:
            return
        sequence.stopped = True
        if self.sequence is sequence:
            self.sequence = None
        sequence.ring.close()
        # Anything an error or an unexpected call left on the device.
        self.finish({sequence.ring.device} | self.offload())

    def _sequence_take(self, params):
        """Serve this step call from the sequence; False (sequence ended) if it is not the next batch."""
        sequence = self.sequence
        index = sequence.next
        expected = sequence.parts[index][0] if index < len(sequence.parts) else None
        if (
            sequence.current is not None
            or expected is None
            or len(expected) != len(params)
            or any(a is not b for a, b in zip(expected, params))
        ):
            self._sequence_end(sequence)
            return False
        ring = sequence.ring
        while len(sequence.placed) < len(sequence.parts):
            slot = ring.place(sequence.parts[len(sequence.placed)][1])
            if slot is None:
                break
            sequence.placed.append(slot)
        ring.use(sequence.placed[index], self.optimizer.state)
        sequence.current = index
        sequence.next = index + 1
        return True

    def _step_chunks(self, optimizer, args, kwargs):
        kwargs.pop("closure", None)
        params = [
            p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
        ]
        chunks = self._chunks(params)
        # Chunk-safe steps never touch parameters without a gradient, so their state stays on host.
        if len(chunks) <= 1:
            if self.sequence is None or not self._sequence_take(params):
                self.restore(params)
            return
        # Step each chunk through the unhooked class step, so registered step hooks run once for
        # the logical step, then hide every gradient so the caller's own step finds nothing left.
        step = type(optimizer).step
        if getattr(step, "hooked", False):
            step = getattr(step, "__wrapped__", step)
        if self.sequence is not None:
            # Not a per-weight call of the sequence: its prefetched copies must not go stale.
            self._sequence_end(self.sequence)
        grads = [p.grad for p in params]
        devices = set()
        plan = self._ring_plan(params, chunks)
        ring = self._new_ring(plan[0], plan[1]) if plan is not None else None
        self.inside = True
        try:
            if ring is not None:
                self._step_ring(optimizer, step, args, kwargs, params, grads, plan, ring, devices)
            else:
                for chunk in chunks:
                    members = set(chunk)
                    for p, g in zip(params, grads):
                        p.grad = g if p in members else None
                    self.restore(chunk)
                    step(optimizer, *args, **kwargs)
                    devices |= self.offload(chunk)
        except BaseException:
            for p, g in zip(params, grads):
                p.grad = g
            self.finish(devices | self.offload())
            raise
        finally:
            self.inside = False
        for p in params:
            p.grad = None
        self.hidden = (params, grads)
        # The chunk copies are still in flight; the post hook's offload finds nothing left to copy.
        self.pending = devices

    def _post_step(self, optimizer, args, kwargs):
        if self.inside:
            return
        if self.hidden is not None:
            for p, g in zip(*self.hidden):
                p.grad = g
            self.hidden = None
        self.steps += 1
        sequence = self.sequence
        if sequence is not None and sequence.current is not None:
            members = sequence.parts[sequence.current][0]
            sequence.ring.copy_out(sequence.placed[sequence.current], self.optimizer.state)
            sequence.placed[sequence.current] = None
            sequence.current = None
            # Copies still in flight; the sequence synchronizes once when it ends.
            self.offload(members)
            return
        devices, self.pending = self.pending, set()
        self.finish(devices | self.offload())


def offload_optimizer_state(
    optimizer,
    pin_memory = True,
    chunk_bytes = None,
):
    """Keep `optimizer`'s state on the host between steps. Idempotent; returns the offloader or None."""
    optimizer = _innermost(optimizer)
    if not isinstance(optimizer, torch.optim.Optimizer):
        return None
    # LayerWiseDummyOptimizer keeps real optimizers in optimizer_dict, stepped by grad hooks.
    layerwise = getattr(optimizer, "optimizer_dict", None)
    if isinstance(layerwise, dict):
        unique = {id(o): o for o in layerwise.values()}.values()
        offloaders = [offload_optimizer_state(o, pin_memory, chunk_bytes) for o in unique]
        offloaders = [o for o in offloaders if o is not None]
        return offloaders or None
    existing = getattr(optimizer, "_unsloth_state_offloader", None)
    if existing is not None:
        return existing
    # Paged bitsandbytes state already spills to host through unified memory.
    if getattr(optimizer, "is_paged", False):
        return None
    if not any(p.device.type != "cpu" for group in optimizer.param_groups for p in group["params"]):
        return None

    # The fused 8-bit AdamW (flat_adamw8bit) runs stock under an offloader; drop its packed buffers.
    optimizer.__dict__.pop("_unsloth_flat_plans", None)
    offloader = _StateOffloader(optimizer, pin_memory = pin_memory, chunk_bytes = chunk_bytes)
    # Step hooks, not a step override: accelerate swaps the instance's step on each fp16 step.
    optimizer.register_step_pre_hook(offloader._pre_step)
    optimizer.register_step_post_hook(offloader._post_step)

    # Instance override, not register_load_state_dict_post_hook: bitsandbytes replaces
    # Optimizer.load_state_dict and does not run torch's post hooks. Loading moves every
    # state tensor to its parameter's device, so offload again right after.
    load_state_dict = optimizer.load_state_dict

    def _load_state_dict_and_offload(state_dict, *args, **kwargs):
        moved = {}
        original = vars(torch.optim.Optimizer).get("_process_value_according_to_param_policy")
        if isinstance(original, staticmethod):
            policy = original.__func__

            # torch casts every loaded state tensor onto its parameter's device; keep it on the
            # host instead so a resume never holds the whole state on device at once.
            def _host_policy(param, value, *policy_args, **policy_kwargs):
                if param.device.type in ("cpu", "meta"):
                    return policy(param, value, *policy_args, **policy_kwargs)
                stand_in = torch.empty(0, dtype = param.dtype, device = "meta")
                out = policy(stand_in, value, *policy_args, **policy_kwargs)
                if out.device.type != "meta":
                    return out
                out = value.to(dtype = out.dtype)
                moved[id(out)] = param.device
                if DTensor is not None and isinstance(out, DTensor):
                    # offload() looks a DTensor up by its local shard.
                    moved[id(out._local_tensor)] = param.device
                return out

            torch.optim.Optimizer._process_value_according_to_param_policy = staticmethod(
                _host_policy
            )
        try:
            result = load_state_dict(state_dict, *args, **kwargs)
        finally:
            if isinstance(original, staticmethod):
                torch.optim.Optimizer._process_value_according_to_param_policy = original
        if moved:
            device_of = lambda param, key, local: moved.get(id(local))
        elif callable(getattr(optimizer, "to_gpu", None)) and hasattr(optimizer, "initialized"):
            # bitsandbytes loads state on the host and its first step moves all of it to the device
            # (to_gpu), defeating the chunks: do its first-step setup here, minus that move.
            if not optimizer.initialized:
                if callable(getattr(optimizer, "check_overrides", None)):
                    optimizer.check_overrides()
                optimizer.initialized = True
            device_of = lambda param, key, local: (
                param.device
                if not isinstance(key, tuple)
                and param.device.type != "cpu"
                and not getattr(local, "is_paged", False)
                else None
            )
        else:
            device_of = None
        offloader.finish(offloader.offload(device_of = device_of))
        return result

    optimizer.load_state_dict = _load_state_dict_and_offload
    optimizer._unsloth_state_offloader = offloader
    offloader.finish(offloader.offload())
    return offloader


_PRINTED = False


def install_optimizer_state_offload(trainer):
    """Offload the trainer's optimizer state once its optimizer is final (inside create_scheduler)."""
    if not optimizer_state_offload_enabled():
        return False
    if getattr(trainer, "is_deepspeed_enabled", False):
        print(
            "Unsloth: UNSLOTH_OFFLOAD_OPTIMIZER_STATE is ignored under DeepSpeed (use its offload_optimizer)."
        )
        return False
    create_scheduler = trainer.create_scheduler
    if getattr(create_scheduler, "_unsloth_state_offload", False):
        return True

    def _create_scheduler_offload_state(*args, **kwargs):
        global _PRINTED
        scheduler = create_scheduler(*args, **kwargs)
        optimizer = kwargs.get("optimizer", args[1] if len(args) > 1 else None)
        optimizer = optimizer if optimizer is not None else getattr(trainer, "optimizer", None)
        if (
            optimizer is not None
            and offload_optimizer_state(optimizer) is not None
            and not _PRINTED
        ):
            _PRINTED = True
            print(
                "Unsloth: Optimizer state is offloaded to CPU between steps (UNSLOTH_OFFLOAD_OPTIMIZER_STATE=1)."
            )
        return scheduler

    _create_scheduler_offload_state._unsloth_state_offload = True
    trainer.create_scheduler = _create_scheduler_offload_state
    return True

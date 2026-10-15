# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Optimizer-state-only CPU offload (UNSLOTH_OFFLOAD_OPTIMIZER_STATE=1).

State lives in pinned host memory between steps and visits the GPU only inside optimizer.step,
so peak memory is max(activations, optimizer state) instead of their sum. The step runs in chunks
of UNSLOTH_OFFLOAD_OPTIMIZER_STATE_CHUNK_MB (default 1024) of state, so the step itself never holds
all of it on device either (parameter-local optimizers only, see _CHUNK_SAFE_MODULES).
"""

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


def _innermost(optimizer):
    # accelerate's AcceleratedOptimizer is itself an Optimizer subclass wrapping .optimizer.
    while isinstance(getattr(optimizer, "optimizer", None), torch.optim.Optimizer):
        optimizer = optimizer.optimizer
    return optimizer


# Classes whose step only touches parameters that have a gradient, so stepping one chunk at a time
# is the same update. Others (schedulefree advances group["k"] on every call) step unchunked.
_CHUNK_SAFE_MODULES = ("torch.optim.", "bitsandbytes.", "transformers.optimization", "unsloth.optimizers.")


def _chunk_safe(optimizer):
    for cls in type(optimizer).__mro__:
        if "step" in vars(cls):
            return cls.__module__.startswith(_CHUNK_SAFE_MODULES)
    return False


def _tensor_attrs(value):
    # Tensors held by state objects such as GaLoreProjector (__slots__, no __dict__).
    if isinstance(value, (torch.Tensor, dict, list, tuple, str, int, float, bool)) or value is None:
        return ()
    names = set(getattr(value, "__dict__", ()))
    for cls in type(value).__mro__:
        slots = vars(cls).get("__slots__", ())
        names.update((slots,) if isinstance(slots, str) else slots)
    return [
        (name, getattr(value, name)) for name in sorted(names)
        if isinstance(getattr(value, name, None), torch.Tensor)
    ]


def _held_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    # LBFGS keeps its history (old_dirs, old_stps, ro) as lists of tensors.
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, torch.Tensor)]
    return [t for _, t in _tensor_attrs(value)]


def _source_device(local, param, key, device_of):
    if local.device.type != "cpu":
        return local.device
    return device_of(param, key, local) if device_of is not None else None


def _synchronize(device):
    module = getattr(torch, device.type, None)
    sync = getattr(module, "synchronize", None)
    if sync is not None:
        sync(device)


class _StateOffloader:
    def __init__(self, optimizer, pin_memory = True, chunk_bytes = None):
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

    def offload(self, params = None, device_of = None):
        # device_of(param, key, tensor) names the device of state that is already on the host
        # (a checkpoint just loaded there), None to leave it host-only.
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
                        if device is not None and isinstance(local, torch.Tensor) and local.device != device:
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
                nbytes = sum(v.numel() * v.element_size() for v in held)
            else:
                # Not stepped yet: two moments, a third with amsgrad or centered RMSprop with
                # momentum (square_avg, momentum_buffer, grad_avg), at least fp32 each.
                group = groups.get(p, {})
                third = group.get("amsgrad") or (group.get("centered") and group.get("momentum", 0) > 0)
                moments = 3 if third else 2
                nbytes = moments * max(p.element_size(), 4) * p.numel()
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
            self.restore()
            return
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

    def _step_chunks(self, optimizer, args, kwargs):
        kwargs.pop("closure", None)
        params = [
            p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
        ]
        chunks = self._chunks(params)
        # Chunk-safe steps never touch parameters without a gradient, so their state stays on host.
        if len(chunks) <= 1:
            self.restore(params)
            return
        # Step each chunk through the unhooked class step, so registered step hooks run once for
        # the logical step, then hide every gradient so the caller's own step finds nothing left.
        step = type(optimizer).step
        if getattr(step, "hooked", False):
            step = getattr(step, "__wrapped__", step)
        grads = [p.grad for p in params]
        devices = set()
        self.inside = True
        try:
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
        devices, self.pending = self.pending, set()
        self.finish(devices | self.offload())


def offload_optimizer_state(optimizer, pin_memory = True, chunk_bytes = None):
    """Keep `optimizer`'s state on the host between steps. Idempotent; returns the offloader or None."""
    optimizer = _innermost(optimizer)
    if not isinstance(optimizer, torch.optim.Optimizer):
        return None
    # transformers' LayerWiseDummyOptimizer (galore/apollo *_layerwise): the real per-parameter
    # optimizers live in optimizer_dict and step from gradient hooks.
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
    if not any(
        p.device.type != "cpu" for group in optimizer.param_groups for p in group["params"]
    ):
        return None

    offloader = _StateOffloader(optimizer, pin_memory = pin_memory, chunk_bytes = chunk_bytes)
    # Hooks, not an optimizer.step override: accelerate swaps the instance's step attribute on
    # every fp16 step, and hooks run whichever bound step a caller kept.
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
                return out

            torch.optim.Optimizer._process_value_according_to_param_policy = staticmethod(_host_policy)
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
                if not isinstance(key, tuple) and param.device.type != "cpu"
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
        print("Unsloth: UNSLOTH_OFFLOAD_OPTIMIZER_STATE is ignored under DeepSpeed (use its offload_optimizer).")
        return False
    create_scheduler = trainer.create_scheduler
    if getattr(create_scheduler, "_unsloth_state_offload", False):
        return True

    def _create_scheduler_offload_state(*args, **kwargs):
        global _PRINTED
        scheduler = create_scheduler(*args, **kwargs)
        optimizer = kwargs.get("optimizer", args[1] if len(args) > 1 else None)
        optimizer = optimizer if optimizer is not None else getattr(trainer, "optimizer", None)
        if optimizer is not None and offload_optimizer_state(optimizer) is not None and not _PRINTED:
            _PRINTED = True
            print("Unsloth: Optimizer state is offloaded to CPU between steps (UNSLOTH_OFFLOAD_OPTIMIZER_STATE=1).")
        return scheduler

    _create_scheduler_offload_state._unsloth_state_offload = True
    trainer.create_scheduler = _create_scheduler_offload_state
    return True

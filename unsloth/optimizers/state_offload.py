# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimizer-state-only CPU offload (UNSLOTH_OFFLOAD_OPTIMIZER_STATE=1).

State lives in pinned host memory between steps and visits the GPU only inside optimizer.step,
so peak memory is max(activations, optimizer state) instead of their sum. The step runs in chunks
of UNSLOTH_OFFLOAD_OPTIMIZER_STATE_CHUNK_MB (default 1024) of state, so the step itself never holds
all of it on device either.
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

    def offload(self, params = None):
        devices = set()
        for param, state in self._states(params):
            for key, value in state.items():
                is_dtensor = DTensor is not None and isinstance(value, DTensor)
                local = value._local_tensor if is_dtensor else value
                if not isinstance(local, torch.Tensor) or local.device.type == "cpu":
                    continue
                devices.add(local.device)
                host = self._host_buffer(param, key, local)
                self.offloaded[(param, key)] = local.device
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
            for key, value in state.items():
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
        for p in params:
            held = [v for v in state.get(p, {}).values() if isinstance(v, torch.Tensor)]
            nbytes = sum(v.numel() * v.element_size() for v in held) if held else 8 * p.numel()
            if chunk and size + nbytes > self.chunk_bytes:
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append(p)
            size += nbytes
        if chunk:
            chunks.append(chunk)
        return chunks

    def step(self, step, args, kwargs):
        self.steps += 1
        params = [
            p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None
        ]
        chunks = self._chunks(params)
        closure = args[0] if args else kwargs.get("closure")
        # A closure recomputes the loss on every call, so it must run once over all parameters.
        if len(chunks) <= 1 or closure is not None:
            self.restore()
            try:
                return step(*args, **kwargs)
            finally:
                self.finish(self.offload())
        grads = [p.grad for p in params]
        devices = set()
        try:
            for chunk in chunks:
                members = set(chunk)
                for p, g in zip(params, grads):
                    p.grad = g if p in members else None
                self.restore(chunk)
                result = step(*args, **kwargs)
                devices |= self.offload(chunk)
        finally:
            for p, g in zip(params, grads):
                p.grad = g
            # A failed chunk can leave restored state of other params on device.
            devices |= self.offload()
            self.finish(devices)
        return result


def offload_optimizer_state(optimizer, pin_memory = True, chunk_bytes = None):
    """Keep `optimizer`'s state on the host between steps. Idempotent; returns the offloader or None."""
    optimizer = _innermost(optimizer)
    if not isinstance(optimizer, torch.optim.Optimizer):
        return None
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
    # Instance attribute: accelerate, GradScaler and LR schedulers all call optimizer.step.
    step = optimizer.step

    def _step_offloaded(*args, **kwargs):
        return offloader.step(step, args, kwargs)

    optimizer.step = _step_offloaded

    # Instance override, not register_load_state_dict_post_hook: bitsandbytes replaces
    # Optimizer.load_state_dict and does not run torch's post hooks. Loading moves every
    # state tensor to its parameter's device, so offload again right after.
    load_state_dict = optimizer.load_state_dict

    def _load_state_dict_and_offload(state_dict, *args, **kwargs):
        result = load_state_dict(state_dict, *args, **kwargs)
        offloader.finish(offloader.offload())
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

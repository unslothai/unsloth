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

Optimizer state (Adam moments, bitsandbytes 8-bit state + absmax) lives in pinned host memory
between steps and is copied to the accelerator only for optimizer.step, so the GPU peak becomes
max(activations, optimizer state) instead of their sum. Weights and gradients stay on the GPU.
Works on any torch.optim.Optimizer subclass through step pre / post hooks, so the optimizer
object, its class and every caller (accelerate, GradScaler, LR schedulers) stay unchanged.
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
    def __init__(self, optimizer, pin_memory = True):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self.buffers = {}
        # (param, key) -> device the tensor came from, restored before every step.
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

    def offload(self):
        devices = set()
        total = 0
        for param, state in self.optimizer.state.items():
            for key, value in state.items():
                is_dtensor = DTensor is not None and isinstance(value, DTensor)
                local = value._local_tensor if is_dtensor else value
                if not isinstance(local, torch.Tensor) or local.device.type == "cpu":
                    continue
                devices.add(local.device)
                host = self._host_buffer(param, key, local)
                total += host.numel() * host.element_size()
                self.offloaded[(param, key)] = local.device
                if is_dtensor:
                    value = copy.copy(value)
                    value._local_tensor = host
                    state[key] = value
                else:
                    state[key] = host
        # The host copies must be complete before anything reads them (torch.save, the next step).
        for device in devices:
            _synchronize(device)
        if total:
            self.offloaded_bytes = total

    def restore(self):
        for param, state in self.optimizer.state.items():
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

    def _pre_step(self, optimizer, args, kwargs):
        self.restore()

    def _post_step(self, optimizer, args, kwargs):
        self.steps += 1
        self.offload()


def offload_optimizer_state(optimizer, pin_memory = True):
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
    if not hasattr(optimizer, "register_step_pre_hook"):
        return None

    offloader = _StateOffloader(optimizer, pin_memory = pin_memory)
    optimizer.register_step_pre_hook(offloader._pre_step)
    optimizer.register_step_post_hook(offloader._post_step)

    # Instance override, not register_load_state_dict_post_hook: bitsandbytes replaces
    # Optimizer.load_state_dict and does not run torch's post hooks. Loading moves every
    # state tensor to its parameter's device, so offload again right after.
    load_state_dict = optimizer.load_state_dict

    def _load_state_dict_and_offload(state_dict, *args, **kwargs):
        result = load_state_dict(state_dict, *args, **kwargs)
        offloader.offload()
        return result

    optimizer.load_state_dict = _load_state_dict_and_offload
    optimizer._unsloth_state_offloader = offloader
    # A state that already exists (optimizer passed in after some steps) goes to host now.
    offloader.offload()
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

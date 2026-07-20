# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Faster safetensors weight loading on unified-memory (integrated) GPUs.

A direct ``safe_open(..., device=<cuda>)`` on CUDA/HIP UMA GPUs (AMD APUs,
NVIDIA GB10 Spark) misses torch's fast pinned-DMA path: the mmap-backed
safetensors buffers aren't recognized, so it falls to a slow per-tensor copy
with page faults. Cloning each tensor into a normal torch CPU allocation before
moving it restores the fast path; outputs are bit-identical.

CUDA/HIP only, and only for loads that pass a CUDA device to ``safe_open``
directly: Intel XPU iGPUs and the CPU-open + later ``.to()`` flows (e.g. bnb /
HQQ quantized loads) keep the stock path until they can be validated on real
hardware.
"""

import os
import functools

import torch

__all__ = [
    "is_integrated_unified_memory_gpu",
    "patch_unified_memory_safetensors_load",
]


@functools.lru_cache(maxsize = None)
def is_integrated_unified_memory_gpu():
    """True only when EVERY visible CUDA/HIP device is integrated (UMA).

    Discrete and mixed discrete+iGPU boxes return False (pinned-DMA already
    works there). Test override: ``UNSLOTH_FORCE_UMA=1`` / ``=0``.
    """
    _force = os.environ.get("UNSLOTH_FORCE_UMA")
    if _force == "1":
        return True
    if _force == "0":
        return False
    try:
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            return False
        count = torch.cuda.device_count()
        if count == 0:
            return False
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            if not getattr(props, "is_integrated", 0):
                return False
        return True
    except Exception:
        return False


def _is_cuda_target(device):
    """Does a ``safe_open`` ``device=`` arg name a CUDA/HIP device?"""
    if isinstance(device, bool):
        return False
    if isinstance(device, int):
        return True
    if isinstance(device, str):
        return device == "cuda" or device.startswith("cuda:")
    try:
        return isinstance(device, torch.device) and device.type == "cuda"
    except Exception:
        return False


def patch_unified_memory_safetensors_load():
    """Wrap ``transformers.modeling_utils.safe_open`` so CUDA-target shard loads
    open on CPU then clone+``.to(device)``, restoring the UMA fast path.

    Gated to integrated GPUs (no-op on discrete/CPU/XPU/MLX), ``framework="pt"``
    CUDA targets only, idempotent. Opt out: ``UNSLOTH_DISABLE_UMA_CLONE_LOAD=1``.

    The gate runs lazily inside the wrapper, never here: probing device
    properties at install would init CUDA during ``import unsloth`` -- breaking
    fork multiprocessing and preempting ``patch_dgx_spark_memory_config``'s
    allocator config. Returns ``True`` if the wrapper was installed.
    """
    if os.environ.get("UNSLOTH_DISABLE_UMA_CLONE_LOAD") == "1":
        return False
    try:
        from transformers import modeling_utils as _mu
    except Exception:
        return False
    real_safe_open = getattr(_mu, "safe_open", None)
    if real_safe_open is None:
        return False
    if getattr(real_safe_open, "_unsloth_uma_clone", False):
        return True

    def _clone_move(tensor, device):
        # Clone into a regular CPU allocation to restore fast pinned-DMA, then
        # move. The clone transiently doubles the tensor's CPU footprint and can
        # OOM a low-memory UMA box; fall back to the direct, allocation-free move
        # (a genuine non-memory error re-raises identically from it).
        try:
            return tensor.clone().to(device, non_blocking = False)
        except (MemoryError, RuntimeError):
            return tensor.to(device, non_blocking = False)

    class _ClonedSlice:
        """Proxy over a safetensors ``PySafeSlice`` that clones+moves on read."""

        __slots__ = ("_real", "_device")

        def __init__(self, real, device):
            self._real = real
            self._device = device

        def __getattr__(self, name):
            if name in ("_real", "_device"):
                raise AttributeError(name)
            return getattr(self._real, name)

        def __getitem__(self, key):
            return _clone_move(self._real[key], self._device)

    class _ClonedSafeOpen:
        """Safetensors-handle proxy: load on CPU, clone+move tensors to CUDA."""

        __slots__ = ("_real", "_device")

        def __init__(self, args, kwargs):
            self._device = kwargs.get("device", args[2] if len(args) > 2 else "cpu")
            # Open on CPU; move ourselves.
            if len(args) > 2:
                args = args[:2] + ("cpu",) + tuple(args[3:])
            else:
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            self._real = real_safe_open(*args, **kwargs)

        def __enter__(self):
            self._real.__enter__()
            return self

        def __exit__(self, *exc):
            return self._real.__exit__(*exc)

        def __getattr__(self, name):
            if name in ("_real", "_device"):
                raise AttributeError(name)
            return getattr(self._real, name)

        def get_slice(self, name):
            return _ClonedSlice(self._real.get_slice(name), self._device)

        def get_tensor(self, name):
            return _clone_move(self._real.get_tensor(name), self._device)

    @functools.wraps(real_safe_open)
    def _uma_safe_open(*args, **kwargs):
        framework = kwargs.get("framework", args[1] if len(args) > 1 else None)
        device = kwargs.get("device", args[2] if len(args) > 2 else "cpu")
        # Device check first: non-CUDA loads must not trigger the CUDA-init gate.
        if (
            framework in ("pt", "pytorch")
            and _is_cuda_target(device)
            and is_integrated_unified_memory_gpu()
        ):
            return _ClonedSafeOpen(args, kwargs)
        return real_safe_open(*args, **kwargs)

    _uma_safe_open._unsloth_uma_clone = True
    _mu.safe_open = _uma_safe_open
    return True

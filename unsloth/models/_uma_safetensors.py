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

On unified-memory GPUs (AMD APUs / "Strix Halo", NVIDIA GB10 "Spark", Intel
iGPUs) the GPU shares the system memory pool. PyTorch's fast pinned-DMA
host -> device path does not recognize the Rust-allocated, mmap-backed buffers
that ``safetensors`` hands back, so a *direct* safetensors GPU load
(``safe_open(..., device=<cuda>)``) drops onto a slow per-tensor copy that, on
unified memory, additionally triggers page-attribute changes and page faults.

Cloning each tensor into a normal torch CPU allocation first and *then* moving
it to the GPU puts the copy back on the fast path. The data, dtype and final
device are unchanged, so model outputs are bit-identical -- only *how* the bytes
reach the GPU changes.

This module is intentionally dependency-light (``os``, ``functools``, ``torch``)
so it can be imported and unit-tested in isolation.
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
    """True only when every visible CUDA/HIP device is an integrated
    (unified-memory) GPU -- an APU / iGPU / Spark-class part that shares the
    system memory pool.

    Discrete GPUs report ``is_integrated == 0`` so this is a strict ``False``
    there (their pinned-DMA path already works -- no change, no regression). We
    require *all* visible devices to be integrated so a mixed discrete+iGPU box
    is left alone too. Uses the standard device property
    (``cudaDeviceProp.integrated`` / ``hipDeviceProp_t.integrated``). Overridable
    for testing via ``UNSLOTH_FORCE_UMA=1`` (force on) / ``=0`` (force off).
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
    """Whether a ``safe_open`` ``device=`` argument names a CUDA/HIP device."""
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
    """Clone-then-move safetensors weights on unified-memory GPUs.

    Wraps ``transformers.modeling_utils.safe_open`` so that, when transformers
    asks it to load a shard directly onto a CUDA/HIP device, the shard is opened
    on CPU and each tensor is ``.clone()``-d into a normal torch allocation
    before ``.to(device)``. This restores the fast host -> device DMA path that
    the mmap-backed safetensors buffers otherwise miss on unified memory.

    Accuracy-neutral: data, dtype and final device are unchanged, so outputs are
    bit-identical (verified ``max|logit diff| == 0``). Strictly gated to
    integrated/unified-memory GPUs (:func:`is_integrated_unified_memory_gpu`) --
    a hard no-op on discrete NVIDIA/AMD GPUs, CPU, XPU and MLX. Only intercepts
    ``framework="pt"`` CUDA-device targets; CPU / disk-offload loads
    (``device="cpu"``) are left untouched. Idempotent (marked via
    ``_unsloth_uma_clone``); opt out with ``UNSLOTH_DISABLE_UMA_CLONE_LOAD=1``.

    Returns ``True`` if the wrapper was installed, else ``False``.
    """
    if not is_integrated_unified_memory_gpu():
        return False
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
        return True  # already patched (idempotent)

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
            return self._real[key].clone().to(self._device, non_blocking = False)

    class _ClonedSafeOpen:
        """Proxy over a safetensors file handle that loads on CPU and
        clones+moves each tensor to the originally-requested CUDA device."""

        __slots__ = ("_real", "_device")

        def __init__(self, args, kwargs):
            self._device = kwargs.get("device", args[2] if len(args) > 2 else "cpu")
            # Re-target the real open at CPU; we do the device move ourselves.
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
            return self._real.get_tensor(name).clone().to(self._device, non_blocking = False)

    @functools.wraps(real_safe_open)
    def _uma_safe_open(*args, **kwargs):
        framework = kwargs.get("framework", args[1] if len(args) > 1 else None)
        device = kwargs.get("device", args[2] if len(args) > 2 else "cpu")
        if framework in ("pt", "pytorch") and _is_cuda_target(device):
            return _ClonedSafeOpen(args, kwargs)
        return real_safe_open(*args, **kwargs)

    _uma_safe_open._unsloth_uma_clone = True
    _mu.safe_open = _uma_safe_open
    return True

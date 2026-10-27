# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measured-budget memory policy for the local diffusion backend.

From the resolved device target, a free-memory snapshot and a coarse model footprint estimate, this
picks a CPU-offload policy and VAE slice/tile settings, then applies them to a built diffusers
pipeline. A model that will not fit resident is kept running by streaming weights through the GPU
one module at a time, which is lossless (offload / VAE slicing change placement and decode chunking,
not numerics).

The choice is coarse (sizes the model, not every activation), so ``auto`` is best-effort and the
explicit ``fast`` / ``balanced`` / ``low_vram`` modes are a hard override. torch / psutil imported
lazily.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import math
import os
import sys
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Optional
from .host_pinning import host_pinning_allowed
from .rocm_bf16 import is_rocm_torch


MEMORY_MODE_AUTO = "auto"
MEMORY_MODE_FAST = "fast"
MEMORY_MODE_BALANCED = "balanced"
MEMORY_MODE_LOW_VRAM = "low_vram"
MEMORY_MODES = (
    MEMORY_MODE_AUTO,
    MEMORY_MODE_FAST,
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_LOW_VRAM,
)

# Tiers: none, model, group, streaming, sequential (sequential is broken for GGUF
# through diffusers 0.39, kept as an escape hatch).
OFFLOAD_NONE = "none"
OFFLOAD_MODEL = "model"
OFFLOAD_GROUP = "group"
OFFLOAD_STREAMING = "streaming"
OFFLOAD_SEQUENTIAL = "sequential"

# Transformer blocks resident per group under group offloading: fewer = lower VRAM, more host-to-device traffic.
DEFAULT_GROUP_BLOCKS = 1

DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024
# flat allowance for fixed pipeline costs (scheduler, embeddings, CUDA context, fragmentation)
DEFAULT_BASE_OVERHEAD_MIB = 2048
# Reserve kind for an mps reading bounded by Metal's recommended working set (see snapshot_device_memory).
METAL_WORKING_SET = "metal_working_set"


_host_memory_reclaim_warning_logged = False
_host_memory_reclaim_unsupported_logged = False

# Optional (no _ctypes); a module attribute because the reclaimer tests monkeypatch it.
try:
    import ctypes
except Exception:  # noqa: BLE001
    ctypes = None  # type: ignore[assignment]


@functools.lru_cache(maxsize = 1)
def _resolve_host_memory_reclaimer() -> Optional[Callable[[], None]]:
    """Resolve this process allocator's native pressure API once, if the OS exposes one."""
    if ctypes is None:
        return None
    try:
        if sys.platform.startswith("linux"):
            allocator = ctypes.CDLL(None)
            pressure = allocator.malloc_trim
            pressure.argtypes = [ctypes.c_size_t]
            pressure.restype = ctypes.c_int

            def reclaim() -> None:
                pressure(0)

        elif sys.platform == "darwin":
            allocator = ctypes.CDLL(None)
            pressure = allocator.malloc_zone_pressure_relief
            pressure.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            pressure.restype = ctypes.c_size_t

            def reclaim() -> None:
                pressure(None, 0)

        elif sys.platform == "win32":
            pressure = None
            for library_name in ("ucrtbase.dll", "msvcrt.dll"):
                try:
                    allocator = ctypes.CDLL(library_name)
                    pressure = allocator._heapmin
                    break
                except Exception:
                    continue
            if pressure is None:
                return None
            pressure.argtypes = []
            pressure.restype = ctypes.c_int

            def reclaim() -> None:
                if pressure() == -1:
                    raise OSError("_heapmin failed")

        else:
            return None
    except Exception:
        return None
    return reclaim


OFFLOAD_KEEP_CPU_ENV = "UNSLOTH_DIFFUSION_OFFLOAD_KEEP_CPU"
_KEEP_ATTR = "_unsloth_offload_keep_cpu"
# diffusion_vae_upcast.DTYPE_ONLY_TO_ATTR
_DTYPE_ONLY_TO_ATTR = "_unsloth_dtype_only_to"


def keep_cpu_weights_on_offload(pipe: Any, logger: Any = None) -> int:
    """Offload by re-pointing unmodified weights at their host tensors; enable_model_cpu_offload is wrapped since diffusers rebuilds hooks."""
    if (os.environ.get(OFFLOAD_KEEP_CPU_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return 0
    try:
        from accelerate.hooks import CpuOffload
    except Exception:  # noqa: BLE001 - no accelerate, no offload hooks to wrap
        return 0
    enable = getattr(pipe, "enable_model_cpu_offload", None)
    if callable(enable) and not getattr(enable, _KEEP_ATTR, False):

        def _enable(*args: Any, **kwargs: Any) -> Any:
            # The re-enable after every call starts with pipe.to("cpu"), which would copy back the module run last.
            _offload_through_kept_hooks(pipe, CpuOffload, logger)
            with _keep_device_cache(pipe):
                out = enable(*args, **kwargs)
            _wrap_cpu_offload_hooks(pipe, CpuOffload, logger)
            return out

        setattr(_enable, _KEEP_ATTR, True)
        try:
            pipe.enable_model_cpu_offload = _enable
        except Exception:  # noqa: BLE001 - a pipeline refusing the attribute keeps the stock hooks
            return 0
    return _wrap_cpu_offload_hooks(pipe, CpuOffload, logger)


def _offload_through_kept_hooks(
    pipe: Any,
    hook_cls: type,
    logger: Any = None,
) -> None:
    components = getattr(pipe, "components", None) or {}
    for module in components.values():
        hook = getattr(module, "_hf_hook", None)
        if not isinstance(hook, hook_cls) or not getattr(hook, _KEEP_ATTR, False):
            continue
        try:
            onloaded = [*module.parameters(), *(b for _, b in _named_buffers(module))]
            if any(t.device.type != "cpu" for t in onloaded):
                hook.init_hook(module)
        except Exception as exc:  # noqa: BLE001 - the stock re-enable still moves it
            if logger is not None:
                logger.debug(
                    "diffusion.memory: kept offload of %s before re-enable failed: %s",
                    type(module).__name__,
                    exc,
                )


def _gguf_parameter_class() -> Optional[type]:
    module = sys.modules.get("diffusers.quantizers.gguf.utils")
    return getattr(module, "GGUFParameter", None) if module is not None else None


def _wrapper_inner(t: Any) -> Optional[list]:
    """``[(attr, tensor), ...]`` of a traceable wrapper subclass (torchao int8 / fp8 weights) whose inner tensors are
    all plain tensors, else None."""
    import torch

    flatten = getattr(type(t), "__tensor_flatten__", None)
    if flatten is None or type(t) in (torch.Tensor, torch.nn.Parameter):
        return None
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if not is_traceable_wrapper_subclass(t):
            return None
        names, _ctx = t.__tensor_flatten__()
        inner = [(name, getattr(t, name)) for name in names]
    except Exception:  # noqa: BLE001
        return None
    if not inner or any(type(x) is not torch.Tensor for _, x in inner):
        return None
    return inner


def _tensor_ptr(t: Any) -> Any:
    """Storage address of ``t``; for a wrapper subclass, the tuple of its inner tensors' addresses."""
    inner = _wrapper_inner(t)
    if inner is not None:
        return tuple(int(x.data_ptr()) for _, x in inner)
    return t.data_ptr()


def _repoint(param: Any, kept: Any) -> None:
    """Point ``param`` at its kept host tensor without a copy. A wrapper subclass cannot change device through
    ``.data``; it swaps with a parameter over the kept tensors, as ``Module.to`` moves it."""
    if _wrapper_inner(kept) is None:
        param.data = kept
        return
    import torch

    torch.utils.swap_tensors(param, torch.nn.Parameter(kept, requires_grad = param.requires_grad))


def _keepable(param: Any) -> bool:
    import torch

    data = param.data
    if type(data) is torch.Tensor:
        return True
    # torchao weights (int8 / fp8): a wrapper of plain tensors; kept and re-pointed like a dense weight.
    if _wrapper_inner(data) is not None and type(data) is type(param):
        return True
    # GGUF weights are packed bytes; the quant type lives on the Parameter, which the swap never replaces.
    gguf = _gguf_parameter_class()
    return gguf is not None and type(param) is gguf and type(data) is gguf


def _wrap_cpu_offload_hooks(
    pipe: Any,
    hook_cls: type,
    logger: Any = None,
) -> int:
    wrapped = 0
    components = getattr(pipe, "components", None) or {}
    for module in components.values():
        hook = getattr(module, "_hf_hook", None)
        if not isinstance(hook, hook_cls) or getattr(hook, _KEEP_ATTR, False):
            continue
        try:
            _wrap_cpu_offload_hook(hook, module, logger)
            module.__dict__[_KEEP_ATTR]["exact_pin"] = bool(
                getattr(pipe, _FAST_PIN_REQUEST_ATTR, False)
            )
            module.__dict__[_KEEP_ATTR]["exact_pin_device"] = getattr(
                hook, "execution_device", None
            )
            wrapped += 1
        except Exception as exc:  # noqa: BLE001 - that module keeps the stock copy
            if logger is not None:
                logger.debug(
                    "diffusion.memory: keep-cpu offload not applied to %s: %s",
                    type(module).__name__,
                    exc,
                )
    return wrapped


OFFLOAD_PIN_ENV = "UNSLOTH_DIFFUSION_OFFLOAD_PIN"
# The host allocator rounds every block up to a power of two, so weights share chunks instead of one pin each.
_PIN_CHUNK_BYTES = 1 << 30
_PIN_ALIGN = 256
_PIN_RESERVE_MIN_BYTES = 4 << 30
_PIN_RESERVE_FRACTION = 0.15


def _pin_mode() -> str:
    raw = (os.environ.get(OFFLOAD_PIN_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "on"
    return "auto"


def _pow2_ceil(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def release_pinned_host_memory() -> None:
    _host_empty_cache()


def _host_empty_cache() -> None:
    try:
        import torch
        empty = getattr(torch._C, "_host_emptyCache", None)
        if callable(empty) and torch.cuda.is_available():
            empty()
    except Exception:  # noqa: BLE001
        pass


def _pinnable_layout(data: Any) -> bool:
    if data.is_contiguous():
        return True
    import torch

    fmt = {4: torch.channels_last, 5: torch.channels_last_3d}.get(data.dim())
    return fmt is not None and data.is_contiguous(memory_format = fmt)


def _pin_host_weights(
    module: Any,
    host: dict,
    logger: Any = None,
    buffer_host: Optional[dict] = None,
    layout: Optional[list] = None,
    exact: bool = False,
    device: Any = None,
) -> int:
    """Pack kept host weights and buffers into page-locked chunks; returns bytes pinned. Each copy keeps its source strides.
    ``layout`` (if given) receives ``(chunk, [(name, start, view), ...])`` per chunk for ``_upload_pinned_chunks``.
    ``exact`` (video loads, see ``_exact_pin_supported``) sizes each chunk to its bytes instead of a power of two."""
    import torch

    mode = _pin_mode()
    if mode == "off" or not torch.cuda.is_available() or not host_pinning_allowed(device):
        return 0
    # (name, owner tensor, inner attribute or None, the tensor to pin)
    params: list = []
    for name, p in module.named_parameters():
        if host.get(name) is None or p.device.type != "cpu" or not _keepable(p):
            continue
        inner = _wrapper_inner(p.data)
        if inner is None:
            if _pinnable_layout(p.data) and not p.data.is_pinned():
                params.append((name, p, None, p.data))
            continue
        for attr, t in inner:
            if t.device.type == "cpu" and _pinnable_layout(t) and not t.is_pinned():
                params.append((name, p, attr, t))
    buffers = [
        (name, b)
        for name, b in _named_buffers(module)
        if (buffer_host or {}).get(name) is b
        and b.device.type == "cpu"
        and _keepable_buffer(b)
        and b.is_contiguous()
        and not b.is_pinned()
    ]
    params = params + [("\0" + name, b, None, b) for name, b in buffers]
    sizes = [-(-t.nbytes // _PIN_ALIGN) * _PIN_ALIGN for _, _, _, t in params]
    if not sizes or sum(sizes) == 0:
        return 0
    # Lay out chunks first so the RAM gate counts chunk tails and last-chunk rounding, not just weight bytes.
    chunk = max(_PIN_CHUNK_BYTES, _pow2_ceil(max(sizes)))
    chunks: list = []
    slots: list = []
    off, left = 0, sum(sizes)
    for size in sizes:
        if not chunks or off + size > chunks[-1]:
            chunks.append(chunk if left >= chunk else _pow2_ceil(left))
            off = 0
        slots.append((len(chunks) - 1, off))
        off += size
        left -= size
    exact = exact and _exact_pin_supported(device)
    if exact:
        # The same slots, so the device copies are unchanged; only the host side loses the rounding.
        used = [0] * len(chunks)
        for size, (index, start) in zip(sizes, slots):
            used[index] = max(used[index], start + size)
        chunks = [-(-n // _EXACT_PIN_PAGE) * _EXACT_PIN_PAGE for n in used]
    need = sum(chunks)
    if mode == "auto":
        # Pinned pages are charged to an enforcing cgroup: size from the cgroup-capped readings.
        available_mib = _available_system_memory_mib()
        total_mib = _host_ram_capacity_mib()
        if available_mib is None or total_mib is None:
            return 0  # no way to size it, so do not lock memory
        available, total = int(available_mib) << 20, int(total_mib) << 20
        reserve = max(_PIN_RESERVE_MIN_BYTES, int(total * _PIN_RESERVE_FRACTION))
        if available - need < reserve:
            if logger is not None:
                logger.info(
                    "diffusion.memory: not pinning %s (%.1f GiB): %.1f GiB of host RAM available",
                    type(module).__name__,
                    need / 2**30,
                    available / 2**30,
                )
            return 0
    if exact:
        try:
            return _pin_host_weights_exact(
                module, host, logger, buffer_host, layout, params, slots, chunks, device
            )
        except Exception as exc:  # noqa: BLE001 - registration refused before any weight moved: the pinned allocator
            if logger is not None:
                logger.info(
                    "diffusion.memory: exact pin of %s failed (%s); using the pinned allocator",
                    type(module).__name__,
                    exc,
                )
            return _pin_host_weights(
                module, host, logger, buffer_host = buffer_host, layout = layout, device = device
            )
    bufs: list = []
    placed: list = []
    try:
        for (name, p, attr, src), (index, start) in zip(params, slots):
            if index == len(bufs):
                bufs.append(torch.empty(chunks[index], dtype = torch.uint8, pin_memory = True))
            view = bufs[index][start : start + src.nbytes].view(src.dtype)
            view = (
                view.view(src.shape)
                if src.is_contiguous()
                else view.as_strided(src.shape, src.stride())
            )
            view.copy_(src)
            placed.append((name, p, attr, view, index, start))
    except Exception as exc:  # noqa: BLE001 - e.g. a WSL pinned-memory cap: keep the pageable weights
        bufs.clear()
        placed.clear()
        view = None
        _host_empty_cache()
        if logger is not None:
            logger.info(
                "diffusion.memory: pinning %s failed (%s); weights stay pageable",
                type(module).__name__,
                exc,
            )
        return 0
    members: list = [[] for _ in bufs]
    wrapped: dict = {}
    for name, p, attr, view, index, start in placed:
        # A torchao wrapper weight gets one entry per inner tensor, under the weight's name.
        members[index].append((name, start, view))
        if name.startswith("\0"):
            name = name[1:]
            _set_buffer(module, name, view)
            buffer_host[name] = view
            continue
        if attr is not None:
            setattr(p, attr, view)
            wrapped[name] = p
            continue
        p.data = view
        host[name] = view
    for name, p in wrapped.items():
        host[name] = p.data  # the kept host weight now reads the pinned inner tensors
    if layout is not None:
        layout[:] = [(buf, entries) for buf, entries in zip(bufs, members) if entries]
    return need


# Video loads only. "0" restores the power-of-two pinned chunks of the whole-module offload tier.
EXACT_PIN_ENV = "UNSLOTH_VIDEO_EXACT_PIN"
_EXACT_PIN_PAGE = 4096


def _exact_pin_supported(device: Any = None) -> bool:
    if (os.environ.get(EXACT_PIN_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    return _fast_pin_supported(device)


def _pin_host_weights_exact(
    module: Any,
    host: dict,
    logger: Any,
    buffer_host: Optional[dict],
    layout: Optional[list],
    params: list,
    slots: list,
    chunks: list,
    device: Any = None,
) -> int:
    """``_pin_host_weights`` into page-granular registered chunks (torch's allocator rounds up to a power of two).
    Weights move per chunk so the module is never held twice; a refusal after the first chunk keeps the moved
    weights pinned and leaves the rest pageable (a first-chunk refusal raises)."""
    import torch

    target = torch.device(device) if device is not None else None
    scope = (
        torch.cuda.device(target)
        if target is not None and target.type == "cuda" and target.index is not None
        else contextlib.nullcontext()
    )
    with scope:
        return _fill_exact_chunks(module, host, logger, buffer_host, layout, params, slots, chunks)


def _fill_exact_chunks(
    module: Any,
    host: dict,
    logger: Any,
    buffer_host: Optional[dict],
    layout: Optional[list],
    params: list,
    slots: list,
    chunks: list,
) -> int:
    import numpy as np
    import torch

    members: list = [[] for _ in chunks]
    for item, (index, start) in zip(range(len(params)), slots):
        members[index].append((item, start))
    done: list = []
    pinned = 0
    for index, size in enumerate(chunks):
        try:
            holder = _RegisteredHostBuffer(size)
            buf = torch.from_numpy(np.asarray(holder))
            placed = []
            for item, start in members[index]:
                name, p, attr, src = params[item]
                view = buf[start : start + src.nbytes].view(src.dtype)
                view = (
                    view.view(src.shape)
                    if src.is_contiguous()
                    else view.as_strided(src.shape, src.stride())
                )
                view.copy_(src)
                placed.append((item, start, view))
            holder.register()
        except Exception:
            if not done:
                raise
            if logger is not None:
                logger.info(
                    "diffusion.memory: pinned %d of %d chunks of %s; the rest stays pageable",
                    len(done),
                    len(chunks),
                    type(module).__name__,
                )
            break
        entries = []
        wrapped: dict = {}
        for item, start, view in placed:
            name, p, attr, _src = params[item]
            params[item] = None
            entries.append((name, start, view))
            if name.startswith("\0"):
                name = name[1:]
                _set_buffer(module, name, view)
                buffer_host[name] = view
                continue
            if attr is not None:
                setattr(p, attr, view)
                wrapped[name] = p
                continue
            p.data = view
            host[name] = view
        for name, p in wrapped.items():
            host[name] = p.data  # the kept host weight now reads the pinned inner tensors
        done.append((buf, entries))
        pinned += size
    if layout is not None:
        layout[:] = [(buf, entries) for buf, entries in done if entries]
    return pinned


OFFLOAD_FAST_UPLOAD_ENV = "UNSLOTH_DIFFUSION_OFFLOAD_FAST_UPLOAD"


def _on_device(module: Any) -> bool:
    first = next(iter(module.parameters()), None)
    return first is not None and first.device.type != "cpu"


@contextlib.contextmanager
def _keep_device_cache(pipe: Any):
    """Skip the empty_cache inside diffusers' enable_model_cpu_offload (re-run after every call) once the
    components onload through pinned chunks: the chunks freed by the last offload are reused by the next onload,
    while an emptied cache makes every onload cudaMalloc again and every offload cudaFree."""
    module = sys.modules.get("diffusers.pipelines.pipeline_utils")
    original = getattr(module, "empty_device_cache", None) if module is not None else None
    components = getattr(pipe, "components", None) or {}
    engaged = _fast_upload_enabled() and any(
        bool((getattr(m, "__dict__", {}).get(_KEEP_ATTR) or {}).get("pin_layout"))
        for m in components.values()
    )
    if not engaged or not callable(original):
        yield
        return
    # Process-global patch: overlapping enables share one count, and the last one out restores the real function.
    with _cache_patch_lock:
        if not _cache_patch:
            _cache_patch.update(module = module, original = original, depth = 0)
            module.empty_device_cache = _skip_empty_device_cache
        _cache_patch["depth"] += 1
    try:
        yield
    finally:
        with _cache_patch_lock:
            _cache_patch["depth"] -= 1
            if _cache_patch["depth"] <= 0:
                if _cache_patch["module"].empty_device_cache is _skip_empty_device_cache:
                    _cache_patch["module"].empty_device_cache = _cache_patch["original"]
                _cache_patch.clear()


_cache_patch_lock = threading.Lock()
_cache_patch: dict = {}


def _skip_empty_device_cache(*args: Any, **kwargs: Any) -> None:
    return None


def _fast_upload_enabled() -> bool:
    return (os.environ.get(OFFLOAD_FAST_UPLOAD_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _same_view(tensor: Any, view: Any) -> bool:
    return (
        tensor.device.type == "cpu"
        and tensor.data_ptr() == view.data_ptr()
        and tensor.dtype == view.dtype
        and tensor.shape == view.shape
        and tensor.stride() == view.stride()
    )


def _prune_pin_layout(layout: list, host: dict, buffer_host: dict) -> None:
    """Drop, in place, the layout entries whose kept host weight is no longer their pinned view, and the chunks left
    with none: the layout's own references would otherwise keep those pinned chunks alive."""
    kept = []
    for buf, entries in layout:
        live = []
        for entry in entries:
            name, view = entry[0], entry[2]
            tensor = buffer_host.get(name[1:]) if name.startswith("\0") else host.get(name)
            if tensor is None:
                continue
            # A torchao wrapper weight has one entry per inner tensor (as ``_upload_pinned_chunks`` reads them).
            inner = None if name.startswith("\0") else _wrapper_inner(tensor)
            if inner is not None:
                if any(_same_view(t, view) for _, t in inner):
                    live.append(entry)
            elif _same_view(tensor, view):
                live.append(entry)
        if live:
            kept.append((buf, live))
    layout[:] = kept


def _prune_state_layout(state: dict) -> None:
    layout = state.get("pin_layout")
    if layout:
        _prune_pin_layout(layout, state.get("host") or {}, state.get("buffer_host") or {})
        if not layout:
            state.pop("pin_layout", None)


def _upload_pinned_chunks(
    module: Any, layout: list, host: dict, owner: dict, buffer_host: dict, device: Any
) -> int:
    """Onload a module's pinned weights chunk by chunk: one device allocation and one non_blocking copy per pinned
    chunk, then every weight is re-pointed at its slice. The stock ``module.to`` allocates every tensor on its own,
    and with the device cache emptied between components each of those was a cudaMalloc (and a cudaFree on the way
    out): under host load that ran at 57-120 MiB/s against a 50 GiB/s link. Chunks are near-uniform, so the next
    component reuses the previous one's cached chunks. Same bytes, same strides, so outputs are bit-identical.
    Weights no longer at their pinned host view (replaced, reassigned) are left for the stock copy. A torchao
    wrapper weight (int8 / fp8) is rebuilt over its inner tensors' device slices with ``__tensor_unflatten__`` and
    swapped in, as the stock ``_to_copy`` + ``Module.to`` do. Returns bytes."""
    import torch

    params = dict(module.named_parameters())
    buffers = dict(_named_buffers(module))
    copied = 0
    moved: list = []
    wrapped: dict = {}
    for name, p in params.items():
        kept = host.get(name)
        inner = _wrapper_inner(kept) if kept is not None else None
        if (
            inner is None
            or owner.get(name) is not p
            or p.device.type != "cpu"
            or type(p) is not type(kept)
            or _tensor_ptr(p.data) != _tensor_ptr(kept)
        ):
            continue
        wrapped[name] = dict(inner)
    # (name, inner attr) -> device slice once uploaded
    inner_dev: dict = {}

    def _inner_attr(name: str, view: Any) -> Optional[str]:
        for attr, t in (wrapped.get(name) or {}).items():
            if _same_view(t, view):
                return attr
        return None

    for _buf, entries in layout:
        for name, _start, view in entries:
            attr = _inner_attr(name, view)
            if attr is not None:
                inner_dev[(name, attr)] = None
    # Only a wrapper with every inner tensor in the layout is rebuilt; the rest stay for the stock module.to.
    whole = {
        name for name, inner in wrapped.items() if all((name, attr) in inner_dev for attr in inner)
    }
    try:
        for buf, entries in layout:
            live = []
            for name, start, view in entries:
                # By address, not identity: ``p.data`` and the re-captured host map hand out fresh tensor objects.
                if name.startswith("\0"):
                    leaf = name[1:]
                    b = buffers.get(leaf)
                    if b is not None and buffer_host.get(leaf) is b and _same_view(b, view):
                        live.append((name, start, view))
                    continue
                if name in wrapped:
                    if name in whole and _inner_attr(name, view) is not None:
                        live.append((name, start, view))
                    continue
                p = params.get(name)
                kept = host.get(name)
                if (
                    p is not None
                    and owner.get(name) is p
                    and kept is not None
                    and _wrapper_inner(kept) is None
                    and _same_view(kept, view)
                    and _same_view(p.data, view)
                ):
                    live.append((name, start, view))
            if not live:
                continue
            # Pack live weights so skipped ones leave no allocated holes.
            used, slots = 0, []
            for name, start, view in sorted(live, key = lambda e: e[1]):
                used = -(-used // _PIN_ALIGN) * _PIN_ALIGN
                slots.append((name, start, view, used))
                used += view.nbytes
            dense = all(start == at for _, start, _, at in slots)
            # Same stream as the forward that follows, so no sync: kernels queue behind the copy.
            dev_buf = torch.empty(used, dtype = torch.uint8, device = device)
            if not torch._has_compatible_shallow_copy_type(buf, dev_buf):
                # e.g. meta: the stock move hands back new Parameters there instead of re-pointing them.
                raise TypeError("device tensors cannot back host parameters")
            if dense:
                dev_buf.copy_(buf[:used], non_blocking = True)
            else:
                for _, start, view, at in slots:
                    dev_buf[at : at + view.nbytes].copy_(
                        buf[start : start + view.nbytes], non_blocking = True
                    )
            copied += used
            for name, _, view, at in slots:
                dev = dev_buf[at : at + view.nbytes].view(view.dtype)
                dev = (
                    dev.view(view.shape)
                    if view.is_contiguous()
                    else dev.as_strided(view.shape, view.stride())
                )
                if name.startswith("\0"):
                    _set_buffer(module, name[1:], dev)
                elif name in wrapped:
                    inner_dev[(name, _inner_attr(name, view))] = dev
                    continue
                else:
                    params[name].data = dev
                moved.append((name, view))
        for name in sorted(whole):
            p, kept = params[name], host[name]
            names, ctx = kept.__tensor_flatten__()
            rebuilt = type(kept).__tensor_unflatten__(
                {attr: inner_dev[(name, attr)] for attr in names}, ctx, kept.shape, kept.stride()
            )
            torch.utils.swap_tensors(p, torch.nn.Parameter(rebuilt, requires_grad = p.requires_grad))
            moved.append((name, kept))
    except Exception:
        # Put every weight back on its host view; the stock per-tensor copy then runs (and raises a real OOM itself).
        for name, view in moved:
            try:
                if name.startswith("\0"):
                    _set_buffer(module, name[1:], view)
                elif name in wrapped:
                    _repoint(params[name], view)
                else:
                    params[name].data = view
            except Exception:  # noqa: BLE001
                pass
        return 0
    return copied


def _named_buffers(module: Any) -> list:
    named = getattr(module, "named_buffers", None)
    return list(named()) if callable(named) else []


def _keepable_buffer(buffer: Any) -> bool:
    import torch
    return type(buffer) is torch.Tensor


def _buffer_version(buffer: Any) -> Optional[int]:
    """``_version``, or None for an inference tensor (moved under inference_mode), which has no version counter."""
    try:
        return None if buffer.is_inference() else int(buffer._version)
    except Exception:  # noqa: BLE001 - untracked: the stock copy handles it
        return None


def _set_buffer(module: Any, name: str, tensor: Any) -> None:
    prefix, _, leaf = name.rpartition(".")
    owner = module.get_submodule(prefix) if prefix else module
    owner._buffers[leaf] = tensor


def _wrap_cpu_offload_hook(
    hook: Any,
    module: Any,
    logger: Any = None,
) -> None:
    # Keyed by name: a move can hand the module new Parameter objects, and a stale key must miss, not alias.
    state = module.__dict__.get(_KEEP_ATTR)
    if state is None:
        state = {"host": {}, "owner": {}, "version": {}}
        module.__dict__[_KEEP_ATTR] = state
    state.setdefault("owner", {})
    host, owner, version = state["host"], state["owner"], state["version"]
    # Plain buffers too: the stock offload re-copied them every call.
    buffer_host = state.setdefault("buffer_host", {})
    buffer_version = state.setdefault("buffer_version", {})

    def _capture(mod: Any) -> None:
        host.clear()
        owner.clear()
        buffer_host.clear()
        for name, p in mod.named_parameters():
            if p.device.type == "cpu" and _keepable(p):
                host[name] = p.data
                owner[name] = p
        for name, b in _named_buffers(mod):
            if b.device.type == "cpu" and _keepable_buffer(b):
                buffer_host[name] = b

    _capture(module)
    version.clear()
    buffer_version.clear()
    init_hook, pre_forward = hook.init_hook, hook.pre_forward

    def _init_hook(mod: Any) -> Any:
        for name, b in _named_buffers(mod):
            kept = buffer_host.get(name)
            seen = buffer_version.get(name)
            if (
                kept is None
                or b.device.type == "cpu"
                or seen is None
                or seen[0] is not b
                or seen[1] != _buffer_version(b)
                or seen[2] != b.data_ptr()
            ):
                continue
            try:
                _set_buffer(mod, name, kept)
            except Exception:  # noqa: BLE001 - the stock copy below handles it
                pass
        for name, p in mod.named_parameters():
            kept = host.get(name)
            seen = version.get(name)
            # Identity and data_ptr too: a replacement can match the version, and `p.data = ...` does not bump it.
            if (
                kept is None
                or p.device.type == "cpu"
                or seen is None
                or seen[0] is not p
                or seen[1] != p._version
                or seen[2] != _tensor_ptr(p)
            ):
                continue
            try:
                _repoint(p, kept)
            except Exception:  # noqa: BLE001 - incompatible tensor types: the stock copy below handles it
                pass
        out = init_hook(mod)
        _capture(mod)
        # a weight cast or replaced on the device comes back as a new host tensor: release its pinned chunk now
        _prune_state_layout(state)
        version.clear()
        buffer_version.clear()
        return out

    def _pre_forward(mod: Any, *args: Any, **kwargs: Any) -> Any:
        # `host` gate: an all-subclass module (torchao) never fills `version`, so would rescan every forward.
        onload = not version and not buffer_version and bool(host or buffer_host)
        if onload:
            for name, p in mod.named_parameters():
                kept = host.get(name)
                if kept is not None and (
                    owner.get(name) is not p
                    or p.device.type != "cpu"
                    or _tensor_ptr(p.data) != _tensor_ptr(kept)
                ):
                    host.pop(name, None)
                    owner.pop(name, None)
            current = dict(_named_buffers(mod))
            for name in list(buffer_host):
                if current.get(name) is not buffer_host[name]:
                    buffer_host.pop(name, None)
            # changed while offloaded (a dtype cast, a replacement): its pinned chunk is not kept for nothing
            _prune_state_layout(state)
        if onload and not state.get("pin_tried"):
            # Once per module, on its first onload, so loading pays nothing.
            state["pin_tried"] = True
            layout: list = []
            try:
                pinned = _pin_host_weights(
                    mod,
                    host,
                    logger,
                    buffer_host = buffer_host,
                    layout = layout,
                    exact = bool(state.get("exact_pin")),
                    device = state.get("exact_pin_device"),
                )
            except Exception:  # noqa: BLE001 - pinning is an optimisation, never a failure
                pinned = 0
            if pinned:
                state["pin_layout"] = layout
            if pinned and logger is not None:
                logger.info(
                    "diffusion.memory: pinned %.1f GiB of %s host weights",
                    pinned / 2**30,
                    type(mod).__name__,
                )
        layout = state.get("pin_layout") if onload and _fast_upload_enabled() else None
        prior_to = mod.__dict__.get("to")
        if layout and prior_to is not None and not getattr(prior_to, _DTYPE_ONLY_TO_ATTR, False):
            # An instance-level mover is already installed (stable arena, pinned swap): defer to it. A dtype-only
            # one (diffusion_vae_upcast) passes device moves through, so the pinned chunks still go up first.
            layout = None
        if not layout:
            out = pre_forward(mod, *args, **kwargs)
        else:
            # Offload the previous component through its kept hook here, so the stock pre_forward finds it on the
            # host and skips its empty_cache: the freed chunks stay in the caching allocator for this onload.
            prev = getattr(hook, "prev_module_hook", None)
            prev_model = getattr(prev, "model", None)
            if (
                prev_model is not None
                and getattr(getattr(prev, "hook", None), _KEEP_ATTR, False)
                and callable(getattr(prev, "offload", None))
                and _on_device(prev_model)
            ):
                prev.offload()

            # The stock pre_forward offloads the previous component, then calls module.to(device) only if the module
            # is not there yet. Shadow that one call so the pinned chunks go up first; module.to then moves whatever
            # the chunks did not cover (unpinned or replaced weights) exactly as before.
            def _put_back_to() -> None:
                if prior_to is None:
                    mod.__dict__.pop("to", None)
                else:
                    mod.__dict__["to"] = prior_to

            def _to(*to_args: Any, **to_kwargs: Any) -> Any:
                _put_back_to()
                import torch

                target = to_args[0] if to_args else to_kwargs.get("device")
                try:
                    target = torch.device(target)
                except Exception:  # noqa: BLE001 - not a plain device move: stock only
                    target = None
                if (
                    target is not None
                    and target.type != "cpu"
                    and len(to_args) + len(to_kwargs) == 1
                ):
                    _upload_pinned_chunks(mod, layout, host, owner, buffer_host, target)
                return mod.to(*to_args, **to_kwargs)

            mod.__dict__["to"] = _to
            try:
                out = pre_forward(mod, *args, **kwargs)
            finally:
                _put_back_to()
        if onload:
            for name, p in mod.named_parameters():
                if name in host and owner.get(name) is p and p.device.type != "cpu":
                    version[name] = (p, p._version, _tensor_ptr(p))
            for name, b in _named_buffers(mod):
                if (
                    name in buffer_host
                    and b.device.type != "cpu"
                    and _buffer_version(b) is not None
                ):
                    buffer_version[name] = (b, _buffer_version(b), b.data_ptr())
        return out

    try:
        import torch

        # Same as accelerate's own pre_forward: a compiled forward must not trace the hook.
        _init_hook, _pre_forward = (
            torch.compiler.disable(_init_hook),
            torch.compiler.disable(_pre_forward),
        )
    except Exception:  # noqa: BLE001 - older torch: the hook runs outside any compiled region anyway
        pass
    hook.init_hook, hook.pre_forward = _init_hook, _pre_forward
    setattr(hook, _KEEP_ATTR, True)


def reclaim_offload_host_memory(offload_policy: str, logger: Any = None) -> bool:
    """Return unused allocator pages after whole-model CPU offload, without touching live
    tensors, Python GC, or device caches. Unsupported allocators and failures are non-fatal."""
    if offload_policy != OFFLOAD_MODEL:
        return False
    return reclaim_host_memory(logger = logger)


def reclaim_host_memory(logger: Any = None) -> bool:
    """Return freed allocator pages to the OS regardless of offload policy (unload / teardown).
    Best effort: unsupported allocators and failures return False."""
    global _host_memory_reclaim_warning_logged
    global _host_memory_reclaim_unsupported_logged
    try:
        reclaim = _resolve_host_memory_reclaimer()
        if reclaim is None:
            # The call site discards the result, so a permanent no-op is otherwise invisible.
            if logger is not None and not _host_memory_reclaim_unsupported_logged:
                _host_memory_reclaim_unsupported_logged = True
                try:
                    logger.info(
                        "diffusion.memory: no host allocator pressure API on this platform "
                        "(%s); freed host pages will not be returned early",
                        sys.platform,
                    )
                except Exception:  # noqa: BLE001
                    pass
            return False
        reclaim()
        _host_empty_cache()
        return True
    except Exception as exc:  # noqa: BLE001
        if logger is not None and not _host_memory_reclaim_warning_logged:
            _host_memory_reclaim_warning_logged = True
            try:
                logger.warning("diffusion.memory: host allocator reclamation failed: %s", exc)
            except Exception:  # noqa: BLE001
                pass
        return False


def normalize_memory_mode(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested mode (accepting dashes); None passes through. Raises ValueError
    for an unsupported mode so the route rejects it as a 4xx before any GPU work."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized:
        return None
    if normalized not in MEMORY_MODES:
        valid = ", ".join(MEMORY_MODES)
        raise ValueError(f"Unsupported diffusion memory_mode '{value}'. Use one of: {valid}.")
    return normalized


@dataclass(frozen = True)
class DeviceMemory:
    """Point-in-time view of the active device's memory, in MiB.

    ``memory_kind`` distinguishes discrete VRAM (CPU offload helps) from unified / system memory
    (offload moves bytes within the same pool, so it does not)."""

    backend: str
    device: str
    memory_kind: str  # "discrete_vram" | "unified_memory" | "system_memory" | "unknown"
    free_mib: Optional[int] = None
    total_mib: Optional[int] = None
    # Metal's recommended working set when it bounds free / total (mps); None everywhere else.
    metal_working_set_mib: Optional[int] = None

    @property
    def is_unified(self) -> bool:
        return self.memory_kind in ("unified_memory", "system_memory")

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "memory_kind": self.memory_kind,
            "free_mib": self.free_mib,
            "total_mib": self.total_mib,
        }


@dataclass(frozen = True)
class MemoryPlan:
    """The chosen runtime profile for one load."""

    requested_mode: str
    offload_policy: str
    vae_tiling: bool
    vae_slicing: bool
    device_memory: DeviceMemory
    estimates: dict[str, Optional[int]]
    reasons: tuple[str, ...] = ()
    # Off by default; set only when streaming text encoders is what makes group offload fit at all.
    stream_text_encoders: bool = False
    # False only on the tier keeping the transformer resident and streaming just the text encoders.
    stream_transformer: bool = True
    # MiB of the streamed denoiser / encoders kept resident (whole groups); None streams every group.
    resident_transformer_mib: Optional[int] = None
    resident_text_encoder_mib: Optional[int] = None
    secondary_ordinal: Optional[int] = None
    secondary_components: Optional[tuple[str, ...]] = None
    # The single-card plan a split replaced, applied on the primary if the split placement fails (a torchao denoiser
    # cannot take the whole-module offload a blind fallback would choose).
    fallback_plan: Optional["MemoryPlan"] = field(default = None, compare = False, repr = False)
    layer_split: Optional[Any] = None

    @property
    def engages_offload(self) -> bool:
        return self.offload_policy != OFFLOAD_NONE

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "offload_policy": self.offload_policy,
            "vae_tiling": self.vae_tiling,
            "vae_slicing": self.vae_slicing,
            "device_memory": self.device_memory.as_public_dict(),
            "estimates": dict(self.estimates),
            "reasons": list(self.reasons),
            "stream_text_encoders": self.stream_text_encoders,
            "stream_transformer": self.stream_transformer,
            "resident_transformer_mib": self.resident_transformer_mib,
            "resident_text_encoder_mib": self.resident_text_encoder_mib,
            "secondary_ordinal": self.secondary_ordinal,
            "secondary_components": (
                list(self.secondary_components) if self.secondary_components is not None else None
            ),
            **(
                {"layer_split": self.layer_split.as_public_dict()}
                if self.layer_split is not None
                else {}
            ),
        }


def snapshot_device_memory(target: Any) -> DeviceMemory:
    """Free / total memory for ``target``'s device. Never raises: a probe failure yields None
    counts, which the planner treats as "budget unknown" (stay resident)."""
    device = getattr(target, "device", "cpu")
    backend = getattr(target, "backend", device)

    if device == "cuda":
        free, total, kind = _cuda_memory(backend)
        return DeviceMemory(backend, device, kind, free, total)
    if device == "xpu":
        free, total = _xpu_memory()
        return DeviceMemory(backend, device, "discrete_vram", free, total)
    if device == "mps":
        # One CPU/GPU pool, so offload is pointless. The device is Metal's recommended working set (what the GPU holds
        # with good performance; allocations past it succeed and page), bounded by the RAM available right now.
        total, free = _system_memory_mib()
        ram = DeviceMemory(backend, device, "unified_memory", free, total)
        working_set, in_use = _metal_working_set_mib()
        if working_set is None:
            return ram
        ws_free = None if free is None else min(int(free), max(0, working_set - in_use))
        ws_total = working_set if total is None else min(int(total), working_set)
        metal = DeviceMemory(
            backend, device, "unified_memory", ws_free, ws_total, metal_working_set_mib = working_set
        )
        # The working set only widens the RAM reading: on an idle Mac with more RAM available than the working set,
        # 15% of it can leave less than RAM minus 20%, and a load that fit under the RAM reading must keep fitting.
        if free is not None and total is not None:
            ram_budget = int(free) - _reserve_mib("unified_memory", int(total))
            if ram_budget > int(ws_free) - _reserve_mib(METAL_WORKING_SET, ws_total):
                return ram
        return metal

    total, free = _system_memory_mib()
    return DeviceMemory(backend, device, "system_memory", free, total)


def _metal_working_set_mib() -> tuple[Optional[int], int]:
    """(Metal's recommended working set, what this process already holds in Metal) in MiB, or (None, 0) when torch
    cannot report it (``recommended_max_memory`` arrived in torch 2.5)."""
    try:
        import torch
        working_set = int(torch.mps.recommended_max_memory()) // (1024 * 1024)
        in_use = int(torch.mps.driver_allocated_memory()) // (1024 * 1024)
    except Exception:  # noqa: BLE001 - no reading: size against system RAM as before
        return None, 0
    return (working_set, in_use) if working_set > 0 else (None, 0)


def _graph_held_cache_bytes(device: Optional[int] = None) -> int:
    """Bytes inside ``memory_reserved() - memory_allocated()`` that ordinary allocations cannot reuse: the CUDA graph
    pools (per-block pools whole, the whole-step graphs' shared pool's unallocated part). ``device`` None counts every
    block pool and reads the whole-step pool on the current card."""
    held = 0
    try:
        from .diffusion_block_graph import pool_bytes as _block_graph_pool_bytes
        held += int(_block_graph_pool_bytes(device))
    except Exception:  # noqa: BLE001
        pass
    try:
        # The whole-step pool reading is per current card, so a named card other than it is made current for the read.
        from .diffusion_cuda_graph import live_pool_free_bytes

        scope = contextlib.nullcontext()
        if device is not None:
            import torch
            try:
                current = int(torch.cuda.current_device())
            except Exception:  # noqa: BLE001 - no current card to compare: switch to the named one
                current = None
            if current != int(device):
                scope = torch.cuda.device(int(device))
        with scope:
            held += int(live_pool_free_bytes())
    except Exception:  # noqa: BLE001 -- no graph layer: nothing held
        pass
    return held


def cuda_reclaimable_bytes(device: Optional[int] = None) -> Optional[int]:
    """Bytes this process's caching allocator holds reserved-but-unallocated on CUDA ``device`` (the current card
    when None) that an ordinary allocation can take back, minus the CUDA graph pools that only their graphs may reuse
    (``_graph_held_cache_bytes``). None when the allocator cannot be read. May be negative; callers credit only a
    positive figure."""
    try:
        import torch
        if device is None:
            reclaimable = int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated())
        else:
            reclaimable = int(torch.cuda.memory_reserved(device)) - int(
                torch.cuda.memory_allocated(device)
            )
    except Exception:  # noqa: BLE001
        return None
    pool_device = device
    if pool_device is None:
        try:
            pool_device = torch.cuda.current_device()
        except Exception:  # noqa: BLE001 - no current CUDA device: count every pool
            pool_device = None
    return reclaimable - _graph_held_cache_bytes(pool_device)


def cuda_reusable_cache_bytes(device: Any = None) -> int:
    """``cuda_reclaimable_bytes`` (excludes CUDA graph pools) floored at 0 for any device form; resolved to an int
    index because ``_graph_held_cache_bytes`` would silently skip the graph-pool count for a torch.device."""
    index = None
    if device is not None:
        try:
            import torch
            index = device if isinstance(device, int) else torch.device(device).index
        except Exception:  # noqa: BLE001 - unparseable: read the current card
            index = None
    return max(0, cuda_reclaimable_bytes(index) or 0)


def reclaimable_snapshot_device_memory(target: Any) -> DeviceMemory:
    """Credits reserved-minus-allocated as free without flushing; cuda only, own allocator only, CUDA graph pool uncredited, plain snapshot on failure."""
    if getattr(target, "device", "cpu") != "cuda":
        return snapshot_device_memory(target)
    snapshot = snapshot_device_memory(target)
    if snapshot.free_mib is None:
        return snapshot
    reclaimable = cuda_reclaimable_bytes()
    if reclaimable is None:  # no allocator reading: the plain snapshot still stands
        return snapshot
    if reclaimable <= 0:
        return snapshot
    free = int(snapshot.free_mib) + reclaimable // (1024 * 1024)
    if snapshot.total_mib is not None:
        free = min(free, int(snapshot.total_mib))
    return DeviceMemory(
        snapshot.backend, snapshot.device, snapshot.memory_kind, free, snapshot.total_mib
    )


def _settle_delay(delay_s: float) -> float:
    """Spacing of memory re-reads so a transient undercount can clear; UNSLOTH_SETTLE_DELAY_S overrides."""
    override = os.environ.get("UNSLOTH_SETTLE_DELAY_S")
    if override is None:
        return delay_s
    try:
        return max(0.0, float(override))
    except (TypeError, ValueError):
        return delay_s  # a typo in the env must not change production behaviour


def settled_snapshot_device_memory(
    target: Any,
    attempts: int = 3,
    delay_s: float = 1.0,
) -> DeviceMemory:
    """Max free over spaced reads (a transient only shrinks free); on mps, empty_cache first so freed pool is not budgeted as used memory."""
    device = getattr(target, "device", "cpu")
    if device == "mps":
        try:
            import torch
            empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception:  # noqa: BLE001 - settle is best-effort; the snapshot below still runs
            pass
        return snapshot_device_memory(target)
    if device != "cuda":
        return snapshot_device_memory(target)
    try:
        import torch
        from .diffusion_cuda_graph import synchronize_outside_capture

        synchronize_outside_capture()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - settle is best-effort; the snapshot below still runs
        pass
    best = snapshot_device_memory(target)
    delay_s = _settle_delay(delay_s)
    for _ in range(max(0, attempts - 1)):
        if best.free_mib is not None and best.total_mib is not None:
            # Free already within the reserve of total: nothing transient to wait out.
            if best.free_mib >= best.total_mib - max(2048, int(best.total_mib * 0.10)):
                break
        try:
            import time
            time.sleep(delay_s)
        except Exception:  # noqa: BLE001
            break
        nxt = snapshot_device_memory(target)
        if nxt.free_mib is not None and (best.free_mib is None or nxt.free_mib > best.free_mib):
            best = nxt
    return best


def _cuda_memory(backend: str) -> tuple[Optional[int], Optional[int], str]:
    try:
        import torch

        # Not torch.cuda.mem_get_info: Windows ROCm over-reports free memory.
        from utils.hardware import trusted_mem_get_info

        free, total = trusted_mem_get_info()
        kind = "discrete_vram"
        try:
            # Current device, not 0: mem_get_info reports the current one.
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            if bool(getattr(props, "integrated", False) or getattr(props, "is_integrated", False)):
                kind = "unified_memory"  # e.g. Jetson / integrated SoC
        except Exception:
            pass
        free_mib, total_mib = int(free // (1024 * 1024)), int(total // (1024 * 1024))
        # ROCm APUs over-report free (free == total on Windows HIP): never credit host memory.
        if kind == "unified_memory" and not _torch_is_rocm(torch):
            free_mib, total_mib = _unified_reclaimable_memory_mib(free_mib, total_mib)
        return free_mib, total_mib, kind
    except Exception:
        return None, None, "discrete_vram"


def _torch_is_rocm(torch) -> bool:
    """Delegates to LlamaCppBackend so the two agree; version.hip alone misses some ROCm builds."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._torch_is_rocm(torch)
    except Exception:  # noqa: BLE001 - the answer still has to be right
        return is_rocm_torch(torch)


def _unified_reclaimable_memory_mib(free_mib: int, total_mib: int) -> tuple[int, int]:
    """Credits page cache via MemAvailable, clamped to driver reading and device total; also returns capacity (cgroup LIMIT, not remainder)."""
    available_mib = _available_system_memory_mib()
    cgroup_mib = _cgroup_available_memory_mib()
    if available_mib is None:
        credited = free_mib
    else:
        credited = max(free_mib, min(int(available_mib), total_mib))
    if cgroup_mib is not None and int(cgroup_mib) <= credited:
        # `<=`: the host reading is already cgroup-capped, so equality is normal in a container.
        credited = int(cgroup_mib)
    # Capacity is the cgroup LIMIT, never the remainder, which shrinks as the container fills.
    limit_mib = _cgroup_memory_limit_mib()
    if limit_mib is None:
        capacity = total_mib
    else:
        capacity = min(total_mib, int(limit_mib))
        # Memory above the limit cannot be charged, so it is not free either.
        credited = min(credited, capacity)
    return credited, capacity


def _available_system_memory_mib() -> Optional[int]:
    """Usable host RAM in MiB: system available capped by any enforcing cgroup's headroom."""
    from utils import host_memory
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._available_system_memory_mib()
    except Exception:  # noqa: BLE001 - the shared reader still applies the cgroup cap
        return host_memory.usable_mib(_system_memory_mib()[1], _cgroup_available_memory_mib())


def _cgroup_available_memory_mib() -> Optional[int]:
    """What an enforcing cgroup will still let this process charge, else None."""
    from utils import host_memory
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._cgroup_available_memory_mib()
    except Exception:  # noqa: BLE001 - read the cgroup directly rather than drop the cap
        try:
            return host_memory.cgroup_headroom_mib()
        except Exception:  # noqa: BLE001 - no readable limit is the same answer as none
            return None


def _cgroup_memory_limit_mib() -> Optional[int]:
    """The capacity an enforcing cgroup allows, else None. Not the remainder above."""
    from utils import host_memory
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._cgroup_memory_limit_mib()
    except Exception:  # noqa: BLE001 - read the cgroup directly rather than drop the cap
        try:
            return host_memory.cgroup_limit_mib()
        except Exception:  # noqa: BLE001 - no readable limit is the same answer as none
            return None


def _host_ram_capacity_mib() -> Optional[int]:
    """Host RAM this process may ever charge, in MiB: system total capped by any cgroup limit."""
    from utils import host_memory
    return host_memory.usable_mib(_system_memory_mib()[0], _cgroup_memory_limit_mib())


def _windows_commit_headroom_mib() -> Optional[int]:
    """(CommitLimit - CommitTotal) in MiB on Windows via ``K32GetPerformanceInfo``; 0 when exhausted, None when
    unreadable or not Windows. The diffusion guards' failure mode is commit exhaustion (#13432), which a
    physical `available` reading does not see."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class _PERFORMANCE_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("CommitTotal", ctypes.c_size_t),
                ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t),
                ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t),
                ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t),
                ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong),
                ("ProcessCount", ctypes.c_ulong),
                ("ThreadCount", ctypes.c_ulong),
            ]

        info = _PERFORMANCE_INFORMATION()
        info.cb = ctypes.sizeof(_PERFORMANCE_INFORMATION)
        # kernel32 exports the K32-prefixed name only; the unprefixed one lives in psapi
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        if not kernel32.K32GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None
        headroom = (int(info.CommitLimit) - int(info.CommitTotal)) * int(info.PageSize)
        return max(0, headroom // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return None


def _xpu_memory() -> tuple[Optional[int], Optional[int]]:
    try:
        import torch
        mem_get_info = getattr(getattr(torch, "xpu", None), "mem_get_info", None)
        if callable(mem_get_info):
            free, total = mem_get_info()
            return int(free // (1024 * 1024)), int(total // (1024 * 1024))
    except Exception:
        pass
    return None, None


def _system_memory_mib() -> tuple[Optional[int], Optional[int]]:
    """(total, available) host RAM in MiB, via psutil then POSIX sysconf."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total // (1024 * 1024)), int(vm.available // (1024 * 1024))
    except Exception:
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page
        avail = os.sysconf("SC_AVPHYS_PAGES") * page
        return int(total // (1024 * 1024)), int(avail // (1024 * 1024))
    except Exception:
        return None, None


def file_size_mib(path: Any) -> Optional[int]:
    """On-disk size of ``path`` in MiB, or None if it can't be stat'd."""
    try:
        from pathlib import Path
        return max(1, int(Path(path).expanduser().stat().st_size // (1024 * 1024)))
    except Exception:
        return None


def safetensors_prefix_mib(path: Any, prefix: str) -> Optional[int]:
    """MiB of the ``prefix*`` tensors, from the safetensors header alone; None if unreadable or unmatched."""
    from .diffusion_component_sizes import entry_nbytes, read_safetensors_header, tensor_entries

    header = read_safetensors_header(path)
    if header is None:
        return None
    try:
        total = 0
        for name, meta in tensor_entries(header):
            if name.startswith(prefix):
                total += entry_nbytes(meta)
    except Exception:  # noqa: BLE001 - not a readable safetensors header
        return None
    return -(-total // (1024 * 1024)) if total > 0 else None


def estimate_gguf_resident_mib(storage_mib: Optional[int]) -> Optional[int]:
    """GGUF weights stay packed, so resident size is about the on-disk size, not unpacked bf16."""
    if storage_mib is None:
        return None
    return int(storage_mib * 1.05)  # margin for allocator + bf16 norms/biases


def estimate_safetensors_dense_mib(
    storage_mib: Optional[int], *, fp8_upcast: bool = False
) -> Optional[int]:
    """Upcast fp8 checkpoints load as bf16 with no quantization_config, so resident size is about 2x."""
    if storage_mib is None:
        return None
    if fp8_upcast:
        return storage_mib * 2
    return storage_mib


# qwen-image-layered's layer_count / layer_resolution (pinned equal by a test; no families import here).
_QWEN_LAYERED_LAYERS = 2
_QWEN_LAYERED_CANVAS = 640


def estimate_image_runtime_mib(
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    family: Optional[str] = None,
    condition_pixels: int = 0,
) -> int:
    """Per-call activation / latent headroom for an image gen, scaled by pixel area and batch.
    Distilled / turbo models (few steps, no CFG) need less. ``condition_pixels`` (already weighted)
    join the target's token sequence once per batch image, so they count like target pixels."""
    w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
    h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
    batch = max(1, int(batch_size or 1))
    cond = max(0, int(condition_pixels or 0))
    if width is None and height is None and "qwen-image-layered" in (family or "").lower():
        # Planning call: reserve what the generation guard charges a decomposition, not one 1024x1024 frame.
        w = h = _QWEN_LAYERED_CANVAS
        cond = (_QWEN_LAYERED_LAYERS + 1) * w * h
    pixel_scale = ((w * h + cond) * batch) / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT)
    return max(1024, int(8192 * max(0.25, pixel_scale) * _family_activation_multiplier(family)))


def estimate_video_runtime_mib(
    *, width: Optional[int], height: Optional[int], num_frames: Optional[int]
) -> int:
    """VAE decode is the video peak: decoded frames times 3 for intermediates and export."""
    w = max(64, int(width or 768))
    h = max(64, int(height or 512))
    frames = max(1, int(num_frames or 121))
    decoded_mib = (frames * w * h * 3 * 4) / float(1024 * 1024)
    return max(3072, int(4096 + 3.0 * decoded_mib))


@dataclass(frozen = True)
class CalibratedImageActivation:
    """MiB above the resident weights, margin included: each phase at 1024x1024, then the 2048x2048 worst case."""

    text_encoder_mib: int
    denoise_mib: int
    decode_mib: int
    tiled_decode_mib: int
    max_canvas_mib: int
    resident_only: bool = False

    def headroom(self, tiled: bool) -> int:
        return max(
            self.text_encoder_mib,
            self.denoise_mib,
            self.tiled_decode_mib if tiled else self.decode_mib,
        )


# NVIDIA worst case per (off / eager / default, max) tier; U-Nets unlisted (cannot stream encoders).
_ACTIVATION_MARGIN = 1.2
_MEASURED_IMAGE_ACTIVATION_MIB: dict[
    str, tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]
] = {
    # Text encoder, denoise, untiled/tiled decode at 1024x1024; max at 2048x2048; Qwen tiles are 512 px.
    "qwen-image-2.1": ((1_849, 666, 7_648, 1_697, 2_479), (1_849, 2_489, 7_648, 1_697, 9_602)),
    "flux.1": ((288, 892, 2_666, 2_456, 2_674), (288, 892, 2_666, 2_456, 2_674)),
    "flux.2-klein": ((1_516, 1_160, 2_645, 2_456, 3_876), (1_516, 1_205, 2_677, 2_456, 3_924)),
    # FLUX.3 Klein (distilled, no CFG), measured B200 bf16, 1024^2 t2i: TE 301 MiB at a ~560-token window, denoise eager
    # 594 bf16 / 901 int8 / 850 fp8, compiled 543-676 (int8 / fp8, regional dynamic=True), untiled AutoencoderKLFlux2
    # decode 2_437. Tiled decode is unmeasured (flux.2-klein's same VAE reads 2_456): capped at the untiled figure, which
    # a tiled decode cannot exceed. The compiled tier keeps the eager 901 rather than its measured 543-676, since the
    # table's tiers may only grow with speed. The 2048^2 column is max(denoise eager fp8 3_359, tiled decode) since the
    # untiled 2048 decode is 9_738. References are charged through ``condition_pixel_weight``. The base (CFG) is
    # unmeasured and keeps the flat estimate.
    "flux.3-klein": ((301, 901, 2_437, 2_437, 3_359), (301, 901, 2_437, 2_437, 3_359)),
    # FLUX.3 Klein base, B200 bf16: per column the larger of Klein and base (same DiT / VAE, one Flux2 formula in ComfyUI).
    "flux.3-klein-base": ((301, 901, 2_512, 2_512, 3_359), (301, 901, 2_512, 2_512, 3_359)),
    # MOSS-Image SFT, B200 bf16 measurements; both tiers take the eager worst.
    "moss-image": ((1_940, 528, 3_666, 3_662, 3_750), (1_940, 528, 3_666, 3_662, 3_750)),
    "z-image": ((744, 1_199, 2_666, 2_456, 4_327), (744, 1_271, 2_669, 2_456, 4_327)),
    # SDXL base, T4 fp16, 1024 (2048 tiled decode 2_455; untiled 7_706 left to the OOM retry); compiled unmeasured.
    "sdxl": ((9, 327, 2_434, 2_434, 2_455), (9, 327, 2_434, 2_434, 2_455)),
}
# U-Nets cannot stream encoders on group tiers: the measurement only decides all-resident vs whole-module offload.
_RESIDENT_ONLY_CALIBRATED = frozenset({"sdxl"})


def _resident_only_hint(family: Optional[str]) -> bool:
    parts = str(family or "").split()
    return bool(parts) and parts[0] in _RESIDENT_ONLY_CALIBRATED


def calibrated_image_activation(
    family: Optional[str], *, max_speed: bool = True
) -> Optional[CalibratedImageActivation]:
    measured = _MEASURED_IMAGE_ACTIVATION_MIB.get(str(family or ""))
    if measured is None:
        return None
    return CalibratedImageActivation(
        *(int(v * _ACTIVATION_MARGIN) for v in measured[1 if max_speed else 0]),
        resident_only = str(family or "") in _RESIDENT_ONLY_CALIBRATED,
    )


# ComfyUI's minimum_inference_memory() (comfy/model_management.py); activations are priced separately here.
COMFY_MIN_INFERENCE_MIB = 800
COMFY_EXTRA_RESERVED_MIB = 400
COMFY_EXTRA_RESERVED_WINDOWS_MIB = 600
COMFY_EXTRA_RESERVED_WINDOWS_BIG_CARD_MIB = 100
# Placements measured on T4 / 8-16 GB cards are tuned against 2 GiB; ComfyUI's 1.2 GiB alone flips them.
DISCRETE_RESERVE_FLOOR_MIB = 2048


def comfy_min_inference_mib(total_mib: Optional[int] = None) -> int:
    """ComfyUI's ``minimum_inference_memory()`` in MiB: 1200 on Linux, 1400 / 1500 (card above 15 GiB) on Windows."""
    if sys.platform != "win32":
        return COMFY_MIN_INFERENCE_MIB + COMFY_EXTRA_RESERVED_MIB
    extra = COMFY_EXTRA_RESERVED_WINDOWS_MIB
    if int(total_mib or 0) > 15 * 1024:
        extra += COMFY_EXTRA_RESERVED_WINDOWS_BIG_CARD_MIB
    return COMFY_MIN_INFERENCE_MIB + extra


def discrete_reserve_mib(total_mib: Optional[int] = None) -> int:
    """A discrete card's fixed reserve: ComfyUI's, floored at 2 GiB."""
    return max(DISCRETE_RESERVE_FLOOR_MIB, comfy_min_inference_mib(total_mib))


_ROCM_APU_POOL = "rocm_apu_pool"


def _reserve_mib(memory_kind: str, base: int) -> int:
    if memory_kind == METAL_WORKING_SET:
        # Only other apps' growth during the load: the working set already sits below RAM. At least as strict as
        # llama.cpp (_APPLE_UNIFIED_MEMORY_FRACTION), which keeps 15% of the lower of available RAM and working set.
        return int(base * 0.15)
    if memory_kind == "unified_memory":
        return max(2048, int(base * 0.20))  # OS + CPU share this pool
    if memory_kind == "system_memory":
        return max(1024, int(base * 0.10))
    if memory_kind == _ROCM_APU_POOL:
        # The amdgpu GTT pool of an APU whose host RAM already covers the OS: still shared with the CPU side, so it
        # keeps the uncapped 10%.
        return max(2048, int(base * 0.10))
    # Fixed, not a share: a capped 10% kept 2.4-4 GiB idle on 24-40 GiB cards.
    return discrete_reserve_mib(base)


def _rocm_linux_apu_os_room_outside_pool(memory: DeviceMemory) -> bool:
    """Linux ROCm APU whose host RAM outside the pool's free part already covers the unified OS reserve: the pool is
    the amdgpu GTT cap, below physical RAM, so reserving 20% of it too reserves twice. Not Windows (HIP over-reports
    free, #7072); unknown readings answer False."""
    if (
        memory.memory_kind != "unified_memory"
        or str(getattr(memory, "device", None) or "").split(":")[0] != "cuda"
        or not sys.platform.startswith("linux")
        or memory.free_mib is None
    ):
        return False
    torch = sys.modules.get("torch")
    if torch is None or not _torch_is_rocm(torch):
        return False
    available = _available_system_memory_mib()
    if available is None:
        return False
    os_reserve = _reserve_mib("unified_memory", memory.total_mib or memory.free_mib)
    return int(available) - int(memory.free_mib) >= os_reserve


def _budget_reserve_kind(memory: DeviceMemory) -> str:
    if getattr(memory, "metal_working_set_mib", None) is not None:
        return METAL_WORKING_SET
    return _ROCM_APU_POOL if _rocm_linux_apu_os_room_outside_pool(memory) else memory.memory_kind


_LOAD_BUDGET_FRACTIONS: dict[str, float] = {}
_LOAD_BUDGET_LOCK = threading.Lock()
_BUDGET_SCOPE: contextvars.ContextVar[Optional[tuple]] = contextvars.ContextVar(
    "unsloth_vram_budget_scope", default = None
)
_UPCOMING_LOAD_FRACTION: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "unsloth_vram_budget_upcoming_load", default = None
)


def _live_vram_budget_fraction() -> float:
    from utils.vram_budget_settings import get_vram_budget_fraction
    return float(get_vram_budget_fraction())


def _vram_budget_fraction() -> float:
    """The fraction the reserve readers charge."""
    scope = _BUDGET_SCOPE.get()
    with _LOAD_BUDGET_LOCK:
        if scope is not None and scope[0] == "load" and scope[2] is not None:
            return float(scope[2])
        if scope is not None and scope[1] in _LOAD_BUDGET_FRACTIONS:
            return _LOAD_BUDGET_FRACTIONS[scope[1]]
        if scope is None:
            upcoming = _UPCOMING_LOAD_FRACTION.get()
            if upcoming is not None:
                return float(upcoming)
        pinned = tuple(_LOAD_BUDGET_FRACTIONS.values())
    return min(pinned) if pinned else _live_vram_budget_fraction()


def vram_budget_scope(owner: str, *, load: bool = False) -> Callable:
    """Decorator: run a backend load (snapshots the live Budget) or generation under ``owner``'s budget scope."""

    def wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def scoped(*args: Any, **kwargs: Any) -> Any:
            outer = _BUDGET_SCOPE.get()
            if load and outer is not None and outer[0] == "load" and outer[1] == owner:
                return fn(*args, **kwargs)
            if load:
                fraction: Optional[float] = _UPCOMING_LOAD_FRACTION.get()
                if fraction is None:
                    try:
                        fraction = _live_vram_budget_fraction()
                    except Exception:  # noqa: BLE001 - settings are a convenience; a load must never fail on them
                        fraction = None
                scope = ("load", owner, fraction)
            else:
                scope = ("generate", owner, None)
            token = _BUDGET_SCOPE.set(scope)
            try:
                return fn(*args, **kwargs)
            finally:
                _BUDGET_SCOPE.reset(token)

        return scoped

    return wrap


def snapshot_upcoming_load_budget() -> Optional[float]:
    """Snapshot the live VRAM Budget for the load this request is preparing, before it ranks GPUs, so the ranking, the plan and the load itself agree."""
    fraction = _UPCOMING_LOAD_FRACTION.get()
    if fraction is not None:
        return fraction
    try:
        fraction = _live_vram_budget_fraction()
    except Exception:  # noqa: BLE001
        return None
    _UPCOMING_LOAD_FRACTION.set(fraction)
    return fraction


@contextlib.contextmanager
def upcoming_load_budget() -> Iterator[Optional[float]]:
    """``snapshot_upcoming_load_budget`` for a block only (a ranking helper that runs outside a request's own context)."""
    if _UPCOMING_LOAD_FRACTION.get() is not None:
        yield _UPCOMING_LOAD_FRACTION.get()
        return
    try:
        fraction: Optional[float] = _live_vram_budget_fraction()
    except Exception:  # noqa: BLE001
        fraction = None
    token = _UPCOMING_LOAD_FRACTION.set(fraction)
    try:
        yield fraction
    finally:
        _UPCOMING_LOAD_FRACTION.reset(token)


def carry_load_budget(fn: Callable) -> Callable:
    """Wrap a load's worker-thread target so it inherits the fraction the calling load began with."""
    scope = _BUDGET_SCOPE.get()
    fraction = (
        scope[2] if scope is not None and scope[0] == "load" else _UPCOMING_LOAD_FRACTION.get()
    )
    if fraction is None:
        return fn

    @functools.wraps(fn)
    def carried(*args: Any, **kwargs: Any) -> Any:
        token = _UPCOMING_LOAD_FRACTION.set(float(fraction))
        try:
            return fn(*args, **kwargs)
        finally:
            _UPCOMING_LOAD_FRACTION.reset(token)

    return carried


def commit_load_budget_fraction(owner: str) -> None:
    """Publish the fraction ``owner``'s load planned with, alongside its committed state. Never raises."""
    scope = _BUDGET_SCOPE.get()
    fraction = scope[2] if scope is not None and scope[0] == "load" and scope[1] == owner else None
    if fraction is None:
        try:
            fraction = _live_vram_budget_fraction()
        except Exception:  # noqa: BLE001
            return
    with _LOAD_BUDGET_LOCK:
        _LOAD_BUDGET_FRACTIONS[owner] = float(fraction)


def release_load_budget_fraction(owner: str) -> None:
    """Drop ``owner``'s snapshot when its committed pipeline is torn down."""
    with _LOAD_BUDGET_LOCK:
        _LOAD_BUDGET_FRACTIONS.pop(owner, None)


def _user_budget_reserve_mib(base: Optional[int]) -> int:
    """What the user's VRAM Budget setting keeps free of ``base``."""
    if not base or int(base) <= 0:
        return 0
    try:
        from utils.vram_budget_settings import VRAM_FRACTION_DEFAULT
        fraction = _vram_budget_fraction()
    except Exception:  # noqa: BLE001 - settings are a convenience; a load must never fail on them
        return 0
    if not fraction < VRAM_FRACTION_DEFAULT:
        return 0
    return int(base * (1.0 - fraction))


def user_budget_reserve_share() -> float:
    """The share of each card the user's VRAM Budget keeps free."""
    try:
        from utils.vram_budget_settings import VRAM_FRACTION_DEFAULT
        fraction = _vram_budget_fraction()
    except Exception:  # noqa: BLE001 - settings are a convenience; a load must never fail on them
        return 0.0
    return 1.0 - fraction if fraction < VRAM_FRACTION_DEFAULT else 0.0


def user_budget_reserve_bytes(device: Any, total_bytes: Optional[int]) -> int:
    """What the user's VRAM Budget keeps free on ``device`` (``total_bytes`` total), in bytes."""
    kind = str(getattr(device, "type", device) or "").split(":")[0]
    if kind in ("cpu", "mps") or not total_bytes:
        return 0
    try:
        return _user_budget_reserve_mib(int(total_bytes) >> 20) << 20
    except Exception:  # noqa: BLE001
        return 0


def _device_user_budget_reserve_mib(memory: DeviceMemory, base: Optional[int]) -> int:
    """:func:`_user_budget_reserve_mib` for ``memory``, 0 where the LLM path ignores the setting too."""
    backend = str(getattr(memory, "backend", "") or "").split(":")[0]
    if (
        backend in ("mps", "cpu")
        or getattr(memory, "memory_kind", None) == "system_memory"
        or getattr(memory, "metal_working_set_mib", None) is not None
    ):
        return 0
    return _user_budget_reserve_mib(base)


def _planner_reserve_mib(memory: DeviceMemory, base: int) -> int:
    return max(
        _reserve_mib(_budget_reserve_kind(memory), base),
        _device_user_budget_reserve_mib(memory, base),
    )


def _safe_device_budget_mib(memory: DeviceMemory) -> Optional[int]:
    """Free memory minus a headroom reserve (room for fragmentation + other tenants). None when
    free memory is unknown."""
    if memory.free_mib is None:
        return None
    base = memory.total_mib or memory.free_mib
    return max(0, int(memory.free_mib) - _planner_reserve_mib(memory, base))


def _fast_device_budget_mib(memory: DeviceMemory) -> Optional[int]:
    """Free memory minus half the standard reserve (min 2 GiB): an explicit ``fast`` offloads only when resident
    would not fit, not to keep ``auto``'s headroom for other tenants."""
    if memory.free_mib is None:
        return None
    base = memory.total_mib or memory.free_mib
    reserve = max(
        2048,
        _reserve_mib(_budget_reserve_kind(memory), base) // 2,
        _device_user_budget_reserve_mib(memory, base),
    )
    return max(0, int(memory.free_mib) - reserve)


def plan_keeps_transformer_resident(plan: Any) -> bool:
    """Whether ``plan`` never moves the denoiser after placement: torchao survives placement, not per-forward hooks."""
    policy = getattr(plan, "offload_policy", OFFLOAD_NONE)
    if policy == OFFLOAD_NONE:
        return True
    return policy == OFFLOAD_GROUP and not bool(getattr(plan, "stream_transformer", True))


PREQUANT_SEED_ON_HOST_ENV = "UNSLOTH_DIFFUSION_PREQUANT_SEED_ON_HOST"


def prequant_seed_device(
    plan: Any,
    device: str,
    scheme: Optional[str] = None,
) -> str:
    """Where a seeded pre-quantized denoiser is materialised: ``device`` when ``plan`` keeps it resident, else "cpu"
    for the schemes measured under offload. Loading it onto the GPU first adds a whole-denoiser spike that left the
    streaming hooks no room for their first block (Qwen-Image-2.1 int8 on 8 GB)."""
    if str(os.environ.get(PREQUANT_SEED_ON_HOST_ENV, "")).strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return device
    if plan is None or plan_keeps_transformer_resident(plan):
        return device
    if str(scheme) not in _TORCHAO_GROUP_OFFLOAD_MIN:
        return device
    return "cpu"


# Oldest torchao measured bit-exact under streamed group offload; 0.17 int8 (v1, no copy stream) ran 14x slower.
_TORCHAO_GROUP_OFFLOAD_MIN = {"int8": (0, 18), "fp8": (0, 17)}
# Without install_torchao_v1_int8_pin_ops, fresh torchao 0.17 installs got fp8 when offloading.
INT8_STREAM_TORCHAO17_ENV = "UNSLOTH_DIFFUSION_INT8_STREAM_TORCHAO17"
_TORCHAO17_INT8_STREAM_MIN = (0, 17)


def _int8_tensor_pinnable() -> bool:
    """Whether the installed torchao's ``Int8Tensor`` implements the pin ops the stream needs (0.17+)."""
    try:
        from torchao.quantization.quantize_.workflows.int8 import int8_tensor
        import torch

        table = getattr(int8_tensor.Int8Tensor, "_ATEN_OP_TABLE", None)
        if isinstance(table, dict):
            ops = set()
            for value in table.values():
                ops.update(value.keys() if isinstance(value, dict) else ())
            if ops:
                return (
                    torch.ops.aten._pin_memory.default in ops
                    and torch.ops.aten.is_pinned.default in ops
                )
        return True  # no readable op table: the class exists, which is what 0.17 added
    except Exception:  # noqa: BLE001 - no Int8Tensor: keep the 0.18 floor
        return False


def _int8_stream_floor() -> tuple:
    if str(os.environ.get(INT8_STREAM_TORCHAO17_ENV, "")).strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return _TORCHAO_GROUP_OFFLOAD_MIN["int8"]
    if not _int8_tensor_pinnable():
        return _TORCHAO_GROUP_OFFLOAD_MIN["int8"]
    return _TORCHAO17_INT8_STREAM_MIN


_TORCHAO_STREAM_SAFE_CLASSES = frozenset(("Int8Tensor", "Float8Tensor"))
# Before 0.38 diffusers moved only the torchao wrapper, leaving quantised data on the host.
_DIFFUSERS_TORCHAO_GROUP_OFFLOAD_MIN = (0, 38)
_UNSET: Any = object()


def _installed_version(package: str) -> Optional[tuple[int, int]]:
    try:
        import re
        from importlib.metadata import version

        major, minor = re.match(r"(\d+)\.(\d+)", version(package)).groups()
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 - absent / unreadable metadata
        return None


@functools.lru_cache(maxsize = 1)
def _installed_torchao_version() -> Optional[tuple[int, int]]:
    return _installed_version("torchao")


@functools.lru_cache(maxsize = 1)
def _installed_diffusers_version() -> Optional[tuple[int, int]]:
    return _installed_version("diffusers")


def _model_offload_fits_quantised(plan: Any) -> bool:
    """Quantised denoiser and encoders (SUM stands in for the largest) must each fit the budget whole."""
    try:
        est = plan.estimates
        budget = est.get("safe_device_budget_mib")
        model = est.get("model_dense_mib")
        companions = est.get("companion_dense_mib")
        if budget is None or model is None or companions is None:
            return False
        denoiser = max(0, int(model) - int(companions))
        overhead = int(est.get("runtime_headroom_mib") or 0) + int(
            est.get("base_overhead_mib") or 0
        )
        encoders = est.get("text_encoder_dense_mib")
        if encoders is None:
            encoders = companions
        return denoiser + overhead <= int(budget) and int(encoders) <= int(budget)
    except Exception:  # noqa: BLE001 - malformed plan: keep the resident rule
        return False


def torchao_offload_plan(
    plan: Any,
    scheme: Optional[str],
    *,
    torchao_version: Any = _UNSET,
    reclaimable_host_mib: int = 0,
) -> Optional[Any]:
    """Placement a torchao ``scheme`` denoiser survives under ``plan``, or None (sequential offload: never measured).
    ``reclaimable_host_mib``: host RAM the outgoing pipeline frees before this one pins."""
    if plan_keeps_transformer_resident(plan):
        return plan
    policy = getattr(plan, "offload_policy", OFFLOAD_NONE)
    if policy == OFFLOAD_MODEL and _model_offload_fits_quantised(plan):
        return plan
    if policy not in (OFFLOAD_MODEL, OFFLOAD_GROUP, OFFLOAD_STREAMING):
        return None
    if not torchao_scheme_streams(scheme, torchao_version = torchao_version):
        return None
    if not _torchao_stream_pinnable(plan, reclaimable_host_mib):
        return None
    if policy != OFFLOAD_MODEL:
        return plan
    return torchao_streaming_plan(plan)


def _torchao_stream_pinnable(plan: Any, reclaimable_host_mib: int = 0) -> bool:
    """Lazy pinning refuses torchao and the stream-free fallback ran ~35x slower, so require an up-front pin."""
    forced = str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower()
    if forced in ("0", "off", "false", "no"):
        return False
    if forced in ("1", "on", "true", "yes"):
        return True
    if _pinned_memory_capped():
        return False
    try:
        est = plan.estimates
        denoiser = int(est["model_dense_mib"]) - int(est["companion_dense_mib"])
    except Exception:  # noqa: BLE001 - an unsized plan cannot prove the pin fits
        return False
    budget = _pin_budget_mib()
    return budget is not None and 0 <= denoiser <= budget + max(0, int(reclaimable_host_mib))


def pipeline_host_mib(pipe: Any) -> int:
    """Host MiB held by ``pipe``'s components, which an unload hands back."""
    if pipe is None:
        return 0
    total = 0
    try:
        components = getattr(pipe, "components", None) or {}
        for module in components.values():
            if hasattr(module, "parameters"):
                total += _module_host_mib_on_cpu(module)
    except Exception:  # noqa: BLE001 - unsizeable: credit nothing
        return 0
    return total


def _module_host_mib_on_cpu(module: Any) -> int:
    try:
        seen: set[int] = set()
        nbytes = 0
        for tensor in (*module.parameters(), *module.buffers()):
            if id(tensor) in seen or getattr(tensor, "device", None) is None:
                continue
            seen.add(id(tensor))
            if tensor.device.type == "cpu":
                nbytes += sum(_storage_nbytes(tensor))
        return nbytes >> 20
    except Exception:  # noqa: BLE001
        return 0


def torchao_survives_plan(
    plan: Any,
    scheme: Optional[str],
    *,
    torchao_version: Any = _UNSET,
    reclaimable_host_mib: int = 0,
) -> bool:
    return (
        torchao_offload_plan(
            plan,
            scheme,
            torchao_version = torchao_version,
            reclaimable_host_mib = reclaimable_host_mib,
        )
        is not None
    )


def torchao_scheme_streams(scheme: Optional[str], *, torchao_version: Any = _UNSET) -> bool:
    floor = _TORCHAO_GROUP_OFFLOAD_MIN.get(str(scheme))
    if floor is None:
        return False
    if str(scheme) == "int8":
        floor = _int8_stream_floor()
    diffusers_version = _installed_diffusers_version()
    if diffusers_version is None or diffusers_version < _DIFFUSERS_TORCHAO_GROUP_OFFLOAD_MIN:
        return False
    version = _installed_torchao_version() if torchao_version is _UNSET else torchao_version
    return version is not None and tuple(version)[:2] >= floor


def torchao_streaming_plan(plan: Any) -> Any:
    return replace(
        plan,
        offload_policy = OFFLOAD_STREAMING,
        reasons = tuple(plan.reasons)
        + (
            "the quantised transformer or a text encoder does not fit the device budget whole; streaming "
            "transformer blocks and text-encoder layers",
        ),
    )


def _torchao_weight_classes(module: Any) -> set[str]:
    classes: set[str] = set()
    try:
        for param in module.parameters():
            for tensor in (param, getattr(param, "data", None)):
                if tensor is not None and type(tensor).__module__.startswith("torchao"):
                    classes.add(type(tensor).__name__)
    except Exception:  # noqa: BLE001 - unreadable: report nothing, the caller keeps its kwargs
        return set()
    return classes


def _torchao_group_offload_kwargs(
    module: Any,
    kwargs: dict[str, Any],
    pinned_mib: Optional[list] = None,
) -> dict[str, Any]:
    """Group-offload kwargs a torchao ``module`` survives. Frozen first (swap_tensors on a requires_grad torchao weight
    hits an unimplemented ``aten.view``); the stream needs an up-front pin within ``pinned_mib``, a shared running total."""
    classes = _torchao_weight_classes(module)
    if not classes:
        return kwargs
    _freeze_torchao_weights(module)
    install_group_offload_torchao_swap_retry()
    if not kwargs.get("use_stream"):
        return kwargs
    if classes <= _TORCHAO_STREAM_SAFE_CLASSES or (
        classes <= _TORCHAO_V1_INT8_CLASSES and install_torchao_v1_int8_pin_ops()
    ):
        if not kwargs.get("low_cpu_mem_usage"):
            return kwargs
        # Same override the planner's _torchao_stream_pinnable honoured, so the two cannot disagree.
        forced = str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower()
        if forced in ("1", "on", "true", "yes"):
            return {**kwargs, "low_cpu_mem_usage": False}
        budget = (
            None
            if forced in ("0", "off", "false", "no") or _pinned_memory_capped()
            else _pin_budget_mib()
        )
        need = _module_pin_mib(module)
        already = pinned_mib[0] if pinned_mib else 0
        if budget is not None and already + need <= budget:
            if pinned_mib is not None:
                pinned_mib[0] = already + need
            return {**kwargs, "low_cpu_mem_usage": False}
    safe = {
        k: v
        for k, v in kwargs.items()
        if k not in ("non_blocking", "record_stream", "low_cpu_mem_usage")
    }
    safe["use_stream"] = False
    return safe


# torchao <= 0.17's default int8 weight (LinearActivationQuantizedTensor over an AffineQuantizedTensor).
_TORCHAO_V1_INT8_CLASSES = frozenset(("LinearActivationQuantizedTensor", "AffineQuantizedTensor"))
_V1_INT8_PIN_OPS_INSTALLED = False


def install_torchao_v1_int8_pin_ops() -> bool:
    """Register ``is_pinned`` / ``pin_memory`` on torchao's v1 int8 classes so group offload streams them (else
    synchronous copies, 14x slower). Re-wraps the same int8 data, scale and zero point, so outputs stay bit-identical.
    Only where torchao ships the classes without the ops (<= 0.17)."""
    global _V1_INT8_PIN_OPS_INSTALLED
    if _V1_INT8_PIN_OPS_INSTALLED:
        return True
    if str(os.environ.get(INT8_STREAM_TORCHAO17_ENV, "")).strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return False
    try:
        import torch
        from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor as AQT
        from torchao.quantization.linear_activation_quantized_tensor import (
            LinearActivationQuantizedTensor as LAQT,
        )

        aten = torch.ops.aten
        pin_ops = [aten._pin_memory.default, aten.pin_memory.default]
        for cls in (LAQT, AQT):
            table = getattr(cls, "_ATEN_OP_TABLE", {}).get(cls, {})
            if aten.is_pinned.default in table or aten._pin_memory.default in table:
                return False  # this torchao implements them itself: never override its dispatch

        def _payload(t: Any) -> list:
            if isinstance(t, LAQT):
                return _payload(t.original_weight_tensor)
            if isinstance(t, AQT):
                impl = t.tensor_impl
                return [
                    x
                    for x in (getattr(impl, n, None) for n in ("int_data", "scale", "zero_point"))
                    if x is not None
                ]
            return [t]

        def _pinned(t: Any) -> Any:
            if isinstance(t, LAQT):
                return t._apply_fn_to_data(_pinned)
            if isinstance(t, AQT):
                return t._apply_fn_to_data(
                    lambda impl: impl._apply_fn_to_data(lambda x: x.pin_memory())
                )
            return t.pin_memory()

        # v1 lacks tensor_data_names, so diffusers could not restore it; payload stayed on GPU after offload.
        from diffusers.hooks import group_offloading as go

        restore = go._restore_torchao_tensor
        if not getattr(restore, "_unsloth_v1_int8", False):

            def _restore_v1(param: Any, source: Any) -> None:
                if isinstance(source, LAQT):
                    param.original_weight_tensor = source.original_weight_tensor
                elif isinstance(source, AQT):
                    param.tensor_impl = source.tensor_impl
                else:
                    restore(param, source)

            _restore_v1._unsloth_v1_int8 = True
            go._restore_torchao_tensor = _restore_v1

        record = go._record_stream_torchao_tensor
        if not getattr(record, "_unsloth_v1_int8", False):

            def _record_v1(param: Any, stream: Any) -> None:
                if isinstance(param, (LAQT, AQT)):
                    for x in _payload(param):
                        x.record_stream(stream)
                else:
                    record(param, stream)

            _record_v1._unsloth_v1_int8 = True
            go._record_stream_torchao_tensor = _record_v1

        for cls in (LAQT, AQT):

            @cls.implements([aten.is_pinned.default])
            def _is_pinned(func: Any, types: Any, args: Any, kwargs: Any) -> bool:
                return all(bool(x.is_pinned()) for x in _payload(args[0]))

            @cls.implements(pin_ops)
            def _pin(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
                return _pinned(args[0])

        _V1_INT8_PIN_OPS_INSTALLED = True
        return True
    except Exception:  # noqa: BLE001 - no v1 classes / no dispatch table: keep the synchronous fallback
        return False


def _freeze_torchao_weights(module: Any) -> None:
    try:
        for param in module.parameters():
            if type(param).__module__.startswith("torchao") and getattr(
                param, "requires_grad", False
            ):
                param.requires_grad_(False)
    except Exception:  # noqa: BLE001 - an inference-only flag: leave the module as it is
        pass


def _holds_torchao_weights(module: Any) -> bool:
    """Whether any parameter of ``module`` is a torchao tensor subclass (GGUF and native int8 are not)."""
    try:
        for param in module.parameters():
            for tensor in (param, getattr(param, "data", None)):
                if tensor is not None and type(tensor).__module__.startswith("torchao"):
                    return True
    except Exception:  # noqa: BLE001 - an unreadable module is treated as movable, today's behaviour
        return False
    return False


def _pipe_denoisers_hold_torchao(pipe: Any) -> bool:
    return any(
        _holds_torchao_weights(getattr(pipe, name, None))
        for name in ("transformer", "transformer_2", "unconditional_transformer")
        if getattr(pipe, name, None) is not None
    )


def _holds_fp8_stored(module: Any) -> bool:
    """Whether ``module`` holds fp8-STORED Linears; codes and scales are buffers, so a parameter scan reads it dense."""
    if module is None:
        return False
    try:
        from .diffusion_comfy_quant import fp8_stored_layers
        return fp8_stored_layers(module) > 0
    except Exception:  # noqa: BLE001 - unreadable: today's parameter verdict stands
        return False


def _pipe_denoisers_hold_packed(pipe: Any) -> bool:
    """GGUF / int8 / fp8-stored weights dequantize a whole Linear per forward: unmeasured by the dense eager table."""
    try:
        import torch
        for name in ("transformer", "transformer_2", "unconditional_transformer", "unet"):
            if _holds_fp8_stored(getattr(pipe, name, None)):
                return True
            params = getattr(getattr(pipe, name, None), "parameters", None)
            if callable(params) and any(
                type(p).__name__ == "GGUFParameter" or p.dtype in (torch.int8, torch.uint8)
                for p in params()
            ):
                return True
    except Exception:  # noqa: BLE001
        return True
    return False


def plan_fits_total_capacity(plan: Any) -> bool:
    """Separates a free-memory dip from a too-small device; False on any missing input so unknown sizes keep current behaviour."""
    try:
        required = plan.estimates.get("resident_required_mib")
        budget = total_capacity_budget_mib(plan.device_memory)
    except Exception:  # noqa: BLE001 - malformed plan: no retry
        return False
    if required is None or budget is None:
        return False
    return int(required) <= budget


def mps_plan_tight(plan: Any) -> bool:
    """Whether an Apple MPS plan, which stays resident whatever it needs (unified memory), needs more than the 0.85
    share of its budget a discrete plan stays resident in. False on every other device or when a size is unknown."""
    try:
        if plan.device_memory.device != "mps":
            return False
        required = plan.estimates.get("resident_required_mib")
        budget = plan.estimates.get("safe_device_budget_mib")
    except Exception:  # noqa: BLE001 - malformed plan: not tight
        return False
    if required is None or budget is None:
        return False
    return int(required) > int(int(budget) * 0.85)


def total_capacity_budget_mib(memory: DeviceMemory) -> Optional[int]:
    """TOTAL capacity minus the same reserve the free budget takes, times the 0.85 resident margin. None when the
    total is unknown. Shared with the dense prefetch gate so the two cannot disagree."""
    total = memory.total_mib
    if total is None:
        return None
    return int((int(total) - _planner_reserve_mib(memory, int(total))) * 0.85)


# Opt-in escape hatch for the unified-memory refusal below: the shortfall check is an estimate, so an operator who
# believes it is wrong can still attempt the load. Desktop / API callers send ``allow_oversized_load`` per load instead.
UNIFIED_OVERSIZE_ENV = "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD"
UNIFIED_OVERSIZE_REQUEST_FIELD = "allow_oversized_load"
# Must match the Settings switch label and location in frontend unified-memory-load-refusal.ts.
UNIFIED_OVERSIZE_SETTING_LABEL = "Allow oversized model loads"
UNIFIED_OVERSIZE_SETTING_LOCATION = "Settings > System > Model memory"
UNIFIED_OVERSIZE_REFUSAL_CODE = "unified-memory-load"

# Unlike the env var, the per-load flag does NOT lift the MPS allocator cap: that cap is process-wide and set once,
# and keeping it turns an over-commit into an MPS OOM rather than the OS killing the process.
_UNIFIED_OVERSIZE_REQUESTED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_allow_oversized_load", default = False
)


class UnifiedMemoryShortfallError(RuntimeError):
    """A RuntimeError so existing handlers still catch it."""

    code = UNIFIED_OVERSIZE_REFUSAL_CODE


def unified_memory_refusal_code(exc: Optional[BaseException]) -> Optional[str]:
    """``UNIFIED_OVERSIZE_REFUSAL_CODE`` when ``exc`` is, or was raised from, the unified-memory refusal."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, UnifiedMemoryShortfallError):
            return UNIFIED_OVERSIZE_REFUSAL_CODE
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


@contextlib.contextmanager
def unified_oversize_requested(allowed: Any):
    """Per-load ``UNIFIED_OVERSIZE_ENV``; a ContextVar so concurrent loads / plans on other threads are unaffected."""
    token = _UNIFIED_OVERSIZE_REQUESTED.set(bool(allowed))
    try:
        yield
    finally:
        _UNIFIED_OVERSIZE_REQUESTED.reset(token)


def unified_oversize_scoped(allowed: Any, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def _scoped(*args: Any, **kwargs: Any) -> Any:
        with unified_oversize_requested(allowed):
            return fn(*args, **kwargs)

    return _scoped


def scoped_allow_oversized_load(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Kwarg is read, not popped, so ``fn`` keeps it in its signature; nested calls keep an outer True."""

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        allowed = (
            bool(kwargs.get(UNIFIED_OVERSIZE_REQUEST_FIELD)) or _UNIFIED_OVERSIZE_REQUESTED.get()
        )
        with unified_oversize_requested(allowed):
            return fn(*args, **kwargs)

    return _wrapper


def _unified_oversize_env() -> bool:
    return os.environ.get(UNIFIED_OVERSIZE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _unified_oversize_override() -> bool:
    return _UNIFIED_OVERSIZE_REQUESTED.get() or _unified_oversize_env()


def _unified_oversize_source() -> str:
    if _UNIFIED_OVERSIZE_REQUESTED.get():
        return UNIFIED_OVERSIZE_REQUEST_FIELD
    return f"{UNIFIED_OVERSIZE_ENV}=1"


def unified_capacity_memory(memory: DeviceMemory) -> DeviceMemory:
    """``memory`` at CAPACITY (free = total), so the verdict does not depend on what is loaded right now.

    On mps the snapshot picks the RAM or working-set view from the CURRENT free reading; re-read RAM here so the
    same pick is not judged against the smaller working set just because something is resident."""
    capacity = replace(memory, free_mib = memory.total_mib)
    if getattr(memory, "device", None) != "mps" or memory.metal_working_set_mib is None:
        return capacity
    try:
        total, _free = _system_memory_mib()
    except Exception:  # noqa: BLE001 - no RAM reading: keep the working-set view
        return capacity
    if total is None:
        return capacity
    ram = DeviceMemory(memory.backend, memory.device, "unified_memory", int(total), int(total))
    ram_budget = _safe_device_budget_mib(ram)
    capacity_budget = _safe_device_budget_mib(capacity)
    if ram_budget is not None and (capacity_budget is None or ram_budget > capacity_budget):
        return ram
    return capacity


def _unified_memory_shortfall_text(plan: Any, family: Optional[str]) -> Optional[str]:
    """Ignores the overrides."""
    try:
        memory = plan.device_memory
        # system_memory excluded: the CPU path has swap and is not Metal-killed.
        if getattr(memory, "memory_kind", None) != "unified_memory":
            return None
        estimates = plan.estimates
        budget = estimates.get("safe_device_budget_mib")
        weights = estimates.get("model_dense_mib")
        overhead = estimates.get("base_overhead_mib")
        companions = estimates.get("companion_dense_mib")
        free = getattr(memory, "free_mib", None)
    except Exception:  # noqa: BLE001 - malformed plan: never block the load
        return None
    if budget is None or weights is None:
        return None
    required = int(weights) + int(overhead or 0)
    if required <= int(budget):
        return None
    what = f"'{family}'" if family else "This model"
    if free is None:
        free_note = "after reserving room for the operating system"
    elif getattr(memory, "metal_working_set_mib", None) is not None:
        free_note = (
            f"of the {int(free) / 1024:.0f} GB the GPU can use right now, after reserving room "
            "for other applications"
        )
    else:
        free_note = (
            f"of the {int(free) / 1024:.0f} GB currently free, after reserving room for the "
            "operating system"
        )
    # The text encoders and VAE load dense whatever quant the main model is, so past this floor no quant of it fits.
    floor = int(companions or 0) + int(overhead or 0)
    if companions and floor > int(budget):
        advice = (
            f"Its text encoders and VAE alone need about {floor / 1024:.0f} GB, so a lower quant of "
            "the main model will not fit either. Use a smaller model, or free memory by closing "
            "other applications."
        )
    elif companions and int(companions) < int(weights) and int(budget) - floor >= 1024:
        room = int(budget) - floor
        advice = (
            f"Its text encoders and VAE need about {floor / 1024:.0f} GB, which leaves room for a "
            f"main model of up to about {room / 1024:.0f} GB: use a lower GGUF quant (or another "
            "quantized build) of this model at or below that size, or free memory by closing "
            "other applications."
        )
    else:
        advice = (
            "Use a smaller or more quantized model (for example a lower GGUF quant), or free memory "
            "by closing other applications."
        )
    # One decimal when whole GB would print the same number twice ("needs about 14 GB, only about 14 GB usable").
    digits = 1 if f"{required / 1024:.0f}" == f"{int(budget) / 1024:.0f}" else 0
    return (
        f"{what} needs about {required / 1024:.{digits}f} GB of memory for its weights, but only "
        f"about {int(budget) / 1024:.{digits}f} GB is usable on this device ({free_note}). This device "
        "has unified memory, so the CPU and GPU share one pool: offloading weights to the CPU "
        "frees nothing, and the operating system stops an oversized load outright instead of "
        f"reporting an out-of-memory error. {advice} The size is an estimate. To try anyway, "
        f"choose Load anyway, or turn on '{UNIFIED_OVERSIZE_SETTING_LABEL}' in "
        f"{UNIFIED_OVERSIZE_SETTING_LOCATION} (API callers can send {UNIFIED_OVERSIZE_REQUEST_FIELD}; "
        f"server installs can set {UNIFIED_OVERSIZE_ENV}=1). An oversized load can exhaust memory, "
        "swap heavily, stop Unsloth or make the computer unresponsive."
    )


def unified_memory_shortfall_message(plan: Any, *, family: Optional[str] = None) -> Optional[str]:
    """On UNIFIED device memory, a user-facing refusal when the WEIGHTS alone cannot fit the safe
    budget (else None).

    Unified memory is the one placement with no fallback left. On discrete VRAM an oversized model
    still loads, degrading to group / whole-module CPU offload and streaming from host RAM. On Apple
    Silicon (and integrated CUDA) the CPU and GPU share one pool, so offload moves bytes within that
    pool and frees nothing: ``plan_diffusion_memory`` correctly returns ``none`` and the load then
    allocates past physical memory. ``apply_mps_memory_cap`` bounds MPS allocations just below RAM,
    but that only turns the overrun into a late OOM after minutes of loading (and integrated CUDA has
    no such cap), so refusing up front is still the better failure.

    Weights only (``model_dense_mib`` plus the flat base overhead): the per-call runtime headroom is
    a coarse activation / VAE-decode estimate and the mps path already turns on VAE tiling and
    slicing, so counting it would refuse marginal loads that would in fact complete. The weights are
    unavoidable resident bytes, and if they alone do not fit then nothing at generation time can
    rescue the load. Same reasoning as the llama.cpp APU guard.

    None under the env var or a per-load ``allow_oversized_load``, as the loaders' speculative probes also see.

    Fail-open on anything unknown (no budget, no size), matching the planner's own "budget or model
    size unknown; staying resident".
    """
    if _unified_oversize_override():
        return None
    return _unified_memory_shortfall_text(plan, family)


def raise_on_unified_memory_shortfall(
    plan: Any,
    *,
    family: Optional[str] = None,
    logger: Any = None,
) -> None:
    """Refuse a load whose weights cannot fit unified device memory. No-op on every other
    placement, so the discrete-VRAM path is untouched.

    Lives outside ``plan_diffusion_memory`` on purpose: the planner is a pure sizing function
    that both loaders call SPECULATIVELY (the image loader re-plans candidate quantisations, and
    both re-plan against a settled snapshot), and a planner that raised would turn those probes
    into load failures instead of letting a smaller candidate win. Call this once, on the plan
    the loader has committed to, after the previous pipeline has been evicted so the free
    reading is the memory the load actually gets. Under an override the
    load proceeds with a warning."""
    message = _unified_memory_shortfall_text(plan, family)
    if message is None:
        return
    if _unified_oversize_override():
        if logger is not None:
            logger.warning(
                "diffusion.memory: loading past the unified-memory estimate (%s): %s",
                _unified_oversize_source(),
                message,
            )
        return
    if logger is not None:
        logger.error("diffusion.memory: refusing oversized unified-memory load: %s", message)
    raise UnifiedMemoryShortfallError(message)


def _sum_required(*values: Optional[int]) -> Optional[int]:
    total = 0
    for value in values:
        if value is None:
            return None
        total += int(value)
    return total


def plan_diffusion_memory(
    *,
    target: Any,
    device_memory: DeviceMemory,
    model_dense_mib: Optional[int],
    runtime_headroom_mib: int,
    companion_dense_mib: Optional[int] = None,
    text_encoder_dense_mib: Optional[int] = None,
    base_overhead_mib: int = DEFAULT_BASE_OVERHEAD_MIB,
    requested_mode: Optional[str] = None,
    explicit_offload: bool = False,
    calibrated_activation: Optional[CalibratedImageActivation] = None,
) -> MemoryPlan:
    """Encoders run once before step 0, so streaming them adds one transfer pass per call, not per step."""
    mode = normalize_memory_mode(requested_mode) or MEMORY_MODE_AUTO
    can_offload = bool(getattr(target, "supports_model_cpu_offload", False))
    budget = _safe_device_budget_mib(device_memory)
    required = _sum_required(model_dense_mib, runtime_headroom_mib, base_overhead_mib)
    # The resident floor under group offload: companions stay, the transformer streams.
    group_floor = _sum_required(companion_dense_mib, runtime_headroom_mib, base_overhead_mib)
    # Second floor with text encoders streamed too; computed only when both terms are known, and clamped at 0 since they can differ in source.
    group_floor_streamed_te = (
        _sum_required(
            max(0, int(companion_dense_mib) - int(text_encoder_dense_mib)),
            runtime_headroom_mib,
            base_overhead_mib,
        )
        if companion_dense_mib is not None and text_encoder_dense_mib is not None
        else None
    )
    # Floor for a resident transformer with streamed text encoders (they run once per call, not per step).
    resident_transformer_floor = (
        _sum_required(
            max(0, int(model_dense_mib) - int(text_encoder_dense_mib)),
            runtime_headroom_mib,
            base_overhead_mib,
        )
        if model_dense_mib is not None
        and text_encoder_dense_mib is not None
        and int(text_encoder_dense_mib) > 0
        else None
    )
    reasons: list[str] = []
    stream_text_encoders = False
    stream_transformer = True
    estimates: dict[str, Optional[int]] = {
        "safe_device_budget_mib": budget,
        "model_dense_mib": model_dense_mib,
        "companion_dense_mib": companion_dense_mib,
        "text_encoder_dense_mib": text_encoder_dense_mib,
        "runtime_headroom_mib": runtime_headroom_mib,
        "base_overhead_mib": base_overhead_mib,
        "resident_required_mib": required,
        "group_floor_mib": group_floor,
        "group_floor_streamed_te_mib": group_floor_streamed_te,
        "resident_transformer_floor_mib": resident_transformer_floor,
    }

    def _group_fits() -> bool:
        # Group offload only helps if the resident companions fit; a too-big text encoder needs whole-module offload.
        return group_floor is not None and budget is not None and group_floor <= budget

    def _group_fits_streamed_te() -> bool:
        return (
            group_floor_streamed_te is not None
            and budget is not None
            and group_floor_streamed_te <= budget
        )

    def _resident_transformer_fits() -> bool:
        return (
            resident_transformer_floor is not None
            and budget is not None
            and resident_transformer_floor <= budget
        )

    # Speed order: group, group + streamed encoders, then whole-module offload (per-step paging).
    def _offload_tier() -> tuple[str, bool, bool]:
        if _resident_transformer_fits():
            return OFFLOAD_GROUP, True, False
        if _group_fits():
            return OFFLOAD_GROUP, False, True
        if _group_fits_streamed_te():
            return OFFLOAD_GROUP, True, True
        return OFFLOAD_MODEL, False, True

    _STREAMED_TE_REASON = (
        "companions exceed budget, but they fit with the text encoders streamed too "
        "(they run once, before step 0); streaming them beats paging every component per step"
    )
    _RESIDENT_TRANSFORMER_REASON = (
        "the transformer fits resident once the text encoders are streamed (they run once, "
        "before step 0); every denoise step runs at resident speed"
    )

    if not can_offload or device_memory.is_unified:
        # No separate device to stream to on MPS/CPU; unified memory offload just shuffles bytes in one pool.
        policy = OFFLOAD_NONE
        if device_memory.is_unified:
            reasons.append("unified/system memory: CPU offload frees no device memory")
        else:
            reasons.append(f"{device_memory.backend}: CPU offload unavailable; staying resident")
    elif mode == MEMORY_MODE_FAST:
        policy = OFFLOAD_NONE
        fast_budget = _fast_device_budget_mib(device_memory)
        estimates["resident_budget_mib"] = fast_budget
        fast_required = required
        if calibrated_activation is not None and calibrated_activation.resident_only:
            # the flat 8 GiB guess demoted an explicit fast SDXL on 15-16 GB cards, where it fits with room to spare
            fast_required = _sum_required(
                model_dense_mib, calibrated_activation.headroom(False), base_overhead_mib
            )
            estimates["calibrated_headroom_mib"] = calibrated_activation.headroom(False)
        if fast_budget is not None and fast_required is not None and fast_required > fast_budget:
            policy, stream_text_encoders, stream_transformer = _offload_tier()
            reasons.append(
                f"fast requested but weights do not fit resident ({fast_required} MiB needed, "
                f"{fast_budget} MiB free after the reserve); offloading"
            )
            if not stream_transformer:
                reasons.append(_RESIDENT_TRANSFORMER_REASON)
            elif stream_text_encoders:
                reasons.append(_STREAMED_TE_REASON)
        else:
            reasons.append("fast requested; weights resident on device")
    elif mode == MEMORY_MODE_BALANCED:
        policy = OFFLOAD_GROUP
        reasons.append("balanced requested; streamed block-level transformer offload")
    elif mode == MEMORY_MODE_LOW_VRAM:
        policy = OFFLOAD_MODEL
        reasons.append("low_vram requested; whole-module offload of every component")
    elif budget is None or required is None:
        policy = OFFLOAD_NONE
        reasons.append("device budget or model size unknown; staying resident")
    elif required <= int(budget * 0.85):
        policy = OFFLOAD_NONE
        reasons.append("weights fit resident with headroom")
    elif _resident_transformer_fits():
        policy = OFFLOAD_GROUP
        stream_text_encoders = True
        stream_transformer = False
        reasons.append(_RESIDENT_TRANSFORMER_REASON)
    elif _group_fits():
        policy = OFFLOAD_GROUP
        reasons.append("tight fit; stream the transformer, companions resident")
    elif _group_fits_streamed_te():
        policy = OFFLOAD_GROUP
        stream_text_encoders = True
        reasons.append(_STREAMED_TE_REASON)
    else:
        policy = OFFLOAD_MODEL
        # An UNKNOWN companion split fails the group tiers too; don't report it as 'exceeds budget'.
        if group_floor is None:
            reasons.append(
                "companion size unknown, so no streamed tier can be sized; "
                "whole-module offload of every component"
            )
        else:
            reasons.append("companions exceed budget; whole-module offload of every component")

    # The legacy cpu_offload flag applies only when no memory_mode was supplied, so an explicit `fast` stays resident.
    if explicit_offload and normalize_memory_mode(requested_mode) is None:
        estimates["explicit_cpu_offload"] = 1
    if (
        explicit_offload
        and normalize_memory_mode(requested_mode) is None
        and policy == OFFLOAD_NONE
        and can_offload
        and not device_memory.is_unified
    ):
        policy = OFFLOAD_MODEL
        reasons.append("explicit cpu_offload overrides resident placement")

    # VAE slicing is exact on any tier; tiling is bit-identical only for one tile (<= 1 MP).
    any_offload = policy != OFFLOAD_NONE or device_memory.backend in ("mps", "cpu")
    tile = policy in (OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL) or device_memory.backend in ("mps", "cpu")
    if (
        calibrated_activation is not None
        and mode == MEMORY_MODE_AUTO
        and not (explicit_offload and normalize_memory_mode(requested_mode) is None)
        and can_offload
        and not device_memory.is_unified
    ):
        faster = _calibrated_faster_tier(
            calibrated_activation,
            budget = budget,
            model_dense_mib = model_dense_mib,
            companion_dense_mib = companion_dense_mib,
            text_encoder_dense_mib = text_encoder_dense_mib,
            base_overhead_mib = base_overhead_mib,
            free_mib = (
                None
                if device_memory.free_mib is None
                else int(device_memory.free_mib)
                - _device_user_budget_reserve_mib(
                    device_memory, device_memory.total_mib or device_memory.free_mib
                )
            ),
            policy = policy,
            stream_transformer = stream_transformer,
        )
        if faster is not None:
            policy, stream_text_encoders, stream_transformer, tile, reason = faster
            any_offload = policy != OFFLOAD_NONE
            reasons.append(reason)
            estimates["calibrated_headroom_mib"] = calibrated_activation.headroom(tile)
    if calibrated_activation is not None and "calibrated_headroom_mib" not in estimates:
        from .diffusion_multigpu import current_selection
        if current_selection() is not None:
            # Untiled: a layer split keeps the VAE resident on the primary.
            estimates["calibrated_split_headroom_mib"] = calibrated_activation.headroom(False)
    split = _multigpu_companion_split(
        target = target,
        device_memory = device_memory,
        mode = mode,
        requested_mode = requested_mode,
        explicit_offload = explicit_offload,
        can_offload = can_offload,
        policy = policy,
        budget = budget,
        model_dense_mib = model_dense_mib,
        companion_dense_mib = companion_dense_mib,
        runtime_headroom_mib = runtime_headroom_mib,
        base_overhead_mib = base_overhead_mib,
        text_encoder_dense_mib = text_encoder_dense_mib,
        calibrated_activation = calibrated_activation,
    )
    single_card = MemoryPlan(
        requested_mode = mode,
        offload_policy = policy,
        vae_tiling = tile,
        vae_slicing = any_offload,
        device_memory = device_memory,
        estimates = dict(estimates),
        reasons = tuple(reasons),
        # only ever meaningful under group offload; every other tier already places the encoders
        stream_text_encoders = stream_text_encoders and policy == OFFLOAD_GROUP,
        stream_transformer = stream_transformer or policy != OFFLOAD_GROUP,
    )
    if split is not None:
        estimates["multigpu_primary_required_mib"] = split.primary_required_mib
        estimates["multigpu_secondary_required_mib"] = split.secondary_required_mib
        return MemoryPlan(
            requested_mode = mode,
            offload_policy = OFFLOAD_NONE,
            vae_tiling = False,
            vae_slicing = False,
            device_memory = device_memory,
            estimates = estimates,
            reasons = tuple(reasons) + (split.reason,),
            secondary_ordinal = split.secondary_ordinal,
            secondary_components = split.components,
            fallback_plan = single_card,
        )
    return single_card


def _multigpu_companion_split(
    *,
    target: Any,
    device_memory: DeviceMemory,
    mode: str,
    requested_mode: Optional[str],
    explicit_offload: bool,
    can_offload: bool,
    policy: str,
    budget: Optional[int],
    model_dense_mib: Optional[int],
    companion_dense_mib: Optional[int],
    runtime_headroom_mib: int,
    base_overhead_mib: int,
    text_encoder_dense_mib: Optional[int] = None,
    calibrated_activation: Optional[CalibratedImageActivation] = None,
) -> Any:
    """The multi-GPU companion split for this plan, or None (one card: every existing decision unchanged). Only for
    ``auto`` / ``fast`` on a discrete CUDA card that is the primary of a multi-card selection; explicit offload modes
    and the legacy ``cpu_offload`` flag keep their meaning."""
    from .diffusion_multigpu import (
        current_selection,
        plan_companion_split,
        secondary_resident_limit_mib,
    )

    selection = current_selection()
    if selection is None or not can_offload or device_memory.is_unified:
        return None
    if getattr(device_memory, "device", None) != "cuda" or budget is None:
        return None
    ordinal = getattr(target, "ordinal", None)
    if ordinal is None and selection.automatic:
        from .diffusion_device import placed_cuda_ordinal
        ordinal = placed_cuda_ordinal(target)
    if ordinal != selection.primary:
        return None
    if mode not in (MEMORY_MODE_AUTO, MEMORY_MODE_FAST):
        return None
    if explicit_offload and normalize_memory_mode(requested_mode) is None:
        return None
    if mode == MEMORY_MODE_FAST:
        limit = _fast_device_budget_mib(device_memory)
    else:
        limit = int(budget * 0.85)
    return plan_companion_split(
        single_card_policy = policy,
        model_dense_mib = model_dense_mib,
        companion_dense_mib = companion_dense_mib,
        runtime_headroom_mib = runtime_headroom_mib,
        base_overhead_mib = base_overhead_mib,
        primary_limit_mib = limit,
        secondary_limit_mib = lambda ordinal: secondary_resident_limit_mib(ordinal, mode),
        selection = selection,
        text_encoder_dense_mib = text_encoder_dense_mib,
        phase_headroom_mib = (
            (
                calibrated_activation.text_encoder_mib,
                calibrated_activation.denoise_mib,
                calibrated_activation.decode_mib,
            )
            if calibrated_activation is not None
            else None
        ),
    )


def _calibrated_faster_tier(
    act: CalibratedImageActivation,
    *,
    budget: Optional[int],
    model_dense_mib: Optional[int],
    companion_dense_mib: Optional[int],
    text_encoder_dense_mib: Optional[int],
    base_overhead_mib: int,
    free_mib: Optional[int],
    policy: str,
    stream_transformer: bool,
) -> Optional[tuple[str, bool, bool, bool, str]]:
    """Strictly faster tier than the flat pick, else None; resident tiers must fit the 2048 denoise (it cannot tile).
    A flat all-resident pick whose measured 2048 denoise does not fit is demoted to the fastest tier that does:
    nothing it holds has offload hooks a request could release. A resident-transformer pick (encoders streamed) is
    never demoted: like ComfyUI, which sizes residency per call, the generation guard handles an oversized request.
    Every check grows with the budget, so more VRAM never picks a slower tier."""
    if budget is None or free_mib is None or model_dense_mib is None or companion_dense_mib is None:
        return None
    te = max(0, int(text_encoder_dense_mib or 0))
    transformer = max(0, int(model_dense_mib) - int(companion_dense_mib))
    others = max(0, int(companion_dense_mib) - te)
    overhead = max(0, int(base_overhead_mib))
    budget = int(budget)
    free = int(free_mib) - overhead
    room = free - act.max_canvas_mib
    if act.resident_only:
        # Larger requests are still checked by the generation guard; an oversized decode is tiled by its OOM retry.
        weights = transformer + te + others
        if (
            policy != OFFLOAD_NONE
            and weights + act.headroom(False) + overhead <= budget
            and weights <= room
        ):
            return (
                OFFLOAD_NONE,
                False,
                True,
                False,
                "measured activations fit resident; the U-Net's only other tier uploads every component per call",
            )
        return None
    if policy == OFFLOAD_NONE:
        flat_rank = 0
    elif policy == OFFLOAD_GROUP and not stream_transformer:
        flat_rank = 1
    elif policy == OFFLOAD_MODEL:
        flat_rank = 4
    elif policy == OFFLOAD_GROUP:
        flat_rank = 5
    else:
        return None
    model_viable = max(transformer, te, others) <= budget and (
        flat_rank < 5
        or (
            transformer <= room
            and te + act.text_encoder_mib <= free
            and others + act.tiled_decode_mib <= free
        )
    )
    resident_streamed_te = te > 0 and transformer + others <= room
    candidates = (
        (
            transformer + te + others + act.headroom(False) + overhead <= int(budget * 0.85)
            and transformer + te + others <= room,
            (OFFLOAD_NONE, False, True, False, "measured activations fit resident with headroom"),
        ),
        (
            resident_streamed_te
            and transformer + others + act.headroom(False) + overhead <= budget,
            (
                OFFLOAD_GROUP,
                True,
                False,
                False,
                "measured activations keep the transformer resident with the text encoders streamed",
            ),
        ),
        (
            resident_streamed_te and transformer + others + act.headroom(True) + overhead <= budget,
            (
                OFFLOAD_GROUP,
                True,
                False,
                True,
                "measured activations keep the transformer resident with the text encoders "
                "streamed and the VAE decode tiled",
            ),
        ),
        (
            model_viable and others + act.decode_mib + overhead <= budget,
            (
                OFFLOAD_MODEL,
                False,
                True,
                False,
                "whole-module offload uploads the transformer once per call, and the measured "
                "VAE decode fits untiled",
            ),
        ),
        (
            model_viable,
            (
                OFFLOAD_MODEL,
                False,
                True,
                True,
                "whole-module offload uploads the transformer once per call instead of every step",
            ),
        ),
    )
    # Only the 2048 denoise is checked here: it cannot tile, while an oversized decode is tiled by the generation guard.
    if flat_rank == 0 and transformer + te + others > room:
        for fits, (tier_policy, stream_te, stream_tr, tile, reason) in candidates[flat_rank + 1 :]:
            if fits:
                return (
                    tier_policy,
                    stream_te,
                    stream_tr,
                    tile,
                    "the measured 2048 denoise does not fit beside the flat resident pick; "
                    + reason,
                )
        return None
    for rank, (fits, tier) in enumerate(candidates):
        if rank >= flat_rank:
            return None
        if fits:
            return tier
    return None


def _streamable_components(pipe: Any, torch: Any) -> dict[str, tuple[Any, str]]:
    """Only denoisers and text encoders can be streamed; the VAE and other components stay resident."""
    streamed: dict[str, tuple[Any, str]] = {}
    for name in ("transformer", "transformer_2", "unconditional_transformer"):
        module = getattr(pipe, name, None)
        if isinstance(module, torch.nn.Module):
            streamed[name] = (module, "block_level")
    for name, module in getattr(pipe, "components", {}).items():
        if str(name).startswith("text_encoder") and isinstance(module, torch.nn.Module):
            streamed[str(name)] = (module, "leaf_level")
    return streamed


def _module_storage_bytes(module: Any, seen: set[int]) -> int:
    storage_bytes = 0
    for tensor in list(module.parameters(recurse = True)) + list(module.buffers(recurse = True)):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        storage_bytes += int(tensor.numel()) * int(tensor.element_size())
    return storage_bytes


def largest_streamable_companion_mib(pipe: Any) -> Optional[int]:
    """MiB of the largest text encoder refinement could stream, as loaded, or None."""
    try:
        import torch
        mib = 1024 * 1024
        sizes = [
            (_module_storage_bytes(module, set()) + mib - 1) // mib
            for name, (module, offload_type) in _streamable_components(pipe, torch).items()
            if offload_type == "leaf_level"
        ]
    except Exception:  # noqa: BLE001 - a sizing aid; the caller treats unknown as unmeasured
        return None
    return max(sizes) if sizes else None


BALANCED_FIT_CHECK_ENV = "UNSLOTH_DIFFUSION_BALANCED_FIT_CHECK"


def _balanced_fit_check_enabled() -> bool:
    return str(os.environ.get(BALANCED_FIT_CHECK_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def refine_balanced_plan_for_components(pipe: Any, plan: MemoryPlan) -> MemoryPlan:
    """Fit-checks balanced plans on the loaded encoders, since pre-load estimates can be the dense size."""
    if not _balanced_fit_check_enabled():
        return plan
    if getattr(plan, "requested_mode", None) != MEMORY_MODE_BALANCED:
        return plan
    if plan.offload_policy != OFFLOAD_GROUP or not bool(getattr(plan, "stream_transformer", True)):
        return plan
    estimates = dict(plan.estimates)
    budget = estimates.get("safe_device_budget_mib")
    if budget is None:
        return plan
    try:
        import torch

        components = getattr(pipe, "components", {})
        if not isinstance(components, dict):
            return plan
        streamed = _streamable_components(pipe, torch)
        mib = 1024 * 1024
        resident = 0
        encoders = 0
        for name, component in components.items():
            if not isinstance(component, torch.nn.Module):
                continue
            if name in streamed and streamed[name][1] == "block_level":
                continue  # the denoisers already stream under group offload
            size = (_module_storage_bytes(component, set()) + mib - 1) // mib
            resident += size
            if name in streamed:
                encoders += size
        overhead = int(estimates.get("runtime_headroom_mib") or 0) + int(
            estimates.get("base_overhead_mib") or 0
        )
    except Exception:  # noqa: BLE001 - a sizing aid; keep the plan the user asked for
        return plan

    estimates["balanced_resident_companions_mib"] = resident
    if resident + overhead <= int(budget):
        return plan
    if encoders > 0 and bool(getattr(plan, "stream_text_encoders", False)) is False:
        if (resident - encoders) + overhead <= int(budget):
            return replace(
                plan,
                stream_text_encoders = True,
                estimates = estimates,
                reasons = plan.reasons
                + (
                    f"balanced: the loaded companions ({resident} MiB) do not fit the {int(budget)} MiB budget "
                    "beside the runtime headroom; streaming the text encoders too (they run once, before step 0)",
                ),
            )
    torchao = _pipe_denoisers_hold_torchao(pipe)
    return replace(
        plan,
        offload_policy = OFFLOAD_STREAMING if torchao else OFFLOAD_MODEL,
        stream_text_encoders = False,
        vae_tiling = True,
        vae_slicing = True,
        estimates = estimates,
        reasons = plan.reasons
        + (
            f"balanced: the loaded companions ({resident} MiB) do not fit the {int(budget)} MiB budget even with "
            "the text encoders streamed; "
            + (
                "streaming transformer blocks and text-encoder layers"
                if torchao
                else "whole-module offload of every component"
            ),
        ),
    )


def refine_memory_plan_for_components(pipe: Any, plan: MemoryPlan) -> MemoryPlan:
    """Switches to granular streaming only when a loaded component cannot fit; keeps whole-module offload if all fit, and checks streamed parts fit together."""
    if plan.offload_policy != OFFLOAD_MODEL:
        return plan
    budget = plan.estimates.get("safe_device_budget_mib")
    if budget is None or int(budget) <= 0:
        return plan

    try:
        import torch

        components = getattr(pipe, "components", {})
        transformer = getattr(pipe, "transformer", None)
        if not isinstance(components, dict) or not isinstance(transformer, torch.nn.Module):
            return plan
        # Streaming cannot move torchao weights; the loader already checked fit (torchao numel reads bf16-sized here).
        if _pipe_denoisers_hold_torchao(pipe):
            return plan
        streamable = _streamable_components(pipe, torch)

        sizes: dict[str, int] = {}
        mib = 1024 * 1024
        for name, component in components.items():
            if not isinstance(component, torch.nn.Module):
                continue
            sizes[str(name)] = (_module_storage_bytes(component, set()) + mib - 1) // mib
    except Exception:  # noqa: BLE001 - runtime measurement is an optional refinement
        return plan

    streamed_sizes = {n: m for n, m in sizes.items() if n in streamable}
    if not streamed_sizes:
        return plan
    largest_name, largest_mib = max(streamed_sizes.items(), key = lambda item: item[1])
    if largest_mib <= int(budget):
        return plan
    # what streaming leaves resident, all at once: over budget here means streaming OOMs too
    resident_mib = sum(m for n, m in sizes.items() if n not in streamable)
    if resident_mib > int(budget):
        return plan

    estimates = dict(plan.estimates)
    estimates["largest_component_mib"] = largest_mib
    estimates["streaming_resident_mib"] = resident_mib
    return replace(
        plan,
        offload_policy = OFFLOAD_STREAMING,
        estimates = estimates,
        reasons = plan.reasons
        + (
            f"loaded {largest_name} is {largest_mib} MiB, above the {int(budget)} MiB "
            "device budget; streaming transformer blocks and text-encoder layers",
        ),
    )


# Flat planner's 8192 MiB/MP activation reserve overstates peaks, so it streams models that fit.
MEASURED_ACTIVATION_ENV = "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION"
PARTIAL_RESIDENT_ENV = "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT"

# Worst measured CUDA MiB above the resident weights, one 1024x1024 image, encoder + every step + VAE decode, torchao
# int8 / fp8 denoisers on the compiled tiers (Qwen-Image-2.1: encoder 1849 streamed, denoise 1442, decode 1730).
# FLUX.1 / Z-Image: their worst phase is the untiled 16-channel VAE decode (_MEASURED_IMAGE_ACTIVATION_MIB).
# MOSS-Image: 2905 MiB worst-phase peak fully streamed (B200 capped at 16 GiB).
# Qwen-Image-Edit-2511: 6046 MiB whole render (1024^2, one 1024^2 source, 40 steps, true CFG 3), int8 DiT + encoder.
_MEASURED_IMAGE_PEAK_MIB: dict[str, int] = {
    "qwen-image-2.1": 1849,
    "flux.1": 2666,
    "z-image": 2666,
    "moss-image": 2976,
    "qwen-image-edit": 6046,
}
# "0": Qwen-Image-Edit places from its learned peak only.
MEASURED_ACTIVATION_QWEN_EDIT_ENV = "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_QWEN_EDIT"
_MEASURED_PEAK_SPEED_MODES = ("default", "max")
_MEASURED_PEAK_MARGIN = 1.15
_MEASURED_PEAK_ROUND_MIB = 256

# Dense denoisers, eager tier: worst CUDA MiB above resident weights, fp16, 1024x1024, encode + steps + VAE decode.
# family -> (peak MiB, largest loaded DiT MiB it covers; a bigger DiT keeps the flat estimate)
MEASURED_ACTIVATION_DENSE_ENV = "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE"
_MEASURED_DENSE_EAGER_PEAK_MIB: dict[str, tuple[int, int]] = {
    "flux.2-klein": (2455, 7800),
    # FLUX.3 Klein: worst phase is the untiled VAE decode (2_437 MiB @1024^2, bf16); DiT 10_583 MiB bf16.
    "flux.3-klein": (2437, 10600),
    "flux.1": (2448, 23800),
    "qwen-image": (4248, 40900),
}
_MEASURED_DENSE_SPEED_MODES = ("off",)
# Dense fp16 / fp32 denoisers (pre-Ampere cards: T4, V100, Turing) on the default and max tiers, seeded from the
# family's measured peaks instead of the flat 8 GiB per megapixel estimate. "0" restores the flat estimate.
MEASURED_ACTIVATION_DENSE_TIERS_ENV = "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE_TIERS"


# Dense bf16, eager tier: worst recorded whole-render peak at 1024x1024, all groups streamed (what a learned peak holds).
# family -> (peak MiB, largest loaded DiT MiB it covers). "0" = learned peak alone.
MEASURED_ACTIVATION_BF16_ENV = "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_BF16"
_MEASURED_DENSE_BF16_EAGER_PEAK_MIB: dict[str, tuple[int, int]] = {
    # FLUX.1-schnell, Colab L4, 4 steps: 3417 (4 loads), 3581 (1 load).
    "flux.1": (3581, 23800),
    # Not seeded: FLUX.2-klein-4B's 2446 used a tiled decode; Qwen-Image-2.1's 5486 was recorded resident, unstreamed.
}


# Small-host int8 DiT, eager: (peak MiB above resident weights at 1024^2, largest int8 DiT MiB covered).
# FLUX.2-klein-4B, Kaggle T4 fp16, 4 steps, sliced VAE. "0" = learned peak alone.
MEASURED_ACTIVATION_SMALL_HOST_ENV = "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_SMALL_HOST"
_MEASURED_SMALL_HOST_INT8_EAGER_PEAK_MIB: dict[str, tuple[int, int]] = {
    "flux.2-klein": (2805, 3800),
}


def _env_off(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("0", "off", "false", "no")


def _small_host_int8_seed_mib(
    family: Optional[str], speed_mode: Optional[str], int8_dit_mib: int, widen: int
) -> Optional[int]:
    """Seed MiB for a small-host int8 denoiser (``widen`` = the compute element size), or None."""
    if _env_off(MEASURED_ACTIVATION_SMALL_HOST_ENV):
        return None
    entry = _MEASURED_SMALL_HOST_INT8_EAGER_PEAK_MIB.get(str(family or "").lower())
    if entry is None or str(speed_mode or "") not in _MEASURED_DENSE_SPEED_MODES:
        return None
    if not 0 < int(int8_dit_mib) <= entry[1]:
        return None
    return entry[0] * max(2, int(widen)) // 2


def pipe_small_host_int8(pipe: Any) -> bool:
    """Whether the small-host route stored this pipe's denoiser as int8 weights (diffusion_small_host.mark)."""
    info = getattr(pipe, "_unsloth_small_host", None)
    comps = info.get("components") if isinstance(info, dict) else None
    if not isinstance(comps, dict):
        return False
    return any(
        str(comps.get(name, "")).startswith("int8 weights")
        for name in ("transformer", "transformer_2", "unconditional_transformer")
    )


def dense_bf16_seeded(family: Optional[str]) -> bool:
    return not _env_off(MEASURED_ACTIVATION_BF16_ENV) and (
        str(family or "").lower() in _MEASURED_DENSE_BF16_EAGER_PEAK_MIB
    )


def _dense_seed_mib(
    family: str,
    speed_mode: str,
    dense_mib: int,
    widen: int,
    tier_seeds: bool,
    bf16: bool = False,
) -> Optional[int]:
    """Seed MiB for a dense fp16 / fp32 denoiser (``widen`` = its element size), or None. Eager: the dense table. Default
    / max: the larger of the dense table and the tier's worst phase (encoder, denoise, untiled decode), only for a
    family the dense table measured at this DiT size. Families it lacks keep the flat estimate: on a T4, Z-Image's 11.5
    GiB DiT would then be kept partly resident, which ran slower per step than streaming it whole (3.8-4.4 vs 3.2-3.6
    s/step). ``tier_seeds``: the target is where the phase table was measured (NVIDIA with a sub-quadratic attention
    kernel, ``diffusion._measured_target``); elsewhere (math-only attention, ROCm) the tiers keep the flat estimate.
    ``bf16``: the bf16 table, eager tier and ``tier_seeds`` targets only (all it was measured on)."""
    if bf16:
        entry = _MEASURED_DENSE_BF16_EAGER_PEAK_MIB.get(family)
        if (
            entry is None
            or not tier_seeds
            or _env_off(MEASURED_ACTIVATION_BF16_ENV)
            or speed_mode not in _MEASURED_DENSE_SPEED_MODES
            or not 0 < dense_mib <= entry[1]
        ):
            return None
        return entry[0]
    entry = _MEASURED_DENSE_EAGER_PEAK_MIB.get(family)
    if entry is None or not 0 < dense_mib <= entry[1] * widen // 2:
        return None
    peak = entry[0] * widen // 2
    if speed_mode in _MEASURED_DENSE_SPEED_MODES:
        return peak
    phases = _MEASURED_IMAGE_ACTIVATION_MIB.get(family)
    if (
        speed_mode not in _MEASURED_PEAK_SPEED_MODES
        or not tier_seeds
        or phases is None
        or _env_off(MEASURED_ACTIVATION_DENSE_TIERS_ENV)
    ):
        return None
    return max(peak, max(phases[1 if speed_mode == "max" else 0][:3]) * widen // 2)


def measured_image_runtime_mib(
    family: Optional[str],
    speed_mode: Optional[str],
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    batch_size: int = 1,
    dense_transformer_mib: Optional[int] = None,
    compute_bytes: int = 2,
    quant: Optional[str] = None,
    phase_seeds: bool = False,
    static_seed: bool = True,
    dense_tier_seeds: bool = False,
    phase: Optional[str] = None,
    dense_bf16: bool = False,
    small_host_int8: bool = False,
) -> Optional[int]:
    """Measured runtime headroom for ``family`` at this size, or None (unmeasured: use the flat estimate).
    ``phase`` (``diffusion_measured_peaks.PHASES``) reads that phase's learned peak instead of the whole render's.
    ``dense_transformer_mib`` selects the dense eager table; ``compute_bytes`` scales it (fp32 Qwen-Image: 8480 vs 4248).

    Seeds: the peak tables above, then the per-phase calibration (``_MEASURED_IMAGE_ACTIVATION_MIB``, the same
    quantity per phase) for a quantised family the peak table lacks, only with ``phase_seeds`` (the target is where
    those were measured: NVIDIA with a sub-quadratic attention kernel, ``_calibrated_activation``). ``quant`` (``denoiser_quant_label``) adds the
    peak this machine recorded for (family, quant, speed tier, resolution bucket) at an earlier render
    (diffusion_measured_peaks): it raises a seed and stands in for a missing one, never lowers a seed.
    ``static_seed=False`` (a denoiser no static table was measured on: bf16 or GGUF / int8 packed dense weights)
    answers from the learned peak alone. ``dense_tier_seeds`` enables the dense default / max tier seeds (``_dense_seed_mib``)
    on the target the phase table was measured on."""
    if _env_off(MEASURED_ACTIVATION_ENV):
        return None
    seed: Optional[int] = None
    if dense_transformer_mib is not None and _env_off(MEASURED_ACTIVATION_DENSE_ENV):
        return None
    if static_seed and dense_transformer_mib is not None:
        seed = _dense_seed_mib(
            str(family or "").lower(),
            str(speed_mode or ""),
            int(dense_transformer_mib),
            max(2, int(compute_bytes or 2)),
            dense_tier_seeds,
            bf16 = dense_bf16,
        )
    elif small_host_int8 and dense_transformer_mib is not None and phase is None:
        seed = _small_host_int8_seed_mib(
            family, speed_mode, int(dense_transformer_mib), max(2, int(compute_bytes or 2))
        )
    elif static_seed and str(speed_mode or "") in _MEASURED_PEAK_SPEED_MODES:
        seed = _MEASURED_IMAGE_PEAK_MIB.get(str(family or "").lower())
        if str(family or "").lower() == "qwen-image-edit" and _env_off(
            MEASURED_ACTIVATION_QWEN_EDIT_ENV
        ):
            seed = None
        if seed is None and phase_seeds:
            phases = _MEASURED_IMAGE_ACTIVATION_MIB.get(str(family or "").lower())
            if phases is not None:
                # worst of encoder / denoise / untiled decode at 1024x1024 for this tier
                seed = max(phases[1 if speed_mode == "max" else 0][:3])
    w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
    h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
    batch = max(1, int(batch_size or 1))
    learned = None
    if quant is not None:
        from .diffusion_measured_peaks import lookup as learned_peak
        learned = learned_peak(
            family = family,
            quant = quant,
            speed_mode = speed_mode,
            width = w,
            height = h,
            batch_size = batch,
            phase = phase,
        )
    if seed is None and learned is None:
        return None
    scale = max(1.0, (w * h * batch) / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT))
    need = max(float(seed or 0) * scale, float(learned or 0)) * _MEASURED_PEAK_MARGIN
    step = _MEASURED_PEAK_ROUND_MIB
    return int(-(-need // step) * step)


def learned_peak_label(
    pipe: Any,
    attention: Optional[str] = None,
    model: Optional[str] = None,
    vae: Optional[str] = None,
    te: Optional[str] = None,
    cache: Optional[str] = None,
    graphs: Optional[str] = None,
) -> str:
    """``denoiser_quant_label`` plus the loaded model (variants of one family differ in size), the text encoders'
    precision (a dense streamed encoder peaks above a quantised one), the VAE savers (an untiled decode peaks far
    above a tiled one) and the engaged attention backend (backends with different workspaces): none shares a learned
    peak. The VAE savers and the backend can change after the load; ``with_label_part`` rewrites them."""
    label = denoiser_quant_label(pipe)
    for name, value in (
        ("model", model),
        ("te", te),
        ("cache", cache),
        ("vae", vae),
        ("attn", attention),
        ("graphs", graphs),
    ):
        if value is not None:
            label = f"{label}|{name}:{value}"
    return label


# Renders without guidance on a CFG-capable family; read only when no CFG peak exists.
CFG_OFF_PART = "cfg:off"
LEARNED_PEAKS_CFG_OFF_ENV = "UNSLOTH_DIFFUSION_LEARNED_PEAKS_CFG_OFF"


def cfg_off_peaks_enabled() -> bool:
    return not _env_off(LEARNED_PEAKS_CFG_OFF_ENV)


# Deferred profiles record and price their own tier's learned peak; 0 = flat estimate for compiled renders.
TIER_PEAKS_ENV = "UNSLOTH_DIFFUSION_TIER_PEAKS"
TIER_MOVED_ATTR = "_unsloth_peak_tier_moved"


def tier_peaks_enabled() -> bool:
    return not _env_off(TIER_PEAKS_ENV)


def without_cfg_off(label: str) -> str:
    return "|".join(part for part in str(label).split("|") if part != CFG_OFF_PART)


def _learned_read_label(
    pipe: Any, family: Optional[str], quant: str, speed_mode: Optional[str]
) -> tuple[str, bool]:
    """``(label, cfg_off_only)``: True means CFG requests must be sized from the flat estimate."""
    from .diffusion_measured_peaks import lookup_bucket

    if (
        not getattr(pipe, "_unsloth_peak_expects_cfg", False)
        or not cfg_off_peaks_enabled()
        or lookup_bucket(family = family, quant = quant, speed_mode = speed_mode) is not None
    ):
        return quant, False
    off = f"{quant}|{CFG_OFF_PART}"
    if lookup_bucket(family = family, quant = off, speed_mode = speed_mode) is None:
        return quant, False
    return off, True


def with_label_part(label: str, name: str, value: str) -> str:
    """``label`` with its ``name:`` part replaced by ``value`` (unchanged when it has no such part)."""
    return "|".join(
        f"{name}:{value}" if part.startswith(f"{name}:") else part for part in str(label).split("|")
    )


# Non-txt2img workflows record under ``wf:<workflow>`` so an edit's larger peak never inflates the next txt2img load.
WORKFLOW_PART = "wf"
TXT2IMG_WORKFLOW = "txt2img"


def with_workflow(label: str, workflow: Optional[str]) -> str:
    """``label`` keyed for ``workflow``: unchanged for text-to-image (and an unknown workflow), else ``|wf:<name>``."""
    name = str(workflow or "").strip().lower()
    if not name or name == TXT2IMG_WORKFLOW:
        return str(label)
    return f"{label}|{WORKFLOW_PART}:{name}"


def learned_phase_split(
    family: Optional[str], quant: Optional[str], speed_mode: Optional[str]
) -> Optional[dict]:
    """Learned phases (+ ``pixels`` / ``quadratic``) at the default bucket, or None unless every phase shares the
    whole-render bucket and the larger padded phase reaches the whole-render peak."""
    from .diffusion_measured_peaks import PHASE_DECODE, PHASE_PRE_DECODE, PHASES, lookup_bucket

    if quant is None:
        return None
    total = lookup_bucket(family = family, quant = quant, speed_mode = speed_mode)
    if total is None:
        return None
    found = {}
    for phase in PHASES:
        value = lookup_bucket(family = family, quant = quant, speed_mode = speed_mode, phase = phase)
        if value is None or value[1] != total[1]:
            return None
        found[phase] = int(value[0])
    if max(found[PHASE_PRE_DECODE], found[PHASE_DECODE]) < int(total[0]):
        return None
    found.update(pixels = int(total[1]), quadratic = bool(total[2]))
    return found


_VAE_SAVERS = (
    ("use_tiling", "enable_vae_tiling", "enable_tiling"),
    ("use_slicing", "enable_vae_slicing", "enable_slicing"),
)


def _vae_saver_can(pipe: Any, pipe_method: str, vae_method: str) -> bool:
    return callable(getattr(pipe, pipe_method, None)) or callable(
        getattr(getattr(pipe, "vae", None), vae_method, None)
    )


def _vae_savers_observable(pipe: Any) -> bool:
    """False when a saver can be engaged but the VAE keeps no ``use_*`` flag to read it back: its state is unknown."""
    vae = getattr(pipe, "vae", None)
    return all(
        hasattr(vae, flag) or not _vae_saver_can(pipe, pm, vm) for flag, pm, vm in _VAE_SAVERS
    )


def planned_vae_saver_tag(pipe: Any, plan: Any) -> str:
    """The VAE savers ``plan`` asks for that this pipeline can engage (what ``apply_memory_plan`` will turn on);
    ``unknown`` (no learned peak used or recorded) when the VAE cannot report them."""
    if not _vae_savers_observable(pipe):
        return "unknown"
    return vae_saver_tag(
        getattr(plan, "vae_tiling", False) and _vae_saver_can(pipe, *_VAE_SAVERS[0][1:]),
        getattr(plan, "vae_slicing", False) and _vae_saver_can(pipe, *_VAE_SAVERS[1][1:]),
    )


def live_vae_saver_tag(pipe: Any) -> str:
    """The VAE savers in force right now (``unknown`` as in ``planned_vae_saver_tag``)."""
    if not _vae_savers_observable(pipe):
        return "unknown"
    vae = getattr(pipe, "vae", None)
    return vae_saver_tag(getattr(vae, "use_tiling", False), getattr(vae, "use_slicing", False))


def stream_window_pad_mib(pipe: Any) -> Optional[int]:
    """MiB a learned peak measured with no denoiser group streaming adds to reach the streamed baseline (the prefetch
    window a streamed placement holds above its start allocation), so every placement reads one conservative scale.
    None when that window cannot be sized, or the denoiser is neither group-streamed nor on the card (whole-module /
    sequential offload onloads it whole inside the render): the peak is then not recorded. Resident text encoders add
    their leaf-streaming window the same way (``_encoder_stream_window_mib``)."""
    encoders = _encoder_stream_window_mib(pipe)
    if encoders is None:
        return None
    streaming = partial = False
    for name, module in (getattr(pipe, "components", {}) or {}).items():
        if str(name) not in ("transformer", "transformer_2", "unconditional_transformer"):
            continue
        if module is None:
            continue  # an empty slot (Wan2.2-TI2V-5B and Wan2.1 register transformer_2 = None)
        state = _streaming_state(module)
        if state is None:
            return None
        if state:
            streaming = True
            partial = partial or any(
                getattr(g, "_unsloth_resident", False) for g in _offload_groups(module)
            )
    if streaming and not partial:
        return encoders
    window = _stream_window_mib(pipe)
    if window < 0:
        return None
    if streaming:
        # some groups resident: the measured window is the largest STREAMED group's, a fully streamed load holds the
        # largest of all; pad the difference
        streamed = _streamed_group_window_mib(pipe)
        if streamed < 0:
            return None
        window = max(0, window - streamed)
    return window + encoders


def _streamed_group_window_mib(pipe: Any) -> int:
    """Device MiB the prefetcher holds for the denoiser groups that stream now: ``depth + 1`` of the largest, or -1."""
    try:
        from .diffusion_offload_prefetch import prefetch_depth

        largest = 0
        for name, module in (getattr(pipe, "components", {}) or {}).items():
            if str(name) not in ("transformer", "transformer_2", "unconditional_transformer"):
                continue
            if module is None:
                continue
            for group in _offload_groups(module):
                if getattr(group, "_unsloth_resident", False):
                    continue
                largest = max(largest, _group_nbytes(group))
        if largest <= 0:
            return -1
        return -(-largest * (prefetch_depth() + 1) // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return -1


def _group_nbytes(group: Any) -> int:
    """Storage bytes of one offload group's modules, parameters and buffers (shared tensors once)."""
    seen: set[int] = set()
    nbytes = 0
    for member in getattr(group, "modules", None) or ():
        for t in (*member.parameters(), *member.buffers()):
            if id(t) not in seen:
                seen.add(id(t))
                nbytes += sum(_storage_nbytes(t))
    for t in (
        *(getattr(group, "parameters", None) or ()),
        *(getattr(group, "buffers", None) or ()),
    ):
        if id(t) not in seen:
            seen.add(id(t))
            nbytes += sum(_storage_nbytes(t))
    return nbytes


def _streaming_state(module: Any) -> Optional[bool]:
    """True when ``module`` streams offload groups, False when it sits on the card, None when neither can be told
    (whole-module / sequential offload, weights off the card without hooks)."""
    try:
        groups = _offload_groups(module) or ()
        if any(not getattr(g, "_unsloth_resident", False) for g in groups):
            return True
        if not groups:
            if getattr(module, "_hf_hook", None) is not None:
                return None
            first = next(iter(module.parameters()), None)
            if first is not None and getattr(first.device, "type", None) != "cuda":
                return None
        return False
    except Exception:  # noqa: BLE001
        return None


def _encoder_stream_window_mib(pipe: Any) -> Optional[int]:
    """MiB a load that streams the text encoders (leaf-level groups) holds in flight that a render with them resident
    did not: ``depth + 1`` of the largest leaf of each resident encoder, less the streamed leaves' window of a
    partially resident one. None when an encoder's placement is unknown."""
    try:
        import torch

        from .diffusion_offload_prefetch import prefetch_depth

        total = 0
        for name, module in (getattr(pipe, "components", {}) or {}).items():
            if not str(name).startswith("text_encoder") or not isinstance(module, torch.nn.Module):
                continue
            state = _streaming_state(module)
            if state is None:
                return None
            groups = list(_offload_groups(module) or ()) if state else []
            if state and not any(getattr(g, "_unsloth_resident", False) for g in groups):
                continue  # fully streamed during the render: its whole window is in the peak
            largest = 0
            for sub in module.modules():
                seen: set[int] = set()
                nbytes = 0
                for t in (*sub.parameters(recurse = False), *sub.buffers(recurse = False)):
                    if id(t) not in seen:
                        seen.add(id(t))
                        nbytes += sum(_storage_nbytes(t))
                largest = max(largest, nbytes)
            window = -(-largest * (prefetch_depth() + 1) // (1024 * 1024))
            if state:
                # some leaves resident: the peak holds the largest STREAMED leaf's window, a fully streamed encoder
                # the largest of all; pad the difference
                streamed = max(
                    (
                        _group_nbytes(g)
                        for g in groups
                        if not getattr(g, "_unsloth_resident", False)
                    ),
                    default = 0,
                )
                window = max(0, window - -(-streamed * (prefetch_depth() + 1) // (1024 * 1024)))
            total += window
        return total
    except Exception:  # noqa: BLE001
        return None


def vae_saver_tag(tiling: Any, slicing: Any) -> str:
    """The VAE savers in force, as the learned-peak key names them."""
    return "+".join(name for name, on in (("tiled", tiling), ("sliced", slicing)) if on) or "full"


def denoiser_quant_label(pipe: Any) -> str:
    """Stable label of the denoiser weights' format for the learned-peak key: torchao subclasses, GGUF, int8 / uint8
    packed, fp8-stored (codes in buffers, widened per layer: ``fp8_stored:<float dtypes>#<variant>``), or the dense
    dtype."""
    try:
        import torch

        dits = [
            getattr(pipe, name, None)
            for name in ("transformer", "transformer_2", "unconditional_transformer", "unet")
        ]
        dits = [d for d in dits if isinstance(d, torch.nn.Module)]
        if not dits:
            return "none"
        classes: set[str] = set()
        for dit in dits:
            classes |= _torchao_weight_classes(dit)
        if classes:
            return "torchao:" + "+".join(sorted(classes)) + _quant_variant_tag(dits)
        dtypes: set[str] = set()
        for dit in dits:
            for param in dit.parameters():
                if type(param).__name__ == "GGUFParameter":
                    return "gguf"
                if param.dtype in (torch.int8, torch.uint8):
                    dtypes.add("int8")
                elif param.is_floating_point():
                    dtypes.add(str(param.dtype).replace("torch.", ""))
        if "int8" in dtypes:
            return "packed:int8" + _quant_variant_tag(dits)
        if any(_holds_fp8_stored(dit) for dit in dits):
            # Never the dense key: same model file, same float dtype, but a per-layer widen the dense run lacks.
            return "fp8_stored:" + "+".join(sorted(dtypes)) + _quant_variant_tag(dits)
        return "dense:" + "+".join(sorted(dtypes)) if dtypes else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _quant_variant_tag(dits: list) -> str:
    """``#`` plus a short digest of the denoisers' quant graph variant (``graph_variant``, the compile cache's key):
    plain vs ConvRot, W8A8 vs weight-only, hosted vs runtime scales share a weight class but not their kernels or
    workspaces. ``#unknown`` when it cannot be read (the label then neither reads nor records a peak)."""
    try:
        import hashlib
        import json

        from .diffusion_compile_cache import graph_variant

        variants = [graph_variant(dit) for dit in dits]
        if any(isinstance(v.get("weights"), str) for v in variants):
            return "#unknown"
        blob = json.dumps(variants, sort_keys = True, default = str)
        return "#" + hashlib.sha256(blob.encode()).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return "#unknown"


def _loaded_component_mib(pipe: Any) -> Optional[dict[str, tuple[int, str]]]:
    """name -> (MiB, role in 'dit' / 'text_encoder' / 'other') from storage bytes (a torchao int8 weight counts its
    int8 data + scales); None when unreadable."""
    try:
        import torch

        out: dict[str, tuple[int, str]] = {}
        for name, module in (getattr(pipe, "components", {}) or {}).items():
            if not isinstance(module, torch.nn.Module):
                continue
            seen: set[int] = set()
            nbytes = 0
            for tensor in (*module.parameters(), *module.buffers()):
                if id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                nbytes += sum(_storage_nbytes(tensor))
            name = str(name)
            role = (
                "dit"
                if name in ("transformer", "transformer_2", "unconditional_transformer")
                else "text_encoder"
                if name.startswith("text_encoder")
                else "other"
            )
            out[name] = (-(-nbytes // (1024 * 1024)), role)
        return out
    except Exception:  # noqa: BLE001 - an optional refinement
        return None


RESIDENT_DIT_ENV = "UNSLOTH_DIFFUSION_RESIDENT_DIT"
# Resident slack: 10% of card (min 1 GiB) for fragmentation and lazy kernels; peak covers the rest.
_RESIDENT_DIT_SLACK_FRACTION = 0.10
_RESIDENT_DIT_SLACK_MIN_MIB = 1024


def _resident_dit_slack_mib(memory: Any) -> int:
    """The measured-residency margin, or what the user's VRAM Budget keeps free when larger (as the planner reserve)."""
    base = getattr(memory, "total_mib", None) or getattr(memory, "free_mib", None) or 0
    return max(
        _RESIDENT_DIT_SLACK_MIN_MIB,
        int(int(base) * _RESIDENT_DIT_SLACK_FRACTION),
        _device_user_budget_reserve_mib(memory, int(base)),
    )


def _resident_dit_fits(memory: Any, dit_mib: int, headroom_mib: int, other_mib: int) -> bool:
    """Whole denoiser + non-encoder companions + measured peak x margin + slack within free memory, encoders streamed."""
    if _env_off(RESIDENT_DIT_ENV):
        return False
    free = getattr(memory, "free_mib", None)
    if free is None:
        return False
    return int(dit_mib) + int(other_mib) + int(headroom_mib) + _resident_dit_slack_mib(
        memory
    ) <= int(free)


STREAMED_RESIDENCY_ENV = "UNSLOTH_DIFFUSION_STREAMED_RESIDENCY"


def _stream_window_mib(pipe: Any) -> int:
    """Device MiB the block prefetcher holds in flight: ``depth + 1`` of the largest denoiser block, or -1."""
    try:
        import torch

        from .diffusion_offload_prefetch import prefetch_depth

        largest = 0
        for name, module in (getattr(pipe, "components", {}) or {}).items():
            if str(name) not in ("transformer", "transformer_2", "unconditional_transformer"):
                continue
            if not isinstance(module, torch.nn.Module):
                continue
            for child in module.children():
                if not isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)):
                    continue
                for block in child:
                    seen: set[int] = set()
                    nbytes = 0
                    for t in (*block.parameters(), *block.buffers()):
                        if id(t) not in seen:
                            seen.add(id(t))
                            nbytes += sum(_storage_nbytes(t))
                    largest = max(largest, nbytes * DEFAULT_GROUP_BLOCKS)
        if largest <= 0:
            return -1
        return -(-largest * (prefetch_depth() + 1) // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return -1


def _streamed_dit_room_mib(memory: Any, headroom_mib: int, other_mib: int, window_mib: int) -> int:
    """Partial room: the whole-resident fit plus the prefetch window, not the flat reserve AND base overhead."""
    free = getattr(memory, "free_mib", None)
    if free is None or window_mib < 0 or _env_off(STREAMED_RESIDENCY_ENV):
        return 0
    return max(
        0,
        int(free)
        - _resident_dit_slack_mib(memory)
        - int(other_mib)
        - int(headroom_mib)
        - int(window_mib),
    )


def _denoiser_compute_bytes(pipe: Any) -> Optional[int]:
    """Denoiser compute element size for the dense table: 2 (fp16), 4 (fp32); None for bf16 / unreadable."""
    try:
        import torch
        for name in ("transformer", "unet"):
            module = getattr(pipe, name, None)
            dtype = getattr(module, "dtype", None) if module is not None else None
            if dtype is not None:
                return {torch.float16: 2, torch.float32: 4}.get(dtype)
    except Exception:  # noqa: BLE001
        pass
    return None


_DENOISER_FIRST_REASON = (
    "the text encoders stream instead of staying resident: that keeps {streamed} MiB of the denoiser resident "
    "rather than {resident} MiB (the denoiser runs every step, the encoders once per prompt)"
)


def refine_plan_from_loaded_weights(
    pipe: Any,
    plan: MemoryPlan,
    *,
    family: Optional[str],
    speed_mode: Optional[str],
    logger: Any = None,
    **labels: Optional[str],
) -> MemoryPlan:
    """Re-place a streamed ``auto`` load from its LOADED weights; the denoiser has priority over the text encoders.

    The flat plan keeps the encoders resident once their floor fits, and their bytes come out of the denoiser's room.
    The encoders-streamed plan is taken only when it keeps strictly more of the denoiser resident; plans where the
    encoders cost nothing are unchanged."""
    refined = _refine_plan_from_loaded_weights(
        pipe, plan, family = family, speed_mode = speed_mode, logger = logger, **labels
    )
    if (
        getattr(plan, "requested_mode", None) != MEMORY_MODE_AUTO
        or plan.offload_policy != OFFLOAD_GROUP
        or bool(getattr(plan, "stream_text_encoders", False))
        or not bool(getattr(plan, "stream_transformer", True))
        or plan.estimates.get("text_encoder_dense_mib") is None
    ):
        return refined
    try:
        streamed = _refine_plan_from_loaded_weights(
            pipe,
            replace(plan, stream_text_encoders = True),
            family = family,
            speed_mode = speed_mode,
            logger = None,
            **labels,
        )
        resident_dit = int(getattr(refined, "resident_transformer_mib", None) or 0)
        streamed_dit = int(getattr(streamed, "resident_transformer_mib", None) or 0)
        if streamed_dit <= resident_dit:
            return refined
        reason = _DENOISER_FIRST_REASON.format(streamed = streamed_dit, resident = resident_dit)
        if logger is not None:
            logger.info("diffusion.memory: %s", reason)
        return replace(
            streamed,
            reasons = streamed.reasons + (reason,),
            estimates = {**streamed.estimates, "denoiser_first_resident_te_dit_mib": resident_dit},
        )
    except Exception as exc:  # noqa: BLE001 - keep the encoders-resident refinement
        if logger is not None:
            logger.debug("diffusion.memory: streamed-encoder comparison skipped (%s)", exc)
        return refined


def _denoiser_is_dense_bf16(pipe: Any) -> bool:
    """A plain bf16 denoiser: no torchao, packed (GGUF / int8) or fp8-stored weights."""
    try:
        import torch

        module = next(
            (
                getattr(pipe, n, None)
                for n in ("transformer", "unet")
                if getattr(pipe, n, None) is not None
            ),
            None,
        )
        if module is None or getattr(module, "dtype", None) != torch.bfloat16:
            return False
        if _pipe_denoisers_hold_torchao(pipe) or _pipe_denoisers_hold_packed(pipe):
            return False
        return not any(
            _holds_fp8_stored(getattr(pipe, n, None))
            for n in ("transformer", "transformer_2", "unet")
        )
    except Exception:  # noqa: BLE001 - unreadable: keep the streamed plan
        return False


_WHOLE_MODULE_KEPT_REASON = (
    "whole-module offload ran the {dit} MiB denoiser whole below the streamed tiers' {floor} MiB floor; it still "
    "fits, and uploading it once per call beats streaming it every step"
)
_WHOLE_MODULE_COST_REASON = (
    " (measured {h2d:.0f} GiB/s host-to-device and {tflops:.0f} bf16 TFLOPS: {model:.2f} s vs {group:.2f} s "
    "streamed for a {steps}-step {side}px render)"
)
_WHOLE_MODULE_DECLINED_REASON = (
    "streaming kept over whole-module offload of the {dit} MiB denoiser: measured {h2d:.0f} GiB/s host-to-device "
    "and {tflops:.0f} bf16 TFLOPS price a {steps}-step {side}px render at {model:.2f} s whole vs {group:.2f} s streamed"
)

# ``auto``: whole-module offload only where priced cheaper; ``1`` always (#15129); ``0`` never.
WHOLE_MODULE_OFFLOAD_ENV = "UNSLOTH_DIFFUSION_WHOLE_MODULE_OFFLOAD"
WHOLE_MODULE_PER_REQUEST_ENV = "UNSLOTH_DIFFUSION_WHOLE_MODULE_PER_REQUEST"
WHOLE_TIER_ATTR = "_unsloth_whole_tier"


def per_request_whole_enabled() -> bool:
    return not _env_off(WHOLE_MODULE_PER_REQUEST_ENV)


# Un-overlapped whole upload vs one streamed step's transfer: 1.47x B200, 1.27-1.58x RTX PRO 6000 (klein-4B bf16).
_WHOLE_UPLOAD_FACTOR = 1.5
# Resident step time / probe-priced FLOPs: 1.38 B200, 0.97 RTX PRO 6000; 1.0 picks right on both, 1.35 does not.
_DIT_COMPUTE_EFFICIENCY = 1.0
_COST_CANVAS_PX = 1024
# 2x2 patches of an 8x VAE latent
_DIT_PIXELS_PER_TOKEN = 16

_OFFLOAD_COST_PROBES: dict = {}


def _offload_cost_measurements() -> Optional[tuple[float, float]]:
    """(pinned host-to-device GiB/s, bf16 GEMM TFLOPS) of the current CUDA device, cached per device; None off CUDA.
    Best of a few runs, so a transient stall on a shared card is not read as the card."""
    result = None
    key = None
    try:
        import torch

        # integrated GPU: the probe's pinned buffer only takes RAM from the card; unpriced keeps the whole-module plan
        if not torch.cuda.is_available() or is_rocm_torch(torch) or not host_pinning_allowed():
            return None
        key = torch.cuda.current_device()
        if key in _OFFLOAD_COST_PROBES:
            return _OFFLOAD_COST_PROBES[key]
        device = torch.device("cuda", key)
        with torch.cuda.device(device):
            nbytes = 256 << 20
            host = torch.empty(nbytes, dtype = torch.uint8, pin_memory = True)
            dev = torch.empty(nbytes, dtype = torch.uint8, device = device)
            a = torch.randn(4096, 3072, dtype = torch.bfloat16, device = device)
            b = torch.randn(3072, 3072, dtype = torch.bfloat16, device = device)
            start, end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)

            def _best_ms(fn, reps: int, inner: int) -> float:
                fn()
                torch.cuda.synchronize()
                best = float("inf")
                for _ in range(reps):
                    start.record()
                    for _ in range(inner):
                        fn()
                    end.record()
                    end.synchronize()
                    best = min(best, start.elapsed_time(end) / inner)
                return best

            copy_ms = _best_ms(lambda: dev.copy_(host, non_blocking = True), 3, 2)
            gemm_ms = _best_ms(lambda: torch.mm(a, b), 3, 10)
            del host, dev, a, b
            torch.cuda.empty_cache()
        if copy_ms > 0 and gemm_ms > 0:
            result = (
                (nbytes / 2**30) / (copy_ms / 1000.0),
                2.0 * 4096 * 3072 * 3072 / (gemm_ms / 1000.0) / 1e12,
            )
    except Exception:  # noqa: BLE001 - unmeasurable: the caller keeps its unpriced choice
        result = None
    if key is not None:
        _OFFLOAD_COST_PROBES[key] = result
    return result


def _whole_module_offload_costs(
    dit_mib: int,
    *,
    steps: int,
    h2d_gib_s: float,
    bf16_tflops: float,
    side_px: int = _COST_CANVAS_PX,
    tokens: Optional[int] = None,
) -> tuple[float, float]:
    """(whole-module, streamed) denoiser seconds for one ``steps``-step render.

    streamed = ``steps * max(compute, transfer)``; transfer counts the whole denoiser even when partly resident (B200
    klein step stayed 0.145-0.170 s at 0-6751 of 7393 MiB resident). whole = one un-overlapped upload + resident steps.
    """
    gib = max(0, int(dit_mib)) / 1024.0
    transfer = gib / max(float(h2d_gib_s), 1e-6)
    params = max(0, int(dit_mib)) * 2**20 / 2.0  # bf16
    if tokens is None:
        tokens = (int(side_px) // _DIT_PIXELS_PER_TOKEN) ** 2
    compute = (
        _DIT_COMPUTE_EFFICIENCY * 2.0 * params * tokens / (max(float(bf16_tflops), 1e-6) * 1e12)
    )
    n = max(1, int(steps))
    whole = _WHOLE_UPLOAD_FACTOR * transfer + n * compute
    streamed = n * max(compute, transfer)
    return whole, streamed


def prefer_whole_module_offload(
    pipe: Any,
    refined: MemoryPlan,
    *,
    logger: Any = None,
    steps: Optional[int] = None,
) -> MemoryPlan:
    """``refined``, or whole-module offload in its place where it beats streaming. With ``steps`` the switch is
    priced from this card's measured bandwidth and bf16 throughput (RTX PRO 6000 streams faster at 4 steps, B200 does
    not); unmeasurable or without ``steps`` it is kept. Plain bf16 only: an fp16 T4 run pointed the other way.
    """
    switch = (os.environ.get(WHOLE_MODULE_OFFLOAD_ENV) or "auto").strip().lower()
    if switch in ("0", "off", "false", "no", "never"):
        return refined
    try:
        if (
            getattr(refined, "requested_mode", None) != MEMORY_MODE_AUTO
            or refined.offload_policy != OFFLOAD_GROUP
            or not bool(getattr(refined, "stream_transformer", True))
            or getattr(refined, "secondary_ordinal", None) is not None
        ):
            return refined
        est = refined.estimates
        model_dense, companions = est.get("model_dense_mib"), est.get("companion_dense_mib")
        floors = [
            int(f)
            for f in (est.get("group_floor_mib"), est.get("group_floor_streamed_te_mib"))
            if f is not None
        ]
        if model_dense is None or companions is None or not floors:
            return refined
        transformer = int(model_dense) - int(companions)
        te = max(0, int(est.get("text_encoder_dense_mib") or 0))
        others = max(0, int(companions) - te)
        floor = min(floors)
        if transformer <= 0 or max(transformer, te, others) >= floor:
            return refined
        if not _denoiser_is_dense_bf16(pipe):
            return refined
        # A refined plan pinning the whole denoiser runs every step resident: kept.
        sizes = _loaded_component_mib(pipe) or {}
        dit = sum(m for m, r in sizes.values() if r == "dit") or transformer
        if int(getattr(refined, "resident_transformer_mib", None) or 0) >= dit:
            return refined
        reason = _WHOLE_MODULE_KEPT_REASON.format(dit = transformer, floor = floor)
        priced: dict = {}
        if (
            steps is not None
            and switch not in ("1", "on", "true", "yes", "always")
            and getattr(pipe, "transformer", None) is not None
        ):
            measured = _offload_cost_measurements()
            if measured is not None:
                h2d, tflops = measured
                whole, streamed = _whole_module_offload_costs(
                    dit, steps = int(steps), h2d_gib_s = h2d, bf16_tflops = tflops
                )
                fields = dict(
                    dit = dit,
                    h2d = h2d,
                    tflops = tflops,
                    model = whole,
                    group = streamed,
                    steps = int(steps),
                    side = _COST_CANVAS_PX,
                )
                priced = {
                    "whole_module_cost_s": round(whole, 4),
                    "streamed_cost_s": round(streamed, 4),
                    "whole_module_cost_steps": int(steps),
                    "measured_h2d_gib_s": round(h2d, 2),
                    "measured_bf16_tflops": round(tflops, 1),
                }
                if whole >= streamed:
                    declined = _WHOLE_MODULE_DECLINED_REASON.format(**fields)
                    if logger is not None:
                        logger.info("diffusion.memory: %s", declined)
                    if per_request_whole_enabled():
                        try:
                            setattr(
                                pipe,
                                WHOLE_TIER_ATTR,
                                {"dit_mib": int(dit), "h2d_gib_s": h2d, "bf16_tflops": tflops},
                            )
                        except Exception:  # noqa: BLE001 - the load-time plan then stays for every request
                            pass
                    return replace(
                        refined,
                        reasons = refined.reasons + (declined,),
                        estimates = {**refined.estimates, **priced},
                    )
                reason += _WHOLE_MODULE_COST_REASON.format(**fields)
        if logger is not None:
            logger.info("diffusion.memory: %s", reason)
        return replace(
            refined,
            offload_policy = OFFLOAD_MODEL,
            stream_text_encoders = False,
            stream_transformer = True,
            vae_tiling = True,
            vae_slicing = True,
            resident_transformer_mib = None,
            resident_text_encoder_mib = None,
            reasons = refined.reasons + (reason,),
            estimates = {**refined.estimates, "whole_module_kept_below_floor_mib": floor, **priced},
        )
    except Exception as exc:  # noqa: BLE001 - keep the streamed refinement
        if logger is not None:
            logger.debug("diffusion.memory: whole-module comparison skipped (%s)", exc)
        return refined


def whole_tier_request_costs(
    pipe: Any,
    *,
    steps: Optional[int],
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    cfg: Optional[bool] = None,
    condition_pixels: int = 0,
) -> Optional[tuple[float, float]]:
    """(whole, streamed) seconds for this request's own tokens (canvas, condition images, batch, CFG); None when the
    load did not record :data:`WHOLE_TIER_ATTR` or the per-request choice is off."""
    tier = getattr(pipe, WHOLE_TIER_ATTR, None)
    if not isinstance(tier, dict) or steps is None or not per_request_whole_enabled():
        return None
    try:
        w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
        h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
        per_image = (w // _DIT_PIXELS_PER_TOKEN) * (h // _DIT_PIXELS_PER_TOKEN) + max(
            0, int(condition_pixels or 0)
        ) // (_DIT_PIXELS_PER_TOKEN**2)
        tokens = per_image * max(1, int(batch_size or 1)) * (2 if cfg else 1)
        return _whole_module_offload_costs(
            int(tier["dit_mib"]),
            steps = int(steps),
            h2d_gib_s = float(tier["h2d_gib_s"]),
            bf16_tflops = float(tier["bf16_tflops"]),
            tokens = tokens,
        )
    except Exception:  # noqa: BLE001 - unpriced: the load's streamed plan
        return None


_MISSING = object()


def pin_denoiser_for_call(
    pipe: Any,
    device: Any,
    *,
    activation_mib: int,
    logger: Any = None,
) -> Optional[Callable[[], None]]:
    """Pin every streamed offload group of ``pipe.transformer`` for one call; returns the release, or None.

    The hooks stay and blocks compile below them, so compiled code never sees the pin. Pinned only if the groups plus
    ``activation_mib`` fit free memory now (VAE reserve kept); all or nothing."""
    transformer = getattr(pipe, "transformer", None)
    try:
        import torch

        groups = _offload_groups(transformer) if transformer is not None else []
        if not groups or any(getattr(g, "offload_to_disk_path", None) for g in groups):
            return None
        streamed = [g for g in groups if not getattr(g, "_unsloth_resident", False)]
        if not streamed:
            return None
        onload = torch.device(device if device is not None else "cuda")
        if onload.type != "cuda":
            return None
        if onload.index is None:
            onload = torch.device("cuda", torch.cuda.current_device())
        unique: dict = {}
        for group in streamed:
            for t in _resident_tensors(group):
                unique.setdefault(id(t), t)
        need = sum(sum(_storage_nbytes(t)) for t in unique.values())
        free, total = (int(x) for x in torch.cuda.mem_get_info(onload))
        free -= user_budget_reserve_bytes(onload, total)
        cached = max(
            0,
            int(torch.cuda.memory_reserved(onload))
            - int(torch.cuda.memory_allocated(onload))
            - _graph_held_cache_bytes(onload.index),
        )
        room = (free + cached) >> 20
        if (need >> 20) + max(0, int(activation_mib)) > room:
            if logger is not None:
                logger.info(
                    "diffusion.memory: the denoiser keeps streaming: running it whole needs %d MiB + %d MiB of "
                    "activations, %d MiB free",
                    need >> 20,
                    int(activation_mib),
                    room,
                )
            return None
        every: dict = {}
        for group in groups:
            for t in _resident_tensors(group):
                every.setdefault(id(t), t)
        whole_mib = (sum(sum(_storage_nbytes(t)) for t in every.values()) >> 20) + 2
        saved = {
            name: transformer.__dict__.get(name, _MISSING)
            for name in ("_unsloth_resident_room", "_unsloth_resident_device")
        }
        _keep_groups_resident(transformer, whole_mib, onload, logger)
        pinned = [g for g in streamed if getattr(g, "_unsloth_resident", False)]
        done = [False]

        def release() -> None:
            if done[0]:
                return
            done[0] = True
            state = getattr(transformer, "_unsloth_stream_state", None)
            if isinstance(state, dict):
                state["streamed"] = (
                    1  # a release that fails part way still waits on the copy stream
                )
            try:
                for group in reversed(pinned):
                    if getattr(group, "_unsloth_resident", False):
                        _release_group(group)
                        _release_group_range(transformer, group)
            finally:
                if isinstance(state, dict):
                    state["streamed"] = _streamed_group_count(groups)
                for name, value in saved.items():
                    try:
                        if value is _MISSING:
                            transformer.__dict__.pop(name, None)
                        else:
                            setattr(transformer, name, value)
                    except Exception:  # noqa: BLE001
                        pass
                from .diffusion_cuda_graph import synchronize_outside_capture

                synchronize_outside_capture(onload)

        if len(pinned) != len(streamed):
            release()
            return None
        return release
    except Exception as exc:  # noqa: BLE001 - streaming every group is the safe state
        if logger is not None:
            logger.warning("diffusion.memory: whole denoiser for this call skipped (%s)", exc)
        return None


def _refine_plan_from_loaded_weights(
    pipe: Any,
    plan: MemoryPlan,
    *,
    family: Optional[str],
    speed_mode: Optional[str],
    logger: Any = None,
    attention: Optional[str] = None,
    model: Optional[str] = None,
    vae: Optional[str] = None,
    te: Optional[str] = None,
    cache: Optional[str] = None,
    graphs: Optional[str] = None,
) -> MemoryPlan:
    """Re-place a streamed ``auto`` load from its LOADED weights and the family's measured activation peak.

    Keeps the flat plan's offload hooks and makes whole groups resident (denoiser first, then encoders) within the
    safe budget left after the measured peak x margin and the base overhead. No-op for explicit modes, non-CUDA / unified memory, unmeasured families (dense denoisers: the eager-tier table)
    or speed tiers and ``model`` plans. A bf16, GGUF / int8 packed or fp8-stored denoiser (no static table measured it)
    places only from a learned peak recorded for its exact build."""
    try:
        if getattr(plan, "requested_mode", None) != MEMORY_MODE_AUTO:
            return plan
        policy = plan.offload_policy
        if policy not in (OFFLOAD_GROUP, OFFLOAD_STREAMING):
            return plan
        memory = plan.device_memory
        if memory.is_unified or getattr(memory, "device", None) != "cuda":
            return plan
        budget = plan.estimates.get("safe_device_budget_mib")
        if budget is None:
            return plan
        budget = int(budget)
        sizes = _loaded_component_mib(pipe)
        if not sizes:
            return plan
        dit = sum(m for m, r in sizes.values() if r == "dit")
        encoders = sum(m for m, r in sizes.values() if r == "text_encoder")
        other = sum(m for m, r in sizes.values() if r == "other")
        if dit <= 0:
            return plan
        dense_mib = None if _pipe_denoisers_hold_torchao(pipe) else dit
        compute_bytes = _denoiser_compute_bytes(pipe)
        # Dense tables cover unpacked fp16 / fp32 / seeded bf16 only; stored fp8 widens per forward, never priced.
        packed = dense_mib is not None and _pipe_denoisers_hold_packed(pipe)
        fp8_stored = any(
            _holds_fp8_stored(getattr(pipe, name, None))
            for name in ("transformer", "transformer_2", "unconditional_transformer", "unet")
        )
        dense_bf16 = bool(
            dense_mib is not None
            and compute_bytes is None
            and not packed
            and not fp8_stored
            and dense_bf16_seeded(family)
            and _denoiser_is_dense_bf16(pipe)
            and bool(getattr(pipe, "_unsloth_dense_tier_seeds", False))
        )
        static_seed = not (
            fp8_stored
            or (dense_mib is not None and ((compute_bytes is None and not dense_bf16) or packed))
        )
        small_host_int8 = bool(
            packed and not fp8_stored and compute_bytes is not None and pipe_small_host_int8(pipe)
        )
        quant = learned_peak_label(pipe, attention, model, vae, te, cache, graphs)
        try:
            pipe._unsloth_peak_identity = (family, quant, speed_mode)
        except Exception:  # noqa: BLE001 - a pipe refusing attributes only skips recording
            pass
        read_quant, cfg_off_only = _learned_read_label(pipe, family, quant, speed_mode)
        headroom = measured_image_runtime_mib(
            family,
            speed_mode,
            dense_transformer_mib = dense_mib,
            compute_bytes = compute_bytes or 2,
            quant = read_quant,
            phase_seeds = bool(getattr(pipe, "_unsloth_phase_seeds", False)),
            dense_tier_seeds = bool(getattr(pipe, "_unsloth_dense_tier_seeds", False)),
            static_seed = static_seed,
            dense_bf16 = dense_bf16,
            small_host_int8 = small_host_int8,
        )
        if headroom is None:
            return plan
        from .diffusion_measured_peaks import lookup_bucket

        learned, learned_pixels, learned_quadratic = lookup_bucket(
            family = family, quant = read_quant, speed_mode = speed_mode
        ) or (None, None, False)
        overhead = int(plan.estimates.get("base_overhead_mib") or DEFAULT_BASE_OVERHEAD_MIB)
        floor = headroom + overhead + other
        estimates = dict(plan.estimates)
        estimates.update(
            measured_runtime_headroom_mib = headroom,
            measured_learned_peak_mib = learned,
            measured_learned_peak_pixels = learned_pixels,
            measured_learned_peak_quadratic = learned_quadratic,
            measured_dense_transformer_mib = dense_mib,
            measured_compute_bytes = compute_bytes,
            measured_static_seed = static_seed,
            measured_dense_bf16 = dense_bf16,
            measured_small_host_int8 = small_host_int8,
            measured_learned_cfg_off = bool(cfg_off_only and learned),
            # tier / phase seed measured on text-to-image: unpriced references size flat (measured_request_extra_mib)
            measured_omits_references = bool(
                getattr(pipe, "_unsloth_unpriced_references", False)
                and str(speed_mode or "") in _MEASURED_PEAK_SPEED_MODES
                and (
                    (dense_mib is not None and getattr(pipe, "_unsloth_dense_tier_seeds", False))
                    or (dense_mib is None and getattr(pipe, "_unsloth_phase_seeds", False))
                )
            ),
            loaded_transformer_mib = dit,
            loaded_text_encoder_mib = encoders,
            loaded_other_mib = other,
        )
        if _env_off(PARTIAL_RESIDENT_ENV):
            return plan
        # Flat hooks stay installed; release_resident_groups lets an oversized request stream groups again.
        stream_te = bool(getattr(plan, "stream_text_encoders", False))
        # The whole-render peak includes the stream prefetch window, which an all-resident placement never allocates:
        # all-resident is checked against the unpadded peak, else encoders against the padded pre-decode phase. Extra
        # residency is lent to the decode; release_for_decode re-streams it only if free memory is short then.
        pre_headroom = resident_headroom = None
        split = None
        if stream_te and policy == OFFLOAD_GROUP:
            split = learned_phase_split(family, read_quant, speed_mode)
        if split is not None:
            from .diffusion_measured_peaks import PHASE_PRE_DECODE, PHASE_UNPADDED

            def _phase_headroom(phase: str) -> Optional[int]:
                value = measured_image_runtime_mib(
                    family,
                    speed_mode,
                    dense_transformer_mib = dense_mib,
                    compute_bytes = compute_bytes or 2,
                    quant = read_quant,
                    phase_seeds = bool(getattr(pipe, "_unsloth_phase_seeds", False)),
                    dense_tier_seeds = bool(getattr(pipe, "_unsloth_dense_tier_seeds", False)),
                    static_seed = static_seed,
                    phase = phase,
                    dense_bf16 = dense_bf16,
                )
                return value if value is not None and value < headroom else None

            pre_headroom = _phase_headroom(PHASE_PRE_DECODE)
            if not bool(getattr(plan, "stream_transformer", True)):
                resident_headroom = _phase_headroom(PHASE_UNPADDED)
        room = budget - floor
        if policy == OFFLOAD_GROUP and not stream_te:
            room -= encoders  # resident companions
        whole_dit = wide_dit = False
        if not bool(getattr(plan, "stream_transformer", True)):
            room -= dit
            dit_room = 0
        else:
            dit_room = min(max(room, 0), dit)
            if (
                dit_room < dit
                # torchao denoisers only: the slack was measured on the int8 route, not the dense eager table
                and dense_mib is None
                and (policy == OFFLOAD_STREAMING or stream_te)
                and _resident_dit_fits(memory, dit, headroom, other)
            ):
                # pin it whole; during the encode it drops back to the flat room (install_encode_release)
                encode_room = int(dit_room)
                dit_room, room, whole_dit = dit, dit, True
            elif (
                dit_room < dit and dense_mib is None and (policy == OFFLOAD_STREAMING or stream_te)
            ):
                # partial: the encode keeps the flat room (install_encode_release)
                window = _stream_window_mib(pipe)
                streamed_room = min(_streamed_dit_room_mib(memory, headroom, other, window), dit)
                if streamed_room > dit_room:
                    encode_room = int(dit_room)
                    dit_room, room, wide_dit = streamed_room, streamed_room, True
                    estimates["stream_window_mib"] = window
        te_room = (
            min(max(0, room - dit_room), encoders) if stream_te and policy == OFFLOAD_GROUP else 0
        )
        lend = 0
        phase_headroom = None
        if split is not None and not whole_dit and not wide_dit:
            phase_te_room = te_room
            if (
                resident_headroom is not None
                and room + (headroom - resident_headroom) - dit_room >= encoders
            ):
                phase_headroom, phase_te_room = resident_headroom, encoders
            elif pre_headroom is not None:
                phase_headroom = pre_headroom
                phase_te_room = min(max(0, room + (headroom - pre_headroom) - dit_room), encoders)
            lend = max(0, phase_te_room - te_room)
            if lend > 0:
                from .diffusion_measured_peaks import PHASE_DECODE_UNPADDED
                te_room = phase_te_room
                estimates.update(
                    measured_phase_headroom_mib = phase_headroom,
                    measured_phase_all_resident = phase_headroom == resident_headroom,
                    measured_decode_peak_mib = split[PHASE_DECODE_UNPADDED],
                    measured_decode_peak_pixels = split["pixels"],
                    measured_decode_peak_quadratic = split["quadratic"],
                    decode_lend_text_encoder_mib = lend,
                )
        if dit_room <= 0 and te_room <= 0:
            return plan
        if whole_dit or wide_dit:
            estimates["resident_dit_slack_mib"] = _resident_dit_slack_mib(memory)
            estimates["encode_resident_transformer_mib"] = encode_room
        new = replace(
            plan,
            resident_transformer_mib = int(dit_room) if dit_room > 0 else None,
            resident_text_encoder_mib = int(te_room) if te_room > 0 else None,
            estimates = {
                **estimates,
                "resident_transformer_mib": int(dit_room),
                "resident_text_encoder_mib": int(te_room),
            },
            reasons = plan.reasons
            + (
                (
                    f"the {dit} MiB transformer is resident (not streamed by the plan)"
                    if not bool(getattr(plan, "stream_transformer", True))
                    else f"{int(dit_room)} MiB of the {dit} MiB transformer"
                )
                + f" and {int(te_room)} MiB of the {encoders} MiB "
                f"encoders stay resident (measured peak {headroom} MiB"
                + (
                    (
                        f"; {phase_headroom} MiB without the streaming window, every weight resident"
                        if phase_headroom == resident_headroom
                        else f"; {phase_headroom} MiB before the VAE decode"
                    )
                    + f", the decode may stream {lend} MiB of them again"
                    if lend > 0
                    else ""
                )
                + "); the rest streams"
                + (
                    f" (whole transformer within free memory less a {_resident_dit_slack_mib(memory)} MiB slack;"
                    f" {encode_room} MiB of it while the encoders run)"
                    if whole_dit
                    else f" (free memory less a {_resident_dit_slack_mib(memory)} MiB slack and a "
                    f"{estimates.get('stream_window_mib')} MiB prefetch window; {encode_room} MiB of it while the "
                    "encoders run)"
                    if wide_dit
                    else ""
                ),
            ),
        )
        if logger is not None:
            logger.info(
                "diffusion.memory: measured-activation placement %s -> %s (%s)",
                policy,
                new.offload_policy,
                new.reasons[-1],
            )
        return new
    except Exception as exc:  # noqa: BLE001 - keep the flat plan
        if logger is not None:
            logger.debug("diffusion.memory: measured-activation placement skipped (%s)", exc)
        return plan


def _placed_on(tensor: Any, device_type: str, is_torchao: Callable[[Any], bool]) -> bool:
    """torchao subclasses are judged by inner tensors: after a streamed offload the wrapper can report the wrong device."""
    if is_torchao(tensor):
        try:
            names, _ = tensor.__tensor_flatten__()
            inner = [getattr(tensor, n) for n in names]
            return all(getattr(t, "device", tensor.device).type == device_type for t in inner)
        except Exception:  # noqa: BLE001 - unflattenable: move it, the swap is idempotent
            return False
    return tensor.device.type == device_type


def _keep_groups_resident(
    module: Any,
    room_mib: int,
    device: Any,
    logger: Any = None,
    only: Optional[set] = None,
) -> int:
    """Make whole offload groups of a streamed ``module`` resident within ``room_mib`` (top-level group first, then
    blocks in order); their onload / offload become no-ops and the rest keeps streaming. Returns the MiB kept.
    ``only`` (group ids): a restore pins back what its own release streamed, not an enclosing release's groups."""
    if room_mib is None or int(room_mib) <= 0 or _env_off(PARTIAL_RESIDENT_ENV):
        return 0
    try:
        import torch
        from diffusers.hooks import group_offloading as go

        groups = _offload_groups(module)
        if not groups:
            return 0
        top = [g for g in groups if getattr(g, "offload_leader", None) is module]
        ordered = top + [g for g in groups if getattr(g, "offload_leader", None) is not module]
        is_torchao = getattr(go, "_is_torchao_tensor", lambda t: False)
        onload = torch.device(device)
        left = int(room_mib) * 1024 * 1024
        kept = 0
        try:
            from .diffusion_stable_arena import pin_group_in_range, stable_repin_wanted
            stable = stable_repin_wanted(module)
        except Exception:  # noqa: BLE001 - re-pins move the stock way
            stable = False
        _tensors = _resident_tensors
        disable = getattr(getattr(torch, "compiler", None), "disable", None)

        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        # Groups of this module still on the copy stream; with none, nothing can be in flight to wait for.
        state = getattr(module, "_unsloth_stream_state", None)
        if not isinstance(state, dict):
            state = {"streamed": 1}
            try:
                module._unsloth_stream_state = state
            except AttributeError:
                pass

        def _resident_onload(stream: Any) -> Callable[[], None]:
            # a prefetching predecessor skips its own copy-stream wait and relies on this onload_ to do it
            def onload_(*args: Any, **kwargs: Any) -> None:
                # event-fenced prefetch: streamed groups wait on their own copy; a resident block may start the forward's fill
                kick = state.get("kick")
                if callable(kick):
                    kick()
                if stream is not None and state["streamed"] and not state.get("fenced"):
                    stream.synchronize()

            return disable(onload_) if callable(disable) else onload_

        noop = disable(_noop) if callable(disable) else _noop

        try:
            module._unsloth_resident_room = int(room_mib)
            module._unsloth_resident_device = onload
        except AttributeError:
            pass
        # Tied weights (an LM head sharing the embedding, as in Qwen3 text encoders) sit in two groups. A resident
        # group next to a streamed partner breaks: the partner's offload points the shared tensor back at its host
        # copy and the resident group's no-op onload never brings it back. Such groups go resident together or not.
        tied = _tied_group_closures(ordered, _tensors)
        done: set = set()
        for group in ordered:
            if getattr(group, "offload_to_disk_path", None):
                break
            if id(group) in done:
                continue
            closure = tied.get(id(group)) or [group]
            done.update(id(g) for g in closure)
            unique: dict = {}
            for member in closure:
                for t in _tensors(member):
                    unique.setdefault(id(t), t)
            tensors = list(unique.values())
            need = sum(sum(_storage_nbytes(t)) for t in tensors)
            if all(getattr(g, "_unsloth_resident", False) for g in closure):
                left -= need
                continue
            if (
                need > left
                or any(getattr(g, "offload_to_disk_path", None) for g in closure)
                or (only is not None and any(id(g) not in only for g in closure))
            ):
                # too large (a later, smaller group may still fit), or another release's group
                continue
            for member in closure:
                cpu = getattr(member, "cpu_param_dict", None) or {}
                members_tensors = _tensors(member)
                placed: set = set()
                if stable and onload.type == "cuda":
                    # The step graph recorded the resident groups' addresses: a re-pin lands where the last one did.
                    try:
                        placed = pin_group_in_range(
                            module,
                            member,
                            members_tensors,
                            lambda t: not _placed_on(t, onload.type, is_torchao),
                            lambda t, cpu = cpu: cpu.get(t, t),
                            onload,
                            go._swap_torchao_tensor,
                        )
                    except Exception as exc:  # noqa: BLE001 - the stock move still places the group
                        placed = set()
                        # Unmap an empty range so the stock move below does not pay for a second copy.
                        try:
                            from .diffusion_stable_arena import release_group_range
                            release_group_range(module, member, members_tensors)
                        except Exception:  # noqa: BLE001
                            pass
                        if logger is not None:
                            logger.warning(
                                "diffusion.memory: stable re-pin range unavailable (%s); the group moves the stock way",
                                exc,
                            )
                for t in members_tensors:
                    if id(t) in placed or _placed_on(t, onload.type, is_torchao):
                        continue
                    moved = cpu.get(t, t).to(onload)
                    if is_torchao(t):
                        go._swap_torchao_tensor(t, moved)
                    else:
                        t.data = moved
            for i, member in enumerate(closure):
                if getattr(member, "_unsloth_resident", False):
                    continue
                # host copies stay: release_resident_groups streams the group again for an oversized request
                member._unsloth_streamed_hooks = (
                    member.__dict__.get("onload_"),
                    member.__dict__.get("offload_"),
                )
                member.onload_ = _resident_onload(getattr(member, "stream", None))
                member.offload_ = noop
                member._unsloth_resident = True
                member._unsloth_resident_bytes = need if i == 0 else 0
                member._unsloth_tied_groups = [g for g in closure if g is not member]
            left -= need
            kept += need
        state["streamed"] = _streamed_group_count(ordered)
        if kept:
            _bump_placement_epoch()
        if kept and onload.type == "cuda":
            from .diffusion_cuda_graph import synchronize_outside_capture
            synchronize_outside_capture(onload)
        if logger is not None and kept:
            logger.info(
                "diffusion.memory: %s keeps %d MiB resident (%d of %d offload groups); the rest streams",
                type(module).__name__,
                kept >> 20,
                sum(1 for g in ordered if getattr(g, "_unsloth_resident", False)),
                len(ordered),
            )
        return kept >> 20
    except Exception as exc:  # noqa: BLE001 - streaming every group is the safe state
        if logger is not None:
            logger.warning("diffusion.memory: partial residency skipped (%s)", exc)
        return 0


def _tied_group_closures(groups: list, tensors_of: Callable[[Any], list]) -> dict:
    """id(group) -> every group that shares a tensor with it, transitively (only for groups that share any)."""
    owners: dict[int, list] = {}
    for group in groups:
        for t in tensors_of(group):
            owners.setdefault(id(t), []).append(group)
    closures: dict = {}
    for group in groups:
        if id(group) in closures:
            continue
        stack, seen = [group], {id(group): group}
        while stack:
            current = stack.pop()
            for t in tensors_of(current):
                for other in owners.get(id(t), ()):
                    if id(other) not in seen:
                        seen[id(other)] = other
                        stack.append(other)
        if len(seen) > 1:
            members = [g for g in groups if id(g) in seen]
            for g in members:
                closures[id(g)] = members
    return closures


def _bump_placement_epoch() -> None:
    try:
        from .diffusion_offload_prefetch import bump_placement_epoch
        bump_placement_epoch()
    except Exception:  # noqa: BLE001 - no graph layer to invalidate
        pass


def _streamed_group_count(groups: list) -> int:
    return sum(
        1
        for g in groups
        if getattr(g, "stream", None) is not None and not getattr(g, "_unsloth_resident", False)
    )


def _resident_tensors(group: Any) -> list:
    """Every tensor of an offload group, each once (its modules' parameters, their buffers, then the group's own)."""
    out: list = []
    seen: set[int] = set()
    for t in (
        [p for m in group.modules for p in m.parameters()]
        + [b for m in group.modules for b in m.buffers()]
        + list(group.parameters or [])
        + list(group.buffers or [])
    ):
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


def _release_group(group: Any) -> None:
    onload_, offload_ = getattr(group, "_unsloth_streamed_hooks", (None, None))
    for name, fn in (("onload_", onload_), ("offload_", offload_)):
        if fn is not None:
            setattr(group, name, fn)
        else:
            group.__dict__.pop(name, None)
    group._unsloth_resident = False
    group.offload_()
    _bump_placement_epoch()


def _release_group_range(owner: Any, group: Any) -> None:
    """A released group's weights are on their host copies again: its stable re-pin range gives the physical memory
    back (the addresses stay reserved)."""
    if "_unsloth_repin_range" in getattr(group, "__dict__", {}):
        from .diffusion_stable_arena import release_group_range
        release_group_range(owner, group, _resident_tensors(group))


def release_resident_groups(
    pipe: Any,
    need_mib: int,
    logger: Any = None,
    denoisers_only: bool = False,
    reason: str = "an oversized request",
    device: Any = None,
) -> Optional[Callable[[], None]]:
    """Stream resident offload groups again until ``need_mib`` is freed (text encoders first, then the denoiser's
    blocks from the last); returns a callable pinning exactly those groups again, or None when nothing was resident.
    A request larger than the measured placement reserved (reference images, a bigger canvas, a batch) then runs as
    the flat plan. ``denoisers_only`` leaves the text encoders' groups alone (the prompt encode needs them).
    ``device``: only modules resident on that card."""
    released: list = []
    streamed: set = set()

    def restore(strict: bool = False) -> None:
        """Pin the released groups again. ``strict``: raise when any of them is still not resident afterwards (the
        re-pin catches its own failures and leaves those groups streamed, which callers that must retry need to
        know)."""
        for module in released:
            _keep_groups_resident(
                module,
                module._unsloth_resident_room,
                module._unsloth_resident_device,
                logger,
                only = streamed,
            )
        if strict:
            left = [
                g
                for module in released
                for g in (_offload_groups(module) or [])
                if id(g) in streamed and not getattr(g, "_unsloth_resident", False)
            ]
            if left:
                raise RuntimeError(f"re-pinning resident offload groups left {len(left)} streamed")

    try:
        import torch

        modules = [
            m
            for m in (getattr(pipe, "components", {}) or {}).values()
            if getattr(m, "_unsloth_resident_room", None)
            and not (denoisers_only and _is_text_encoder_module(pipe, m))
        ]
        if device is not None:
            from .diffusion_vae_oom import _same_card
            modules = [
                m
                for m in modules
                if _same_card(getattr(m, "_unsloth_resident_device", None) or device, device)
            ]
        if not modules or int(need_mib) <= 0:
            return None
        modules.sort(key = lambda m: 0 if _is_text_encoder_module(pipe, m) else 1)
        left = int(need_mib) * 1024 * 1024
        for module in modules:
            for group in reversed(_offload_groups(module) or []):
                if left <= 0:
                    break
                if not getattr(group, "_unsloth_resident", False):
                    continue
                # Before demoting: a release that fails part way must still wait on the copy stream.
                state = getattr(module, "_unsloth_stream_state", None)
                if isinstance(state, dict):
                    state["streamed"] = 1
                if module not in released:
                    released.append(module)
                # tied partners stream together (a resident partner would keep reading the host copy)
                for member in [group, *(getattr(group, "_unsloth_tied_groups", None) or [])]:
                    if not getattr(member, "_unsloth_resident", False):
                        continue
                    # Recorded before demoting: a release that fails part way is still re-pinned by ``restore``.
                    streamed.add(id(member))
                    _release_group(member)
                    _release_group_range(module, member)
                    left -= int(getattr(member, "_unsloth_resident_bytes", 0))
        if not released:
            return None
        for module in released:
            state = getattr(module, "_unsloth_stream_state", None)
            if isinstance(state, dict):
                state["streamed"] = _streamed_group_count(_offload_groups(module))
        device = getattr(released[0], "_unsloth_resident_device", None)
        if device is not None and torch.device(device).type == "cuda":
            from .diffusion_cuda_graph import synchronize_outside_capture
            synchronize_outside_capture(device)
        if logger is not None:
            logger.info(
                "diffusion.memory: streaming %d MiB of resident groups for %s",
                (int(need_mib) * 1024 * 1024 - max(left, 0)) >> 20,
                reason,
            )

        return restore
    except Exception as exc:  # noqa: BLE001 - the guard still refuses what cannot fit
        if logger is not None:
            logger.warning("diffusion.memory: releasing resident groups failed (%s)", exc)
        # groups released before the failure stay streamed until the caller re-pins them: hand back their restore
        return restore if released else None


DECODE_LEND_ATTR = "_unsloth_decode_lend"
# Kept free beside the decode's learned peak x margin when deciding whether lent encoder groups must stream again.
_DECODE_SLACK_MIB = 256


def decode_lend_from_plan(plan: Any) -> Optional[tuple[int, int, int, bool]]:
    """``(lent MiB, decode peak MiB, bucket pixels, quadratic)`` when ``plan`` lent encoder residency to the decode."""
    estimates = getattr(plan, "estimates", None) or {}
    lend = estimates.get("decode_lend_text_encoder_mib")
    if not lend or not getattr(plan, "resident_text_encoder_mib", None):
        return None
    return (
        int(lend),
        int(estimates.get("measured_decode_peak_mib") or 0),
        int(
            estimates.get("measured_decode_peak_pixels")
            or DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT
        ),
        bool(estimates.get("measured_decode_peak_quadratic")),
    )


def decode_need_mib(
    lend: Any,
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
) -> int:
    """MiB the decode of a ``width x height x batch`` request needs above the allocation at its start (margin
    included), scaled from the learned decode peak the way the store scales (``diffusion_measured_peaks``)."""
    from .diffusion_measured_peaks import resolution_bucket

    _, decode_mib, covered, quadratic = lend
    w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
    h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
    batch = max(1, int(batch_size or 1))
    covered = max(1, int(covered))
    scale = 1.0
    if quadratic:
        if w * h > covered:
            scale = (resolution_bucket(w, h, 1) / float(covered)) ** 2
        scale *= batch
    elif w * h * batch > covered:
        scale = resolution_bucket(w, h, batch) / float(covered)
    return int(math.ceil(float(decode_mib) * scale * _MEASURED_PEAK_MARGIN))


def release_for_decode(
    pipe: Any,
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    device: Any = None,
    logger: Any = None,
) -> Optional[Callable[[], None]]:
    """If free memory (driver + the allocator cache outside CUDA graph pools) is short of the learned decode peak x margin, stream the shortfall of
    lent groups again (text encoders first) and return the restore; else None. Mirrors ComfyUI's
    ``load_models_gpu(memory_required=...)``. The decode is untouched, so output is bit-identical."""
    lend = getattr(pipe, DECODE_LEND_ATTR, None)
    if not lend:
        return None
    try:
        import torch

        dev = torch.device(device if device is not None else "cuda")
        if dev.type != "cuda":
            return None
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        need = decode_need_mib(lend, width = width, height = height, batch_size = batch_size)
        free, total = (int(x) for x in torch.cuda.mem_get_info(dev))
        # keep the VRAM Budget reserve free, like every other decode-time check
        free -= user_budget_reserve_bytes(dev, total)
        # only the cache an ordinary decode allocation can reuse: a CUDA graph pool's free blocks are not
        cached = max(
            0,
            int(torch.cuda.memory_reserved(dev))
            - int(torch.cuda.memory_allocated(dev))
            - _graph_held_cache_bytes(dev.index),
        )
        short = need + _DECODE_SLACK_MIB - ((free + cached) >> 20)
    except Exception as exc:  # noqa: BLE001 - unreadable: the decode's OOM retry still frees the card
        if logger is not None:
            logger.debug("diffusion.memory: decode residency check skipped (%s)", exc)
        return None
    if short <= 0:
        return None
    return release_resident_groups(pipe, short, logger, reason = "the VAE decode", device = dev)


def resident_group_mib(pipe: Any) -> int:
    """MiB an oversized request could release: every resident offload group of the pipeline's components."""
    total = 0
    for module in (getattr(pipe, "components", {}) or {}).values():
        if not getattr(module, "_unsloth_resident_room", None):
            continue
        for group in _offload_groups(module) or []:
            if getattr(group, "_unsloth_resident", False):
                total += int(getattr(group, "_unsloth_resident_bytes", 0))
    return total >> 20


def _any_layer_split(pipe: Any) -> bool:
    from .diffusion_layer_split import is_layer_split
    from .diffusion_multigpu import DENOISER_ATTRS
    return any(is_layer_split(getattr(pipe, name, None)) for name in DENOISER_ATTRS)


def hook_resident_denoiser(
    pipe: Any,
    device: Any,
    logger: Any = None,
    resident: bool = True,
) -> bool:
    """Give a hookless resident torchao transformer block-level offload hooks, every group kept resident, so
    release_resident_groups can stream part of it. On demand, not at load: hooks at load recompile every load's second
    render. True when installed. ``resident=False``: every group streams (the load's flat plan), for a transformer
    that cannot come back to the card whole. A failure whose own return to the card fails raises, the transformer
    split between host and card with no hooks (return_torchao_denoiser / hook_resident_denoiser(resident=False) are
    the ways out)."""
    transformer = getattr(pipe, "transformer", None)
    if (
        transformer is None
        or _offload_groups(transformer)
        or not _pipe_denoisers_hold_torchao(pipe)
        or _any_layer_split(pipe)
    ):
        return False
    import torch

    onload = torch.device(device)
    if onload.type != "cuda":
        return False
    dit_mib = _module_pin_mib(transformer)
    # the load's _torchao_stream_pinnable rule: unpinned, the apply copies the whole transformer to pageable host RAM
    if not _streamed_pin_plan(dit_mib, 0, device = onload)[0]:
        if logger is not None:
            logger.info(
                "diffusion.memory: the resident transformer keeps no offload hooks (%d MiB cannot be pinned)",
                dit_mib,
            )
        return False
    torch_hooks = _module_forward_hook_ids(transformer)
    try:
        import inspect

        from diffusers.hooks import apply_group_offloading

        install_group_offload_buffer_restore()
        install_group_offload_hooks_eager()
        gkwargs: dict[str, Any] = {
            "onload_device": onload,
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": DEFAULT_GROUP_BLOCKS,
            "use_stream": True,
        }
        params = inspect.signature(apply_group_offloading).parameters
        for name in ("non_blocking", "record_stream"):
            if name in params:
                gkwargs[name] = True
        if "low_cpu_mem_usage" in params:
            gkwargs["low_cpu_mem_usage"] = False
        pinned_mib = [0]
        _apply_group_offloading_packed(
            apply_group_offloading,
            transformer,
            _torchao_group_offload_kwargs(transformer, gkwargs, pinned_mib),
            logger,
        )
        if not _pin_top_level_group(transformer, logger, pinned_mib, device = onload):
            _skip_top_level_copy_back(transformer, logger)
        if gkwargs.get("record_stream"):
            install_group_prefetch(transformer, onload, logger)
        kept = 0
        if resident:
            kept = _keep_groups_resident(transformer, dit_mib + 1, onload, logger)
            if not all(
                getattr(g, "_unsloth_resident", False) for g in _offload_groups(transformer)
            ):
                raise RuntimeError("not every offload group stayed resident")
    except Exception as exc:  # noqa: BLE001 - the guard still refuses what cannot fit
        # the apply already moved the weights to their host copies: back to the card, as before the call (a return
        # that itself fails raises: the caller owns the way back)
        _unhook_and_return(transformer, onload, torch_hooks)
        if logger is not None:
            logger.warning("diffusion.memory: resident transformer left without hooks (%s)", exc)
        return False
    if logger is not None:
        logger.info(
            "diffusion.memory: the resident transformer takes offload hooks (%d MiB kept resident) so this request "
            "can stream part of it",
            kept,
        )
    return True


_FORWARD_HOOK_DICTS = (
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
)


def _module_forward_hook_ids(module: Any) -> dict[str, set]:
    return {name: set(getattr(module, name, None) or ()) for name in _FORWARD_HOOK_DICTS}


def _unhook_and_return(transformer: Any, onload: Any, torch_hooks: Optional[dict]) -> None:
    _remove_group_offload_hooks(transformer)
    if torch_hooks is not None:
        _drop_module_forward_hooks_since(transformer, torch_hooks)
    from .diffusion_offload_prefetch import PREFETCHER_ATTR

    for attr in (PREFETCHER_ATTR, "_unsloth_stream_state"):
        transformer.__dict__.pop(attr, None)
    _return_module_tensors(transformer, onload)


def return_torchao_denoiser(
    pipe: Any,
    device: Any,
    logger: Any = None,
    torch_hooks: Optional[dict] = None,
) -> None:
    """The way back to where a hookless resident torchao transformer started, from wherever a failed
    hook_resident_denoiser left it: streamed under offload hooks (settled) -> every group resident again, the state
    a successful hook_resident_denoiser leaves; split with no hooks -> whole on the card. Raises if it is not there."""
    import torch

    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return
    onload = torch.device(device)
    if _offload_groups(transformer):
        _keep_groups_resident(transformer, _module_host_mib(transformer) + 1, onload, logger)
        if not all(getattr(g, "_unsloth_resident", False) for g in _offload_groups(transformer)):
            raise RuntimeError("the transformer's offload groups are not all resident again")
        return
    _unhook_and_return(transformer, onload, torch_hooks)


def module_forward_hook_ids(module: Any) -> dict[str, set]:
    return _module_forward_hook_ids(module)


def _drop_module_forward_hooks_since(module: Any, before: dict[str, set]) -> None:
    """Remove the torch forward hooks registered on ``module`` after ``before`` was taken (the group prefetcher's)."""
    for name, ids in before.items():
        hooks = getattr(module, name, None)
        if hooks is not None:
            for key in set(hooks) - ids:
                hooks.pop(key, None)


def _return_module_tensors(module: Any, onload: Any) -> None:
    """Move ``module``'s weights back to ``onload`` after its hooks are gone. A torchao weight on its host copy can
    still report the card (``Module.to`` skips or fails on it), so its inner tensors move under a new wrapper."""
    import torch
    from diffusers.hooks import group_offloading as go

    is_torchao = getattr(go, "_is_torchao_tensor", lambda t: False)
    for t in [*module.parameters(), *module.buffers()]:
        if _placed_on(t, onload.type, is_torchao):
            continue
        if is_torchao(t):
            names, ctx = t.__tensor_flatten__()
            moved = type(t).__tensor_unflatten__(
                {name: getattr(t, name).to(onload) for name in names}, ctx, t.size(), t.stride()
            )
            go._swap_torchao_tensor(t, moved)
        else:
            t.data = t.data.to(onload)
    if onload.type == "cuda":
        from .diffusion_cuda_graph import synchronize_outside_capture
        synchronize_outside_capture(onload)


def _unwind_group_offload(
    pipe: Any,
    snapshots: list,
    logger: Any = None,
) -> bool:
    """Undo a partial ``_apply_group_offload``: group hooks, the prefetchers' torch hooks and attributes, deferred
    pins. The weights stay where the hooks left them (host copies or the card); the caller's retry moves them."""
    try:
        from .diffusion_offload_prefetch import PREFETCHER_ATTR
        from .diffusion_small_host import ENCODER_PREFETCH_ATTR

        for module, before in snapshots:
            _remove_group_offload_hooks(module)
            _drop_module_forward_hooks_since(module, before)
            for attr in (PREFETCHER_ATTR, ENCODER_PREFETCH_ATTR, "_unsloth_stream_state"):
                module.__dict__.pop(attr, None)
            _drop_deferred_pinning(pipe, module)
        return bool(snapshots)
    except Exception as exc:  # noqa: BLE001 - the original failure propagates
        if logger is not None:
            logger.warning("diffusion.memory: could not unwind partial group offload (%s)", exc)
        return False


def _is_text_encoder_module(pipe: Any, module: Any) -> bool:
    for name, component in (getattr(pipe, "components", {}) or {}).items():
        if component is module:
            return str(name).startswith("text_encoder")
    return False


# "0": pin the denoiser back after every text encoder instead of once per encode_prompt call.
ENCODE_RELEASE_ONCE_ENV = "UNSLOTH_DIFFUSION_ENCODE_RELEASE_ONCE"
# "0": the prompt encode always streams the whole surplus, even when free memory holds the encoder's need.
ENCODE_SHORTFALL_ENV = "UNSLOTH_DIFFUSION_ENCODE_SHORTFALL"
_ENCODE_SLACK_MIB = 512
# Per token, in hidden_size x 2 bytes (plus one per layer: output_hidden_states keeps every layer).
_ENCODE_TOKEN_HIDDEN_FACTOR = 32
# Vision inputs are not priced.
_ENCODE_VISUAL_KWARGS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "images",
)


def _encoder_width(module: Any) -> tuple[Optional[int], int, int]:
    config = getattr(module, "config", None)
    for cfg in (config, getattr(config, "text_config", None)):
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if hidden:
            heads = (
                getattr(cfg, "num_attention_heads", None) or getattr(cfg, "num_heads", None) or 1
            )
            layers = (
                getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None) or 0
            )
            return int(hidden), int(heads), int(layers)
    return None, 1, 0


def encode_need_mib(module: Any, args: Any, kwargs: Any) -> Optional[int]:
    """MiB one encoder call needs: the ``depth + 1`` largest streamed groups plus prompt activations, with margin.
    None when unpriceable (visual inputs, no token ids, unknown width)."""
    try:
        import torch

        from .diffusion_offload_prefetch import prefetch_depth

        kwargs = kwargs or {}
        if any(kwargs.get(name) is not None for name in _ENCODE_VISUAL_KWARGS):
            return None
        ids = kwargs.get("input_ids")
        if ids is None:
            ids = next(
                (
                    a
                    for a in (args or ())
                    if isinstance(a, torch.Tensor)
                    and a.dim() == 2
                    and not torch.is_floating_point(a)
                ),
                None,
            )
        if ids is None:
            embeds = kwargs.get("inputs_embeds")
            if not isinstance(embeds, torch.Tensor) or embeds.dim() != 3:
                return None
            batch, tokens = int(embeds.shape[0]), int(embeds.shape[1])
        elif isinstance(ids, torch.Tensor) and ids.dim() == 2:
            batch, tokens = int(ids.shape[0]), int(ids.shape[1])
        else:
            return None
        hidden, heads, layers = _encoder_width(module)
        if not hidden:
            return None
        streamed = sorted(
            (
                _group_nbytes(g)
                for g in (_offload_groups(module) or ())
                if not getattr(g, "_unsloth_resident", False)
            ),
            reverse = True,
        )
        window = sum(streamed[: prefetch_depth() + 1])
        acts = (
            batch
            * tokens
            * (hidden * 2 * (_ENCODE_TOKEN_HIDDEN_FACTOR + layers) + heads * tokens * 4)
        )
        return (
            int(math.ceil((window + acts) / float(1 << 20) * _MEASURED_PEAK_MARGIN))
            + _ENCODE_SLACK_MIB
        )
    except Exception:  # noqa: BLE001 - unpriced: the whole surplus streams
        return None


def encode_available_mib(device: Any = None) -> Optional[int]:
    """Driver free less the VRAM Budget reserve, plus allocator cache outside CUDA graph pools, in MiB."""
    try:
        import torch

        dev = torch.device(device if device is not None else "cuda")
        if dev.type != "cuda":
            return None
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        free, total = (int(x) for x in torch.cuda.mem_get_info(dev))
        free -= user_budget_reserve_bytes(dev, total)
        cached = max(
            0,
            int(torch.cuda.memory_reserved(dev))
            - int(torch.cuda.memory_allocated(dev))
            - _graph_held_cache_bytes(dev.index),
        )
        # clamp after adding the cache: a reserve larger than driver free still claims part of it
        return max(free + cached, 0) >> 20
    except Exception:  # noqa: BLE001
        return None


def _encode_device(pipe: Any) -> Any:
    for module in (getattr(pipe, "components", {}) or {}).values():
        device = getattr(module, "_unsloth_resident_device", None)
        if device is not None and getattr(module, "_unsloth_resident_room", None):
            return device
    return None


def encode_release_mib(pipe: Any, module: Any, args: Any, kwargs: Any, surplus: int) -> int:
    """Only the shortfall (ComfyUI's ``free_memory(memory_required - free)``), at most ``surplus``; unpriced: all."""
    if _env_off(ENCODE_SHORTFALL_ENV):
        return surplus
    need = encode_need_mib(module, args, kwargs)
    if need is None:
        return surplus
    available = encode_available_mib(_encode_device(pipe))
    if available is None:
        return surplus
    return max(0, min(int(surplus), int(need) - int(available)))


def install_encode_release(
    pipe: Any,
    plan: Any,
    logger: Any = None,
) -> int:
    """While a text encoder runs, stream the whole-resident (or widened partial) denoiser back to the flat room (the
    partial placement's encode state); pin it back on return, or once encode_prompt returns (one move per prompt with
    several encoders). Returns the number of encoders hooked."""
    estimates = getattr(plan, "estimates", None) or {}
    encode_room = estimates.get("encode_resident_transformer_mib")
    whole = getattr(plan, "resident_transformer_mib", None)
    if encode_room is None or not whole:
        return 0
    surplus = int(whole) - max(int(encode_room), 0)
    if surplus <= 0:
        return 0
    try:
        import weakref

        import torch

        try:
            pipe_ref = weakref.ref(pipe)  # the hooks live on the encoder, which the pipe owns
        except TypeError:
            pipe_ref = lambda: pipe  # noqa: E731 - not weak-referenceable
        disable = getattr(getattr(torch, "compiler", None), "disable", None)
        pending: list = []
        asked = [0]
        active = [0]

        def _before(
            module: Any,
            args: Any,
            kwargs: Any = None,
        ) -> None:
            active[0] += 1
            owner = pipe_ref()
            if owner is None or active[0] > 1:
                return  # nested encoder call: the outer one already released
            # Each encoder prices its own shortfall; a prompt never streams more than the surplus in total.
            need = min(
                encode_release_mib(owner, module, args, kwargs, surplus),
                surplus - asked[0],
            )
            if need <= 0:
                return
            asked[0] += need
            restore = release_resident_groups(
                owner, need, logger, denoisers_only = True, reason = "the prompt encode"
            )
            # No empty_cache: the allocator frees cached segments and retries on OOM; flushing per prompt cost a sync.
            if restore is not None:
                pending.append(restore)

        # encode_prompt calls in flight: their encoders keep one release and the outermost return pins back
        scope = [0]

        def _restore_pending() -> None:
            asked[0] = 0
            while pending:
                pending.pop()()

        def _after(module: Any, args: Any, output: Any) -> None:
            active[0] = max(0, active[0] - 1)
            if active[0] > 0 or scope[0] > 0:
                return  # the next encoder of this prompt reuses the room; encode_prompt's return pins back
            _restore_pending()

        before = disable(_before) if callable(disable) else _before
        after = disable(_after) if callable(disable) else _after
        hooked = 0
        for name, module in (getattr(pipe, "components", {}) or {}).items():
            if not str(name).startswith("text_encoder") or not isinstance(module, torch.nn.Module):
                continue
            handles = (
                module.register_forward_pre_hook(before, with_kwargs = True),
                # always_call: an encode that raises (cancel, OOM) still pins the denoiser back
                module.register_forward_hook(after, always_call = True),
            )
            module._unsloth_encode_release = handles
            hooked += 1
        encode = getattr(pipe, "encode_prompt", None)
        if (
            hooked > 1
            and callable(encode)
            and not _env_off(ENCODE_RELEASE_ONCE_ENV)
            and not getattr(encode, "_unsloth_encode_release_scope", False)
        ):

            @functools.wraps(encode)
            def encode_prompt(*args: Any, **kwargs: Any) -> Any:
                scope[0] += 1
                try:
                    return encode(*args, **kwargs)
                finally:
                    scope[0] -= 1
                    if scope[0] == 0:
                        _restore_pending()

            encode_prompt._unsloth_encode_release_scope = True
            try:
                pipe.encode_prompt = encode_prompt
            except Exception:  # noqa: BLE001 - a read-only attribute keeps the per-encoder pin-back
                pass
        if hooked and logger is not None:
            logger.info(
                "diffusion.memory: the prompt encode streams %d MiB of the resident transformer (%d MiB stay), "
                "pinned again before step 0",
                surplus,
                max(int(encode_room), 0),
            )
        return hooked
    except Exception as exc:  # noqa: BLE001 - fall back to the partial placement rather than risk the encode
        if logger is not None:
            logger.warning(
                "diffusion.memory: encode release not installed (%s); the transformer streams past the flat room",
                exc,
            )
        release_resident_groups(
            pipe, surplus, logger, denoisers_only = True, reason = "the partial placement"
        )
        return 0


def measured_request_extra_mib(
    pipe: Any,
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    condition_pixels: int = 0,
    cfg: Optional[bool] = None,
    unpriced_references: bool = False,
    workflow: Optional[str] = None,
    tier_peak_mib: Optional[int] = None,
) -> int:
    """MiB a request needs beyond what the measured placement reserved (0 when it fits or none was made).  ``tier_peak_mib``:
    a deferred profile's tier peak (``TIER_MOVED_ATTR``), raising eager pricing; None keeps the flat estimate. ``workflow``:
    a workflow other than text-to-image also reads the peak recorded under its own key (``with_workflow``). ``cfg``:
    whether the request runs classifier-free guidance (None: unknown, treated as running it). ``unpriced_references``:
    the request carries reference images whose latents join the denoiser's sequence but are not in
    ``condition_pixels`` (FLUX.2 Klein's primary reference): a placement made from a dense default / max tier seed,
    measured on text-to-image, then sizes the request from the flat estimate."""
    reserve = getattr(pipe, "_unsloth_measured_reserve", None)
    if not reserve:
        return 0
    headroom, family, speed_mode = reserve[:3]
    dense_mib = reserve[3] if len(reserve) > 3 else None
    compute_bytes = reserve[4] if len(reserve) > 4 and reserve[4] else 2
    identity = getattr(pipe, "_unsloth_peak_identity", None)
    sized = dict(
        width = width,
        height = height,
        batch_size = batch_size,
        dense_transformer_mib = dense_mib,
        compute_bytes = compute_bytes,
    )
    # A seeded family keeps the seed and the learned peak the load placed from: a peak this session's renders recorded
    # after the load only places the NEXT load, it must not stream resident groups for a request the same size as one
    # that just completed. An unseeded family was placed from its learned peak, which is all there is to check against.
    sized["phase_seeds"] = bool(getattr(pipe, "_unsloth_phase_seeds", False))
    sized["dense_tier_seeds"] = bool(getattr(pipe, "_unsloth_dense_tier_seeds", False))
    sized["static_seed"] = bool(reserve[8]) if len(reserve) > 8 and reserve[8] is not None else True
    sized["dense_bf16"] = bool(len(reserve) > 10 and reserve[10])
    sized["small_host_int8"] = bool(len(reserve) > 11 and reserve[11])
    need = measured_image_runtime_mib(family, speed_mode, **sized)
    learned = reserve[5] if len(reserve) > 5 else None
    # placed from a peak recorded without CFG: a request that runs it doubles the denoiser batch no measurement covers
    cfg_unmeasured = bool(len(reserve) > 9 and reserve[9]) and cfg is not False
    # tier and phase seeds were measured on text-to-image: an unsized reference is not covered
    tier_seeded = str(speed_mode or "") in _MEASURED_PEAK_SPEED_MODES and (
        (dense_mib is not None and sized["dense_tier_seeds"])
        or (dense_mib is None and sized["phase_seeds"])
    )
    tier_moved = bool(getattr(pipe, TIER_MOVED_ATTR, False))
    tier_peak = int(tier_peak_mib) if tier_moved and tier_peak_mib else None
    if (unpriced_references and tier_seeded) or (
        # eager-only seeds: same invalidation as a learned peak
        (learned or need is None or sized["dense_bf16"] or sized["small_host_int8"])
        and (
            getattr(pipe, "_unsloth_peak_static", False)
            or (tier_moved and tier_peak is None)
            or getattr(pipe, "_unsloth_loras", None)
            or cfg_unmeasured
        )
    ):
        # the learned peak no longer describes this pipeline (runtime LoRA adapters, a planned VAE saver that did not
        # engage): requests size from the flat estimate
        flat = estimate_image_runtime_mib(
            width = width,
            height = height,
            batch_size = batch_size,
            family = family,
            condition_pixels = condition_pixels,
        )
        return max(0, int(flat) - int(headroom))
    if learned:
        # the learned peak the load reserved, scaled here (not reread from the store, which another worker may evict
        # or a failed write may leave empty while this pipeline stays loaded)
        from .diffusion_measured_peaks import resolution_bucket

        w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
        h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
        batch = max(1, int(batch_size or 1))
        covered = (
            reserve[6]
            if len(reserve) > 6 and reserve[6]
            else DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT
        )
        # the learned store's own scaling: a request past the covered bucket takes its bucket's ceiling, squared over
        # the image's pixels for a math-attention entry (linear over the batch)
        scale = 1.0
        if len(reserve) > 7 and reserve[7]:
            if w * h > int(covered):
                scale = (resolution_bucket(w, h, 1) / float(covered)) ** 2
            scale *= batch
        elif w * h * batch > int(covered):
            scale = resolution_bucket(w, h, batch) / float(covered)
        step = _MEASURED_PEAK_ROUND_MIB
        scaled = int(-(-(float(learned) * scale * _MEASURED_PEAK_MARGIN) // step) * step)
        seeded = need is not None
        need = scaled if need is None else max(need, scaled)
        if not seeded and isinstance(identity, tuple) and len(identity) == 3:
            stored = measured_image_runtime_mib(family, speed_mode, quant = identity[1], **sized)
            if stored is not None:
                need = max(need, stored)
    if need is None and isinstance(identity, tuple) and len(identity) == 3:
        need = measured_image_runtime_mib(family, speed_mode, quant = identity[1], **sized)
    if tier_peak is not None:
        need = tier_peak if need is None else max(int(need), tier_peak)
    cond = max(0, int(condition_pixels or 0)) * max(1, int(batch_size or 1))
    allowance = int(
        8192 * cond / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT) * _MEASURED_PEAK_MARGIN
    )
    if need is not None:
        need += allowance
    if isinstance(identity, tuple) and len(identity) == 3:
        keyed = with_workflow(identity[1], workflow)
        if keyed != identity[1]:
            own = None
            if (
                cfg is False
                and getattr(pipe, "_unsloth_peak_expects_cfg", False)
                and cfg_off_peaks_enabled()
            ):
                # a guidance-off render on a CFG-capable family records ``|cfg:off`` before the workflow part
                off = with_workflow(f"{without_cfg_off(identity[1])}|{CFG_OFF_PART}", workflow)
                own = measured_image_runtime_mib(family, speed_mode, quant = off, **sized)
            if own is None:
                own = measured_image_runtime_mib(family, speed_mode, quant = keyed, **sized)
            if own is not None:
                # the learned workflow peak is the other complete estimate: it was measured with the condition images
                # resident, so the allowance is not charged on top of it
                need = own if need is None else max(need, own)
    if need is None:
        return 0
    return max(0, int(need) - int(headroom))


def apply_memory_plan(
    pipe: Any,
    plan: MemoryPlan,
    *,
    device: str,
    placement_device: Optional[str] = None,
    logger: Any = None,
) -> tuple[str, bool]:
    """placement_device must be indexed (cuda:1); a bare cuda lets CPU offload fall back to cuda:0."""
    placement = placement_device or device
    try:
        pipe._unsloth_applied_plan = None
    except Exception:  # noqa: BLE001
        pass
    tiling_engaged = False
    if plan.vae_tiling:
        tiling_engaged = _enable_vae_saver(pipe, "enable_vae_tiling", "enable_tiling", logger)
    if plan.vae_slicing:
        _enable_vae_saver(pipe, "enable_vae_slicing", "enable_slicing", logger)

    def _fallback_to_model_offload() -> None:
        # Group plan set vae_tiling=False; model-offload fallback can OOM on decode, so enable tiling.
        nonlocal tiling_engaged
        pipe.enable_model_cpu_offload(device = placement)
        keep_cpu_weights_on_offload(pipe, logger)
        if not tiling_engaged:
            tiling_engaged = _enable_vae_saver(pipe, "enable_vae_tiling", "enable_tiling", logger)

    policy = plan.offload_policy
    if policy == OFFLOAD_MODEL:
        pipe.enable_model_cpu_offload(device = placement)
        keep_cpu_weights_on_offload(pipe, logger)
    elif policy == OFFLOAD_GROUP:
        # getattr, not attribute access: manually built / duck-typed plans predate this field.
        group_kwargs: dict[str, Any] = {
            "stream_text_encoders": bool(getattr(plan, "stream_text_encoders", False))
        }
        if not bool(getattr(plan, "stream_transformer", True)):
            group_kwargs["stream_transformer"] = False
        resident_mib = getattr(plan, "resident_transformer_mib", None)
        if resident_mib:
            group_kwargs["resident_transformer_mib"] = int(resident_mib)
        resident_te_mib = getattr(plan, "resident_text_encoder_mib", None)
        if resident_te_mib and group_kwargs["stream_text_encoders"]:
            group_kwargs["resident_text_encoder_mib"] = int(resident_te_mib)
        _set_group_offload_failure(pipe, None)
        applied = _apply_group_offload(pipe, placement, logger, **group_kwargs)
        failure = getattr(pipe, _GROUP_OFFLOAD_FAILURE_ATTR, None)
        if not applied and failure is not None and _is_oom(failure):
            _set_group_offload_failure(pipe, None)  # the retry records its own failure, if any
            lowered = _retry_group_offload_lower(pipe, plan, placement, logger, failure)
            if lowered is not None:
                applied = True
                try:
                    pipe._unsloth_applied_plan = lowered
                except Exception:  # noqa: BLE001
                    pass
        if not applied:
            if "stream_transformer" in group_kwargs and _pipe_denoisers_hold_torchao(pipe):
                raise RuntimeError(
                    "the text encoder could not be streamed beside the resident quantised "
                    "transformer, and whole-module offload cannot move torchao weights"
                )
            if _pipe_denoisers_hold_torchao(pipe) and not _model_offload_fits_quantised(plan):
                # Whole-module offload would onload the quantised transformer whole, which is what streaming avoided.
                raise RuntimeError(
                    "group offloading could not be set up for the quantised transformer, and it does not "
                    "fit the GPU whole for whole-module offload"
                )
            _fallback_to_model_offload()
            policy = OFFLOAD_MODEL
    elif policy == OFFLOAD_STREAMING:
        resident_mib = getattr(plan, "resident_transformer_mib", None)
        if resident_mib:
            _apply_streaming_offload(
                pipe, placement, logger, resident_transformer_mib = int(resident_mib)
            )
        else:
            _apply_streaming_offload(pipe, placement, logger)
    elif policy == OFFLOAD_SEQUENTIAL:
        try:
            pipe.enable_sequential_cpu_offload(device = placement)
        except Exception as exc:  # noqa: BLE001 - keep the model loadable
            if logger is not None:
                logger.warning(
                    "diffusion.memory: sequential offload failed (%s); "
                    "falling back to whole-module offload",
                    exc,
                )
            _fallback_to_model_offload()
            policy = OFFLOAD_MODEL
    elif (
        getattr(plan, "secondary_ordinal", None) is not None
        or getattr(plan, "layer_split", None) is not None
    ):
        if getattr(plan, "layer_split", None) is not None:
            from .diffusion_layer_split_plan import apply_layer_split_plan as _apply_split
        else:
            from .diffusion_multigpu import apply_split_plan as _apply_split
        if not _apply_split(pipe, plan, placement = placement, logger = logger):
            fallback = getattr(plan, "fallback_plan", None)
            if (
                fallback is not None
                and getattr(fallback, "secondary_ordinal", None) is None
                and getattr(fallback, "layer_split", None) is None
            ):
                # the policy one card would have used (group offload keeps a torchao denoiser loadable)
                result = apply_memory_plan(
                    pipe, fallback, device = device, placement_device = placement_device, logger = logger
                )
                # applied_plan() must report the fallback (or the lower re-plan it recorded), not the split
                if getattr(pipe, "_unsloth_applied_plan", None) is None:
                    try:
                        pipe._unsloth_applied_plan = fallback
                    except Exception:  # noqa: BLE001
                        pass
                return result
            _fallback_to_model_offload()
            policy = OFFLOAD_MODEL
            try:
                pipe._unsloth_applied_plan = replace(
                    plan,
                    offload_policy = OFFLOAD_MODEL,
                    secondary_ordinal = None,
                    secondary_components = None,
                    fallback_plan = None,
                    layer_split = None,
                )
            except Exception:  # noqa: BLE001
                pass
    else:
        from .diffusion_fast_load import fast_upload

        components = getattr(pipe, "components", None)
        modules = list(components.values()) if isinstance(components, dict) else []
        with fast_upload(modules, placement, logger = logger):
            pipe.to(placement)
    return policy, tiling_engaged


_GROUP_OFFLOAD_FAILURE_ATTR = "_unsloth_group_offload_failure"


class _GroupOffloadFailure(Exception):
    """Type and message of a failed group offload setup. The live exception is not kept: its traceback holds the
    failed frames, whose tensors would stay allocated through the very retry that needs the room."""

    def __init__(self, exc: BaseException) -> None:
        super().__init__(f"{type(exc).__name__}: {exc}")
        self.type_name = type(exc).__name__
        self.oom = _is_oom(exc)


def _set_group_offload_failure(pipe: Any, exc: Optional[BaseException]) -> None:
    """What made ``_apply_group_offload`` return False (it absorbs the exception so the caller can fall back)."""
    if exc is not None and not isinstance(exc, _GroupOffloadFailure):
        exc = _GroupOffloadFailure(exc)
    try:
        setattr(pipe, _GROUP_OFFLOAD_FAILURE_ATTR, exc)
    except Exception:  # noqa: BLE001
        pass


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, _GroupOffloadFailure):
        return exc.oom
    from .diffusion_batched import is_oom_error
    return is_oom_error(exc)


def applied_plan(pipe: Any, plan: Any) -> Any:
    """The plan ``apply_memory_plan`` actually placed: ``plan``, or the lower-residency re-plan a transient setup OOM
    forced (callers gate graphs, fused GEMMs and the encode release on residency)."""
    return getattr(pipe, "_unsloth_applied_plan", None) or plan


def _retry_group_offload_lower(
    pipe: Any, plan: MemoryPlan, placement: str, logger: Any, exc: BaseException
) -> Optional[MemoryPlan]:
    """One retry after an OOM while setting up group offload (another process can move VRAM between the plan's
    snapshot and placement): return every denoiser and text encoder to the host, empty the cache, and stream the
    denoisers and encoders with nothing kept resident. Returns the re-plan, or None (the caller refuses as before)."""
    try:
        import gc

        import torch

        for name, comp in (getattr(pipe, "components", {}) or {}).items():
            name = str(name)
            if not isinstance(comp, torch.nn.Module):
                continue
            if name in (
                "transformer",
                "transformer_2",
                "unconditional_transformer",
            ) or name.startswith("text_encoder"):
                _return_module_tensors(comp, torch.device("cpu"))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if logger is not None:
            logger.warning(
                "diffusion.memory: group offload setup hit OOM (%s); emptied the cache and re-planning once with "
                "the transformer and text encoders streamed",
                str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__,
            )
    except Exception as retry_exc:  # noqa: BLE001 - the caller's refusal stands
        if logger is not None:
            logger.warning(
                "diffusion.memory: lower-residency group offload retry failed (%s)", retry_exc
            )
        return None
    # outside the catch: a raise here left hooks on part of the pipe, which the caller's whole-module fallback rejects
    if not _apply_group_offload(pipe, placement, logger, stream_text_encoders = True):
        return None
    estimates = dict(plan.estimates or {})
    estimates.update(resident_transformer_mib = 0, resident_text_encoder_mib = 0, setup_oom_retry = True)
    return replace(
        plan,
        stream_text_encoders = True,
        stream_transformer = True,
        resident_transformer_mib = None,
        resident_text_encoder_mib = None,
        estimates = estimates,
        reasons = tuple(plan.reasons)
        + (
            "group offload setup hit OOM; re-planned once with the transformer and text encoders streamed",
        ),
    )


def _enable_vae_saver(pipe: Any, pipe_method: str, vae_method: str, logger: Any) -> bool:
    """Turn on a VAE memory saver, trying the pipeline shortcut first then the VAE submodule
    (some pipelines, e.g. Z-Image, only expose it on ``pipe.vae``). Returns whether it engaged."""
    for owner, method in ((pipe, pipe_method), (getattr(pipe, "vae", None), vae_method)):
        fn = getattr(owner, method, None)
        if not callable(fn):
            continue
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 - a VAE saver is an optimisation, never fatal
            if logger is not None:
                logger.warning("diffusion.memory: %s() failed: %s", method, exc)
    return False


def _pin_vision_embedding_device(module: Any) -> int:
    """Keeps Qwen3-VL's vision pos_embed on the compute device, which group offload otherwise leaves
    on CPU."""
    import types

    walk = getattr(module, "modules", None)
    if not callable(walk):
        return 0
    patched = 0
    for sub in walk():
        original = getattr(sub, "fast_pos_embed_interpolate", None)
        if not callable(original) or getattr(original, "_unsloth_device_pinned", False):
            continue

        def _on_grid_device(
            self: Any,
            grid_thw: Any,
            *,
            _original: Any = original,
        ) -> Any:
            out = _original(grid_thw)
            device = getattr(grid_thw, "device", None)
            return out.to(device) if device is not None and out.device != device else out

        _on_grid_device._unsloth_device_pinned = True  # type: ignore[attr-defined]
        sub.fast_pos_embed_interpolate = types.MethodType(_on_grid_device, sub)
        patched += 1
    return patched


# ``0`` never pins the streamed-encoder tiers, ``1`` always does.
GROUP_OFFLOAD_PIN_ENV = "UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN"
_PIN_RESERVE_MIN_MIB = _PIN_RESERVE_MIN_BYTES >> 20


def _module_host_mib(module: Any) -> int:
    """Pinned host MiB for ``module``; per tensor rounded to a power of two like torch's pinned allocator."""
    try:
        seen: set[int] = set()
        total = 0
        tensors = list(module.parameters(recurse = True)) + list(module.buffers(recurse = True))
        for tensor in tensors:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            for nbytes in _storage_nbytes(tensor):
                if nbytes > 0:
                    total += 1 << (nbytes - 1).bit_length()
        return total // (1024 * 1024)
    except Exception:  # noqa: BLE001 - an unsizeable module is priced as nothing to pin
        return 0


_HOST_PAGE = 4096


def _page_bytes(nbytes: int) -> int:
    return -(-int(nbytes) // _HOST_PAGE) * _HOST_PAGE


def _module_registered_mib(module: Any) -> int:
    """Pinned host MiB of ``module`` in registered host memory (``registered_host_copy``): each tensor page-rounded."""
    try:
        total = sum(
            _page_bytes(nbytes)
            for tensor in _unique_tensors(module)
            for nbytes in _storage_nbytes(tensor)
            if nbytes > 0
        )
        return -(-total // (1024 * 1024))
    except Exception:  # noqa: BLE001 - priced as the per-tensor pin
        return _module_host_mib(module)


def _module_offload_registered_mib(module: Any) -> int:
    """Registered MiB a group offload pins: page-rounded, counted once per owning submodule (UMT5's tied embedding pins twice)."""
    try:
        total = 0
        for sub in module.modules():
            seen: set[int] = set()
            for tensor in list(sub.parameters(recurse = False)) + list(sub.buffers(recurse = False)):
                if id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                total += sum(
                    _page_bytes(nbytes) for nbytes in _storage_nbytes(tensor) if nbytes > 0
                )
        return -(-total // (1024 * 1024))
    except Exception:  # noqa: BLE001 - priced as the per-tensor pin
        return _module_host_mib(module)


def _unique_tensors(module: Any) -> list:
    seen: set[int] = set()
    out = []
    for tensor in list(module.parameters(recurse = True)) + list(module.buffers(recurse = True)):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            out.append(tensor)
    return out


def _module_pin_mib(module: Any) -> int:
    """Pinned host MiB of an up-front pin of ``module``: slab packing when the arena is on, else per-tensor rounding."""
    from .diffusion_pinned_arena import arena_reserved_bytes, pin_arena_enabled

    if not pin_arena_enabled():
        return _module_host_mib(module)
    try:
        return -(-arena_reserved_bytes(_unique_tensors(module)) // (1024 * 1024))
    except Exception:  # noqa: BLE001 - priced as the per-tensor pin
        return _module_host_mib(module)


def _apply_group_offloading_packed(
    apply: Callable[..., Any],
    module: Any,
    kwargs: dict[str, Any],
    logger: Any = None,
) -> Any:
    """``apply(module, **kwargs)`` with an up-front pin (stream, ``low_cpu_mem_usage=False``) packed into slabs.

    diffusers pins each tensor through torch's host allocator, which rounds it up to a power of two: a 23.8 GB bf16
    Flux DiT held 37.2 GB pinned. Slabs hold the same tensors in ~weight bytes; the H2D copies read the same pinned
    pages, so overlap is unchanged. Lazy (``low_cpu_mem_usage``) and unstreamed applies run as before."""
    from .diffusion_pinned_arena import packable_bytes, pinned_arena_for_group_offload

    pack = (
        bool(kwargs.get("use_stream"))
        and not kwargs.get("low_cpu_mem_usage", False)
        and host_pinning_allowed(kwargs.get("onload_device"))
    )
    expected = None
    if pack:
        try:
            expected = packable_bytes(_unique_tensors(module))
        except Exception:  # noqa: BLE001 - whole slabs, as without a size
            expected = None
    with pinned_arena_for_group_offload(
        enabled = None if pack else False, expected_bytes = expected
    ) as arena:
        out = apply(module, **kwargs)
    if arena is not None and arena.payload_bytes:
        try:
            module._unsloth_pin_arena_bytes = (arena.payload_bytes, arena.reserved_bytes)
        except Exception:  # noqa: BLE001 - only a diagnostic
            pass
        if logger is not None:
            logger.info(
                "diffusion.memory: %s pinned in %.2f GB of slabs (%.2f GB of weights)",
                type(module).__name__,
                arena.reserved_bytes / 1e9,
                arena.payload_bytes / 1e9,
            )
    return out


def _storage_nbytes(tensor: Any, depth: int = 0) -> list[int]:
    """Bytes per allocation; torchao subclasses report their logical bf16 size, so size the packed inner tensors."""
    flatten = getattr(type(tensor), "__tensor_flatten__", None)
    if flatten is not None and depth < 4:
        try:
            names, _ctx = tensor.__tensor_flatten__()
            inner = [getattr(tensor, name, None) for name in names]
            if inner and all(t is not None for t in inner):
                return [n for t in inner for n in _storage_nbytes(t, depth + 1)]
        except Exception:  # noqa: BLE001 - fall back to the logical size, which only over-counts
            pass
    return [int(tensor.numel()) * int(tensor.element_size())]


def _pin_budget_mib() -> Optional[int]:
    """Pinnable host MiB leaving ``max(4 GiB, 15%)`` of the cgroup-capped total free, or None if unreadable."""
    total = _host_ram_capacity_mib()
    available = _available_system_memory_mib()
    if total is None or available is None:
        return None
    reserve = max(_PIN_RESERVE_MIN_MIB, int(int(total) * _PIN_RESERVE_FRACTION))
    return max(0, int(available) - reserve)


def _remove_group_offload_hooks(module: Any) -> None:
    """Undo a group offloading apply that raised part way (best effort)."""
    try:
        from diffusers.hooks import group_offloading as go
        from diffusers.hooks.hooks import HookRegistry

        registry = HookRegistry.check_if_exists_or_initialize(module)
        for name in (
            "_GROUP_OFFLOADING",
            "_LAZY_PREFETCH_GROUP_OFFLOADING",
            "_LAYER_EXECUTION_TRACKER",
        ):
            hook = getattr(go, name, None)
            if isinstance(hook, str):
                registry.remove_hook(hook, recurse = True)
    except Exception:  # noqa: BLE001 - the fallback reports its own failure
        pass


def _pinned_memory_capped() -> bool:
    """Windows (WDDM) and WSL2 cap pinned host memory near 1 GiB (NVIDIA CUDA on WSL known limitation)."""
    if sys.platform == "win32":
        return True
    try:
        with open("/proc/version", "r", encoding = "utf-8") as f:
            return "microsoft" in f.read().lower()
    except Exception:  # noqa: BLE001 - not Linux, or unreadable: not WSL
        return False


def _streamed_pin_plan(
    transformer_mib: int,
    encoder_mib: int,
    logger: Any = None,
    device: Any = None,
) -> tuple[bool, bool]:
    """(pin transformer, pin encoders), transformer first. ``device``: the onload target the pins serve."""
    forced = str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower()
    if forced in ("0", "off", "false", "no") or not host_pinning_allowed(device):
        return False, False
    if forced in ("1", "on", "true", "yes"):
        return True, True
    budget = None if _pinned_memory_capped() else _pin_budget_mib()
    if budget is None or budget <= 0:
        pin_transformer = pin_encoders = False
    else:
        pin_transformer = transformer_mib <= budget
        pinned = transformer_mib if pin_transformer else 0
        pin_encoders = pinned + encoder_mib <= budget
    if logger is not None:
        try:
            logger.info(
                "diffusion.memory: streamed-encoder tier pins transformer=%s (%d MiB) encoders=%s (%d MiB) "
                "against %s MiB of pinnable host RAM",
                pin_transformer,
                transformer_mib,
                pin_encoders,
                encoder_mib,
                budget,
            )
        except Exception:  # noqa: BLE001
            pass
    return pin_transformer, pin_encoders


def install_group_offload_torchao_swap_retry() -> bool:
    """``swap_tensors`` refuses a weakref'd tensor; retry once after gc so uncollected compile garbage does not fail."""
    try:
        from diffusers.hooks import group_offloading as go
    except Exception:  # noqa: BLE001 - no group offload in this diffusers
        return False
    original = getattr(go, "_swap_torchao_tensor", None)
    if original is None or getattr(original, "_unsloth_swap_retry", False):
        return False

    @functools.wraps(original)
    def _swap_torchao_tensor(param: Any, source: Any) -> None:
        try:
            original(param, source)
        except RuntimeError as exc:
            if "weakref" not in str(exc):
                raise
            import gc

            gc.collect()
            original(param, source)

    _swap_torchao_tensor._unsloth_swap_retry = True
    go._swap_torchao_tensor = _swap_torchao_tensor
    return True


BACKGROUND_PIN_ENV = "UNSLOTH_DIFFUSION_BACKGROUND_PIN"
_BG_PIN_ATTR = "_unsloth_background_pin"
_PENDING_PINS_ATTR = "_unsloth_background_pins"
_BACKGROUND_PIN_REQUEST_ATTR = "_unsloth_background_pin_requested"


def request_background_pins(pipe: Any) -> None:
    """Ask the next group offload apply on ``pipe`` to pin off the load path (see _GroupPinner)."""
    try:
        setattr(pipe, _BACKGROUND_PIN_REQUEST_ATTR, True)
    except Exception:  # noqa: BLE001 - a pipe refusing attributes pins eagerly, as before
        pass


def _background_pin_enabled(device: Any = None) -> bool:
    return (os.environ.get(BACKGROUND_PIN_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    ) and host_pinning_allowed(device)


# "0": image-load streamed encoders pin into slabs up front instead of on a background worker.
ENCODER_BACKGROUND_PIN_ENV = "UNSLOTH_DIFFUSION_ENCODER_BACKGROUND_PIN"
# "0": an encoder that does not fit the pin budget whole pins nothing instead of a fitting prefix.
PARTIAL_ENCODER_PIN_ENV = "UNSLOTH_DIFFUSION_PARTIAL_ENCODER_PIN"
_ENCODER_PIN_REQUEST_ATTR = "_unsloth_background_encoder_pin_requested"
_PARTIAL_PIN_MIN_MIB = 256
# "0": old behaviour (power-of-two pin price, no partial encoder prefix).
VIDEO_EXACT_PIN_PLAN_ENV = "UNSLOTH_VIDEO_EXACT_PIN_PLAN"
_CAST_WRITES_PENDING_ATTR = "_unsloth_cast_writes_pending"


def request_background_encoder_pins(pipe: Any) -> None:
    """Mark ``pipe`` so the next group offload defers its streamed encoder pins; denoisers still pin up front."""
    try:
        setattr(pipe, _ENCODER_PIN_REQUEST_ATTR, True)
    except Exception:  # noqa: BLE001 - a pipe refusing attributes pins eagerly, as before
        pass


def finish_encoder_pins_at_load(pipe: Any, logger: Any = None) -> float:
    """Wait for the encoder pins; left running into the first prompt they cost what they saved (FLUX.1 B200)."""
    if pipe is None:
        return 0.0
    start_background_pins(pipe)
    waited = finish_background_pins(pipe)
    if waited >= 0.5 and logger is not None:
        logger.info("diffusion.memory: the load waited %.1f s for the encoder pins", waited)
    return waited


def _encoder_background_pin_wanted(pipe: Any, device: Any = None) -> bool:
    # Needs the registered pin: per-tensor pin_memory() rounds to a power of two, past the weight-byte budget.
    return (
        bool(getattr(pipe, _ENCODER_PIN_REQUEST_ATTR, False))
        and not _env_off(ENCODER_BACKGROUND_PIN_ENV)
        and _background_pin_enabled(device)
        and _fast_pin_supported(device)
    )


def note_cast_cache_writes(pipe: Any, writes: Any) -> None:
    """Record the cast-cache writes a first load in a Studio home deferred; the pin plan reads them (``done()``)."""
    try:
        setattr(pipe, _CAST_WRITES_PENDING_ATTR, writes)
    except Exception:  # noqa: BLE001 - a pipe refusing attributes keeps the exact plan
        pass


def _cast_cache_write_pending(pipe: Any) -> bool:
    writes = getattr(pipe, _CAST_WRITES_PENDING_ATTR, None)
    if writes is None:
        return False
    try:
        return not writes.done()
    except Exception:  # noqa: BLE001 - unreadable: treat as still writing
        return True


def _video_exact_pins(
    pipe: Any,
    device: Any = None,
    logger: Any = None,
) -> bool:
    """Video load priced at page-rounded bytes; a first load still writing its cast cache keeps the default plan (extra pins cost it 3-11 s)."""
    if not (
        bool(getattr(pipe, _FAST_PIN_REQUEST_ATTR, False))
        and not _env_off(VIDEO_EXACT_PIN_PLAN_ENV)
        and _fast_pin_supported(device)
    ):
        return False
    if _cast_cache_write_pending(pipe):
        if logger is not None:
            logger.info(
                "diffusion.memory: this load writes its cast cache (first load in this home): its pins keep the "
                "default plan, no encoder prefix"
            )
        return False
    return True


def _late_encoder_pin_cap(
    pipe: Any,
    module: Any,
    device: Any = None,
) -> Callable[[], int]:
    """Encoder pin cap read when the pinner starts, not at plan time (the load still holds source shards then), minus earlier pinners' unplaced bytes."""

    def cap() -> int:
        budget = _streamed_pin_leftover_mib(0, False, device)
        owed = 0
        for pinner in list(getattr(pipe, _PENDING_PINS_ATTR, None) or ()):
            if pinner.module is module:
                break
            owed += pinner.outstanding_bytes()
        return max(0, (int(budget) << 20) - owed)

    return cap


def _streamed_pin_leftover_mib(
    transformer_mib: int,
    pin_transformer: bool,
    device: Any = None,
) -> int:
    """Pin budget MiB left after the transformer; 0 wherever ``_streamed_pin_plan`` would not pin by budget."""
    if _env_off(PARTIAL_ENCODER_PIN_ENV) or not host_pinning_allowed(device):
        return 0
    forced = str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower()
    if forced in ("0", "off", "false", "no", "1", "on", "true", "yes") or _pinned_memory_capped():
        return 0
    budget = _pin_budget_mib()
    if budget is None:
        return 0
    left = int(budget) - (int(transformer_mib) if pin_transformer else 0)
    return left if left >= _PARTIAL_PIN_MIN_MIB else 0


_DENOISER_NAMES = ("transformer", "transformer_2", "unconditional_transformer", "unet")


def denoisers_pinned_resident(pipe: Any) -> bool:
    """Every denoiser unhooked or with all its offload groups pinned, at least one pinned. The hooks stay so an
    oversized request can stream it again (``release_resident_groups``)."""
    pinned = False
    for name in _DENOISER_NAMES:
        module = getattr(pipe, name, None)
        if module is None:
            continue
        try:
            if any(getattr(sub, "_hf_hook", None) is not None for sub in module.modules()):
                return False  # accelerate (model / sequential) offload moves it
        except Exception:  # noqa: BLE001 - unreadable: assume it moves
            return False
        groups = _offload_groups(module)
        if not groups:
            registry = getattr(module, "_diffusers_hook", None)
            if any("offload" in str(key) for key in (getattr(registry, "hooks", None) or {})):
                return False
            continue
        if not all(getattr(g, "_unsloth_resident", False) for g in groups):
            return False
        pinned = True
    return pinned


def _has_offload_groups(module: Any) -> bool:
    try:
        return bool(_offload_groups(module))
    except Exception:  # noqa: BLE001 - an unreadable module counts as unhooked
        return False


def _offload_groups(module: Any) -> list:
    """The diffusers offload groups hooked under ``module``, in registration (block) order."""
    try:
        from diffusers.hooks import group_offloading as go
        name = getattr(go, "_GROUP_OFFLOADING", "group_offloading")
    except Exception:  # noqa: BLE001
        return []
    groups: list = []
    seen: set = set()
    for sub in module.modules():
        registry = getattr(sub, "_diffusers_hook", None)
        get_hook = getattr(registry, "get_hook", None)
        group = getattr(get_hook(name), "group", None) if callable(get_hook) else None
        if group is not None and id(group) not in seen:
            seen.add(id(group))
            groups.append(group)
    return groups


# Video loads only. "0" restores the per-tensor ``pin_memory()``.
FAST_PIN_ENV = "UNSLOTH_VIDEO_FAST_PIN"
_FAST_PIN_REQUEST_ATTR = "_unsloth_fast_pin_requested"
_FAST_PIN_THREADS = 8


def request_fast_pins(pipe: Any) -> None:
    """Ask the background pinners of ``pipe`` to use registered host memory."""
    try:
        setattr(pipe, _FAST_PIN_REQUEST_ATTR, True)
    except Exception:  # noqa: BLE001 - a pipe refusing attributes pins the default way
        pass


def _fast_pin_supported(device: Any = None) -> bool:
    if (os.environ.get(FAST_PIN_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    # WDDM / WSL2 cap page-locked memory; ROCm's register path is unmeasured.
    if not sys.platform.startswith("linux") or _pinned_memory_capped():
        return False
    try:
        import torch

        if (
            is_rocm_torch(torch)
            or not torch.cuda.is_available()
            or not host_pinning_allowed(device)
        ):
            return False
        cudart = torch.cuda.cudart()
        return hasattr(cudart, "cudaHostRegister") and hasattr(cudart, "cudaHostUnregister")
    except Exception:  # noqa: BLE001
        return False


class _RegisteredHostBuffer:
    """Anonymous mapping page-locked with ``cudaHostRegister``; tensor views keep it alive, the last one unregisters it.

    Torch's pinned allocator pins on one thread (10.9-13.1 s for LTX-2.3's 24.6 GiB Gemma3); 8 threads + register: 2.3 s."""

    def __init__(self, nbytes: int):
        import ctypes
        import mmap

        import torch

        self._cudart = torch.cuda.cudart()
        self._size = max(int(nbytes), 1)
        self._map = mmap.mmap(-1, self._size)
        self._anchor = ctypes.c_char.from_buffer(self._map)
        self.ptr = ctypes.addressof(self._anchor)
        self._registered = False
        self._device: Optional[int] = None
        self.__array_interface__ = {
            "shape": (int(nbytes),),
            "typestr": "|u1",
            "data": (self.ptr, False),
            "version": 3,
        }

    def register(self) -> None:
        err = self._cudart.cudaHostRegister(self.ptr, self._size, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister failed ({err})")
        import torch

        # A bare synchronize() on another thread would wait on (and open a context on) device 0.
        self._device = torch.cuda.current_device()
        self._registered = True

    def __del__(self) -> None:
        try:
            if self._registered:
                import torch

                # An async upload may still read these pages; never unmap under it.
                torch.cuda.synchronize(self._device)
                self._cudart.cudaHostUnregister(self.ptr)
                self._registered = False
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass
        try:
            del self._anchor
            self._map.close()
        except Exception:  # noqa: BLE001
            pass


# "0" restores torch's pinned allocator for up-front pinned slabs.
REGISTERED_SLAB_ENV = "UNSLOTH_DIFFUSION_REGISTERED_SLABS"


def registered_pinned_empty(nbytes: int) -> Optional[Any]:
    """Page-locked uint8 host tensor in registered memory, or None (use torch's pinned allocator).
    Torch's allocator faults pages on one thread (~18 s for MiniMax-H3); ``zero_`` here faults them in parallel."""
    if (os.environ.get(REGISTERED_SLAB_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return None
    if int(nbytes) <= 0 or not _fast_pin_supported():
        return None
    import numpy as np
    import torch

    try:
        holder = _RegisteredHostBuffer(int(nbytes))
        buf = torch.from_numpy(np.asarray(holder))
        buf.zero_()
        holder.register()
    except Exception:  # noqa: BLE001
        return None
    return buf


def registered_host_copy(src: Any) -> Any:
    """A contiguous copy of host tensor ``src`` in CUDA-registered (pinned) memory."""
    import numpy as np
    import torch

    holder = _RegisteredHostBuffer(src.numel() * src.element_size())
    out = torch.from_numpy(np.asarray(holder)).view(src.dtype).view(src.shape)
    out.copy_(src)
    holder.register()
    return out


def _registered_pin(src: Any, device: Any = None) -> Any:
    """``src.pin_memory()`` in registered host memory where the type allows (GGUF re-wrapped, avoiding pow2 rounding)."""
    import torch

    if not host_pinning_allowed(device):
        return src  # integrated GPU: a pinned copy only takes RAM from the card
    if type(src) is torch.Tensor:
        return registered_host_copy(src)
    gguf = _gguf_parameter_class()
    if gguf is not None and type(src) is gguf and _exact_pin_supported(device):
        pinned = registered_host_copy(src.as_tensor())
        return gguf(pinned, requires_grad = False, quant_type = src.quant_type)
    return src.pin_memory()


class _GroupPinner:
    """Pins a streamed module's offload groups on a worker thread, first block first.

    Pinning at load reads every streamed byte from disk before the load returns: 125-139 s of LTX-2.3's 37 GB DiT,
    20-44 s of Wan2.2-5B's text encoder on Colab. Off the load path the read overlaps the prompt encode and the
    first compile. A group's ``onload_`` waits for that group only, and the swap happens before the wait releases,
    so no group is ever onloaded while its host copy is being replaced."""

    def __init__(
        self,
        module: Any,
        groups: list,
        device: Any,
        logger: Any = None,
        fast: bool = False,
        byte_cap: Optional[int] = None,
    ):
        import threading

        self.fast = bool(fast)
        # Pin whole groups in order while they fit; the rest stay pageable.
        if byte_cap is None or callable(byte_cap):
            self.byte_cap = byte_cap
        else:
            self.byte_cap = max(0, int(byte_cap))
        self.module = module
        self.label = type(module).__name__
        self.groups = groups
        self.device = device
        self.logger = logger
        self.pinned = 0
        self.waited_s = 0.0
        self._done = {id(g): threading.Event() for g in groups}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target = self._run, name = "unsloth-offload-pin", daemon = True)
        self._started = False
        for group in groups:
            setattr(group, _BG_PIN_ATTR, self)

    def start(self) -> None:
        with self._lock:
            if not self._started:
                self._started = True
                self._thread.start()

    def wait(self, group: Any) -> None:
        import threading

        event = self._done.get(id(group))
        if event is None or threading.current_thread() is self._thread:
            return
        self.start()
        if not event.is_set():
            import time

            began = time.perf_counter()
            event.wait()
            self.waited_s += time.perf_counter() - began

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait for the worker; True once it has exited (or when called from the worker itself)."""
        import threading

        if threading.current_thread() is self._thread:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout)

    def _unpinned(self, group: Any) -> list:
        import torch

        # GGUF host copies are GGUFParameter (packed bytes + quant type): pin_memory keeps the subclass, which the
        # per-onload fallback already relies on. Leaving them unpinned kept every CUDA graph off the GGUF stream.
        gguf = _gguf_parameter_class()
        return [
            (tensor, src)
            for tensor, src in list(group.cpu_param_dict.items())
            if (type(src) is torch.Tensor or (gguf is not None and type(src) is gguf))
            and src.device.type == "cpu"
            and src.numel() > 0
            and not src.is_pinned()
        ]

    @staticmethod
    def _fast_result(future: Any, src: Any) -> Any:
        try:
            return future.result()
        except Exception:  # noqa: BLE001 - registration refused: the allocator's pin for this tensor
            return src.pin_memory()

    def outstanding_bytes(self) -> int:
        """Bytes this pinner may still place: its unpinned groups (placed ones already left that sum), within what
        its byte cap has left."""
        try:
            need = sum(
                _page_bytes(src.nbytes)
                for group in self.groups
                for _t, src in self._unpinned(group)
            )
        except Exception:  # noqa: BLE001 - unreadable: owes nothing
            return 0
        cap = self.byte_cap
        if callable(cap) or cap is None:
            return need
        return min(need, max(0, int(cap) - int(self.pinned)))

    def _capped_groups(self) -> list:
        if self.byte_cap is None:
            return list(self.groups)
        left = self.byte_cap
        out: list = []
        for group in self.groups:
            need = sum(_page_bytes(src.nbytes) for _t, src in self._unpinned(group))
            if need > left:
                break
            left -= need
            out.append(group)
        return out

    def _run(self) -> None:
        import time

        import torch

        start = time.perf_counter()
        failed = None
        pool = None
        try:
            if callable(self.byte_cap):
                self.byte_cap = max(0, int(self.byte_cap()))
                if self.logger is not None:
                    self.logger.info(
                        "diffusion.memory: %s pins up to %.1f GiB of its offload groups (cap read at pin start)",
                        self.label,
                        self.byte_cap / 2**30,
                    )
            if self.byte_cap is not None:
                kept = self._capped_groups()
                kept_ids = {id(g) for g in kept}
                # A group past the cap never pins here: its onload must not wait for the prefix.
                for group in self.groups:
                    if id(group) not in kept_ids:
                        self._done[id(group)].set()
                self.groups = kept
            if (
                getattr(self.device, "type", None) == "cuda"
                and getattr(self.device, "index", None) is not None
            ):
                torch.cuda.set_device(self.device)
            copies: dict = {}
            if self.fast:
                from concurrent.futures import ThreadPoolExecutor

                # Queued in group order, so the first group is ready first.
                pool = ThreadPoolExecutor(
                    max_workers = _FAST_PIN_THREADS,
                    thread_name_prefix = "unsloth-fast-pin",
                    initializer = torch.cuda.set_device,
                    initargs = (torch.cuda.current_device(),),
                )
                for group in self.groups:
                    copies[id(group)] = [
                        (
                            tensor,
                            src,
                            pool.submit(_registered_pin, src, self.device),
                        )
                        for tensor, src in self._unpinned(group)
                    ]
            for group in self.groups:
                if self._stop.is_set():
                    break
                if self.fast:
                    placed = [
                        (tensor, src, self._fast_result(future, src))
                        for tensor, src, future in copies.pop(id(group))
                    ]
                else:
                    # One pin per tensor, as the eager apply does; chunk views are slower to stream on LTX-2.3.
                    placed = [
                        (tensor, src, src.pin_memory()) for tensor, src in self._unpinned(group)
                    ]
                cpu = group.cpu_param_dict
                for tensor, src, pinned in placed:
                    if cpu.get(tensor) is src:
                        cpu[tensor] = pinned
                        if tensor.device.type == "cpu" and tensor.data_ptr() == src.data_ptr():
                            tensor.data = pinned
                        self.pinned += src.nbytes
                # Done before the epoch: a capture check that sees the new epoch must also see this group ready.
                self._done[id(group)].set()
                _bump_placement_epoch()
        except Exception as exc:  # noqa: BLE001 - diffusers pins what is left on each onload
            failed = exc
        finally:
            if pool is not None:
                pool.shutdown(wait = True, cancel_futures = True)
            for event in self._done.values():
                event.set()
            if failed is not None:
                _bump_placement_epoch()
        if self.logger is not None:
            try:
                if failed is not None:
                    self.logger.warning(
                        "diffusion.memory: background pinning of %s stopped (%s); the rest pins on each onload",
                        self.label,
                        failed,
                    )
                self.logger.info(
                    "diffusion.memory: pinned %.1f GiB of %s host weights in the background in %.1f s "
                    "(onloads waited %.1f s%s)",
                    self.pinned / 2**30,
                    self.label,
                    time.perf_counter() - start,
                    self.waited_s,
                    ", registered" if self.fast else "",
                )
            except Exception:  # noqa: BLE001
                pass


def install_group_pin_wait() -> bool:
    """Make a diffusers offload group wait for its background pin before it onloads."""
    try:
        import torch
        from diffusers.hooks import group_offloading as go
    except Exception:  # noqa: BLE001
        return False
    group_cls = getattr(go, "ModuleGroup", None)
    original = getattr(group_cls, "onload_", None)
    if original is None or getattr(original, "_unsloth_pin_wait", False):
        return False

    @functools.wraps(original)
    def onload_(self, *args: Any, **kwargs: Any) -> Any:
        pinner = getattr(self, _BG_PIN_ATTR, None)
        if pinner is not None:
            pinner.wait(self)
        return original(self, *args, **kwargs)

    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    wrapped = disable(onload_) if callable(disable) else onload_
    wrapped._unsloth_pin_wait = True
    group_cls.onload_ = wrapped
    return True


def start_background_pins(pipe: Any) -> int:
    """Start the pinners the load deferred; returns how many."""
    pinners = list(getattr(pipe, _PENDING_PINS_ATTR, None) or ())
    for pinner in pinners:
        pinner.start()
    return len(pinners)


def finish_background_pins(pipe: Any, cancel: Any = None) -> float:
    """Blocks until deferred pinners finish, since a render overlapping one runs measurably slower."""
    import time

    start = time.perf_counter()
    for pinner in list(getattr(pipe, _PENDING_PINS_ATTR, None) or ()):
        pinner.start()
        # Polled so a cancelled request leaves now; the pin itself keeps running for the next render.
        while not pinner.join(0.25):
            if cancel is not None and cancel.is_set():
                return time.perf_counter() - start
    return time.perf_counter() - start


def stop_background_pins(pipe: Any, timeout: Optional[float] = 30.0) -> None:
    for pinner in list(getattr(pipe, _PENDING_PINS_ATTR, None) or ()):
        try:
            pinner.stop(timeout)
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass


def _drop_deferred_pinning(pipe: Any, module: Any) -> None:
    pending = getattr(pipe, _PENDING_PINS_ATTR, None)
    if pending:
        pending[:] = [pinner for pinner in pending if pinner.module is not module]


def _defer_pinning(
    pipe: Any,
    module: Any,
    device: Any,
    logger: Any,
    *,
    fast: Optional[bool] = None,
    byte_cap: Optional[int] = None,
) -> bool:
    groups = _offload_groups(module)
    if not groups:
        return False
    if fast is None:
        fast = bool(getattr(pipe, _FAST_PIN_REQUEST_ATTR, False))
    fast = bool(fast) and _fast_pin_supported(device)
    pinner = _GroupPinner(module, groups, device, logger, fast = fast, byte_cap = byte_cap)
    pending = getattr(pipe, _PENDING_PINS_ATTR, None)
    if pending is None:
        pending = []
        try:
            setattr(pipe, _PENDING_PINS_ATTR, pending)
        except Exception:  # noqa: BLE001 - nowhere to park it: pin now, like the eager path
            pinner.start()
            return True
    pending.append(pinner)
    return True


EAGER_OFFLOAD_HOOKS_ENV = "UNSLOTH_DIFFUSION_EAGER_OFFLOAD_HOOKS"


def install_group_offload_hooks_eager() -> bool:
    """Keep diffusers' group-offload hooks out of compiled blocks: traced, the first-forward layer tracker guards on the
    block name and recompiles per block (44 recompiles on Qwen-Image-2.1 at 16 GB, 5 eager). Must run before the hooks register."""
    if _env_off(EAGER_OFFLOAD_HOOKS_ENV):
        return False
    try:
        import torch
        from diffusers.hooks import group_offloading as go
    except Exception:  # noqa: BLE001 - no group offload in this diffusers
        return False
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if not callable(disable):
        return False
    patched = False
    for cls_name, methods in (
        ("GroupOffloadingHook", ("pre_forward", "post_forward")),
        ("LayerExecutionTrackerHook", ("pre_forward",)),
        ("LazyPrefetchGroupOffloadingHook", ("post_forward",)),
    ):
        cls = getattr(go, cls_name, None)
        for name in methods:
            fn = getattr(cls, "__dict__", {}).get(name) if cls is not None else None
            if fn is None or getattr(fn, "_unsloth_eager", False):
                continue
            eager = disable(fn)
            eager._unsloth_eager = True
            eager._unsloth_orig = fn
            setattr(cls, name, eager)
            patched = True
    return patched


def _install_group_offload_torchao_host_copy(go: Any) -> bool:
    """Give aliased torchao host copies a separate wrapper so ``swap_tensors`` cannot move them to CUDA."""
    group_cls = getattr(go, "ModuleGroup", None)
    original = getattr(group_cls, "_to_cpu", None)
    is_torchao = getattr(go, "_is_torchao_tensor", None)
    if (
        original is None
        or not callable(is_torchao)
        or getattr(original, "_unsloth_host_copy", False)
    ):
        return False

    def _to_cpu(tensor: Any, low_cpu_mem_usage: bool) -> Any:
        copy = original(tensor, low_cpu_mem_usage)
        if copy is not tensor or not is_torchao(tensor):
            return copy
        names, ctx = tensor.__tensor_flatten__()
        return type(tensor).__tensor_unflatten__(
            {name: getattr(tensor, name) for name in names}, ctx, tensor.size(), tensor.stride()
        )

    _to_cpu._unsloth_host_copy = True
    group_cls._to_cpu = staticmethod(_to_cpu)
    return True


def install_group_offload_buffer_restore() -> bool:
    """Fix buffer restoration and torchao host-copy aliasing in diffusers stream group offload."""
    try:
        from diffusers.hooks import group_offloading as go
    except Exception:  # noqa: BLE001 - no group offload in this diffusers
        return False
    _install_group_offload_torchao_host_copy(go)
    group_cls = getattr(go, "ModuleGroup", None)
    original = getattr(group_cls, "_offload_to_memory", None)
    if original is None or getattr(original, "_unsloth_buffer_restore", False):
        return False

    @functools.wraps(original)
    def _offload_to_memory(self, *args: Any, **kwargs: Any) -> Any:
        out = original(self, *args, **kwargs)
        cpu_copies = getattr(self, "cpu_param_dict", None)
        if getattr(self, "stream", None) is None or not cpu_copies:
            return out
        restore_torchao = getattr(go, "_restore_torchao_tensor", None)
        is_torchao = getattr(go, "_is_torchao_tensor", None)
        for group_module in getattr(self, "modules", None) or ():
            for buffer in group_module.buffers():
                cpu = cpu_copies.get(buffer)
                if cpu is None:
                    continue
                if callable(is_torchao) and callable(restore_torchao) and is_torchao(buffer):
                    restore_torchao(buffer, cpu)
                else:
                    buffer.data = cpu
        return out

    _offload_to_memory._unsloth_buffer_restore = True
    group_cls._offload_to_memory = _offload_to_memory
    return True


PIN_TOP_GROUP_ENV = "UNSLOTH_DIFFUSION_PIN_TOP_GROUP"


def _pin_top_level_group(
    module: Any,
    logger: Any = None,
    pinned_mib: Optional[list] = None,
    device: Any = None,
) -> bool:
    """Onloads from one pinned host copy, async; skipped for torchao weights (not re-pointable) and when the copy does not fit pinnable host RAM.
    ``device``: the group's onload target (None: the current device)."""
    if (os.environ.get(PIN_TOP_GROUP_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    if not host_pinning_allowed(device):
        return False
    try:
        import torch
        from diffusers.hooks import group_offloading as go

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
            or getattr(group, "_unsloth_pinned_top", False)
            or getattr(group, "_unsloth_top_no_copy_back", False)
        ):
            return False
        tensors: list = []
        seen: set = set()
        for tensor in (
            [p for m in group.modules for p in m.parameters()]
            + [b for m in group.modules for b in m.buffers()]
            + list(group.parameters or [])
            + list(group.buffers or [])
        ):
            if id(tensor) not in seen:
                seen.add(id(tensor))
                tensors.append(tensor)
        is_torchao = getattr(go, "_is_torchao_tensor", None)
        if not tensors or (callable(is_torchao) and any(is_torchao(t) for t in tensors)):
            return False
        if any(type(t) not in (torch.Tensor, torch.nn.Parameter) for t in tensors):
            return False
        # the user's "pin nothing" override wins, as on every other streamed path
        if str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower() in (
            "0",
            "off",
            "false",
            "no",
        ):
            return False
        # per tensor rounded to a power of two, like torch's pinned allocator (and _module_host_mib)
        need_mib = sum(
            1 << (int(t.numel()) * int(t.element_size()) - 1).bit_length()
            for t in tensors
            if int(t.numel()) * int(t.element_size()) > 0
        ) // (1024 * 1024)
        budget = None if _pinned_memory_capped() else _pin_budget_mib()
        # on the running total the encoders and torchao denoisers already pinned (or will, deferred) count against
        already = pinned_mib[0] if pinned_mib else 0
        if budget is None or already + need_mib > budget:
            return False
        host = {
            t: (t.data if t.data.device.type == "cpu" else t.data.cpu()).pin_memory()
            for t in tensors
        }
        device = group.onload_device

        def onload_() -> None:
            for tensor, pinned in list(host.items()):
                current = tensor.data
                if current.device.type == "cpu" and current.data_ptr() != pinned.data_ptr():
                    # replaced while offloaded (a .to() conversion, an adapter fused on the host): re-pin what is there now
                    pinned = current if current.is_pinned() else current.pin_memory()
                    host[tensor] = pinned
                tensor.data = pinned.to(device, non_blocking = True)

        def offload_() -> None:
            for tensor, pinned in host.items():
                tensor.data = pinned

        disable = getattr(getattr(torch, "compiler", None), "disable", None)
        if callable(disable):
            onload_, offload_ = disable(onload_), disable(offload_)
        offload_()
        group.onload_ = onload_
        group.offload_ = offload_
        group._unsloth_pinned_top = True
        if pinned_mib is not None:
            pinned_mib[0] = already + need_mib
        if logger is not None:
            logger.info(
                "diffusion.memory: %s top-level weights (%d MiB) onload from a pinned copy, no copy back",
                type(module).__name__,
                need_mib,
            )
        return True
    except Exception as exc:  # noqa: BLE001 - diffusers keeps its own (slower) path
        if logger is not None:
            logger.debug("diffusion.memory: top-level group left as diffusers built it (%s)", exc)
        return False


def _skip_top_level_copy_back(module: Any, logger: Any = None) -> bool:
    """Fallback when ``_pin_top_level_group`` cannot pin: offload re-points to the group's existing host tensors
    instead of copying the (inference-constant) weights back on the compute stream. Same kill switch as the pinned path."""
    if (os.environ.get(PIN_TOP_GROUP_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    try:
        import torch
        from diffusers.hooks import group_offloading as go

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
            or getattr(group, "_unsloth_pinned_top", False)
            or getattr(group, "_unsloth_top_no_copy_back", False)
        ):
            return False
        tensors: list = []
        seen: set = set()
        for tensor in (
            [p for m in group.modules for p in m.parameters()]
            + [b for m in group.modules for b in m.buffers()]
            + list(group.parameters or [])
            + list(group.buffers or [])
        ):
            if id(tensor) not in seen:
                seen.add(id(tensor))
                tensors.append(tensor)
        is_torchao = getattr(go, "_is_torchao_tensor", None)
        if not tensors or (callable(is_torchao) and any(is_torchao(t) for t in tensors)):
            return False
        if any(type(t) not in (torch.Tensor, torch.nn.Parameter) for t in tensors):
            return False
        host = {t: (t.data if t.data.device.type == "cpu" else t.data.cpu()) for t in tensors}
        device = group.onload_device

        def onload_() -> None:
            for tensor, cpu in list(host.items()):
                current = tensor.data
                if current.device.type == "cpu" and current.data_ptr() != cpu.data_ptr():
                    # replaced while offloaded (a .to() conversion, an adapter fused on the host): upload that instead
                    cpu = current
                    host[tensor] = cpu
                tensor.data = cpu.to(device)

        def offload_() -> None:
            for tensor, cpu in host.items():
                tensor.data = cpu

        disable = getattr(getattr(torch, "compiler", None), "disable", None)
        if callable(disable):
            onload_, offload_ = disable(onload_), disable(offload_)
        offload_()
        group.onload_ = onload_
        group.offload_ = offload_
        group._unsloth_top_no_copy_back = True
        if logger is not None:
            logger.info(
                "diffusion.memory: %s top-level weights (%d MiB) not pinnable here; offload re-points to the host "
                "copy instead of copying them back",
                type(module).__name__,
                sum(int(t.numel()) * int(t.element_size()) for t in tensors) >> 20,
            )
        return True
    except Exception as exc:  # noqa: BLE001 - diffusers keeps its own path
        if logger is not None:
            logger.debug(
                "diffusion.memory: top-level group copy-back left as diffusers built it (%s)", exc
            )
        return False


def _apply_group_offload(
    pipe: Any,
    device: str,
    logger: Any,
    *,
    stream_text_encoders: bool = False,
    stream_transformer: bool = True,
    background_pin: Optional[bool] = None,
    resident_transformer_mib: Optional[int] = None,
    resident_text_encoder_mib: Optional[int] = None,
) -> bool:
    """Streams transformer blocks in groups; encoders stay resident unless stream_text_encoders is set; returns False on any failure, caller falls back."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return False
    if not stream_transformer and not stream_text_encoders:
        return False
    if _any_layer_split(pipe):
        from .diffusion_layer_split_plan import undo_layer_split
        undo_layer_split(pipe)
        if logger is not None:
            logger.info("diffusion.memory: removed the layer split before applying group offload")
    installed = 0
    # every component's torch forward hooks before this call: an OOM after hooks exist unwinds to this state
    hook_snapshots: list = []
    grouped_before: set = set()
    try:
        import inspect

        import torch
        from diffusers.hooks import apply_group_offloading

        install_group_offload_buffer_restore()
        install_group_offload_hooks_eager()
        hook_snapshots = [
            (comp, _module_forward_hook_ids(comp))
            for comp in (getattr(pipe, "components", {}) or {}).values()
            if isinstance(comp, torch.nn.Module)
        ]
        grouped_before = {id(comp) for comp, _ in hook_snapshots if _has_offload_groups(comp)}

        # Dual-DiT pipelines (Ideogram 4): second DiT is as large as the first, so stream all DiTs.
        streamed: dict[str, Any] = {}
        if stream_transformer:
            streamed["transformer"] = transformer
            for extra in ("transformer_2", "unconditional_transformer"):
                module = getattr(pipe, extra, None)
                if isinstance(module, torch.nn.Module):
                    streamed[extra] = module
        # Encoders stream separately from DiTs, in their own dict so the resident placement loop skips them.
        streamed_encoders: dict[str, Any] = {}
        if stream_text_encoders:
            # Encoders run once before step 0, so stream them; match components by name prefix text_encoder.
            for name, comp in getattr(pipe, "components", {}).items():
                if name.startswith("text_encoder") and isinstance(comp, torch.nn.Module):
                    streamed_encoders[name] = comp

        onload = torch.device(device)
        # overlap H2D copies with compute; a stream makes diffusers pin, so not on an integrated GPU
        use_stream = onload.type == "cuda" and host_pinning_allowed(onload)
        gkwargs: dict[str, Any] = {
            "onload_device": onload,
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": DEFAULT_GROUP_BLOCKS,
            "use_stream": use_stream,
        }
        # Gated on the signature so older diffusers still works.
        _params = inspect.signature(apply_group_offloading).parameters
        if use_stream:
            if "non_blocking" in _params:
                gkwargs["non_blocking"] = True
            if "record_stream" in _params:
                gkwargs["record_stream"] = True
        # background_pin: applied unpinned; a _GroupPinner pins it once the load commits (start_background_pins).
        if background_pin is None:
            background_pin = bool(getattr(pipe, _BACKGROUND_PIN_REQUEST_ATTR, False))
        defer = (
            background_pin
            and use_stream
            and "low_cpu_mem_usage" in _params
            and _background_pin_enabled(onload)
        )
        # an up-front pin packs into slabs (_apply_group_offloading_packed); a deferred one pins per tensor
        pin_mib = _module_host_mib if defer else _module_pin_mib
        # Power-of-two pricing read Wan's 12.5 GiB UMT5 as 17 GiB, so it never pinned on 31-51 GiB hosts.
        exact_defer = defer and _video_exact_pins(pipe, onload, logger)
        if exact_defer:
            _host_price = pin_mib

            def pin_mib(module: Any) -> int:
                if _torchao_weight_classes(module):
                    return _host_price(module)
                return _module_offload_registered_mib(module)

        # Image loads: only the streamed encoders defer (registered pins, priced at weight bytes); torchao ones cannot.
        encoders_defer = (
            not defer
            and stream_text_encoders
            and use_stream
            and "low_cpu_mem_usage" in _params
            and _encoder_background_pin_wanted(pipe, onload)
        )

        def _encoder_defers(module: Any) -> bool:
            return encoders_defer and not _torchao_weight_classes(module)

        def enc_pin_mib(module: Any) -> int:
            return _module_registered_mib(module) if _encoder_defers(module) else pin_mib(module)

        pin_streamed = (True, True)
        dit_pin_mib = 0
        if stream_text_encoders and "low_cpu_mem_usage" in _params:
            # The streamed path PINS every offloaded parameter in host RAM when a copy stream is in use (diffusers
            # group_offloading `_init_cpu_param_dict`), which is a fine trade when group offload was already the plan.
            # Pin only what host RAM covers (unbounded pinning hurt #8188); unpinned re-pins on every onload.
            dit_pin_mib = sum(pin_mib(m) for m in streamed.values())
            pin_streamed = _streamed_pin_plan(
                dit_pin_mib,
                sum(enc_pin_mib(m) for m in streamed_encoders.values()),
                logger,
                device = onload,
            )
            gkwargs["low_cpu_mem_usage"] = not pin_streamed[0]
        small_host = bool(getattr(pipe, "_unsloth_small_host", None))
        if small_host and "low_cpu_mem_usage" in _params:
            # pinning would copy the small-host route's memory-mapped bytes back into host RAM
            pin_streamed = (False, False)
            gkwargs["low_cpu_mem_usage"] = True
        encoder_cap_mib = 0
        late_prefix = (
            exact_defer
            and stream_text_encoders
            and not pin_streamed[1]
            and not small_host
            and not _env_off(PARTIAL_ENCODER_PIN_ENV)
        )
        if late_prefix and logger is not None:
            logger.info(
                "diffusion.memory: streamed encoders pin the prefix of their offload groups that fits the host RAM "
                "free when the load returns"
            )
        if encoders_defer and stream_text_encoders and not pin_streamed[1] and not small_host:
            encoder_cap_mib = _streamed_pin_leftover_mib(dit_pin_mib, pin_streamed[0], onload)
            if encoder_cap_mib and logger is not None:
                logger.info(
                    "diffusion.memory: streamed encoders pin a %d MiB prefix of their offload groups in the background",
                    encoder_cap_mib,
                )
        if defer or encoders_defer:
            install_group_pin_wait()
        # Place companions BEFORE group hooks: a companion OOM then leaves no hooks, and
        # diffusers rejects enable_model_cpu_offload once group hooks exist.
        for name, comp in getattr(pipe, "components", {}).items():
            if name in streamed or name in streamed_encoders:
                continue
            if isinstance(comp, torch.nn.Module):
                comp.to(onload)
        # Encoders the plan pins count against the same budget as any torchao denoiser pinned below.
        pinned_mib = [
            sum(enc_pin_mib(m) for m in streamed_encoders.values())
            if stream_text_encoders and pin_streamed[1]
            else encoder_cap_mib
        ]
        for module in streamed.values():
            # torchao weights need their up-front pin (lazy pinning refuses them), so they never defer.
            if (
                defer
                and not gkwargs.get("low_cpu_mem_usage", False)
                and not _torchao_weight_classes(module)
            ):
                apply_group_offloading(module, **{**gkwargs, "low_cpu_mem_usage": True})
                _defer_pinning(pipe, module, onload, logger)
            else:
                _apply_group_offloading_packed(
                    apply_group_offloading,
                    module,
                    _torchao_group_offload_kwargs(module, gkwargs, pinned_mib),
                    logger,
                )
            installed += 1
            if use_stream and not _pin_top_level_group(module, logger, pinned_mib, device = onload):
                _skip_top_level_copy_back(module, logger)
            if use_stream and gkwargs.get("record_stream"):
                install_group_prefetch(module, onload, logger)
        if resident_transformer_mib:
            room = int(resident_transformer_mib)
            for module in streamed.values():
                room -= _keep_groups_resident(module, room, onload, logger)
        # Encoders after the DiTs, one by one, failures absorbed: a refusing encoder stays resident
        # since hooks already exist. Leaf level: encoders are not uniform block stacks.
        ekwargs = {k: v for k, v in gkwargs.items() if k != "num_blocks_per_group"}
        ekwargs["offload_type"] = "leaf_level"
        if "low_cpu_mem_usage" in gkwargs:
            ekwargs["low_cpu_mem_usage"] = not pin_streamed[1]
        transformer_demoted = False
        te_room = int(resident_text_encoder_mib or 0)
        encoder_left_mib = encoder_cap_mib
        for name, module in streamed_encoders.items():
            try:
                whole = not ekwargs.get("low_cpu_mem_usage", False)
                if defer and whole:
                    apply_group_offloading(module, **{**ekwargs, "low_cpu_mem_usage": True})
                    _defer_pinning(pipe, module, onload, logger)
                elif late_prefix and use_stream and not _torchao_weight_classes(module):
                    apply_group_offloading(module, **{**ekwargs, "low_cpu_mem_usage": True})
                    _defer_pinning(
                        pipe,
                        module,
                        onload,
                        logger,
                        fast = True,
                        byte_cap = _late_encoder_pin_cap(pipe, module, onload),
                    )
                elif _encoder_defers(module) and (whole or encoder_left_mib > 0):
                    apply_group_offloading(module, **{**ekwargs, "low_cpu_mem_usage": True})
                    cap = None if whole else int(encoder_left_mib) << 20
                    _defer_pinning(pipe, module, onload, logger, fast = True, byte_cap = cap)
                    if not whole:
                        encoder_left_mib = max(0, encoder_left_mib - _module_registered_mib(module))
                else:
                    _apply_group_offloading_packed(apply_group_offloading, module, ekwargs, logger)
                installed += 1
                _pin_vision_embedding_device(module)
                if te_room > 0:
                    te_room -= _keep_groups_resident(module, te_room, onload, logger)
                if getattr(pipe, "_unsloth_small_host", None):
                    from .diffusion_small_host import install_encoder_prefetch
                    install_encoder_prefetch(module, onload, logger)
            except Exception as exc:  # noqa: BLE001 -- degrade this encoder, never fail the load
                if not stream_transformer and installed == 0:
                    # Resident encoder here would OOM; fall back to model offload, which rejects partial hooks.
                    _remove_group_offload_hooks(module)
                    raise
                if not stream_transformer and not transformer_demoted:
                    if _pipe_denoisers_hold_torchao(pipe):
                        raise
                    # Model offload is gone once hooks exist: stream the transformer before this encoder goes resident.
                    for dit_name in ("transformer", "transformer_2", "unconditional_transformer"):
                        dit = getattr(pipe, dit_name, None)
                        if isinstance(dit, torch.nn.Module):
                            dkwargs = dict(gkwargs)
                            if "low_cpu_mem_usage" in _params and use_stream:
                                dkwargs["low_cpu_mem_usage"] = True
                            apply_group_offloading(
                                dit, **_torchao_group_offload_kwargs(dit, dkwargs, pinned_mib)
                            )
                            installed += 1
                    transformer_demoted = True
                    if logger is not None:
                        logger.warning(
                            "diffusion.memory: %s refused group offload on the resident-transformer tier; "
                            "streaming the transformer instead",
                            name,
                        )
                if logger is not None:
                    logger.warning(
                        "diffusion.memory: group offload unavailable for %s (%s); "
                        "keeping it resident",
                        name,
                        exc,
                    )
                # A failed leaf apply may leave partial hooks; resident must mean no hooks.
                _remove_group_offload_hooks(module)
                _drop_deferred_pinning(pipe, module)
                module.to(onload)
        return True
    except Exception as exc:  # noqa: BLE001 - fall back to whole-module offload
        # apply_group_offloading can raise after hooking part of a module, before ``installed`` counts it
        if not installed and any(
            id(comp) not in grouped_before and _has_offload_groups(comp)
            for comp, _ in hook_snapshots
        ):
            installed = 1
        if installed and _is_oom(exc) and _unwind_group_offload(pipe, hook_snapshots, logger):
            # An OOM after some hooks exist (a later DiT, an encoder moved resident): every hook is gone again, so
            # the caller's one lower-residency retry can run instead of the load failing.
            installed = 0
        if installed:
            # Partial group-offload hooks make enable_model_cpu_offload reject the pipe; raise the real failure.
            if logger is not None:
                logger.warning(
                    "diffusion.memory: group offload failed after installing hooks on %d "
                    "module(s) (%s); cannot fall back to whole-module offload",
                    installed,
                    exc,
                )
            raise
        _set_group_offload_failure(pipe, exc)
        if logger is not None:
            logger.warning(
                "diffusion.memory: group offload failed (%s); falling back to "
                "whole-module offload",
                exc,
            )
        return False


def stream_text_encoders_onto(
    pipe: Any,
    device: Any,
    logger: Any = None,
) -> list[str]:
    """Leaf-level group offload of every ``text_encoder*`` onto ``device`` and nothing else: the denoisers and the other
    components are not touched (a layer split places them; ``_apply_group_offload`` would move every unstreamed
    component onto one card). A text encoder runs once, before step 0, so it pages in for that encode. Returns the
    streamed names; raises (with this call's hooks removed) when an encoder refuses, and the caller falls back."""
    import inspect

    import torch
    from diffusers.hooks import apply_group_offloading

    install_group_offload_buffer_restore()
    install_group_offload_hooks_eager()
    onload = torch.device(device)
    use_stream = onload.type == "cuda" and host_pinning_allowed(onload)
    kwargs: dict[str, Any] = {
        "onload_device": onload,
        "offload_device": torch.device("cpu"),
        "offload_type": "leaf_level",
        "use_stream": use_stream,
    }
    params = inspect.signature(apply_group_offloading).parameters
    if use_stream:
        if "non_blocking" in params:
            kwargs["non_blocking"] = True
        if "record_stream" in params:
            kwargs["record_stream"] = True
    encoders = {
        str(n): m
        for n, m in (getattr(pipe, "components", {}) or {}).items()
        if str(n).startswith("text_encoder") and isinstance(m, torch.nn.Module)
    }
    if "low_cpu_mem_usage" in params:
        pin = _streamed_pin_plan(
            0, sum(_module_pin_mib(m) for m in encoders.values()), logger, device = onload
        )
        kwargs["low_cpu_mem_usage"] = not pin[1]
    done: list[str] = []
    try:
        for name, module in encoders.items():
            module.to("cpu")
            done.append(name)
            _apply_group_offloading_packed(apply_group_offloading, module, kwargs, logger)
            _pin_vision_embedding_device(module)
    except Exception:
        for name in done:
            _remove_group_offload_hooks(encoders[name])
        raise
    if logger is not None and done:
        logger.info(
            "diffusion.memory: streaming %s onto %s for the encode", ", ".join(done), onload
        )
    return done


# Generate-time activation guard. The load-time plan cannot know the output resolution: a model is loaded once and
# then generates at whatever size the sliders say, so ``_plan_memory`` budgets the 1024x1024 default. That is the
# right call for PLACEMENT, but it means a request for a much larger frame is never checked against anything, so this
# re-checks per generation with the real dimensions. Opt-in escape hatch, mirroring the load-time one: the activation
# estimate is coarse, so an operator who believes it is wrong keeps a way through. Also sent per request
# (``allow_oversized``): a desktop install has no terminal.
OVERSIZED_GENERATE_ENV = "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_GENERATE"

# Must match the Images page setting label (frontend memory-refusal.ts).
OVERSIZED_GENERATE_SETTING_LABEL = "Allow oversized generations"

# Exposed through CORS in main.py: the desktop app is cross-origin (tauri://localhost).
IMAGE_REFUSAL_HEADER = "X-Unsloth-Refusal"
IMAGE_REFUSAL_MEMORY_ESTIMATE = "memory-estimate"

ACTIVATION_RUN = "run"
ACTIVATION_TILE = "tile"
ACTIVATION_REFUSE = "refuse"

# Denoiser MiB per output megapixel per image; linear only for sub-quadratic (flash / mem-efficient) attention.
DENOISE_MIB_PER_MEGAPIXEL = 1024

DEFAULT_VAE_TILE_SIDE = 1024


class ImageActivationShortfallError(ValueError):
    """This generation's activations cannot fit the free device budget.

    A ValueError subclass so ``/images/generate``'s existing mapping still turns it into a 400
    with the reason. The distinct type is what lets the OpenAI-compatible route, whose boundary
    sanitises every other exception into a bare 500, recognise the one message here that is
    written FOR the caller and hand it back instead of "Image generation failed." The Images page
    recognises it too (the route tags the 400), so it can offer a retry with the override.
    """


@dataclass(frozen = True)
class ImageActivationVerdict:
    """Generate-time guard decision (MiB figures); ACTIVATION_TILE = tile and slice the VAE for this call."""

    action: str
    message: Optional[str] = None
    needed_mib: Optional[int] = None
    tiled_needed_mib: Optional[int] = None
    budget_mib: Optional[int] = None
    overridden: bool = False


def _oversized_generate_override() -> bool:
    return os.environ.get(OVERSIZED_GENERATE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def vae_tile_side(vae: Any) -> Optional[int]:
    """The pixel side a VAE tiles at, or None when it cannot tile at all."""
    if vae is None or not callable(getattr(vae, "enable_tiling", None)):
        return None
    decode_side = getattr(vae, "_unsloth_decode_tile_side", None)
    if isinstance(decode_side, int) and not isinstance(decode_side, bool) and decode_side > 0:
        # diffusion_vae_tiling decodes in wider tiles than the attributes
        return max(64, min(4096, decode_side))
    sides: list[int] = []
    for attr in ("tile_sample_min_size", "tile_sample_min_height", "tile_sample_min_width"):
        value = getattr(vae, attr, None)
        values = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            if isinstance(item, bool):
                continue
            try:
                side = int(item)
            except (TypeError, ValueError):
                continue
            if side > 0:
                sides.append(side)
    if not sides:
        return DEFAULT_VAE_TILE_SIDE
    return max(64, min(4096, max(sides)))


def vae_can_slice(vae: Any) -> bool:
    if vae is None:
        return False
    return bool(getattr(vae, "use_slicing", False)) or callable(
        getattr(vae, "enable_slicing", None)
    )


def vae_is_sliced(vae: Any) -> bool:
    if vae is None:
        return False
    return bool(getattr(vae, "use_slicing", callable(getattr(vae, "enable_slicing", None))))


def engage_vae_tiling_for_call(pipe: Any, logger: Any = None) -> Optional[Callable[[], None]]:
    """Tile and slice ``pipe``'s VAE for one generation; returns the undo, or None if nothing changed."""
    return engage_vae_tiling(pipe, logger = logger)[0]


def engage_vae_tiling(pipe: Any, logger: Any = None) -> tuple[Optional[Callable[[], None]], bool]:
    """``engage_vae_tiling_for_call`` plus whether tiling is actually on (``enable_tiling()`` may fail)."""
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return None, False
    tiled = bool(getattr(vae, "use_tiling", False))
    undo: list[str] = []
    for flag, enable, disable in (
        ("use_tiling", "enable_tiling", "disable_tiling"),
        ("use_slicing", "enable_slicing", "disable_slicing"),
    ):
        if bool(getattr(vae, flag, False)):
            continue
        fn = getattr(vae, enable, None)
        if not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a VAE saver is an optimisation, never fatal
            if logger is not None:
                logger.warning("diffusion.memory: %s() failed: %s", enable, exc)
            continue
        undo.append(disable)
        if enable == "enable_tiling":
            tiled = bool(getattr(vae, flag, True))
    if not undo:
        return None, tiled

    def _restore() -> None:
        for method in undo:
            fn = getattr(vae, method, None)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - restoring is best-effort
                if logger is not None:
                    logger.warning("diffusion.memory: %s() failed: %s", method, exc)

    return _restore, tiled


def _family_activation_multiplier(family: Optional[str]) -> float:
    fam = (family or "").lower()
    multiplier = 1.0
    if "edit" in fam:
        multiplier *= 1.35
    if "turbo" in fam or "distilled" in fam or "schnell" in fam:
        multiplier *= 0.85
    return multiplier


def estimate_tiled_image_runtime_mib(
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    family: Optional[str] = None,
    condition_pixels: int = 0,
    tile_side: Optional[int] = None,
    vae_sliced: bool = False,
) -> int:
    """Per-call MiB with the VAE tiled: max(one-tile decode, denoiser); unsliced tiles carry the whole batch.
    Invalid under the SDPA math fallback, whose score matrix grows with the square of the tokens."""
    w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
    h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
    batch = max(1, int(batch_size or 1))
    cond = max(0, int(condition_pixels or 0))
    side = int(tile_side or DEFAULT_VAE_TILE_SIDE)
    megapixel = float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT)
    vae = estimate_image_runtime_mib(
        width = min(w, side),
        height = min(h, side),
        batch_size = 1 if vae_sliced else batch,
        family = family,
    )
    denoise = batch * (
        DENOISE_MIB_PER_MEGAPIXEL * (w * h / megapixel) * _family_activation_multiplier(family)
        + 8192 * (cond / megapixel)
    )
    return max(1024, vae, int(denoise))


def _activation_refusal_message(
    *,
    width: int,
    height: int,
    batch: int,
    total_mib: int,
    overhead_mib: int,
    budget_mib: int,
    free_mib: int,
    source_driven: bool,
    condition_pixels: int,
    tiled: bool,
    controlnet: bool = False,
    calibrated: bool = False,
) -> str:
    batch_note = f" at a batch of {batch}" if batch > 1 else ""
    cond_note = (
        " with ControlNet" if controlnet else " with its input images" if condition_pixels else ""
    )
    if source_driven:
        remedy = (
            "Upload a smaller source image (this workflow takes its output size from the image, "
            "not the Resolution setting)"
        )
    else:
        remedy = "Generate at a smaller resolution"
    # Two decimals and overhead included: refusals are decided by tens of MiB and must not contradict themselves.
    return (
        f"Generating at {width}x{height}{batch_note}{cond_note} needs about {total_mib / 1024:.2f} GB "
        f"of working memory{' even with tiled VAE decoding' if tiled else ''} (including about "
        f"{overhead_mib / 1024:.2f} GB of fixed overhead), but only about {budget_mib / 1024:.2f} GB "
        f"is usable on this device (of the {free_mib / 1024:.2f} GB currently free, after reserving "
        "room for fragmentation and other processes). Working memory holds the image being "
        "generated, so unlike model weights it cannot be moved to the CPU. "
        # Smaller-batch hint only when batch > 1: a one-image refusal cannot be fixed by asking for fewer.
        f"{remedy}"
        f"{' or a smaller batch size' if batch > 1 else ''}"
        f"{', use fewer input images or a lower reference detail' if condition_pixels else ''}"
        f"{', generate without ControlNet' if controlnet else ''}"
        f"{', reload the model with the balanced memory mode' if calibrated else ''}, "
        "or free GPU memory by closing other applications. To try anyway, turn on "
        f"'{OVERSIZED_GENERATE_SETTING_LABEL}' under Advanced on the Images page "
        "(API callers of /api/inference/images/generate can send allow_oversized; server installs "
        f"can set {OVERSIZED_GENERATE_ENV}=1)."
    )


def image_activation_verdict(
    *,
    device_memory: DeviceMemory,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    family: Optional[str] = None,
    base_overhead_mib: int = DEFAULT_BASE_OVERHEAD_MIB,
    source_driven: bool = False,
    condition_pixels: int = 0,
    vae_tile_side: Optional[int] = None,
    vae_sliced: bool = False,
    quadratic_attention: bool = False,
    allow_oversized: bool = False,
    calibrated_placement: bool = False,
    controlnet: bool = False,
) -> ImageActivationVerdict:
    """Activations cannot be offloaded, so tile the VAE decode or refuse; refuse here, not in torch, since Windows WDDM spills silently; fail open when unknown."""
    override = bool(allow_oversized) or _oversized_generate_override()
    try:
        # Unified memory skips the check: 'free' moves with the OS, and load-time refusal already owns it.
        if getattr(device_memory, "is_unified", False):
            return ImageActivationVerdict(ACTIVATION_RUN)
        # CUDA / ROCm only (ROCm reports "cuda"); the estimate was measured on discrete VRAM.
        if getattr(device_memory, "device", None) != "cuda":
            return ImageActivationVerdict(ACTIVATION_RUN)
        free = getattr(device_memory, "free_mib", None)
        if free is None:
            return ImageActivationVerdict(ACTIVATION_RUN)
        budget = _safe_device_budget_mib(device_memory)
        if budget is None:
            return ImageActivationVerdict(ACTIVATION_RUN)
        # Calibrated tiers measured only the unconditioned denoise; a ControlNet counts as one output-sized image.
        # Not on a U-Net: its ControlNet adds no tokens and stays under the measured decode peak.
        if controlnet and _resident_only_hint(family):
            controlnet = False
        input_pixels = max(0, int(condition_pixels or 0))
        conditioned = bool(calibrated_placement) and (input_pixels > 0 or bool(controlnet))
        if calibrated_placement and controlnet:
            condition_pixels = input_pixels + max(64, int(width or DEFAULT_IMAGE_WIDTH)) * max(
                64, int(height or DEFAULT_IMAGE_HEIGHT)
            )
        needed = estimate_image_runtime_mib(
            width = width,
            height = height,
            batch_size = batch_size,
            family = family,
            condition_pixels = condition_pixels,
        )
        # Load budget: same estimator and family hint as _plan_memory, at default resolution.
        planned = estimate_image_runtime_mib(
            width = None,
            height = None,
            batch_size = 1,
            family = family,
        )
        tiled = None
        if vae_tile_side is not None and not quadratic_attention:
            tiled = estimate_tiled_image_runtime_mib(
                width = width,
                height = height,
                batch_size = batch_size,
                family = family,
                condition_pixels = condition_pixels,
                tile_side = vae_tile_side,
                vae_sliced = vae_sliced,
            )
    except Exception:  # noqa: BLE001 -- a broken probe must never block a generation
        return ImageActivationVerdict(ACTIVATION_RUN)
    overhead = max(0, int(base_overhead_mib))
    # Base overhead counts here too: the load plan sums it with the activations.
    numbers = dict(needed_mib = int(needed), tiled_needed_mib = tiled, budget_mib = int(budget))
    if int(needed) + overhead <= int(budget):
        return ImageActivationVerdict(ACTIVATION_RUN, **numbers)
    if not conditioned and needed <= planned:
        if (
            tiled is not None
            and int(tiled) + overhead <= int(budget)
            and _device_user_budget_reserve_mib(
                device_memory, device_memory.total_mib or device_memory.free_mib
            )
            > _reserve_mib(
                _budget_reserve_kind(device_memory),
                device_memory.total_mib or device_memory.free_mib,
            )
        ):
            return ImageActivationVerdict(ACTIVATION_TILE, **numbers)
        return ImageActivationVerdict(ACTIVATION_RUN, **numbers)
    if tiled is not None and (
        int(tiled) + overhead <= int(budget) or (not conditioned and tiled <= planned)
    ):
        return ImageActivationVerdict(ACTIVATION_TILE, **numbers)
    if override:
        # Tile even under quadratic attention: the estimate is untrusted there, but tiling still lowers the peak.
        return ImageActivationVerdict(
            ACTIVATION_TILE if vae_tile_side is not None else ACTIVATION_RUN,
            overridden = True,
            **numbers,
        )
    w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
    h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
    message = _activation_refusal_message(
        width = w,
        height = h,
        batch = max(1, int(batch_size or 1)),
        total_mib = int(tiled if tiled is not None else needed) + overhead,
        overhead_mib = overhead,
        budget_mib = int(budget),
        free_mib = int(free),
        source_driven = source_driven,
        condition_pixels = input_pixels,
        tiled = tiled is not None,
        controlnet = bool(controlnet),
        calibrated = conditioned,
    )
    return ImageActivationVerdict(ACTIVATION_REFUSE, message = message, **numbers)


def image_activation_shortfall_message(
    *,
    device_memory: DeviceMemory,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    family: Optional[str] = None,
    base_overhead_mib: int = DEFAULT_BASE_OVERHEAD_MIB,
    source_driven: bool = False,
    condition_pixels: int = 0,
    vae_tile_side: Optional[int] = None,
    vae_sliced: bool = False,
    quadratic_attention: bool = False,
    allow_oversized: bool = False,
    calibrated_placement: bool = False,
    controlnet: bool = False,
) -> Optional[str]:
    return image_activation_verdict(
        device_memory = device_memory,
        width = width,
        height = height,
        batch_size = batch_size,
        family = family,
        base_overhead_mib = base_overhead_mib,
        source_driven = source_driven,
        condition_pixels = condition_pixels,
        vae_tile_side = vae_tile_side,
        vae_sliced = vae_sliced,
        quadratic_attention = quadratic_attention,
        allow_oversized = allow_oversized,
        calibrated_placement = calibrated_placement,
        controlnet = controlnet,
    ).message


def raise_on_image_activation_shortfall(
    *,
    device_memory: DeviceMemory,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    family: Optional[str] = None,
    base_overhead_mib: int = DEFAULT_BASE_OVERHEAD_MIB,
    source_driven: bool = False,
    condition_pixels: int = 0,
    vae_tile_side: Optional[int] = None,
    vae_sliced: bool = False,
    quadratic_attention: bool = False,
    allow_oversized: bool = False,
    calibrated_placement: bool = False,
    controlnet: bool = False,
    logger: Any = None,
) -> ImageActivationVerdict:
    """Raises ValueError so the route answers 400 with the reason; RuntimeError becomes an opaque 500."""
    verdict = image_activation_verdict(
        device_memory = device_memory,
        width = width,
        height = height,
        batch_size = batch_size,
        family = family,
        base_overhead_mib = base_overhead_mib,
        source_driven = source_driven,
        condition_pixels = condition_pixels,
        vae_tile_side = vae_tile_side,
        vae_sliced = vae_sliced,
        quadratic_attention = quadratic_attention,
        allow_oversized = allow_oversized,
        calibrated_placement = calibrated_placement,
        controlnet = controlnet,
    )
    if verdict.action == ACTIVATION_REFUSE:
        if logger is not None:
            logger.error("diffusion.memory: refusing oversized generation: %s", verdict.message)
        raise ImageActivationShortfallError(verdict.message)
    if logger is not None and (verdict.action == ACTIVATION_TILE or verdict.overridden):
        logger.warning(
            "diffusion.memory: %dx%d needs ~%s MiB untiled, ~%s MiB tiled, budget %s MiB: %s",
            int(width or DEFAULT_IMAGE_WIDTH),
            int(height or DEFAULT_IMAGE_HEIGHT),
            verdict.needed_mib,
            verdict.tiled_needed_mib,
            verdict.budget_mib,
            (
                "running anyway (allow_oversized)"
                if verdict.overridden
                else "tiling the VAE for this generation"
            ),
        )
    return verdict


STREAMING_PREFETCH_ENV = "UNSLOTH_DIFFUSION_STREAMING_PREFETCH"


def install_group_prefetch(
    module: Any,
    device: Any,
    logger: Any = None,
) -> int:
    """Event-fenced, deeper prefetch for a block-streamed module's offload groups (diffusion_offload_prefetch)."""
    from .diffusion_offload_prefetch import install_group_prefetch as _install
    return _install(module, device, logger)


def _streaming_prefetch_enabled() -> bool:
    """Kill switch for the streaming tier's overlapped onload (pinned host copies + record_stream); default on."""
    return (os.environ.get(STREAMING_PREFETCH_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _apply_streaming_offload(
    pipe: Any,
    device: str,
    logger: Any,
    *,
    resident_transformer_mib: Optional[int] = None,
) -> None:
    """Streams blocks and encoder leaves; a whole-component onload would OOM, so setup failure is fatal."""
    installed = 0
    try:
        import inspect

        import torch
        from diffusers.hooks import apply_group_offloading

        install_group_offload_buffer_restore()
        install_group_offload_hooks_eager()

        components = getattr(pipe, "components", {})
        if not isinstance(components, dict):
            raise RuntimeError("pipeline does not expose its components")

        # same selection the planner sized against, so what it promised to stream is what streams
        streamed = _streamable_components(pipe, torch)
        if "transformer" not in streamed:
            raise RuntimeError("pipeline has no transformer to stream")

        onload = torch.device(device)
        offload = torch.device("cpu")
        params = inspect.signature(apply_group_offloading).parameters
        use_stream = (
            onload.type == "cuda" and "low_cpu_mem_usage" in params and host_pinning_allowed(onload)
        )

        # Small companions (tiled VAE) stay resident; transformers and encoders stay on CPU behind hooks.
        for name, component in components.items():
            if str(name) in streamed:
                continue
            if isinstance(component, torch.nn.Module):
                component.to(onload)

        pinned_mib = [0]
        # Overlap needs a pinned host copy and record_stream (record_stream=False syncs the compute stream per group).
        prefetch = use_stream and _streaming_prefetch_enabled()
        pin_dits, pin_encoders = False, False
        defer = (
            prefetch
            and bool(getattr(pipe, _BACKGROUND_PIN_REQUEST_ATTR, False))
            and _background_pin_enabled(onload)
        )
        # an up-front pin packs into slabs (_apply_group_offloading_packed); a deferred one pins per tensor
        pin_mib = _module_host_mib if defer else _module_pin_mib
        exact_defer = defer and _video_exact_pins(pipe, onload, logger)
        if exact_defer:
            _host_price = pin_mib

            def pin_mib(module: Any) -> int:
                if _torchao_weight_classes(module):
                    return _host_price(module)
                return _module_offload_registered_mib(module)

        encoders_defer = not defer and prefetch and _encoder_background_pin_wanted(pipe, onload)

        def _encoder_defers(name: Any, module: Any, offload_type: str) -> bool:
            return (
                encoders_defer
                and offload_type != "block_level"
                and str(name).startswith("text_encoder")
                and not _torchao_weight_classes(module)
            )

        def leaf_pin_mib(name: Any, module: Any, offload_type: str) -> int:
            if _encoder_defers(name, module, offload_type):
                return _module_registered_mib(module)
            return pin_mib(module)

        dit_pin_mib = sum(pin_mib(m) for m, t in streamed.values() if t == "block_level")
        leaf_mib = 0
        if prefetch:
            leaf_mib = sum(
                leaf_pin_mib(n, m, t) for n, (m, t) in streamed.items() if t != "block_level"
            )
            pin_dits, pin_encoders = _streamed_pin_plan(
                dit_pin_mib, leaf_mib, logger, device = onload
            )
        late_prefix = (
            exact_defer
            and prefetch
            and not pin_encoders
            and not getattr(pipe, "_unsloth_small_host", None)
            and not _env_off(PARTIAL_ENCODER_PIN_ENV)
            and host_pinning_allowed(onload)
        )
        if late_prefix and logger is not None:
            logger.info(
                "diffusion.memory: streamed encoders pin the prefix of their offload groups that fits the host RAM "
                "free when the load returns"
            )
        encoder_left_mib = 0
        if encoders_defer and not pin_encoders:
            encoder_left_mib = _streamed_pin_leftover_mib(dit_pin_mib, pin_dits, onload)
            if encoder_left_mib and logger is not None:
                logger.info(
                    "diffusion.memory: streamed encoders pin a %d MiB prefix of their offload groups in the background",
                    encoder_left_mib,
                )
        if defer or encoders_defer:
            install_group_pin_wait()
        # pinned encoders count against the budget a torchao denoiser pins within, as on the group tier
        if pin_encoders:
            pinned_mib[0] = leaf_mib
        else:
            pinned_mib[0] = encoder_left_mib

        for comp_name, (module, offload_type) in streamed.items():
            kwargs: dict[str, Any] = {
                "onload_device": onload,
                "offload_device": offload,
                "offload_type": offload_type,
            }
            if offload_type == "block_level":
                kwargs["num_blocks_per_group"] = DEFAULT_GROUP_BLOCKS
            if "use_stream" in params:
                kwargs["use_stream"] = use_stream
            if use_stream and "non_blocking" in params:
                kwargs["non_blocking"] = True
            if use_stream and "record_stream" in params:
                kwargs["record_stream"] = prefetch
            pin = pin_dits if offload_type == "block_level" else pin_encoders
            enc_defer = _encoder_defers(comp_name, module, offload_type) and (
                pin or encoder_left_mib > 0
            )
            if use_stream and "low_cpu_mem_usage" in params:
                kwargs["low_cpu_mem_usage"] = not (pin and not defer and not enc_defer)
            kwargs = _torchao_group_offload_kwargs(module, kwargs, pinned_mib)
            _apply_group_offloading_packed(apply_group_offloading, module, kwargs, logger)
            if pin and defer and kwargs.get("use_stream") and kwargs.get("low_cpu_mem_usage"):
                _defer_pinning(pipe, module, onload, logger)
            elif (
                late_prefix
                and not pin
                and offload_type != "block_level"
                and str(comp_name).startswith("text_encoder")
                and not _torchao_weight_classes(module)
                and kwargs.get("use_stream")
                and kwargs.get("low_cpu_mem_usage")
            ):
                _defer_pinning(
                    pipe,
                    module,
                    onload,
                    logger,
                    fast = True,
                    byte_cap = _late_encoder_pin_cap(pipe, module, onload),
                )
            elif enc_defer and kwargs.get("use_stream") and kwargs.get("low_cpu_mem_usage"):
                cap = None if pin else int(encoder_left_mib) << 20
                _defer_pinning(pipe, module, onload, logger, fast = True, byte_cap = cap)
                if not pin:
                    encoder_left_mib = max(0, encoder_left_mib - _module_registered_mib(module))
            installed += 1
            if (
                use_stream
                and offload_type == "block_level"
                and not _pin_top_level_group(module, logger, pinned_mib, device = onload)
            ):
                _skip_top_level_copy_back(module, logger)
            if use_stream and prefetch and offload_type == "block_level":
                install_group_prefetch(module, onload, logger)
            if offload_type == "leaf_level":
                _pin_vision_embedding_device(module)
        if resident_transformer_mib:
            room = int(resident_transformer_mib)
            for module, offload_type in streamed.values():
                if offload_type == "block_level":
                    room -= _keep_groups_resident(module, room, onload, logger)
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "diffusion.memory: granular streaming offload failed after installing hooks "
                "on %d module(s): %s",
                installed,
                exc,
            )
        raise RuntimeError(
            "A diffusion component is larger than the available GPU memory, and granular "
            f"streaming offload could not be enabled: {exc}"
        ) from exc

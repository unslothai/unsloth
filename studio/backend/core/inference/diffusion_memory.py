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

import functools
import os
import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional


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

# none -- all weights resident (fastest; fits only with room). model -- enable_model_cpu_offload(): one top-level
# module on the GPU at a time. group -- apply_group_offloading() on the transformer: stream a few blocks at a time
# with a prefetch stream. streaming -- group-offload the transformer and leaf-offload text encoders that cannot fit
# whole. sequential -- enable_sequential_cpu_offload(): submodule-level (broken for GGUF through diffusers 0.39, kept
# as an escape hatch).
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


_host_memory_reclaim_warning_logged = False
_host_memory_reclaim_unsupported_logged = False

# Optional: a Python without _ctypes must not break the inference stack that imports this module.
# Must stay a module attribute, not a lazy local: the reclaimer tests monkeypatch it.
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


def _keepable(param: Any) -> bool:
    import torch

    data = param.data
    if type(data) is torch.Tensor:
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
) -> int:
    """Pack kept host weights and buffers into page-locked chunks; returns bytes pinned. Each copy keeps its source strides."""
    import torch

    mode = _pin_mode()
    if mode == "off" or not torch.cuda.is_available():
        return 0
    params = [
        (name, p)
        for name, p in module.named_parameters()
        if host.get(name) is not None
        and p.device.type == "cpu"
        and _keepable(p)
        and _pinnable_layout(p.data)
        and not p.data.is_pinned()
    ]
    buffers = [
        (name, b)
        for name, b in _named_buffers(module)
        if (buffer_host or {}).get(name) is b
        and b.device.type == "cpu"
        and _keepable_buffer(b)
        and b.is_contiguous()
        and not b.is_pinned()
    ]
    params = params + [("\0" + name, b) for name, b in buffers]
    sizes = [-(-p.data.nbytes // _PIN_ALIGN) * _PIN_ALIGN for _, p in params]
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
    need = sum(chunks)
    if mode == "auto":
        try:
            import psutil
            vm = psutil.virtual_memory()
        except Exception:  # noqa: BLE001 - no way to size it, so do not lock memory
            return 0
        # psutil reads the host; pinned pages are charged to an enforcing cgroup, so size from the container.
        available, total = vm.available, vm.total
        remainder = _cgroup_available_memory_mib()
        if remainder is not None:
            available = min(available, int(remainder) << 20)
        limit = _cgroup_memory_limit_mib()
        if limit is not None:
            total = min(total, int(limit) << 20)
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
    bufs: list = []
    placed: list = []
    try:
        for (name, p), (index, start) in zip(params, slots):
            if index == len(bufs):
                bufs.append(torch.empty(chunks[index], dtype = torch.uint8, pin_memory = True))
            view = bufs[index][start : start + p.data.nbytes].view(p.dtype)
            view = (
                view.view(p.shape)
                if p.data.is_contiguous()
                else view.as_strided(p.shape, p.data.stride())
            )
            view.copy_(p.data)
            placed.append((name, p, view))
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
    for name, p, view in placed:
        if name.startswith("\0"):
            name = name[1:]
            _set_buffer(module, name, view)
            buffer_host[name] = view
            continue
        p.data = view
        host[name] = view
    return need


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
    # Plain buffers too (torchao-free int8): the stock offload re-copied them every call, 6.3 s vs a 1.2 s denoise on Wan2.2-5B.
    # Kept while the device copy is the same tensor at the same version.
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
                or seen[2] != p.data_ptr()
            ):
                continue
            try:
                p.data = kept
            except Exception:  # noqa: BLE001 - incompatible tensor types: the stock copy below handles it
                pass
        out = init_hook(mod)
        _capture(mod)
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
                    or p.data.data_ptr() != kept.data_ptr()
                ):
                    host.pop(name, None)
                    owner.pop(name, None)
            current = dict(_named_buffers(mod))
            for name in list(buffer_host):
                if current.get(name) is not buffer_host[name]:
                    buffer_host.pop(name, None)
        if onload and not state.get("pin_tried"):
            # Once per module, on its first onload, so loading pays nothing.
            state["pin_tried"] = True
            try:
                pinned = _pin_host_weights(mod, host, logger, buffer_host = buffer_host)
            except Exception:  # noqa: BLE001 - pinning is an optimisation, never a failure
                pinned = 0
            if pinned and logger is not None:
                logger.info(
                    "diffusion.memory: pinned %.1f GiB of %s host weights",
                    pinned / 2**30,
                    type(mod).__name__,
                )
        out = pre_forward(mod, *args, **kwargs)
        if onload:
            for name, p in mod.named_parameters():
                if name in host and owner.get(name) is p and p.device.type != "cpu":
                    version[name] = (p, p._version, p.data_ptr())
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
    # Under group offload, stream the TEXT ENCODERS alongside the transformer instead of keeping them resident.
    # Defaulted so every existing construction is unchanged; set only where that is what makes group offload fit at
    # all (see plan_diffusion_memory).
    stream_text_encoders: bool = False
    # False only on the tier keeping the transformer resident and streaming just the text encoders.
    stream_transformer: bool = True
    # MiB of the streamed denoiser / encoders kept resident (whole groups); None streams every group.
    resident_transformer_mib: Optional[int] = None
    resident_text_encoder_mib: Optional[int] = None

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
        # Apple Silicon shares one CPU/GPU pool: system memory is the budget, offload pointless
        total, free = _system_memory_mib()
        return DeviceMemory(backend, device, "unified_memory", free, total)

    total, free = _system_memory_mib()
    return DeviceMemory(backend, device, "system_memory", free, total)


def reclaimable_snapshot_device_memory(target: Any) -> DeviceMemory:
    """``snapshot_device_memory`` with the caching allocator's RECLAIMABLE bytes credited back as free,
    without flushing it.

    ``torch.cuda.mem_get_info`` reports driver-level free memory, so every block the caching
    allocator holds for reuse counts as used even though the next allocation would take it straight
    back, and a card that has already generated looks much smaller than it is.
    ``settled_snapshot_device_memory`` fixes that with ``empty_cache()``, right once per load but
    wrong per generation: releasing every cached block forces the next forward back to
    ``cudaMalloc`` for all its activations, the exact cost the caching allocator exists to avoid.

    ``memory_reserved() - memory_allocated()`` is that same figure without the flush. Adding it back
    deliberately over-estimates at the margin, because this feeds a REFUSAL and over-estimating free
    memory can only make the guard quieter, never more trigger-happy.

    Only the process's own allocator is credited. Host memory pinned by ``enable_model_cpu_offload``
    lives outside it and is not counted here, which is correct: it is not device memory this
    generation can allocate into.

    A captured CUDA graph's pool (``diffusion_cuda_graph``) is reserved but not allocated, so it is
    credited here, yet ordinary allocations cannot reuse it while a graph holds it, so on a graphed
    load the guard reads about one step of activations high.

    Falls back to the plain snapshot on any failure or non-cuda device."""
    if getattr(target, "device", "cpu") != "cuda":
        return snapshot_device_memory(target)
    snapshot = snapshot_device_memory(target)
    if snapshot.free_mib is None:
        return snapshot
    try:
        import torch
        reclaimable = int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated())
    except Exception:  # noqa: BLE001 -- no allocator reading: the plain snapshot still stands
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
    """How long to wait between the retried reads, honouring ``UNSLOTH_SETTLE_DELAY_S``.

    The retry loop rejects a TRANSIENT undercount, and the ``max`` over the reads does that whatever
    the spacing; the spacing only gives a real transient time to clear on a live card, so production
    keeps the full second.

    A test reaching this through ``_plan_memory`` cannot pass ``delay_s`` and pays the wait for
    nothing, since its snapshots are stubs whose answers do not change with time
    (``test_diffusion_backend.py`` alone spent 142s of a 328s suite here). Callers that can pass
    ``delay_s = 0`` already do.
    """
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
    """``snapshot_device_memory`` hardened against TRANSIENT free-VRAM undercounts on cuda.

    ``torch.cuda.mem_get_info`` is device-wide and instantaneous: a neighbouring process (or a
    just-spawned subprocess context) briefly holding tens of GB at the wrong moment makes an
    empty card look full, and the planner then silently declines the resident/quant fast path
    (measured on B200: a cold FLUX.2-dev int8 load saw free < 74 GB on an idle 183 GB card and
    fell back to offloaded GGUF; the identical retry saw >= 124 GB and went resident). Settle
    the allocator (synchronize + empty_cache, best-effort) and take the MAX free over a few
    spaced reads: a transient can only SHRINK free, so the max rejects transient undercounts
    while a persistent tenant still caps every read. Non-cuda targets keep the single read.

    On mps the budget is system memory, and torch's MPS caching allocator holds the previous
    pipeline's freed buffers as reserved -- which reads as used system memory. Release them first,
    or a swap is budgeted against a pool that only looks too small
    (torch.mps.empty_cache: "Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications")."""
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
        torch.cuda.synchronize()
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

        # Not torch.cuda.mem_get_info directly: on Windows ROCm its free half is an over-report that does not track
        # residency, and this feeds the activation refusal that exists BECAUSE Windows WDDM spills to host RAM instead
        # of raising (#8403). Imported lazily to keep this module free of backend imports at module scope.
        from utils.hardware import trusted_mem_get_info

        free, total = trusted_mem_get_info()
        kind = "discrete_vram"
        try:
            # Query the CURRENT device (mem_get_info reports it); hardcoding 0 would inspect the wrong GPU and
            # misclassify it.
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            if bool(getattr(props, "integrated", False) or getattr(props, "is_integrated", False)):
                kind = "unified_memory"  # e.g. Jetson / integrated SoC
        except Exception:
            pass
        free_mib, total_mib = int(free // (1024 * 1024)), int(total // (1024 * 1024))
        # A ROCm APU sets the same integrated flag and reaches `unified_memory` too, but
        # its free reading is wrong in the OPPOSITE direction (Windows HIP reports
        # free == total, #7072): crediting host memory would enlarge an over-report.
        if kind == "unified_memory" and not _torch_is_rocm(torch):
            free_mib, total_mib = _unified_reclaimable_memory_mib(free_mib, total_mib)
        return free_mib, total_mib, kind
    except Exception:
        return None, None, "discrete_vram"


def _torch_is_rocm(torch) -> bool:
    """Whether this torch is a ROCm build.

    ``version.hip`` alone is not the test: AMD SDK and Radeon wheels leave it unset and
    tag ``__version__`` only, and reading one as CUDA would credit host memory onto an
    APU's already optimistic free reading. Taken from ``LlamaCppBackend`` so the two
    cannot drift, restated inline only for an import that cannot be satisfied.
    """
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._torch_is_rocm(torch)
    except Exception:  # noqa: BLE001 - the answer still has to be right
        return (
            getattr(getattr(torch, "version", None), "hip", None) is not None
            or "rocm" in getattr(torch, "__version__", "").lower()
        )


def _unified_reclaimable_memory_mib(free_mib: int, total_mib: int) -> tuple[int, int]:
    """Credit reclaimable page cache back to an integrated CUDA device's free reading.

    ``cudaMemGetInfo`` reports the kernel's ``MemFree`` here, which counts the page cache
    as used, so a model's own download collapses the budget the load that follows is
    measured against: ``flux.2-klein`` refused at "about 0 GB usable (of the 3 GB
    currently free)" on a 121 GiB machine (#9919). That cache is reclaimed on demand.

    ``MemAvailable`` is the kernel's estimate of what an allocation can have without
    swapping, a floor rather than an optimistic figure, and it is clamped to the driver
    reading and the device total, so a genuinely full machine is refused as it is today.

    Through the llama.cpp helper, which caps it by the cgroup remainder, then applied
    AGAIN as a ceiling: as a lower bound it is thrown away whenever the driver's
    host-wide ``MemFree`` is larger, the normal case in a container.

    Returns the CAPACITY too, since on this device it is the same pool. ``_reserve_mib``
    takes 20% of the total, so leaving a 32 GiB container's at the host's 121 GiB
    reserved 24 GiB and left about 8 GiB usable. Uncapped hosts keep the device total.
    """
    available_mib = _available_system_memory_mib()
    cgroup_mib = _cgroup_available_memory_mib()
    if available_mib is None:
        credited = free_mib
    else:
        credited = max(free_mib, min(int(available_mib), total_mib))
    if cgroup_mib is not None and int(cgroup_mib) <= credited:
        # `<=`, not `<`: the host reading is cgroup-capped already, so equality is the
        # ordinary result in a container, not a sign that the limit does not bind.
        credited = int(cgroup_mib)
    # Capacity is a separate question, and a finite limit answers it whether or not the
    # remainder is what caps the free reading: a tighter host figure does not make a
    # 64 GiB container a 121 GiB device. The LIMIT, never the remainder, which shrinks
    # as the container fills and would refuse a model that fits once one is evicted.
    limit_mib = _cgroup_memory_limit_mib()
    if limit_mib is None:
        capacity = total_mib
    else:
        capacity = min(total_mib, int(limit_mib))
        # Memory above the limit cannot be charged, so it is not free either.
        credited = min(credited, capacity)
    return credited, capacity


def _available_system_memory_mib() -> Optional[int]:
    """Available host RAM in MiB, capped by any enforcing cgroup limit."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._available_system_memory_mib()
    except Exception:  # noqa: BLE001 - the host reading still stands
        return _system_memory_mib()[1]


def _cgroup_available_memory_mib() -> Optional[int]:
    """What an enforcing cgroup will still let this process charge, else None."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._cgroup_available_memory_mib()
    except Exception:  # noqa: BLE001 - no readable limit is the same answer as none
        return None


def _cgroup_memory_limit_mib() -> Optional[int]:
    """The capacity an enforcing cgroup allows, else None. Not the remainder above."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._cgroup_memory_limit_mib()
    except Exception:  # noqa: BLE001 - no readable limit is the same answer as none
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
    import json
    import struct

    try:
        with open(path, "rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            if not 0 < header_len <= 256 * 1024 * 1024:
                return None
            header = json.loads(fh.read(header_len))
        total = 0
        for name, meta in header.items():
            if name != "__metadata__" and name.startswith(prefix):
                start, end = meta["data_offsets"]
                total += int(end) - int(start)
    except Exception:  # noqa: BLE001 - not a readable safetensors header
        return None
    return -(-total // (1024 * 1024)) if total > 0 else None


def estimate_gguf_resident_mib(storage_mib: Optional[int]) -> Optional[int]:
    """Approximate the RESIDENT device size of a GGUF transformer under ``GGUFQuantizationConfig``.

    Weights stay PACKED as quantised bytes; ``GGUFLinear.forward`` dequantises each transiently
    for its matmul and frees it, so the persistent footprint is ~= on-disk size, not unpacked
    bf16. Measured on Z-Image-Turbo: Q2_K 3.64 -> 3.68 GiB, Q8_0 7.22 -> 7.25 GiB resident. The
    transient dequant is covered by the separate runtime headroom. (The prior per-quant expansion
    assumed a full unpack that never happens, over-estimating Q2 ~7.6x and forcing needless offload.)"""
    if storage_mib is None:
        return None
    return int(storage_mib * 1.05)  # margin for allocator + bf16 norms/biases


def estimate_safetensors_dense_mib(
    storage_mib: Optional[int], *, fp8_upcast: bool = False
) -> Optional[int]:
    """Resident size of a safetensors checkpoint, in MiB.

    Usually loads near on-disk size (None passes through). Exception: ``fp8_upcast`` -- an fp8
    single-file transformer loads with no quantization_config, so diffusers upcasts to bf16 (~2x)."""
    if storage_mib is None:
        return None
    if fp8_upcast:
        return storage_mib * 2
    return storage_mib


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
    pixel_scale = ((w * h + cond) * batch) / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT)
    return max(1024, int(8192 * max(0.25, pixel_scale) * _family_activation_multiplier(family)))


def estimate_video_runtime_mib(
    *, width: Optional[int], height: Optional[int], num_frames: Optional[int]
) -> int:
    """Per-call activation / latent / decode headroom for a video generation.

    The pixel-area image estimator undershoots video: the VAE DECODE is the peak -- the clip
    materialises as num_frames full-res fp32 frames plus decoder intermediates. Scale by the
    decoded-clip footprint (frames x H x W x 3 x 4 bytes) with a 3x factor for intermediates +
    the export copy, on top of a fixed denoise-side base.
    """
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
    # text encoder, denoise, untiled decode, tiled decode (all at 1024x1024), max(denoise, tiled decode) at 2048x2048
    "qwen-image-2.1": ((1_849, 666, 7_648, 449, 2_479), (1_849, 2_489, 7_648, 449, 9_602)),
    "flux.1": ((288, 892, 2_666, 2_456, 2_674), (288, 892, 2_666, 2_456, 2_674)),
    "flux.2-klein": ((1_516, 1_160, 2_645, 2_456, 3_876), (1_516, 1_205, 2_677, 2_456, 3_924)),
    "z-image": ((744, 1_199, 2_666, 2_456, 4_327), (744, 1_271, 2_669, 2_456, 4_327)),
}


def calibrated_image_activation(
    family: Optional[str], *, max_speed: bool = True
) -> Optional[CalibratedImageActivation]:
    measured = _MEASURED_IMAGE_ACTIVATION_MIB.get(str(family or ""))
    if measured is None:
        return None
    return CalibratedImageActivation(
        *(int(v * _ACTIVATION_MARGIN) for v in measured[1 if max_speed else 0])
    )


def _reserve_mib(memory_kind: str, base: int) -> int:
    if memory_kind == "unified_memory":
        return max(2048, int(base * 0.20))  # OS + CPU share this pool
    if memory_kind == "system_memory":
        return max(1024, int(base * 0.10))
    return max(2048, int(base * 0.10))


def _rocm_linux_apu_os_room_outside_pool(memory: DeviceMemory) -> bool:
    """Linux ROCm APU whose host RAM outside the pool's free part already covers the unified OS reserve: the pool is
    the amdgpu GTT cap, below physical RAM, so reserving 20% of it too reserves twice. Not Windows (HIP over-reports
    free, #7072); unknown readings answer False."""
    if (
        memory.memory_kind != "unified_memory"
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
    return "discrete_vram" if _rocm_linux_apu_os_room_outside_pool(memory) else memory.memory_kind


def _safe_device_budget_mib(memory: DeviceMemory) -> Optional[int]:
    """Free memory minus a headroom reserve (room for fragmentation + other tenants). None when
    free memory is unknown."""
    if memory.free_mib is None:
        return None
    base = memory.total_mib or memory.free_mib
    return max(0, int(memory.free_mib) - _reserve_mib(_budget_reserve_kind(memory), base))


def _fast_device_budget_mib(memory: DeviceMemory) -> Optional[int]:
    """Free memory minus half the standard reserve (min 2 GiB): an explicit ``fast`` offloads only when resident
    would not fit, not to keep ``auto``'s headroom for other tenants."""
    if memory.free_mib is None:
        return None
    base = memory.total_mib or memory.free_mib
    reserve = max(2048, _reserve_mib(_budget_reserve_kind(memory), base) // 2)
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
# torchao 0.17 already ships the pinnable Int8Tensor (hosted int8 checkpoints) and its v1 int8 weights get the pin
# ops (install_torchao_v1_int8_pin_ops): without this a fresh install (torchao 0.17) got fp8 whenever it offloaded.
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
        need = _module_host_mib(module)
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

        # diffusers restores / record_streams torchao weights via ``tensor_data_names``, which v1 lacks: the payload stayed
        # on the GPU after offload.
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


def _pipe_denoisers_hold_packed(pipe: Any) -> bool:
    """GGUF / int8 weights dequantize a whole Linear per forward: unmeasured by the dense eager table."""
    try:
        import torch
        for name in ("transformer", "transformer_2", "unconditional_transformer", "unet"):
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
    """Whether ``plan``'s resident requirement fits TOTAL device capacity under the standard
    reserve + the 0.85 resident margin -- i.e. an offload decision can only stem from the
    instantaneous FREE reading (something else held VRAM at snapshot time), never from the
    device being too small. Used to retry a declined resident/quant plan once with a fresh
    settled snapshot instead of trusting a single transient undercount. False on any missing
    input (unknown sizes keep today's behaviour)."""
    try:
        required = plan.estimates.get("resident_required_mib")
        budget = total_capacity_budget_mib(plan.device_memory)
    except Exception:  # noqa: BLE001 - malformed plan: no retry
        return False
    if required is None or budget is None:
        return False
    return int(required) <= budget


def total_capacity_budget_mib(memory: DeviceMemory) -> Optional[int]:
    """TOTAL capacity minus the same reserve the free budget takes, times the 0.85 resident margin. None when the
    total is unknown. Shared with the dense prefetch gate so the two cannot disagree."""
    total = memory.total_mib
    if total is None:
        return None
    return int((int(total) - _reserve_mib(_budget_reserve_kind(memory), int(total))) * 0.85)


# Opt-in escape hatch for the unified-memory refusal below: the shortfall check is an estimate, so an operator who
# believes it is wrong can still attempt the load.
UNIFIED_OVERSIZE_ENV = "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD"


def _unified_oversize_override() -> bool:
    return os.environ.get(UNIFIED_OVERSIZE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def unified_memory_shortfall_message(plan: Any, *, family: Optional[str] = None) -> Optional[str]:
    """On UNIFIED device memory, a user-facing refusal when the WEIGHTS alone cannot fit the safe
    budget (else None).

    Unified memory is the one placement with no fallback left. On discrete VRAM an oversized model
    still loads, degrading to group / whole-module CPU offload and streaming from host RAM. On Apple
    Silicon (and integrated CUDA) the CPU and GPU share one pool, so offload moves bytes within that
    pool and frees nothing: ``plan_diffusion_memory`` correctly returns ``none`` and the load then
    allocates past physical memory. There is no torch OOM to catch, because ``_mps_or_cpu_target``
    sets PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable the MPS allocator's hard limit, so the
    failure is the OS killing the process with no Python exception.

    Weights only (``model_dense_mib`` plus the flat base overhead): the per-call runtime headroom is
    a coarse activation / VAE-decode estimate and the mps path already turns on VAE tiling and
    slicing, so counting it would refuse marginal loads that would in fact complete. The weights are
    unavoidable resident bytes, and if they alone do not fit then nothing at generation time can
    rescue the load. Same reasoning as the llama.cpp APU guard.

    Fail-open on anything unknown (no budget, no size), matching the planner's own "budget or model
    size unknown; staying resident".
    """
    if _unified_oversize_override():
        return None
    try:
        memory = plan.device_memory
        # ``system_memory`` (plain CPU) is deliberately excluded: it is an opt-in fringe path, it has swap, and it is
        # not what gets Metal-killed. Only the accelerator-on-system-pool case is guarded.
        if getattr(memory, "memory_kind", None) != "unified_memory":
            return None
        estimates = plan.estimates
        budget = estimates.get("safe_device_budget_mib")
        weights = estimates.get("model_dense_mib")
        overhead = estimates.get("base_overhead_mib")
        free = getattr(memory, "free_mib", None)
    except Exception:  # noqa: BLE001 - malformed plan: never block the load
        return None
    if budget is None or weights is None:
        return None
    required = int(weights) + int(overhead or 0)
    if required <= int(budget):
        return None
    what = f"'{family}'" if family else "This model"
    free_note = (
        f"of the {int(free) / 1024:.0f} GB currently free, after reserving room for the "
        "operating system"
        if free is not None
        else "after reserving room for the operating system"
    )
    return (
        f"{what} needs about {required / 1024:.0f} GB of memory for its weights, but only "
        f"about {int(budget) / 1024:.0f} GB is usable on this device ({free_note}). This device "
        "has unified memory, so the CPU and GPU share one pool: offloading weights to the CPU "
        "frees nothing, and the operating system stops an oversized load outright instead of "
        "reporting an out-of-memory error. Use a smaller or more quantized model (for example a "
        "lower GGUF quant), or free memory by closing other applications. (Server "
        f"installs that know this estimate is wrong can set {UNIFIED_OVERSIZE_ENV}=1 to attempt "
        "the load anyway.)"
    )


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
    reading is the memory the load actually gets."""
    message = unified_memory_shortfall_message(plan, family = family)
    if message is None:
        return
    if logger is not None:
        logger.error("diffusion.memory: refusing oversized unified-memory load: %s", message)
    raise RuntimeError(message)


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
    """Pick an offload policy plus VAE memory savers for the current load.

    ``model_dense_mib`` is the resident size of all weights; ``companion_dense_mib`` is just the
    companions, which stay resident under group offload while the transformer streams block by
    block. ``text_encoder_dense_mib`` is the TEXT-ENCODER share of that companion total, which
    unlocks a second group tier; None means "no split available" and reproduces the pre-split
    decision exactly. ``explicit_offload`` is the back-compat ``cpu_offload=True`` request (forces
    model offload).

    Policies by speed/VRAM tradeoff:

    none  - everything resident: fastest, highest VRAM.

    group - stream the transformer, companions resident: near-resident speed, moderate cut.

    group + streamed text encoders - as above, but the encoders stream too: they run ONCE, before
    step 0, so this costs one extra host-to-device pass per call rather than a per-step one.

    model - offload every component: lowest VRAM, slow.

    streaming - stream transformer blocks and text-encoder leaves when one component cannot fit.
    """
    mode = normalize_memory_mode(requested_mode) or MEMORY_MODE_AUTO
    can_offload = bool(getattr(target, "supports_model_cpu_offload", False))
    budget = _safe_device_budget_mib(device_memory)
    required = _sum_required(model_dense_mib, runtime_headroom_mib, base_overhead_mib)
    # The resident floor under group offload: companions stay, the transformer streams.
    group_floor = _sum_required(companion_dense_mib, runtime_headroom_mib, base_overhead_mib)
    # A SECOND floor, for the same tier with the text encoders streamed as well. The encoders are the largest companion
    # on most families (Z-Image: 8.0 of 8.2 GB) and they are used exactly once, before step 0, so holding them resident
    # for the whole denoise reserves their bytes for nothing. Streaming them leaves the VAE as the only resident
    # companion. Computed only when BOTH terms are known: an unknown split must reproduce the previous decision, never
    # guess a smaller floor. Clamped at 0 because the two terms can come from different sources.
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

    # The best tier available when the weights do not fit resident, in speed order: plain group (companions resident)
    # beats group with streamed encoders (one extra host-to-device pass per CALL) beats whole-module offload (every
    # component paged per STEP -- the 48-minute case).
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
        # MPS / CPU cannot stream to a separate device; on unified memory offload just shuffles bytes within the same
        # pool.
        policy = OFFLOAD_NONE
        if device_memory.is_unified:
            reasons.append("unified/system memory: CPU offload frees no device memory")
        else:
            reasons.append(f"{device_memory.backend}: CPU offload unavailable; staying resident")
    elif mode == MEMORY_MODE_FAST:
        policy = OFFLOAD_NONE
        fast_budget = _fast_device_budget_mib(device_memory)
        estimates["resident_budget_mib"] = fast_budget
        if fast_budget is not None and required is not None and required > fast_budget:
            policy, stream_text_encoders, stream_transformer = _offload_tier()
            reasons.append(
                f"fast requested but weights do not fit resident ({required} MiB needed, "
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
        # Both group tiers also fail when the companion split is simply UNKNOWN, and reporting that as "exceeds
        # budget" sends anyone reading the log looking for a card that is too small.
        if group_floor is None:
            reasons.append(
                "companion size unknown, so no streamed tier can be sized; "
                "whole-module offload of every component"
            )
        else:
            reasons.append("companions exceed budget; whole-module offload of every component")

    # The legacy cpu_offload flag applies only when no memory_mode was supplied, so an explicit `fast` stays resident.
    if (
        explicit_offload
        and normalize_memory_mode(requested_mode) is None
        and policy == OFFLOAD_NONE
        and can_offload
        and not device_memory.is_unified
    ):
        policy = OFFLOAD_MODEL
        reasons.append("explicit cpu_offload overrides resident placement")

    # VAE savers cap the high-res decode spike. Slicing (one image at a time) is EXACT, so enable it on any offload
    # tier. Tiling is only bit-identical for a single tile (<=1MP), so restrict it to the lowest tiers. Group offload
    # keeps the VAE resident.
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
            free_mib = device_memory.free_mib,
            policy = policy,
            stream_transformer = stream_transformer,
        )
        if faster is not None:
            policy, stream_text_encoders, stream_transformer, tile, reason = faster
            any_offload = policy != OFFLOAD_NONE
            reasons.append(reason)
            estimates["calibrated_headroom_mib"] = calibrated_activation.headroom(tile)
    return MemoryPlan(
        requested_mode = mode,
        offload_policy = policy,
        vae_tiling = tile,
        vae_slicing = any_offload,
        device_memory = device_memory,
        estimates = estimates,
        reasons = tuple(reasons),
        # only ever meaningful under group offload; every other tier already places the encoders
        stream_text_encoders = stream_text_encoders and policy == OFFLOAD_GROUP,
        stream_transformer = stream_transformer or policy != OFFLOAD_GROUP,
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
    for rank, (fits, tier) in enumerate(candidates):
        if rank >= flat_rank:
            return None
        if fits:
            return tier
    return None


def _streamable_components(pipe: Any, torch: Any) -> dict[str, tuple[Any, str]]:
    """Component name -> (module, group-offload type) for what streaming keeps off the device.

    Every denoiser streams block by block; every text encoder streams leaf by leaf. Anything else
    (the VAE, an image encoder) has no granular hook here and stays resident, so this is also the
    set ``refine_memory_plan_for_components`` is allowed to size the policy against."""
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
    """Fit-check an explicit ``memory_mode=balanced`` plan against the LOADED companions.

    Balanced never checked the budget, so Qwen-Image-2.1 int8 kept its 9 GiB fp8 encoder resident and OOMed at 16 /
    12 / 8 GB. Judged after the load because the pre-load estimate can be the dense encoder size (16.7 vs 9.0 GB).
    Walks only the streamed tiers: encoders streamed too, else whole-module offload (granular streaming for torchao
    denoisers, which cannot take per-forward ``Module.to()``)."""
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
    """Replace whole-module offload when a loaded component cannot fit on the device.

    The coarse planner runs before the pipeline exists; at this point the weights are still on CPU,
    so their actual packed storage is a better signal than family or cache estimates. Keep
    whole-module offload when every component fits, preserving its faster execution; when one
    cannot, use granular streaming so no forward needs to materialise that component in full.

    Only a STREAMABLE component justifies the switch, and only if the components streaming cannot
    hook still fit resident TOGETHER: whole-module offload onloads one at a time, streaming holds
    all of them at once, so refining past either bound would trade one OOM for another.
    """
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


# The flat planner reserves 8192 MiB/MP of activations: Qwen-Image-2.1 (16525 MiB of weights, 1849 MiB measured peak)
# streamed its encoder on 24 GB.
MEASURED_ACTIVATION_ENV = "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION"
PARTIAL_RESIDENT_ENV = "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT"

# Worst measured CUDA MiB above the resident weights, one 1024x1024 image, encoder + every step + VAE decode, torchao
# int8 / fp8 denoisers on the compiled tiers (Qwen-Image-2.1: encoder 1849 streamed, denoise 1442, decode 1730).
_MEASURED_IMAGE_PEAK_MIB: dict[str, int] = {"qwen-image-2.1": 1849}
_MEASURED_PEAK_SPEED_MODES = ("default", "max")
_MEASURED_PEAK_MARGIN = 1.15
_MEASURED_PEAK_ROUND_MIB = 256

# Dense denoisers, eager tier: worst CUDA MiB above resident weights, fp16, 1024x1024, encode + steps + VAE decode.
# family -> (peak MiB, largest loaded DiT MiB it covers; a bigger DiT keeps the flat estimate)
MEASURED_ACTIVATION_DENSE_ENV = "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE"
_MEASURED_DENSE_EAGER_PEAK_MIB: dict[str, tuple[int, int]] = {
    "flux.2-klein": (2455, 7800),
    "flux.1": (2448, 23800),
    "qwen-image": (4248, 40900),
}
_MEASURED_DENSE_SPEED_MODES = ("off",)


def _env_off(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("0", "off", "false", "no")


def measured_image_runtime_mib(
    family: Optional[str],
    speed_mode: Optional[str],
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    batch_size: int = 1,
    dense_transformer_mib: Optional[int] = None,
    compute_bytes: int = 2,
) -> Optional[int]:
    """Measured runtime headroom for ``family`` at this size, or None (unmeasured: use the flat estimate).
    ``dense_transformer_mib`` selects the dense eager table; ``compute_bytes`` scales it (fp32 Qwen-Image: 8480 vs 4248)."""
    if _env_off(MEASURED_ACTIVATION_ENV):
        return None
    if dense_transformer_mib is not None:
        if _env_off(MEASURED_ACTIVATION_DENSE_ENV):
            return None
        entry = _MEASURED_DENSE_EAGER_PEAK_MIB.get(str(family or "").lower())
        if entry is None or str(speed_mode or "") not in _MEASURED_DENSE_SPEED_MODES:
            return None
        peak, max_dit = entry
        widen = max(2, int(compute_bytes or 2))
        peak, max_dit = peak * widen // 2, max_dit * widen // 2
        if int(dense_transformer_mib) <= 0 or int(dense_transformer_mib) > max_dit:
            return None
    else:
        peak = _MEASURED_IMAGE_PEAK_MIB.get(str(family or "").lower())
        if peak is None or str(speed_mode or "") not in _MEASURED_PEAK_SPEED_MODES:
            return None
    w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
    h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
    scale = max(
        1.0,
        (w * h * max(1, int(batch_size or 1))) / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT),
    )
    need = peak * scale * _MEASURED_PEAK_MARGIN
    step = _MEASURED_PEAK_ROUND_MIB
    return int(-(-need // step) * step)


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
# Whole-resident tier slack: 10% of the card, min 1 GiB, instead of the flat reserve + base overhead sized for an
# unmeasured activation. Free memory is read after CUDA init and the measured peak covers encode, steps and decode, so
# the slack only covers fragmentation and lazily loaded kernels.
_RESIDENT_DIT_SLACK_FRACTION = 0.10
_RESIDENT_DIT_SLACK_MIN_MIB = 1024


def _resident_dit_slack_mib(memory: Any) -> int:
    base = getattr(memory, "total_mib", None) or getattr(memory, "free_mib", None) or 0
    return max(_RESIDENT_DIT_SLACK_MIN_MIB, int(int(base) * _RESIDENT_DIT_SLACK_FRACTION))


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


def refine_plan_from_loaded_weights(
    pipe: Any,
    plan: MemoryPlan,
    *,
    family: Optional[str],
    speed_mode: Optional[str],
    logger: Any = None,
) -> MemoryPlan:
    """Re-place a streamed ``auto`` load from its LOADED weights and the family's measured activation peak.

    Keeps the flat plan's offload hooks and makes whole groups resident (denoiser first, then encoders) within the
    safe budget left after the measured peak x margin and the base overhead. No-op for explicit modes, non-CUDA / unified memory, unmeasured families (dense denoisers: the eager-tier table)
    or speed tiers, non-torchao denoisers and ``model`` plans."""
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
        if dense_mib is not None and (compute_bytes is None or _pipe_denoisers_hold_packed(pipe)):
            # measured on fp16 / fp32 cards only; bf16 cards can compile later
            return plan
        headroom = measured_image_runtime_mib(
            family, speed_mode, dense_transformer_mib = dense_mib, compute_bytes = compute_bytes or 2
        )
        if headroom is None:
            return plan
        overhead = int(plan.estimates.get("base_overhead_mib") or DEFAULT_BASE_OVERHEAD_MIB)
        floor = headroom + overhead + other
        estimates = dict(plan.estimates)
        estimates.update(
            measured_runtime_headroom_mib = headroom,
            measured_dense_transformer_mib = dense_mib,
            measured_compute_bytes = compute_bytes,
            loaded_transformer_mib = dit,
            loaded_text_encoder_mib = encoders,
            loaded_other_mib = other,
        )
        if _env_off(PARTIAL_RESIDENT_ENV):
            return plan
        # The flat plan's hooks stay installed and whole groups are kept resident within the room, so an oversized
        # request can stream them again (release_resident_groups) and run exactly as the flat plan would.
        stream_te = bool(getattr(plan, "stream_text_encoders", False))
        room = budget - floor
        if policy == OFFLOAD_GROUP and not stream_te:
            room -= encoders  # resident companions
        whole_dit = False
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
        te_room = max(0, room - dit_room) if stream_te and policy == OFFLOAD_GROUP else 0
        if dit_room <= 0 and te_room <= 0:
            return plan
        if whole_dit:
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
                f"{int(dit_room)} MiB of the {dit} MiB transformer and {int(te_room)} MiB of the {encoders} MiB "
                f"encoders stay resident (measured peak {headroom} MiB); the rest streams"
                + (
                    f" (whole transformer within free memory less a {_resident_dit_slack_mib(memory)} MiB slack;"
                    f" {encode_room} MiB of it while the encoders run)"
                    if whole_dit
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

        def _tensors(group: Any) -> list:
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
        for group in ordered:
            if getattr(group, "offload_to_disk_path", None):
                break
            tensors = _tensors(group)
            need = sum(sum(_storage_nbytes(t)) for t in tensors)
            if getattr(group, "_unsloth_resident", False):
                left -= need
                continue
            if need > left or (only is not None and id(group) not in only):
                # too large (a later, smaller group may still fit), or another release's group
                continue
            cpu = getattr(group, "cpu_param_dict", None) or {}
            for t in tensors:
                if _placed_on(t, onload.type, is_torchao):
                    continue
                moved = cpu.get(t, t).to(onload)
                if is_torchao(t):
                    go._swap_torchao_tensor(t, moved)
                else:
                    t.data = moved
            # host copies stay: release_resident_groups streams the group again for an oversized request
            group._unsloth_streamed_hooks = (
                group.__dict__.get("onload_"),
                group.__dict__.get("offload_"),
            )
            group.onload_ = _resident_onload(getattr(group, "stream", None))
            group.offload_ = noop
            group._unsloth_resident = True
            group._unsloth_resident_bytes = need
            left -= need
            kept += need
        state["streamed"] = _streamed_group_count(ordered)
        if kept and onload.type == "cuda":
            torch.cuda.synchronize(onload)
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


def _streamed_group_count(groups: list) -> int:
    return sum(
        1
        for g in groups
        if getattr(g, "stream", None) is not None and not getattr(g, "_unsloth_resident", False)
    )


def _release_group(group: Any) -> None:
    onload_, offload_ = getattr(group, "_unsloth_streamed_hooks", (None, None))
    for name, fn in (("onload_", onload_), ("offload_", offload_)):
        if fn is not None:
            setattr(group, name, fn)
        else:
            group.__dict__.pop(name, None)
    group._unsloth_resident = False
    group.offload_()


def release_resident_groups(
    pipe: Any,
    need_mib: int,
    logger: Any = None,
    denoisers_only: bool = False,
    reason: str = "an oversized request",
) -> Optional[Callable[[], None]]:
    """Stream resident offload groups again until ``need_mib`` is freed (text encoders first, then the denoiser's
    blocks from the last); returns a callable pinning exactly those groups again, or None when nothing was resident.
    A request larger than the measured placement reserved (reference images, a bigger canvas, a batch) then runs as
    the flat plan. ``denoisers_only`` leaves the text encoders' groups alone (the prompt encode needs them)."""
    try:
        import torch

        modules = [
            m
            for m in (getattr(pipe, "components", {}) or {}).values()
            if getattr(m, "_unsloth_resident_room", None)
            and not (denoisers_only and _is_text_encoder_module(pipe, m))
        ]
        if not modules or int(need_mib) <= 0:
            return None
        modules.sort(key = lambda m: 0 if _is_text_encoder_module(pipe, m) else 1)
        left = int(need_mib) * 1024 * 1024
        released: list = []
        streamed: set = set()
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
                _release_group(group)
                streamed.add(id(group))
                left -= int(getattr(group, "_unsloth_resident_bytes", 0))
                if module not in released:
                    released.append(module)
        if not released:
            return None
        for module in released:
            state = getattr(module, "_unsloth_stream_state", None)
            if isinstance(state, dict):
                state["streamed"] = _streamed_group_count(_offload_groups(module))
        device = getattr(released[0], "_unsloth_resident_device", None)
        if device is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        if logger is not None:
            logger.info(
                "diffusion.memory: streaming %d MiB of resident groups for %s",
                (int(need_mib) * 1024 * 1024 - max(left, 0)) >> 20,
                reason,
            )

        def restore() -> None:
            for module in released:
                _keep_groups_resident(
                    module,
                    module._unsloth_resident_room,
                    module._unsloth_resident_device,
                    logger,
                    only = streamed,
                )

        return restore
    except Exception as exc:  # noqa: BLE001 - the guard still refuses what cannot fit
        if logger is not None:
            logger.warning("diffusion.memory: releasing resident groups failed (%s)", exc)
        return None


def _is_text_encoder_module(pipe: Any, module: Any) -> bool:
    for name, component in (getattr(pipe, "components", {}) or {}).items():
        if component is module:
            return str(name).startswith("text_encoder")
    return False


def install_encode_release(
    pipe: Any,
    plan: Any,
    logger: Any = None,
) -> int:
    """While a text encoder runs, stream the whole-resident denoiser back to the flat room (the partial placement's
    encode state); pin it back on return. Returns the number of encoders hooked."""
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

        def _before(module: Any, args: Any) -> None:
            owner = pipe_ref()
            if owner is None or pending:
                return  # nested encoder call: the outer one already released
            restore = release_resident_groups(
                owner, surplus, logger, denoisers_only = True, reason = "the prompt encode"
            )
            pending.append(restore)
            if restore is not None and torch.cuda.is_available():
                # freed blocks sit in the default stream's pool; the encoder onloads on the copy stream
                torch.cuda.empty_cache()

        def _after(module: Any, args: Any, output: Any) -> None:
            if not pending:
                return
            restore = pending.pop()
            if restore is not None:
                restore()

        before = disable(_before) if callable(disable) else _before
        after = disable(_after) if callable(disable) else _after
        hooked = 0
        for name, module in (getattr(pipe, "components", {}) or {}).items():
            if not str(name).startswith("text_encoder") or not isinstance(module, torch.nn.Module):
                continue
            handles = (
                module.register_forward_pre_hook(before),
                # always_call: an encode that raises (cancel, OOM) still pins the denoiser back
                module.register_forward_hook(after, always_call = True),
            )
            module._unsloth_encode_release = handles
            hooked += 1
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
) -> int:
    """MiB a request needs beyond what the measured placement reserved (0 when it fits or none was made)."""
    reserve = getattr(pipe, "_unsloth_measured_reserve", None)
    if not reserve:
        return 0
    headroom, family, speed_mode = reserve[:3]
    dense_mib = reserve[3] if len(reserve) > 3 else None
    compute_bytes = reserve[4] if len(reserve) > 4 and reserve[4] else 2
    need = measured_image_runtime_mib(
        family,
        speed_mode,
        width = width,
        height = height,
        batch_size = batch_size,
        dense_transformer_mib = dense_mib,
        compute_bytes = compute_bytes,
    )
    if need is None:
        return 0
    cond = max(0, int(condition_pixels or 0)) * max(1, int(batch_size or 1))
    need += int(
        8192 * cond / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT) * _MEASURED_PEAK_MARGIN
    )
    return max(0, int(need) - int(headroom))


def apply_memory_plan(
    pipe: Any,
    plan: MemoryPlan,
    *,
    device: str,
    placement_device: Optional[str] = None,
    logger: Any = None,
) -> tuple[str, bool]:
    """Apply ``plan`` to a built diffusers pipeline: enable the VAE savers then place / offload
    the weights. Exactly one placement call runs (fully resident or wired for offload, never both).

    Returns the ``(offload_policy, vae_tiling)`` ACTUALLY engaged, which can differ from the plan:
    tiling is a no-op where there's no tiling control, and group / sequential offload fall back to
    whole-module offload if unsupported (e.g. sequential is broken for GGUF through diffusers 0.39).

    ``placement_device`` is the INDEXED string when a card was selected ("cuda:1"), and is what
    every diffusers handoff below receives. A bare "cuda" is not equivalent to the CPU-offload
    APIs: ``enable_model_cpu_offload`` reads the index off the device and, finding none, falls
    back to ``_offload_gpu_id = 0`` and onloads to cuda:0 (pipeline_utils.py, diffusers 0.39), so
    the modules would page onto the very card the selection existed to avoid while generation ran
    on another. ``device`` stays bare for anything reading it as a policy string."""
    placement = placement_device or device
    tiling_engaged = False
    if plan.vae_tiling:
        tiling_engaged = _enable_vae_saver(pipe, "enable_vae_tiling", "enable_tiling", logger)
    if plan.vae_slicing:
        _enable_vae_saver(pipe, "enable_vae_slicing", "enable_slicing", logger)

    def _fallback_to_model_offload() -> None:
        # The GROUP plan set vae_tiling=False (the VAE stays resident). Dropping to whole-module offload is the low-VRAM
        # case where the decode spike can OOM, so turn tiling on now.
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
        if not _apply_group_offload(pipe, placement, logger, **group_kwargs):
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
    else:
        from .diffusion_fast_load import fast_upload

        components = getattr(pipe, "components", None)
        modules = list(components.values()) if isinstance(components, dict) else []
        with fast_upload(modules, placement, logger = logger):
            pipe.to(placement)
    return policy, tiling_engaged


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
    """Keep a leaf-offloaded Qwen3-VL vision tower's position embeddings on the compute device.

    ``Qwen3VLVisionModel.fast_pos_embed_interpolate`` reads ``self.pos_embed.weight.device`` while
    group offload still holds that weight on the CPU, so every image-conditioned Qwen-Image-2.1 call
    on a streamed encoder raised a device mismatch. The result follows ``grid_thw``'s device."""
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
    """Pinnable host MiB leaving ``max(4 GiB, 15%)`` free, or None if unreadable."""
    total, _available = _system_memory_mib()
    available = _available_system_memory_mib()
    if total is None or available is None:
        return None
    # Same container sizing as _pin_host_weights: pinned pages are charged to an enforcing cgroup.
    limit = _cgroup_memory_limit_mib()
    if limit is not None:
        total = min(int(total), int(limit))
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
) -> tuple[bool, bool]:
    """(pin transformer, pin encoders), transformer first."""
    forced = str(os.environ.get(GROUP_OFFLOAD_PIN_ENV, "")).strip().lower()
    if forced in ("0", "off", "false", "no"):
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


def _background_pin_enabled() -> bool:
    return (os.environ.get(BACKGROUND_PIN_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


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


def _fast_pin_supported() -> bool:
    if (os.environ.get(FAST_PIN_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    # WDDM / WSL2 cap page-locked memory; ROCm's register path is unmeasured.
    if not sys.platform.startswith("linux") or _pinned_memory_capped():
        return False
    try:
        import torch

        if getattr(torch.version, "hip", None) or not torch.cuda.is_available():
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


def registered_host_copy(src: Any) -> Any:
    """A contiguous copy of host tensor ``src`` in CUDA-registered (pinned) memory."""
    import numpy as np
    import torch

    holder = _RegisteredHostBuffer(src.numel() * src.element_size())
    out = torch.from_numpy(np.asarray(holder)).view(src.dtype).view(src.shape)
    out.copy_(src)
    holder.register()
    return out


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
    ):
        import threading

        self.fast = bool(fast)
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
        return [
            (tensor, src)
            for tensor, src in list(group.cpu_param_dict.items())
            if type(src) is torch.Tensor
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

    def _run(self) -> None:
        import time

        import torch

        start = time.perf_counter()
        failed = None
        pool = None
        try:
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
                        (tensor, src, pool.submit(registered_host_copy, src))
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
                    # One pin per tensor, exactly what the eager apply makes: chunk views streamed ~10% slower on
                    # LTX-2.3.
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
                self._done[id(group)].set()
        except Exception as exc:  # noqa: BLE001 - diffusers pins what is left on each onload
            failed = exc
        finally:
            if pool is not None:
                pool.shutdown(wait = True, cancel_futures = True)
            for event in self._done.values():
                event.set()
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
    """Block until every deferred pinner on ``pipe`` is done; returns the seconds waited.

    A render that overlaps the pinner ran its warm renders 2.5 to 3 s slower on an A100 (LTX-2.3, n=15 per arm);
    letting the pinner finish first matched the eager pin. So the pin overlaps the idle time after the load, not
    the first render."""
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


def _defer_pinning(pipe: Any, module: Any, device: Any, logger: Any) -> bool:
    groups = _offload_groups(module)
    if not groups:
        return False
    fast = bool(getattr(pipe, _FAST_PIN_REQUEST_ATTR, False)) and _fast_pin_supported()
    pinner = _GroupPinner(module, groups, device, logger, fast = fast)
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
) -> bool:
    """Onload a block-streamed DiT's top-level group (embedders, norm_out, proj_out) from one pinned host copy.

    diffusers gives that group no stream: every forward uploaded it from pageable memory and every offload copied it
    back to a fresh host buffer, which inference never needs. Now the onload is an async H2D on the compute stream and
    the offload only re-points the tensors; VRAM unchanged. Skipped for torchao weights (not re-pointable via
    ``.data``) and when the copy does not fit the pinnable host RAM."""
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
    """Stream the transformer a few blocks at a time via diffusers group offloading, keeping the
    smaller components resident. Returns False (caller falls back to whole-module) on any failure.

    ``stream_text_encoders`` extends the streamed set to every ``text_encoder*`` module. Off by
    default: keeping them resident is faster when there is room. The planner turns it on only
    where it is the difference between group offload and whole-module offload.

    ``stream_transformer=False`` (only with ``stream_text_encoders``) keeps every DiT resident."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return False
    if not stream_transformer and not stream_text_encoders:
        return False
    installed = 0
    try:
        import inspect

        import torch
        from diffusers.hooks import apply_group_offloading

        install_group_offload_buffer_restore()
        install_group_offload_hooks_eager()

        # A dual-DiT pipeline (Ideogram 4) carries a second denoiser as large as the first, so stream every DiT and keep
        # only smaller companions resident.
        streamed: dict[str, Any] = {}
        if stream_transformer:
            streamed["transformer"] = transformer
            for extra in ("transformer_2", "unconditional_transformer"):
                module = getattr(pipe, extra, None)
                if isinstance(module, torch.nn.Module):
                    streamed[extra] = module
        # The text encoders are streamed SEPARATELY from the DiTs, and tolerantly (see the apply loop below). Kept in
        # their own dict so the resident placement loop still skips them.
        streamed_encoders: dict[str, Any] = {}
        if stream_text_encoders:
            # A text encoder runs ONCE, before step 0, so residency buys it nothing while it costs its bytes for every
            # step of the denoise. Streaming it does two things: the resident loop below skips it (it is no longer
            # placed with comp.to(onload)), and group hooks page it in for that single encode. Component names, not
            # attributes, so a family with text_encoder / text_encoder_2 / text_encoder_3 is covered without a
            # per-family list.
            for name, comp in getattr(pipe, "components", {}).items():
                if name.startswith("text_encoder") and isinstance(comp, torch.nn.Module):
                    streamed_encoders[name] = comp

        onload = torch.device(device)
        use_stream = onload.type == "cuda"  # overlap H2D copies with compute
        gkwargs: dict[str, Any] = {
            "onload_device": onload,
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": DEFAULT_GROUP_BLOCKS,
            "use_stream": use_stream,
        }
        # On the CUDA stream path, overlap each block's H2D copy with compute. Lossless, and gated on the signature so
        # older diffusers still works.
        _params = inspect.signature(apply_group_offloading).parameters
        if use_stream:
            if "non_blocking" in _params:
                gkwargs["non_blocking"] = True
            if "record_stream" in _params:
                gkwargs["record_stream"] = True
        pin_streamed = (True, True)
        if stream_text_encoders and "low_cpu_mem_usage" in _params:
            # The streamed path PINS every offloaded parameter in host RAM when a copy stream is in use (diffusers
            # group_offloading `_init_cpu_param_dict`), which is a fine trade when group offload was already the plan.
            # Pin only what host RAM covers (unbounded pinning hurt #8188); unpinned re-pins on every onload.
            pin_streamed = _streamed_pin_plan(
                sum(_module_host_mib(m) for m in streamed.values()),
                sum(_module_host_mib(m) for m in streamed_encoders.values()),
                logger,
            )
            gkwargs["low_cpu_mem_usage"] = not pin_streamed[0]
        if getattr(pipe, "_unsloth_small_host", None) and "low_cpu_mem_usage" in _params:
            # pinning would copy the small-host route's memory-mapped bytes back into host RAM
            pin_streamed = (False, False)
            gkwargs["low_cpu_mem_usage"] = True
        # ``background_pin``: a module the plan pins is applied unpinned and handed to a _GroupPinner, which the caller
        # starts with start_background_pins once the load has committed.
        if background_pin is None:
            background_pin = bool(getattr(pipe, _BACKGROUND_PIN_REQUEST_ATTR, False))
        defer = (
            background_pin
            and use_stream
            and "low_cpu_mem_usage" in _params
            and _background_pin_enabled()
        )
        if defer:
            install_group_pin_wait()
        # Place the smaller components resident BEFORE attaching the transformer group-offload hooks: a companion .to()
        # OOM then returns False with no hooks installed, and diffusers rejects enable_model_cpu_offload once group
        # hooks exist.
        for name, comp in getattr(pipe, "components", {}).items():
            if name in streamed or name in streamed_encoders:
                continue
            if isinstance(comp, torch.nn.Module):
                comp.to(onload)
        # Encoders the plan pins count against the same budget as any torchao denoiser pinned below.
        pinned_mib = [
            sum(_module_host_mib(m) for m in streamed_encoders.values())
            if stream_text_encoders and pin_streamed[1]
            else 0
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
                apply_group_offloading(
                    module, **_torchao_group_offload_kwargs(module, gkwargs, pinned_mib)
                )
            installed += 1
            if use_stream and not _pin_top_level_group(module, logger, pinned_mib):
                _skip_top_level_copy_back(module, logger)
            if use_stream and gkwargs.get("record_stream"):
                install_group_prefetch(module, onload, logger)
        if resident_transformer_mib:
            room = int(resident_transformer_mib)
            for module in streamed.values():
                room -= _keep_groups_resident(module, room, onload, logger)
        # The encoders come AFTER the DiTs and are applied one by one, each failure absorbed. A text encoder is a far
        # less well-trodden target for block-level group offloading than a DiT (a family whose encoder exposes no
        # recognisable block list can refuse), and this tier is a rescue: the alternative to streaming an encoder is
        # keeping it resident, which is what happened before this tier existed. Letting one refusal join the all-or-
        # nothing DiT loop would turn a slow-but-working load into a hard failure, because by then hooks are installed
        # and whole-module offload can no longer be used as a fallback. So a refusal places that encoder resident
        # instead: the plan's floor becomes optimistic by that encoder's bytes, and the load still runs. Leaf level, not
        # the DiTs' block level: an encoder is not a stack of uniform blocks, so _streamable_components and
        # _apply_streaming_offload already classify every text_encoder* that way. Reusing the transformer's kwargs here
        # grouped the whole encoder as one unit, which is the residency the planner's floor was chosen to avoid -- the
        # plan said leaf and the application said block. num_blocks_per_group goes with it: leaf level has no blocks.
        ekwargs = {k: v for k, v in gkwargs.items() if k != "num_blocks_per_group"}
        ekwargs["offload_type"] = "leaf_level"
        if "low_cpu_mem_usage" in gkwargs:
            ekwargs["low_cpu_mem_usage"] = not pin_streamed[1]
        transformer_demoted = False
        te_room = int(resident_text_encoder_mib or 0)
        for name, module in streamed_encoders.items():
            try:
                if defer and not ekwargs.get("low_cpu_mem_usage", False):
                    apply_group_offloading(module, **{**ekwargs, "low_cpu_mem_usage": True})
                    _defer_pinning(pipe, module, onload, logger)
                else:
                    apply_group_offloading(module, **ekwargs)
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
                # a leaf-level apply can raise after hooking part of the encoder; resident means no hooks at all, or
                # the applied VRAM floor reads the whole encoder as streamed while its unhooked layers stay on the card
                _remove_group_offload_hooks(module)
                _drop_deferred_pinning(pipe, module)
                module.to(onload)
        return True
    except Exception as exc:  # noqa: BLE001 - fall back to whole-module offload
        if installed:
            # An earlier streamed module already has hooks but a later one failed: the pipe is in a PARTIAL
            # group-offload state enable_model_cpu_offload rejects, so propagate the real failure instead of a
            # misleading hook error.
            if logger is not None:
                logger.warning(
                    "diffusion.memory: group offload failed after installing hooks on %d "
                    "module(s) (%s); cannot fall back to whole-module offload",
                    installed,
                    exc,
                )
            raise
        if logger is not None:
            logger.warning(
                "diffusion.memory: group offload failed (%s); falling back to "
                "whole-module offload",
                exc,
            )
        return False


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
    """Decide whether this generation's ACTIVATIONS fit the free device budget: run it as loaded,
    run it with the VAE tiled, or refuse.

    ``source_driven`` says the refused size comes from an UPLOADED image rather than the Resolution
    control (inpaint / extend / upscale / edit); telling those callers to generate at a smaller
    resolution points them at a control that cannot change the number in the refusal. Same verdict
    either way, only the remedy sentence differs.

    Why tiling and not another offload tier: weights can be offloaded, activations cannot. Every
    offload tier moves WEIGHTS between host and device, while the latents, attention buffers and VAE
    intermediates a forward pass allocates have to be on the device while it runs. What CAN shrink
    is the VAE decode, by tiling (as ComfyUI does); not under ``quadratic_attention``, where the
    denoiser is what grows.

    Refusing has to happen HERE rather than being left to torch. On Linux the overrun raises
    ``torch.OutOfMemoryError`` and the job dies cleanly; on Windows WDDM (including ROCm) it does
    not raise at all -- the driver satisfies the overflow from system RAM as "non-local" GPU memory,
    so the process quietly grows past the card into tens of GB of host RAM and pagefile and the
    desktop stops responding with no error anywhere.

    It does NOT second-guess the load. The flat headroom this estimate is built on is a deliberately
    generous PLANNING figure for picking a tier, and that tier already runs the 1024x1024 default on
    cards whose whole budget is under 8 GB (measured: a 8 GB card's safe budget is 5898 MiB against
    a 6963 MiB default-resolution estimate, and those generations complete). So a refusal needs
    BOTH conditions -- over the free budget AND over what the load already budgeted -- which
    confines it to the resolution-driven overrun it is for. The tiled look applies the same rule.

    Fail-open on anything unknown (no free reading, no budget) and on any device class where the
    estimate or the offload story means something different, so a broken probe can never block a
    generation that would have worked.
    """
    override = bool(allow_oversized) or _oversized_generate_override()
    try:
        # Unified / system memory: offload moves bytes within one pool, "free" is a moving target shared with the OS,
        # and the load-time unified refusal already owns that device class.
        if getattr(device_memory, "is_unified", False):
            return ImageActivationVerdict(ACTIVATION_RUN)
        # CUDA / ROCm only. ROCm's torch reports device "cuda", so this covers both. XPU / MPS / CPU keep today's
        # behaviour exactly: their allocators and offload semantics differ and this estimate was measured against a
        # discrete VRAM pool.
        if getattr(device_memory, "device", None) != "cuda":
            return ImageActivationVerdict(ACTIVATION_RUN)
        free = getattr(device_memory, "free_mib", None)
        if free is None:
            return ImageActivationVerdict(ACTIVATION_RUN)
        budget = _safe_device_budget_mib(device_memory)
        if budget is None:
            return ImageActivationVerdict(ACTIVATION_RUN)
        # Calibrated tiers measured only the unconditioned denoise; a ControlNet counts as one output-sized image.
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
        # What the LOAD budgeted: the same estimator at the default resolution, i.e. the exact call _plan_memory makes.
        # Same function and same family hint, so the comparison is between two points on one curve rather than between
        # two different guesses.
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
    # The flat base overhead rides along with the activations: the CUDA context, the scheduler state and the
    # fragmentation allowance all have to coexist with this pass's tensors, and the load-time plan already sums them
    # additively for exactly that reason. Leaving it out made the guard silent by a few hundred MiB on the very card
    # #8188 was reported from (15.92 GiB: 13,872 MiB of activations against a 14,254 MiB budget). It cannot cause a
    # false refusal at or below the default resolution, because the `needed <= planned` arm already exempts every
    # request the load itself budgeted for.
    numbers = dict(needed_mib = int(needed), tiled_needed_mib = tiled, budget_mib = int(budget))
    if int(needed) + overhead <= int(budget) or (not conditioned and needed <= planned):
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
    """Refuse a generation whose activations cannot fit the free device budget; else return the verdict.

    ``ValueError`` on purpose: ``/images/generate`` maps ValueError to HTTP 400 with the message
    as the reason, so this surfaces as an actionable refusal in the UI. RuntimeError there is
    reserved for the two client-state sentinels (not loaded / cancelled) and otherwise becomes an
    opaque 500."""
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
    """Stream transformer blocks and text-encoder leaves without whole-component onloads.

    This is selected only after measuring a component larger than the safe device budget, so a
    model-offload fallback would deterministically OOM. Any setup failure is therefore fatal and
    reports the granular-offload failure directly.
    """
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
        use_stream = onload.type == "cuda" and "low_cpu_mem_usage" in params

        # Keep small companions such as the tiled VAE resident. Every transformer and text encoder remains on CPU behind
        # a granular hook.
        for name, component in components.items():
            if str(name) in streamed:
                continue
            if isinstance(component, torch.nn.Module):
                component.to(onload)

        pinned_mib = [0]
        # Overlap needs a pinned host copy and record_stream (record_stream=False syncs the compute stream per group).
        prefetch = use_stream and _streaming_prefetch_enabled()
        pin_dits, pin_encoders = False, False
        if prefetch:
            pin_dits, pin_encoders = _streamed_pin_plan(
                sum(_module_host_mib(m) for m, t in streamed.values() if t == "block_level"),
                sum(_module_host_mib(m) for m, t in streamed.values() if t != "block_level"),
                logger,
            )
        defer = (
            prefetch
            and bool(getattr(pipe, _BACKGROUND_PIN_REQUEST_ATTR, False))
            and _background_pin_enabled()
        )
        if defer:
            install_group_pin_wait()
        # pinned encoders count against the budget a torchao denoiser pins within, as on the group tier
        if pin_encoders:
            pinned_mib[0] = sum(
                _module_host_mib(m) for m, t in streamed.values() if t != "block_level"
            )

        for module, offload_type in streamed.values():
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
            if use_stream and "low_cpu_mem_usage" in params:
                kwargs["low_cpu_mem_usage"] = not (pin and not defer)
            kwargs = _torchao_group_offload_kwargs(module, kwargs, pinned_mib)
            apply_group_offloading(module, **kwargs)
            if pin and defer and kwargs.get("use_stream") and kwargs.get("low_cpu_mem_usage"):
                _defer_pinning(pipe, module, onload, logger)
            installed += 1
            if (
                use_stream
                and offload_type == "block_level"
                and not _pin_top_level_group(module, logger, pinned_mib)
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

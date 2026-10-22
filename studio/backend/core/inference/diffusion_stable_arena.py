# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stable device addresses for a whole-module CPU-offloaded denoiser (``memory_mode=low_vram``).

accelerate's ``CpuOffload`` hook onloads the denoiser with ``module.to(device)`` at the start of every render, so the
caching allocator hands its weights new addresses each time and a CUDA graph recorded on render 1 cannot replay on
render 2 (``diffusion_cuda_graph.MODEL_OFFLOAD_REASON``). Here the denoiser's device copy lives in one CUDA virtual
address range reserved once for the load (driver VMM: ``cuMemAddressReserve``). An onload creates physical memory,
maps it at that range and copies the kept host weights in; the offload syncs the card, unmaps and releases the
physical memory, so the text encoder and VAE phases get the VRAM back exactly as with the stock hook. The weights
land at the same addresses on every render, so the recorded step graph keeps replaying.

ComfyUI does the same for its graph replays over offloaded weights (comfy_aimdo ``model_vbar``).

Plain dense tensors (with their strides) and torchao-style wrapper weights (int8 / fp8: their plain inner tensors,
rebuilt over the range) are placed; a module holding any other parameter type (GGUF) is left to the stock hook.
Opt out with ``UNSLOTH_DIFFUSION_MODEL_OFFLOAD_ARENA=0``.
"""

from __future__ import annotations

import ctypes
import os
import threading
import weakref
from typing import Any, Optional

ARENA_ENV = "UNSLOTH_DIFFUSION_MODEL_OFFLOAD_ARENA"
_ARENA_ATTR = "_unsloth_stable_arena"
_ALIGN = 512
_OFF = ("0", "off", "false", "no")

# CUDA driver enums
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


_LIB: Any = None
_LIB_LOCK = threading.Lock()


def _driver() -> Any:
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            lib = None
            for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
                try:
                    lib = ctypes.CDLL(name)
                    break
                except OSError:
                    continue
            if lib is None:
                raise OSError("CUDA driver library not found")
            u64, sz = ctypes.c_uint64, ctypes.c_size_t
            for fn, args in (
                (
                    "cuMemGetAllocationGranularity",
                    [ctypes.POINTER(sz), ctypes.POINTER(_AllocProp), ctypes.c_int],
                ),
                ("cuMemAddressReserve", [ctypes.POINTER(u64), sz, sz, u64, ctypes.c_ulonglong]),
                ("cuMemAddressFree", [u64, sz]),
                (
                    "cuMemCreate",
                    [
                        ctypes.POINTER(ctypes.c_ulonglong),
                        sz,
                        ctypes.POINTER(_AllocProp),
                        ctypes.c_ulonglong,
                    ],
                ),
                ("cuMemRelease", [ctypes.c_ulonglong]),
                ("cuMemMap", [u64, sz, sz, ctypes.c_ulonglong, ctypes.c_ulonglong]),
                ("cuMemUnmap", [u64, sz]),
                ("cuMemSetAccess", [u64, sz, ctypes.POINTER(_AccessDesc), sz]),
                ("cuCtxGetCurrent", [ctypes.POINTER(ctypes.c_void_p)]),
            ):
                f = getattr(lib, fn)
                f.argtypes = args
                f.restype = ctypes.c_int
            _LIB = lib
    return _LIB


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed (CUresult {rc})")


def arena_disabled() -> bool:
    return (os.environ.get(ARENA_ENV) or "").strip().lower() in _OFF


class _CudaArray:
    """A ``__cuda_array_interface__`` over raw device bytes, so ``torch.as_tensor`` wraps them without a copy."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (int(nbytes),),
            "typestr": "|u1",
            "data": (int(ptr), False),
            "version": 3,
            "strides": None,
            "stream": None,
        }


class VmmArena:
    """One reserved device VA range of ``nbytes`` (rounded to the allocation granularity) on ``device_index``; physical
    memory exists only between ``map()`` and ``unmap()``."""

    def __init__(self, nbytes: int, device_index: int) -> None:
        import torch

        self.device_index = int(device_index)
        self.device = torch.device("cuda", self.device_index)
        lib = _driver()
        self._ensure_context()
        prop = self._prop()
        gran = ctypes.c_size_t(0)
        _check(
            lib.cuMemGetAllocationGranularity(
                ctypes.byref(gran), ctypes.byref(prop), _CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            ),
            "cuMemGetAllocationGranularity",
        )
        self.granularity = int(gran.value) or (2 << 20)
        self.size = -(-int(nbytes) // self.granularity) * self.granularity
        va = ctypes.c_uint64(0)
        _check(
            lib.cuMemAddressReserve(ctypes.byref(va), self.size, self.granularity, 0, 0),
            "cuMemAddressReserve",
        )
        self.va = int(va.value)
        self._handle: Optional[int] = None
        self._base: Any = None
        self.maps = 0
        self.unmaps = 0
        self.closed = False

    def _prop(self) -> _AllocProp:
        prop = _AllocProp()
        prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device_index
        return prop

    def _ensure_context(self) -> None:
        import torch

        ctx = ctypes.c_void_p(0)
        _driver().cuCtxGetCurrent(ctypes.byref(ctx))
        if not ctx.value:
            # The calling thread has no current context yet: any runtime call on the device makes the primary one current.
            with torch.cuda.device(self.device_index):
                torch.cuda.current_stream().query()

    @property
    def mapped(self) -> bool:
        return self._handle is not None

    def map(self) -> Any:
        """Back the range with new physical memory; returns a uint8 tensor over the whole range."""
        import torch

        if self.closed:
            raise RuntimeError("arena closed")
        if self._handle is not None:
            return self._base
        lib = _driver()
        with torch.cuda.device(self.device_index):
            self._ensure_context()
            prop = self._prop()
            handle = ctypes.c_ulonglong(0)
            rc = lib.cuMemCreate(ctypes.byref(handle), self.size, ctypes.byref(prop), 0)
            if rc != 0:
                # The caching allocator may hold the memory as free cached blocks.
                torch.cuda.empty_cache()
                rc = lib.cuMemCreate(ctypes.byref(handle), self.size, ctypes.byref(prop), 0)
            _check(rc, "cuMemCreate")
            try:
                _check(lib.cuMemMap(self.va, self.size, 0, handle.value, 0), "cuMemMap")
                access = _AccessDesc()
                access.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
                access.location.id = self.device_index
                access.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                rc = lib.cuMemSetAccess(self.va, self.size, ctypes.byref(access), 1)
                if rc != 0:
                    lib.cuMemUnmap(self.va, self.size)
                    _check(rc, "cuMemSetAccess")
            except Exception:
                lib.cuMemRelease(handle.value)
                raise
            self._handle = int(handle.value)
            self._base = torch.as_tensor(_CudaArray(self.va, self.size), device = self.device)
        self.maps += 1
        return self._base

    def view(
        self,
        offset: int,
        shape: tuple,
        dtype: Any,
        stride: Optional[tuple] = None,
    ) -> Any:
        import torch

        base = self.map()
        numel = 1
        for s in shape:
            numel *= int(s)
        itemsize = torch.empty((), dtype = dtype).element_size()
        if stride is None:
            span = numel
        else:
            span = 1 + sum((int(s) - 1) * int(st) for s, st in zip(shape, stride)) if numel else 0
        flat = base[offset : offset + span * itemsize].view(dtype)
        if stride is None:
            return flat.view(shape)
        return flat.as_strided(shape, stride)

    def unmap(self) -> None:
        """Release the physical memory. The caller guarantees no tensor over the range is used again before ``map()``
        (pending work on the card is waited for here)."""
        import torch

        if self._handle is None:
            return
        lib = _driver()
        with torch.cuda.device(self.device_index):
            torch.cuda.synchronize(self.device_index)
            self._base = None
            _check(lib.cuMemUnmap(self.va, self.size), "cuMemUnmap")
            _check(lib.cuMemRelease(self._handle), "cuMemRelease")
        self._handle = None
        self.unmaps += 1

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.unmap()
        finally:
            self.closed = True
            try:
                _driver().cuMemAddressFree(self.va, self.size)
            except Exception:  # noqa: BLE001 - process teardown
                pass


def _plain(t: Any) -> bool:
    import torch
    return type(t) is torch.Tensor or (
        type(t) is torch.nn.Parameter and type(t.data) is torch.Tensor
    )


def _inner(t: Any) -> Optional[list]:
    """A torchao-style wrapper weight's plain inner tensors (``diffusion_memory._wrapper_inner``), else None."""
    from .diffusion_memory import _wrapper_inner
    return _wrapper_inner(t)


def _placeable(t: Any) -> bool:
    return _plain(t) or _inner(t) is not None


def _pieces(name: str, t: Any) -> list:
    """(layout key, tensor) for each storage of ``t``: its own data, or a wrapper weight's inner tensors."""
    if _plain(t):
        return [(name, t.data)]
    return [(f"{name}#{attr}", x) for attr, x in (_inner(t) or ())]


def _set_tensor(owner: Any, leaf: str, kind: str, value: Any) -> None:
    if kind == "p":
        owner._parameters[leaf].data = value
    else:
        owner._buffers[leaf] = value


def _entries(module: Any) -> list:
    """(qualified name, owner module, leaf name, kind, tensor) for every parameter and buffer, each tensor once."""
    out: list = []
    seen: set = set()
    for mod_name, mod in module.named_modules():
        for kind, table in (("p", mod._parameters), ("b", mod._buffers)):
            for leaf, t in table.items():
                if t is None or id(t) in seen:
                    continue
                seen.add(id(t))
                out.append((f"{mod_name}.{leaf}" if mod_name else leaf, mod, leaf, kind, t))
    return out


class ModuleArena:
    """The stable device range of one whole-module offloaded denoiser, installed as the module's own ``to``."""

    def __init__(
        self,
        module: Any,
        device: Any,
        logger: Any = None,
    ) -> None:
        import torch

        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.logger = logger
        self.layout: Optional[dict] = None
        self.arena: Optional[VmmArena] = None
        self.failed: Optional[str] = None
        self.stats = {"onloads": 0, "placed": 0, "stock": 0, "maps": 0, "unmaps": 0}
        self._module = weakref.ref(module)
        self._stock_to = type(module).to
        module.__dict__["to"] = self._to
        weakref.finalize(module, ModuleArena._finalize, self.__dict__)

    @staticmethod
    def _finalize(state: dict) -> None:
        arena = state.get("arena")
        if arena is not None:
            try:
                arena.close()
            except Exception:  # noqa: BLE001 - process teardown
                pass

    def detach(self) -> None:
        module = self._module()
        if module is not None and module.__dict__.get("to") == self._to:
            module.__dict__.pop("to", None)
        if self.arena is not None and module is not None and not self._referenced(module):
            self.arena.close()
            self.arena = None

    def _build(self, entries: list) -> None:
        offset = 0
        layout: dict = {}
        for name, _owner, _leaf, _kind, whole in entries:
            if whole.device.type != "cpu" or not _placeable(whole):
                continue
            for key, t in _pieces(name, whole):
                # strided layouts (channels_last) keep their strides over the span they reach
                span = (
                    (1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))) * t.element_size()
                    if t.numel()
                    else 0
                )
                layout[key] = (offset, tuple(t.shape), tuple(t.stride()), t.dtype, span)
                offset += -(-max(span, 1) // _ALIGN) * _ALIGN
        self.layout = layout
        self.arena = VmmArena(max(offset, _ALIGN), self.device.index)

    def _referenced(self, module: Any) -> bool:
        if self.arena is None or not self.arena.mapped:
            return False
        lo, hi = self.arena.va, self.arena.va + self.arena.size
        for name, _owner, _leaf, _kind, whole in _entries(module):
            try:
                if whole.device.type != "cuda":
                    continue
                pieces = _pieces(name, whole) if _placeable(whole) else [(name, whole)]
                if any(lo <= x.data_ptr() < hi for _, x in pieces):
                    return True
            except Exception:  # noqa: BLE001
                return True
        return False

    @staticmethod
    def _on_host(module: Any) -> bool:
        for t in module.parameters():
            return t.device.type == "cpu"
        return False

    def _onload(self, module: Any, non_blocking: bool) -> None:
        import torch

        entries = _entries(module)
        if self.layout is None:
            self._build(entries)
        assert self.arena is not None and self.layout is not None
        self.arena.map()
        placed = 0
        for name, owner, leaf, kind, whole in entries:
            if whole.device.type != "cpu" or not _placeable(whole):
                continue
            pieces = _pieces(name, whole)
            slots = [self.layout.get(key) for key, _ in pieces]
            if any(
                slot is None
                or (tuple(t.shape), tuple(t.stride()), t.dtype) != (slot[1], slot[2], slot[3])
                for slot, (_, t) in zip(slots, pieces)
            ):
                continue
            views = []
            for slot, (_, t) in zip(slots, pieces):
                view = self.arena.view(slot[0], slot[1], slot[3], slot[2])
                view.copy_(t, non_blocking = non_blocking)
                views.append(view)
            if _plain(whole):
                _set_tensor(owner, leaf, kind, views[0])
            else:
                # a wrapper weight is rebuilt over its placed inner tensors and swapped in, as Module.to moves it
                names, ctx = whole.__tensor_flatten__()
                rebuilt = type(whole).__tensor_unflatten__(
                    dict(zip(names, views)), ctx, whole.shape, whole.stride()
                )
                if kind == "p":
                    torch.utils.swap_tensors(
                        whole, torch.nn.Parameter(rebuilt, requires_grad = whole.requires_grad)
                    )
                else:
                    owner._buffers[leaf] = rebuilt
            placed += 1
        self.stats["placed"] = placed

    def _to(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        module = self._module()
        try:
            device, dtype, non_blocking, fmt = torch._C._nn._parse_to(*args, **kwargs)
        except Exception:  # noqa: BLE001 - let the stock call raise its own error
            device = dtype = fmt = None
            non_blocking = False
        if (
            module is not None
            and self.failed is None
            and dtype is None
            and fmt is None
            and device is not None
            and device.type == "cuda"
            and (device.index if device.index is not None else torch.cuda.current_device())
            == self.device.index
        ):
            try:
                # accelerate calls ``to`` on every forward (its device check compares "cuda" with "cuda:0"): only a
                # module whose weights are on the host is placed
                if self._on_host(module):
                    self._onload(module, True)
                    self.stats["onloads"] += 1
            except Exception as exc:  # noqa: BLE001 - the stock move still places the module
                self.failed = f"{type(exc).__name__}: {exc}"
                if self.logger is not None:
                    self.logger.warning(
                        "diffusion.stable_arena: onload into the stable range failed (%s)", exc
                    )
                self._release(module)
        out = self._stock_to(module, *args, **kwargs)
        if module is not None:
            self._release(module)
        return out

    def _release(self, module: Any) -> None:
        """Unmap the range once no tensor of the module lives in it (its weights went back to the host)."""
        if self.arena is None or not self.arena.mapped or self._referenced(module):
            return
        self.arena.unmap()
        self.stats["unmaps"] += 1

    def describe(self) -> dict:
        arena = self.arena
        return {
            **self.stats,
            "mib": None if arena is None else arena.size >> 20,
            "mapped": bool(arena is not None and arena.mapped),
            "failed": self.failed,
        }


def attach(
    module: Any,
    device: Any = None,
    logger: Any = None,
) -> tuple:
    """``(arena, None)`` when ``module``'s onloads will land in one stable range from now on, else ``(None, why)``."""
    if arena_disabled():
        return None, f"disabled by {ARENA_ENV}=0"
    existing = module.__dict__.get(_ARENA_ATTR)
    if existing is not None:
        return (existing, None) if existing.failed is None else (None, existing.failed)
    try:
        import torch
        if not torch.cuda.is_available():
            return None, "no CUDA device"
        _driver()
    except Exception as exc:  # noqa: BLE001
        return None, f"CUDA virtual memory unavailable ({exc})"
    odd = [name for name, _o, _l, _k, t in _entries(module) if not _placeable(t)]
    if odd:
        return None, f"{len(odd)} parameter(s) are not plain tensors ({odd[0]})"
    if device is None:
        hook = getattr(module, "_hf_hook", None)
        device = getattr(hook, "execution_device", None)
    if device is None:
        return None, "no execution device"
    try:
        arena = ModuleArena(module, device, logger)
    except Exception as exc:  # noqa: BLE001
        return None, f"stable range unavailable ({exc})"
    module.__dict__[_ARENA_ATTR] = arena
    return arena, None


def arena_of(module: Any) -> Optional[ModuleArena]:
    try:
        return module.__dict__.get(_ARENA_ATTR)
    except Exception:  # noqa: BLE001
        return None

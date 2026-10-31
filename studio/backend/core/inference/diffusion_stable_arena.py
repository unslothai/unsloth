# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keeps offloaded denoiser weights at fixed VMM addresses so recorded CUDA graphs replay; GGUF or
other parameter types left to the stock hook"""

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


# Physical bytes mapped through every range, per CUDA ordinal: memory the caching allocator's counters do not see.
_MAPPED: dict = {}
# High-water mark of ``_MAPPED`` since the last ``reset_peak_mapped`` (the stable ranges' max_memory_allocated).
_PEAK_MAPPED: dict = {}
_MAPPED_LOCK = threading.Lock()


def mapped_bytes(device_index: Optional[int]) -> int:
    """Bytes currently mapped in stable ranges on ``device_index`` (outside ``torch.cuda.memory_allocated``)."""
    with _MAPPED_LOCK:
        return int(_MAPPED.get(device_index, 0))


def peak_mapped_bytes(device_index: Optional[int]) -> int:
    """Most bytes mapped at once on ``device_index`` since the last ``reset_peak_mapped``."""
    with _MAPPED_LOCK:
        return int(max(_PEAK_MAPPED.get(device_index, 0), _MAPPED.get(device_index, 0)))


def reset_peak_mapped(device_index: Optional[int]) -> None:
    with _MAPPED_LOCK:
        _PEAK_MAPPED[device_index] = _MAPPED.get(device_index, 0)


def _count_mapped(device_index: int, delta: int) -> None:
    with _MAPPED_LOCK:
        now = _MAPPED.get(device_index, 0) + int(delta)
        _MAPPED[device_index] = now
        _PEAK_MAPPED[device_index] = max(_PEAK_MAPPED.get(device_index, 0), now)


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
            try:
                base = torch.as_tensor(_CudaArray(self.va, self.size), device = self.device)
            except Exception:
                # nothing may keep the mapping a failed map() leaves no tensor over: unmap and release it, so the
                # arena is unmapped and retryable and the caller's stock fallback holds the only copy
                lib.cuMemUnmap(self.va, self.size)
                lib.cuMemRelease(handle.value)
                raise
            self._handle = int(handle.value)
            self._base = base
        _count_mapped(self.device_index, self.size)
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
            from .diffusion_cuda_graph import synchronize_outside_capture
            synchronize_outside_capture(self.device_index)
            self._release_synced(lib)

    def _release_synced(self, lib: Any) -> None:
        """Unmap and release, the card already synchronized with no capture recording."""
        self._base = None
        _check(lib.cuMemUnmap(self.va, self.size), "cuMemUnmap")
        _check(lib.cuMemRelease(self._handle), "cuMemRelease")
        self._handle = None
        _count_mapped(self.device_index, -self.size)
        self.unmaps += 1

    def _close_idle(self) -> None:
        """``close`` for ``release_when_capture_idle``: run under the capture lock with no capture recording."""
        import torch

        if self.closed:
            return
        try:
            if self._handle is not None:
                with torch.cuda.device(self.device_index):
                    torch.cuda.synchronize(self.device_index)
                    self._release_synced(_driver())
        finally:
            self.closed = True
            try:
                _driver().cuMemAddressFree(self.va, self.size)
            except Exception:  # noqa: BLE001 - process teardown
                pass

    def close_from_finalizer(self) -> None:
        """``close`` for weakref finalizers: never blocks on the capture gate (the finalizer's thread may hold its
        lock). Runs now when no capture records and the lock is free, else at the next gated point; the memory is
        freed only after a device sync with no capture recording, as ``unmap`` does."""
        if self.closed:
            return
        from .diffusion_cuda_graph import release_when_capture_idle

        release_when_capture_idle(self._close_idle)

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
    """(qualified name, registrations, kind, tensor) for every parameter and buffer, each tensor once.

    ``registrations`` lists every (owner module, leaf name, qualified name) the tensor is registered under: a buffer
    shared by several submodules (or names) has one registry slot per registration, and each must point at the placed
    copy."""
    out: list = []
    index: dict = {}
    for mod_name, mod in module.named_modules(remove_duplicate = False):
        for kind, table in (("p", mod._parameters), ("b", mod._buffers)):
            for leaf, t in table.items():
                if t is None:
                    continue
                qual = f"{mod_name}.{leaf}" if mod_name else leaf
                at = index.get(id(t))
                if at is not None:
                    regs = out[at][1]
                    if not any(o is mod and l == leaf for o, l, _q in regs):
                        regs.append((mod, leaf, qual))
                    continue
                index[id(t)] = len(out)
                out.append((qual, [(mod, leaf, qual)], kind, t))
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
        self.retiring = False
        self.stats = {"onloads": 0, "placed": 0, "stock": 0, "maps": 0, "unmaps": 0, "rebuilds": 0}
        self._module = weakref.ref(module)
        self._stock_to = type(module).to
        module.__dict__["to"] = self._to
        # accelerate's remove_hook_from_module pops ``to`` from the instance (its _accelerate_added_attributes) and
        # diffusers re-hooks before every render: put it back before the hook's pre-forward onloads the module.
        self._reinstall_handle = module.register_forward_pre_hook(self._reinstall)
        weakref.finalize(module, ModuleArena._finalize, self.__dict__)

    def _reinstall(self, module: Any, _args: Any) -> None:
        if self.retiring:
            self.retire()
            return
        if module.__dict__.get("to") != self._to and self.failed is None:
            module.__dict__["to"] = self._to
            self.stats["reinstalls"] = self.stats.get("reinstalls", 0) + 1

    @staticmethod
    def _finalize(state: dict) -> None:
        arena = state.get("arena")
        if arena is not None:
            try:
                arena.close_from_finalizer()
            except Exception:  # noqa: BLE001 - process teardown
                pass

    def detach(self) -> None:
        """Give the module its stock onload back and close the range. Detached while the weights still live in the
        range (graphing stopped mid-render), it stops placing at once and finishes on the offload that moves them
        out, so the range is never left mapped."""
        module = self._module()
        if self.arena is not None and module is not None and self._referenced(module):
            self.failed = self.failed or "detached"
            self.retiring = True
            return
        self.retiring = False
        if module is not None and module.__dict__.get("to") == self._to:
            module.__dict__.pop("to", None)
        try:
            self._reinstall_handle.remove()
        except Exception:  # noqa: BLE001
            pass
        if module is not None and module.__dict__.get(_ARENA_ATTR) is self:
            module.__dict__.pop(_ARENA_ATTR, None)
        if self.arena is not None and module is not None and not self._referenced(module):
            self.arena.close()
            self.arena = None

    def retire(self) -> None:
        """Restore the stock onload; detach waits for the offload that takes mapped weights home."""
        self.retiring = True
        module = self._module()
        if (
            module is None
            or self.arena is None
            or not self.arena.mapped
            or not self._referenced(module)
        ):
            if self.arena is not None and self.arena.mapped and module is not None:
                self._release(module)
            self.detach()

    def _build(self, entries: list) -> None:
        offset = 0
        layout: dict = {}
        for name, regs, _kind, whole in entries:
            if whole.device.type != "cpu" or not _placeable(whole):
                continue
            for key, t in _pieces(name, whole):
                # strided layouts (channels_last) keep their strides over the span they reach
                span = (
                    (1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))) * t.element_size()
                    if t.numel()
                    else 0
                )
                slot = (offset, tuple(t.shape), tuple(t.stride()), t.dtype, span)
                # Every registration name of a shared tensor maps to its one slot: Module.to() moves each
                # registration on its own, so after an offload they come back as separate (equal) host copies.
                suffix = key[len(name) :]
                for _owner, _leaf, qual in regs:
                    layout[qual + suffix] = slot
                offset += -(-max(span, 1) // _ALIGN) * _ALIGN
        self.layout = layout
        self.arena = VmmArena(max(offset, _ALIGN), self.device.index)

    def _stale(self, entries: list) -> bool:
        """Whether a placeable host weight has no slot of its shape, stride and dtype in the layout, or a slot outlived
        its weight (an unloaded adapter): a removed piece must not keep its share of the range mapped."""
        assert self.layout is not None
        seen: set = set()
        for name, regs, _kind, whole in entries:
            if whole.device.type != "cpu" or not _placeable(whole):
                continue
            for key, t in _pieces(name, whole):
                slot = self.layout.get(key)
                if slot is None or (tuple(t.shape), tuple(t.stride()), t.dtype) != (
                    slot[1],
                    slot[2],
                    slot[3],
                ):
                    return True
                suffix = key[len(name) :]
                seen.update(qual + suffix for _owner, _leaf, qual in regs)
        return not seen.issuperset(self.layout)

    def _referenced(self, module: Any) -> bool:
        if self.arena is None or not self.arena.mapped:
            return False
        lo, hi = self.arena.va, self.arena.va + self.arena.size
        for name, _regs, _kind, whole in _entries(module):
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
        if self.layout is not None and self._stale(entries):
            # Host weights changed (adapter, dtype, shape): re-lay out, else the stale range stays mapped beside them.
            if self.arena is not None:
                self.arena.close()
            self.arena = None
            self.layout = None
            self.stats["rebuilds"] += 1
        if self.layout is None:
            self._build(entries)
        assert self.arena is not None and self.layout is not None
        self.arena.map()
        placed = 0
        filled: dict = {}  # slot offset -> (host tensor copied into it during this onload, its view)
        for name, regs, kind, whole in entries:
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
            # A slot already filled by another registration of a once-shared buffer is reused only when the copies
            # still hold the same values; otherwise this one keeps the stock move.
            if any(
                slot[0] in filled and not torch.equal(filled[slot[0]][0], t)
                for slot, (_, t) in zip(slots, pieces)
            ):
                continue
            views = []
            for slot, (_, t) in zip(slots, pieces):
                if slot[0] in filled:
                    # the same view object: the registrations share one tensor again, as on the first onload, so the
                    # graph's placement fingerprint (one entry per tensor) does not change
                    views.append(filled[slot[0]][1])
                    continue
                view = self.arena.view(slot[0], slot[1], slot[3], slot[2])
                view.copy_(t, non_blocking = non_blocking)
                filled[slot[0]] = (t, view)
                views.append(view)
            if _plain(whole):
                for owner, leaf, _qual in regs:
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
                    for owner, leaf, _qual in regs:
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
            and not self.retiring
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
            if self.retiring and not (self.arena is not None and self.arena.mapped):
                self.detach()
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
    odd = [name for name, _r, _k, t in _entries(module) if not _placeable(t)]
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


# ---------------------------------------------------------------------------------------------------------------------
# Block offload: stable addresses for the resident groups a prompt encode streams and pins again.
#
# The stock re-pin after an encode release lands at new allocator addresses, re-recording the step graph every new
# prompt. A reserved range per group unmaps on release and remaps at the same addresses, so the step keeps replaying.
# Opt out with UNSLOTH_DIFFUSION_GROUP_REPIN_RANGE=0.

REPIN_ENV = "UNSLOTH_DIFFUSION_GROUP_REPIN_RANGE"
_REPIN_ATTR = "_unsloth_repin_range"
_REPIN_WANTED = "_unsloth_repin_wanted"
_REPIN_STATS = "_unsloth_repin_stats"


def repin_disabled() -> bool:
    return (os.environ.get(REPIN_ENV) or "").strip().lower() in _OFF


def want_stable_repin(module: Any) -> Optional[str]:
    """Mark ``module`` (a block-offloaded denoiser whose step records a graph) so its group re-pins land in stable
    ranges from now on. Returns why not, else None."""
    if repin_disabled():
        return f"disabled by {REPIN_ENV}=0"
    try:
        import torch
        if not torch.cuda.is_available():
            return "no CUDA device"
        _driver()
    except Exception as exc:  # noqa: BLE001
        return f"CUDA virtual memory unavailable ({exc})"
    try:
        module.__dict__[_REPIN_WANTED] = True
        module.__dict__.setdefault(
            _REPIN_STATS,
            {
                "groups": 0,
                "pins": 0,
                "placed": 0,
                "stock": 0,
                "maps": 0,
                "unmaps": 0,
                "bytes": 0,
                "rebuilds": 0,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return f"module not writable ({exc})"
    return None


def drop_stable_repin(module: Any) -> None:
    """Graphing stopped for ``module``: later re-pins move the stock way. A mapped range is unmapped by the next
    release as before; the address ranges themselves go with the module."""
    try:
        module.__dict__.pop(_REPIN_WANTED, None)
    except Exception:  # noqa: BLE001
        pass


def stable_repin_wanted(module: Any) -> bool:
    try:
        return bool(module.__dict__.get(_REPIN_WANTED)) and not repin_disabled()
    except Exception:  # noqa: BLE001
        return False


def repin_stats(module: Any) -> Optional[dict]:
    try:
        stats = module.__dict__.get(_REPIN_STATS)
    except Exception:  # noqa: BLE001
        return None
    return dict(stats) if isinstance(stats, dict) else None


def _host_pieces(t: Any) -> Optional[list]:
    """(attr or None, plain tensor) for each storage of a host copy, else None (a type the range cannot hold)."""
    if _plain(t):
        return [(None, t.data)]
    inner = _inner(t)
    return None if inner is None else list(inner)


def _span(t: Any) -> int:
    if not t.numel():
        return 0
    return (1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))) * t.element_size()


class GroupRange:
    """The reserved device range of one offload group's resident copy: ``layout[i]`` holds the (offset, shape, stride,
    dtype) of each storage of the group's i-th tensor, fixed at the first re-pin."""

    def __init__(self, layout: dict, nbytes: int, device_index: int) -> None:
        self.layout = layout
        self.arena = VmmArena(max(nbytes, _ALIGN), device_index)

    def referenced(self, tensors: list) -> bool:
        if not self.arena.mapped:
            return False
        lo, hi = self.arena.va, self.arena.va + self.arena.size
        for t in tensors:
            try:
                pieces = _host_pieces(t) or [(None, t)]
                if any(x.device.type == "cuda" and lo <= x.data_ptr() < hi for _, x in pieces):
                    return True
            except Exception:  # noqa: BLE001 - unreadable: keep the memory
                return True
        return False


def _close_ranges(ranges: list) -> None:
    for rng in ranges:
        try:
            rng.arena.close_from_finalizer()
        except Exception:  # noqa: BLE001 - process teardown
            pass


def _range_for(owner: Any, group: Any, sources: list, device: Any) -> Optional[GroupRange]:
    """``group``'s range, built on first use from the host copies ``sources`` (index -> tensor) it places."""
    offset = 0
    layout: dict = {}
    for i, src in sources:
        pieces = _host_pieces(src)
        if pieces is None or any(x.device.type != "cpu" for _, x in pieces):
            continue
        slots = []
        for _attr, x in pieces:
            slots.append((offset, tuple(x.shape), tuple(x.stride()), x.dtype))
            offset += -(-max(_span(x), 1) // _ALIGN) * _ALIGN
        layout[i] = tuple(slots)
    held = owner.__dict__.get("_unsloth_repin_ranges")
    stats = owner.__dict__.get(_REPIN_STATS)
    rng = getattr(group, "__dict__", {}).get(_REPIN_ATTR)
    if rng is not None:
        if rng.layout == layout or rng.arena.mapped:
            # unchanged, or still holding weights (a release that could not move them all out): reused as is
            return rng
        # Host tensors changed (LoRA, dtype, shape): re-lay out, else the stale range stays mapped beside them.
        rng.arena.close()
        group.__dict__.pop(_REPIN_ATTR, None)
        if held is not None and rng in held:
            held.remove(rng)
        if isinstance(stats, dict):
            stats["rebuilds"] = stats.get("rebuilds", 0) + 1
            stats["groups"] -= 1
            stats["bytes"] -= rng.arena.size
    if not layout:
        return None
    rng = GroupRange(layout, offset, device.index)
    group.__dict__[_REPIN_ATTR] = rng
    # The ranges outlive any one group object only as long as the denoiser does.
    if held is None:
        held = []
        owner.__dict__["_unsloth_repin_ranges"] = held
        weakref.finalize(owner, _close_ranges, held)
    held.append(rng)
    if isinstance(stats, dict):
        stats["groups"] += 1
        stats["bytes"] += rng.arena.size
    return rng


def pin_group_in_range(
    owner: Any,
    group: Any,
    tensors: list,
    pending: Any,
    host_of: Any,
    device: Any,
    swap_wrapper: Any,
) -> set:
    """Place ``group``'s ``tensors`` for which ``pending`` holds (not on the card yet) into its stable range, copied
    from ``host_of(t)`` (queued on the current stream: the caller synchronizes). The range is laid out over the
    tensors pending at the first re-pin (a tied tensor another group placed takes no room here); a slot is keyed on
    the tensor's position in ``tensors``. Returns the ids placed; the caller moves the rest the stock way."""
    import torch

    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    todo = [(i, t, host_of(t)) for i, t in enumerate(tensors) if pending(t)]
    rng = _range_for(owner, group, [(i, src) for i, _t, src in todo], device)
    if rng is None:
        return set()
    rng.arena.map()
    stats = owner.__dict__.get(_REPIN_STATS)
    placed: set = set()
    undo: list = []  # how to point each tensor installed so far back at its original storage
    try:
        for i, t, src in todo:
            slots = rng.layout.get(i)
            pieces = _host_pieces(src)
            if (
                slots is None
                or pieces is None
                or len(slots) != len(pieces)
                or any(
                    (tuple(x.shape), tuple(x.stride()), x.dtype) != slot[1:]
                    for slot, (_attr, x) in zip(slots, pieces)
                )
            ):
                continue
            views = []
            for slot, (_attr, x) in zip(slots, pieces):
                view = rng.arena.view(slot[0], slot[1], slot[3], slot[2])
                view.copy_(x, non_blocking = True)
                views.append(view)
            if pieces[0][0] is None:
                prev = t.data
                t.data = views[0]
                undo.append((t, prev, False))
            else:
                _names, ctx = src.__tensor_flatten__()
                rebuilt = type(src).__tensor_unflatten__(
                    dict(zip([a for a, _ in pieces], views)), ctx, src.shape, src.stride()
                )
                swap_wrapper(t, rebuilt)  # a swap: ``rebuilt`` now holds t's original contents
                undo.append((t, rebuilt, True))
            placed.add(id(t))
    except BaseException:
        # Leave the range empty so the caller unmaps it and moves the whole group the stock way.
        for t, prev, wrapped in reversed(undo):
            if wrapped:
                swap_wrapper(t, prev)
            else:
                t.data = prev
        raise
    if isinstance(stats, dict):
        stats["pins"] += 1
        stats["maps"] = sum(r.arena.maps for r in owner.__dict__.get("_unsloth_repin_ranges", ()))
        stats["placed"] += len(placed)
        stats["stock"] += len(todo) - len(placed)
    return placed


def release_group_range(owner: Any, group: Any, tensors: list) -> bool:
    """After ``group`` streamed again (its tensors back on their host copies): unmap its range. False when a tensor
    still lives in it (the memory then stays until the next re-pin reuses it)."""
    rng = getattr(group, "__dict__", {}).get(_REPIN_ATTR)
    if rng is None or not rng.arena.mapped:
        return True
    if rng.referenced(tensors):
        return False
    rng.arena.unmap()
    stats = owner.__dict__.get(_REPIN_STATS) if owner is not None else None
    if isinstance(stats, dict):
        stats["unmaps"] += 1
    return True

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Decode KV buffers on CUDA virtual memory (cuMem*), so training can take their physical memory back while the
captured decode graphs keep their addresses: sleep() unmaps and frees the pages but keeps the virtual range,
wake() maps fresh pages at the same addresses (zeroed), and the graphs replay without a re-capture.

The idea is vLLM's sleep mode (vllm/device_allocator/cumem.py). Only the KV cache sleeps here; the graphs'
private pool (small decode intermediates) stays resident. Driver calls go through ctypes on libcuda, inside the
device's primary context, so no compiled extension, MemPool or pluggable allocator is needed (any torch).
UNSLOTH_DECODE_GRAPHS_VMM=0 turns it off (plain torch.zeros buffers, as before)."""

import ctypes
import os
import sys
import threading

import torch

__all__ = ["available", "zeros", "buffer_of", "sleep", "wake"]

_LOCK = threading.Lock()
_CUDA = None
_SUPPORTED = {}
SLEEPS = 0
WAKES = 0


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _Prop(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


_PINNED, _DEVICE, _READWRITE, _VMM_ATTRIBUTE = 1, 1, 3, 102


def _lib():
    global _CUDA
    if _CUDA is None:
        _CUDA = False
        if sys.platform.startswith("linux") and torch.version.hip is None:
            try:
                lib = ctypes.CDLL("libcuda.so.1")
                u64, size = ctypes.c_ulonglong, ctypes.c_size_t
                for name, args in (
                    ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
                    ("cuDeviceGetAttribute", [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]),
                    ("cuDevicePrimaryCtxRetain", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]),
                    ("cuCtxPushCurrent_v2", [ctypes.c_void_p]),
                    ("cuCtxPopCurrent_v2", [ctypes.POINTER(ctypes.c_void_p)]),
                    ("cuMemGetAllocationGranularity", [ctypes.POINTER(size), ctypes.POINTER(_Prop), ctypes.c_int]),
                    ("cuMemAddressReserve", [ctypes.POINTER(u64), size, size, u64, u64]),
                    ("cuMemAddressFree", [u64, size]),
                    ("cuMemCreate", [ctypes.POINTER(u64), size, ctypes.POINTER(_Prop), u64]),
                    ("cuMemRelease", [u64]),
                    ("cuMemMap", [u64, size, size, u64, u64]),
                    ("cuMemUnmap", [u64, size]),
                    ("cuMemSetAccess", [u64, size, ctypes.POINTER(_Access), size]),
                ):  # fmt: skip
                    fn = getattr(lib, name)
                    fn.argtypes, fn.restype = args, ctypes.c_int
                _CUDA = lib
            except Exception:
                _CUDA = False
    return _CUDA


_CUDA_ERROR_OUT_OF_MEMORY = 2


def _check(result, what):
    if result == _CUDA_ERROR_OUT_OF_MEMORY:
        # An OutOfMemoryError reaches generate_with_fallback's recovery path instead of aborting the rollout.
        raise torch.cuda.OutOfMemoryError(f"Unsloth: {what} is out of memory (CUresult {result})")
    if result != 0:
        raise RuntimeError(f"Unsloth: {what} failed (CUresult {result})")


class _Context:
    """The device's primary context (the one the CUDA runtime and torch use) for the driver calls."""

    def __init__(self, index):
        self.index = index

    def __enter__(self):
        lib = _lib()
        ctx = ctypes.c_void_p()
        _check(
            lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self.index), "cuDevicePrimaryCtxRetain"
        )
        _check(lib.cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent")
        return self

    def __exit__(self, *exc):
        _lib().cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
        return False


def _prop(index):
    prop = _Prop()
    prop.type, prop.location.type, prop.location.id = _PINNED, _DEVICE, index
    return prop


def available(device):
    """VMM decode buffers on this CUDA device (Linux NVIDIA, driver support, not turned off)."""
    if os.environ.get("UNSLOTH_DECODE_GRAPHS_VMM", "1") == "0":
        return False
    if getattr(device, "type", None) != "cuda" or not _lib():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    ok = _SUPPORTED.get(index)
    if ok is None:
        try:
            flag = ctypes.c_int()
            dev = ctypes.c_int()
            _check(_lib().cuDeviceGet(ctypes.byref(dev), index), "cuDeviceGet")
            _check(
                _lib().cuDeviceGetAttribute(ctypes.byref(flag), _VMM_ATTRIBUTE, dev.value),
                "attribute",
            )
            ok = bool(flag.value)
        except Exception:
            ok = False
        _SUPPORTED[index] = ok
    return ok


class _Interface:
    # The tensor's storage holds this object, and so the range: it is freed only with the last view.
    def __init__(self, buffer):
        self.buffer = buffer
        self.__cuda_array_interface__ = {
            "shape": (buffer.size,),
            "typestr": "|u1",
            "data": (buffer.ptr, False),
            "version": 2,
        }


class VMMBuffer:
    """One virtual range with its pages mapped (awake) or not (asleep). The tensor views the range."""

    def __init__(self, nbytes, index):
        lib = _lib()
        self.index = index
        prop = _prop(index)
        granularity = ctypes.c_size_t()
        with _Context(index):
            _check(
                lib.cuMemGetAllocationGranularity(ctypes.byref(granularity), ctypes.byref(prop), 0),
                "granularity",
            )
            g = granularity.value
            self.size = -(-max(nbytes, 1) // g) * g
            ptr = ctypes.c_ulonglong()
            _check(
                lib.cuMemAddressReserve(ctypes.byref(ptr), self.size, 0, 0, 0),
                "cuMemAddressReserve",
            )
        self.ptr = ptr.value
        self.handle = None
        self.baseline = None
        self._map()

    def _map(self):
        lib = _lib()
        handle = ctypes.c_ulonglong()
        prop = _prop(self.index)
        access = _Access()
        access.location.type, access.location.id, access.flags = _DEVICE, self.index, _READWRITE
        with _Context(self.index):
            result = lib.cuMemCreate(ctypes.byref(handle), self.size, ctypes.byref(prop), 0)
            if result == _CUDA_ERROR_OUT_OF_MEMORY:
                # Training's freed blocks sit in torch's caching allocator, which cuMemCreate cannot reuse.
                torch.cuda.synchronize(self.index)
                torch.cuda.empty_cache()
                result = lib.cuMemCreate(ctypes.byref(handle), self.size, ctypes.byref(prop), 0)
            _check(result, "cuMemCreate")
            try:
                _check(lib.cuMemMap(self.ptr, self.size, 0, handle.value, 0), "cuMemMap")
                try:
                    _check(
                        lib.cuMemSetAccess(self.ptr, self.size, ctypes.byref(access), 1),
                        "cuMemSetAccess",
                    )
                except Exception:
                    lib.cuMemUnmap(self.ptr, self.size)
                    raise
            except Exception:
                lib.cuMemRelease(handle.value)
                raise
        self.handle = handle.value

    @property
    def awake(self):
        return self.handle is not None

    def sleep(self):
        if self.handle is None:
            return
        # Every queued kernel on the range must be done before its pages go.
        torch.cuda.synchronize(self.index)
        lib = _lib()
        with _Context(self.index):
            _check(lib.cuMemUnmap(self.ptr, self.size), "cuMemUnmap")
            _check(lib.cuMemRelease(self.handle), "cuMemRelease")
        self.handle = None

    def wake(self):
        if self.handle is None:
            self._map()

    def __del__(self):
        try:
            lib = _lib()
            if not lib or self.ptr is None:
                return
            torch.cuda.synchronize(self.index)
            with _Context(self.index):
                if self.handle is not None:
                    lib.cuMemUnmap(self.ptr, self.size)
                    lib.cuMemRelease(self.handle)
                lib.cuMemAddressFree(self.ptr, self.size)
            self.ptr = None
        except Exception:
            pass


def zeros(shape, dtype, device):
    """A zeroed tensor on a VMM range (or None when VMM is unavailable / fails: the caller allocates as before)."""
    if not available(device):
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    numel = 1
    for s in shape:
        numel *= int(s)
    nbytes = numel * torch.empty((), dtype = dtype).element_size()
    try:
        with _LOCK:
            buffer = VMMBuffer(nbytes, index)
        raw = torch.as_tensor(_Interface(buffer), device = torch.device("cuda", index))
    except Exception:
        return None
    tensor = raw[:nbytes].view(dtype).view(shape)
    del raw
    tensor._unsloth_vmm = buffer
    # Tensors on the range right now (this one and its view chain): any more at sleep() means a caller still
    # holds a view (a returned cache), which must keep its pages.
    buffer.baseline = _use_count(tensor)
    tensor.zero_()
    return tensor


def _use_count(tensor):
    count = getattr(torch._C, "_storage_Use_Count", None)
    if count is None:
        return None
    return count(tensor.untyped_storage()._cdata)


def buffer_of(tensor):
    return getattr(tensor, "_unsloth_vmm", None) if isinstance(tensor, torch.Tensor) else None


def in_use(tensor):
    """Another tensor views this buffer (or it cannot be told): it must not sleep."""
    buffer = buffer_of(tensor)
    count = _use_count(tensor)
    return buffer is not None and (
        count is None or buffer.baseline is None or count > buffer.baseline
    )


def sleep(tensor):
    """Frees the pages (addresses stay); do not touch before wake(). Check in_use() first."""
    global SLEEPS
    buffer = buffer_of(tensor)
    if buffer is None or not buffer.awake:
        return 0
    buffer.sleep()
    SLEEPS += 1
    return buffer.size


def wake(tensor):
    """Maps fresh pages at the same addresses and zeroes them; True if it was asleep."""
    global WAKES
    buffer = buffer_of(tensor)
    if buffer is None or buffer.awake:
        return False
    buffer.wake()
    tensor.zero_()
    WAKES += 1
    return True

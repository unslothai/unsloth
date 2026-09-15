#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Probe: what does every layer report as this GPU's memory, and how big a single
allocation does the HIP runtime actually allow?

Five independent readings of the same machine, so a number can be attributed to
the layer that produced it rather than to "the GPU":

  host         the OS: physical RAM, the per-adapter VRAM the registry records
               (Windows `HardwareInformation.qwMemorySize`, which is 64-bit;
               WMI `AdapterRAM` is uint32 and saturates at 4 GiB, so it is
               collected only as the trap it is), page file.
  hip          amdhip64_7.dll: hipMemGetInfo, hipDeviceTotalMem, runtime and
               driver versions, and the module path Windows actually loaded.
  vulkan_raw   vulkan-1.dll: every memory heap of every physical device, its
               flags, and its VK_EXT_memory_budget budget/usage when supported.
  ggml_vulkan  ggml_backend_vk_get_device_memory, i.e. the number llama.cpp and
               Unsloth Studio see. Run in a child process: a Vulkan instance in
               the reporting process would outlive the reading.
  fit          llama-server --list-devices and llama-fit-params, i.e. what the
               loader believes it may place.

  alloc_cap    the largest single hipMalloc that also memsets and reads back,
               found by bisection with one fresh process per candidate. This is
               the only reading that measures the allocator rather than asking
               it, which matters because hipMemGetInfo cannot express a cap that
               is a constant rather than a measurement (ROCm/rocm-systems#10974
               clamps maxAllocSize_ at 64 GiB on stock PAL).

Standard library only: the Windows runners have the system Python and no venv.
Every section is independent, and a section that fails records why instead of
reporting a plausible zero.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024**3
IS_WIN = os.name == "nt"

# hipMemcpyKind; stable across every HIP release.
HIP_MEMCPY_DEVICE_TO_HOST = 2


# --------------------------------------------------------------------------- utils


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_version(path: Path) -> str:
    """PE FileVersion, so a swapped DLL is identified by more than its size."""
    if not IS_WIN:
        return ""
    try:
        size = ctypes.windll.version.GetFileVersionInfoSizeW(str(path), None)
        if not size:
            return ""
        buf = ctypes.create_string_buffer(size)
        ctypes.windll.version.GetFileVersionInfoW(str(path), 0, size, buf)
        block = ctypes.c_void_p()
        length = ctypes.c_uint()
        if not ctypes.windll.version.VerQueryValueW(
            buf, "\\", ctypes.byref(block), ctypes.byref(length)
        ):
            return ""
        ffi = ctypes.cast(block, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
        ms, ls = ffi[2], ffi[3]
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}>"


def describe_file(path: Path) -> dict:
    try:
        st = path.stat()
        return {
            "path": str(path),
            "bytes": st.st_size,
            "sha256": sha256(path),
            "file_version": file_version(path),
        }
    except Exception as e:  # noqa: BLE001
        return {"path": str(path), "error": f"{type(e).__name__}: {e}"}


def loaded_module_path(handle: int) -> str:
    """Where Windows actually found the DLL. Putting a file next to the exe is a
    request, not a guarantee: the loader may already hold another copy."""
    if not IS_WIN:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetModuleFileNameW(ctypes.c_void_p(handle), buf, 32768)
        return buf.value if n else ""
    except Exception:  # noqa: BLE001
        return ""


def find_lib(directory: Path, stem: str) -> Path | None:
    exact = directory / stem
    if exact.is_file():
        return exact
    try:
        for entry in sorted(os.listdir(directory)):
            if entry.startswith(stem + "."):
                return directory / entry
    except OSError:
        pass
    return None


def run(
    cmd: list[str],
    timeout: int = 300,
    env: dict | None = None,
) -> dict:
    try:
        r = subprocess.run(
            cmd,
            capture_output = True,
            text = True,
            timeout = timeout,
            encoding = "utf-8",
            errors = "replace",
            env = env,
        )
        return {
            "cmd": cmd,
            "rc": r.returncode,
            "stdout": r.stdout[-20000:],
            "stderr": r.stderr[-8000:],
        }
    except Exception as e:  # noqa: BLE001
        return {"cmd": cmd, "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- host


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def read_host() -> dict:
    out: dict = {"platform": platform.platform(), "python": sys.version.split()[0]}
    if IS_WIN:
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        out["total_phys_bytes"] = int(st.ullTotalPhys)
        out["avail_phys_bytes"] = int(st.ullAvailPhys)
        out["total_pagefile_bytes"] = int(st.ullTotalPageFile)
        out["adapters"] = read_windows_adapters()
        # AdapterRAM is uint32 and cannot express more than 4 GiB. Recorded to
        # show the trap, never used as a capacity.
        out["wmi_adapter_ram_trap"] = run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
            ],
            120,
        )
    else:
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding = "utf-8")
            for line in meminfo.splitlines():
                if line.startswith(("MemTotal", "MemAvailable")):
                    k, v = line.split(":", 1)
                    out[k.lower() + "_bytes"] = int(v.split()[0]) * 1024
        except OSError as e:
            out["meminfo_error"] = str(e)
        out["rocminfo"] = run(["rocminfo"], 120)
    return out


def read_windows_adapters() -> list:
    """Per-adapter VRAM from the driver's registry key. 64-bit, and readable
    without administrator rights, which is what makes it usable here."""
    import winreg  # noqa: PLC0415  (Windows-only)

    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    found = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError as e:
        return [{"error": f"{type(e).__name__}: {e}"}]
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(key, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        row: dict = {"index": sub}
        try:
            with winreg.OpenKey(key, sub) as k:
                for name, out_name in (
                    ("DriverDesc", "name"),
                    ("HardwareInformation.qwMemorySize", "qw_memory_size"),
                    ("HardwareInformation.MemorySize", "memory_size"),
                    ("DriverVersion", "driver_version"),
                ):
                    try:
                        value, kind = winreg.QueryValueEx(k, name)
                    except OSError:
                        continue
                    if isinstance(value, bytes):
                        value = int.from_bytes(value, "little")
                    row[out_name] = value
                    row[out_name + "_kind"] = kind
        except OSError as e:
            row["error"] = f"{type(e).__name__}: {e}"
        if len(row) > 1:
            found.append(row)
    return found


# --------------------------------------------------------------------------- HIP


class Hip:
    """The HIP runtime from one specific directory, with the loaded path proved."""

    def __init__(self, hip_dir: Path | None):
        self.dir = Path(hip_dir) if hip_dir else None
        self._dll_dir = None
        name = "amdhip64_7.dll" if IS_WIN else "libamdhip64.so"
        path = None
        if self.dir:
            path = find_lib(self.dir, name)
            if IS_WIN:
                try:
                    self._dll_dir = os.add_dll_directory(str(self.dir))
                except Exception:  # noqa: BLE001
                    pass
        self.requested = str(path) if path else name
        self.lib = ctypes.CDLL(self.requested)
        self.loaded = loaded_module_path(self.lib._handle) or self.requested
        for fn, argtypes in (
            ("hipInit", [ctypes.c_uint]),
            ("hipGetDeviceCount", [ctypes.POINTER(ctypes.c_int)]),
            ("hipSetDevice", [ctypes.c_int]),
            ("hipDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
            ("hipDeviceTotalMem", [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]),
            ("hipMemGetInfo", [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]),
            ("hipRuntimeGetVersion", [ctypes.POINTER(ctypes.c_int)]),
            ("hipDriverGetVersion", [ctypes.POINTER(ctypes.c_int)]),
            ("hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
            ("hipFree", [ctypes.c_void_p]),
            ("hipMemset", [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]),
            ("hipMemcpy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]),
            ("hipDeviceSynchronize", []),
            ("hipGetLastError", []),
        ):
            f = getattr(self.lib, fn)
            f.argtypes = argtypes
            f.restype = ctypes.c_int
        self.lib.hipGetErrorString.argtypes = [ctypes.c_int]
        self.lib.hipGetErrorString.restype = ctypes.c_char_p

    def err(self, code: int) -> str:
        try:
            return (self.lib.hipGetErrorString(code) or b"").decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return f"hip error {code}"

    def inventory(self) -> dict:
        out: dict = {
            "requested_dll": self.requested,
            "loaded_dll": self.loaded,
            "dll": describe_file(Path(self.loaded)) if self.loaded else {},
        }
        rc = self.lib.hipInit(0)
        out["hipInit_rc"] = rc
        if rc != 0:
            out["error"] = self.err(rc)
            return out
        for fn, key in (
            ("hipRuntimeGetVersion", "runtime_version"),
            ("hipDriverGetVersion", "driver_version"),
        ):
            v = ctypes.c_int(0)
            if getattr(self.lib, fn)(ctypes.byref(v)) == 0:
                out[key] = v.value
        n = ctypes.c_int(0)
        self.lib.hipGetDeviceCount(ctypes.byref(n))
        out["device_count"] = n.value
        devices = []
        for i in range(n.value):
            d: dict = {"index": i}
            self.lib.hipSetDevice(i)
            buf = ctypes.create_string_buffer(256)
            if self.lib.hipDeviceGetName(buf, 256, i) == 0:
                d["name"] = buf.value.decode("utf-8", "replace")
            total = ctypes.c_size_t(0)
            if self.lib.hipDeviceTotalMem(ctypes.byref(total), i) == 0:
                d["device_total_bytes"] = total.value
            free_b, total_b = ctypes.c_size_t(0), ctypes.c_size_t(0)
            rc = self.lib.hipMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b))
            d["hipMemGetInfo_rc"] = rc
            if rc == 0:
                # On Windows this "free" counts only this process's allocations,
                # so it is an optimistic ceiling, not machine-wide occupancy.
                d["free_bytes"], d["total_bytes"] = free_b.value, total_b.value
            devices.append(d)
        out["devices"] = devices
        return out

    def try_alloc(
        self,
        nbytes: int,
        device: int = 0,
    ) -> dict:
        """One allocation, written and read back. A hipMalloc that returns a
        pointer nothing was ever stored in is not evidence the memory exists."""
        res: dict = {"bytes": nbytes}
        rc = self.lib.hipInit(0)
        if rc != 0:
            return {**res, "ok": False, "stage": "hipInit", "error": self.err(rc)}
        rc = self.lib.hipSetDevice(device)
        if rc != 0:
            return {**res, "ok": False, "stage": "hipSetDevice", "error": self.err(rc)}
        ptr = ctypes.c_void_p()
        t0 = time.monotonic()
        rc = self.lib.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        res["malloc_seconds"] = round(time.monotonic() - t0, 3)
        if rc != 0 or not ptr:
            return {**res, "ok": False, "stage": "hipMalloc", "rc": rc, "error": self.err(rc)}
        try:
            rc = self.lib.hipMemset(ptr, 0xA5, ctypes.c_size_t(nbytes))
            if rc != 0:
                return {**res, "ok": False, "stage": "hipMemset", "rc": rc, "error": self.err(rc)}
            rc = self.lib.hipDeviceSynchronize()
            if rc != 0:
                return {
                    **res,
                    "ok": False,
                    "stage": "hipDeviceSynchronize",
                    "rc": rc,
                    "error": self.err(rc),
                }
            probe = ctypes.create_string_buffer(64)
            for offset in (0, nbytes // 2, nbytes - 64):
                src = ctypes.c_void_p(ptr.value + max(0, offset))
                rc = self.lib.hipMemcpy(probe, src, 64, HIP_MEMCPY_DEVICE_TO_HOST)
                if rc != 0:
                    return {
                        **res,
                        "ok": False,
                        "stage": f"hipMemcpy@{offset}",
                        "rc": rc,
                        "error": self.err(rc),
                    }
                if probe.raw[:64] != b"\xa5" * 64:
                    return {
                        **res,
                        "ok": False,
                        "stage": f"readback@{offset}",
                        "error": "buffer did not read back as written",
                    }
            res["ok"] = True
            free_b, total_b = ctypes.c_size_t(0), ctypes.c_size_t(0)
            if self.lib.hipMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b)) == 0:
                res["free_after_bytes"], res["total_after_bytes"] = free_b.value, total_b.value
            return res
        finally:
            self.lib.hipFree(ptr)


def read_hip(hip_dir: Path | None) -> dict:
    return Hip(hip_dir).inventory()


# --------------------------------------------------------------------------- Vulkan

VK_MAX_MEMORY_TYPES = 32
VK_MAX_MEMORY_HEAPS = 16
VK_MAX_EXTENSION_NAME_SIZE = 256
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2 = 1000059006
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT = 1000237000
VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1
VK_MEMORY_HEAP_MULTI_INSTANCE_BIT = 0x2
VK_DEVICE_TYPES = {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "cpu"}
MEMORY_PROPERTY_BITS = [
    (0x1, "DEVICE_LOCAL"),
    (0x2, "HOST_VISIBLE"),
    (0x4, "HOST_COHERENT"),
    (0x8, "HOST_CACHED"),
    (0x10, "LAZILY_ALLOCATED"),
    (0x20, "PROTECTED"),
]


class VkMemoryType(ctypes.Structure):
    _fields_ = [("propertyFlags", ctypes.c_uint32), ("heapIndex", ctypes.c_uint32)]


class VkMemoryHeap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]


class VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", VkMemoryType * VK_MAX_MEMORY_TYPES),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", VkMemoryHeap * VK_MAX_MEMORY_HEAPS),
    ]


class VkPhysicalDeviceMemoryProperties2(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("memoryProperties", VkPhysicalDeviceMemoryProperties),
    ]


class VkPhysicalDeviceMemoryBudgetPropertiesEXT(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("heapBudget", ctypes.c_uint64 * VK_MAX_MEMORY_HEAPS),
        ("heapUsage", ctypes.c_uint64 * VK_MAX_MEMORY_HEAPS),
    ]


class VkPhysicalDeviceProperties(ctypes.Structure):
    # Only the head is parsed; the tail is spare room for limits and sparse
    # properties, which the driver writes and this probe does not read.
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * 256),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("tail", ctypes.c_uint8 * 4096),
    ]


class VkExtensionProperties(ctypes.Structure):
    _fields_ = [
        ("extensionName", ctypes.c_char * VK_MAX_EXTENSION_NAME_SIZE),
        ("specVersion", ctypes.c_uint32),
    ]


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def _flag_names(flags: int) -> list[str]:
    return [name for bit, name in MEMORY_PROPERTY_BITS if flags & bit]


def read_vulkan_raw() -> dict:
    """Every heap of every physical device, from the loader directly.

    This is the ground truth the ggml reading is compared against: on an
    integrated part the same physical RAM is exposed through several
    device-local heaps, so summing them counts it twice."""
    lib_name = "vulkan-1.dll" if IS_WIN else "libvulkan.so.1"
    vk = ctypes.CDLL(lib_name)
    out: dict = {"library": loaded_module_path(vk._handle) or lib_name}

    api_version = (1 << 22) | (1 << 12)  # 1.1: memory properties 2 is core there
    try:
        vk.vkEnumerateInstanceVersion.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        vk.vkEnumerateInstanceVersion.restype = ctypes.c_int
        v = ctypes.c_uint32(0)
        if vk.vkEnumerateInstanceVersion(ctypes.byref(v)) == 0:
            out["loader_api_version"] = (
                f"{v.value >> 22}.{(v.value >> 12) & 0x3FF}.{v.value & 0xFFF}"
            )
            api_version = min(api_version, v.value)
    except AttributeError:
        out["loader_api_version"] = "1.0"

    app = VkApplicationInfo(
        sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName = b"amd_ci_memory_report",
        pEngineName = b"amd_ci",
        apiVersion = api_version,
    )
    ci = VkInstanceCreateInfo(
        sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo = ctypes.cast(ctypes.byref(app), ctypes.c_void_p),
    )
    vk.vkCreateInstance.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    vk.vkCreateInstance.restype = ctypes.c_int
    inst = ctypes.c_void_p()
    rc = vk.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
    if rc != 0:
        out["error"] = f"vkCreateInstance rc={rc}"
        return out

    try:
        vk.vkEnumeratePhysicalDevices.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        vk.vkEnumeratePhysicalDevices.restype = ctypes.c_int
        count = ctypes.c_uint32(0)
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), None)
        devices = (ctypes.c_void_p * max(1, count.value))()
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), devices)

        vk.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        vk.vkGetPhysicalDeviceProperties.restype = None
        vk.vkGetPhysicalDeviceMemoryProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        vk.vkGetPhysicalDeviceMemoryProperties.restype = None
        vk.vkEnumerateDeviceExtensionProperties.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        vk.vkEnumerateDeviceExtensionProperties.restype = ctypes.c_int
        mem2 = _memory_properties2(vk, inst)

        rows = []
        for i in range(count.value):
            pd = devices[i]
            props = VkPhysicalDeviceProperties()
            vk.vkGetPhysicalDeviceProperties(pd, ctypes.byref(props))
            ext_count = ctypes.c_uint32(0)
            vk.vkEnumerateDeviceExtensionProperties(pd, None, ctypes.byref(ext_count), None)
            exts = (VkExtensionProperties * max(1, ext_count.value))()
            vk.vkEnumerateDeviceExtensionProperties(pd, None, ctypes.byref(ext_count), exts)
            names = {e.extensionName.decode("utf-8", "replace") for e in exts[: ext_count.value]}
            has_budget = "VK_EXT_memory_budget" in names

            budget = VkPhysicalDeviceMemoryBudgetPropertiesEXT(
                sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
            )
            mp2 = VkPhysicalDeviceMemoryProperties2(
                sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
                pNext = ctypes.cast(ctypes.byref(budget), ctypes.c_void_p)
                if (has_budget and mem2)
                else None,
            )
            if mem2:
                mem2(pd, ctypes.byref(mp2))
                mp = mp2.memoryProperties
            else:
                mp = VkPhysicalDeviceMemoryProperties()
                vk.vkGetPhysicalDeviceMemoryProperties(pd, ctypes.byref(mp))

            heaps = []
            for h in range(mp.memoryHeapCount):
                row = {
                    "index": h,
                    "size_bytes": int(mp.memoryHeaps[h].size),
                    "flags": int(mp.memoryHeaps[h].flags),
                    "device_local": bool(mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT),
                    "multi_instance": bool(
                        mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT
                    ),
                }
                if has_budget and mem2:
                    row["budget_bytes"] = int(budget.heapBudget[h])
                    row["usage_bytes"] = int(budget.heapUsage[h])
                heaps.append(row)
            types = [
                {
                    "index": t,
                    "heap": int(mp.memoryTypes[t].heapIndex),
                    "flags": int(mp.memoryTypes[t].propertyFlags),
                    "names": _flag_names(int(mp.memoryTypes[t].propertyFlags)),
                }
                for t in range(mp.memoryTypeCount)
            ]
            local = [h["size_bytes"] for h in heaps if h["device_local"]]
            rows.append(
                {
                    "index": i,
                    "name": props.deviceName.decode("utf-8", "replace"),
                    "type": VK_DEVICE_TYPES.get(int(props.deviceType), int(props.deviceType)),
                    "vendor_id": hex(int(props.vendorID)),
                    "device_id": hex(int(props.deviceID)),
                    "api_version": f"{props.apiVersion >> 22}.{(props.apiVersion >> 12) & 0x3FF}"
                    f".{props.apiVersion & 0xFFF}",
                    "driver_version": int(props.driverVersion),
                    "memory_budget_ext": has_budget,
                    "heaps": heaps,
                    "memory_types": types,
                    # The two candidate readings, side by side, so the over-report is
                    # visible without recomputing anything downstream.
                    "device_local_sum_bytes": sum(local),
                    "device_local_max_bytes": max(local) if local else 0,
                    "all_heaps_sum_bytes": sum(h["size_bytes"] for h in heaps),
                }
            )
        out["devices"] = rows
    finally:
        try:
            vk.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            vk.vkDestroyInstance(inst, None)
        except Exception:  # noqa: BLE001
            pass
    return out


def _memory_properties2(vk, inst):
    """vkGetPhysicalDeviceMemoryProperties2, statically or through the loader."""
    for name in ("vkGetPhysicalDeviceMemoryProperties2", "vkGetPhysicalDeviceMemoryProperties2KHR"):
        try:
            fn = getattr(vk, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            fn.restype = None
            return fn
        except AttributeError:
            pass
    try:
        vk.vkGetInstanceProcAddr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        vk.vkGetInstanceProcAddr.restype = ctypes.c_void_p
        for name in (
            b"vkGetPhysicalDeviceMemoryProperties2",
            b"vkGetPhysicalDeviceMemoryProperties2KHR",
        ):
            addr = vk.vkGetInstanceProcAddr(inst, name)
            if addr:
                proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
                return proto(addr)
    except Exception:  # noqa: BLE001
        pass
    return None


# ------------------------------------------------------------------- ggml Vulkan


def read_ggml_vulkan(bin_dir: Path) -> dict:
    """What ggml reports, which is what llama.cpp and Studio place against.

    Same load path as Studio's own `_vulkan_probe.py`, in a child process, so a
    driver that dies taking its instance with it does not take the rest of the
    report."""
    r = run(
        [sys.executable, str(Path(__file__).resolve()), "--ggml-vulkan-only", str(bin_dir)], 300
    )
    out: dict = {"bin_dir": str(bin_dir), "rc": r.get("rc"), "stderr": r.get("stderr", "")}
    if r.get("rc") != 0:
        # A directory without the Vulkan backend is a wrong argument, not a
        # machine with no Vulkan memory; say so instead of returning no devices.
        raise RuntimeError(
            f"ggml Vulkan child rc={r.get('rc')}: " f"{(r.get('stderr') or '').strip()[:300]}"
        )
    try:
        out["devices"] = json.loads(r.get("stdout") or "[]")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"unparsable ggml Vulkan child output: {e}") from e
    return out


def ggml_vulkan_child(bin_dir: str) -> int:
    base_name = "ggml-base.dll" if IS_WIN else "libggml-base.so"
    vk_name = "ggml-vulkan.dll" if IS_WIN else "libggml-vulkan.so"
    directory = Path(bin_dir)
    if IS_WIN:
        try:
            os.add_dll_directory(bin_dir)
        except Exception:  # noqa: BLE001
            pass
    base_path, vk_path = find_lib(directory, base_name), find_lib(directory, vk_name)
    if not base_path or not vk_path:
        print(f"ggml vulkan libraries not found in {bin_dir}", file = sys.stderr)
        return 1
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    base = ctypes.CDLL(str(base_path), mode = mode)
    lib = ctypes.CDLL(str(vk_path), mode = mode)
    lib.ggml_backend_vk_get_device_count.restype = ctypes.c_int
    lib.ggml_backend_vk_get_device_memory.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ggml_backend_vk_get_device_memory.restype = None
    count = lib.ggml_backend_vk_get_device_count()

    types: dict[int, int] = {}
    names: dict[int, str] = {}
    try:
        lib.ggml_backend_vk_reg.restype = ctypes.c_void_p
        base.ggml_backend_reg_dev_count.argtypes = [ctypes.c_void_p]
        base.ggml_backend_reg_dev_count.restype = ctypes.c_size_t
        base.ggml_backend_reg_dev_get.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        base.ggml_backend_reg_dev_get.restype = ctypes.c_void_p
        base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
        base.ggml_backend_dev_type.restype = ctypes.c_int
        base.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
        base.ggml_backend_dev_description.restype = ctypes.c_char_p
        reg = lib.ggml_backend_vk_reg()
        for i in range(min(count, base.ggml_backend_reg_dev_count(reg))):
            dev = base.ggml_backend_reg_dev_get(reg, i)
            types[i] = base.ggml_backend_dev_type(dev)
            names[i] = (base.ggml_backend_dev_description(dev) or b"").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for i in range(count):
        free_b, total_b = ctypes.c_size_t(0), ctypes.c_size_t(0)
        lib.ggml_backend_vk_get_device_memory(i, ctypes.byref(free_b), ctypes.byref(total_b))
        rows.append(
            {
                "index": i,
                "free_bytes": free_b.value,
                "total_bytes": total_b.value,
                "ggml_dev_type": types.get(i),
                "is_igpu": types.get(i) == 2,
                "name": names.get(i, ""),
            }
        )
    print(json.dumps(rows))
    return 0


# --------------------------------------------------------------------------- fit


def read_fit(
    bin_dir: Path,
    model: str | None,
    env_extra: dict | None = None,
) -> dict:
    exe = ".exe" if IS_WIN else ""
    env = dict(os.environ)
    if not IS_WIN:
        env["LD_LIBRARY_PATH"] = f"{bin_dir}{os.pathsep}" + env.get("LD_LIBRARY_PATH", "")
    env.update(env_extra or {})
    out: dict = {"bin_dir": str(bin_dir)}
    server = bin_dir / f"llama-server{exe}"
    out["version"] = run([str(server), "--version"], 120, env)
    out["list_devices"] = run([str(server), "--list-devices"], 300, env)
    fit = bin_dir / f"llama-fit-params{exe}"
    if model and fit.is_file():
        out["fit_params"] = run([str(fit), "-m", model, "-ngl", "999", "-c", "8192"], 900, env)
    return out


# ----------------------------------------------------------------- allocation cap


def alloc_once_subprocess(
    hip_dir: Path | None,
    nbytes: int,
    device: int,
    timeout: int = 900,
    settle: float = 2.0,
) -> dict:
    """One candidate, one process. HIP's Windows free-memory accounting is
    process-scoped, so a stale allocator state would otherwise leak between
    candidates and turn a cap into a fragmentation measurement."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--try-alloc",
        str(nbytes),
        "--alloc-device",
        str(device),
    ]
    if hip_dir:
        cmd += ["--hip-dir", str(hip_dir)]
    r = run(cmd, timeout)
    # Let the driver hand the pages back before the next candidate asks for them,
    # so a refusal is the cap rather than the previous attempt still unwinding.
    time.sleep(settle)
    try:
        return json.loads((r.get("stdout") or "").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {
            "bytes": nbytes,
            "ok": False,
            "stage": "child",
            "error": f"rc={r.get('rc')} {(r.get('stderr') or '')[-400:]}",
        }


def read_alloc_cap(
    hip_dir: Path | None,
    lo_gib: float,
    hi_gib: float,
    reps: int,
    device: int,
    resolution_gib: float,
    attempt_timeout: int = 900,
) -> dict:
    """Bisect for the largest allocation that succeeds and reads back.

    Ascending and descending confirmation around the boundary, because an
    allocator with hysteresis or fragmentation gives different answers depending
    on the order candidates are tried, and one pass cannot tell the two apart."""
    attempts: list[dict] = []

    def attempt(gib: float) -> bool:
        res = alloc_once_subprocess(hip_dir, int(gib * GIB), device, attempt_timeout)
        res["gib"] = round(gib, 3)
        attempts.append(res)
        return bool(res.get("ok"))

    out: dict = {
        "hip_dir": str(hip_dir) if hip_dir else None,
        "device": device,
        "search_lo_gib": lo_gib,
        "search_hi_gib": hi_gib,
        "resolution_gib": resolution_gib,
        "attempt_timeout_s": attempt_timeout,
    }
    if not attempt(lo_gib):
        out["error"] = (
            f"the {lo_gib} GiB floor already fails, so there is no interval to "
            f"bisect; this is a broken runtime, not a cap"
        )
        out["attempts"] = attempts
        return out
    if attempt(hi_gib):
        out["max_ok_gib"] = hi_gib
        out["capped"] = False
        out["note"] = "the ceiling itself succeeded: the cap is above the search range"
        out["attempts"] = attempts
        return out

    lo, hi = lo_gib, hi_gib  # lo known good, hi known bad
    while hi - lo > resolution_gib:
        mid = (lo + hi) / 2
        if attempt(mid):
            lo = mid
        else:
            hi = mid
    out["max_ok_gib"] = round(lo, 3)
    out["min_fail_gib"] = round(hi, 3)
    out["capped"] = True

    # Confirmation: the boundary must reproduce from both directions.
    conf = {"below_ok": [], "above_fail": []}
    for _ in range(max(0, reps)):
        conf["below_ok"].append(attempt(lo))
        conf["above_fail"].append(not attempt(hi))
    out["confirmation"] = conf
    out["boundary_stable"] = all(conf["below_ok"]) and all(conf["above_fail"])
    out["attempts"] = attempts
    return out


# --------------------------------------------------------------------------- main

SECTIONS = ("host", "hip", "vulkan_raw", "ggml_vulkan", "fit", "alloc_cap")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default = "solo")
    ap.add_argument("--checkout", default = "", help = "directory holding the llama.cpp binaries")
    ap.add_argument("--out", default = "")
    ap.add_argument(
        "--hip-dir",
        default = "",
        help = "directory whose amdhip64_7.dll is loaded; defaults to --checkout",
    )
    ap.add_argument(
        "--vulkan-dir", default = "", help = "directory holding ggml-vulkan (the vulkan release)"
    )
    ap.add_argument("--fit-model", default = "")
    ap.add_argument("--sections", default = "host,hip,vulkan_raw,ggml_vulkan,fit")
    ap.add_argument("--alloc-lo-gib", type = float, default = 1.0)
    ap.add_argument("--alloc-hi-gib", type = float, default = 200.0)
    ap.add_argument("--alloc-resolution-gib", type = float, default = 0.5)
    ap.add_argument("--alloc-reps", type = int, default = 2)
    ap.add_argument("--alloc-device", type = int, default = 0)
    ap.add_argument(
        "--alloc-attempt-timeout",
        type = int,
        default = 900,
        help = "seconds one candidate allocation may take before it is a failure",
    )
    ap.add_argument(
        "--try-alloc",
        type = int,
        default = 0,
        help = "internal: attempt one allocation and print the JSON result",
    )
    ap.add_argument(
        "--ggml-vulkan-only",
        default = "",
        help = "internal: print the ggml Vulkan inventory of this directory",
    )
    a = ap.parse_args()

    if a.ggml_vulkan_only:
        return ggml_vulkan_child(a.ggml_vulkan_only)
    if a.try_alloc:
        hip_dir = Path(a.hip_dir) if a.hip_dir else None
        try:
            res = Hip(hip_dir).try_alloc(a.try_alloc, a.alloc_device)
        except Exception as e:  # noqa: BLE001
            res = {
                "bytes": a.try_alloc,
                "ok": False,
                "stage": "load",
                "error": f"{type(e).__name__}: {e}",
            }
        print(json.dumps(res))
        return 0 if res.get("ok") else 3

    checkout = Path(a.checkout) if a.checkout else None
    hip_dir = Path(a.hip_dir) if a.hip_dir else checkout
    vulkan_dir = Path(a.vulkan_dir) if a.vulkan_dir else None
    wanted = [s.strip() for s in a.sections.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")

    res: dict = {
        "state": a.state,
        "checkout": a.checkout,
        "hip_dir": str(hip_dir or ""),
        "vulkan_dir": a.vulkan_dir,
        "sections_requested": wanted,
        "platform": platform.platform(),
        "sections": {},
        "errors": {},
    }
    if hip_dir:
        name = "amdhip64_7.dll" if IS_WIN else "libamdhip64.so"
        found = find_lib(hip_dir, name)
        res["hip_dll_file"] = (
            describe_file(found) if found else {"error": f"{name} not in {hip_dir}"}
        )

    todo = {
        "host": lambda: read_host(),
        "hip": lambda: read_hip(hip_dir),
        "vulkan_raw": lambda: read_vulkan_raw(),
        "ggml_vulkan": lambda: read_ggml_vulkan(vulkan_dir or checkout),
        "fit": lambda: read_fit(checkout, a.fit_model or None),
        "alloc_cap": lambda: read_alloc_cap(
            hip_dir,
            a.alloc_lo_gib,
            a.alloc_hi_gib,
            a.alloc_reps,
            a.alloc_device,
            a.alloc_resolution_gib,
            a.alloc_attempt_timeout,
        ),
    }
    for name in wanted:
        if name in ("ggml_vulkan", "fit") and not (vulkan_dir or checkout):
            res["errors"][name] = "no --checkout or --vulkan-dir given"
            continue
        t0 = time.monotonic()
        try:
            res["sections"][name] = todo[name]()
        except Exception as e:  # noqa: BLE001
            res["errors"][name] = f"{type(e).__name__}: {e}"
        print(
            f"== {name}: {round(time.monotonic() - t0, 1)}s"
            f"{' ERROR ' + res['errors'][name] if name in res['errors'] else ''}",
            flush = True,
        )

    res["ok"] = not res["errors"]
    text = json.dumps(res, indent = 2)
    if a.out:
        Path(a.out).parent.mkdir(parents = True, exist_ok = True)
        Path(a.out).write_text(text, encoding = "utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "sections"}, indent = 2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

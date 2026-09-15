#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""NVIDIA inventory read from the driver's own libraries, for a host nvidia-smi cannot answer for.

An absent, stale or hanging nvidia-smi read as a CPU host, which is fatal to the install.
Ollama, Jan and lemonade ask NVML or the CUDA driver API instead, which ship with the driver.

* NVML (`libnvidia-ml.so.1` / `nvml.dll`): the PHYSICAL inventory, unmasked like nvidia-smi;
  the caller applies CUDA_VISIBLE_DEVICES.
* the CUDA driver API (`libcuda.so.1` / `nvcuda.dll`): honours the mask; used when NVML is
  unavailable.

Always a child process with a deadline: these libraries block in-process in the states that
hang nvidia-smi. Failing both, `/proc/driver/nvidia/version` names the driver, and the driver
major bounds the CUDA major (13 needs R580+, 12 R525+, 11 R450+).
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class NvidiaLibraryInventory:
    source: str  # "nvml" or "cuda"
    cuda_driver_version: tuple[int, int] | None
    driver_version: str
    # index, uuid, name, compute_cap ("8.9"); NVML order is physical and unmasked.
    devices: list[dict[str, str]] = field(default_factory = list)


def _library_candidates(kind: str) -> list[str]:
    if sys.platform == "darwin":
        return []
    if sys.platform == "win32":
        system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        if kind == "nvml":
            nvsmi = [
                os.path.join(root, "NVIDIA Corporation", "NVSMI", "nvml.dll")
                for root in (os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles"))
                if root
            ]
            return [os.path.join(system32, "nvml.dll"), *nvsmi, "nvml.dll"]
        return [os.path.join(system32, "nvcuda.dll"), "nvcuda.dll"]
    if kind == "nvml":
        return ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    return ["libcuda.so.1", "libcuda.so"]


def _load(kind: str) -> ctypes.CDLL | None:
    for name in _library_candidates(kind):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _split_cuda_version(packed: int) -> tuple[int, int] | None:
    # 13010 -> (13, 1), as CUDA packs it.
    if packed <= 0:
        return None
    return packed // 1000, (packed % 1000) // 10


class _NvmlMemory(ctypes.Structure):
    # nvmlMemory_t (v1): bytes, in this order.
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _mig_children(nvml, handle) -> list:
    """(handle, uuid, memory) per MIG instance of a device; empty without MIG or on an NVML
    too old to ask. Unused instance slots answer NOT_FOUND and are skipped."""
    get_max = getattr(nvml, "nvmlDeviceGetMaxMigDeviceCount", None)
    get_mig = getattr(nvml, "nvmlDeviceGetMigDeviceHandleByIndex", None)
    if get_max is None or get_mig is None:
        return []
    children = []
    try:
        count = ctypes.c_uint(0)
        if get_max(handle, ctypes.byref(count)) != 0:
            return []
        for index in range(count.value):
            mig = ctypes.c_void_p()
            if get_mig(handle, index, ctypes.byref(mig)) != 0:
                continue
            uuid = ctypes.create_string_buffer(96)
            if nvml.nvmlDeviceGetUUID(mig, uuid, 96) != 0:
                continue
            memory = _NvmlMemory()
            if nvml.nvmlDeviceGetMemoryInfo(mig, ctypes.byref(memory)) != 0:
                memory.total = memory.free = 0
            children.append((mig, uuid.value.decode("ascii", "replace"), memory))
    except Exception:
        return children
    return children


def _probe_nvml() -> dict | None:
    nvml = _load("nvml")
    if nvml is None:
        return None
    init = getattr(nvml, "nvmlInit_v2", None) or getattr(nvml, "nvmlInit", None)
    if init is None or init() != 0:
        return None
    try:
        version = ""
        buf = ctypes.create_string_buffer(96)
        if nvml.nvmlSystemGetDriverVersion(buf, 96) == 0:
            version = buf.value.decode("ascii", "replace")
        cuda = None
        packed = ctypes.c_int(0)
        get_cuda = getattr(nvml, "nvmlSystemGetCudaDriverVersion_v2", None) or getattr(
            nvml, "nvmlSystemGetCudaDriverVersion", None
        )
        if get_cuda is not None and get_cuda(ctypes.byref(packed)) == 0:
            cuda = _split_cuda_version(packed.value)
        count = ctypes.c_uint(0)
        get_count = getattr(nvml, "nvmlDeviceGetCount_v2", None) or nvml.nvmlDeviceGetCount
        if get_count(ctypes.byref(count)) != 0:
            return None
        get_handle = getattr(nvml, "nvmlDeviceGetHandleByIndex_v2", None) or (
            nvml.nvmlDeviceGetHandleByIndex
        )
        devices = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if get_handle(index, ctypes.byref(handle)) != 0:
                return None  # a GPU this reader cannot see is a GPU the selectors would miss
            name = ctypes.create_string_buffer(96)
            uuid = ctypes.create_string_buffer(96)
            major, minor = ctypes.c_int(0), ctypes.c_int(0)
            nvml.nvmlDeviceGetName(handle, name, 96)
            nvml.nvmlDeviceGetUUID(handle, uuid, 96)
            cap = ""
            if (
                nvml.nvmlDeviceGetCudaComputeCapability(
                    handle, ctypes.byref(major), ctypes.byref(minor)
                )
                == 0
            ):
                cap = f"{major.value}.{minor.value}"
            memory = _NvmlMemory()
            if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                memory.total = memory.free = 0
            devices.append(
                {
                    "index": str(index),
                    "uuid": uuid.value.decode("ascii", "replace"),
                    "name": name.value.decode("utf-8", "replace"),
                    "compute_cap": cap,
                    "memory_total_mib": str(memory.total // (1024 * 1024)),
                    "memory_free_mib": str(memory.free // (1024 * 1024)),
                }
            )
            # MIG slices, so a CUDA_VISIBLE_DEVICES=MIG-... assignment can be named: same
            # parent index and capability, their own uuid and memory, marked "mig".
            for mig, mig_uuid, mig_memory in _mig_children(nvml, handle):
                devices.append(
                    {
                        "index": str(index),
                        "uuid": mig_uuid,
                        "name": name.value.decode("utf-8", "replace") + " MIG",
                        "compute_cap": cap,
                        "memory_total_mib": str(mig_memory.total // (1024 * 1024)),
                        "memory_free_mib": str(mig_memory.free // (1024 * 1024)),
                        "mig": "1",
                    }
                )
        return {
            "source": "nvml",
            "cuda_driver_version": list(cuda) if cuda else None,
            "driver_version": version,
            "devices": devices,
        }
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


def _probe_cuda_driver() -> dict | None:
    cuda = _load("cuda")
    if cuda is None or cuda.cuInit(0) != 0:
        return None
    packed = ctypes.c_int(0)
    version = None
    if cuda.cuDriverGetVersion(ctypes.byref(packed)) == 0:
        version = _split_cuda_version(packed.value)
    count = ctypes.c_int(0)
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
        count.value = 0
    devices = []
    for index in range(count.value):
        device = ctypes.c_int(0)
        if cuda.cuDeviceGet(ctypes.byref(device), index) != 0:
            continue
        name = ctypes.create_string_buffer(96)
        cuda.cuDeviceGetName(name, 96, device)
        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, _MINOR = 76.
        ok = cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device) == 0
        ok = ok and cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device) == 0
        devices.append(
            {
                "index": str(index),
                "uuid": "",
                "name": name.value.decode("utf-8", "replace"),
                "compute_cap": f"{major.value}.{minor.value}" if ok else "",
                # Memory needs a context, which this probe never creates.
                "memory_total_mib": "0",
                "memory_free_mib": "0",
            }
        )
    return {
        "source": "cuda",
        "cuda_driver_version": list(version) if version else None,
        "driver_version": "",
        "devices": devices,
    }


def probe_in_process() -> dict | None:
    """The raw inventory, read in THIS process. Prefer `probe`, which bounds it."""
    for reader in (_probe_nvml, _probe_cuda_driver):
        try:
            result = reader()
        except Exception:
            result = None
        if result is not None:
            return result
    return None


def _from_payload(payload: object) -> NvidiaLibraryInventory | None:
    if not isinstance(payload, dict) or payload.get("source") not in ("nvml", "cuda"):
        return None
    version = payload.get("cuda_driver_version")
    cuda = tuple(int(part) for part in version[:2]) if isinstance(version, list) else None
    keys = ("index", "uuid", "name", "compute_cap", "memory_total_mib", "memory_free_mib")
    devices = [
        {key: str(row.get(key, "")) for key in keys}
        for row in payload.get("devices") or []
        if not row.get("mig")
        if isinstance(row, dict)
    ]
    return NvidiaLibraryInventory(
        source = str(payload["source"]),
        cuda_driver_version = cuda if cuda and len(cuda) == 2 else None,
        driver_version = str(payload.get("driver_version") or ""),
        devices = devices,
    )


def enabled() -> bool:
    """UNSLOTH_NVIDIA_LIBRARY_PROBE=0 turns every source here off, the /proc bound included.

    The test suites set it, so a test faking a CPU host does not find the real GPUs here.
    """
    return os.environ.get("UNSLOTH_NVIDIA_LIBRARY_PROBE", "1") != "0"


def probe(timeout: float = 20) -> NvidiaLibraryInventory | None:
    """The inventory, read in a child with a deadline. None when nothing answered in time."""
    if sys.platform == "darwin" or not enabled():
        return None
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [sys.executable, "-I", os.path.abspath(__file__), "--json"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
            **kwargs,
        )
    except Exception:
        return None
    if result.returncode not in (0, 1):
        return None
    try:
        return _from_payload(json.loads(result.stdout or "null"))
    except ValueError:
        return None


_DRIVER_MAJOR_CUDA = (
    (580, (13, 0)),
    (570, (12, 8)),
    (560, (12, 6)),
    (555, (12, 5)),
    (550, (12, 4)),
    (545, (12, 3)),
    (535, (12, 2)),
    (525, (12, 0)),
    (450, (11, 0)),
)


def cuda_version_for_driver(driver_version: str) -> tuple[int, int] | None:
    """The CUDA version a driver of this release is known to carry (a floor, never above)."""
    match = re.match(r"\s*(\d+)\.", driver_version or "")
    if match is None:
        return None
    major = int(match.group(1))
    for floor, cuda in _DRIVER_MAJOR_CUDA:
        if major >= floor:
            return cuda
    return None


def proc_driver_version(path: str = "/proc/driver/nvidia/version") -> str:
    """The kernel module's version from /proc, or "": present on Linux whenever the driver is."""
    try:
        with open(path, encoding = "utf-8", errors = "replace") as handle:
            head = handle.readline()
    except OSError:
        return ""
    match = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", head)
    return match.group(1) if match else ""


def main(argv: list[str]) -> int:
    payload = probe_in_process()
    if "--json" in argv:
        print(json.dumps(payload))
    elif payload and payload["devices"]:
        for row in payload["devices"]:
            print(f"GPU {row['index']}: {row['name']} (compute {row['compute_cap']})")
    return 0 if payload and payload["devices"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

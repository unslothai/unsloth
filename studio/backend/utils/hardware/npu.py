# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detect AMD NPUs without starting a runtime.

XDNA 2 (1022:17f0) is supported; XDNA 1 (1022:1502) is not.
Runtime readiness is checked separately by ``flm validate``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

_AMD_VENDOR = "1022"
_XDNA2_DEVICE = "17f0"
_XDNA1_DEVICE = "1502"
_PCI_ROOT = Path("/sys/bus/pci/devices")
_MODULE_ROOT = Path("/sys/module")


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding = "utf-8", errors = "replace").strip()
    except OSError:
        return None


def _linux_probe(pci_root: Path = _PCI_ROOT, module_root: Path = _MODULE_ROOT) -> dict[str, Any]:
    try:
        devices = sorted(pci_root.iterdir())
    except OSError:
        devices = []
    for device in devices:
        vendor = (_read(device / "vendor") or "").lower().removeprefix("0x")
        product = (_read(device / "device") or "").lower().removeprefix("0x")
        if vendor != _AMD_VENDOR or product not in (_XDNA2_DEVICE, _XDNA1_DEVICE):
            continue
        driver_link = device / "driver"
        driver = driver_link.resolve().name if driver_link.exists() else None
        accel = sorted((device / "accel").glob("accel*")) if (device / "accel").is_dir() else []
        return {
            "present": True,
            "family": "XDNA2" if product == _XDNA2_DEVICE else "XDNA1",
            "name": _read(device / "vbnv") or "AMD NPU",
            "driver": driver,
            "driver_version": _read(module_root / driver / "version") if driver else None,
            "firmware_version": _read(device / "fw_version"),
            "device_node": f"/dev/accel/{accel[0].name}" if accel else None,
        }
    return {"present": False}


def _windows_probe() -> dict[str, Any]:
    import winreg

    base = r"SYSTEM\CurrentControlSet\Enum\PCI"
    try:
        pci = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return {"present": False}
    with pci:
        index = 0
        while True:
            try:
                hardware_id = winreg.EnumKey(pci, index)
            except OSError:
                return {"present": False}
            index += 1
            upper = hardware_id.upper()
            if not upper.startswith(f"VEN_{_AMD_VENDOR.upper()}&DEV_"):
                continue
            product = upper.split("&DEV_", 1)[1][:4].lower()
            if product not in (_XDNA2_DEVICE, _XDNA1_DEVICE):
                continue
            driver_version = None
            name = None
            try:
                with winreg.OpenKey(pci, hardware_id) as by_id:
                    instance = winreg.EnumKey(by_id, 0)
                    with winreg.OpenKey(by_id, instance) as device:
                        name = _reg_value(device, "FriendlyName") or _reg_value(
                            device, "DeviceDesc"
                        )
                        driver_key = _reg_value(device, "Driver")
                if driver_key:
                    with winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE,
                        rf"SYSTEM\CurrentControlSet\Control\Class\{driver_key}",
                    ) as driver:
                        driver_version = _reg_value(driver, "DriverVersion")
            except OSError:
                pass
            if name and ";" in name:
                # DeviceDesc is stored as "@oemNN.inf,%key%;Readable name".
                name = name.rsplit(";", 1)[1]
            return {
                "present": True,
                "family": "XDNA2" if product == _XDNA2_DEVICE else "XDNA1",
                "name": name or "AMD NPU",
                "driver": "installed" if driver_version else None,
                "driver_version": driver_version,
                "firmware_version": None,
                "device_node": None,
            }


def _reg_value(key, name: str) -> Optional[str]:
    import winreg
    try:
        value, _kind = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return str(value) if value is not None else None


def detect_amd_npu() -> dict[str, Any]:
    """Return NPU hardware details and whether FastFlowLM supports this platform."""
    try:
        if sys.platform.startswith("linux"):
            info = _linux_probe()
        elif sys.platform == "win32":
            info = _windows_probe()
        else:
            info = {"present": False}
    except Exception as exc:  # noqa: BLE001 -- detection must never break startup or /status
        info = {"present": False, "error": str(exc)}
    info["supported"] = bool(info.get("present")) and info.get("family") == "XDNA2"
    return info

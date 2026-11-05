# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adds the Windows facts a native llama-server crash report needs to its attempt log."""

from __future__ import annotations

import ntpath
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

_APPLICATION_ERROR_QUERY = (
    "*[System[Provider[@Name='Application Error'] and EventID=1000"
    " and TimeCreated[timediff(@SystemTime) <= 3600000]]]"
)
# Windows Error Reporting writes the event shortly after the process dies.
_EVENT_WAIT_ATTEMPTS = 10
_EVENT_WAIT_INTERVAL_S = 1.0
_WEVTUTIL_TIMEOUT_S = 5.0

# Event 1000 insertion strings, by position: older builds leave them unnamed.
_MODULE_NAME = 3
_MODULE_VERSION = 4
_EXCEPTION_CODE = 6
_FAULT_OFFSET = 7
_PROCESS_ID = 8

_DISPLAY_ADAPTERS_KEY = (
    r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
)
_IMPLICIT_LAYERS_KEY = r"SOFTWARE\Khronos\Vulkan\ImplicitLayers"


def _hex(value: str) -> str:
    # The event stores the exception code and offset as bare hex.
    return value if value.lower().startswith("0x") else f"0x{value}"


def _fault_from_events(xml_text: str, pid: int) -> Optional[str]:
    """The fault line for ``pid`` from ``wevtutil /f:xml`` output, which has no root element."""
    try:
        events = ET.fromstring(f"<Events>{xml_text}</Events>")
    except ET.ParseError:
        return None
    for event in events:
        fields = [node.text or "" for node in event.iter() if node.tag.endswith("}Data")]
        if len(fields) <= _PROCESS_ID:
            continue
        try:
            event_pid = int(fields[_PROCESS_ID], 16)
        except ValueError:
            continue
        if event_pid == pid:
            return (
                f"module={fields[_MODULE_NAME]} version={fields[_MODULE_VERSION]} "
                f"exception={_hex(fields[_EXCEPTION_CODE])} offset={_hex(fields[_FAULT_OFFSET])}"
            )
    return None


def _query_fault(pid: int) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "wevtutil",
                "qe",
                "Application",
                f"/q:{_APPLICATION_ERROR_QUERY}",
                "/c:20",
                "/rd:true",
                "/f:xml",
            ],
            capture_output = True,
            timeout = _WEVTUTIL_TIMEOUT_S,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _fault_from_events(result.stdout.decode("utf-8", "replace"), pid)


def _wait_for_fault(pid: int) -> Optional[str]:
    for attempt in range(_EVENT_WAIT_ATTEMPTS):
        if attempt:
            time.sleep(_EVENT_WAIT_INTERVAL_S)
        fault = _query_fault(pid)
        if fault is not None:
            return fault
    return None


def _gpu_drivers() -> list[str]:
    import winreg

    drivers = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_ADAPTERS_KEY) as adapters:
            for index in range(winreg.QueryInfoKey(adapters)[0]):
                try:
                    with winreg.OpenKey(adapters, winreg.EnumKey(adapters, index)) as adapter:
                        name = winreg.QueryValueEx(adapter, "DriverDesc")[0]
                        version = winreg.QueryValueEx(adapter, "DriverVersion")[0]
                except OSError:
                    continue
                drivers.append(f"{name} {version}")
    except OSError:
        pass
    return drivers


def _enabled_implicit_layers() -> list[str]:
    import winreg

    layers = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, _IMPLICIT_LAYERS_KEY) as key:
                for index in range(winreg.QueryInfoKey(key)[1]):
                    path, disabled, _ = winreg.EnumValue(key, index)
                    # The Vulkan loader skips any manifest whose value is nonzero.
                    if disabled == 0:
                        layers.append(ntpath.basename(path))
        except OSError:
            continue
    return layers


def _append_details(log_path: Path, pid: int) -> None:
    lines = [
        f"[studio] crash {_wait_for_fault(pid) or 'module=unknown'}",
        f"[studio] gpu drivers: {', '.join(_gpu_drivers()) or 'unknown'}",
        f"[studio] vulkan implicit layers: {', '.join(_enabled_implicit_layers()) or 'none'}",
    ]
    try:
        with open(log_path, "a", encoding = "utf-8") as log:
            log.write("\n".join(lines) + "\n")
    except OSError:
        pass


def record_crash_details(log_path: Path, pid: int) -> None:
    """Appends the details from a daemon thread, so a CPU fallback never waits on the event log."""
    if sys.platform != "win32":
        return
    try:
        threading.Thread(
            target = _append_details,
            args = (log_path, pid),
            daemon = True,
            name = "llama-crash-details",
        ).start()
    except RuntimeError:
        pass

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows AMD detection survives a missing or unhealthy WMI repository."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"


def _load_stack_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("studio_install_python_stack_registry", STACK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stack = _load_stack_module()


class _RegistryKey:
    def __init__(self, subkeys: tuple[str, ...] = (), values: dict[str, str] | None = None):
        self.subkeys = subkeys
        self.values = values or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeWinreg:
    HKEY_LOCAL_MACHINE = object()

    def __init__(self):
        self.adapters = {
            "0000": _RegistryKey(
                values={
                    "MatchingDeviceId": "PCI\\VEN_1002&DEV_743F",
                    "DriverDesc": "AMD Radeon RX 6700S",
                }
            )
        }
        self.root = _RegistryKey(subkeys=tuple(self.adapters))

    def OpenKey(self, parent, name):
        if parent is self.HKEY_LOCAL_MACHINE:
            return self.root
        return self.adapters[name]

    @staticmethod
    def QueryInfoKey(key):
        return len(key.subkeys), 0, 0

    @staticmethod
    def EnumKey(key, index):
        return key.subkeys[index]

    @staticmethod
    def QueryValueEx(key, name):
        if name not in key.values:
            raise OSError(name)
        return key.values[name], 1


def test_registry_fallback_recovers_an_amd_gfx_target(monkeypatch):
    fake_winreg = _FakeWinreg()
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    monkeypatch.setattr(stack, "IS_WINDOWS", True)
    monkeypatch.setattr(stack.shutil, "which", lambda _name: None)
    monkeypatch.setattr(stack, "_pick_visible_index", lambda _count, warn = True: 0)
    monkeypatch.setattr(stack, "_safe_print", lambda *_args, **_kwargs: None)

    class _FailedWmi:
        returncode = 1
        stdout = b""

    monkeypatch.setattr(stack.subprocess, "run", lambda *_args, **_kwargs: _FailedWmi())

    assert stack._windows_amd_gpu_names_from_registry() == ["AMD Radeon RX 6700S"]
    assert stack._detect_windows_gfx_arch() == "gfx1030"


def test_registry_fallback_is_after_the_existing_wmi_probe():
    source = STACK_PATH.read_text(encoding = "utf-8")
    assert "_windows_amd_gpu_names_from_registry()" in source
    assert source.rindex("_windows_amd_gpu_names_from_registry()") > source.index(
        "Get-CimInstance Win32_VideoController"
    )

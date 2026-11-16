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
    spec = importlib.util.spec_from_file_location(
        "studio_install_python_stack_registry", STACK_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stack = _load_stack_module()


class _RegistryKey:
    def __init__(
        self,
        subkeys: tuple[str, ...] = (),
        values: dict[str, str] | None = None,
    ):
        self.subkeys = subkeys
        self.values = values or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeWinreg:
    HKEY_LOCAL_MACHINE = object()

    def __init__(self, adapters: dict[str, dict[str, str]]):
        self.adapters = {k: _RegistryKey(values = v) for k, v in adapters.items()}
        self.root = _RegistryKey(subkeys = tuple(self.adapters))

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


_RX_6700S = {"MatchingDeviceId": "PCI\\VEN_1002&DEV_73EF", "DriverDesc": "AMD Radeon RX 6700S"}


def _windows_host(
    monkeypatch,
    adapters,
    wmi_returncode,
    wmi_stdout = b"",
):
    monkeypatch.setitem(sys.modules, "winreg", _FakeWinreg(adapters))
    monkeypatch.setattr(stack, "IS_WINDOWS", True)
    monkeypatch.setattr(stack.shutil, "which", lambda _name: None)
    monkeypatch.setattr(stack.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(stack, "_safe_print", lambda *_args, **_kwargs: None)
    for var in ("UNSLOTH_ROCM_GFX_ARCH", "HIP_PATH", "ROCM_PATH", "HIP_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)

    class _Wmi:
        returncode = wmi_returncode
        stdout = wmi_stdout

    monkeypatch.setattr(stack.subprocess, "run", lambda *_args, **_kwargs: _Wmi())


def test_registry_fallback_recovers_the_card_when_wmi_fails(monkeypatch):
    _windows_host(monkeypatch, {"0000": _RX_6700S}, wmi_returncode = 1)
    assert stack._windows_amd_gpu_names_from_registry() == ["AMD Radeon RX 6700S"]
    assert stack._detect_windows_gfx_arch() == "gfx1032"


def test_stale_registry_key_does_not_overrule_a_working_wmi(monkeypatch):
    stale_amd = {
        "MatchingDeviceId": "PCI\\VEN_1002&DEV_73DF",
        "DriverDesc": "AMD Radeon RX 6700 XT",
    }
    _windows_host(monkeypatch, {"0000": stale_amd}, wmi_returncode = 0, wmi_stdout = b"")
    assert stack._detect_windows_gfx_arch() is None


def test_wmi_named_rx_6700s_gets_its_own_device_pack(monkeypatch):
    _windows_host(monkeypatch, {}, wmi_returncode = 0, wmi_stdout = b"AMD Radeon RX 6700S|0\r\n")
    gfx = stack._detect_windows_gfx_arch()
    assert gfx == "gfx1032"
    assert stack._windows_multiarch_torch_pkg_specs(gfx)[0].startswith("torch[device-gfx1032]==")

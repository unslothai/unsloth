# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows driver-only AMD hosts: a WMI GPU name picks the per-arch device pack for that exact chip."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"


def _load_stack_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("studio_install_python_stack_name", STACK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stack = _load_stack_module()


@pytest.mark.parametrize(
    "wmi_name,expected",
    [
        ("AMD Radeon RX 6700S", "gfx1032"),
        ("AMD Radeon RX 6800S", "gfx1032"),
        ("AMD Radeon RX 6700 XT", "gfx1031"),
        ("AMD Radeon RX 6800M", "gfx1031"),
        ("AMD Radeon RX 6800 XT", "gfx1030"),
        ("AMD Radeon RX 6750 GRE 10GB", "gfx1032"),
        # TheRock: gfx1153 is "AMD Radeon 820M iGPU"; AMD ships a Windows device pack for it.
        ("AMD Radeon 820M Graphics", "gfx1153"),
        ("AMD Ryzen AI 5 330 w/ Radeon 820M", "gfx1153"),
    ],
)
def test_wmi_name_gets_its_own_device_pack(monkeypatch, wmi_name, expected):
    monkeypatch.setattr(stack, "IS_WINDOWS", True)
    monkeypatch.setattr(stack.shutil, "which", lambda _name: None)
    monkeypatch.setattr(stack.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(stack, "_safe_print", lambda *_args, **_kwargs: None)
    for var in (
        "UNSLOTH_ROCM_GFX_ARCH",
        "HIP_PATH",
        "ROCM_PATH",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(var, raising = False)

    class _Wmi:
        returncode = 0
        stdout = f"{wmi_name}|0\r\n".encode()

    monkeypatch.setattr(stack.subprocess, "run", lambda *_args, **_kwargs: _Wmi())
    gfx = stack._detect_windows_gfx_arch()
    assert gfx == expected
    assert stack._windows_multiarch_torch_pkg_specs(gfx)[0].startswith(
        f"torch[device-{expected}]=="
    )

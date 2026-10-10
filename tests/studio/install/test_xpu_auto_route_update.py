# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""install.sh's Intel GPU route survives `unsloth studio update`: _ensure_xpu_torch acts on the
resolved xpu backend, or the xpu flavor the install recorded, and on nothing else unpinned."""

import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_xpu_auto", ROOT / "studio" / "install_python_stack.py"
)
stack = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stack
_SPEC.loader.exec_module(stack)


@pytest.fixture(autouse = True)
def _linux(monkeypatch):
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    for var in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
        monkeypatch.delenv(var, raising = False)
    stack._invalidate_torch_runtime_probe()
    yield
    stack._invalidate_torch_runtime_probe()


def _run(
    backend,
    recorded,
    version = "2.11.0+cpu",
):
    with (
        patch.object(stack, "_TORCH_BACKEND", backend),
        patch.object(stack, "_RECORDED_TORCH_TAG", recorded),
        patch.object(stack, "_probe_torch_runtime", return_value = (True, True, version, None, None)),
        patch.object(stack, "pip_install") as pip,
    ):
        stack._ensure_xpu_torch()
    return pip


@pytest.mark.parametrize(
    ("backend", "recorded"),
    [("xpu", None), ("", "xpu")],
    ids = ["install.sh route", "standalone update, recorded xpu"],
)
def test_the_auto_route_repairs_a_cpu_wheel_from_the_xpu_index(backend, recorded):
    pip = _run(backend, recorded)
    assert pip.called
    assert pip.call_args.args[-2:] == ("--index-url", "https://download.pytorch.org/whl/xpu")


@pytest.mark.parametrize(
    ("backend", "recorded"),
    [("", None), ("", "cpu"), ("cpu", "xpu"), ("cuda", None), ("rocm", "xpu")],
)
def test_no_xpu_signal_or_another_stated_backend_leaves_torch_alone(backend, recorded):
    assert not _run(backend, recorded).called


def test_another_family_pin_wins_over_a_recorded_xpu(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cpu")
    assert not _run("", "xpu").called


def test_a_supported_xpu_wheel_is_left_alone():
    assert not _run("xpu", None, version = "2.9.1+xpu").called


def test_install_sh_exports_the_xpu_backend():
    source = (ROOT / "install.sh").read_text(encoding = "utf-8")
    assert re.search(r'^\s*xpu\)\s*export UNSLOTH_TORCH_BACKEND="xpu"', source, re.M)

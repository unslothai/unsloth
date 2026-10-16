# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Setup installs flash-attn but not causal-conv1d or mamba-ssm: those (~550 MB) install when a
model that needs them is trained or loaded (core/training/worker.py, utils/ssm_runtime.py), and
`unsloth install-kernels` still provisions them ahead of time for offline images. After torch
moves, setup still removes an installed copy whose compiled extension no longer imports."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))
sys.path.insert(0, str(STUDIO_DIR / "backend"))

import install_python_stack as ips  # noqa: E402

SOURCE = (STUDIO_DIR / "install_python_stack.py").read_text(encoding = "utf-8")
TREE = ast.parse(SOURCE)


def _called_names() -> set[str]:
    return {
        node.func.id
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_setup_has_no_ssm_kernel_step():
    assert not hasattr(ips, "_ensure_ssm_kernels")
    assert "install_kernel" not in _called_names()
    assert '_progress("mamba kernels")' not in SOURCE
    assert "causal-conv1d + mamba-ssm" not in SOURCE


def test_flash_attn_and_the_stale_kernel_check_run_at_all_three_sites():
    calls = [m.start() for m in re.finditer(r"\n\s+_ensure_flash_attn\(\)\n", SOURCE)]
    assert len(calls) == 3
    for start in calls:
        following = SOURCE[start : start + 120]
        assert "_remove_stale_ssm_kernels()" in following, following
    assert "base_total += 1  # flash-attn, Linux and Windows" in SOURCE


class _Kernels:
    """Which SSM distributions are installed and how their extension import ends: True loads,
    False raises ImportError, None hangs, an int is a raw exit code (negative = signal)."""

    def __init__(
        self,
        installed: dict,
        torch_imports = True,
    ):
        self.installed = dict(installed)
        self.torch_imports = torch_imports
        self.probes: list[str] = []
        self.removed: list[str] = []

    def version(self, distribution):
        return "1.0" if distribution in self.installed else None

    def run(self, cmd, *args, **kwargs):
        code = cmd[-1]
        self.probes.append(code)
        if code == "import torch":
            if self.torch_imports is None:
                raise subprocess.TimeoutExpired(cmd, 1)
            if self.torch_imports:
                return subprocess.CompletedProcess(cmd, 0, stderr = "")
            return subprocess.CompletedProcess(
                cmd, 1, stderr = "OSError: libtorch_cuda.so: cannot open shared object file"
            )
        assert "import torch" in code  # the extension probe proves torch imports too
        extension = "causal_conv1d_cuda" if "causal_conv1d_cuda" in code else "selective_scan_cuda"
        distribution = "causal-conv1d" if extension == "causal_conv1d_cuda" else "mamba-ssm"
        state = self.installed[distribution]
        if state is None:
            raise subprocess.TimeoutExpired(cmd, 1)
        if state is True:
            return subprocess.CompletedProcess(cmd, 0, stderr = "")
        if state is False:
            return subprocess.CompletedProcess(
                cmd,
                ips._SSM_EXTENSION_BROKEN_EXIT,
                stderr = f"{extension}: ImportError: {extension}.so: undefined symbol: _ZN3c10",
            )
        return subprocess.CompletedProcess(cmd, state, stderr = "Killed")

    def uninstall(self, distribution):
        self.removed.append(distribution)
        self.installed.pop(distribution, None)
        return True


def _run_stale_check(monkeypatch, host: _Kernels, **flags):
    monkeypatch.setattr(ips, "NO_TORCH", flags.get("no_torch", False))
    monkeypatch.setattr(ips, "IS_MACOS", flags.get("macos", False))
    monkeypatch.setattr(ips, "_installed_distribution_version", host.version)
    monkeypatch.setattr(ips, "_uninstall_distribution", host.uninstall)
    monkeypatch.setattr(ips, "_count_install_action", lambda: None)
    warnings = []
    monkeypatch.setattr(ips, "_step", lambda label, value, color_fn = None: warnings.append(value))
    monkeypatch.setattr(ips, "_note", lambda message, color_fn = None: warnings.append(message))
    monkeypatch.setattr(ips.subprocess, "run", host.run)
    installs = []
    monkeypatch.setattr(ips, "install_prebuilt", lambda *a, **k: installs.append(a) or "installed")
    ips._remove_stale_ssm_kernels()
    assert installs == []  # never installs anything
    return warnings


def test_stale_kernels_that_no_longer_import_are_removed(monkeypatch):
    host = _Kernels({"causal-conv1d": False, "mamba-ssm": False})
    warnings = _run_stale_check(monkeypatch, host)
    assert host.removed == ["causal-conv1d", "mamba-ssm"]
    assert len(warnings) == 2
    # The probe's stderr tail says why.
    assert "undefined symbol" in warnings[0] and "selective_scan_cuda" in warnings[1]


def test_kernels_that_still_import_are_kept(monkeypatch):
    host = _Kernels({"causal-conv1d": True, "mamba-ssm": False})
    _run_stale_check(monkeypatch, host)
    assert host.removed == ["mamba-ssm"]


def test_absent_kernels_are_not_probed_or_installed(monkeypatch):
    host = _Kernels({})
    _run_stale_check(monkeypatch, host)
    assert host.probes == [] and host.removed == []


def test_a_hung_probe_is_inconclusive_and_keeps_the_kernel(monkeypatch):
    host = _Kernels({"causal-conv1d": None})
    _run_stale_check(monkeypatch, host)
    assert host.removed == []


def test_no_torch_or_macos_skips_the_check(monkeypatch):
    for flags in ({"no_torch": True}, {"macos": True}):
        host = _Kernels({"causal-conv1d": False})
        _run_stale_check(monkeypatch, host, **flags)
        assert host.probes == []


@pytest.mark.parametrize("torch_imports", [False, None])
def test_a_torch_that_does_not_import_keeps_every_kernel(monkeypatch, torch_imports):
    # A half-finished repair or a transient failure in torch says nothing about the kernels, and an
    # offline machine could not reinstall them.
    host = _Kernels({"causal-conv1d": False, "mamba-ssm": False}, torch_imports = torch_imports)
    warnings = _run_stale_check(monkeypatch, host)
    assert host.removed == []
    assert host.probes == ["import torch"]
    if torch_imports is False:
        assert "libtorch_cuda.so" in warnings[0]


@pytest.mark.parametrize("exit_code", [-9, -11, 1, 2])
def test_a_killed_or_unrelated_failure_is_inconclusive(monkeypatch, exit_code):
    # A signal (OOM killer, segfault) or an exit that is not the extension's ImportError / OSError.
    host = _Kernels({"causal-conv1d": exit_code, "mamba-ssm": exit_code})
    warnings = _run_stale_check(monkeypatch, host)
    assert host.removed == []
    assert all("kept it" in message for message in warnings)


def test_the_probed_extensions_match_the_kernel_checks():
    for name, extension in ips._SSM_KERNEL_EXTENSIONS.items():
        assert ips.KERNELS[name][1] == f"import torch, {extension}"


def test_the_kernel_cli_still_provisions_both_ahead_of_time():
    from utils import kernel_install
    assert {"causal_conv1d", "mamba_ssm"} <= set(kernel_install.KERNELS)
    assert callable(kernel_install.install_kernel)

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
    """Which SSM distributions are installed and whether their compiled extension imports."""

    def __init__(self, installed: dict[str, bool]):
        self.installed = dict(installed)
        self.probes: list[str] = []
        self.removed: list[str] = []

    def version(self, distribution):
        return "1.0" if distribution in self.installed else None

    def run(self, cmd, *args, **kwargs):
        check = cmd[-1]
        self.probes.append(check)
        distribution = "causal-conv1d" if "causal_conv1d" in check else "mamba-ssm"
        if self.installed.get(distribution) is None:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0 if self.installed[distribution] else 1)

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


def test_the_kernel_cli_still_provisions_both_ahead_of_time():
    from utils import kernel_install
    assert {"causal_conv1d", "mamba_ssm"} <= set(kernel_install.KERNELS)
    assert callable(kernel_install.install_kernel)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""amd-smi on Windows that exits 0 with no stdout must be spawned once, not on every poll."""

from __future__ import annotations

import subprocess

import pytest

from utils.hardware import amd


@pytest.fixture
def spawns(monkeypatch):
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout = "", stderr = "Invalid platform\n")

    monkeypatch.setattr(amd, "_amd_smi_disabled", False)
    monkeypatch.setattr(amd, "_amd_smi_consecutive_failures", 0)
    monkeypatch.setattr(amd, "_amd_smi_allowed", lambda: True)
    monkeypatch.setattr(amd.shutil, "which", lambda name: "C:/Windows/System32/amd-smi.exe")
    monkeypatch.setattr(amd.subprocess, "run", _fake_run)
    return calls


def test_windows_silent_exit_disables_after_one_spawn(spawns, monkeypatch):
    monkeypatch.setattr(amd.platform, "system", lambda: "Windows")
    for _ in range(5):
        assert amd._run_amd_smi("metric") is None
    assert len(spawns) == 1
    assert amd._amd_smi_disabled is True


def test_linux_silent_exit_keeps_polling(spawns, monkeypatch):
    monkeypatch.setattr(amd.platform, "system", lambda: "Linux")
    for _ in range(3):
        assert amd._run_amd_smi("metric") is None
    assert len(spawns) == 3
    assert amd._amd_smi_disabled is False


@pytest.mark.parametrize(
    "flag, allowed",
    [(None, False), ("0", False), ("1", True), ("true", True)],
)
def test_windows_needs_explicit_opt_in_even_with_a_hip_sdk(monkeypatch, tmp_path, flag, allowed):
    # A HIP SDK on PATH used to allow amd-smi; the driver's amd-smi.exe still prompts there.
    real_isfile, real_exists = amd.os.path.isfile, amd.os.path.exists
    is_hipinfo = lambda p: str(p).lower().endswith("hipinfo.exe")
    monkeypatch.setattr(amd.os.path, "isfile", lambda p: is_hipinfo(p) or real_isfile(p))
    monkeypatch.setattr(amd.os.path, "exists", lambda p: is_hipinfo(p) or real_exists(p))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HIP_PATH", str(tmp_path))
    if flag is None:
        monkeypatch.delenv("UNSLOTH_ENABLE_AMD_SMI", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_ENABLE_AMD_SMI", flag)
    monkeypatch.setattr(amd.platform, "system", lambda: "Windows")
    assert amd._amd_smi_allowed() is allowed


def test_linux_always_allowed(monkeypatch):
    monkeypatch.setattr(amd.platform, "system", lambda: "Linux")
    monkeypatch.setenv("UNSLOTH_ENABLE_AMD_SMI", "0")
    assert amd._amd_smi_allowed() is True

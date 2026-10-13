"""amd-smi on Windows that exits 0 with no stdout must be spawned once, not on every poll.

The driver's System32 amd-smi.exe on hosts without wmic prints "Invalid platform" to
stderr, exits 0, and pops a DiskPart UAC prompt per spawn. A zero exit kept it out of
the failure breaker, so the System page's live monitor spawned it forever.
"""

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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for Studio's PowerShell resolution (#9440)."""

import ntpath
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unsloth_cli._studio_runtime_gate import _resolve_windows_powershell  # noqa: E402


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows PowerShell")
def test_resolves_through_path_when_available(monkeypatch):
    monkeypatch.setenv("SystemRoot", os.environ.get("SystemRoot", r"C:\Windows"))
    resolved = _resolve_windows_powershell()
    assert ntpath.isabs(resolved)
    assert os.path.isfile(resolved)


def test_falls_back_to_the_builtin_location_when_path_lacks_powershell(monkeypatch, tmp_path):
    # Simulate the stripped PATH from a GUI launch.
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("SystemRoot", os.environ.get("SystemRoot", r"C:\Windows"))

    resolved = _resolve_windows_powershell()

    expected_root = os.environ.get("SystemRoot", r"C:\Windows")
    if os.path.isfile(
        ntpath.join(expected_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    ):
        assert resolved == ntpath.join(
            expected_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
        )
    else:
        pytest.skip("builtin Windows PowerShell not present on this host")


def test_returns_the_bare_name_as_a_last_resort(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    monkeypatch.delenv("ProgramFiles", raising = False)

    # Preserve the familiar subprocess error when no host exists.
    assert _resolve_windows_powershell() == "powershell.exe"

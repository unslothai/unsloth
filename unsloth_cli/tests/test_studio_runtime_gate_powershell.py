"""The Studio-update gate must find PowerShell without PATH help (#9440).

A GUI-launched Desktop app can run with a PATH that omits
``System32\\WindowsPowerShell\\v1.0``; the gate's ``subprocess`` call then died
with ``FileNotFoundError: [WinError 2]`` during ``unsloth studio setup`` before
PowerShell ever started.
"""

import ntpath
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unsloth_cli._studio_runtime_gate import _resolve_windows_powershell  # noqa: E402


def test_resolves_through_path_when_available(monkeypatch):
    monkeypatch.setenv("SystemRoot", os.environ.get("SystemRoot", r"C:\Windows"))
    resolved = _resolve_windows_powershell()
    # Either a PATH hit or the builtin absolute location — never a bare name that
    # depends on the caller's PATH again.
    assert ntpath.isabs(resolved)
    assert os.path.isfile(resolved)


def test_falls_back_to_the_builtin_location_when_path_lacks_powershell(monkeypatch, tmp_path):
    # A PATH with no PowerShell (a stripped GUI environment) must still resolve:
    # the builtin System32 location is absolute and does not depend on PATH.
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

    # No PATH hit, no builtin under the (faked) SystemRoot, no pwsh: return the
    # bare name so the subprocess error stays the familiar WinError 2 rather
    # than a new failure mode.
    assert _resolve_windows_powershell() == "powershell.exe"

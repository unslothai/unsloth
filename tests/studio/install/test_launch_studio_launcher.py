"""Guard install.ps1's launch-studio.vbs against re-introducing the AV-heuristic
shape: a WScript .vbs spawning a hidden, ExecutionPolicy-Bypass PowerShell."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_PS1 = REPO_ROOT / "install.ps1"


def _vbs_block() -> str:
    text = INSTALL_PS1.read_text(encoding = "utf-8")
    m = re.search(r'\$vbsContent\s*=\s*@"\r?\n(.*?)\r?\n"@', text, re.S)
    assert m, "could not locate the $vbsContent here-string in install.ps1"
    return m.group(1)


def test_install_ps1_present():
    assert INSTALL_PS1.is_file(), f"missing {INSTALL_PS1}"


def test_vbs_does_not_pass_windowstyle_hidden():
    vbs = _vbs_block()
    assert "-WindowStyle Hidden" not in vbs, (
        "launch-studio.vbs must not pass -WindowStyle Hidden to PowerShell: the "
        "window is already hidden by shell.Run(cmd, 0, False); the redundant flag "
        "only adds the hidden-PowerShell token that AV heuristics flag."
    )


def test_vbs_stays_windowless_via_shell_run():
    vbs = _vbs_block()
    assert re.search(r"shell\.Run\s+cmd\s*,\s*0\s*,\s*False", vbs), (
        "launcher must remain windowless via shell.Run(cmd, 0, False) "
        "(intWindowStyle 0 = hidden)."
    )


def test_vbs_keeps_bypass_and_file_invocation():
    # Bypass lets the unsigned local .ps1 run under the default Restricted policy.
    vbs = _vbs_block()
    assert "-ExecutionPolicy Bypass" in vbs
    assert "-File" in vbs
    assert "powershell" in vbs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""Guard install.ps1's Unsloth launcher against the AV-heuristic shape (Kaspersky
HEUR:Trojan.VBS.Agent.gen): a WScript .vbs spawning a hidden ExecutionPolicy-Bypass PowerShell.
The shortcut must stay windowless via powershell.exe -WindowStyle Hidden over launch-studio.ps1,
never a .vbs/WScript.Shell.Run wrapper, and any pre-existing .vbs must be deleted on upgrade."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_PS1 = REPO_ROOT / "install.ps1"


def _text() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


def test_install_ps1_present():
    assert INSTALL_PS1.is_file(), f"missing {INSTALL_PS1}"


def test_no_vbs_launcher_generated():
    text = _text()
    # No here-string building a .vbs body, no .vbs written (legacy cleanup checked separately).
    assert "$vbsContent" not in text, (
        "install.ps1 must not generate a launch-studio.vbs: a WScript.Shell .vbs "
        "spawning a hidden ExecutionPolicy-Bypass PowerShell is the exact shape "
        "VBS-dropper heuristics flag (Kaspersky HEUR:Trojan.VBS.Agent.gen)."
    )
    assert 'CreateObject("WScript.Shell")' not in text
    assert "shell.Run" not in text
    assert not re.search(r"Set-Content\s+-LiteralPath\s+\$launcherVbs", text)
    assert "//B //Nologo" not in text


def test_legacy_vbs_removed_on_upgrade():
    # An upgrade must DELETE a pre-existing launch-studio.vbs, not just stop generating it, or AV keeps flagging the
    text = _text()
    assert re.search(
        r"Remove-Item\s+-LiteralPath\s+\$legacyLauncherVbs", text
    ), "upgrades must remove a pre-existing launch-studio.vbs so AV stops flagging it"


def test_shortcut_target_is_not_wscript():
    text = _text()
    assert "wscript.exe" not in text.lower()


def test_launcher_is_windowless_powershell():
    # The .lnk must not launch through wscript.exe (the VBS script host).
    # The shortcut runs powershell.exe -WindowStyle Hidden over launch-studio.ps1.
    text = _text()
    assert re.search(
        r"-WindowStyle\s+Hidden", text
    ), "the launcher must run powershell.exe with -WindowStyle Hidden over launch-studio.ps1."
    assert "launch-studio.ps1" in text


def test_hidden_window_never_pairs_with_execution_policy_bypass():
    # The pair AV detections key on;
    # install.rs already refuses it for the app's own launch.
    text = _text()
    for line in text.splitlines():
        if re.search(r"-WindowStyle\s+Hidden", line):
            assert not re.search(r"-ExecutionPolicy\s+Bypass", line), (
                "a hidden-window PowerShell launch must not also pass -ExecutionPolicy Bypass: "
                f"{line.strip()}"
            )
    assert re.search(
        r"-WindowStyle\s+Hidden\s+-ExecutionPolicy\s+RemoteSigned", text
    ), "the shortcut must launch launch-studio.ps1 under RemoteSigned, not Bypass."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

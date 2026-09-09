# SPDX-License-Identifier: AGPL-3.0-only
import sys
from types import SimpleNamespace

from core.inference import srt_setup


def test_windows_setup_uses_selected_interpreter_without_tool_replay(monkeypatch):
    commands = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(srt_setup.srt_probe, "invalidate_cache", lambda **kwargs: None)
    monkeypatch.setattr(
        srt_setup.subprocess,
        "run",
        lambda command, **kwargs: (commands.append(command) or SimpleNamespace(returncode = 0)),
    )
    result = srt_setup.install_windows_sandbox()
    assert result["status"] == "installed"
    assert len(commands) == 1
    assert commands[0][0] == sys.executable
    assert commands[0][-1] == "--windows-install"
    assert "no command will be retried" in result["message"]


def test_setup_does_not_spawn_on_linux_or_macos(monkeypatch):
    monkeypatch.setattr(
        srt_setup.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected setup process")),
    )
    for platform in ("linux", "darwin"):
        monkeypatch.setattr(sys, "platform", platform)
        assert srt_setup.install_windows_sandbox()["status"] == "unavailable"

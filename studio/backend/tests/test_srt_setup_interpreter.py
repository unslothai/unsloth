# SPDX-License-Identifier: AGPL-3.0-only
import sys
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("exit_code", [0, 1])
def test_windows_setup_installs_srt_and_propagates_failure(tmp_path, exit_code):
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        pytest.skip("PowerShell is required to execute the Windows setup hook")
    source = (Path(__file__).parents[2] / "setup.ps1").read_text(encoding = "utf-8")
    hook = re.search(r"function Install-WindowsSrt \{.*?^\}", source, re.M | re.S).group()
    installer = tmp_path / "installer.py"
    installer.write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(" + str(exit_code) + ")\n",
        encoding = "utf-8",
    )
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    harness = (
        "$ErrorActionPreference = 'Stop'\n"
        "function step { param($name, $message) }\n"
        "function Exit-SetupFailure { param($message) throw $message }\n"
        + hook
        + "\nInstall-WindowsSrt -PythonExe "
        + quote(sys.executable)
        + " -InstallerPath "
        + quote(installer)
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-Command", harness], capture_output = True, text = True
    )
    assert json.loads(result.stdout.strip()) == ["--windows-install"]
    assert (result.returncode == 0) == (exit_code == 0)
    if exit_code:
        assert "Windows sandbox setup failed" in result.stderr


@pytest.mark.parametrize(
    "ready,force,provisions", [(True, False, False), (False, False, True), (True, True, True)]
)
def test_installer_preserves_working_windows_configuration(
    monkeypatch, tmp_path, ready, force, provisions
):
    path = Path(__file__).parents[2] / "install_srt_runtime.py"
    spec = importlib.util.spec_from_file_location("srt_installer_test", path)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    monkeypatch.setattr(sys, "platform", "win32")
    cli = tmp_path / "node_modules/npm/bin/npm-cli.js"
    cli.parent.mkdir(parents = True)
    cli.write_text("", encoding = "utf-8")
    monkeypatch.setattr(installer.shutil, "which", lambda name: str(tmp_path / name))
    monkeypatch.setattr(installer, "_existing_windows_ready", lambda root: ready)
    ranges = []
    monkeypatch.setattr(
        installer, "_select_windows_range", lambda *a, **k: ranges.append(1) or None
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode = 0, stdout = "v24.9.0")

    monkeypatch.setattr(installer.subprocess, "run", run)
    assert installer.install(windows_install = True, windows_force = force) == 0
    installs = [command for command in commands if "windows-install" in command]
    assert len(installs) == int(provisions)
    assert len(ranges) == int(provisions)
    if installs:
        assert ("--force" in installs[0]) == force


@pytest.mark.parametrize("ready,cleanup_fails", [(True, False), (False, False), (True, True)])
def test_installer_reuse_requires_probe_and_cleanup(tmp_path, ready, cleanup_fails):
    path = Path(__file__).parents[2] / "install_srt_runtime.py"
    spec = importlib.util.spec_from_file_location("srt_installer_probe_test", path)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    root = tmp_path / "backend/core/inference/srt_runtime"
    root.mkdir(parents = True)
    (root.parent / "srt_probe.py").write_text(
        f"def probe(*, force):\n    assert force\n    return {ready}, 'test'\n",
        encoding = "utf-8",
    )
    marker = tmp_path / "cleanup.txt"
    (root.parent / "srt_windows_read_lease.py").write_text(
        "from pathlib import Path\ndef shutdown():\n"
        f"    Path({str(marker)!r}).write_text('closed')\n"
        + ("    raise RuntimeError('cleanup failed')\n" if cleanup_fails else ""),
        encoding = "utf-8",
    )
    assert installer._existing_windows_ready(root) == (ready and not cleanup_fails)
    assert marker.read_text() == "closed"

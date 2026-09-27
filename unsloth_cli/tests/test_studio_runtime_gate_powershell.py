# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for Unsloth's PowerShell resolution and launch arguments."""

import ntpath
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unsloth_cli._studio_runtime_gate import resolve_windows_powershell  # noqa: E402


@pytest.mark.skipif(sys.platform != "win32", reason = "requires Windows PowerShell")
def test_resolves_through_path_when_available(monkeypatch):
    monkeypatch.setenv("SystemRoot", os.environ.get("SystemRoot", r"C:\Windows"))
    resolved = resolve_windows_powershell()
    assert ntpath.isabs(resolved)
    assert os.path.isfile(resolved)


def test_falls_back_to_the_builtin_location_when_path_lacks_powershell(monkeypatch, tmp_path):
    # Simulate the stripped PATH from a GUI launch.
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("SystemRoot", os.environ.get("SystemRoot", r"C:\Windows"))

    resolved = resolve_windows_powershell()

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
    assert resolve_windows_powershell() == "powershell.exe"


# The callers. Resolving in the gate alone does not fix #9440: setup() and update() both run the
# gate and then hand off to PowerShell again, so every spawn on that path has to use the resolver
# or the install dies at the next one with the same WinError 2.

# ── the callers ────────────────────────────────────────────────────────────────────
_RESOLVED = ntpath.join(r"C:\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe")


def _windows_studio(monkeypatch):
    from unsloth_cli.commands import studio

    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        studio._studio_runtime_gate, "resolve_windows_powershell", lambda: _RESOLVED
    )
    return studio


class _Process:
    returncode = 0

    def wait(self):
        return 0


def test_the_setup_handoff_spawns_the_resolved_interpreter(monkeypatch, tmp_path):
    """_run_setup_script is the gate's own next line, and its Popen has no OSError handler."""
    studio = _windows_studio(monkeypatch)
    monkeypatch.setattr(studio, "_probe_profile_proxy_defaults", lambda hosts: None)
    repo_root = tmp_path / "repo"
    (repo_root / "studio").mkdir(parents = True)
    (repo_root / "studio" / "setup.ps1").write_text("")

    spawned = []
    monkeypatch.setattr(
        studio.subprocess, "Popen", lambda argv, **kw: spawned.append(list(argv)) or _Process()
    )

    studio._run_setup_script(repo_root = repo_root)

    # The handoff, not merely the first spawn: _run_setup_script asks uv where its cache
    # is before it hands over, and subprocess.run is built on Popen, so that probe lands
    # here too. What this test is about is which interpreter the handoff itself uses.
    handoffs = [argv for argv in spawned if argv and argv[-1].endswith("*>&1")]
    assert len(handoffs) == 1, spawned
    assert handoffs[0][0] == _RESOLVED, spawned


def test_the_profile_probe_falls_back_to_the_resolved_interpreter(monkeypatch, tmp_path):
    """_profile_probe_hosts() keeps only hosts shutil.which finds, so the PATH that broke the
    gate empties it and the fallback is the only host the proxy probe gets."""
    studio = _windows_studio(monkeypatch)
    monkeypatch.setattr(studio, "_profile_probe_hosts", list)
    monkeypatch.delenv("_UNSLOTH_PS_PROXY_DEFAULTS", raising = False)
    repo_root = tmp_path / "repo"
    (repo_root / "studio").mkdir(parents = True)
    (repo_root / "studio" / "setup.ps1").write_text("")

    probed = []
    monkeypatch.setattr(
        studio, "_probe_profile_proxy_defaults", lambda hosts: probed.append(list(hosts))
    )
    monkeypatch.setattr(studio.subprocess, "Popen", lambda argv, **kw: _Process())

    studio._run_setup_script(repo_root = repo_root)

    assert probed == [[_RESOLVED]], probed


def test_the_launcher_refresh_spawns_the_resolved_interpreter(monkeypatch, tmp_path):
    """update() ends in _refresh_desktop_shortcuts. It degrades instead of crashing, so a bare
    name silently drops the refresh -- after fetching an installer it cannot launch either."""
    studio = _windows_studio(monkeypatch)
    installer = tmp_path / "install.ps1"
    installer.write_text("")
    monkeypatch.setattr(studio, "_installers_on_disk", lambda candidates: [installer])
    monkeypatch.setattr(
        studio,
        "_fetch_installer",
        lambda *a, **k: pytest.fail("a launchable installer was on disk"),
    )

    spawned = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(
        studio.subprocess, "run", lambda argv, **kw: spawned.append(list(argv)) or _Result()
    )

    studio._refresh_desktop_shortcuts()

    assert spawned and spawned[0][0] == _RESOLVED, spawned


@pytest.mark.parametrize("interactive", [False, True], ids = ["redirected", "console"])
@pytest.mark.parametrize("flow", ["setup", "local-refresh", "fetched-refresh"])
def test_windows_launch_uses_process_flags_without_windowstyle(
    monkeypatch, tmp_path, interactive, flow
):
    """Cover Python argv for test_installer_av_shapes.py's Hidden/Bypass rule.

    Inspect the actual handoffs, including refresh's separate -Command/-File paths;
    the shell-script scan cannot see arguments assembled across Python functions.
    """
    studio = _windows_studio(monkeypatch)
    monkeypatch.setattr(studio.sys.stdout, "isatty", lambda: interactive)
    monkeypatch.setattr(studio.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising = False)
    monkeypatch.setattr(studio, "_with_studio_uv_cache", lambda env, **kw: env)
    monkeypatch.setattr(studio, "_backfill_uv_cache_marker", lambda env: None)
    monkeypatch.setattr(studio, "_probe_profile_proxy_defaults", lambda hosts: None)

    repo_root = tmp_path / "owner's repo"
    (repo_root / "studio").mkdir(parents = True)
    setup_script = repo_root / "studio" / "setup.ps1"
    setup_script.write_text("", encoding = "utf-8")
    installer = repo_root / "install.ps1"
    installer.write_text("", encoding = "utf-8")
    fetched = b"Write-Output 'refresh'"
    spawned = []

    def capture(argv, **kwargs):
        spawned.append((list(argv), kwargs))
        if "-File" in argv:
            path = Path(argv[argv.index("-File") + 1])
            assert path.read_bytes() == b"\xef\xbb\xbf" + fetched
        return _Process()

    monkeypatch.setattr(studio.subprocess, "Popen", capture)
    monkeypatch.setattr(studio.subprocess, "run", capture)
    if flow == "setup":
        studio._run_setup_script(repo_root = repo_root)
    else:
        monkeypatch.setattr(
            studio,
            "_installers_on_disk",
            lambda candidates: [installer] if flow == "local-refresh" else [],
        )
        monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **kw: fetched)
        studio._refresh_desktop_shortcuts()

    assert len(spawned) == 1
    argv, kwargs = spawned[0]
    assert argv[0] == _RESOLVED
    assert "-NoProfile" in argv
    assert "-WindowStyle" not in argv
    assert argv[argv.index("-ExecutionPolicy") + 1] == "Bypass"
    if interactive:
        assert "-NonInteractive" not in argv
        assert "creationflags" not in kwargs
        assert "startupinfo" not in kwargs
    else:
        assert "-NonInteractive" in argv
        assert "-NoLogo" in argv
        assert kwargs["creationflags"] & 0x08000000
    if flow == "fetched-refresh":
        script_path = Path(argv[argv.index("-File") + 1])
        assert argv[-1] == "--shortcuts-only"
        assert not script_path.exists()
    else:
        script = setup_script if flow == "setup" else installer
        quoted = str(script).replace("'", "''")
        command = argv[argv.index("-Command") + 1]
        assert f"& '{quoted}'" in command
        assert command.endswith("*>&1")
        if flow == "local-refresh":
            assert "--shortcuts-only" in command
        else:
            assert "stdout" in kwargs
            assert "stderr" in kwargs

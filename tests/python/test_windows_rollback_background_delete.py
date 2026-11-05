# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""install.ps1 deletes a committed venv rollback in a separate process.

The foreground delete cost 30-40 s after "Setup Complete" on a Studio-only update. These
pin that the commit returns at once, prints nothing new, does not hold a caller's output
pipe open (the desktop app reads the installer's stdout to EOF), that the tree is gone
shortly after, that a later run's sweep never collides with a running deletion, and that
the sweep removes what a stopped deletion left behind.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]
FUNCTIONS = (
    "Test-StudioPathPresent",
    "Remove-StudioVenvTreeWithRetry",
    "Test-StudioVenvRollbackMustBePreserved",
    "Remove-StaleStudioVenvRollbacks",
    "Start-StudioVenvRollbackBackgroundDelete",
    "Complete-StudioVenvRollback",
)
SWEEP_NAME = re.compile(r"^unsloth_studio\.rollback\.[0-9]{14}\.([0-9]+)(?:\.[0-9]+)?$")

windows_only = pytest.mark.skipif(
    os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell behaviour"
)


def _blocks() -> str:
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    out = []
    for name in FUNCTIONS:
        match = re.search(rf"    function {name} \{{.*?\n    \}}\n", source, flags = re.DOTALL)
        assert match is not None, f"install.ps1 function not found: {name}"
        out.append(match.group(0))
    return "".join(out)


def _prelude() -> str:
    return (
        '$ErrorActionPreference = "Stop"\n'
        # Every line the helpers print goes through these two, so the captured stdout is
        # exactly what the installer would have shown.
        'function substep { param([string]$Text, [string]$Color) Write-Output ("SUBSTEP " + $Text) }\n'
        "function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) "
        'Write-Output ("LINE " + $Message) }\n'
        "$StudioHome = $env:TEST_STUDIO_HOME\n" + _blocks()
    )


def _run(
    shell: str,
    script: str,
    home: Path,
    timeout: int = 120,
) -> tuple[str, float]:
    env = os.environ.copy()
    env["TEST_STUDIO_HOME"] = str(home)
    started = time.monotonic()
    # Pipes on both streams, as the desktop app spawns the installer: an inherited handle in
    # the deleter would keep this call blocked until the deletion ended.
    result = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", _prelude() + script],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = env,
        timeout = timeout,
    )
    wall = time.monotonic() - started
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout, wall


def _make_tree(root: Path, dirs: int, files_per_dir: int) -> None:
    for d in range(dirs):
        sub = root / "Lib" / "site-packages" / f"pkg{d}"
        sub.mkdir(parents = True)
        for f in range(files_per_dir):
            (sub / f"m{f}.py").write_bytes(b"x = 1\n")
    (root / "pyvenv.cfg").write_text("home = C:\\nowhere\n", encoding = "utf-8")


def _rollbacks(home: Path) -> list[Path]:
    return sorted(p for p in home.iterdir() if p.name.startswith("unsloth_studio.rollback."))


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


COMMIT = """
$script:StudioVenvRollbackActive = $true
$script:StudioVenvRollbackDir = $env:TEST_BACKUP
$script:StudioVenvRollbackPartial = $false
$script:StudioVenvRollbackPreserve = $false
$sw = [System.Diagnostics.Stopwatch]::StartNew()
Complete-StudioVenvRollback
$sw.Stop()
Write-Output ("MARK commit_ms=" + [int]$sw.Elapsed.TotalMilliseconds)
Write-Output ("MARK active=" + $script:StudioVenvRollbackActive + " committed=" + $script:StudioInstallCommitted)
Write-Output ("MARK original_present=" + (Test-Path -LiteralPath $env:TEST_BACKUP))
"""


def test_commit_disarms_restore_before_handing_off_the_delete():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    body = re.search(r"    function Complete-StudioVenvRollback \{.*?\n    \}\n", source, re.DOTALL)
    assert body is not None
    text = body.group(0)
    handoff = text.index("Start-StudioVenvRollbackBackgroundDelete -Path $backup")
    for disarm in (
        "$script:StudioInstallCommitted = $true",
        "$script:StudioVenvRollbackActive = $false",
        "$script:StudioVenvRollbackDir = $null",
    ):
        assert text.index(disarm) < handoff, disarm
    # The foreground delete stays as the fallback.
    assert text.index("Remove-StudioVenvTreeWithRetry -Path $backup") > handoff


@windows_only
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_rollback_is_deleted_in_the_background(tmp_path: Path, shell: str):
    home = tmp_path / "studio"
    home.mkdir()
    backup = home / "unsloth_studio.rollback.20260101000000.4000001"
    _make_tree(backup, dirs = 200, files_per_dir = 100)
    twin = tmp_path / "twin"
    shutil.copytree(backup, twin)

    # Baseline: what the commit cost before, on an identical tree.
    out, _ = _run(
        shell,
        """
$sw = [System.Diagnostics.Stopwatch]::StartNew()
Remove-StudioVenvTreeWithRetry -Path $env:TEST_TWIN -Label "environment rollback" | Out-Null
$sw.Stop()
Write-Output ("MARK foreground_ms=" + [int]$sw.Elapsed.TotalMilliseconds)
""".replace("$env:TEST_TWIN", f"'{twin}'"),
        home,
    )
    foreground_ms = int(re.search(r"foreground_ms=(\d+)", out).group(1))
    assert not twin.exists()

    os.environ["TEST_BACKUP"] = str(backup)
    try:
        out, wall = _run(shell, COMMIT, home)
    finally:
        os.environ.pop("TEST_BACKUP", None)
    returned_at = time.monotonic()
    parked = _rollbacks(home)
    commit_ms = int(re.search(r"commit_ms=(\d+)", out).group(1))
    print(f"[{shell}] foreground_ms={foreground_ms} commit_ms={commit_ms} wall_s={wall:.2f}")

    # No output beyond the markers: the installer prints exactly what it did before.
    assert [l for l in out.splitlines() if l and not l.startswith("MARK ")] == [], out
    assert "active=False committed=True" in out, out
    assert "original_present=False" in out, out
    # Returned while the deletion is still running, and the caller's pipe reached EOF first.
    assert len(parked) == 1, (parked, out)
    match = SWEEP_NAME.match(parked[0].name)
    assert match, parked[0].name
    deleter_pid = int(match.group(1))
    assert commit_ms < 5000, out

    # A later run sweeping now must not touch the tree while its deleter is alive.
    sweep_out, _ = _run(shell, "Remove-StaleStudioVenvRollbacks\nWrite-Output 'MARK swept'\n", home)
    assert [l for l in sweep_out.splitlines() if l and not l.startswith("MARK ")] == [], sweep_out

    deadline = returned_at + 300
    while time.monotonic() < deadline and parked[0].exists():
        time.sleep(0.5)
    gone_after = time.monotonic() - returned_at
    print(
        f"[{shell}] background delete finished {gone_after:.1f}s after return (pid {deleter_pid})"
    )
    assert not parked[0].exists(), "background delete did not finish in 300 s"
    assert _rollbacks(home) == []


@windows_only
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_stopped_delete_is_swept_by_the_next_run(tmp_path: Path, shell: str):
    home = tmp_path / "studio"
    home.mkdir()
    backup = home / "unsloth_studio.rollback.20260101000000.4000002"
    _make_tree(backup, dirs = 50, files_per_dir = 100)
    stale = home / f"unsloth_studio.rollback.20250101000000.{_dead_pid()}"
    _make_tree(stale, dirs = 2, files_per_dir = 5)
    live = home / f"unsloth_studio.rollback.20250101000000.{os.getpid()}"
    _make_tree(live, dirs = 1, files_per_dir = 1)

    # The deleter is in the installer's job; a job closed with kill-on-close stops it like this.
    script = (
        COMMIT
        + """
$parked = @(Get-ChildItem -LiteralPath $StudioHome -Directory -Force |
    Where-Object { $_.Name -like 'unsloth_studio.rollback.2026*' })
$deleterPid = [int]($parked[0].Name -split '\\.')[-1]
Stop-Process -Id $deleterPid -Force
Wait-Process -Id $deleterPid -Timeout 30 -ErrorAction SilentlyContinue
Write-Output ("MARK parked=" + $parked[0].FullName)
"""
    )
    os.environ["TEST_BACKUP"] = str(backup)
    try:
        out, _ = _run(shell, script, home)
    finally:
        os.environ.pop("TEST_BACKUP", None)
    parked = Path(re.search(r"MARK parked=(.+)", out).group(1).strip())
    assert parked.exists(), f"deleter finished before it was stopped; nothing to sweep: {out}"

    sweep_out, _ = _run(shell, "Remove-StaleStudioVenvRollbacks\n", home)
    assert not parked.exists(), sweep_out
    assert not stale.exists(), sweep_out
    assert live.exists(), "a rollback owned by a live process must survive the sweep"
    assert f"SUBSTEP removed stale environment rollback {parked.name}" in sweep_out, sweep_out
    assert "LINE [WARN]" not in sweep_out, sweep_out

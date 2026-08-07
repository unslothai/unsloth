# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Windows desktop setup log must be UTF-8 and must print each step once.

"Getting things ready..." used to produce::

      ?? Unsloth Studio Setup
      ????????????????????????????????????????????????????
      gpu
    none (chat-only / GGUF)
      gpu            none (chat-only / GGUF)

Encoding: 5.1 encodes redirected output with the OEM code page while the desktop
app decodes the pipe as UTF-8 (``from_utf8_lossy``, src-tauri/src/install.rs).
U+1F9A5 has no OEM form so PowerShell writes one ``?`` per UTF-16 surrogate;
U+2500 has one, so it becomes a bare 0xC4 and arrives as U+FFFD.

Duplication: ``step``/``substep`` wrote through Write-Host *and* a console-handle
mirror, and the CLI spawns setup.ps1 as ``-Command "& '...' *>&1"``, which merges
the Information stream into stdout.

Splitting: ``step`` built one line from two Write-Host calls with -NoNewline,
which a redirected consumer splits at the record boundary.

The byte-level tests assert on raw bytes; decoding first would hide the exact
regression being guarded.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
INSTALL_PS1 = REPO_ROOT / "install.ps1"
EXTRACTOR = REPO_ROOT / "tests" / "studio_setup_ps1" / "Get-FunctionSource.ps1"

SLOTH = "\U0001f9a5"
RULE_CHAR = "─"
REPLACEMENT = "�"

# Windows PowerShell 5.1 is what the desktop app spawns (install.rs pins
# System32\WindowsPowerShell\v1.0\powershell.exe). pwsh is a stand-in for the
# behaviour on other hosts; the OEM-code-page bug itself only reproduces on 5.1,
# which the staging Windows runner covers.
_PWSH = shutil.which("powershell") if sys.platform == "win32" else shutil.which("pwsh")
pwsh_only = pytest.mark.skipif(_PWSH is None, reason = "PowerShell is unavailable")


def _harness(redirected_probe: bool) -> str:
    """Emit a known banner + steps using the real helpers.

    Extracted from setup.ps1 rather than restated, so a change in shape fails
    here instead of drifting.
    """
    sink = "$true" if redirected_probe else "[Console]::IsOutputRedirected"
    return f"""
$ErrorActionPreference = 'Stop'
$_UnslothUtf8NoBom = New-Object System.Text.UTF8Encoding $false
try {{ [Console]::OutputEncoding = $_UnslothUtf8NoBom }} catch {{ }}
$OutputEncoding = $_UnslothUtf8NoBom

. '{EXTRACTOR.as_posix()}'
foreach ($fn in @('Get-StudioAnsi', 'Write-StudioStdoutMirror', 'step', 'substep')) {{
    $src = Get-FunctionSource -Path '{SETUP_PS1.as_posix()}' -Name $fn
    if (-not $src) {{ throw "missing $fn" }}
    . ([scriptblock]::Create($src))
}}
$script:StudioVtOk = $false
$script:StudioStdoutRedirected = {sink}

$Rule = [string]::new([char]0x2500, 52)
$Sloth = [char]::ConvertFromUtf32(0x1F9A5)
if ($script:StudioStdoutRedirected) {{
    [Console]::Out.WriteLine("  $Sloth Unsloth Studio Setup")
    [Console]::Out.WriteLine("  $Rule")
    [Console]::Out.Flush()
}} else {{
    Write-Host ("  " + $Sloth + " Unsloth Studio Setup")
    Write-Host ("  " + $Rule)
}}
step "gpu" "none (chat-only / GGUF)"
step "long paths" "enabled"
substep "installing OXC validator runtime..."
"""


def _run_capturing_bytes(script: str, use_command_shape: bool) -> bytes:
    """Run through a real pipe, in both launch shapes the product uses.

    ``-File`` is how the desktop app spawns the installer; ``-Command ... *>&1``
    is how the CLI spawns setup for ``unsloth studio update``. Piped stdout is
    required to reproduce, and is captured as bytes, never decoded here.
    """
    tmp = REPO_ROOT / "tests" / "python" / "_setup_output_probe.ps1"
    tmp.write_text(script, encoding = "utf-8")
    try:
        base = [_PWSH, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
        if use_command_shape:
            literal = str(tmp).replace("'", "''")
            argv = base + ["-Command", f"& '{literal}' *>&1"]
        else:
            argv = base + ["-File", str(tmp)]
        proc = subprocess.run(argv, stdout = subprocess.PIPE, stderr = subprocess.PIPE, timeout = 180)
        assert proc.returncode == 0, proc.stderr.decode("utf-8", errors = "replace")
        return proc.stdout
    finally:
        tmp.unlink(missing_ok = True)


@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_setup_output_is_valid_utf8(use_command_shape: bool) -> None:
    """Strict decode. Lossy decoding here would hide the exact regression."""
    raw = _run_capturing_bytes(_harness(redirected_probe = True), use_command_shape)
    text = raw.decode("utf-8")  # strict on purpose; UnicodeDecodeError is the failure
    assert REPLACEMENT not in text, "output contains U+FFFD (OEM bytes decoded as UTF-8)"


@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_banner_glyphs_survive_the_pipe(use_command_shape: bool) -> None:
    raw = _run_capturing_bytes(_harness(redirected_probe = True), use_command_shape)
    text = raw.decode("utf-8")
    assert text.count(SLOTH) == 1, "sloth emoji lost or duplicated"
    assert "??" not in text, "emoji was transcoded to '?' by a non-UTF-8 code page"
    assert RULE_CHAR * 52 in text, "the 52-char rule did not survive intact"


@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_every_step_appears_exactly_once(use_command_shape: bool) -> None:
    raw = _run_capturing_bytes(_harness(redirected_probe = True), use_command_shape)
    text = raw.decode("utf-8")
    for sentinel in ("none (chat-only / GGUF)", "enabled", "installing OXC validator runtime..."):
        assert (
            text.count(sentinel) == 1
        ), f"{sentinel!r} appeared {text.count(sentinel)} times, expected 1"


@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_step_label_and_value_stay_on_one_line(use_command_shape: bool) -> None:
    """The `gpu` / newline / `none (chat-only / GGUF)` split."""
    raw = _run_capturing_bytes(_harness(redirected_probe = True), use_command_shape)
    lines = raw.decode("utf-8").splitlines()
    matches = [line for line in lines if "gpu" in line]
    assert len(matches) == 1, f"expected one gpu line, got {matches!r}"
    assert matches[0] == "  gpu            none (chat-only / GGUF)"


# --------------------------------------------------------------------------
# Source contracts. These run everywhere, including the Linux backend CI job,
# so a regression is caught without waiting for a Windows runner.
# --------------------------------------------------------------------------


def _strip_comments(source: str) -> str:
    return re.sub(r"(?m)#.*$", "", source)


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_entry_scripts_set_the_utf8_invariant(path: Path) -> None:
    source = path.read_text(encoding = "utf-8")
    assert "[Console]::OutputEncoding = $_UnslothUtf8NoBom" in source
    assert "$env:PYTHONUTF8 = '1'" in source
    assert "$env:PYTHONIOENCODING = 'utf-8'" in source


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_entry_scripts_have_no_bom(path: Path) -> None:
    """5.1 parses BOM-less scripts as ANSI, so the fix stays ASCII-only.

    A BOM would be a far wider packaging change: these get concatenated and
    streamed through `irm | iex`.
    """
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_step_helper_emits_one_record(path: Path) -> None:
    """-NoNewline splits a logical line once a redirected consumer sees records."""
    source = _strip_comments(path.read_text(encoding = "utf-8"))
    match = re.search(r"(?m)^\s*function step\b", source)
    assert match, f"no step function in {path.name}"
    body = source[match.start() : match.start() + 2000]
    assert "-NoNewline" not in body


def test_setup_resolves_the_redirect_sink_once() -> None:
    source = SETUP_PS1.read_text(encoding = "utf-8")
    assert "$script:StudioStdoutRedirected = [Console]::IsOutputRedirected" in source


def test_refresh_environment_cannot_clobber_the_python_encoding_vars() -> None:
    """Refresh-Environment reloads the registry repeatedly through a long run.

    Without the guard a registry PYTHONUTF8=0 reloads over ours and every later
    Python child goes back to mojibake.
    """
    source = SETUP_PS1.read_text(encoding = "utf-8")
    assert "$key -eq 'PYTHONUTF8' -or $key -eq 'PYTHONIOENCODING'" in source


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_entry_scripts_bind_a_utf8_writer_when_there_is_no_console(path: Path) -> None:
    """The setter needs a console handle, and the desktop spawns us without one.

    It P/Invokes SetConsoleOutputCP and drops the cached writer BEFORE throwing,
    assigning OutputEncoding only after, so Console.Out would rebuild on the old
    code page. Swallowing the exception alone leaves redirected step/substep,
    whose only sink is Console.Out, still emitting locale-encoded bytes.
    """
    source = path.read_text(encoding = "utf-8")
    assert "[Console]::OpenStandardOutput()" in source
    assert "[Console]::SetOut(" in source


def test_update_command_uses_the_utf8_switch_not_just_env() -> None:
    """-I implies -E, so the isolated update child ignores every PYTHON* var.

    https://docs.python.org/3/using/cmdline.html#cmdoption-I
    """
    source = (REPO_ROOT / "studio" / "src-tauri" / "src" / "update.rs").read_text(encoding = "utf-8")
    assert '"-X", "utf8"' in source, "isolated Python child needs -X utf8, env vars are ignored"


@pytest.mark.parametrize(
    "rust_file",
    ["install.rs", "update.rs", "process.rs"],
)
def test_rust_windows_spawns_force_utf8(rust_file: str) -> None:
    """The Rust readers decode as UTF-8, so their Windows children must emit it."""
    source = (REPO_ROOT / "studio" / "src-tauri" / "src" / rust_file).read_text(encoding = "utf-8")
    assert 'cmd.env("PYTHONUTF8", "1");' in source, f"{rust_file} does not force PYTHONUTF8"
    assert (
        'cmd.env("PYTHONIOENCODING", "utf-8");' in source
    ), f"{rust_file} does not force PYTHONIOENCODING"

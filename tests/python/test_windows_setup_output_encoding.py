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

Sink: fixing ``step``/``substep`` left every other line on Write-Host, which
5.1's console host writes with its own OEM-code-page writer rather than the
UTF-8 one bound to ``[Console]::Out``. The banner and the footer are not steps,
so they kept arriving as U+FFFD. Both entry scripts now funnel through
``Write-StudioLine``, and Write-Host survives only inside helpers that have
already ruled out the redirected sink.

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

# The desktop app spawns Windows PowerShell 5.1; pwsh stands in elsewhere. The
# OEM-code-page bug only reproduces on 5.1, which the Windows runner covers.
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


def _section(source: str, title: str) -> str:
    """The statements under a ``# <title>`` box header, up to the blank line.

    Sliced out of setup.ps1 rather than restated, so the banner and the footer
    are exercised as written. A rewrite that drops them back onto Write-Host
    fails here.
    """
    match = re.search(rf"(?m)^# {re.escape(title)}\n#[^\n]*\n", source)
    assert match, f"no '{title}' section header in {SETUP_PS1.name}"
    body = source[match.end() :]
    return body[: body.index("\n\n")]


def _banner_footer_harness() -> str:
    """Print setup.ps1's real banner and footer with the redirected sink on."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    return f"""
$ErrorActionPreference = 'Stop'
$_UnslothUtf8NoBom = New-Object System.Text.UTF8Encoding $false
try {{ [Console]::OutputEncoding = $_UnslothUtf8NoBom }} catch {{ }}
$OutputEncoding = $_UnslothUtf8NoBom

. '{EXTRACTOR.as_posix()}'
foreach ($fn in @('Get-StudioAnsi', 'Write-StudioLine', 'Write-StudioStdoutMirror', 'step', 'substep')) {{
    $src = Get-FunctionSource -Path '{SETUP_PS1.as_posix()}' -Name $fn
    if (-not $src) {{ throw "missing $fn" }}
    . ([scriptblock]::Create($src))
}}
# No console handle under CREATE_NO_WINDOW, so the real run never takes the
# ANSI branch; pin it here instead of depending on the test host.
$script:StudioVtOk = $false
$script:StudioStdoutRedirected = $true
$script:LlamaCppDegraded = $false
$env:SKIP_STUDIO_BASE = '1'

$Rule = [string]::new([char]0x2500, 52)
{_section(source, "Banner")}
{_section(source, "Footer")}
"""


def _run_capturing_bytes(script: str, use_command_shape: bool, stem: str = "setup_output") -> bytes:
    """Run through a real pipe, in both launch shapes the product uses.

    ``-File`` is how the desktop app spawns the installer; ``-Command ... *>&1``
    is how the CLI spawns setup for ``unsloth studio update``. Piped stdout is
    required to reproduce, and is captured as bytes, never decoded here.
    """
    tmp = REPO_ROOT / "tests" / "python" / f"_{stem}_probe_{int(use_command_shape)}.ps1"
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


# The banner and the footer are the two blocks a user actually reads in the
# desktop setup log, and neither goes through step/substep, so they need their
# own byte-level coverage.


@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_banner_and_footer_are_valid_utf8(use_command_shape: bool) -> None:
    raw = _run_capturing_bytes(_banner_footer_harness(), use_command_shape, stem = "banner_footer")
    text = raw.decode("utf-8")  # strict on purpose
    assert REPLACEMENT not in text, "banner/footer contain U+FFFD (OEM bytes decoded as UTF-8)"
    assert "??" not in text, "emoji was transcoded to '?' by a non-UTF-8 code page"


@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_banner_and_footer_print_once_each(use_command_shape: bool) -> None:
    """One sink, so no line survives twice even when *>&1 merges the streams."""
    raw = _run_capturing_bytes(_banner_footer_harness(), use_command_shape, stem = "banner_footer")
    text = raw.decode("utf-8")
    assert text.count(SLOTH) == 1, "sloth emoji lost or duplicated"
    assert text.count(f"  {SLOTH} Unsloth Studio Setup") == 1
    assert text.count("  Unsloth Studio Setup Complete") == 1
    # One rule under the banner, two around the footer.
    assert text.count("  " + RULE_CHAR * 52) == 3, "the 52-char rule did not survive intact"


# Source contracts. These run everywhere, including the Linux backend CI job,
# so a regression is caught without waiting for a Windows runner.


def _strip_comments(source: str) -> str:
    return re.sub(r"(?m)#.*$", "", source)


def _mask_literals(source: str) -> str:
    """Blank comments and string literals, keeping every offset in place.

    A regex over the raw text would trip over the launcher script install.ps1
    builds in a here-string (it has its own Write-Host and no helper to call)
    and over the commented-out block in setup.ps1.
    """
    out = list(source)
    index, size = 0, len(source)

    def blank(start: int, stop: int) -> None:
        for k in range(start, stop):
            if out[k] != "\n":
                out[k] = " "

    while index < size:
        char = source[index]
        if source.startswith("<#", index):
            stop = source.find("#>", index + 2)
            stop = size if stop < 0 else stop + 2
        elif char == "@" and index + 1 < size and source[index + 1] in "\"'":
            terminator = "\n" + source[index + 1] + "@"
            stop = source.find(terminator, index + 2)
            stop = size if stop < 0 else stop + len(terminator)
        elif char == "#" and (index == 0 or source[index - 1] in " \t\r\n(){};,|=&"):
            stop = source.find("\n", index)
            stop = size if stop < 0 else stop
        elif char in "\"'":
            stop = index + 1
            while stop < size:
                if char == '"' and source[stop] == "`":
                    stop += 2
                    continue
                if source[stop] == char:
                    if stop + 1 < size and source[stop + 1] == char:
                        stop += 2
                        continue
                    stop += 1
                    break
                stop += 1
        else:
            index += 1
            continue
        blank(index, stop)
        index = stop
    return "".join(out)


def _function_span(masked: str, name: str) -> tuple[int, int]:
    """Offsets of `function <name> { ... }`, brace-matched over masked source."""
    match = re.search(r"(?im)^[ \t]*function\s+" + re.escape(name) + r"\b", masked)
    assert match, f"no function {name}"
    start = masked.index("{", match.end())
    depth = 0
    for offset in range(start, len(masked)):
        if masked[offset] == "{":
            depth += 1
        elif masked[offset] == "}":
            depth -= 1
            if depth == 0:
                return match.start(), offset + 1
    raise AssertionError(f"unbalanced braces in {name}")


# Write-Host may only appear inside a helper that has already ruled out the
# redirected sink: Write-StudioLine itself, and setup.ps1's step/substep, which
# return through the console mirror before reaching their interactive branch.
WRITE_HOST_ALLOW_LIST = {
    SETUP_PS1: ("Write-StudioLine", "step", "substep"),
    INSTALL_PS1: ("Write-StudioLine",),
}


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


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_write_host_only_survives_inside_gated_helpers(path: Path) -> None:
    source = path.read_text(encoding = "utf-8")
    masked = _mask_literals(source)
    spans = [_function_span(masked, name) for name in WRITE_HOST_ALLOW_LIST[path]]
    lines = source.splitlines()
    stray = [
        f"  {path.name}:{source.count(chr(10), 0, m.start()) + 1}: "
        f"{lines[source.count(chr(10), 0, m.start())].strip()}"
        for m in re.finditer(r"\bWrite-Host\b", masked)
        if not any(lo <= m.start() < hi for lo, hi in spans)
    ]
    assert not stray, (
        "Write-Host is written by 5.1's console host, not by the UTF-8 writer bound to "
        "[Console]::Out, so under CREATE_NO_WINDOW the desktop app renders these lines as "
        "U+FFFD. Call Write-StudioLine instead (same arguments, same colors when "
        "interactive):\n" + "\n".join(stray)
    )


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_every_allow_listed_helper_rules_out_the_redirected_sink(path: Path) -> None:
    """The allow-list is only safe while each entry still checks the sink."""
    masked = _mask_literals(path.read_text(encoding = "utf-8"))
    for name in WRITE_HOST_ALLOW_LIST[path]:
        lo, hi = _function_span(masked, name)
        assert "$script:StudioStdoutRedirected" in masked[lo:hi], (
            f"{name} in {path.name} reaches Write-Host without testing "
            "$script:StudioStdoutRedirected; drop it from the allow-list or gate it"
        )


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_the_sink_helper_is_defined_before_the_first_line_it_prints(path: Path) -> None:
    """PowerShell resolves functions at call time, but not before their line runs."""
    masked = _mask_literals(path.read_text(encoding = "utf-8"))
    definition = masked.index("function Write-StudioLine")
    first_call = min(m.start() for m in re.finditer(r"\bWrite-StudioLine\b", masked))
    assert first_call == definition + len("function "), (
        f"{path.name} calls Write-StudioLine before defining it"
    )


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_the_sink_helper_writes_through_the_console_handle(path: Path) -> None:
    masked = _mask_literals(path.read_text(encoding = "utf-8"))
    lo, hi = _function_span(masked, "Write-StudioLine")
    body = path.read_text(encoding = "utf-8")[lo:hi]
    assert "[Console]::Out.WriteLine($Message)" in body
    # Tauri reads line by line, so a buffered line is a line the user never sees.
    assert "[Console]::Out.Flush()" in body


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_entry_scripts_resolve_the_redirect_sink_once(path: Path) -> None:
    source = path.read_text(encoding = "utf-8")
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

    It drops the cached writer BEFORE throwing, assigning OutputEncoding only
    after, so Console.Out would rebuild on the old code page and redirected
    step/substep, whose only sink it is, would stay locale-encoded.
    """
    source = path.read_text(encoding = "utf-8")
    assert "[Console]::OpenStandardOutput()" in source
    assert "[Console]::SetOut(" in source
    # stderr is decoded identically and the failure text is built from it.
    assert "[Console]::OpenStandardError()" in source
    assert "[Console]::SetError(" in source


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

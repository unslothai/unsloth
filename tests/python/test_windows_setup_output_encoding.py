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
5.1's console host writes through its own console-attached writer rather than
the UTF-8 one bound to ``[Console]::Out``. The banner and the footer are not
steps, so they never entered the sink #8083 built. Both entry scripts now
funnel through ``Write-StudioLine``, and Write-Host survives only inside
helpers that have already ruled out the redirected sink.

No console: the transcode above needs a console to transcode against. Where
``CREATE_NO_WINDOW`` really leaves the child without one, which is the state
install.rs's own comment assumes, Write-Host has no screen buffer to query and
throws instead, taking the whole script down under ``-ErrorActionPreference
Stop``. The banner is then not mangled, it is absent. That is what
``test_banner_survives_a_console_less_spawn`` measures, and it is the only case
here that separates this fix from what shipped before it.

The byte-level tests assert on raw bytes; decoding first would hide the exact
regression being guarded.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
INSTALL_PS1 = REPO_ROOT / "install.ps1"
EXTRACTOR = REPO_ROOT / "tests" / "studio_setup_ps1" / "Get-FunctionSource.ps1"

SLOTH = "\U0001f9a5"
RULE_CHAR = "─"
REPLACEMENT = "�"

# The desktop app spawns Windows PowerShell 5.1;
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


def _run_capturing_bytes(
    script: str,
    use_command_shape: bool,
    stem: str = "setup_output",
) -> bytes:
    """Run through a real pipe, in both launch shapes the product uses.

    ``-File`` is how the desktop app spawns the installer; ``-Command ... *>&1``
    is how the CLI spawns setup for ``unsloth studio update``. Piped stdout is
    required to reproduce, and is captured as bytes, never decoded here.
    """
    # Unique per call:
    # Unique per call. The name used to be (stem, shape), which several tests share, so under pytest-xdist one case
    tmp = (
        REPO_ROOT
        / "tests"
        / "python"
        / f"_{stem}_probe_{int(use_command_shape)}_{uuid.uuid4().hex}.ps1"
    )
    tmp.write_text(script, encoding = "utf-8")
    try:
        base = [_PWSH, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
        if use_command_shape:
            literal = str(tmp).replace("'", "''")
            argv = base + ["-Command", f"& '{literal}' *>&1"]
        else:
            argv = base + ["-File", str(tmp)]
        # run_pwsh, not subprocess.run: the byte-level cases read this stdout as the setup log, and an interpreter that
        # aborted leaves an empty or truncated stream, which reads as the banner being mangled or lost.
        # See tests/_shared/unsloth_pwsh_runner.py.
        proc = run_pwsh(argv, stdout = subprocess.PIPE, stderr = subprocess.PIPE, timeout = 180)
        assert proc.returncode == 0, proc.stderr.decode("utf-8", errors = "replace")
        return proc.stdout
    finally:
        tmp.unlink(missing_ok = True)


@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_setup_output_is_valid_utf8(use_command_shape: bool) -> None:
    """Strict decode. Lossy decoding here would hide the exact regression."""
    raw = _run_capturing_bytes(_harness(redirected_probe = True), use_command_shape)
    text = raw.decode("utf-8")
    assert REPLACEMENT not in text, "output contains U+FFFD (OEM bytes decoded as UTF-8)"


@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_banner_glyphs_survive_the_pipe(use_command_shape: bool) -> None:
    raw = _run_capturing_bytes(_harness(redirected_probe = True), use_command_shape)
    text = raw.decode("utf-8")  # strict on purpose; UnicodeDecodeError is the failure
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


@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_banner_and_footer_are_valid_utf8(use_command_shape: bool) -> None:
    raw = _run_capturing_bytes(_banner_footer_harness(), use_command_shape, stem = "banner_footer")
    text = raw.decode("utf-8")
    assert REPLACEMENT not in text, "banner/footer contain U+FFFD (OEM bytes decoded as UTF-8)"
    assert "??" not in text, "emoji was transcoded to '?' by a non-UTF-8 code page"


# The banner and the footer are the two blocks a user actually reads in the desktop setup log, and neither goes through
@pwsh_only
@pytest.mark.parametrize("use_command_shape", [False, True], ids = ["-File", "-Command-merged"])
def test_banner_and_footer_print_once_each(use_command_shape: bool) -> None:
    """One sink, so no line survives twice even when *>&1 merges the streams."""
    raw = _run_capturing_bytes(_banner_footer_harness(), use_command_shape, stem = "banner_footer")
    text = raw.decode("utf-8")
    assert text.count(SLOTH) == 1, "sloth emoji lost or duplicated"
    assert text.count(f"  {SLOTH} Unsloth Studio Setup") == 1
    assert text.count("  Unsloth Studio Setup Complete") == 1
    assert text.count("  " + RULE_CHAR * 52) == 3, "the 52-char rule did not survive intact"


# so a regression is caught without waiting for a Windows runner.
# Source contracts. These run everywhere, including the Linux backend CI job, so a regression is caught without waiting
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


def _close_brace(masked: str, open_offset: int) -> int:
    """Offset of the `}` closing the `{` at `open_offset`, over masked source."""
    depth = 0
    for offset in range(open_offset, len(masked)):
        if masked[offset] == "{":
            depth += 1
        elif masked[offset] == "}":
            depth -= 1
            if depth == 0:
                return offset
    raise AssertionError("unbalanced braces")


def _function_span(masked: str, name: str) -> tuple[int, int]:
    """Offsets of `function <name> { ... }`, brace-matched over masked source."""
    match = _function_match(masked, name)
    assert match, f"no function {name}"
    return match.start(), _close_brace(masked, masked.index("{", match.end())) + 1


def _function_match(masked: str, name: str) -> re.Match[str] | None:
    return re.search(r"(?im)^[ \t]*function\s+" + re.escape(name) + r"\b", masked)


# Write-Host may only appear inside a helper that has already ruled out the redirected sink:
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
    assert first_call == definition + len(
        "function "
    ), f"{path.name} calls Write-StudioLine before defining it"


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


def test_managed_cli_command_uses_the_utf8_switch_not_just_env() -> None:
    """The managed CLI children must force UTF-8 on the command line.

    Every Windows spawn of the CLI now goes through build_managed_cli_command in
    process.rs, so that is where the switch has to be; update.rs only calls it.
    The env vars alone are not enough: a caller that already exports
    PYTHONIOENCODING wins over the ones set beside the spawn, and the Rust
    readers decode as UTF-8 regardless. -X utf8 is not overridable that way.

    https://docs.python.org/3/using/cmdline.html#cmdoption-X
    """
    source = (REPO_ROOT / "studio" / "src-tauri" / "src" / "process.rs").read_text(encoding = "utf-8")
    assert re.search(
        r'"-X"\s*,\s*"utf8"', source
    ), "the managed CLI child needs -X utf8, not just the env vars"


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


# Calling FreeConsole() in the child first produces the state install.rs assumes, and there the two diverge hard:

# The console-less spawn.
# A GitHub runner hands a CREATE_NO_WINDOW child a console anyway (GetConsoleOutputCP 437, GetConsoleWindow 0), so the
# UTF-8 setter in the preamble succeeds there and every version of these scripts emits a clean banner.
CREATE_NO_WINDOW = 0x08000000

# studio/src-tauri/src/install.rs::powershell_launch_args, minus the -File the runner appends.
TAURI_FLAGS = ["-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "RemoteSigned"]

# install.rs::powershell_exe.
# Resolved absolutely for the same reason it is there, and not fallen back to a bare `powershell.exe`: pwsh 7 is UTF-8
# by default and would pass this without exercising anything.
_WINDOWS_POWERSHELL = (
    Path(os.environ.get("SystemRoot", r"C:\Windows"))
    / r"System32\WindowsPowerShell\v1.0\powershell.exe"
    if sys.platform == "win32"
    else None
)

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason = "the console-less spawn is a Win32 state"
)
powershell_51_only = pytest.mark.skipif(
    _WINDOWS_POWERSHELL is None or not _WINDOWS_POWERSHELL.is_file(),
    reason = "Windows PowerShell 5.1 is unavailable",
)

# Documented kernel32 calls and nothing else, so the probe reaches the target state without the scripts under test
_FREE_CONSOLE = """Add-Type -Namespace Force -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern bool FreeConsole();
'@
$null = [Force.Native]::FreeConsole()"""

_UTF8_ENCODER = "$_UnslothUtf8NoBom = New-Object System.Text.UTF8Encoding $false"


def _dedent(block: str) -> str:
    """Strip the common indent; install.ps1 defines all of this inside a block."""
    indents = [len(line) - len(line.lstrip()) for line in block.split("\n") if line.strip()]
    cut = min(indents) if indents else 0
    return "\n".join(line[cut:] if line.strip() else "" for line in block.split("\n"))


def _slice_preamble(source: str) -> str:
    """The UTF-8 output invariant, from the encoder to the PYTHONIOENCODING line."""
    head = source.rindex("\n", 0, source.index(_UTF8_ENCODER)) + 1
    tail = source.index("\n", source.index("$env:PYTHONIOENCODING = 'utf-8'")) + 1
    return _dedent(source[head:tail])


def _slice_optional(source: str, pattern: str) -> str | None:
    """None is an answer: main's install.ps1 has neither probe nor sink helper."""
    match = re.search(pattern, source)
    return _dedent(match.group(0)) if match else None


def _slice_if_chain(source: str, masked: str, start: int) -> str:
    """A whole `if {} else {}`; brace-matching alone drops the non-ANSI branch.

    The redirected run is exactly the one that takes that branch.
    """
    end = _close_brace(masked, masked.index("{", start))
    chained = r"[ \t\r\n]*(?:elseif[ \t]*\(.*?\)|else)[ \t\r\n]*\{"
    while True:
        tail = re.match(chained, masked[end + 1 :], re.S)
        if not tail:
            return source[start : end + 1]
        end = _close_brace(masked, end + tail.end())


def _slice_banner(source: str, masked: str) -> str:
    """The blank line, the VT/plain branch and the trailing blank, verbatim."""
    match = re.search(
        r'(?m)^[ \t]*(?:Write-Host|Write-StudioLine) ""\n'
        r"[ \t]*if \(\$script:StudioVtOk -and -not \$env:NO_COLOR\) \{",
        source,
    )
    assert match, "no banner block"
    block = _slice_if_chain(source, masked, match.end() - 1)
    end = match.end() - 1 + len(block)
    trailing = re.match(r'\n[ \t]*(?:Write-Host|Write-StudioLine) ""(?=\n)', source[end:])
    return _dedent(source[match.start() : end + (trailing.end() if trailing else 0)])


def _console_less_probe(path: Path) -> str:
    """Assemble a probe out of the script's own preamble, helpers and banner."""
    source = path.read_text(encoding = "utf-8")
    masked = _mask_literals(source)
    # Sliced too: it is what turns the Write-Host throw into a dead script rather than a skipped line, so restating it
    eap = _slice_optional(source, r'(?m)^[ \t]*\$ErrorActionPreference = "Stop"')
    assert eap, f"{path.name} no longer stops on error before the banner"
    parts = [eap, _FREE_CONSOLE, "", _slice_preamble(source), ""]
    redirect_probe = _slice_optional(
        source,
        r"(?m)^[ \t]*\$script:StudioStdoutRedirected = \$false\n"
        r"[ \t]*try \{ \$script:StudioStdoutRedirected = \[Console\]::IsOutputRedirected \} catch \{ \}",
    )
    parts += [redirect_probe or "$script:StudioStdoutRedirected = $false", ""]
    for name in ("Write-StudioLine", "Enable-StudioVirtualTerminal", "Get-StudioAnsi"):
        if _function_match(masked, name):
            lo, hi = _function_span(masked, name)
            parts += [_dedent(source[lo:hi]), ""]
    parts += ["$script:StudioVtOk = Enable-StudioVirtualTerminal", ""]
    for pattern in (
        r"(?m)^[ \t]*\$Rule = \[string\]::new\(\[char\]0x2500, 52\)",
        r"(?m)^[ \t]*\$Sloth = \[char\]::ConvertFromUtf32\(0x1F9A5\)",
    ):
        # setup.ps1 inlines the sloth in the banner;
        # install.ps1 binds it first.
        assignment = _slice_optional(source, pattern)
        if assignment:
            parts.append(assignment)
    parts += ["", _slice_banner(source, masked), ""]
    # On stderr, which the app reads on a separate reader, so stdout stays exactly the byte stream the log panel is
    parts += [
        '[Console]::Error.WriteLine("psversion=" + $PSVersionTable.PSVersion.ToString())',
        '[Console]::Error.WriteLine("console_outputencoding_codepage=" + [Console]::OutputEncoding.CodePage)',
        '[Console]::Error.WriteLine("output_redirected=" + [Console]::IsOutputRedirected)',
        '[Console]::Error.WriteLine("studio_stdout_redirected=" + $script:StudioStdoutRedirected)',
        '[Console]::Error.WriteLine("studio_vt_ok=" + $script:StudioVtOk)',
    ]
    assembled = "\n".join(parts) + "\n"
    # 5.1 parses a BOM-less file as ANSI, which is why both scripts are ASCII-only.
    assembled.encode("ascii")
    return assembled


@lru_cache(maxsize = None)
def _run_console_less(path: Path, source: str | None = None) -> tuple[int, bytes, str]:
    """Spawn the probe the way install.rs spawns the installer, and read bytes.

    `source` is for the VT parity case, which runs this file's own function beside the one it
    replaced. A str keeps the lru_cache above workable; a dict would not hash.
    """
    with tempfile.TemporaryDirectory() as workdir:
        # A file written here has no Zone.Identifier, so RemoteSigned admits it.
        probe = Path(workdir) / f"{path.stem}_console_less_probe.ps1"
        text = _console_less_probe(path) if source is None else source
        probe.write_bytes(text.replace("\n", "\r\n").encode("ascii"))
        # run_pwsh, not subprocess.run: the console-less cases are phrased as "this run exited non-zero having printed
        # almost nothing", which is also what an aborted interpreter looks like, so the two must not be confused.
        # See tests/_shared/unsloth_pwsh_runner.py.
        proc = run_pwsh(
            [str(_WINDOWS_POWERSHELL), *TAURI_FLAGS, "-File", str(probe)],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            creationflags = CREATE_NO_WINDOW,
            timeout = 180,
        )
    return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", errors = "replace")


def _decode_like_install_rs(raw: bytes) -> str:
    """install.rs: read_until(b'\\n') -> trim_line_endings -> from_utf8_lossy.

    One record per `install-progress` event, so this is what the log panel
    renders. Python's 'replace' emits one U+FFFD per maximal subpart, the rule
    Rust's from_utf8_lossy uses.
    """
    records = raw.split(b"\n")
    if records and records[-1] == b"":
        records.pop()
    return "\n".join(r.rstrip(b"\r\n").decode("utf-8", errors = "replace") for r in records)


def _explain(path: Path, code: int, raw: bytes, err: str) -> str:
    tail = "\n".join(line for line in err.splitlines() if line.strip())[-1200:]
    return (
        f"\n{path.name} under a console-less CREATE_NO_WINDOW spawn: exit {code}, "
        f"{len(raw)} stdout bytes.\nstderr:\n{tail}\n"
    )


# A floor, not the exact byte counts:
# because that is the regression they guard, so they need this floor underneath
# A floor, not the 191 and 207 bytes these banners currently produce.
_MIN_BANNER_BYTES = 64


def _banner_or_explain(path: Path) -> tuple[str, str]:
    """Run the probe, insist the banner actually arrived, and decode it.

    Every console-less case starts here. `raw` being truthy is not enough: the
    aborted run is truthy too.
    """
    code, raw, err = _run_console_less(path)
    detail = _explain(path, code, raw, err)
    assert code == 0, (
        "the banner block aborted the script instead of printing. Write-Host needs a "
        "console screen buffer, and CREATE_NO_WINDOW is documented not to give the "
        "child one, so it throws and -ErrorActionPreference Stop takes the run down. "
        "The desktop setup log gets a PowerShell stack trace and no banner at all. "
        "Route the line through Write-StudioLine." + detail
    )
    assert len(raw) >= _MIN_BANNER_BYTES, (
        f"only {len(raw)} stdout bytes, under the {_MIN_BANNER_BYTES}-byte floor: the "
        "banner was LOST, not mangled. Exiting 0 having printed nothing is the same "
        "empty setup log to the user as throwing." + detail
    )
    return _decode_like_install_rs(raw), detail


@windows_only
@powershell_51_only
@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_banner_survives_a_console_less_spawn(path: Path) -> None:
    """Without the sink this exits 1 with 2 bytes: the banner never arrives."""
    _banner_or_explain(path)


@windows_only
@powershell_51_only
@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_console_less_banner_is_valid_utf8(path: Path) -> None:
    """Lossy first: it names how bad the stream is before the strict decode."""
    lossy, detail = _banner_or_explain(path)
    assert REPLACEMENT not in lossy, (
        "the desktop app decodes this pipe as UTF-8 and got bytes that are not, so the "
        "log shows U+FFFD. With no console the UTF-8 setter throws, and only the "
        "writers bound in its catch branch keep the stream UTF-8." + detail
    )
    # Strict on purpose; UnicodeDecodeError is the failure.
    _run_console_less(path)[1].decode("utf-8")


@windows_only
@powershell_51_only
@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_console_less_banner_keeps_its_glyphs(path: Path) -> None:
    text, detail = _banner_or_explain(path)
    assert SLOTH in text, "the sloth did not reach stdout" + detail
    assert "??" not in text, "the sloth was transcoded to '?' by a non-UTF-8 code page" + detail
    assert text.count(RULE_CHAR * 52) == 1, (
        f"expected one 52-char U+2500 rule, found {text.count(RULE_CHAR * 52)}" + detail
    )


# Sliced back out to rebuild the function this replaced, so parity is measured against the real predecessor.
_VT_FAST_PATH = re.compile(
    r"(?m)^[ \t]*# A redirected stdout is not a console.*?\n"
    r"(?:^[ \t]*#.*\n)*"
    r"^[ \t]*if \(\$script:StudioStdoutRedirected\) \{ return \$false \}\n"
)


def _probe_without_the_vt_fast_path(path: Path) -> str:
    probe = _console_less_probe(path)
    stripped, count = _VT_FAST_PATH.subn("", probe, count = 1)
    assert count == 1, (
        f"{path.name}: the VT fast path is not in the sliced probe in the shape this test "
        f"removes, so nothing was being compared. Update _VT_FAST_PATH."
    )
    return stripped


def _vt_verdict(err: str) -> str:
    for line in err.splitlines():
        if line.startswith("studio_vt_ok="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"the probe printed no studio_vt_ok line:\n{err}")


@windows_only
@powershell_51_only
@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_vt_fast_path_decides_exactly_as_the_compile_did(path: Path) -> None:
    """Skipping csc.exe must not change one byte the user sees.

    This probe is the changed branch, not a bystander: install.rs spawns with a pipe, so
    `$script:StudioStdoutRedirected` is true here and the early return is what runs. The
    reconstructed predecessor reaches Add-Type instead, and has to land on the same verdict.
    """
    new_code, new_raw, new_err = _run_console_less(path)
    old_code, old_raw, old_err = _run_console_less(
        path, source = _probe_without_the_vt_fast_path(path)
    )
    assert new_code == old_code == 0, (
        f"probe exit codes {new_code} (with the fast path) and {old_code} (without)"
        f"{_explain(path, new_code, new_raw, new_err)}"
    )
    assert _vt_verdict(new_err) == _vt_verdict(old_err) == "False", (
        f"a redirected stream cannot render VT: the fast path returned "
        f"{_vt_verdict(new_err)} where the compile returned {_vt_verdict(old_err)}"
    )
    assert new_raw == old_raw, (
        "the banner bytes moved. Same verdict in, same bytes out is the whole contract of "
        "this change" + _explain(path, new_code, new_raw, new_err)
    )

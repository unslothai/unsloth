# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Setup log must be UTF-8 and print each step once; redirected 5.1 output is OEM-encoded."""

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

# The OEM-code-page bug only reproduces on Windows PowerShell 5.1.
_PWSH = shutil.which("powershell") if sys.platform == "win32" else shutil.which("pwsh")
pwsh_only = pytest.mark.skipif(_PWSH is None, reason = "PowerShell is unavailable")


def _harness(redirected_probe: bool) -> str:
    """Extracted from setup.ps1, not restated, so a change in shape fails here instead of drifting."""
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
    """Slices the box-header section out of setup.ps1 so a rewrite back onto Write-Host fails here."""
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
    """Runs through a real pipe in both launch shapes (-File, -Command *>&1), returning undecoded bytes."""
    # Unique per call so xdist workers cannot unlink each other's script.
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
        # run_pwsh: an aborted interpreter would read as a mangled banner.
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
    assert text.count("  " + RULE_CHAR * 52) == 3, "the 52-char rule did not survive intact"


def _strip_comments(source: str) -> str:
    return re.sub(r"(?m)#.*$", "", source)


def _mask_literals(source: str) -> str:
    """Blanks comments and strings so a regex cannot match inside a here-string or commented-out code."""
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


# Write-Host is allowed only in helpers that already ruled out the redirected sink.
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
    """5.1 parses BOM-less scripts as ANSI, so the fix stays ASCII-only; a BOM would change packaging."""
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
    # Tauri reads line by line, so a buffered line is never seen.
    assert "[Console]::Out.Flush()" in body


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_entry_scripts_resolve_the_redirect_sink_once(path: Path) -> None:
    source = path.read_text(encoding = "utf-8")
    assert "$script:StudioStdoutRedirected = [Console]::IsOutputRedirected" in source


def test_refresh_environment_cannot_clobber_the_python_encoding_vars() -> None:
    """A registry PYTHONUTF8=0 reloaded by Refresh-Environment would clobber ours and restore mojibake."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    assert "$key -eq 'PYTHONUTF8' -or $key -eq 'PYTHONIOENCODING'" in source


@pytest.mark.parametrize("path", [SETUP_PS1, INSTALL_PS1], ids = ["setup.ps1", "install.ps1"])
def test_entry_scripts_bind_a_utf8_writer_when_there_is_no_console(path: Path) -> None:
    """The setter throws without a console, so the cached writer must be dropped before it, not after."""
    source = path.read_text(encoding = "utf-8")
    assert "[Console]::OpenStandardOutput()" in source
    assert "[Console]::SetOut(" in source
    assert "[Console]::OpenStandardError()" in source
    assert "[Console]::SetError(" in source


def test_managed_cli_command_uses_the_utf8_switch_not_just_env() -> None:
    """The CLI child needs -X utf8 in its command line; env vars lose to an exported PYTHONIOENCODING."""
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


# Windows only, opt-in off CI. Runners give CREATE_NO_WINDOW children a console anyway, so the
# child calls FreeConsole() itself (Add-Type of kernel32 trips AV, hence the opt-in).
# Not DETACHED_PROCESS: Windows PowerShell 5.1 spawned that way exits 0 without running a line.
CREATE_NO_WINDOW = 0x08000000

_CONSOLE_LESS_OPTED_IN = (
    os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("UNSLOTH_TEST_CONSOLE_LESS") == "1"
)

# install.rs::powershell_launch_args minus -File; RemoteSigned as the shipped spawn uses.
TAURI_FLAGS = ["-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "RemoteSigned"]

# Absolute path, no fallback: pwsh 7 is UTF-8 by default and would pass vacuously.
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
    """Brace matching alone drops the non-ANSI else branch; the redirected run is the one that takes it."""
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
    eap = _slice_optional(source, r'(?m)^[ \t]*\$ErrorActionPreference = "Stop"')
    assert eap, f"{path.name} no longer stops on error before the banner"
    parts = [eap, _FREE_CONSOLE, "", _slice_preamble(source), ""]
    redirect_probe = _slice_optional(
        source,
        r"(?m)^[ \t]*\$script:StudioStdoutRedirected = \$false\n"
        r"[ \t]*try \{ \$script:StudioStdoutRedirected = \[Console\]::IsOutputRedirected \} catch \{ \}",
    )
    parts += [redirect_probe or "$script:StudioStdoutRedirected = $false", ""]
    for name in (
        "Write-StudioLine",
        "Enable-StudioVirtualTerminal",
        "Get-StudioAnsi",
    ):
        if _function_match(masked, name):
            lo, hi = _function_span(masked, name)
            parts += [_dedent(source[lo:hi]), ""]
    parts += ["$script:StudioVtOk = Enable-StudioVirtualTerminal", ""]
    for pattern in (
        r"(?m)^[ \t]*\$Rule = \[string\]::new\(\[char\]0x2500, 52\)",
        r"(?m)^[ \t]*\$Sloth = \[char\]::ConvertFromUtf32\(0x1F9A5\)",
    ):
        assignment = _slice_optional(source, pattern)
        if assignment:
            parts.append(assignment)
    parts += ["", _slice_banner(source, masked), ""]
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
    """Spawns the probe as install.rs does and returns raw bytes; a str source keeps lru_cache usable."""
    if not _CONSOLE_LESS_OPTED_IN:
        pytest.skip(
            "the FreeConsole probe trips AV heuristics; set UNSLOTH_TEST_CONSOLE_LESS=1 to run it"
        )
    with tempfile.TemporaryDirectory() as workdir:
        # A file written here has no Zone.Identifier, so RemoteSigned admits it.
        probe = Path(workdir) / f"{path.stem}_console_less_probe.ps1"
        text = _console_less_probe(path) if source is None else source
        probe.write_bytes(text.replace("\n", "\r\n").encode("ascii"))
        proc = run_pwsh(
            [str(_WINDOWS_POWERSHELL), *TAURI_FLAGS, "-File", str(probe)],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            creationflags = CREATE_NO_WINDOW,
            timeout = 180,
        )
    return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", errors = "replace")


def _decode_like_install_rs(raw: bytes) -> str:
    """Decodes as install.rs does; Python's 'replace' emits U+FFFD per maximal subpart, as Rust does."""
    records = raw.split(b"\n")
    if records and records[-1] == b"":
        records.pop()  # read_until returning Ok(0) at EOF, not an empty line
    return "\n".join(r.rstrip(b"\r\n").decode("utf-8", errors = "replace") for r in records)


def _explain(path: Path, code: int, raw: bytes, err: str) -> str:
    tail = "\n".join(line for line in err.splitlines() if line.strip())[-1200:]
    return (
        f"\n{path.name} under a console-less CREATE_NO_WINDOW spawn: exit {code}, "
        f"{len(raw)} stdout bytes.\nstderr:\n{tail}\n"
    )


# A floor so a dead script (2 bytes) cannot pass the stream checks; not exact, to allow rewording.
_MIN_BANNER_BYTES = 64


def _banner_or_explain(path: Path) -> tuple[str, str]:
    """Requires the banner itself: a truthy raw output is not enough, since an aborted run is truthy too."""
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


# Removing the fast path leaves a bare SupportsVirtualTerminal read, which answers True when
# redirected; assert directly that the shipped code answers False.
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
def test_a_redirected_stream_is_answered_false_and_gets_no_escape_bytes(path: Path) -> None:
    """The branch the desktop app actually takes, asserted on the code that ships.

    install.rs spawns the installer with a pipe, so `$script:StudioStdoutRedirected` is true and
    the early return is what runs. Getting this wrong is not subtle: the Unsloth log panel is that
    pipe, and a True here puts raw escape sequences in front of a user.

    `$Host.UI.SupportsVirtualTerminal` does NOT answer this question on its own. It reports True on
    a redirected stream, which is why the redirect check is consulted first and why deleting it
    would be a user-visible regression rather than a tidy-up. That is measured below rather than
    asserted from the source: the fast path is removed and the remainder must disagree.
    """
    code, raw, err = _run_console_less(path)
    assert code == 0, f"the shipped probe exited {code}{_explain(path, code, raw, err)}"
    assert _vt_verdict(err) == "False", (
        f"a redirected stream was told it can render VT ({_vt_verdict(err)}). install.rs reads this "
        f"pipe, so the banner would arrive with escape sequences in it."
        + _explain(path, code, raw, err)
    )
    assert b"\x1b" not in raw, (
        "an escape byte reached the pipe even though the verdict was False, so something downstream "
        "of Enable-StudioVirtualTerminal is colouring unconditionally"
        + _explain(path, code, raw, err)
    )

    bare_code, _, bare_err = _run_console_less(path, source = _probe_without_the_vt_fast_path(path))
    assert bare_code == 0, f"the fast-path-less probe exited {bare_code}:\n{bare_err}"
    assert _vt_verdict(bare_err) == "True", (
        f"without the redirect check the property answered {_vt_verdict(bare_err)}, not True. The "
        f"early return in Enable-StudioVirtualTerminal is documented as load-bearing because the "
        f"property cannot see a redirected stream; if that has changed, update the comment there "
        f"and this test together.\n{bare_err}"
    )

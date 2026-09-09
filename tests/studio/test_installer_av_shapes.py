# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the shipped installers off the shapes antivirus heuristics score.

An AMSI provider blocked install.ps1 at parse time (#8523) and Microsoft flagged the Linux
AppImage `Trojan:Script/Wacatac.B!ml`. PowerShell hands the whole script block to AMSI before
running a line, so every byte counts, comments included.

Nothing here reproduces either verdict; it pins the constructs that were removed. The output
lock at the bottom is the other half: hardening must not change what a user sees.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]

PS_SCRIPTS = ("install.ps1", "studio/setup.ps1", "scripts/uninstall.ps1")
SH_SCRIPTS = ("install.sh", "studio/setup.sh")
ALL_SCRIPTS = PS_SCRIPTS + SH_SCRIPTS


def _text(name: str) -> str:
    return (REPO / name).read_text(encoding = "utf-8")


_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def _code_lines(name: str):
    """Lines reduced to what the script executes: no comments, here-strings or quoted literals.

    Most checks here scan the whole file, since AMSI does too. The ones about what the script
    *does* use this, so the printed remediation text does not read as an execution.
    """
    in_here_string = False
    for number, line in enumerate(_text(name).splitlines(), start = 1):
        stripped = line.strip()
        if in_here_string:
            if stripped in ("'@", '"@'):
                in_here_string = False
            continue
        if re.search(r"@[\"']$", stripped):
            in_here_string = True
            continue
        if stripped.startswith("#"):
            continue
        yield number, _QUOTED.sub('""', line)


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_script_exists(name: str) -> None:
    assert (REPO / name).is_file(), f"missing {name}"


@pytest.mark.parametrize("name", PS_SCRIPTS)
def test_no_remote_script_is_executed_in_process(name: str) -> None:
    # The construct AMSI and cloud ML scanners score hardest.
    for number, line in _code_lines(name):
        assert not re.search(
            r"Invoke-Expression\s*\(\s*Invoke-(RestMethod|WebRequest)", line
        ), f"{name}:{number} runs downloaded script text in-process: {line.strip()}"
        assert not re.search(
            r"\|\s*(iex|Invoke-Expression)\b", line
        ), f"{name}:{number} pipes into the engine: {line.strip()}"
        assert "scriptblock]::Create" not in line.lower().replace(
            " ", ""
        ), f"{name}:{number} builds a script block from a string: {line.strip()}"


@pytest.mark.parametrize("name", SH_SCRIPTS)
def test_no_remote_script_is_piped_into_a_shell_first(name: str) -> None:
    # The astral fallback stays reachable for unpinned hosts, but must never be tried first.
    text = _text(name)
    if "astral.sh/uv/install.sh" not in text:
        return
    pinned = min(
        (m.start() for m in re.finditer(r"_(setup_install_uv_pinned|uv_install_pinned)\b", text)),
        default = None,
    )
    fallback = text.index("astral.sh/uv/install.sh")
    assert pinned is not None, f"{name} has no pinned uv path"
    assert pinned < fallback, f"{name} reaches the piped fallback before the pinned release"


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_no_encoded_or_base64_command_payloads(name: str) -> None:
    text = _text(name)
    for banned in ("-EncodedCommand", "FromBase64String", "base64 -d", "base64 --decode"):
        assert banned not in text, f"{name} contains {banned}"


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_a_hidden_window_never_pairs_with_a_bypassed_policy(name: str) -> None:
    # Microsoft's detections key on this pair;
    # install.rs already refuses it for the app's own launch.
    # Python setup/refresh argv is exercised at the subprocess boundary by
    # unsloth_cli/tests/test_studio_runtime_gate_powershell.py::
    # test_windows_launch_uses_process_flags_without_windowstyle.
    for number, line in enumerate(_text(name).splitlines(), start = 1):
        if re.search(r"-WindowStyle\s+Hidden", line, re.IGNORECASE):
            assert not re.search(
                r"-ExecutionPolicy\s+Bypass", line, re.IGNORECASE
            ), f"{name}:{number} pairs a hidden window with a bypassed policy: {line.strip()}"


# Every native import left in the installers, however it is declared. Both scripts define theirs through reflection
# emit now, which costs no compile at all: install.ps1 the path resolver, the console thunk, the icon refresh and the
# process-image lookup, studio/setup.ps1 the console thunk. A new entry still needs a reason, and a PowerShell
# equivalent usually exists.
ALLOWED_PINVOKES = {
    # Canonicalising linked ancestors of security-relevant paths.
    # No PS 5.1 equivalent: ResolveLinkTarget is .NET 6+, and .Target misses a linked ancestor of a non-link leaf.
    # Not skippable either: Get-StudioRuntimePathHash hashes this spelling byte for byte and Python derives the same
    # mutex name from its own, so a GetFullPath fast path differing on case or an 8.3 name would let two installers each
    # believe they hold the lock.
    "CreateFileW",
    "GetFinalPathNameByHandleW",
    # ANSI colour on a real console.
    # Skipped entirely when stdout is redirected, see
    # test_virtual_terminal_answers_a_redirected_stream_without_compiling.
    "GetStdHandle",
    "GetConsoleMode",
    "SetConsoleMode",
    # Per-item Explorer icon refresh, standalone path only.
    # ie4uinit.exe -show is the global broadcast, which alone does not recover a stale .lnk, so it is not a substitute.
    "SHChangeNotify",
    # Naming the image behind a pid, so a venv Unsloth still has open is not overwritten.
    # PROCESS_QUERY_LIMITED_INFORMATION only, and it is the rung the others cannot replace: Process.Path goes through
    # MainModule, which needs PROCESS_VM_READ and is refused across users and across bitness, and Win32_Process needs a
    # working WMI service. Without this the scan can find nothing and let the install proceed over an open venv.
    "OpenProcess",
    "QueryFullProcessImageNameW",
    # Closing the handles CreateFileW and OpenProcess opened.
    "CloseHandle",
}


# The two ways a native import can be declared. Add-Type takes C# and runs csc.exe; DefinePInvokeMethod builds the
# same stub in memory. The second is invisible to a DllImport regex, so without this the inventory above would silently
# stop covering install.ps1 the moment it stopped compiling.
def _native_imports(text: str) -> set:
    imported = set()
    for match in re.finditer(
        r"DllImport\(\"[^\"]+\"[^)]*\)\][^;{]*?extern\s+[\w.\[\]]+\s+(\w+)", text
    ):
        imported.add(match.group(1))
    # install.ps1's multi-line declarations put the parameter list on later lines.
    for match in re.finditer(r"extern\s+[\w.<>\[\]]+\s+(\w+)\s*\(", text):
        imported.add(match.group(1))
    if "DefinePInvokeMethod" in text:
        for match in re.finditer(r"@\{\s*Name\s*=\s*\"(\w+)\"", text):
            imported.add(match.group(1))
    return imported


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_no_new_native_imports(name: str) -> None:
    text = _text(name)
    unexpected = _native_imports(text) - ALLOWED_PINVOKES
    assert not unexpected, (
        f"{name} imports {sorted(unexpected)} from native code. Prefer a PowerShell or .NET "
        f"equivalent; if there genuinely is none, add it to ALLOWED_PINVOKES with the reason."
    )


@pytest.mark.parametrize("name", ("install.ps1", "studio/setup.ps1"))
def test_virtual_terminal_answers_a_redirected_stream_without_defining_a_type(name: str) -> None:
    """The answer we already know must come first, before any native work at all.

    Only the redirected case is decided early, and it is decided FALSE. A redirected stdout is
    not a console, GetConsoleMode fails on a non-console handle, and the native path could
    only have returned false too. Anything that claimed VT here would put raw escape sequences
    in the Unsloth log panel, which is a pipe.

    This used to guard an Add-Type, back when the redirect check was the only thing keeping the
    desktop app off csc.exe. Nothing here compiles now, so the ordering is no longer load-bearing
    against a scanner; it is still the cheaper answer, and getting it wrong still corrupts the
    log panel.
    """
    text = _text(name)
    start = text.index("function Enable-StudioVirtualTerminal")
    call = re.compile(r"(?m)^[ \t]*\$null = New-StudioEmittedNativeType\b").search(text, start)
    assert call, f"{name} no longer emits the console thunk; update this guard"
    define_at = call.start()
    fast_path = text.index("if ($script:StudioStdoutRedirected) { return $false }", start)
    assert fast_path < define_at, (
        f"{name} builds the native console thunk before checking the stream: move the redirect "
        f"guard above it, since a redirected stream can never render VT anyway."
    )
    assert "$true" not in text[fast_path:define_at], (
        f"{name} returns something other than $false before the native work. The early answer is "
        f"only sound because a redirected stream can never render VT."
    )


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_no_process_memory_apis(name: str) -> None:
    # The installer reads image paths, nothing more.
    for banned in (
        "VirtualAllocEx",
        "WriteProcessMemory",
        "ReadProcessMemory",
        "CreateRemoteThread",
        "SetWindowsHookEx",
    ):
        assert banned not in _text(name), f"{name} references {banned}"


# What the installers print when they need the user to reinstall. Hardening must not touch user-visible output, and a
# search-and-replace would take exactly these out.
REQUIRED_OUTPUT = {
    "install.ps1": ['Write-StudioLine "          irm https://unsloth.ai/install.ps1 | iex"'],
    "studio/setup.ps1": ['Write-StudioLine "        irm https://unsloth.ai/install.ps1 | iex"'],
    "install.sh": ["curl -fsSL https://unsloth.ai/install.sh | sh"],
}


@pytest.mark.parametrize("name", sorted(REQUIRED_OUTPUT))
def test_printed_remediation_survives_the_hardening(name: str) -> None:
    text = _text(name)
    for snippet in REQUIRED_OUTPUT[name]:
        assert snippet in text, (
            f"{name} no longer prints {snippet!r}. Removing the one-liner from comments is the "
            f"point; removing it from what the user is told to run is a regression."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("name", ("install.ps1", "studio/setup.ps1"))
def test_the_installer_never_runs_the_c_sharp_compiler(name: str) -> None:
    """The desktop app spawns Windows PowerShell 5.1, which compiles Add-Type by writing C# to
    %TEMP% and running csc.exe. A GUI binary launching a windowless PowerShell that launches a
    compiler and drops a DLL in %TEMP% is a dropper's shape whatever the code says, and it was
    blocked in the field. Reflection emit builds the same stub in memory: no compiler process,
    no source on disk, no DLL, nothing in %TEMP%.

    Add-Type in full, not only -TypeDefinition: -MemberDefinition wraps its argument in a class
    and compiles that, so it reaches csc.exe by the same road. -AssemblyName is the exception,
    and the only one: it loads an assembly that already exists on disk and never reaches a
    compiler. Both scripts, not only the bundled one: leaving a compile anywhere means the answer
    to "does this run a compiler" depends on which entrypoint ran and whether an early return
    happened to come first, and a guard that holds only conditionally is what let this reach the
    field.
    """
    text = _text(name)
    hits = re.findall(r"(?m)^[ \t]*Add-Type\b(?![^\r\n]*-AssemblyName).*", text)
    assert not hits, (
        f"{name} compiles C# again ({len(hits)} Add-Type call(s), first: {hits[0].strip()!r}). "
        "Declare native methods with New-StudioEmittedNativeType instead; -MemberDefinition runs "
        "csc.exe just as -TypeDefinition does."
    )
    assert (
        "DefinePInvokeMethod" in text
    ), f"{name} no longer emits its native imports; update this guard"
    # The private-%TEMP% retry is gone with it. Redirecting TEMP to compile again after a block
    # cannot beat a filter driver, and "blocked writing an executable to TEMP, change TEMP, write
    # it again" is itself an evasion heuristic. Scoped to the resolver: Initialize-StudioTempEnvironment
    # legitimately redirects an unusable inherited TEMP, and that is a different thing.
    # Only install.ps1 has the path resolver; setup.ps1 emits the console thunk and nothing else.
    if "function Initialize-StudioFinalPathNativeType" not in text:
        return
    start = text.index("function Initialize-StudioFinalPathNativeType")
    body = text[start : text.index("\n    function ", start + 1)]
    assert (
        "$env:TMP" not in body and "$env:TEMP" not in body
    ), "the native resolver touches the temporary directory again; it should need nothing there"


def test_a_ci_lane_fails_when_a_compiler_actually_runs() -> None:
    """The behavioural half of the guard above.

    Reading the scripts cannot see a compile reached through a module, a dot-sourced
    file or a generated here-string, and it cannot see one a dependency performs while
    our process tree is what a scanner scores. Bitdefender scored the chain, not the
    bytes, so there has to be a lane that runs the installer and fails on the process.

    The positive control is the part worth asserting from here: a detector that sees
    nothing reads exactly like a clean run, and auditing can silently fail to apply. If
    the lane ever loses the control, every later green result stops meaning anything.
    """
    workflow = REPO / ".github" / "workflows" / "windows-no-compiler-ci.yml"
    assert workflow.is_file(), "the runtime guard lane is gone; the text check is alone again"
    body = workflow.read_text(encoding = "utf-8")
    assert "Positive control" in body, "the lane no longer proves its own detector works"
    assert (
        "Add-Type -TypeDefinition" in body
    ), "the positive control must really compile something; a simulated one proves nothing"

    watcher = REPO / ".github" / "scripts" / "Watch-ForCompiler.ps1"
    assert watcher.is_file()
    watcher_body = watcher.read_text(encoding = "utf-8")
    for image in ("csc.exe", "vbc.exe", "cvtres.exe"):
        assert image in watcher_body, f"the watcher no longer looks for {image}"
    # 4688 is what sees a compiler spawned at any depth; the temp sweep is what
    # survives auditing being overridden. Losing either leaves one detector.
    assert "4688" in watcher_body
    assert "*.cmdline" in watcher_body


_WATCHER = REPO / ".github" / "scripts" / "Watch-ForCompiler.ps1"

_FAKE_EVENTS = r"""
function New-FakeEvent {
    param([string]$Image, [string]$CommandLine)
    $xml = "<Event><EventData>" +
        "<Data Name='NewProcessName'>$Image</Data>" +
        "<Data Name='CommandLine'>$CommandLine</Data>" +
        "</EventData></Event>"
    $record = [pscustomobject]@{
        TimeCreated = [datetime]'2026-01-01T00:00:00Z'
        Message     = "New Process Name: $Image`nProcess Command Line: $CommandLine"
    }
    $body = [scriptblock]::Create("return @'`n$xml`n'@")
    return ($record | Add-Member -MemberType ScriptMethod -Name ToXml -Value $body -PassThru)
}
"""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")
@pytest.mark.parametrize(
    ("image", "command_line", "expected"),
    [
        (r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe", "csc.exe /out:x.dll", 1),
        (r"C:\Windows\System32\cmd.exe", "cmd.exe /c echo csc.exe", 0),
        (r"C:\Windows\System32\cmd.exe", r"cmd.exe /c copy a.txt C:\csc.exe.log", 0),
        (r"C:\Users\r\csc.exe.helper.exe", "whatever", 0),
    ],
    ids = ["a-real-compile", "a-command-line-mentioning-one", "a-path-argument", "a-similar-name"],
)
def test_the_watcher_scores_the_image_that_ran_not_the_words_in_the_message(
    tmp_path, image: str, command_line: str, expected: int
) -> None:
    """4688 renders the command line into the message, so a message search is not a detector.

    It scored `cmd.exe /c echo csc.exe` as a compile, and this job's whole output is a
    yes or no about whether a compiler ran under the installer. The record names the
    image it created in its own field; that is what gets read, and matched whole against
    the leaf name rather than as a substring.
    """
    script = tmp_path / "probe.ps1"
    script.write_text(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                f'. "{_WATCHER}"',
                _FAKE_EVENTS,
                f"$e = New-FakeEvent -Image '{image}' -CommandLine '{command_line}'",
                "$hits = Select-StudioCompilerHits -Events @($e)",
                'Write-Output "HITS:$($hits.Count)"',
            ]
        ),
        encoding = "utf-8",
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"HITS:{expected}" in result.stdout, result.stdout


def test_an_unreadable_security_log_is_void_rather_than_clean() -> None:
    """Get-WinEvent throws both for "nothing matched" and for "could not read".

    Swallowing both made a job that could not open the Security log print "no compiler"
    and pass. The positive control does not cover it: it runs in an earlier step, and
    says nothing about whether the log was still readable during the measurement.
    """
    body = _WATCHER.read_text(encoding = "utf-8")
    assert (
        "-MaxEvents 1" in body
    ), "the watcher no longer distinguishes an empty result from an unreadable log"
    assert "void rather than as clean" in body


def test_the_native_resolver_still_has_a_lexical_fallback() -> None:
    """The point of the change is the acquisition, not the ladder. A host where emit fails has to
    degrade exactly as a host that could not compile already did, which is a path the installer
    has always taken and still completes on.
    """
    text = _text("install.ps1")
    assert "Write-StudioFinalPathDegraded" in text
    assert "Get-StudioLexicalPath" in text
    # Constrained Language Mode forbids defining types at all, by emit as by Add-Type.
    assert '$languageMode -ne "FullLanguage"' in text

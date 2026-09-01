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
    for number, line in enumerate(_text(name).splitlines(), start = 1):
        if re.search(r"-WindowStyle\s+Hidden", line, re.IGNORECASE):
            assert not re.search(
                r"-ExecutionPolicy\s+Bypass", line, re.IGNORECASE
            ), f"{name}:{number} pairs a hidden window with a bypassed policy: {line.strip()}"


# Every runtime-compiled P/Invoke left in the installers.
# Each costs a csc.exe compile and is scored, so a new entry needs a reason;
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
    # PID -> image path for the venv-holder check.
    # Win32_Process answers the same question, but test_windows_installer_concurrency_guard.py bans it and $process.Path
    # there: the races #7764 closed came from inferring "in use" from anything but a confirmed executable identity.
    "OpenProcess",
    "QueryFullProcessImageNameW",
    "CloseHandle",
}


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_no_new_native_imports(name: str) -> None:
    text = _text(name)
    imported = set()
    for match in re.finditer(
        r"DllImport\(\"[^\"]+\"[^)]*\)\][^;{]*?extern\s+[\w.\[\]]+\s+(\w+)", text
    ):
        imported.add(match.group(1))
    # install.ps1's multi-line declarations put the parameter list on later lines.
    for match in re.finditer(r"extern\s+[\w.<>\[\]]+\s+(\w+)\s*\(", text):
        imported.add(match.group(1))
    unexpected = imported - ALLOWED_PINVOKES
    assert not unexpected, (
        f"{name} imports {sorted(unexpected)} from native code. Prefer a PowerShell or .NET "
        f"equivalent; if there genuinely is none, add it to ALLOWED_PINVOKES with the reason."
    )


@pytest.mark.parametrize("name", ("install.ps1", "studio/setup.ps1"))
def test_virtual_terminal_answers_a_redirected_stream_without_compiling(name: str) -> None:
    """Add-Type runs the C# compiler, so the answer we already know must come first.

    Only the redirected case is decided early, and it is decided FALSE. A redirected stdout is
    not a console, GetConsoleMode fails on a non-console handle, and the compiled path could
    only have returned false too. Anything that claimed VT here would put raw escape sequences
    in the Unsloth log panel, which is a pipe.
    """
    text = _text(name)
    start = text.index("function Enable-StudioVirtualTerminal")
    # The call, not the comment above it.
    call = re.compile(r"(?m)^[ \t]*Add-Type\b").search(text, start)
    assert call, f"{name} no longer compiles the console thunk; update this guard"
    compile_at = call.start()
    fast_path = text.index("if ($script:StudioStdoutRedirected) { return $false }", start)
    assert fast_path < compile_at, (
        f"{name} compiles C# for colour before checking the stream: move the redirect guard "
        f"above Add-Type, or every install spawns csc.exe again."
    )
    assert "$true" not in text[fast_path:compile_at], (
        f"{name} returns something other than $false before the compile. The early answer is "
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


# What the installers print when they need the user to reinstall. Hardening must not touch
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

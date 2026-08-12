# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the shipped installers off the shapes antivirus heuristics score.

Two false positives motivated this file. A third-party AMSI provider blocked install.ps1 at parse
time with ``ScriptContainedMaliciousContent`` (unsloth#8523) -- PowerShell hands the whole script
block to AMSI before running a line, so every byte of the file counts, comments included. And
Microsoft returned ``Trojan:Script/Wacatac.B!ml`` on the 0.1.701-beta Linux AppImage, a generic ML
verdict on script content.

None of the assertions here claim to reproduce either verdict; they pin the constructs that were
removed so they cannot drift back in. The counterpart to that is the output lock at the bottom:
hardening must never change what a user sees, so the remediation strings the installers print stay
verbatim.
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
    """Lines reduced to what the script actually executes.

    AMSI scans comments and string literals too, so most checks here run over the whole file.
    A few are about what the script *does* -- those use this, which drops comments, here-string
    bodies and quoted literals. The printed remediation text lives in exactly those literals and
    is deliberately unchanged, so it must not trip an "executes remote script" assertion.
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
    # Piping downloaded script text into the engine is the construct AMSI providers and cloud ML
    # scanners score hardest. install.ps1 and studio/setup.ps1 both fetch a pinned archive now.
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
    # The fallback in both files is deliberate and stays reachable, but it must never be what a
    # mainstream host actually takes -- the pinned path has to be tried first.
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
    # Microsoft's own detections key on this pair; studio/src-tauri/src/install.rs refuses it for
    # the app's own launch and the shortcut install.ps1 writes now matches.
    for number, line in enumerate(_text(name).splitlines(), start = 1):
        if re.search(r"-WindowStyle\s+Hidden", line, re.IGNORECASE):
            assert not re.search(
                r"-ExecutionPolicy\s+Bypass", line, re.IGNORECASE
            ), f"{name}:{number} pairs a hidden window with a bypassed policy: {line.strip()}"


# Every runtime-compiled P/Invoke still in the shipped installers, with why it stays. Each entry
# costs a csc.exe compile at install time and is scored by heuristics, so adding one needs a
# reason -- and a pure-PowerShell equivalent is almost always available.
ALLOWED_PINVOKES = {
    # Handle-based canonicalisation of linked ancestors, on security-relevant paths. No faithful
    # Windows PowerShell 5.1 equivalent: ResolveLinkTarget is .NET 6+, and Get-Item .Target does
    # not resolve a linked ancestor of a non-link leaf.
    "CreateFileW",
    "GetFinalPathNameByHandleW",
    # ANSI colour on legacy conhost.
    "GetStdHandle",
    "GetConsoleMode",
    "SetConsoleMode",
    # Per-item Explorer icon refresh; the global broadcast alone does not recover a stale .lnk.
    "SHChangeNotify",
    # Resolving a PID to its image path for the venv-holder check. Opening a handle per process
    # is a shape heuristics score, and Win32_Process.ExecutablePath answers the same question --
    # but tests/python/test_windows_installer_concurrency_guard.py bans Get-CimInstance and
    # $process.Path from that detector outright, because the races #7764 closed came from
    # inferring "in use" out of anything other than a confirmed executable identity. The contract
    # wins: a wrongly blocked install costs more than these three imports.
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
    # The multi-line declarations in install.ps1 need the parameter list skipped too.
    for match in re.finditer(r"extern\s+[\w.<>\[\]]+\s+(\w+)\s*\(", text):
        imported.add(match.group(1))
    unexpected = imported - ALLOWED_PINVOKES
    assert not unexpected, (
        f"{name} imports {sorted(unexpected)} from native code. Prefer a PowerShell or .NET "
        f"equivalent; if there genuinely is none, add it to ALLOWED_PINVOKES with the reason."
    )


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_no_process_memory_apis(name: str) -> None:
    # The installer reads a process's image path and nothing else. Reaching into another
    # process's memory has no legitimate use here and is what the injection heuristics look for.
    for banned in (
        "VirtualAllocEx",
        "WriteProcessMemory",
        "ReadProcessMemory",
        "CreateRemoteThread",
        "SetWindowsHookEx",
    ):
        assert banned not in _text(name), f"{name} references {banned}"


# The lines the installers print when they need the user to reinstall. Hardening must not touch
# user-visible output, and these are the ones a well-meaning search-and-replace would take out.
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

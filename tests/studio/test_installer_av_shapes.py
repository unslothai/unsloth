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

from unsloth_pwsh_runner import pwsh_env


REPO = Path(__file__).resolve().parents[2]

PS_SCRIPTS = ("install.ps1", "studio/setup.ps1", "scripts/uninstall.ps1")
SH_SCRIPTS = ("install.sh", "studio/setup.sh", "scripts/uninstall.sh")
# A shipped .bat is scanned like any other file, and studio/setup.bat launches PowerShell, so the
# same rules apply. Neither it nor scripts/uninstall.sh was in any list here, which is how
# setup.bat's `-ExecutionPolicy Bypass` survived the passes in #7822 and #8586: every guard below
# was reading a set of files that did not include it.
BAT_SCRIPTS = ("studio/setup.bat",)
ALL_SCRIPTS = PS_SCRIPTS + SH_SCRIPTS + BAT_SCRIPTS


# ---------------------------------------------------------------------------
# Why the installers are written the way they are, and which product flagged what
# ---------------------------------------------------------------------------
#
# This record lives here, in a test, rather than in the shipped scripts or in a doc.
#
# Not in the scripts, because PowerShell hands the entire top-level script block to AMSI at compile
# time, before the first statement runs -- so every byte of install.ps1 is classifier input, comments
# included. VirusTotal's analysis of the file quoted one of our own comments back as grounds for
# suspicion: "The presence of comments that suggest the script is designed to evade detection by
# security tools (e.g. 'AMSI scans this file in full before a line of it runs') further adds to the
# suspicion." Prose naming vendors and detection families raises the score of the very file it is
# trying to explain.
#
# Here, because this file already owns the guards that keep those shapes out, it ships to nobody, it
# is not packaged, and nothing scans it. The rule the tests below enforce is:
#
#     a comment in a shipped script says what the code does and what breaks if you change it;
#     this record says which scanner flagged which shape, and when.
#
# Nothing below is an evasion. Every decision makes the installer do strictly LESS than the shape it
# replaced: no compiler, no script engine, no remote script text, no cache purge on a no-op. The
# heuristics are not wrong about the shapes; they are wrong that we are an instance of them, and the
# fix is to stop having the shape.
AV_SHAPES_RECORD = r"""
## Measured detections

| Sample | Verdict |
|---|---|
| install.ps1 at 1ad44677d (the revision reported in #10805) | 1 malicious / 58 undetected: Skyhigh
  (Trellix / McAfee Enterprise) BehavesLike.PS.Suspicious.gr, engine v2021.2.0+4045, definitions
  20260912 |
| the same file uploaded elsewhere as unsloth_install2.ps1 | same single engine, same label |
| tests/security/fixtures/malicious_wheel.whl, malicious_sdist.tar.gz | 2/65 and 2/60 (Tencent,
  Rising), and the cause of Panda flagging the GitHub repo zip as Exploit/CVE-2014-6271. Both are now
  generated at test time rather than committed |

Beyond the one engine verdict, VirusTotal runs 17 Sigma rules against install.ps1 (1 high, 11 medium,
5 low) plus two crowdsourced YARA rules. That rule count, not an engine detection, is what drives the
high scores seen on third-party analysis sites.

## What the hardening has actually bought so far

Measured by scripts/virustotal_delta.py, not asserted:

|                  | baseline (1ad44677d)          | main (2c70592a)   |
|------------------|-------------------------------|-------------------|
| engines flagging | 1 / 60                        | 1 / 61            |
| which            | Skyhigh BehavesLike.PS.Suspicious.gr | the same   |
| Sigma            | 17 (1 high, 11 medium, 5 low) | 14 (10 medium, 4 low) |
| YARA             | 2                             | 0                 |

The high-severity rule is gone, and so are both YARA hits --
INDICATOR_SUSPICIOUS_PWSH_B64Encoded_Concatenated_FileEXEC and Windows_API_Function, the second of
which is the P/Invoke cluster. The Skyhigh verdict has not moved, and that is recorded as unchanged
rather than dressed up: a cloud behavioural verdict is not recomputed because we deleted some code,
and the honest expectation is that the static axis moves first and the engine axis may never move.

This is the first before-and-after this work has had. Six passes shipped without one, which is why
none of them could be shown to have achieved anything.

## Reflection emit instead of Add-Type

Add-Type on Windows PowerShell 5.1 -- the interpreter the desktop app spawns, and the one a clean
Windows box ships -- has no in-process compiler. -TypeDefinition and -MemberDefinition alike write C#
to %TEMP% and run csc.exe, leaving source, a response file and a DLL behind.

Bitdefender blocked the resulting DLL as Gen:Variant.MSILHeracles.272113 (#10540). The shape it scores
is real: a windowless PowerShell spawned by a GUI binary, running a compiler, writing executable
content to %TEMP%. The same call also failed outright with CS2001 where %TEMP% was unusable (#9140),
so this is a robustness fix as much as a detection one.

System.Reflection.Emit builds the identical interop stubs in memory: no compiler process, no source
file, no DLL, and an empty Assembly.Location. windows-no-compiler-ci.yml is the standing proof --
4688 process auditing unioned with a live FileSystemWatcher over every temp root, with a positive
control, requiring zero compiler launches across a full install.

## No .vbs launcher

A WScript.Shell .vbs that spawns a hidden PowerShell with a bypassed execution policy is close to the
canonical shape VBS-dropper heuristics are written for -- Kaspersky HEUR:Trojan.VBS.Agent.gen is the
family. The .lnk files therefore point straight at powershell.exe running launch-studio.ps1, with no
script engine in between, and the installer removes an older launch-studio.vbs if it finds one,
because leaving it behind means a machine that was cleaned still has the flagged file.

## RemoteSigned rather than Bypass, next to a hidden window

A hidden window paired with a bypassed execution policy is a pair Microsoft's own detections key on,
and it appears in Sigma's "Suspicious PowerShell WindowStyle Option" too. The hidden window is a real
requirement -- a console flashing up on every launch is a visible regression, and
tests/studio/install/test_launch_studio_launcher.py pins the flag -- so the half that goes is the
policy.

RemoteSigned is not a weaker guarantee here, it is the same one: it refuses unsigned scripts only in
the Internet and Untrusted zones, and every script involved is written locally by the installer. Where
a mark of the web can exist (a downloaded zip rather than a clone), Unblock-File clears it first
instead of the policy being relaxed to tolerate it.

Three things to know before simplifying any of this:

  - UNC paths are a remote zone. \\wsl.localhost\<distro>\tmp\x.ps1 is refused under RemoteSigned,
    which is why install.sh resolves the Windows %TEMP% for its generated script and skips shortcut
    creation rather than falling back to Bypass when it cannot.
  - Execution policy is evaluated against the script FILE's zone, not against who launched
    PowerShell, and it does not apply to -Command at all. A .cmd or .bat shim therefore cannot
    launder a remote-zone script, and Sigma's "Powershell Execute Batch Script" scores the shim
    itself.
  - The Windows CLIENT default is Restricted, which blocks every script file. Documentation telling a
    user to set a process-scoped policy is load-bearing; it just does not need to say Bypass.

## The icon-cache refresh is gated on a first install or a real icon change

Clearing icon caches and killing StartMenuExperienceHost is the only way to make Explorer pick up a
rewritten .lnk icon, and ie4uinit's global broadcast alone does not do it. But "clear caches, kill a
shell process, repeat on every run" is a cluster behavioural engines score, and on a reinstall that
changed nothing it is also pure waste. So both install.ps1 and install.sh snapshot the icon and run
the heavy path only on a first install or an actual change, preserving start2.bin.

SHChangeNotify stays, and with it one shell32 import: a permanently wrong desktop icon is a worse
outcome than one import.

## uv comes from a pinned archive, not from a remote install script

The upstream one-liner pipes a remote script into the interpreter. Running remote script text
in-process is the highest-scoring thing in this problem space, and download-run-delete is the literal
definition of a dropper. Our fallback reaches the same end state (same archive, same destination, same
user-PATH prepend as astral's installer) by fetching a DATA file with a pinned SHA-256.

Cost: bumping the uv version means bumping all three hashes, one per architecture, in all three
scripts. That is deliberate friction and the comment at each site says so.

There is a floor no shape work removes: while the supported install route is
`irm https://unsloth.ai/install.ps1 | iex`, the documented entry point is itself the top-scoring token
sequence in this space.

## Why the script headers do not repeat the usage text

install.ps1 and scripts/uninstall.ps1 are scanned in full at compile time, before a line of either
runs, and nothing reads a header comment from inside the script. Duplicating the README's option list
into a header adds bytes to a classifier's input and reaches no user.

## Shapes we are keeping, on purpose

Each fires a rule and each is load-bearing. Listed so nobody spends a second pass rediscovering them.

  - -WindowStyle Hidden on the shortcuts. Removing it makes a console window appear on every launch;
    the flag is a pinned contract in test_launch_studio_launcher.py.
  - Authenticode publisher checks on the python.org and vc_redist downloads. Weakening a real security
    control to lower a heuristic score is backwards.
  - New-Object -ComObject WScript.Shell to write the .lnk files. The only shortcut mechanism available
    to PowerShell 5.1 without IShellLink interop; hand-writing the shell-link binary format or
    emitting COM interop are both more suspicious and more fragile.
  - Unblock-File rather than deleting the Zone.Identifier stream directly. The direct delete trades
    one medium indicator ("Suspicious Unblock-File") for another ("Hidden Executable In NTFS Alternate
    Data Stream"), which already fires.
  - ie4uinit, Get-Process, python -X utf8 -c, Invoke-WebRequest. Each is scored; each has no
    equivalent that does the job.

## Which products ship a controlled-folder-access equivalent

Get-SecuritySoftwareNote, in install.ps1 and studio/setup.ps1, explains a denied llama.cpp cache
by naming the security product that is registered and running, because takeown and icacls cannot
clear a filter-driver block and elevation does not either. Defender's own feature is Controlled
folder access, and the script names it directly: that name is Microsoft's, not a third-party
vendor's, and is already in the user-facing string the function returns.

The third-party suites ship the same protected-folders feature under their own product names,
and those names live here rather than in the scripts, because the scripts are AMSI input and a
comment listing security vendors raises the score of the file it is explaining:

| Vendor | Feature |
|---|---|
| Bitdefender | Safe Files, and Ransomware Remediation |
| Kaspersky | Anti-Ransomware / Protected folders |
| Trellix and McAfee Enterprise | Access Protection rules |
| Sophos | CryptoGuard protected folders |

The function does not hard-code any of these. It reads whatever SecurityCenter2 has registered
and names that, so the list above is the reason the code is written to ask rather than the data
it asks with, and it does not need updating when a vendor renames a feature.

## Reporting a detection

Use the "Windows: antivirus or security software blocked the installer" issue form. It requires the
product, the exact detection name, the full error including its FullyQualifiedErrorId, and the output
of a read-only collection script naming the AMSI providers actually loaded.

Those fields are required because clearance is granted per file hash and every vendor submission form
asks for a detection name. Six previous reports (#8523, #6326, #6588, #6648, #10540, #10805) named
none of them, which is why none could be submitted to a vendor or proven fixed.
"""


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
            # PowerShell wants the terminator in column 0, and install.ps1 has
            # indented `"@echo off",` array entries that a stripped comparison
            # closes on.
            if line.startswith(("'@", '"@')):
                in_here_string = False
            continue
        # Quoted literals first. Both install.ps1 and studio/setup.ps1 redact
        # credentials with `-replace ..., '$1<redacted>@'`, whose raw line ends
        # in `@'`; opening a here-string there swallowed everything up to the
        # next terminator -- 780 lines of setup.ps1, 740 of install.ps1 -- and
        # every check below silently stopped looking at them.
        blanked = _QUOTED.sub('""', line)
        if re.search(r"@[\"']$", blanked.strip()):
            in_here_string = True
            continue
        if stripped.startswith("#"):
            continue
        yield number, blanked


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


_HIDDEN = re.compile(r"-WindowStyle\s+Hidden", re.IGNORECASE)
_BYPASS = re.compile(
    r"-ExecutionPolicy\s+Bypass|Set-ExecutionPolicy[^\r\n]*?\bBypass\b", re.IGNORECASE
)
_ASSIGNMENT = re.compile(r"\$(?:script:|env:)?(\w+)\s*(?:=|\+=)\s*(.*)")

# Every relaxed execution policy left in a shipped script, why it is still there, and what removes
# it. A ratchet: these counts may go down, never up, and the test fails BOTH ways -- too many is a
# new site, too few is a stale entry that has stopped guarding anything.
KNOWN_BYPASS_SITES = {
    # install.ps1:3433, the roaming-profile fallback for a launcher on a share. The last one left,
    # and the only one that is genuinely load-bearing: %LOCALAPPDATA% can be folder-redirected to a
    # UNC path, RemoteSigned refuses an unsigned script there, and a desktop shortcut that silently
    # does nothing is worse than the token. Removing it needs launch-studio.ps1 written to a
    # guaranteed-local directory first.
    "install.ps1": 1,
}

# Known (script, variable) pairs where one assignment carries a hidden window and another a relaxed
# policy. Empty is the goal. install.ps1's $shortcutArgs is recorded rather than failed on, so this
# guard can land without also forcing the launcher relocation above.
# Keyed on the CASEFOLDED variable name. PowerShell variable names are not case-sensitive
# (about_Variables: "Variable names aren't case-sensitive"), so `$shortcutArgs` and `$ShortcutArgs`
# are one variable. Grouping on the captured spelling instead would file them as two, each holding
# only one of the two flags, and layer 3 below would wave the pair through on a capitalisation edit.
KNOWN_SPLIT_PAIR_VARIABLES = {("install.ps1", "shortcutargs")}


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_a_hidden_window_never_pairs_with_a_bypassed_policy(name: str) -> None:
    """Three layers, because the same-line check alone never saw the pair we actually shipped.

    Microsoft's detections key on the pair, and install.rs already refuses it for the app's own
    launch. Python setup/refresh argv is exercised at the subprocess boundary by
    unsloth_cli/tests/test_studio_runtime_gate_powershell.py::
    test_windows_launch_uses_process_flags_without_windowstyle.

    The original check compared the two flags only within one physical line. install.ps1 assigns
    `$shortcutArgs` a hidden window at :3419 and then overwrites it with a relaxed policy at :3433,
    fourteen lines apart in one function, and that passed for as long as it existed. A scanner reads
    the file, not the line.
    """
    text = _text(name)
    lines = text.splitlines()

    # 1. Same line. The cheapest check and the one with the clearest message.
    for number, line in enumerate(lines, start = 1):
        if _HIDDEN.search(line):
            assert not _BYPASS.search(
                line
            ), f"{name}:{number} pairs a hidden window with a bypassed policy: {line.strip()}"

    # 2. A ratchet on how many relaxed policies the file contains at all, wherever they sit and
    #    whatever they are near. This is what catches a new one arriving somewhere the other two
    #    layers do not model.
    found = [
        (number, line.strip()) for number, line in enumerate(lines, start = 1) if _BYPASS.search(line)
    ]
    allowed = KNOWN_BYPASS_SITES.get(name, 0)
    assert len(found) <= allowed, (
        f"{name} relaxes the execution policy in {len(found)} place(s) but {allowed} are recorded: "
        f"{found}. RemoteSigned loads any locally written, unmarked script, which covers almost "
        f"every case; if this one genuinely cannot, add it to KNOWN_BYPASS_SITES with the reason "
        f"and what would remove it."
    )
    assert len(found) >= allowed, (
        f"{name} now has {len(found)} relaxed policies but KNOWN_BYPASS_SITES records {allowed}. "
        f"Lower the count in the same commit that removed one, so the ratchet keeps its grip "
        f"instead of leaving slack for the next regression to fit into."
    )

    # 3. Same variable, any distance: the union of everything assigned to one name must not contain
    #    both flags. This is the layer that catches the install.ps1 3419/3433 shape.
    contributions: dict[str, set] = {}
    spellings: dict[str, set] = {}
    for line in lines:
        match = _ASSIGNMENT.match(line.strip())
        if not match:
            continue
        # Casefolded, because PowerShell resolves $shortcutArgs and $ShortcutArgs to one variable.
        key = match.group(1).casefold()
        spellings.setdefault(key, set()).add(match.group(1))
        seen = contributions.setdefault(key, set())
        if _HIDDEN.search(match.group(2)):
            seen.add("hidden")
        if _BYPASS.search(match.group(2)):
            seen.add("bypass")
    for variable, seen in sorted(contributions.items()):
        if seen != {"hidden", "bypass"}:
            continue
        written = " / ".join(sorted(spellings[variable]))
        assert (name, variable) in KNOWN_SPLIT_PAIR_VARIABLES, (
            f"{name}: ${written} is assigned a hidden window in one place and a relaxed policy in "
            f"another. Only one of them reaches the command line, so whichever is dead weight "
            f"should go rather than be recorded here."
        )


# Every native import left in the installers, however it is declared. Both scripts define theirs through reflection
# emit now, which costs no compile: install.ps1 the path resolver, console thunk, icon refresh and process-image
# lookup, studio/setup.ps1 the console thunk. A new entry needs a reason; a PowerShell equivalent usually exists.
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
    # PROCESS_QUERY_LIMITED_INFORMATION only, and the others cannot replace it: Process.Path goes through MainModule,
    # which needs PROCESS_VM_READ and is refused across users and bitness, and Win32_Process needs a working WMI
    # service. Without it the scan can find nothing and proceed over an open venv.
    "OpenProcess",
    "QueryFullProcessImageNameW",
    # Closing the handles CreateFileW and OpenProcess opened.
    "CloseHandle",
    # The NVIDIA driver's own inventory (Get-NvidiaLibraryInventory) for a host whose nvidia-smi is absent, stale
    # or hangs: the CUDA driver version and one compute capability per GPU. No PowerShell or .NET equivalent
    # exists; nvidia-smi is the thing being worked around, and reading the registry names no driver version.
    "nvmlInit_v2",
    "nvmlShutdown",
    "nvmlSystemGetCudaDriverVersion_v2",
    "nvmlDeviceGetCount_v2",
    "nvmlDeviceGetHandleByIndex_v2",
    "nvmlDeviceGetCudaComputeCapability",
    "cuInit",
    "cuDriverGetVersion",
    "cuDeviceGetCount",
    "cuDeviceGet",
    "cuDeviceGetAttribute",
}


# Both ways a native import can be declared: Add-Type runs csc.exe over C#, DefinePInvokeMethod builds the same stub
# in memory. The second is invisible to a DllImport regex, so without it the inventory above would stop covering
# install.ps1 the moment it stopped compiling.
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


def test_setup_bat_clears_the_mark_before_loading_under_remotesigned() -> None:
    """The batch launcher's two calls, in order, and the one flag it must not grow.

    `setup.bat` used to run `powershell -ExecutionPolicy Bypass -File setup.ps1`. RemoteSigned is
    enough, because setup.ps1 ships beside it inside an installed package and is MyComputer-zone.
    The exception is a package unzipped from a download, where setup.ps1 carries a mark of the web
    that RemoteSigned honours and Bypass ignored, so the mark is cleared first. Execution policy
    governs script FILES and not -Command, so that first call runs under any machine policy.

    The launch must keep loading profiles. That is not an oversight: tests/studio/
    test_amd_venv_repair_loop.ps1 drives a profile that sets `Set-StrictMode -Version Latest`
    against setup.ps1, and adding -NoProfile here would silently retire that coverage.
    """
    text = _text("studio/setup.bat")
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().lower().startswith(("rem ", "@echo", "rem\t"))
    ]
    # The path and the policy both travel in variables now, so match on the flags rather than on a
    # literal filename: the path is an environment variable so an apostrophe in the install
    # directory cannot break the quoting, and the policy is chosen by a probe that steps down to
    # Bypass only for a script on a remote share (see
    # test_setup_bat_steps_down_to_bypass_only_for_a_remote_script).
    launches = [line for line in lines if "-File" in line and "-ExecutionPolicy" in line]
    assert len(launches) == 1, f"expected exactly one setup.ps1 launch, found {launches}"
    launch = launches[0]

    assert "-ExecutionPolicy %UNSLOTH_SETUP_POLICY%" in launch, (
        f"studio/setup.bat must load setup.ps1 under the probed policy, not a hardcoded relaxed "
        f"one: {launch}"
    )
    assert 'set "UNSLOTH_SETUP_POLICY=RemoteSigned"' in text, (
        "the probed policy no longer DEFAULTS to RemoteSigned. That default is what makes a probe "
        "which fails to run leave the tightened policy in place instead of restoring the relaxed one."
    )
    assert "-NoProfile" not in launch, (
        "studio/setup.bat must keep loading profiles for setup.ps1. "
        "tests/studio/test_amd_venv_repair_loop.ps1 drives a profile that sets Set-StrictMode "
        "against it, and -NoProfile here would retire that coverage without anything failing."
    )

    unblock = [line for line in lines if "Unblock-File" in line]
    assert len(unblock) == 1, f"expected exactly one Unblock-File call, found {unblock}"
    assert text.index(unblock[0]) < text.index(launch), (
        "the mark of the web has to be cleared before the launch that RemoteSigned would refuse, "
        "not after it"
    )
    # Interpolating the path into the command string breaks on an apostrophe in the install
    # directory, which is a real Windows user name.
    assert "$env:" in unblock[0], (
        f"pass the script path to Unblock-File through an environment variable rather than "
        f"interpolating it into the command string: {unblock[0]}"
    )


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
    """The stronger contract this test's name always implied: nothing native happens here at all.

    It used to assert an ordering -- that the redirect check came *before* the emit call -- because
    the redirect check was the only thing keeping the desktop app off csc.exe. There is no emit call
    now. A CI pre-flight measured Windows PowerShell 5.1 attached to a real console and found the
    console mode already 0x7 before any of our code ran: bit 0x4,
    ENABLE_VIRTUAL_TERMINAL_PROCESSING, is set by the host at startup. The SetConsoleMode this
    replaced was re-setting a bit that was already set, so reading
    $Host.UI.SupportsVirtualTerminal loses nothing.

    Two things still have to hold. The redirected case must still be decided FALSE and decided
    first: a redirected stdout is not a console, and anything claiming VT there puts raw escape
    sequences in the Unsloth log panel, which is a pipe. And the function must stay free of native
    work, or the three kernel32 imports come back one careful commit at a time.
    """
    text = _text(name)
    start = text.index("function Enable-StudioVirtualTerminal")
    # To the end of the function. The next top-level construct after it is the assignment of its
    # result, which is a stable landmark in both files.
    end = text.index("$script:StudioVtOk = Enable-StudioVirtualTerminal", start)
    body = text[start:end]

    fast_path = body.index("if ($script:StudioStdoutRedirected) { return $false }")
    property_read = body.index("$Host.UI.SupportsVirtualTerminal")
    assert fast_path < property_read, (
        f"{name} consults the host before checking whether the stream is redirected. A redirected "
        f"stream can never render VT, so that case has to be decided first and decided false."
    )

    for banned in (
        "New-StudioEmittedNativeType",
        "DefinePInvokeMethod",
        "Add-Type",
        "GetStdHandle",
        "SetConsoleMode",
        "kernel32",
    ):
        assert banned not in _strip_comments(body), (
            f"{name}'s Enable-StudioVirtualTerminal does native work again ({banned}). The host "
            f"already enables virtual terminal processing at startup, measured: the console mode "
            f"is 0x7 before we touch it. Colouring a banner is not worth three kernel32 imports."
        )


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def test_neither_installer_declares_a_console_mode_import() -> None:
    """The console thunk reached zero native surface, and must not drift back.

    Both scripts used to declare GetStdHandle, GetConsoleMode and SetConsoleMode for one consumer: a
    cosmetic ANSI colour banner. A CI pre-flight measured Windows PowerShell 5.1 attached to a real
    console and found the mode already 0x7 before anything of ours ran, so bit 0x4,
    ENABLE_VIRTUAL_TERMINAL_PROCESSING, was already set by the host and the SetConsoleMode was
    re-setting it. Without this test the three imports are one "just add a small helper" away from
    coming back, and nothing else in the suite would notice: every other check here is about how a
    native import is DECLARED rather than whether there is one.

    Scoped to the console imports rather than to all of them. studio/setup.ps1 still emits the nvml
    and nvcuda imports for Get-NvidiaLibraryProbeType, which is what the GPU inventory reads, so a
    blanket "no native imports" assertion would be false and deleting the apparatus to satisfy it
    would break that.
    """
    for name in ("install.ps1", "studio/setup.ps1"):
        declared = _native_imports(_text(name))
        for banned in ("GetStdHandle", "GetConsoleMode", "SetConsoleMode"):
            assert banned not in declared, (
                f"{name} declares {banned} again; the console mode is the host's job and "
                f"$Host.UI.SupportsVirtualTerminal reports its outcome"
            )
    setup = _strip_comments(_text("studio/setup.ps1"))
    assert "StudioVTNative" not in setup, "the emitted console thunk is back in studio/setup.ps1"
    assert "Add-Type" not in setup, "studio/setup.ps1 compiles C# through csc.exe again"


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


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_the_installer_never_runs_the_c_sharp_compiler(name: str) -> None:
    """The desktop app spawns Windows PowerShell 5.1, which compiles Add-Type by writing C# to
    %TEMP% and running csc.exe. A GUI binary launching a windowless PowerShell that launches a
    compiler and drops a DLL in %TEMP% is a dropper's shape whatever the code says, and it was
    blocked in the field. Reflection emit builds the same stub in memory: no compiler process,
    no source on disk, no DLL, nothing in %TEMP%.

    Add-Type in full, not only -TypeDefinition: -MemberDefinition wraps its argument in a class
    and compiles that too. -AssemblyName is the only exception, since it loads an assembly that
    already exists on disk. Every shipped script, because a compile left anywhere makes "does this
    run a compiler" depend on which entrypoint ran and whether an early return came first, and a
    guard that holds only conditionally is what let this reach the field.

    The shell scripts are covered for a concrete reason, not for symmetry. install.sh writes
    PowerShell into a here-string and runs it on the Windows side to create the WSL shortcut, and
    that generated script still carried the `Add-Type -MemberDefinition` this test exists to ban:
    #10540 replaced it in install.ps1 and the install.sh copy was missed, because the parametrise
    list here stopped at the two .ps1 files. The `^[ \\t]*Add-Type` anchor matches inside a
    here-string exactly as it does outside one, so seeing it needs no here-string parsing.
    """
    text = _text(name)
    hits = re.findall(r"(?m)^[ \t]*Add-Type\b(?![^\r\n]*-AssemblyName).*", text)
    assert not hits, (
        f"{name} compiles C# again ({len(hits)} Add-Type call(s), first: {hits[0].strip()!r}). "
        "Declare native methods with New-StudioEmittedNativeType instead, or with an inline "
        "DefinePInvokeMethod block where the script is generated and cannot call it; "
        "-MemberDefinition runs csc.exe just as -TypeDefinition does."
    )
    # Conditional, because "emits its native imports" only means anything for a file that HAS
    # native imports. Three shipped files now declare none: scripts/uninstall.ps1 and
    # studio/setup.sh never did, and studio/setup.ps1 stopped -- its whole emit apparatus existed
    # to colour a banner, and the host turns out to enable virtual terminal processing before our
    # code runs. Demanding the token of a file with zero native surface would be a guard that
    # fails for being satisfied; test_setup_declares_no_native_imports_at_all is what keeps that
    # zero honest.
    if _native_imports(text):
        assert "DefinePInvokeMethod" in text, (
            f"{name} declares native imports but no longer emits them. If they are compiled again "
            f"instead, that is csc.exe on 5.1, which is the shape this whole file exists to keep out."
        )
    # The private-%TEMP% retry is gone with it: redirecting TEMP to compile again cannot beat a
    # filter driver, and "blocked writing an executable to TEMP, change TEMP, write it again" is
    # itself an evasion heuristic. Scoped to the resolver, since Initialize-StudioTempEnvironment
    # legitimately redirects an unusable inherited TEMP.
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

    Reading the scripts cannot see a compile reached through a module, a dot-sourced file
    or a generated here-string, nor one a dependency performs while our process tree is
    what a scanner scores. Bitdefender scored the chain, not the bytes, so a lane has to
    run the installer and fail on the process.

    The positive control is what is worth asserting from here: a detector that sees
    nothing reads exactly like a clean run, and auditing can silently fail to apply.
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

# The .NET host tearing itself down, as opposed to the script under test deciding something.
# Seen on a hosted runner as `System.IO.FileLoadException: The given assembly name was
# invalid.` out of AssemblyName.ParseAsAssemblySpec, followed by "The PowerShell process will
# exit" and SIGABRT, on a probe that passes everywhere else and had no assembly of its own.
_PWSH_HOST_FAULT = (
    "An error has occurred that was not properly handled",
    "System.IO.FileLoadException",
    "Unhandled exception.",
)


def _run_pwsh(script: Path, *, timeout: int):
    """Run `script` under pwsh, skipping rather than failing when the HOST aborts.

    Only an abnormal termination is forgiven, and only with a fault banner on stderr to back
    it up: a clean non-zero exit, or the wrong answer on stdout, is the script under test
    being wrong and still fails. Retried once first, because the fault has never repeated.

    pwsh_env, not run_pwsh: this function's whole job is to look at a crashed
    CompletedProcess and decide, and run_pwsh raises PwshInterpreterCrash instead of
    returning one, so it cannot be the caller here. What it can still take is the private
    startup cache -- and the FileLoadException named in _PWSH_HOST_FAULT above is exactly
    the torn-cache shape that cache directory removes, so this is the call site that most
    needed it. See tests/_shared/unsloth_pwsh_runner.py.
    """
    command = ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)]
    env = pwsh_env()
    for attempt in range(2):
        result = subprocess.run(command, capture_output = True, text = True, timeout = timeout, env = env)
        crashed = result.returncode < 0 and any(
            marker in result.stderr for marker in _PWSH_HOST_FAULT
        )
        if not crashed:
            return result
        if attempt:
            pytest.skip(f"pwsh host aborted ({result.returncode}): {result.stderr.strip()[:400]}")
    raise AssertionError("unreachable")


_FAKE_EVENTS = r"""
function New-FakeEvent {
    param([string]$Image, [string]$CommandLine, [string]$Parent = 'C:\Windows\System32\cmd.exe')
    $xml = "<Event><EventData>" +
        "<Data Name='NewProcessName'>$Image</Data>" +
        "<Data Name='ParentProcessName'>$Parent</Data>" +
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
    """4688 renders the command line into the message, so a message search is not a detector:
    it scored `cmd.exe /c echo csc.exe` as a compile. The record names the image it created
    in its own field; that is what gets read, matched whole against the leaf name rather
    than as a substring.
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
    result = _run_pwsh(script, timeout = 120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"HITS:{expected}" in result.stdout, result.stdout


_CSC = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
_CVTRES = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\cvtres.exe"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")
@pytest.mark.parametrize(
    ("image", "parent", "expected"),
    [
        (_CSC, r"C:\Program Files\PowerShell\7\pwsh.exe", 1),
        (_CVTRES, r"C:\Program Files\PowerShell\7\pwsh.exe", 1),
        (_CVTRES, _CSC, 0),
    ],
    ids = [
        "a-compile-the-shell-started",
        "a-resource-step-with-no-compiler-parent",
        "a-resource-step-the-compiler-started",
    ],
)
def test_a_compiler_started_by_a_compiler_is_one_compile_not_two(
    tmp_path, image: str, parent: str, expected: int
) -> None:
    """csc.exe shells out to cvtres.exe, so a single compile creates two 4688 records and
    scoring both says the action compiled twice.

    It also decides the cross-step bleed the timestamp baseline could not. The Security log
    is written with latency: the positive control's csc.exe started before the installer's
    window opened, its cvtres.exe child landed just inside, and neither was in the log yet
    when the baseline was read, so the subtraction had nothing to subtract and the
    installer was failed for a compile one step earlier.

    A compile the action really starts is still caught, because its ROOT compiler is
    spawned by the installer's shell and the window opens before the action does. That is
    the middle case here: an orphaned resource step with a non-compiler parent still counts.
    """
    script = tmp_path / "probe.ps1"
    script.write_text(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                f'. "{_WATCHER}"',
                _FAKE_EVENTS,
                f"$e = New-FakeEvent -Image '{image}' -CommandLine 'x' -Parent '{parent}'",
                "$hits = Select-StudioCompilerHits -Events @($e)",
                'Write-Output "HITS:$($hits.Count)"',
            ]
        ),
        encoding = "utf-8",
    )
    result = _run_pwsh(script, timeout = 120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"HITS:{expected}" in result.stdout, result.stdout


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")
def test_a_record_with_no_parent_field_is_still_scored(tmp_path) -> None:
    """The whole chain is reported when the schema does not carry ParentProcessName. An
    absent field reads as empty, and empty must not be mistaken for a compiler parent, or a
    log that predates the field would report nothing at all."""
    script = tmp_path / "probe.ps1"
    script.write_text(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                f'. "{_WATCHER}"',
                # The pre-parent schema: NewProcessName and nothing else.
                "function New-OldEvent {",
                "    param([string]$Image)",
                "    $xml = \"<Event><EventData><Data Name='NewProcessName'>$Image</Data>\" +",
                '        "</EventData></Event>"',
                "    $record = [pscustomobject]@{",
                "        TimeCreated = [datetime]'2026-01-01T00:00:00Z'",
                '        Message     = "New Process Name: $Image"',
                "    }",
                "    $body = [scriptblock]::Create(\"return @'`n$xml`n'@\")",
                "    return ($record | Add-Member -MemberType ScriptMethod -Name ToXml "
                "-Value $body -PassThru)",
                "}",
                f"$e = New-OldEvent -Image '{_CSC}'",
                "$hits = Select-StudioCompilerHits -Events @($e)",
                'Write-Output "HITS:$($hits.Count)"',
            ]
        ),
        encoding = "utf-8",
    )
    result = _run_pwsh(script, timeout = 120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "HITS:1" in result.stdout, result.stdout


_FAKE_WINEVENT = r"""
function Get-WinEvent {
    # Off Windows there is no such cmdlet, so this resolves the call. Empty rather than
    # throwing: this exercises the artefact half, and the 4688 half has its own tests.
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    return @()
}
"""


# A new subdirectory's watch is not in place the instant the directory appears. On
# Windows, ReadDirectoryChangesW is recursive in the kernel, so there is no gap at all.
# Off Windows, .NET emulates IncludeSubdirectories by adding an inotify watch per
# directory, and it adds the one for a directory it has just been told about after the
# fact: a file written into a brand-new subdirectory microseconds later can land before
# its watch does, and the creation is never raised.
#
# That is invisible to a test whose files are still on disk at the end, because the
# listing half reports those anyway. It is the whole result for the one below, where the
# live stream is the only detector left. Measured here, idle, 64 cores: 1 miss in 60
# without this settle, 0 in 60 with it. A two-core hosted runner is where it actually
# bit (Backend CI job 105986532571, `Repo tests (CPU, studio)`, COUNT:0).
#
# This is a property of watching a Linux filesystem, not of Watch-ForCompiler.ps1, which
# runs on Windows in anger. csc.exe does not write its intermediates in the same
# microsecond it creates their directory either, so waiting here is the faithful shape.
_SETTLE_FOR_THE_SUBDIRECTORY_WATCH = "Start-Sleep -Milliseconds 500; "


def _run_watch(tmp_path, action: str, setup: str = "") -> tuple[str, list[str]]:
    """Drive the real Invoke-WithCompilerWatch over $Action, with TEMP pointed at tmp_path.

    ``setup`` runs BEFORE the watch starts, for the one case that needs a directory to
    already exist and already be watched when the action writes into it.
    """
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    evidence = tmp_path / "evidence"
    script = tmp_path / "probe.ps1"
    script.write_text(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                f'$env:TEMP = "{temp_root.as_posix()}"',
                f'$env:TMP = "{temp_root.as_posix()}"',
                _FAKE_WINEVENT,
                f'. "{_WATCHER}"',
                setup,
                f"$action = {{ {action} }}",
                "$seen = Invoke-WithCompilerWatch -Name 'probe' -Action $action "
                f'-EvidenceRoot "{evidence.as_posix()}"',
                'foreach ($lib in $seen.TempLibraries) { Write-Output "LIB:$lib" }',
                'Write-Output "COUNT:$($seen.TempLibraries.Count)"',
            ]
        ),
        encoding = "utf-8",
    )
    result = _run_pwsh(script, timeout = 300)
    assert result.returncode == 0, result.stderr + result.stdout
    libraries = [
        line[len("LIB:") :] for line in result.stdout.splitlines() if line.startswith("LIB:")
    ]
    return result.stdout, libraries


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")
def test_the_watcher_sees_intermediates_the_compiler_cleaned_up(tmp_path) -> None:
    """The failure this replaces: the positive control compiled a type, 4688 recorded

        csc.exe /noconfig /fullpaths @"...\\Temp\\vpmyd5eq\\vpmyd5eq.cmdline"

    and the artefact half reported nothing, because CodeDom deletes its intermediate
    directory once the assembly is loaded. Comparing a listing taken before against one
    taken after cannot see a file that no longer exists, so the job failed as a broken
    detector on every run since it was added.
    """
    action = (
        '$dir = Join-Path $env:TEMP "abcd1234"; '
        "New-Item -ItemType Directory -Force -Path $dir | Out-Null; "
        # See _SETTLE_FOR_THE_SUBDIRECTORY_WATCH: off Windows the watch for $dir is added
        # after $dir appears, and this is the one test with no listing half to fall back on.
        + _SETTLE_FOR_THE_SUBDIRECTORY_WATCH
        + 'Set-Content -LiteralPath (Join-Path $dir "abcd1234.cmdline") -Value "/noconfig"; '
        'Set-Content -LiteralPath (Join-Path $dir "abcd1234.dll") -Value "MZ"; '
        "Start-Sleep -Milliseconds 400; "
        # The whole point: gone before the action returns, exactly as CodeDom leaves it.
        "Remove-Item -LiteralPath $dir -Recurse -Force"
    )
    stdout, libraries = _run_watch(tmp_path, action)
    assert libraries, f"a compile that cleaned up after itself was missed again: {stdout}"
    assert any(lib.endswith(".cmdline") for lib in libraries), libraries
    assert any(lib.endswith(".dll") for lib in libraries), libraries


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")
def test_a_deleted_intermediate_is_reported_with_no_subdirectory_race_to_lose(tmp_path) -> None:
    """The same claim as above, with the platform's timing taken out of it.

    The claim is that a file which no longer exists when the action returns is still
    reported, and nothing about that claim needs the file's directory to be new. Here the
    directory is created before the watch starts, so its watch is in place before anything
    is written into it and the live stream is the only thing being measured. If the test
    above ever goes quiet on a loaded runner, this one still fails when the watcher stops
    reporting what it saw, which is the regression either of them is for.
    """
    setup = (
        '$staged = Join-Path $env:TEMP "wxyz9876"; '
        "New-Item -ItemType Directory -Force -Path $staged | Out-Null"
    )
    action = (
        '$dir = Join-Path $env:TEMP "wxyz9876"; '
        'Set-Content -LiteralPath (Join-Path $dir "wxyz9876.cmdline") -Value "/noconfig"; '
        'Set-Content -LiteralPath (Join-Path $dir "wxyz9876.dll") -Value "MZ"; '
        "Start-Sleep -Milliseconds 400; "
        "Remove-Item -LiteralPath $dir -Recurse -Force"
    )
    stdout, libraries = _run_watch(tmp_path, action, setup = setup)
    assert libraries, f"a compile that cleaned up after itself was missed again: {stdout}"
    assert any(lib.endswith("wxyz9876.cmdline") for lib in libraries), libraries
    assert any(lib.endswith("wxyz9876.dll") for lib in libraries), libraries


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")
def test_the_watcher_still_reports_intermediates_that_were_left_behind(tmp_path) -> None:
    """The listing half must keep working; the watcher is added to it, not swapped for it.

    A compile that was NOT cleaned up, so the response file is still next to the assembly.
    This asserted a bare ``leftover.dll`` before, which read as "any DLL under TEMP is a
    compiler artefact"; that is the rule the job died on, and it is not what this test is
    for. The vehicle changed, the listing half it checks did not.
    """
    action = (
        '$dir = Join-Path $env:TEMP "leftover"; '
        "New-Item -ItemType Directory -Force -Path $dir | Out-Null; "
        'Set-Content -LiteralPath (Join-Path $dir "leftover.cmdline") -Value "/noconfig"; '
        'Set-Content -LiteralPath (Join-Path $dir "leftover.dll") -Value "MZ"'
    )
    _, libraries = _run_watch(tmp_path, action)
    assert any(lib.endswith("leftover.dll") for lib in libraries), libraries
    assert any(lib.endswith("leftover.cmdline") for lib in libraries), libraries


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")
def test_an_unpacked_archive_is_not_scored_as_a_compile(tmp_path) -> None:
    """What actually ran on every red run of this job.

    The installer unpacks llama.cpp's checksum-verified prebuilt release into a staging
    directory under TEMP, which lands ~25 DLLs there with no compiler anywhere near them.
    The shape under test is ``csc.exe -> %TEMP%\\<random>.dll``; an unpacked archive is a
    different thing and must not read as one, or the job can never pass and stops meaning
    anything.
    """
    action = (
        '$dir = Join-Path $env:TEMP "unsloth-llama-prebuilt-ay5ptbfd"; '
        '$dir = Join-Path $dir "extract-w613j_am"; '
        "New-Item -ItemType Directory -Force -Path $dir | Out-Null; "
        'foreach ($n in @("ggml.dll", "llama.dll", "mtmd.dll", "ggml-cpu-x64.dll")) { '
        '    Set-Content -LiteralPath (Join-Path $dir $n) -Value "MZ" '
        "}; "
        # A README ships in the archive too, and must stay just as uninteresting.
        'Set-Content -LiteralPath (Join-Path $dir "LICENSE.txt") -Value "MIT"; '
        "Start-Sleep -Milliseconds 400"
    )
    stdout, libraries = _run_watch(tmp_path, action)
    assert not libraries, f"an unpacked release archive was scored as a compile: {stdout}"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")
def test_a_compile_beside_an_unpacked_archive_is_still_caught(tmp_path) -> None:
    """The narrowing is per-directory, so unpacking an archive cannot cover a real compile."""
    action = (
        '$extract = Join-Path $env:TEMP "unsloth-llama-prebuilt-zz\\extract-zz"; '
        "New-Item -ItemType Directory -Force -Path $extract | Out-Null; "
        'Set-Content -LiteralPath (Join-Path $extract "ggml.dll") -Value "MZ"; '
        '$compile = Join-Path $env:TEMP "vpmyd5eq"; '
        "New-Item -ItemType Directory -Force -Path $compile | Out-Null; "
        'Set-Content -LiteralPath (Join-Path $compile "vpmyd5eq.cmdline") -Value "/noconfig"; '
        'Set-Content -LiteralPath (Join-Path $compile "vpmyd5eq.dll") -Value "MZ"; '
        "Start-Sleep -Milliseconds 400"
    )
    _, libraries = _run_watch(tmp_path, action)
    assert any(lib.endswith("vpmyd5eq.dll") for lib in libraries), libraries
    assert not any(lib.endswith("ggml.dll") for lib in libraries), libraries


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")
def test_an_action_that_compiles_nothing_reports_nothing(tmp_path) -> None:
    """Otherwise the real measurement, which requires neither detector to fire, can never pass.

    A text file is written so the action is not a no-op: the watcher sees the creation and
    must still discard it, because the extension is not one a compiler writes.
    """
    action = (
        'Set-Content -LiteralPath (Join-Path $env:TEMP "notes.txt") -Value "hello"; '
        "Start-Sleep -Milliseconds 400"
    )
    stdout, libraries = _run_watch(tmp_path, action)
    assert "COUNT:0" in stdout, stdout
    assert not libraries, libraries


def test_an_unreadable_security_log_is_void_rather_than_clean() -> None:
    """Get-WinEvent throws both for "nothing matched" and for "could not read".

    Swallowing both made a job that could not open the Security log print "no compiler"
    and pass. The positive control runs in an earlier step and says nothing about whether
    the log was still readable during the measurement.
    """
    body = _WATCHER.read_text(encoding = "utf-8")
    assert (
        "-MaxEvents 1" in body
    ), "the watcher no longer distinguishes an empty result from an unreadable log"
    assert "void rather than as clean" in body


def test_the_compiler_window_is_cut_to_size_by_timecreated() -> None:
    """The 4688 window has to be exact at the floor, or it scores the step before it.

    Observed on unslothai/unsloth#10626: a csc.exe recorded at 17:51:57.107 came back from
    a window whose floor was 17:51:57.58 and failed a measurement whose step had not
    printed its first line until 17:51:58.58. The compile belonged to the positive control
    one step earlier. $prior is meant to subtract exactly that, and did not, so the filter
    itself has to hold to the precision it was given rather than to the hashtable's.
    """
    body = _WATCHER.read_text(encoding = "utf-8")
    assert "StartTime = $Since.AddSeconds(-1)" in body
    assert "EndTime   = $Until.AddSeconds(1)" in body
    assert "$_.TimeCreated -ge $Since -and $_.TimeCreated -le $Until" in body, (
        "the padded query is no longer cut back to the real window, so it reports "
        "compiles from before the action began"
    )


def test_the_native_resolver_still_has_a_lexical_fallback() -> None:
    """The point of the change is the acquisition, not the ladder: a host where emit fails must
    degrade exactly as one that could not compile already did.
    """
    text = _text("install.ps1")
    assert "Write-StudioFinalPathDegraded" in text
    assert "Get-StudioLexicalPath" in text
    # Constrained Language Mode forbids defining types at all, by emit as by Add-Type.
    assert '$languageMode -ne "FullLanguage"' in text


# ---------------------------------------------------------------------------
# The shipped scripts must not name detections. The document must.
# ---------------------------------------------------------------------------


# Every file that ships and is scanned, including the two no other check in this file reads.
DOCUMENTED_SCRIPTS = tuple(sorted(set(ALL_SCRIPTS) | {"studio/setup.bat", "scripts/uninstall.sh"}))

# Vendor names, detection families and analyst vocabulary. Not a style rule: PowerShell hands the
# entire top-level script block to AMSI at compile time, so comments are classifier input, and
# VirusTotal's analysis of install.ps1 quoted one of our own comments as grounds for suspicion.
BANNED_TOKENS = (
    "bitdefender",
    "kaspersky",
    "skyhigh",
    "trellix",
    "mcafee",
    "avast",
    "sophos",
    "malwarebytes",
    "tencent",
    # "rising" is deliberately absent. It is a real engine, and one of the two that flagged the
    # fixture archives, but the word is also ordinary English: "rising memory use" is a sentence
    # someone will write, and boundary matching cannot tell it from the vendor. A guard that fails
    # on valid prose gets deleted by the next person, so it is worth less than nothing. "tencent",
    # the other engine that flagged those archives, has no such problem and stays.
    "panda",
    "wacatac",
    "heur:",
    "heracles",
    "gen:variant",
    "behaveslike",
    "trojan",
    "dropper",
    "amsi",
    "smartscreen",
    "virustotal",
    "sigma rule",
    "malware",
)

# Generic words describing a runtime hazard the code actually handles, one of which reaches the
# user. Banning these would delete real operational meaning, so they are deliberately allowed:
# antivirus, quarantine, scanner, security software, blocked.
#
# "false positive" is also deliberately absent, and for a more interesting reason: this test caught
# it at install.ps1 and studio/setup.ps1, where it means a *statistical* false positive in a
# registry probe and has nothing to do with a scanner. A token list is only as good as the words
# having one meaning.


def _banned_pattern(token: str) -> re.Pattern:
    """`token`, matched on word boundaries where the token's own edges are word characters.

    A raw substring search makes several of these unusable. `rising` is inside `surprising`,
    `arising` and `comprising`; `panda` is inside `pandas`, which is a real dependency name. The
    failure message would then accuse an ordinary sentence of naming an antivirus vendor, and the
    fix a reader would reach for is to delete the guard. Boundaries are conditional because
    `heur:` and `gen:variant` end or begin on a colon, where `\b` asserts the opposite of what is
    wanted.
    """
    left = r"\b" if token[:1].isalnum() else ""
    right = r"\b" if token[-1:].isalnum() else ""
    return re.compile(left + re.escape(token) + right, re.IGNORECASE)


@pytest.mark.parametrize("name", DOCUMENTED_SCRIPTS)
@pytest.mark.parametrize("token", BANNED_TOKENS)
def test_no_shipped_script_names_a_detection(name: str, token: str) -> None:
    path = REPO / name
    if not path.is_file():
        pytest.skip(f"{name} is not present")
    found = _banned_pattern(token).search(path.read_text(encoding = "utf-8"))
    assert not found, (
        f"{name} contains {token!r}. Vendor names, detection families and analyst vocabulary "
        f"belong in tests/studio/test_installer_av_shapes.py, not in a file that is itself handed to "
        f"a classifier in full before it runs. Say what the code does and what breaks if it "
        f"changes; link the anchor for which product flagged what."
    )


def test_the_record_survives_and_keeps_its_evidence() -> None:
    """Without this, the ban above is satisfiable by deleting the knowledge instead of moving it.

    Six hardening passes shipped without recording which engine flagged what, which is why none of
    them could be shown to have fixed anything. AV_SHAPES_RECORD is where that record lives now --
    in this file rather than a doc, because a test ships to nobody and nothing scans it, and because
    the guards that enforce the split are right here beside it.
    """
    for section in (
        "## Measured detections",
        "## Reflection emit instead of Add-Type",
        "## No .vbs launcher",
        "## RemoteSigned rather than Bypass, next to a hidden window",
        "## The icon-cache refresh is gated on a first install or a real icon change",
        "## uv comes from a pinned archive, not from a remote install script",
        "## Why the script headers do not repeat the usage text",
        "## Shapes we are keeping, on purpose",
    ):
        assert section in AV_SHAPES_RECORD, f"the record lost its {section!r} section"

    # The sections are the skeleton; these are the point. A record with headings and no evidence is
    # the same loss with extra steps.
    for evidence in (
        "Gen:Variant.MSILHeracles.272113",
        "HEUR:Trojan.VBS.Agent.gen",
        "BehavesLike.PS.Suspicious.gr",
        "#10540",
        "#10805",
        "#9140",
    ):
        assert evidence in AV_SHAPES_RECORD, f"the record no longer names {evidence}"


@pytest.mark.parametrize("name", DOCUMENTED_SCRIPTS)
def test_every_script_that_dropped_its_explanation_points_at_the_record(name: str) -> None:
    """A comment reduced to "security software blocks this" with no forward reference is worse
    than the prose it replaced: the next maintainer cannot tell whether it is still true.
    """
    path = REPO / name
    if not path.is_file():
        pytest.skip(f"{name} is not present")
    text = path.read_text(encoding = "utf-8")
    hints = ("Add-Type", "RemoteSigned", "ClearIconCache", "Sha256", "SHA-256")
    if not any(hint in text for hint in hints):
        pytest.skip(f"{name} carries none of the documented shapes")
    assert "test_installer_av_shapes.py" in text, (
        f"{name} implements one of the recorded shapes but references nothing. Point at "
        f"tests/studio/test_installer_av_shapes.py, where AV_SHAPES_RECORD says which product "
        f"flagged what, so the reasoning is one grep away rather than lost."
    )
    # Assembled, so that a later blanket rename of the doc path cannot silently rewrite this check
    # into asserting the opposite of what it means. That happened once while writing it.
    stale = "docs/windows-installer-" + "av-shapes.md"
    assert stale not in text, (
        f"{name} points at {stale}, which does not exist. The record lives in "
        f"tests/studio/test_installer_av_shapes.py as AV_SHAPES_RECORD."
    )


# -------------------------------------------------------------------------
# studio/setup.bat
# ---------------------------------------------------------------------------


def _setup_bat_probe() -> str:
    """The PowerShell that setup.bat embeds to clear the mark and choose a policy."""
    for line in _text("studio/setup.bat").splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("rem") or stripped.startswith("@rem"):
            continue
        if "-Command" not in line:
            continue
        body = line.split('-Command "', 1)[1]
        return body[: body.rindex('"`)')]
    raise AssertionError(
        "studio/setup.bat no longer embeds a -Command probe. It needs one: Unblock-File has to run "
        "before setup.ps1 is loaded under RemoteSigned, and the remote-path check has to happen "
        "before a policy is chosen."
    )


def test_the_setup_bat_probe_parses() -> None:
    """It is one long line inside a batch `for /f` backquote block, which is a quoting minefield.

    A syntax error here does not fail loudly: the `for /f` captures nothing, the batch default of
    RemoteSigned stands, and the mark of the web is never cleared -- so a user who unzipped a
    download gets a refusal with no hint that the probe was the thing that broke.
    """
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is unavailable")
    probe = _setup_bat_probe()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.ps1"
        path.write_text(probe, encoding = "utf-8")
        result = _run_pwsh_parse(pwsh, path)
    assert (
        result.returncode == 0
    ), f"the probe embedded in studio/setup.bat does not parse:\n{result.stdout}\n{result.stderr}"


def _run_pwsh_parse(pwsh: str, path: Path):
    import os
    from unsloth_pwsh_runner import run_pwsh

    script = (
        "$errors = $null; $tokens = $null; "
        "$null = [System.Management.Automation.Language.Parser]::ParseFile("
        "$env:UNSLOTH_TARGET, [ref]$tokens, [ref]$errors); "
        "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    return run_pwsh(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = 120,
        env = {**os.environ, "UNSLOTH_TARGET": str(path)},
    )


def test_setup_bat_steps_down_to_bypass_only_for_a_remote_script() -> None:
    """The one case where RemoteSigned is a real regression, handled the way install.ps1 handles it.

    Execution policy is judged on the script file's ZONE. A dotted-FQDN UNC, a DFS path or an
    IP-literal share is the Internet zone, where RemoteSigned refuses an unsigned script and
    `Unblock-File` cannot help: with no `Zone.Identifier` stream present the path decides, and there
    is nothing to clear. `install.ps1` already steps down to Bypass for exactly this when it writes
    the shortcut; reusing that logic beats inventing a second answer.
    """
    probe = _setup_bat_probe()
    assert (
        "DriveInfo" in probe and "Network" in probe
    ), "the probe no longer detects a mapped network drive, so a script on H:/Z: would be refused"
    assert "-like '\\\\*'" in probe, "the probe no longer detects a UNC path"
    assert (
        "'Bypass'" in probe and "'RemoteSigned'" in probe
    ), "the probe no longer chooses between the two policies"
    assert (
        "Unblock-File" in probe
    ), "the probe no longer clears the mark of the web, so an unzipped download is refused"

    # The launch line, the -NoProfile asymmetry and the Unblock-File ordering are asserted by
    # test_setup_bat_clears_the_mark_before_loading_under_remotesigned above; not repeated here.


# Whole-line comments only. Both defects this guards against were whole-line, and a trailing `#`
# cannot be told from a `#` inside a string without re-parsing, which would trade a real check for
# a source of false alarms.
_COMMENT_PREFIXES = {".bat": ("rem ", "::"), ".ps1": ("#",), ".sh": ("#",)}

# A repo-relative path, which is a claim about THIS tree, as opposed to a PR number or a URL.
# Anchored on the real top-level directories and required to carry a file extension, so
# `unsloth.ai/install.ps1` (a URL) and a bare directory mention do not match.
_REPO_PATH_IN_PROSE = re.compile(
    r"(?<![\w./-])((?:\.github|docs|tests|scripts|studio|unsloth|unsloth_cli|unsloth_zoo)"
    r"/[\w./-]+\.\w+)"
)


def _comment_lines(text: str, name: str):
    prefixes = _COMMENT_PREFIXES[Path(name).suffix]
    for line in text.splitlines():
        stripped = line.strip().lower() if name.endswith(".bat") else line.strip()
        if stripped.startswith(prefixes):
            yield line


@pytest.mark.parametrize("name", DOCUMENTED_SCRIPTS)
def test_a_comment_never_points_at_a_file_that_is_not_here(name: str) -> None:
    """A comment citing evidence must cite something a reader can actually open.

    Twice now a shipped script has carried a pointer to a file that was not in the tree: first
    `docs/windows-installer-av-shapes.md` after the doc was folded into AV_SHAPES_RECORD, then
    `.github/workflows/windows-vt-preflight.yml`, which lives in a separate PR and therefore does
    not exist on this branch at all. Both read as authoritative and neither could be followed, which
    is worse than saying nothing: the justification for deleting a native call becomes unverifiable.
    Referring to a PR number is fine and stays true; referring to a path is a claim about this tree.
    """
    text = (REPO / name).read_text(encoding = "utf-8")
    cited = set()
    for line in _comment_lines(text, name):
        cited.update(_REPO_PATH_IN_PROSE.findall(line))

    missing = sorted(p for p in cited if not (REPO / p).exists())
    assert not missing, (
        f"{name} has a comment pointing at {missing}, which is not in this tree. Cite a PR number, "
        "or cite tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD), which travels with the repo."
    )

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The helpers that must work under Constrained Language Mode may only use what CLM allows.

CLM is one of the two policies that stop the installer defining a P/Invoke type at runtime, so
every fallback rung exists precisely for hosts running under it. A rung that itself uses a
construct CLM refuses is absent from the population it was written for, and it fails silently:
the ladder catches the throw and moves on.

This is a STATIC check on purpose. The behavioural suite runs the same helpers inside a real
constrained runspace, but a constrained runspace on Linux under pwsh 7 does not enforce the
allowed-type list the way Windows PowerShell 5.1 does, so that suite passed green on this
repository's own CI for a build in which three of these helpers could not run on Windows at all.
The rules below come from the 5.1 documentation rather than from an observation on one host:
https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_language_modes?view=powershell-5.1

  "Only allowed types can be used in PowerShell. Other types aren't permitted."
  "Users can get all properties of allowed types."
  "[ref] - Casting an object to type [ref] ... is not permitted."
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = ROOT / "install.ps1"

# Every helper reachable while the session may be constrained. Adding a rung to the ladder means
# adding it here too: a name that is not listed is simply not checked.
CLM_REACHABLE = (
    "Invoke-StudioEarlyPythonScriptViaCmdlets",
    "New-StudioChildScriptDirectory",
    "Test-StudioChildScriptDirectoryElevated",
    "Get-StudioEarlyPython",
    "Invoke-StudioEarlyPython",
    "Get-StudioPythonFinalPath",
    "Resolve-StudioFinalPathsInOneChild",
    "Get-StudioPythonProcessImageTable",
    "Remove-StudioTrailingNewline",
    "Get-RunningStudioVenvProcesses",
    # The NVIDIA inventory's Python rung. It is REACHED when the emitted probe type declined,
    # and Constrained Language Mode is the commonest reason it declines, so this one runs under
    # CLM more often than any other helper here.
    "Read-NvidiaLibraryRawViaPython",
)

# The allowed-type list from the 5.1 documentation, lowercased, restricted to the spellings this
# installer actually uses. A static call on anything else throws under CLM.
CLM_ALLOWED_TYPES = {
    "array",
    "bool",
    "byte",
    "char",
    "cultureinfo",
    "datetime",
    "decimal",
    "double",
    "float",
    "guid",
    "hashtable",
    "int",
    "int16",
    "int32",
    "int64",
    "long",
    "psobject",
    "pscustomobject",
    "regex",
    "sbyte",
    "single",
    "string",
    "switch",
    "timespan",
    "uint16",
    "uint32",
    "uint64",
    "uri",
    "version",
    "void",
    "xml",
    # Spelled out in full as well, since both forms appear in the file.
    "system.string",
    "system.guid",
    "system.datetime",
    "system.timespan",
    "system.uri",
    "system.version",
    "system.text.regularexpressions.regex",
}

# Property reads CLM refuses because the object is not of an allowed type. Each of these was
# live in the installer and each one broke a rung on Windows PowerShell 5.1.
FORBIDDEN_PROPERTIES = {
    "FullName": "System.IO.FileInfo is not a CLM allowed type; build the path as a string",
    "ExitCode": "System.Diagnostics.Process is not a CLM allowed type; use Select-Object -ExpandProperty",
    "HasExited": "System.Diagnostics.Process is not a CLM allowed type; use Wait-Process -ErrorVariable",
    "MainModule": "System.Diagnostics.Process is not a CLM allowed type",
    "Source": "System.Management.Automation.CommandInfo is not a CLM allowed type; use Select-Object -ExpandProperty",
}

# Cmdlets whose return value is an object of a type CLM will not let script read properties from.
FORBIDDEN_CMDLETS = {
    "New-TemporaryFile": "returns a FileInfo whose path cannot be read under CLM; compose the path from $env:TEMP and a [guid]",
}


def _extent(text: str, name: str) -> str:
    """The full source of a named function, by brace matching.

    A regex to the next 'function ' keyword overshoots into the following helper, which makes
    every count below wrong in the permissive direction.
    """
    start = text.index(f"function {name} {{")
    depth, i = 0, start
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1


def _code_lines(body: str) -> list[str]:
    """Source with whole-line comments dropped.

    The comments here NAME the forbidden constructs, explaining why they are absent. Matching
    raw text would fail on the explanation instead of on the code.
    """
    out = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out.append(line.split(" #", 1)[0] if " #" in line else line)
    return out


def _joined_pipelines(code: str) -> str:
    """One pipeline per line.

    PowerShell continues a pipeline after a trailing `|`, so the projection that makes a
    Get-Process safe often sits on the NEXT line. Read line by line, the rule below fired on the
    formatting rather than on the code.
    """
    out: list[str] = []
    for line in code.split("\n"):
        if out and out[-1].rstrip().endswith("|"):
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


SOURCE = INSTALL_PS1.read_text(encoding = "utf-8")
BODIES = {name: _extent(SOURCE, name) for name in CLM_REACHABLE}


@pytest.mark.parametrize("name", CLM_REACHABLE)
def test_every_static_call_is_on_a_clm_allowed_type(name: str):
    code = "\n".join(_code_lines(BODIES[name]))
    used = {m.lower() for m in re.findall(r"\[([\w.]+)\]::", code)}
    forbidden = sorted(used - CLM_ALLOWED_TYPES)
    assert not forbidden, (
        f"{name} makes a static call on {forbidden}, which Constrained Language Mode refuses. "
        "Use the cmdlet equivalent: Split-Path -Parent for GetDirectoryName, "
        "Split-Path -IsAbsolute for IsPathRooted."
    )


@pytest.mark.parametrize("name", CLM_REACHABLE)
def test_no_property_is_read_off_a_disallowed_type(name: str):
    code = "\n".join(_code_lines(BODIES[name]))
    hits = sorted(
        {
            prop
            for prop in FORBIDDEN_PROPERTIES
            if re.search(r"\$\w+(\.\w+)*\." + prop + r"\b", code)
        }
    )
    assert not hits, "; ".join(f"{name} reads .{p}: {FORBIDDEN_PROPERTIES[p]}" for p in hits)


@pytest.mark.parametrize("name", CLM_REACHABLE)
def test_no_cmdlet_hands_back_an_object_clm_cannot_read(name: str):
    code = "\n".join(_code_lines(BODIES[name]))
    hits = sorted({c for c in FORBIDDEN_CMDLETS if re.search(r"\b" + c + r"\b", code)})
    assert not hits, "; ".join(f"{name} calls {c}, which {FORBIDDEN_CMDLETS[c]}" for c in hits)


@pytest.mark.parametrize("name", CLM_REACHABLE)
def test_nothing_casts_to_ref(name: str):
    # "[ref] - Casting an object to type [ref] or [Management.Automation.PSReference] is not
    # permitted." That rules out [int]::TryParse($text, [ref]$out), whose obvious replacement is
    # a -match on '^\\d+$' followed by an [int] cast.
    code = "\n".join(_code_lines(BODIES[name]))
    assert not re.search(
        r"\[ref\]\s*\$", code
    ), f"{name} casts to [ref], which Constrained Language Mode does not permit"


def test_the_checks_are_not_vacuous():
    """Every rule above must actually fire on the construct it bans.

    Without this the suite would pass just as happily against an empty allowlist, an unanchored
    regex, or a comment-stripper that ate the whole body.
    """
    sample = """function Fake-Helper {
        $f = New-TemporaryFile
        $dir = [System.IO.Path]::GetDirectoryName($f.FullName)
        $null = [int]::TryParse($text, [ref]$parsed)
        if ($proc.ExitCode -ne 0) { return $null }
        $where = $cmd.Source
    }"""
    code = "\n".join(_code_lines(_extent(sample, "Fake-Helper")))
    assert sorted({m.lower() for m in re.findall(r"\[([\w.]+)\]::", code)} - CLM_ALLOWED_TYPES) == [
        "system.io.path"
    ]
    assert sorted(
        p for p in FORBIDDEN_PROPERTIES if re.search(r"\$\w+(\.\w+)*\." + p + r"\b", code)
    ) == [
        "ExitCode",
        "FullName",
        "Source",
    ]
    assert [c for c in FORBIDDEN_CMDLETS if re.search(r"\b" + c + r"\b", code)] == [
        "New-TemporaryFile"
    ]
    assert re.search(r"\[ref\]\s*\$", code)


@pytest.mark.parametrize("name", CLM_REACHABLE)
def test_no_raw_process_object_is_iterated(name: str):
    """Get-Process hands back System.Diagnostics.Process, which CLM will not let script read.

    This one is not a static call or a single named property, it is a SHAPE: the objects come out
    of a cmdlet, so nothing above flags them, and then the loop body reads .Id off each one. Under
    Constrained Language Mode that throws, and in the one place it mattered it threw inside a
    `catch { continue }`, so every process was skipped in silence and the live-process guard went
    blind on exactly the hosts these fallbacks exist for.

    Select-Object projects into a PSCustomObject, which IS on the allowed list, and does the read
    itself inside compiled code. So the rule is: a Get-Process whose output this code walks must be
    piped through Select-Object first.
    """
    code = _joined_pipelines("\n".join(_code_lines(BODIES[name])))
    for match in re.finditer(r"Get-Process\b([^\r\n]*)", code):
        assert "Select-Object" in match.group(1), (
            f"{name} uses the output of Get-Process without projecting it: "
            f"{match.group(0).strip()!r}. Constrained Language Mode refuses property reads on "
            "System.Diagnostics.Process; pipe through Select-Object -Property Id, ProcessName."
        )


def test_the_raw_process_rule_is_not_vacuous():
    """It must fire on the unprojected spelling and stay quiet on the projected one."""
    bad = """function Fake-Scan {
        foreach ($p in @(Get-Process -ErrorAction SilentlyContinue)) { $null = $p.Id }
    }"""
    good = """function Fake-Scan {
        foreach ($p in @(Get-Process -ErrorAction SilentlyContinue | Select-Object -Property Id)) { $null = $p.Id }
    }"""
    wrapped = """function Fake-Scan {
        foreach ($p in @(Get-Process -ErrorAction SilentlyContinue |
            Select-Object -Property Id)) { $null = $p.Id }
    }"""

    def offends(sample: str) -> bool:
        code = _joined_pipelines("\n".join(_code_lines(_extent(sample, "Fake-Scan"))))
        return any(
            "Select-Object" not in m.group(1) for m in re.finditer(r"Get-Process\b([^\r\n]*)", code)
        )

    assert offends(bad)
    assert not offends(good)
    # And a pipeline broken across lines is still one pipeline, which is how the real one is
    # written; without the join the rule fired on its own formatting.
    assert not offends(wrapped)


def test_the_comment_stripper_keeps_the_code():
    """It must drop explanations without eating the statements they explain."""
    body = BODIES["Invoke-StudioEarlyPythonScriptViaCmdlets"]
    code = "\n".join(_code_lines(body))
    assert "Start-Process" in code
    # This helper's comments NAME the construct they explain the absence of, which is exactly
    # why stripping matters: without it the cmdlet rule would fail on its own explanation.
    assert "New-TemporaryFile" in body and "New-TemporaryFile" not in code

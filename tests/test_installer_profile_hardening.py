# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for install.ps1 under a user PowerShell profile.

The installer used to succeed only from a console started with -NoProfile. A profile runs before
"irm https://unsloth.ai/install.ps1 | iex" does and shares its scope, and there is no script file
to re-launch without it, so four couplings were cut individually: $PSDefaultParameterValues,
Set-StrictMode, $PSNativeCommandUseErrorActionPreference, and command discovery finding a profile
alias or function named "uv" ahead of PATH. A fifth lived outside install.ps1 -- the handoff to
`unsloth studio setup` passed -NoProfile only when stdout was not a tty, which is never true for
the console install this is about; setup.ps1 itself is not otherwise covered here.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
STUDIO_COMMAND = REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"


def _framed(record: str, *, banner: str = "") -> str:
    """What the probe child really prints: the framed record, plus whatever the profile said."""
    from unsloth_cli.commands import studio as studio_cmd
    return f"{banner}{studio_cmd._PROXY_PROBE_BEGIN}\n{record}\n{studio_cmd._PROXY_PROBE_END}\n"


requires_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


def _locate(haystack: str, needle: str, what: str) -> int:
    """str.index, but a removed anchor reads as a failed guard rather than a ValueError."""
    index = haystack.find(needle)
    if index < 0:
        pytest.fail(f"install.ps1 no longer contains {what}: {needle!r}")
    return index


def _extract_function(name: str) -> str:
    """install.ps1's own function text, verbatim, so these tests cannot drift from it."""
    src = _install_ps1()
    start = _locate(src, f"    function {name} {{", f"the {name} helper")
    end = _locate(src[start:], "\n    }\n", f"the end of {name}") + start
    return src[start : end + len("\n    }\n")]


def _code_only() -> str:
    """install.ps1 with whole-line comments blanked, so ordering is judged on what executes.

    The hardening block names the very cmdlets it protects, and those mentions would otherwise
    read as the first use of each.
    """
    return "\n".join(
        "" if line.lstrip().startswith("#") else line for line in _install_ps1().splitlines()
    )


def _extract_prologue() -> str:
    """The profile-hardening block, from its opening comment through the last $script: reset."""
    src = _install_ps1()
    start = _locate(src, "    # The user's PowerShell profile has already run", "the prologue")
    end = _locate(src[start:], "$script:UnslothVerbose = ($env:UNSLOTH_VERBOSE", "the prologue end")
    return src[start : start + end]


def _ps_literal(value: object) -> str:
    """A single-quoted PowerShell literal. Doubling escapes an apostrophe, which a Windows
    account named O'Brien puts in tmp_path; unsloth_cli/commands/studio.py does the same."""
    return "'" + str(value).replace("'", "''") + "'"




def test_prologue_neutralizes_profile_state():
    block = _extract_prologue()
    assert (
        "Set-StrictMode -Off" in block
    ), "this script's $env:X -in @(...) and unassigned-$script: idioms are errors under Latest"
    assert (
        "$PSDefaultParameterValues = $_UnslothKeptDefaults" in block
    ), "a profile entry like 'Start-Process:WindowStyle' otherwise rebinds cmdlets here"
    assert (
        "$PSNativeCommandUseErrorActionPreference = $false" in block
    ), "with it on, a failing native command throws past Exit-InstallFailure and skips rollback"


def test_proxy_defaults_are_carried_across_rather_than_dropped():
    """Clearing the table wholesale would break the corporate hosts where a profile proxy
    entry is the only route to python.org -- and those users are the ones who never run
    -NoProfile, so they would be trading one broken install for another."""
    block = _extract_prologue()
    assert (
        ":Proxy(Credential|UseDefaultCredentials)?$" in block
    ), "the filter must keep proxy-shaped keys, which can only ever enable a download"


def test_profile_hardening_precedes_every_use_it_protects():
    """Ordering, not presence: a fix applied after the first unset-env test or the first
    download protects nothing.

    Only in-process first uses are anchored. The first textual Invoke-RestMethod and
    Start-Process are inside the launcher here-string, which is text for a separate process
    that install.ps1 starts with -NoProfile of its own.
    """
    code = _code_only()
    strict_idx = _locate(code, "Set-StrictMode -Off", "the strict-mode pin")
    defaults_idx = _locate(code, "$PSDefaultParameterValues = $_UnslothKeptDefaults", "the filter")
    native_idx = _locate(
        code, "$PSNativeCommandUseErrorActionPreference = $false", "the native pin"
    )
    for marker, what in (
        ("$env:UNSLOTH_NO_TORCH -in @(", "the first unset-env test"),
        ("Invoke-WebRequest", "the first in-process download"),
        (
            "Invoke-ManagedUnslothCli -Python $VenvPython -Arguments $studioArgs",
            "the setup handoff",
        ),
    ):
        idx = _locate(code, marker, what)
        assert strict_idx < idx, f"{what} must be reached with strict mode pinned"
        assert defaults_idx < idx, f"{what} must be reached with the profile's defaults filtered"
        assert native_idx < idx, f"{what} must be reached with native error handling pinned"


def test_script_scoped_uv_state_is_reset_per_invocation():
    """Same hazard as $script:IsIntelXpu: under irm | iex, $script: is the caller's session."""
    src = _install_ps1()
    # Anchored on the newline plus exactly four spaces:
    for reset, first_read in (
        ("\n    $script:UvExe = 'uv'\n", "& $script:UvExe"),
        ("\n    $script:UvInstallDestDir = $null\n", "foreach ($d in @($script:UvInstallDestDir"),
    ):
        init_idx = _locate(src, reset, "a per-invocation reset")
        assert init_idx < _locate(
            src, first_read, "the first read"
        ), f"{reset.strip()} must come before the first read"
    assert (
        "$script:UvExe = 'uv'" in src
    ), "the reset value must stay the bare token, so an unforeseen path behaves as it does today"


def test_no_bare_uv_token_survives_at_a_call_site():
    """A profile alias or function named uv outranks PATH, so every invocation goes through
    the resolved executable."""
    src = _install_ps1()
    assert "{ uv " not in src, "install scriptblocks must invoke $script:UvExe, not the bare token"
    assert "& uv " not in src, "the version probe must invoke the resolved executable"
    assert (
        src.count("{ & $script:UvExe ") >= 27
    ), "every uv scriptblock must route through the resolved path"


def test_uv_is_resolved_as_an_application():
    body = _extract_function("Get-UvExecutableCandidates")
    # The whole invocation, not the flags separately:
    assert "Get-Command uv -CommandType Application -All -ErrorAction SilentlyContinue" in body, (
        "Application-only lookup is what skips an alias or function, and ordering across "
        "several matches is only documented for -All"
    )
    assert "return 'uv'" not in body, (
        "the bare token must NOT come back as a fallback: a profile `function uv` that answers "
        "--version with a plausible number would clear the version gate and then receive every "
        "install command, which is the hijack this function exists to stop"
    )
    assert "-CommandType Alias" in body and "ResolvedCommand" in body, (
        "an alias pointing at a real executable is still a legitimate way to have uv, so follow "
        "it -- but only as far as an Application, and hand back the resolved path"
    )
    probe = _extract_function("Test-UvVersionOk") + _extract_function("Test-UvCandidateVersion")
    assert (
        "Get-UvExecutableCandidates" in probe and "$script:UvExe = $exe" in probe
    ), "the probe must pin the executable that answered, so later PATH edits cannot swap it"


def test_setup_ps1_handoff_never_inherits_the_profile():
    """install.ps1 ends by running `unsloth studio setup`, which re-enters PowerShell. That
    launch used to add -NoProfile only when stdout was not a tty, so the console install this
    whole file is about ran setup.ps1 with the user's profile loaded and its bare `uv` calls
    exposed to the same alias."""
    src = STUDIO_COMMAND.read_text(encoding = "utf-8")
    start = _locate(src, "powershell_args = [powershell]", "the setup.ps1 launch")
    branch = _locate(src[start:], "_should_hide_windows_subprocesses()", "the hidden-window branch")
    assert (
        '"-NoProfile"' in src[start : start + branch]
    ), "-NoProfile must be added unconditionally, before the hidden-window branch"




_HOSTILE_PROFILE = """
Set-StrictMode -Version Latest
Set-Alias uv Write-Host
$PSDefaultParameterValues['Invoke-WebRequest:Proxy'] = 'http://127.0.0.1:9'
$PSDefaultParameterValues['Get-Date:Format'] = 'HOSTILE'
$PSNativeCommandUseErrorActionPreference = $true
"""


def _write_exe(directory: Path, stem: str, posix_body: str, cmd_body: str) -> Path:
    """A tiny executable discoverable as an Application on either platform."""
    directory.mkdir(parents = True, exist_ok = True)
    if os.name == "nt":
        exe = directory / f"{stem}.cmd"
        # newline = "" so write_text does not translate \n and leave \r\r\n on disk.
        exe.write_text(cmd_body, encoding = "ascii", newline = "")
    else:
        exe = directory / stem
        exe.write_text(posix_body, encoding = "ascii", newline = "")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def _fake_uv(directory: Path, version: str = "0.12.1") -> Path:
    """A uv on PATH that reports a modern version."""
    return _write_exe(
        directory,
        "uv",
        f'#!/bin/sh\necho "uv {version}"\n',
        f"@echo uv {version}\r\n",
    )


def _fake_failing_exe(directory: Path) -> Path:
    """A native command that exits non-zero, without being an error in itself."""
    return _write_exe(directory, "failer", "#!/bin/sh\nexit 7\n", "@exit /b 7\r\n")


def _hostile_env(
    tmp_path: Path,
    path_prepend: Path | None = None,
    path_override: Path | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    home = tmp_path / "home"
    home.mkdir(parents = True, exist_ok = True)
    drive, tail = os.path.splitdrive(str(home))
    env.update({"HOME": str(home), "USERPROFILE": str(home), "HOMEDRIVE": drive, "HOMEPATH": tail})
    # HOME on its own does not move $PROFILE on Unix: PowerShell reads $XDG_CONFIG_HOME first and only falls back to
    # $HOME/.config when it is unset, and GitHub's ubuntu image writes XDG_CONFIG_HOME into /etc/environment, so an
    # inherited value kept naming the real account and this file's isolation leaked on a hosted runner and nowhere else.
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    if path_override is not None:
        path_override.mkdir(parents = True, exist_ok = True)
        env["PATH"] = str(path_override)
    if path_prepend is not None:
        env["PATH"] = str(path_prepend) + os.pathsep + env.get("PATH", "")
    return env


_PROFILE_SCOPES = (
    "AllUsersAllHosts",
    "AllUsersCurrentHost",
    "CurrentUserAllHosts",
    "CurrentUserCurrentHost",
)


def _profile_paths(env: dict[str, str]) -> dict[str, Path]:
    """The four profile paths pwsh itself reports under `env`.

    Asked rather than assumed. The path this replaces was hardcoded to the ".config under $HOME"
    branch, which is only PowerShell's fallback when XDG_CONFIG_HOME is unset, so on a host that
    exports it the fixture wrote a file pwsh never opened and the profile silently did not apply.
    """
    # pwsh reporting nothing rather than as pwsh never having started.
    # This asks pwsh where its own profiles live, and every fixture below plants a file at the answer.
    res = run_pwsh(
        [
            shutil.which("pwsh") or "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "foreach ($n in "
            + ", ".join(_ps_literal(scope) for scope in _PROFILE_SCOPES)
            + ') { "$n=$($PROFILE.$n)" }',
        ],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 120,
        env = env,
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    reported = dict(line.strip().split("=", 1) for line in res.stdout.splitlines() if "=" in line)
    return {scope: Path(reported.get(scope, "")) for scope in _PROFILE_SCOPES}


def _run_with_profile(
    tmp_path: Path,
    body: str,
    path_prepend: Path | None = None,
    path_override: Path | None = None,
    profile: str | None = None,
) -> subprocess.CompletedProcess:
    """Run `body` with the hostile profile dot-sourced into the scope that encloses it.

    That is the scope relationship a profile has to the installer: the profile runs at global
    scope, then "irm ... | iex" defines and calls Install-UnslothStudio in that same scope.
    Dot-sourcing reproduces it on every OS, which planting a profile file does not -- Windows
    resolves the Documents folder through the known-folder API, so no environment variable can
    redirect $PROFILE at a fixture. test_a_real_profile_reproduces_the_same_state anchors this
    against a genuinely loaded profile where that is possible.
    """
    profile_path = tmp_path / "hostile_profile.ps1"
    profile_path.write_text(_HOSTILE_PROFILE if profile is None else profile, encoding = "utf-8")
    script = tmp_path / "body.ps1"
    script.write_text(f". {_ps_literal(profile_path)}\n{body}\n", encoding = "utf-8")
    # Absolute path: PATH is replaced in some cases, so pwsh could not be found by name.
    return run_pwsh(
        [shutil.which("pwsh") or "pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 120,
        env = _hostile_env(tmp_path, path_prepend, path_override),
    )


_STATE_PROBE = "\n".join(
    [
        '"ALIAS:$([bool](Get-Alias uv -ErrorAction SilentlyContinue))"',
        "\"PROXY:$($PSDefaultParameterValues['Invoke-WebRequest:Proxy'])\"",
        '"NATIVEEAP:$PSNativeCommandUseErrorActionPreference"',
        'try { $q = $global:neverAssigned; "STRICT:off" } catch { "STRICT:on" }',
    ]
)

_PROBE_PREFIXES = ("ALIAS:", "PROXY:", "NATIVEEAP:", "STRICT:")


def _probe_lines(stdout: str) -> list[str]:
    return [line.strip() for line in stdout.splitlines() if line.startswith(_PROBE_PREFIXES)]


@pytest.mark.skipif(
    os.name == "nt", reason = "Windows resolves $PROFILE through the known-folder API"
)
@requires_pwsh
def test_a_real_profile_reproduces_the_same_state(tmp_path):
    """Anchor for the simulation: a profile pwsh actually loads leaves the same state.

    If these ever diverge, every test below is measuring something the installer never meets.
    Only the probe lines are compared, so a system-wide profile that prints a banner does not
    turn this into a flake.
    """
    env = _hostile_env(tmp_path)
    paths = _profile_paths(env)
    # The two machine-wide profiles load into the real leg only and no environment variable can move them, so one
    for scope in ("AllUsersAllHosts", "AllUsersCurrentHost"):
        if paths[scope].is_file():
            pytest.skip(f"a machine-wide profile at {paths[scope]} loads into the real leg only")
    profile = paths["CurrentUserCurrentHost"]
    # Not an assert: on a host whose $PROFILE cannot be redirected into the fixture at all, the real leg would load
    # whatever the actual account has (or nothing) and prove nothing either way.
    # XDG_CONFIG_HOME in _hostile_env is what keeps this true on Linux and macOS, so the skip is unreachable there and a
    if tmp_path not in profile.parents:
        pytest.skip(f"pwsh resolves $PROFILE to {profile}, which this fixture cannot plant into")
    profile.parent.mkdir(parents = True, exist_ok = True)
    profile.write_text(_HOSTILE_PROFILE, encoding = "utf-8")
    script = tmp_path / "real.ps1"
    script.write_text(_STATE_PROBE, encoding = "utf-8")

    # would be read as the two diverging rather than as one of them never having run.
    # NoProfile deliberately omitted;
    real = run_pwsh(
        [shutil.which("pwsh") or "pwsh", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 120,
        env = env,
    )
    assert real.returncode == 0, f"stdout={real.stdout!r} stderr={real.stderr!r}"
    simulated = _run_with_profile(tmp_path, _STATE_PROBE)
    # All four probes coming back at their defaults means the file was planted somewhere pwsh does not read, which is a
    assert _probe_lines(real.stdout) == _probe_lines(simulated.stdout), (
        f"dot-sourced profile diverges from the one pwsh loaded from {profile}: "
        f"{_probe_lines(simulated.stdout)} vs {_probe_lines(real.stdout)}"
    )


@requires_pwsh
def test_the_hostile_profile_really_is_hostile(tmp_path):
    """Control. Without it, every assertion below could pass on a profile that never applied."""
    res = _run_with_profile(tmp_path, _STATE_PROBE, path_prepend = _fake_uv(tmp_path / "bin").parent)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert _probe_lines(res.stdout) == [
        "ALIAS:True",
        "PROXY:http://127.0.0.1:9",
        "NATIVEEAP:True",
        "STRICT:on",
    ], f"the profile did not apply; the tests below would be vacuous: {res.stdout!r}"


@requires_pwsh
def test_bare_uv_token_is_hijacked_by_the_profile(tmp_path):
    """The bug itself: the pre-fix `& uv --version` runs the alias, so a working uv reads as
    missing and the install stops at 'uv could not be installed'."""
    res = _run_with_profile(
        tmp_path,
        "$raw = (& uv --version 2>$null | Select-Object -First 1)\n"
        '"RAW:[$raw]"\n'
        "if ($raw -match 'uv\\s+([0-9]+(?:\\.[0-9]+)+)') "
        "{ \"MATCHED:$($Matches[1])\" } else { 'MATCHED:none' }\n",
        path_prepend = _fake_uv(tmp_path / "bin").parent,
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert (
        "MATCHED:none" in res.stdout
    ), "if the bare token resolves past the alias, this suite can no longer detect a regression"
    # Write-Host swallowed the arguments instead of running uv, which is the whole failure mode.
    assert "RAW:[]" in res.stdout, f"expected the alias to eat the call; got {res.stdout!r}"


def _guarded(*body: str) -> str:
    """The prologue, verbatim from install.ps1, wrapped in the function it lives in."""
    return "\n".join(
        ["function Install-UnslothStudio {", '    $ErrorActionPreference = "Stop"']
        + [_extract_prologue()]
        + list(body)
        + ["}"]
    )


@requires_pwsh
def test_prologue_clears_profile_state_without_disturbing_the_session(tmp_path):
    body = "\n".join(
        [
            _guarded(
                '    "INSIDE_DEFAULTS:$($PSDefaultParameterValues.Count)"',
                "    \"INSIDE_PROXY:$($PSDefaultParameterValues['Invoke-WebRequest:Proxy'])\"",
                '    "INSIDE_NATIVEEAP:$PSNativeCommandUseErrorActionPreference"',
                "    Show-NestedState",
                "    if ($env:UNSLOTH_TOTALLY_UNSET -in @('1', 'true'))"
                ' { "INSIDE_ENVTEST:hit" } else { "INSIDE_ENVTEST:ok" }',
                '    "INSIDE_UVEXE:$script:UvExe"',
                '    "INSIDE_DESTDIR:[$script:UvInstallDestDir]"',
            ),
            "function Show-NestedState {",
            '    "NESTED_DEFAULTS:$($PSDefaultParameterValues.Count)"',
            "    \"NESTED_DATE:$(Get-Date -Date '2020-01-02T03:04:05')\"",
            '    & { if ($script:neverAssignedAnywhere) { "SB:hit" } else { "SB:ok" } }',
            "}",
            "Install-UnslothStudio",
            '"AFTER_DEFAULTS:$($PSDefaultParameterValues.Count)"',
            '"AFTER_NATIVEEAP:$PSNativeCommandUseErrorActionPreference"',
            'try { $q = $global:neverAssigned2; "AFTER_STRICT:off" } catch { "AFTER_STRICT:on" }',
        ]
    )
    res = _run_with_profile(tmp_path, body)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    # The interfering default is gone;
    assert "INSIDE_DEFAULTS:1" in res.stdout
    assert "NESTED_DEFAULTS:1" in res.stdout
    assert "INSIDE_PROXY:http://127.0.0.1:9" in res.stdout, "a proxy default must survive"
    assert "HOSTILE" not in res.stdout, "a nested cmdlet still bound the profile's Get-Date default"
    assert "INSIDE_NATIVEEAP:False" in res.stdout
    # Strict mode off for the idioms that predate it, in nested functions and scriptblocks too.
    assert "INSIDE_ENVTEST:ok" in res.stdout
    assert "SB:ok" in res.stdout
    # The $script: state a second run must not inherit.
    assert "INSIDE_UVEXE:uv" in res.stdout
    assert "INSIDE_DESTDIR:[]" in res.stdout
    # And the user's own session is left exactly as the profile configured it.
    assert "AFTER_DEFAULTS:2" in res.stdout, "the caller's default-parameter table must survive"
    assert "AFTER_NATIVEEAP:True" in res.stdout, "the caller's native error handling must survive"
    assert "AFTER_STRICT:on" in res.stdout, "the caller's strict mode must survive"


@requires_pwsh
def test_a_profile_cannot_make_a_failing_native_command_terminating(tmp_path):
    """$PSNativeCommandUseErrorActionPreference plus this script's $ErrorActionPreference = Stop
    turns a non-zero exit into a throw, which would leave the setup handoff bypassing
    Exit-InstallFailure and its rollback."""
    failer = _fake_failing_exe(tmp_path / "bin")
    body = "\n".join(
        [
            _guarded(
                f"    & {_ps_literal(failer)}",
                '    "AFTER_NATIVE:$LASTEXITCODE"',
            ),
            'try { Install-UnslothStudio } catch { "THREW:$($_.Exception.GetType().Name)" }',
        ]
    )
    res = _run_with_profile(tmp_path, body)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert (
        "THREW:" not in res.stdout
    ), f"a failing native command became terminating: {res.stdout!r}"
    assert (
        "AFTER_NATIVE:7" in res.stdout
    ), "the exit code must still be readable, which is how every caller here branches"


def _uv_probe_body(*extra: str) -> str:
    return "\n".join(
        [
            _guarded(
                '    $UvMinVersion = "0.8.16"',
                _extract_function("Get-UvExecutableCandidates"),
                _extract_function("Resolve-UvExecutable"),
                _extract_function("Test-UvCandidateVersion"),
                _extract_function("Test-UvVersionOk"),
                '    "OK:$(Test-UvVersionOk)"',
                '    "PINNED:$script:UvExe"',
                *extra,
            ),
            "Install-UnslothStudio",
        ]
    )


@requires_pwsh
def test_uv_probe_finds_the_real_uv_behind_a_profile_alias(tmp_path):
    fake = _fake_uv(tmp_path / "bin")
    # The stub stands in for Invoke-InstallCommand, which is how all 27 real call sites run:
    body = "\n".join(
        [
            "function Invoke-InstallCommandStub "
            '{ param([ScriptBlock]$Command); "CALLSITE:$(& $Command)" }',
            _uv_probe_body("    Invoke-InstallCommandStub { & $script:UvExe pip install nothing }"),
        ]
    )
    res = _run_with_profile(tmp_path, body, path_prepend = fake.parent)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert (
        "OK:True" in res.stdout
    ), "a uv on PATH must be detected even with an alias shadowing the name"
    assert (
        f"PINNED:{fake}" in res.stdout
    ), f"the probe must pin {str(fake)!r} exactly; got {res.stdout!r}"
    # The scriptblock form the installer actually uses must reach the executable, not the alias.
    assert (
        "CALLSITE:uv 0.12.1" in res.stdout
    ), "install scriptblocks must run the resolved uv; the alias would echo the arguments"


@requires_pwsh
def test_uv_probe_rejects_a_too_old_uv_and_leaves_the_reset_value(tmp_path):
    """The version gate still has to fail closed, or a stale uv would be pinned and used.

    PATH is REPLACED, not prepended to: since the gate now walks every candidate, a real uv on
    the machine running this would legitimately rescue the run and hide the branch under test.
    """
    fake = _fake_uv(tmp_path / "bin", version = "0.7.0")
    res = _run_with_profile(tmp_path, _uv_probe_body(), path_override = fake.parent)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "OK:False" in res.stdout
    assert "PINNED:uv" in res.stdout, "a rejected uv must not be pinned over the reset value"


@requires_pwsh
def test_uv_probe_reports_missing_when_only_the_alias_exists(tmp_path):
    """With no uv on PATH the installer must still take its install-uv branch, not pin the alias."""
    # The inherited PATH is replaced, not prepended to:
    res = _run_with_profile(tmp_path, _uv_probe_body(), path_override = tmp_path / "emptybin")
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "OK:False" in res.stdout
    assert "PINNED:uv" in res.stdout


@requires_pwsh
def test_a_convincing_uv_function_is_not_accepted_as_uv(tmp_path):
    """The dangerous shape is a wrapper that ANSWERS the version probe.

    An alias to a missing file fails loudly, so the earlier tests catch it. A profile
    `function uv` that prints a modern version number does not: it clears the version gate,
    gets pinned, and then receives every install command the script runs -- with the user's
    torch, index URL and venv path as arguments. Nothing on PATH, so only the wrapper exists.
    """
    profile = 'function uv { Write-Output "uv 99.0.0" }\n'
    body = _uv_probe_body('    "RESOLVED:$(Resolve-UvExecutable)"')
    res = _run_with_profile(tmp_path, body, path_override = tmp_path / "emptybin", profile = profile)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "RESOLVED:" in res.stdout and "RESOLVED:uv" not in res.stdout, (
        "a function named uv must not resolve to the bare token; it would then be pinned and "
        f"handed every install command. got {res.stdout!r}"
    )
    assert "OK:False" in res.stdout, "the version gate must not be satisfied by a wrapper"
    assert "PINNED:uv" in res.stdout, "the reset value must survive, so uv is installed for real"


@requires_pwsh
def test_an_alias_to_a_real_uv_is_followed_to_the_executable(tmp_path):
    """The legitimate wrapper shape still has to work, and resolve to a PATH, not the alias.

    People do alias uv at a specific build. Refusing that outright would install a second uv
    over a working machine, which is the failure this whole function was written to avoid.
    """
    fake = _fake_uv(tmp_path / "aliased", version = "0.12.4")
    profile = f"Set-Alias uv {_ps_literal(fake)}\n"
    body = _uv_probe_body('    "RESOLVED:$(Resolve-UvExecutable)"')
    res = _run_with_profile(tmp_path, body, path_override = tmp_path / "emptybin", profile = profile)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert f"RESOLVED:{fake}" in res.stdout, (
        f"an alias to a real executable must resolve to {str(fake)!r}, not to the alias name; "
        f"got {res.stdout!r}"
    )
    assert "OK:True" in res.stdout
    assert f"PINNED:{fake}" in res.stdout


def test_no_bare_winget_token_survives_at_a_call_site():
    """winget installs both Python and uv, so an alias on it owns the same ground.

    Detection used a bare Get-Command, and all five invocations were bare tokens, so the fix
    applied to uv left the other half of the bootstrap exposed to the identical wrapper.
    """
    src = _install_ps1()
    assert (
        "Get-Command winget -CommandType Application -All" in src
    ), "winget detection must skip aliases and functions, exactly as uv detection does"
    for bare in ("\n                winget ", "\n                    winget ", "{ winget "):
        assert bare not in src, f"winget must be invoked through the resolved path, found {bare!r}"
    assert (
        src.count("& $script:WingetExe ") >= 5
    ), "every winget invocation must route through the resolved path"


def test_module_autoloading_is_restored():
    """A profile setting 'None' removes Test-Path, Write-Host and Invoke-WebRequest on pwsh 7,
    which loads no modules at startup. Windows PowerShell 5.1 preloads them and survives, which
    is what makes this reproduce on one machine and not another."""
    src = _install_ps1()
    idx = _locate(src, "$PSModuleAutoLoadingPreference = 'All'", "the autoloading reset")
    assert idx < _locate(src, "Write-TauriLog", "the first cmdlet that needs a module")


@requires_pwsh
def test_install_ps1_parses():
    """A syntax error here is a total install failure, and the file is not imported by anything."""
    # A nonzero exit here is claimed to mean install.ps1 has a syntax error, and pwsh aborting before it ever reached
    res = run_pwsh(
        [
            shutil.which("pwsh") or "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$errs = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile("
            f"{_ps_literal(INSTALL_PS1)}, [ref]$null, [ref]$errs); "
            'if ($errs) { $errs | ForEach-Object { "ERR $($_.Extent.StartLineNumber): '
            '$($_.Message)" }; exit 1 }',
        ],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 120,
    )
    assert res.returncode == 0, res.stdout + res.stderr




def _proxy_prelude() -> str:
    """The PowerShell the setup launch prepends to its -Command, read from the shipped source."""
    src = STUDIO_COMMAND.read_text(encoding = "utf-8")
    start = _locate(src, "_PS_PROXY_DEFAULTS_PRELUDE = (", "the proxy handoff prelude")
    namespace: dict = {}
    exec(src[start : src.index(")\n", start) + 1], namespace)  # noqa: S102 - our own source
    return namespace["_PS_PROXY_DEFAULTS_PRELUDE"]


def test_the_setup_launch_reapplies_the_proxy_it_told_the_child_to_forget():
    """-NoProfile on the setup handoff drops the whole $PSDefaultParameterValues table, proxy
    entries included, and setup.ps1 downloads on its own (the VC++ runtime, the uv installer).
    Where a profile proxy entry is the only egress, keeping it in install.ps1 and losing it one
    process later is the same broken install. A PowerShell variable cannot cross a process
    boundary, so it travels as environment."""
    assert "_UNSLOTH_PS_PROXY_DEFAULTS" in _install_ps1(), "install.ps1 must publish the handoff"
    prelude = _proxy_prelude()
    assert "_UNSLOTH_PS_PROXY_DEFAULTS" in prelude, "the child must read it back"

    src = STUDIO_COMMAND.read_text(encoding = "utf-8")
    start = _locate(src, "powershell_args = [powershell]", "the setup.ps1 launch")
    # The f-string itself, not the comment above it that also quotes *>&1.
    command = _locate(src[start:], 'f"', "the -Command f-string")
    assert (
        "_PS_PROXY_DEFAULTS_PRELUDE" in src[start + command : start + command + 200]
    ), "the prelude has to run before setup.ps1, not merely exist"


@requires_pwsh
def test_only_serializable_proxy_keys_reach_the_child(tmp_path):
    """A credential is deliberately left behind: PSCredential does not survive ConvertTo-Json,
    and an environment variable is the wrong place for one. Everything non-proxy stays dropped."""
    block = tmp_path / "block.ps1"
    block.write_text(_extract_prologue(), encoding = "utf-8", newline = "")
    driver = tmp_path / "drive.ps1"
    driver.write_text(
        "$PSDefaultParameterValues = @{\n"
        "  'Invoke-WebRequest:Proxy' = 'http://proxy.corp:8080'\n"
        "  'Invoke-WebRequest:ProxyUseDefaultCredentials' = $true\n"
        "  'Invoke-WebRequest:ProxyCredential' = (New-Object "
        "System.Management.Automation.PSCredential('u',"
        "(ConvertTo-SecureString 'p' -AsPlainText -Force)))\n"
        "  'Start-Process:WindowStyle' = 'Hidden'\n"
        "}\n"
        f". '{block}'\n"
        "Write-Output $script:UnslothProxyHandoffJson\n",
        encoding = "utf-8",
        newline = "",
    )
    # The proxy keys are read straight out of this run's stdout, so an interpreter that died would look like the
    handoff = run_pwsh(
        [shutil.which("pwsh") or "pwsh", "-NoProfile", "-NonInteractive", "-File", str(driver)],
        capture_output = True,
        text = True,
        check = False,
    ).stdout

    # Parsed, not substring-matched: the handoff IS JSON, so an exact value comparison is both stronger and free of the
    # "URL may sit anywhere in the string" reading a bare `in` invites.
    carried = json.loads(handoff.strip().splitlines()[-1])
    assert carried["Invoke-WebRequest:Proxy"] == "http://proxy.corp:8080"
    assert "ProxyUseDefaultCredentials" in handoff
    assert "ProxyCredential" not in handoff, "a credential must not be written to the environment"
    assert "WindowStyle" not in handoff, "only proxy keys are carried"


@requires_pwsh
def test_the_child_restores_the_proxy_and_nothing_else(tmp_path):
    """End to end through real pwsh: with the handoff set the child's table has the proxy back;
    without it, or with a corrupt value, the child is left exactly as -NoProfile made it and the
    launch still succeeds. A prelude that could throw would take setup.ps1 down with it."""
    probe = (
        _proxy_prelude()
        + "Write-Output ('IWR=' + $PSDefaultParameterValues['Invoke-WebRequest:Proxy']); "
        "Write-Output ('OTHER=' + $PSDefaultParameterValues['Start-Process:WindowStyle'])"
    )
    pwsh = shutil.which("pwsh") or "pwsh"

    def _run(value):
        env = {k: v for k, v in os.environ.items() if k != "_UNSLOTH_PS_PROXY_DEFAULTS"}
        if value is not None:
            env["_UNSLOTH_PS_PROXY_DEFAULTS"] = value
        # Each case asserts the child launched successfully and restored exactly the proxy key, so a pwsh that aborted
        # at startup would read as the prelude taking setup.ps1 down.
        return run_pwsh(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", probe],
            capture_output = True,
            text = True,
            check = False,
            env = env,
        )

    restored = _run(
        '{"Invoke-WebRequest:Proxy":"http://proxy.corp:8080",'
        '"Start-Process:WindowStyle":"Hidden"}'
    )
    assert restored.returncode == 0
    assert "IWR=http://proxy.corp:8080" in restored.stdout
    # install.ps1 never publishes a non-proxy key, but the child trusts the table wholesale, so pin that the round trip
    # carries what it was given and the FILTER is the one place that decides.
    assert "OTHER=Hidden" in restored.stdout

    for absent in (None, "{not json", ""):
        result = _run(absent)
        assert result.returncode == 0, f"the prelude must not fail the launch on {absent!r}"
        assert "IWR=\n" in result.stdout or result.stdout.startswith("IWR=\n")
        assert not result.stderr.strip(), f"the prelude leaked an error on {absent!r}"




def test_module_autoloading_is_restored_before_the_handoff_needs_it():
    """Ordering. The handoff calls ConvertTo-Json, which lives in Microsoft.PowerShell.Utility,
    so under a profile's $PSModuleAutoLoadingPreference = 'None' a fresh PowerShell 7 session
    would terminate there -- taking out the exact configuration the handoff exists to support."""
    source = _install_ps1()
    autoload = _locate(source, "$PSModuleAutoLoadingPreference = 'All'", "the autoloading reset")
    convert = _locate(source, "ConvertTo-Json -Compress", "the handoff serialization")
    assert autoload < convert, (
        "ConvertTo-Json runs before module autoloading is restored, so a profile that disables "
        "it kills the install before the proxy is ever carried across"
    )


@requires_pwsh
def test_the_filter_takes_lowercase_keys_and_uri_values(tmp_path):
    """Both are ordinary PowerShell. Parameter names bind case-insensitively, and [uri] is the
    type the Proxy parameter actually takes, so a careful profile writes it that way. Dropping
    either leaves a corporate host with no route out."""
    block = tmp_path / "block.ps1"
    block.write_text(_extract_prologue(), encoding = "utf-8", newline = "")
    driver = tmp_path / "drive.ps1"
    driver.write_text(
        "$PSDefaultParameterValues = @{\n"
        "  'invoke-webrequest:proxy' = [uri]'http://proxy.corp:8080'\n"
        "  'Invoke-RestMethod:PROXYUSEDEFAULTCREDENTIALS' = $true\n"
        "  'Start-Process:WindowStyle' = 'Hidden'\n"
        "}\n"
        f". '{block}'\n"
        "Write-Output $script:UnslothProxyHandoffJson\n",
        encoding = "utf-8",
        newline = "",
    )
    # Same reading here for the casing and [uri] cases:
    handoff = run_pwsh(
        [shutil.which("pwsh") or "pwsh", "-NoProfile", "-NonInteractive", "-File", str(driver)],
        capture_output = True,
        text = True,
        check = False,
    ).stdout

    assert "proxy.corp:8080" in handoff, "a [uri] value was dropped at the process boundary"
    assert "PROXYUSEDEFAULTCREDENTIALS" in handoff, "a lowercase-cmdlet key was filtered out"
    assert "WindowStyle" not in handoff, "only proxy keys are carried"


def test_the_handoff_is_published_only_around_the_setup_child():
    """Under "irm ... | iex" the prologue runs in the CALLER's session, so an environment
    variable set there outlives the install on every path, early returns included. A later
    `unsloth studio update` from that console would then reapply stale proxy JSON."""
    source = _install_ps1()
    prologue = _extract_prologue()
    assert (
        "$env:_UNSLOTH_PS_PROXY_DEFAULTS =" not in prologue
    ), "the prologue publishes the handoff into the session it was invoked from"
    assert "$UnslothProxyHandoffJson" in prologue, "the prologue must hold it instead"
    # Set beside the other child-scoped variables, and restored with them.
    assert "$previousProxyHandoff = $env:_UNSLOTH_PS_PROXY_DEFAULTS" in source
    assert "$env:_UNSLOTH_PS_PROXY_DEFAULTS = $previousProxyHandoff" in source
    gate = _locate(source, "$previousSetupRuntimeGateHandoff =", "the runtime-gate handoff")
    proxy = _locate(source, "$previousProxyHandoff =", "the proxy handoff save")
    call = _locate(
        source,
        "    try {\n        Invoke-ManagedUnslothCli -Python $VenvPython -Arguments $studioArgs",
        "the child invocation",
    )
    assert gate < call and proxy < call, "the handoff must be in place before the child runs"


def test_a_standalone_update_reconstructs_the_proxy_for_itself():
    """install.ps1 publishes the handoff; `unsloth studio update` typed into a console has no
    installer above it, and the profile that holds the proxy ran in a shell whose variables
    never reach this Python process. So it is asked for, before -NoProfile drops it."""
    src = STUDIO_COMMAND.read_text(encoding = "utf-8")
    assert "_probe_profile_proxy_defaults" in src
    start = _locate(src, "powershell_args = [powershell]", "the setup.ps1 launch")
    guard = _locate(src[start:], "_probe_profile_proxy_defaults(", "the probe call")
    noprofile = _locate(src[start:], '"-NoProfile"', "the -NoProfile flag")
    assert guard < noprofile, "the probe has to run while the profile is still reachable"
    # ...and only when the installer did not already hand one over.
    assert "_UNSLOTH_PS_PROXY_DEFAULTS" in src[start : start + guard]


@requires_pwsh
def test_the_probe_reads_a_hostile_profile_without_carrying_anything_else(tmp_path):
    """Executable, against a profile pwsh genuinely loads: strict mode on, module autoloading
    off, a lowercase key and a [uri] value. The probe has to survive all of it and return only
    the proxy entries."""
    src = STUDIO_COMMAND.read_text(encoding = "utf-8")
    # From the markers, not just the script:
    start = _locate(src, "_PROXY_PROBE_BEGIN = ", "the probe framing")
    namespace: dict = {}
    exec(  # noqa: S102 - our own source
        src[start : src.index("\n)\n", start) + 3], namespace
    )

    home = tmp_path / "config"
    (home / "powershell").mkdir(parents = True)
    (home / "powershell" / "Microsoft.PowerShell_profile.ps1").write_text(
        "Set-StrictMode -Version Latest\n"
        "$PSModuleAutoLoadingPreference = 'None'\n"
        "$PSDefaultParameterValues['invoke-webrequest:proxy'] = [uri]'http://proxy.corp:8080'\n"
        "$PSDefaultParameterValues['Start-Process:WindowStyle'] = 'Hidden'\n",
        encoding = "utf-8",
        newline = "",
    )
    # Empty stdout is already handled below as "the planted profile was not loaded" and skips the test, so a crashed
    result = run_pwsh(
        [
            shutil.which("pwsh") or "pwsh",
            "-NonInteractive",
            "-Command",
            namespace["_PS_PROXY_PROBE"],
        ],
        capture_output = True,
        text = True,
        check = False,
        env = {**os.environ, "XDG_CONFIG_HOME": str(home), "HOME": str(tmp_path)},
    )
    from unsloth_cli.commands import studio as studio_cmd

    if "proxy.corp" not in (result.stdout or ""):
        pytest.skip(f"pwsh did not load the planted profile (got {result.stdout!r})")
    payload = studio_cmd._framed_probe_record(result.stdout)
    assert payload, "the probe must emit a framed record"
    assert "WindowStyle" not in payload, "the probe carried a non-proxy default across"
    import json as _json

    assert isinstance(_json.loads(payload), dict), "the probe must emit parseable JSON"


def test_a_profile_that_prints_a_banner_does_not_cost_the_proxy(monkeypatch):
    """The profile has already run by the time the record is printed and is free to say
    anything: a MOTD, a "loading modules" line, a corporate banner. With the record bare, that
    output arrived first, the parse threw, and the answer was dropped -- so the locked-down host
    that needed the proxy handed the -NoProfile child nothing."""
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    noisy = _framed(
        '{"invoke-webrequest:proxy": "http://proxy.corp:8080"}',
        banner = "Loading personal and system profiles took 812ms.\nWelcome to Contoso.\n",
    )
    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result(noisy))

    merged = studio_cmd._probe_profile_proxy_defaults(["pwsh.exe"])
    assert json.loads(merged) == {"invoke-webrequest:proxy": "http://proxy.corp:8080"}


def test_a_profile_that_prints_nothing_useful_is_still_no_answer(monkeypatch):
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result("just a banner\n"))
    assert studio_cmd._probe_profile_proxy_defaults(["pwsh.exe"]) is None


def test_the_caller_edition_is_read_from_the_order_not_the_absence(monkeypatch):
    """A machine can carry BOTH module trees on PSModulePath -- 5.1 launched from a session
    that already had 7's path, or a profile that appends it, the mixed case setup.ps1
    documents. Reading "7 is present, so the caller is 7" then gave the wrong profile
    precedence and let its proxy override the console the command was typed into. Each host
    puts its own module directory first, so the earliest tree names the caller."""
    from unsloth_cli.commands import studio as studio_cmd

    monkeypatch.setattr(studio_cmd.shutil, "which", lambda name: f"C:\\{name}")
    windows = "C:\\Users\\me\\Documents\\WindowsPowerShell\\Modules"
    seven = "C:\\Program Files\\PowerShell\\7\\Modules"

    # Both present, 5.1 first: the caller is Windows PowerShell.
    monkeypatch.setenv("PSModulePath", f"{windows};{seven}")
    assert studio_cmd._profile_probe_hosts()[0] == "powershell.exe"
    # Both present, 7 first: the caller is pwsh.
    monkeypatch.setenv("PSModulePath", f"{seven};{windows}")
    assert studio_cmd._profile_probe_hosts()[0] == "pwsh.exe"
    # Only one tree: unchanged from before.
    monkeypatch.setenv("PSModulePath", windows)
    assert studio_cmd._profile_probe_hosts()[0] == "powershell.exe"
    monkeypatch.setenv("PSModulePath", seven)
    assert studio_cmd._profile_probe_hosts()[0] == "pwsh.exe"
    # Nothing to read: the previous default order.
    monkeypatch.setenv("PSModulePath", "")
    assert studio_cmd._profile_probe_hosts() == ["pwsh.exe", "powershell.exe"]


def test_the_probe_pins_its_own_output_encoding(monkeypatch):
    """Windows PowerShell 5.1 writes REDIRECTED output in the console code page while this
    process decodes UTF-8, so a non-ASCII proxy value came back with replacement characters,
    still parsed as JSON, and handed setup a proxy that does not resolve."""
    from unsloth_cli.commands import studio as studio_cmd

    probe = studio_cmd._PS_PROXY_PROBE
    assert "[Console]::OutputEncoding" in probe
    assert "UTF8Encoding" in probe
    assert probe.index("OutputEncoding") < probe.index("PSDefaultParameterValues")


def test_one_host_owns_a_cmdlet_outright(monkeypatch):
    """Filling a missing companion parameter from the other edition's profile builds a
    configuration neither host has: the earlier host's Proxy with the later host's
    ProxyUseDefaultCredentials, which offers the user's Windows credentials to a proxy whose
    own profile never asked for that."""
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    answers = [
        _framed('{"Invoke-WebRequest:Proxy": "http://first.corp:8080"}'),
        _framed(
            '{"Invoke-WebRequest:ProxyUseDefaultCredentials": true, '
            '"Start-BitsTransfer:Proxy": "http://second.corp:8080"}'
        ),
    ]
    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result(answers.pop(0)))

    merged = json.loads(studio_cmd._probe_profile_proxy_defaults(["pwsh.exe", "powershell.exe"]))

    assert merged == {
        "Invoke-WebRequest:Proxy": "http://first.corp:8080",
        # ...while a cmdlet the first host never configured still comes across.
        "Start-BitsTransfer:Proxy": "http://second.corp:8080",
    }


def test_the_probe_signature_evaluates_on_the_oldest_supported_python():
    """No `from __future__ import annotations` here, so an unquoted `str | list[str]` is
    evaluated at def time -- a TypeError on 3.9 that takes the whole CLI import with it."""
    import ast

    tree = ast.parse(STUDIO_COMMAND.read_text(encoding = "utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__" for node in ast.walk(tree)
    ), "this test's premise changed: the module now postpones annotations"
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for arg in [*node.args.args, *node.args.kwonlyargs]:
            annotation = arg.annotation
            if annotation is None:
                continue
            assert not isinstance(
                annotation, ast.BinOp
            ), f"{node.name}({arg.arg}) uses an unquoted PEP 604 union, which raises on 3.9"


def test_the_probe_asks_both_powershell_editions(monkeypatch):
    """pwsh and powershell.exe keep SEPARATE profiles.

    `unsloth studio update` is typed into whichever host the user has open, and probing only
    powershell.exe missed a proxy living in the PowerShell 7 profile -- the likelier place on
    a machine that has pwsh at all. The -NoProfile child then had no proxy and setup.ps1's
    downloads failed.
    """
    from unsloth_cli.commands import studio as studio_cmd

    asked: list[str] = []

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    answers = {
        "pwsh.exe": _framed('{"invoke-webrequest:proxy": "http://seven.corp:8080"}'),
        "powershell.exe": _framed('{"invoke-restmethod:proxy": "http://five.corp:8080"}'),
    }

    def _run(argv, **kwargs):
        asked.append(argv[0])
        return _Result(answers[argv[0]])

    monkeypatch.setattr(studio_cmd.subprocess, "run", _run)
    merged = studio_cmd._probe_profile_proxy_defaults(["pwsh.exe", "powershell.exe"])

    assert asked == ["pwsh.exe", "powershell.exe"]
    assert json.loads(merged) == {
        "invoke-webrequest:proxy": "http://seven.corp:8080",
        "invoke-restmethod:proxy": "http://five.corp:8080",
    }


def test_the_callers_edition_wins_where_the_two_profiles_disagree(monkeypatch):
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    answers = {
        "pwsh.exe": _framed('{"invoke-webrequest:proxy": "http://seven.corp:8080"}'),
        "powershell.exe": _framed('{"invoke-webrequest:proxy": "http://five.corp:8080"}'),
    }
    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result(answers[argv[0]]))
    merged = studio_cmd._probe_profile_proxy_defaults(["powershell.exe", "pwsh.exe"])
    assert json.loads(merged) == {"invoke-webrequest:proxy": "http://five.corp:8080"}


def test_two_spellings_of_one_key_are_one_key(monkeypatch):
    """$PSDefaultParameterValues is case-insensitive; a Python dict is not.

    With both spellings carried across, the prelude replayed them in order and the
    lower-priority host's value landed last -- the exact reverse of earlier-host-wins, and on
    a stricter host the case-colliding JSON is rejected outright."""
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    answers = {
        "pwsh.exe": _framed('{"Invoke-WebRequest:Proxy": "http://seven.corp:8080"}'),
        "powershell.exe": _framed('{"invoke-webrequest:proxy": "http://five.corp:8080"}'),
    }
    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result(answers[argv[0]]))
    merged = json.loads(studio_cmd._probe_profile_proxy_defaults(["pwsh.exe", "powershell.exe"]))

    assert merged == {"Invoke-WebRequest:Proxy": "http://seven.corp:8080"}


def test_one_profile_that_prints_both_spellings_is_folded_too(monkeypatch):
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    payload = (
        '{"Invoke-WebRequest:Proxy": "http://first.corp:8080",'
        ' "invoke-webrequest:PROXY": "http://second.corp:8080",'
        ' "Invoke-RestMethod:Proxy": "http://rest.corp:8080"}'
    )
    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result(_framed(payload)))
    merged = json.loads(studio_cmd._probe_profile_proxy_defaults(["pwsh.exe"]))

    # The first host owns Invoke-WebRequest whole, so the second host's credential flag for that same cmdlet is
    assert merged == {
        "Invoke-WebRequest:Proxy": "http://first.corp:8080",
        "Invoke-RestMethod:Proxy": "http://rest.corp:8080",
    }


def test_a_wildcard_key_claims_the_whole_cmdlet_family(monkeypatch):
    """The command half of a $PSDefaultParameterValues key may be a wildcard, and PowerShell
    applies such an entry to every cmdlet it matches. Comparing the strings literally let
    Invoke-Web*:Proxy from one host merge with Invoke-WebRequest:ProxyUseDefaultCredentials from
    the other -- one invocation configured from two profiles, offering the user's Windows
    credentials to a proxy whose own profile never asked for that."""
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    answers = {
        "pwsh.exe": _framed('{"Invoke-Web*:Proxy": "http://seven.corp:8080"}'),
        "powershell.exe": _framed(
            '{"Invoke-WebRequest:ProxyUseDefaultCredentials": true,'
            ' "Invoke-RestMethod:Proxy": "http://five.corp:8080"}'
        ),
    }
    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result(answers[argv[0]]))
    merged = json.loads(studio_cmd._probe_profile_proxy_defaults(["pwsh.exe", "powershell.exe"]))

    assert merged == {
        "Invoke-Web*:Proxy": "http://seven.corp:8080",
        "Invoke-RestMethod:Proxy": "http://five.corp:8080",
    }


def test_two_wildcards_that_share_a_cmdlet_are_one_family(monkeypatch):
    """Matching either pattern against the other as a STRING does not establish whether their
    match sets overlap: Invoke-Web* and *-WebRequest both apply to Invoke-WebRequest and neither
    matches the other. Overlap between two patterns is assumed, so the lower-priority profile
    cannot slip ProxyUseDefaultCredentials in beside the other's proxy."""
    # never reaches the child.
    # Same collision inside a single host's answer:
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    answers = {
        "pwsh.exe": _framed('{"Invoke-Web*:Proxy": "http://seven.corp:8080"}'),
        "powershell.exe": _framed(
            '{"*-WebRequest:ProxyUseDefaultCredentials": true,'
            ' "Invoke-RestMethod:Proxy": "http://five.corp:8080"}'
        ),
    }
    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result(answers[argv[0]]))
    merged = json.loads(studio_cmd._probe_profile_proxy_defaults(["pwsh.exe", "powershell.exe"]))

    assert merged == {
        "Invoke-Web*:Proxy": "http://seven.corp:8080",
        # An unrelated family from the second host is still merged.
        "Invoke-RestMethod:Proxy": "http://five.corp:8080",
    }


def test_the_probe_re_pins_utf8_after_the_profiles_have_run():
    """A profile setting [Console]::OutputEncoding is an ordinary customization and it
    overrides the pin at the top of the probe. The parent decodes this stream as UTF-8, so a
    profile that leaves the console on UTF-16 or a legacy code page corrupts the framed record
    and a non-ASCII proxy URI goes with it."""
    from unsloth_cli.commands.studio import _PS_PROXY_PROBE as probe

    pin = "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false"
    assert probe.count(pin) == 2, "pinned before the profiles and again after them"
    assert probe.index(pin) < probe.index("$__unslothProfile") < probe.rindex(pin)
    assert probe.rindex(pin) < probe.index("ConvertTo-Json")


def test_the_record_is_emitted_through_the_builtin_cmdlets():
    """An alias or function named ConvertTo-Json or Write-Output in the profile shadows the bare
    name, and clearing $PSDefaultParameterValues does not cover a command override. A wrapper
    that reshapes the output produces a frame the reader cannot parse, which costs a standalone
    update its only proxy."""
    from unsloth_cli.commands.studio import _PS_PROXY_PROBE as probe

    assert probe.count("Microsoft.PowerShell.Utility\\Write-Output") == 2
    assert "Microsoft.PowerShell.Utility\\ConvertTo-Json -Compress" in probe
    # No bare invocation left to be shadowed.
    assert "; Write-Output " not in probe
    assert "| ConvertTo-Json" not in probe


def test_a_vscode_terminal_still_reads_its_own_host_profile():
    """TERM_PROGRAM=vscode is set by EVERY VS Code integrated terminal, not only the PowerShell
    extension's host. Substituting Microsoft.VSCode_profile.ps1 for the current-host profile
    therefore missed the proxy a plain pwsh terminal in VS Code actually has, while applying an
    unrelated one. Both are read, current-host last."""
    from unsloth_cli.commands.studio import _PS_PROXY_PROBE as probe

    assert "$__unslothProfiles += $PROFILE.CurrentUserCurrentHost; " in probe
    assert "$__unslothProfiles += $PROFILE.AllUsersCurrentHost; " in probe
    # The named host profile is an addition, never an else-branch replacement.
    assert "} else { " not in probe.split("$out = @{}")[0]
    assert probe.index("Split-Path -Parent $PROFILE.CurrentUserCurrentHost") < probe.index(
        "$__unslothProfiles += $PROFILE.CurrentUserCurrentHost; "
    )


def test_the_probe_host_order_follows_the_console_the_user_typed_into(monkeypatch):
    from unsloth_cli.commands import studio as studio_cmd

    monkeypatch.setattr(studio_cmd.shutil, "which", lambda name: f"C:/{name}")

    monkeypatch.setenv("PSModulePath", r"C:\Program Files\PowerShell\7\Modules")
    assert studio_cmd._profile_probe_hosts()[0] == "pwsh.exe"

    monkeypatch.setenv("PSModulePath", r"C:\Windows\System32\WindowsPowerShell\v1.0\Modules")
    assert studio_cmd._profile_probe_hosts()[0] == "powershell.exe"

    # A host that is not installed is not asked.
    monkeypatch.setattr(
        studio_cmd.shutil, "which", lambda name: None if name == "pwsh.exe" else "x"
    )
    assert studio_cmd._profile_probe_hosts() == ["powershell.exe"]


def test_the_uv_alias_is_resolved_before_the_path_candidates():
    """PowerShell resolves an alias ahead of PATH, so `uv` in a profile that aliases it means
    that binary -- and checking PATH first made the alias branch unreachable on any machine
    that also has some uv on PATH."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    start = _locate(src, "function Get-UvExecutableCandidates {", "the uv resolver")
    body = src[start : start + 2500]
    alias = _locate(body, "Get-Command uv -CommandType Alias", "the alias lookup")
    apps = _locate(body, "Get-Command uv -CommandType Application", "the PATH lookup")
    assert alias < apps, "the PATH candidates are consulted before the alias"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_a_stale_alias_does_not_block_a_current_uv_on_path(tmp_path):
    """Executed: the version gate walks EVERY candidate.

    An alias pointing at a real but stale uv used to be the only binary ever probed, so a
    current uv already on PATH -- or one winget had just installed -- could not rescue the
    run, and the install ended at "uv could not be installed" on a machine that had one.
    """
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    gate = src[
        _locate(src, "    function Test-UvVersionOk {", "the version gate") : _locate(
            src, "    # Fallback for hosts without winget", "the end of the gate"
        )
    ]
    script = (
        "$UvMinVersion = '0.9.0'\n"
        f"{gate}\n"
        # Two candidates: a stale alias target first, a current one second.
        "function Get-UvExecutableCandidates { @('stale', 'current') }\n"
        "function Test-UvCandidateVersionShim { }\n"
        "Write-Output ([string](Test-UvVersionOk))\n"
        "Write-Output $script:UvExe\n"
    )
    # The candidates are invoked as executables, so stand in two shims that answer --version.
    stale = tmp_path / "stale.ps1"
    stale.write_text("Write-Output 'uv 0.4.30'", encoding = "utf-8")
    current = tmp_path / "current.ps1"
    current.write_text("Write-Output 'uv 0.12.1'", encoding = "utf-8")
    script = script.replace(
        "@('stale', 'current')",
        f"@('{stale.as_posix()}', '{current.as_posix()}')",
    )
    # The uv gate is judged by the first and last lines of stdout, and an interpreter crash leaves none, which the
    result = run_pwsh(
        [shutil.which("pwsh"), "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        check = False,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert (
        lines and lines[0] == "True"
    ), f"the gate stopped at the stale alias: {result.stdout!r} {result.stderr!r}"
    assert lines[-1].endswith("current.ps1"), "the passing candidate must be the pinned one"


def test_the_parity_workflow_runs_when_the_studio_command_changes():
    """The suite asserts unsloth_cli/commands/studio.py's behaviour directly, so a PR touching
    only that module has to run it. No other workflow invokes this file."""
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cross-platform-parity-ci.yml"
    ).read_text(encoding = "utf-8")
    assert (
        workflow.count("unsloth_cli/commands/studio.py") == 2
    ), "both the pull_request and push path filters need the module"


def test_the_parity_job_installs_what_this_suite_imports():
    """Three tests here import unsloth_cli.commands.studio, which pulls typer -> pyyaml ->
    pydantic -> click. The job installed pip and pytest only, so on a clean setup-python it
    died with ModuleNotFoundError before a single test ran, on both matrix legs."""
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cross-platform-parity-ci.yml"
    ).read_text(encoding = "utf-8")
    install = [line for line in workflow.splitlines() if "pip install" in line and "pytest" in line]
    assert install, "the parity job's install step was not found"
    for package in ("typer", "pyyaml", "pydantic", "click", "rich"):
        assert all(
            package in line for line in install
        ), f"the parity job does not install {package}, which this suite's imports need"


def test_the_probe_output_is_decoded_lossily():
    """The profile ran before the record was printed and may have said anything in any
    encoding. `text=True` alone decodes with the locale codec and STRICT errors, so a UTF-8
    banner on an ANSI console raised UnicodeDecodeError -- neither OSError nor
    SubprocessError, so it escaped the handler and took the update down before the
    -NoProfile child ever ran."""
    import ast

    tree = ast.parse(STUDIO_COMMAND.read_text(encoding = "utf-8"))
    probe = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_probe_profile_proxy_defaults"
    )
    runs = [
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
    ]
    assert runs, "the probe no longer shells out"
    for call in runs:
        keywords = {kw.arg for kw in call.keywords}
        assert "encoding" in keywords and "errors" in keywords, (
            "a strict locale decode of the profile's own output can raise before the framing "
            "is ever consulted"
        )


def test_a_non_ascii_banner_does_not_cost_the_proxy(monkeypatch):
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    banner = "Bienvenue \ufffd\ufffd\ufffd chez Contoso\n"
    noisy = _framed('{"invoke-webrequest:proxy": "http://proxy.corp:8080"}', banner = banner)
    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result(noisy))

    merged = studio_cmd._probe_profile_proxy_defaults(["pwsh.exe"])
    assert json.loads(merged) == {"invoke-webrequest:proxy": "http://proxy.corp:8080"}


def test_the_setup_child_does_not_hand_the_proxy_secret_to_its_descendants():
    """A profile proxy is routinely an authenticated URI (http://user:secret@proxy). The prelude
    copies it into $PSDefaultParameterValues, which is NOT inherited -- but the environment
    variable it read from is, so every native process setup.ps1 starts, and everything they
    start in turn, saw the plaintext credential. It is removed the moment it has been read."""
    from unsloth_cli.commands.studio import _PS_PROXY_DEFAULTS_PRELUDE

    prelude = _PS_PROXY_DEFAULTS_PRELUDE
    read_at = prelude.find("$env:_UNSLOTH_PS_PROXY_DEFAULTS")
    removed_at = prelude.find("Remove-Item Env:_UNSLOTH_PS_PROXY_DEFAULTS")
    assert read_at >= 0, "the prelude must still read the handoff"
    assert removed_at > read_at, "and must clear it straight after reading it"
    # Before anything that could start a child process -- i.e.
    assert removed_at < prelude.find("$PSDefaultParameterValues[$_.Name]")


def test_the_probe_adds_the_callers_host_profile_beside_the_current_host_one(monkeypatch):
    """pwsh.exe and powershell.exe run the CONSOLEHOST profile, so a caller in the VS Code
    Integrated Console -- whose defaults live in Microsoft.VSCode_profile.ps1 -- got no proxy
    and the -NoProfile child could not download.

    Added beside the current-host profile, never in place of it: TERM_PROGRAM=vscode is set by
    every VS Code integrated terminal, so substitution robbed a plain pwsh terminal there of the
    only profile it has. Named hosts only, since a directory-wide sweep of
    Microsoft.*_profile.ps1 ran profiles for hosts nobody was using."""
    # The decode is the parent's job, so this drives the extractor with the mangled text the lossy decode produces: the
    # record is ASCII and has to survive whatever precedes it.
    from unsloth_cli.commands import studio as studio_cmd

    probe = studio_cmd._PS_PROXY_PROBE
    assert "$env:_UNSLOTH_PS_HOST_PROFILE" in probe
    assert "Microsoft.*_profile.ps1" not in probe, "no directory-wide sourcing"
    # The two the caller's session would have loaded, named explicitly...
    assert "$PROFILE.CurrentUserAllHosts" in probe
    assert "CurrentUserCurrentHost" in probe
    assert "Split-Path -Parent $PROFILE.CurrentUserCurrentHost" in probe
    # ...and before the table is read, or it would snapshot the wrong defaults.
    assert probe.find("_UNSLOTH_PS_HOST_PROFILE") < probe.find("$out = @{}")
    # Unconditional and last, so the profile the probe's own host would have loaded is still there and still gets the
    for scope in ("AllUsersCurrentHost", "CurrentUserCurrentHost"):
        named = probe.find(f"Split-Path -Parent $PROFILE.{scope}")
        plain = probe.find(f"$__unslothProfiles += $PROFILE.{scope};")
        assert named >= 0, f"the caller's own host profile is named beside {scope}"
        assert plain > named, f"{scope} is kept and runs after it"

    # The caller's host is named from the environment it announces itself in.
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    assert studio_cmd._profile_probe_env()["_UNSLOTH_PS_HOST_PROFILE"] == (
        "Microsoft.VSCode_profile.ps1"
    )
    # An unidentifiable host gets no extra profile rather than someone else's.
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    assert "_UNSLOTH_PS_HOST_PROFILE" not in studio_cmd._profile_probe_env()
    monkeypatch.delenv("TERM_PROGRAM", raising = False)
    assert "_UNSLOTH_PS_HOST_PROFILE" not in studio_cmd._profile_probe_env()


def test_the_probe_loads_the_all_users_profiles_in_startup_order():
    """A machine-managed proxy lives in an all-users profile on a domain-joined box and the
    user's own profile never mentions it, so sourcing only the current-user pair reported no
    proxy on exactly the locked-down host that has one. Order is part of the fix: the user's
    profile is entitled to override the machine's, which it only does if it runs last."""
    from unsloth_cli.commands.studio import _PS_PROXY_PROBE as probe

    assert "$PROFILE.AllUsersAllHosts" in probe
    assert "Split-Path -Parent $PROFILE.AllUsersCurrentHost" in probe
    assert (
        probe.find("$PROFILE.AllUsersAllHosts")
        < probe.find("$PROFILE.AllUsersCurrentHost")
        < probe.find("$PROFILE.CurrentUserAllHosts")
        < probe.find("$PROFILE.CurrentUserCurrentHost")
    )
    # Still the caller's own host, never a directory sweep, for the all-users pair too.
    assert "Microsoft.*_profile.ps1" not in probe
    assert probe.count("Join-Path (Split-Path -Parent") == 2


def test_the_probe_clears_profile_defaults_before_it_serializes():
    """The profile's $PSDefaultParameterValues aims at every cmdlet in the probe, including the
    two that emit the record. ConvertTo-Json:AsArray = $true is a legitimate setting and it
    turns the payload into a JSON array, which the reader rejects for not being a dictionary --
    dropping the caller's proxy on the host that needed it. $out already holds copies."""
    from unsloth_cli.commands.studio import _PS_PROXY_PROBE as probe

    assert "$PSDefaultParameterValues = @{}" in probe
    assert (
        probe.find("$out = @{}")
        < probe.find("$PSDefaultParameterValues = @{}")
        < probe.find("ConvertTo-Json")
    )


def test_a_script_block_proxy_default_is_evaluated_not_dropped():
    """{ [uri]$env:CORP_PROXY } is PowerShell's supported form for a dynamic default and
    Invoke-WebRequest evaluates it per call, so the caller downloads fine while the handoff
    silently omitted it. Both serializers evaluate the block and carry the RESULT -- executable
    code must not cross into the child."""
    from unsloth_cli.commands.studio import _PS_PROXY_PROBE

    assert "[scriptblock]" in _PS_PROXY_PROBE
    assert "& $v" in _PS_PROXY_PROBE, "the block has to be invoked, not serialized"
    assert "$out[$k] = $r.AbsoluteUri" in _PS_PROXY_PROBE

    installer = INSTALL_PS1.read_text(encoding = "utf-8")
    assert "$_UnslothDefaultValue -is [scriptblock]" in installer
    assert "& $_UnslothDefaultValue" in installer
    assert "$_UnslothDefaultResolved.AbsoluteUri" in installer


def test_an_installer_launch_with_no_proxy_still_skips_the_probe(monkeypatch):
    """The ABSENCE of the handoff is how a standalone update is recognised. install.ps1 used to
    remove the variable when it had no proxy, so an installer launch -- including one started
    with -NoProfile or by the desktop app -- went off and reloaded the very profiles it had
    deliberately discarded, reapplying a stale proxy during setup."""
    installer = INSTALL_PS1.read_text(encoding = "utf-8")
    handoff = installer[
        installer.index("$previousProxyHandoff = $env:_UNSLOTH_PS_PROXY_DEFAULTS") :
    ]
    handoff = handoff[: handoff.index("try {")]
    # After the table has been read, and before anything is written.
    assert (
        "Remove-Item Env:_UNSLOTH_PS_PROXY_DEFAULTS" not in handoff
    ), "the installer must publish an explicit empty handoff, not remove the variable"
    assert "'{}'" in handoff

    # And the CLI keys on presence, so "{}" means "the installer looked, there is none".
    source = STUDIO_COMMAND.read_text(encoding = "utf-8")
    assert 'os.environ.get("_UNSLOTH_PS_PROXY_DEFAULTS") is None' in source


def test_the_profile_probe_shares_one_timeout_across_hosts(monkeypatch):
    """Both editions installed and both profiles hung meant two full timeouts back to back, so
    every standalone setup stalled for twice the cost the helper documents."""
    from unsloth_cli.commands import studio as studio_cmd

    budget = 0.4
    monkeypatch.setattr(studio_cmd, "_PROFILE_PROBE_TIMEOUT_SECONDS", budget)
    asked: list[float] = []

    def _hang(argv, **kwargs):
        # A profile that really does hang burns the timeout it was given, which is what makes a per-host budget cost
        asked.append(kwargs["timeout"])
        time.sleep(kwargs["timeout"])
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(studio_cmd.subprocess, "run", _hang)
    started = time.monotonic()
    assert studio_cmd._probe_profile_proxy_defaults(["pwsh.exe", "powershell.exe"]) is None
    elapsed = time.monotonic() - started
    assert asked, "the probe must have been attempted"
    # One budget for the whole probe, however many hosts it tries.
    assert sum(asked) <= budget + 0.05, asked
    assert elapsed < budget * 1.8, elapsed


def test_the_probe_child_runs_with_no_profile(monkeypatch):
    """Without -NoProfile the probe host loads its OWN ConsoleHost profile first: an unrelated
    profile that prints, rewrites $PSDefaultParameterValues or calls exit got in the way, and it
    is still not the profile a VS Code caller keeps its defaults in. With -NoProfile nothing has
    run, and the profiles the caller's session would have loaded ($PROFILE.CurrentUserAllHosts
    plus its host profile) are dot-sourced by name -- $PROFILE is fully populated under
    -NoProfile because the paths are computed, not loaded."""
    from unsloth_cli.commands import studio as studio_cmd

    seen: list[list[str]] = []

    def _capture(argv, **kwargs):
        seen.append(list(argv))
        raise OSError("no powershell here")

    monkeypatch.setattr(studio_cmd.subprocess, "run", _capture)
    studio_cmd._probe_profile_proxy_defaults(["pwsh.exe"])

    assert seen, "the probe must have been attempted"
    assert "-NoProfile" in seen[0]
    # Ahead of -Command, like every other PowerShell child in the tree spells it.
    assert seen[0].index("-NoProfile") < seen[0].index("-Command")


def test_the_proxy_handoff_does_not_outlive_the_installer():
    """Under the documented `irm ... | iex` path $script: IS the caller's session scope, so a
    serialized authenticated proxy stayed readable in that console after the installer
    returned. Cleanup near the setup child covers one exit out of dozens -- -ShortcutsOnly, an
    argument error, lock contention, a failed dependency install all return earlier -- so the
    value is held in a FUNCTION-local, which dies with the frame on every path."""
    installer = INSTALL_PS1.read_text(encoding = "utf-8")

    assert "$script:UnslothProxyHandoffJson" not in installer
    assert "\n    $UnslothProxyHandoffJson =\n" in installer
    # Still dropped explicitly once the child it exists for has run.
    cleared = "$UnslothProxyHandoffJson = $null"
    assert cleared in installer
    child = installer.index("Invoke-ManagedUnslothCli -Python $VenvPython -Arguments $studioArgs")
    assert child < installer.index(cleared, child)


def test_the_installer_serializes_the_handoff_through_the_builtin():
    """A profile alias or function named ConvertTo-Json would otherwise reshape this record or
    throw out of the prologue, and setup then gets an empty proxy configuration on a host whose
    only egress is that same profile proxy."""
    installer = INSTALL_PS1.read_text(encoding = "utf-8")

    assert "Microsoft.PowerShell.Utility\\ConvertTo-Json -Compress" in installer
    prologue = _extract_prologue()
    assert "| ConvertTo-Json" not in prologue


def test_disjoint_wildcard_families_from_two_hosts_both_survive(monkeypatch):
    """Assuming every wildcard pair overlaps threw away entries that provably cannot: a
    higher-priority Start-Bits* dropped the other host's Invoke-Web*, so setup was left without
    the proxy Invoke-WebRequest needs on a locked-down box."""
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    answers = {
        "pwsh.exe": _framed('{"Start-Bits*:Proxy": "http://seven.corp:8080"}'),
        "powershell.exe": _framed('{"Invoke-Web*:Proxy": "http://five.corp:8080"}'),
    }
    monkeypatch.setattr(studio_cmd.subprocess, "run", lambda argv, **kw: _Result(answers[argv[0]]))
    merged = json.loads(studio_cmd._probe_profile_proxy_defaults(["pwsh.exe", "powershell.exe"]))

    assert merged == {
        "Start-Bits*:Proxy": "http://seven.corp:8080",
        "Invoke-Web*:Proxy": "http://five.corp:8080",
    }


@pytest.mark.parametrize(
    ("left", "right", "overlaps"),
    [
        ("invoke-web*", "*-webrequest", True),
        ("invoke-web*", "invoke-webrequest", True),
        ("*", "invoke-webrequest", True),
        ("invoke-?ebrequest", "invoke-webrequest", True),
        ("invoke-web*", "invoke-restmethod", False),
        ("start-bits*", "invoke-web*", False),
        ("*-webrequest", "*-restmethod", False),
        # A character class is assumed to overlap rather than decided:
        ("invoke-[wr]*", "start-bits*", True),
    ],
)
def test_two_command_patterns_overlap_exactly_when_a_name_matches_both(left, right, overlaps):
    from unsloth_cli.commands import studio as studio_cmd
    assert studio_cmd._patterns_can_overlap(left, right) is overlaps
    assert studio_cmd._patterns_can_overlap(right, left) is overlaps


def _windows_probe_env(monkeypatch, host, module_path):
    from unsloth_cli.commands import studio as studio_cmd

    monkeypatch.setattr(studio_cmd.platform, "system", lambda: "Windows")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("PSModulePath", module_path)
    return studio_cmd._profile_probe_env(host)


# os.environ upper-cases keys on Windows, so a probe env built there carries PSMODULEPATH.
def _module_path(env):
    return env[next(k for k in env if k.upper() == "PSMODULEPATH")]


_PS7_MODULE_PATH = (
    r"C:\Users\me\Documents\PowerShell\Modules;"
    r"C:\Program Files\PowerShell\7\Modules;"
    r"C:\Windows\System32\WindowsPowerShell\v1.0\Modules"
)
_WINDOWS_PS_MODULES = r"C:\Windows\System32\WindowsPowerShell\v1.0\Modules"


def test_the_windows_powershell_probe_is_given_its_own_modules_first(monkeypatch):
    """PowerShell 7 strips its module paths only when IT launches powershell.exe. Through this
    Python process the child kept them in front and loaded PS7 copies of 5.1's own modules, so a
    profile importing one threw while it was dot-sourced and its proxy never reached setup."""
    env = _windows_probe_env(monkeypatch, "powershell.exe", _PS7_MODULE_PATH)

    entries = _module_path(env).split(";")
    assert entries[0] == _WINDOWS_PS_MODULES
    # Reordered, not pruned: everything the caller had is still reachable, just second.
    assert entries[1:] == _PS7_MODULE_PATH.split(";")[:2]
    assert entries.count(_WINDOWS_PS_MODULES) == 1


def test_a_module_path_already_led_by_windows_powershell_is_left_alone(monkeypatch):
    already = _WINDOWS_PS_MODULES + r";C:\Program Files\PowerShell\7\Modules"
    env = _windows_probe_env(monkeypatch, "powershell.exe", already)

    assert _module_path(env) == already


def test_the_pwsh_probe_keeps_the_inherited_module_path(monkeypatch):
    # PowerShell 7 prefixes its own paths at startup, so reordering here would only demote them.
    env = _windows_probe_env(monkeypatch, "pwsh.exe", _PS7_MODULE_PATH)

    assert _module_path(env) == _PS7_MODULE_PATH


def test_the_probe_reads_the_module_path_windows_actually_exports(monkeypatch):
    """dict(os.environ) on Windows is keyed PSMODULEPATH and a plain dict is case-sensitive, so
    reading "PSModulePath" dropped the caller's entries and added a second, case-differing key."""
    from unsloth_cli.commands import studio as studio_cmd

    monkeypatch.setattr(studio_cmd.platform, "system", lambda: "Windows")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.delenv("PSModulePath", raising = False)
    monkeypatch.setenv("PSMODULEPATH", _PS7_MODULE_PATH)

    env = studio_cmd._profile_probe_env("powershell.exe")

    assert [k for k in env if k.upper() == "PSMODULEPATH"] == ["PSMODULEPATH"]
    assert _module_path(env).split(";") == [_WINDOWS_PS_MODULES] + _PS7_MODULE_PATH.split(";")[:2]


def test_the_module_path_is_untouched_off_windows(monkeypatch):
    from unsloth_cli.commands import studio as studio_cmd

    monkeypatch.setattr(studio_cmd.platform, "system", lambda: "Linux")
    monkeypatch.setenv("PSModulePath", _PS7_MODULE_PATH)

    assert _module_path(studio_cmd._profile_probe_env("powershell.exe")) == _PS7_MODULE_PATH


def test_each_probed_host_gets_its_own_module_path(monkeypatch):
    # The wiring: the repair is per host, so the env has to be built from the host being run.
    from unsloth_cli.commands import studio as studio_cmd

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    seen: dict = {}

    def _run(argv, **kwargs):
        seen[argv[0]] = _module_path(kwargs["env"])
        return _Result(_framed('{"Invoke-WebRequest:Proxy": "http://corp:8080"}'))

    monkeypatch.setattr(studio_cmd.platform, "system", lambda: "Windows")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("PSModulePath", _PS7_MODULE_PATH)
    monkeypatch.setattr(studio_cmd.subprocess, "run", _run)
    studio_cmd._probe_profile_proxy_defaults(["pwsh.exe", "powershell.exe"])

    assert seen["pwsh.exe"] == _PS7_MODULE_PATH
    assert seen["powershell.exe"].split(";")[0] == _WINDOWS_PS_MODULES

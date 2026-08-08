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

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
STUDIO_COMMAND = REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"

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


# ── source-level: the couplings stay cut ──


def test_prologue_neutralizes_profile_state():
    block = _extract_prologue()
    assert "Set-StrictMode -Off" in block, (
        "this script's $env:X -in @(...) and unassigned-$script: idioms are errors under Latest"
    )
    assert "$PSDefaultParameterValues = $_UnslothKeptDefaults" in block, (
        "a profile entry like 'Start-Process:WindowStyle' otherwise rebinds cmdlets here"
    )
    assert "$PSNativeCommandUseErrorActionPreference = $false" in block, (
        "with it on, a failing native command throws past Exit-InstallFailure and skips rollback"
    )


def test_proxy_defaults_are_carried_across_rather_than_dropped():
    """Clearing the table wholesale would break the corporate hosts where a profile proxy
    entry is the only route to python.org -- and those users are the ones who never run
    -NoProfile, so they would be trading one broken install for another."""
    block = _extract_prologue()
    assert ":Proxy(Credential|UseDefaultCredentials)?$" in block, (
        "the filter must keep proxy-shaped keys, which can only ever enable a download"
    )


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
        ("& $UnslothExe @studioArgs", "the setup handoff"),
    ):
        idx = _locate(code, marker, what)
        assert strict_idx < idx, f"{what} must be reached with strict mode pinned"
        assert defaults_idx < idx, f"{what} must be reached with the profile's defaults filtered"
        assert native_idx < idx, f"{what} must be reached with native error handling pinned"


def test_script_scoped_uv_state_is_reset_per_invocation():
    """Same hazard as $script:IsIntelXpu: under irm | iex, $script: is the caller's session."""
    src = _install_ps1()
    # Anchored on the newline plus exactly four spaces. The real assignments sit deeper in the
    # function and would otherwise satisfy this by accident, which is how a dropped reset for
    # $script:UvInstallDestDir once slipped past.
    for reset, first_read in (
        ("\n    $script:UvExe = 'uv'\n", "& $script:UvExe"),
        ("\n    $script:UvInstallDestDir = $null\n", "foreach ($d in @($script:UvInstallDestDir"),
    ):
        init_idx = _locate(src, reset, "a per-invocation reset")
        assert init_idx < _locate(src, first_read, "the first read"), (
            f"{reset.strip()} must come before the first read"
        )
    assert "$script:UvExe = 'uv'" in src, (
        "the reset value must stay the bare token, so an unforeseen path behaves as it does today"
    )


def test_no_bare_uv_token_survives_at_a_call_site():
    """A profile alias or function named uv outranks PATH, so every invocation goes through
    the resolved executable."""
    src = _install_ps1()
    assert "{ uv " not in src, "install scriptblocks must invoke $script:UvExe, not the bare token"
    assert "& uv " not in src, "the version probe must invoke the resolved executable"
    assert src.count("{ & $script:UvExe ") >= 27, (
        "every uv scriptblock must route through the resolved path"
    )


def test_uv_is_resolved_as_an_application():
    body = _extract_function("Resolve-UvExecutable")
    # The whole invocation, not the flags separately: the function's own comment mentions both,
    # so a substring test for either passes on a body that no longer uses them.
    assert "Get-Command uv -CommandType Application -All -ErrorAction SilentlyContinue" in body, (
        "Application-only lookup is what skips an alias or function, and ordering across "
        "several matches is only documented for -All"
    )
    assert "return 'uv'" in body, (
        "with nothing on PATH the bare token must come back, so resolution stays exactly today's"
    )
    probe = _extract_function("Test-UvVersionOk")
    assert "Resolve-UvExecutable" in probe and "$script:UvExe = $exe" in probe, (
        "the probe must pin the executable that answered, so later PATH edits cannot swap it"
    )


def test_setup_ps1_handoff_never_inherits_the_profile():
    """install.ps1 ends by running `unsloth studio setup`, which re-enters PowerShell. That
    launch used to add -NoProfile only when stdout was not a tty, so the console install this
    whole file is about ran setup.ps1 with the user's profile loaded and its bare `uv` calls
    exposed to the same alias."""
    src = STUDIO_COMMAND.read_text(encoding = "utf-8")
    start = _locate(src, 'powershell_args = ["powershell.exe"]', "the setup.ps1 launch")
    branch = _locate(src[start:], "_should_hide_windows_subprocesses()", "the hidden-window branch")
    assert '"-NoProfile"' in src[start : start + branch], (
        "-NoProfile must be added unconditionally, before the hidden-window branch"
    )


# ── executable: run the extracted code under a hostile profile ──


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
    # HOME on its own does not move $PROFILE on Unix. PowerShell reads $XDG_CONFIG_HOME first and
    # only falls back to $HOME/.config when it is unset, and GitHub's ubuntu image writes
    # XDG_CONFIG_HOME into /etc/environment, so an inherited value went on naming the real account
    # while the fixture planted its profile under tmp_path -- the whole file's isolation leaked on
    # a hosted runner and nowhere else. Pointed at the same directory HOME already implies, so the
    # two rules agree whichever one the host applies. Windows ignores it: $PROFILE comes from the
    # known-folder API there.
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
    res = subprocess.run(
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
    reported = dict(
        line.strip().split("=", 1) for line in res.stdout.splitlines() if "=" in line
    )
    return {scope: Path(reported.get(scope, "")) for scope in _PROFILE_SCOPES}


def _run_with_profile(
    tmp_path: Path,
    body: str,
    path_prepend: Path | None = None,
    path_override: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run `body` with the hostile profile dot-sourced into the scope that encloses it.

    That is the scope relationship a profile has to the installer: the profile runs at global
    scope, then "irm ... | iex" defines and calls Install-UnslothStudio in that same scope.
    Dot-sourcing reproduces it on every OS, which planting a profile file does not -- Windows
    resolves the Documents folder through the shell's known-folder API, so no environment
    variable can redirect $PROFILE at a fixture. test_a_real_profile_reproduces_the_same_state
    anchors this against a genuinely loaded profile wherever that is possible.
    """
    profile = tmp_path / "hostile_profile.ps1"
    profile.write_text(_HOSTILE_PROFILE, encoding = "utf-8")
    script = tmp_path / "body.ps1"
    script.write_text(f". {_ps_literal(profile)}\n{body}\n", encoding = "utf-8")
    # Absolute path: PATH is replaced in some cases, so pwsh could not be found by name.
    return subprocess.run(
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
    # The two machine-wide profiles load into the real leg only, and no environment variable can
    # move them, so one that touches any probed setting would read as a divergence that says
    # nothing about the simulation. Neither file ships with PowerShell; this skip is for a host
    # that has been administered, and it names the file so the reason is checkable.
    for scope in ("AllUsersAllHosts", "AllUsersCurrentHost"):
        if paths[scope].is_file():
            pytest.skip(f"a machine-wide profile at {paths[scope]} loads into the real leg only")
    profile = paths["CurrentUserCurrentHost"]
    # Not an assert: on a host whose $PROFILE cannot be redirected into the fixture at all, the
    # real leg would load whatever the actual account has (or nothing) and prove nothing either
    # way. XDG_CONFIG_HOME in _hostile_env is what keeps this true on Linux and macOS, so the
    # skip is unreachable there and a regression in the simulation still fails locally and in CI.
    if tmp_path not in profile.parents:
        pytest.skip(f"pwsh resolves $PROFILE to {profile}, which this fixture cannot plant into")
    profile.parent.mkdir(parents = True, exist_ok = True)
    profile.write_text(_HOSTILE_PROFILE, encoding = "utf-8")
    script = tmp_path / "real.ps1"
    script.write_text(_STATE_PROBE, encoding = "utf-8")

    # -NoProfile deliberately omitted; this is the one place the profile is really loaded.
    real = subprocess.run(
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
    # All four probes coming back at their defaults means the file was planted somewhere pwsh
    # does not read, which is a broken fixture rather than a divergence; the path says which.
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
    assert "MATCHED:none" in res.stdout, (
        "if the bare token resolves past the alias, this suite can no longer detect a regression"
    )
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
    # The interfering default is gone; the proxy one is kept, for the installer and its callees.
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
    assert "THREW:" not in res.stdout, (
        f"a failing native command became terminating: {res.stdout!r}"
    )
    assert "AFTER_NATIVE:7" in res.stdout, (
        "the exit code must still be readable, which is how every caller here branches"
    )


def _uv_probe_body(*extra: str) -> str:
    return "\n".join(
        [
            _guarded(
                '    $UvMinVersion = "0.8.16"',
                _extract_function("Resolve-UvExecutable"),
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
    # a scriptblock built in one scope and invoked with & from another. It has to be defined
    # ahead of the call, which is what the "Install-UnslothStudio" line inside _uv_probe_body is.
    body = "\n".join(
        [
            "function Invoke-InstallCommandStub "
            '{ param([ScriptBlock]$Command); "CALLSITE:$(& $Command)" }',
            _uv_probe_body("    Invoke-InstallCommandStub { & $script:UvExe pip install nothing }"),
        ]
    )
    res = _run_with_profile(tmp_path, body, path_prepend = fake.parent)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "OK:True" in res.stdout, (
        "a uv on PATH must be detected even with an alias shadowing the name"
    )
    assert f"PINNED:{fake}" in res.stdout, (
        f"the probe must pin {str(fake)!r} exactly; got {res.stdout!r}"
    )
    # The scriptblock form the installer actually uses must reach the executable, not the alias.
    assert "CALLSITE:uv 0.12.1" in res.stdout, (
        "install scriptblocks must run the resolved uv; the alias would echo the arguments"
    )


@requires_pwsh
def test_uv_probe_rejects_a_too_old_uv_and_leaves_the_reset_value(tmp_path):
    """The version gate still has to fail closed, or a stale uv would be pinned and used."""
    fake = _fake_uv(tmp_path / "bin", version = "0.7.0")
    res = _run_with_profile(tmp_path, _uv_probe_body(), path_prepend = fake.parent)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "OK:False" in res.stdout
    assert "PINNED:uv" in res.stdout, "a rejected uv must not be pinned over the reset value"


@requires_pwsh
def test_uv_probe_reports_missing_when_only_the_alias_exists(tmp_path):
    """With no uv on PATH the installer must still take its install-uv branch, not pin the alias."""
    # The inherited PATH is replaced, not prepended to: the machine running this may well have a
    # real uv, and it would answer the probe and hide the branch under test.
    res = _run_with_profile(tmp_path, _uv_probe_body(), path_override = tmp_path / "emptybin")
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "OK:False" in res.stdout
    assert "PINNED:uv" in res.stdout


@requires_pwsh
def test_install_ps1_parses():
    """A syntax error here is a total install failure, and the file is not imported by anything."""
    res = subprocess.run(
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

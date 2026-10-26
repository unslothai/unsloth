# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
from pathlib import Path
import subprocess
import sys

import pytest
from unsloth_pwsh_runner import PWSH, run_pwsh

REPO = Path(__file__).resolve().parents[2]
POSIX = pytest.mark.skipif(os.name == "nt", reason = "POSIX")
WINDOWS = pytest.mark.skipif(not PWSH, reason = "PowerShell")
PS_PRELUDE = """$ErrorActionPreference = 'Stop'
function Write-StudioLine { param($Message, $ForegroundColor) }
function substep { param($Message) }
function Restore-StudioTempEnvironment {}
function Exit-SetupFailure { param($Message) throw $Message }
"""


def _part(filename, start, end):
    source = (REPO / filename).read_text(encoding = "utf-8")
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


@pytest.fixture
def env(tmp_path):
    result = dict(os.environ, HOME = str(tmp_path), USERPROFILE = str(tmp_path))
    for key in (
        "UNSLOTH_HOME",
        "UNSLOTH_STUDIO_HOME",
        "STUDIO_HOME",
        "UNSLOTH_LLAMA_CPP_PATH",
        "UNSLOTH_SETUP_PYTHON",
        "UNSLOTH_STUDIO_STAGE_ROOT",
    ):
        result.pop(key, None)
    return result


def _run(
    script,
    env,
    cwd,
    powershell = False,
    expected_exit = 0,
):
    if (powershell and not PWSH) or (not powershell and os.name == "nt"):
        pytest.skip("shell unavailable")
    argv = [PWSH, "-NoProfile", "-NonInteractive", "-Command"] if powershell else ["sh", "-c"]
    run = run_pwsh if powershell else subprocess.run
    result = run(argv + [script], env = env, cwd = cwd, capture_output = True, text = True, timeout = 60)
    assert result.returncode == expected_exit, result.stdout + result.stderr
    return result


def _root_script(filename):
    if filename == "install.sh":
        body = _part(filename, "_resolve_studio_destinations() {", "\n}\n")
        return (
            "set -eu\nsubstep() { :; }\n"
            + body
            + '\n}\n_resolve_studio_destinations\necho "$STUDIO_HOME|$_STUDIO_HOME_REDIRECT"'
        )
    if filename == "studio/setup.sh":
        return (
            'set -eu\nsetup_fail() { exit "$1"; }\n'
            + _part(filename, '_studio_override_var=""', "\nSTAGE_ROOT=")
            + '\necho "$STUDIO_HOME"'
        )
    if filename == "install.ps1":
        body = "$defaultProfile = $null\n$defaultDataDir = 'unused'\n$tauriChosenRoot = $false\n"
        body += "$tauriProfile = $env:USERPROFILE\n"
        body += _part(filename, "    $envOverrideVar = $null", "    $defaultProfile = $null")
        for name in (
            "Resolve-StudioLinkTarget",
            "Get-StudioSubstTarget",
            "Get-StudioLexicalPath",
            "Resolve-StudioFinalPathInfo",
            "Get-StudioFinalPath",
            "Test-StudioSameDirectoryByProbe",
            "Split-StudioExistingAncestor",
        ):
            body += _part(filename, f"    function {name} {{", "\n    }\n") + "\n    }\n"
        body += "function Get-StudioPythonFinalPath { param($Path) return $null }\n"
        body += "function Write-StudioFinalPathDegraded { param($Reason) }\n"
        body += _part(
            filename,
            "    # Custom Unsloth roots are not supported with --tauri",
            "    # LOCALAPPDATA may be unset",
        )
        body += _part(
            filename, "    # Cleared per invocation:", "    $VenvDir = Join-Path $StudioHome"
        )
        return PS_PRELUDE + body + '\n"$StudioHome|$StudioRedirectMode"'
    return (
        PS_PRELUDE
        + _part(filename, "$_studioOverrideVar = $null", "$StudioOwnedMarker = ")
        + '\n"$StudioHome"\n'
    )


@pytest.mark.parametrize(
    "filename", ["install.sh", "studio/setup.sh", "install.ps1", "studio/setup.ps1", "cli"]
)
@pytest.mark.parametrize(
    "studio, alias, master, expected",
    [
        ("", "", "", ".unsloth/studio"),
        ("chosen", "", "", "chosen"),
        ("", "alias", "", "alias"),
        ("chosen", "alias", "", "chosen"),
        ("chosen", "", "portable", "chosen"),
        ("", "alias", "portable", "alias"),
        ("", "", "{home}/portable [1]", "portable [1]/studio"),
        (" \t ", " ", " portable [1] ", "portable [1]/studio"),
        ("chosen", "ignored", "portable", "chosen"),
        (" \t ", "alias", "portable", "alias"),
        ("", "", "~/portable", "portable/studio"),
        ("", "", "~", "studio"),
        ("~/chosen space [1]", "", "", "chosen space [1]"),
        ("", "~/alias space [1]", "", "alias space [1]"),
        ("", "", " \t ", ".unsloth/studio"),
    ],
)
def test_install_and_setup_root_precedence(
    tmp_path, env, filename, studio, alias, master, expected
):
    for key, value in zip(
        ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UNSLOTH_HOME"), (studio, alias, master)
    ):
        env[key] = value.format(home = tmp_path)
    root = tmp_path / expected
    if filename.startswith("studio/"):
        root.mkdir(parents = True)
    if filename == "cli":
        env["PYTHONPATH"] = str(REPO)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from unsloth_cli.commands.studio import STUDIO_HOME; print(STUDIO_HOME)",
            ],
            env = env,
            cwd = tmp_path,
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()).resolve() == root.resolve()
        return
    output = _run(
        _root_script(filename), env, tmp_path, powershell = filename.endswith(".ps1")
    ).stdout.strip()
    chosen, *mode = output.split("|")
    assert Path(chosen).resolve() == root.resolve()
    if mode and master.strip():
        assert mode == ["env"]


@POSIX
@pytest.mark.parametrize(
    "master, studio, alias, accepted",
    [
        (" portable [1] ", "", "", False),
        ("portable", " \t ", " ", False),
        ("", " \t ", "custom", False),
        ("portable", "{home}/.unsloth/studio", "ignored", True),
        ("~/.unsloth/", "", "", True),
        (".unsloth", "", "", True),
        ("./.unsloth", "", "", True),
        ("missing/../.unsloth", "", "", True),
        ("", ".unsloth/studio", "", True),
        ("", "", "./.unsloth/studio", True),
    ],
)
def test_desktop_guard_uses_selected_root(tmp_path, env, master, studio, alias, accepted):
    for key, value in zip(
        ("UNSLOTH_HOME", "UNSLOTH_STUDIO_HOME", "STUDIO_HOME"), (master, studio, alias)
    ):
        env[key] = value.format(home = tmp_path)
    # Run argument parsing and the guard, stopping before any install work.
    script = (
        _part("install.sh", "#!/bin/sh", "\n_is_verbose() {")
        + "\necho accepted\n}\n_unsloth_main --tauri"
    )
    result = _run(script, env, tmp_path, expected_exit = 0 if accepted else 1)
    assert (
        "accepted" in result.stdout if accepted else "not supported with --tauri" in result.stderr
    )
    assert not any(tmp_path.iterdir())


@WINDOWS
@pytest.mark.parametrize("handoff", [None, "UNSLOTH_STUDIO_STAGE_ROOT", "UNSLOTH_SETUP_PYTHON"])
def test_setup_reuses_python_without_system_python(tmp_path, env, handoff):
    studio = tmp_path / "portable" / "studio"
    studio.mkdir(parents = True)
    env["UNSLOTH_HOME"] = str(studio.parent)
    runtime = tmp_path / "handoff [1]" if handoff else studio
    venv = runtime / "unsloth_studio"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check = True, timeout = 60
    )
    python = venv / "Scripts" / "python.exe"
    if os.name != "nt":
        python.parent.mkdir()
        python.symlink_to(venv / "bin" / "python")
    if handoff:
        env[handoff] = str(python if handoff == "UNSLOTH_SETUP_PYTHON" else runtime)
    script = (
        _root_script("studio/setup.ps1")
        + """
$env:PATH = ''
function winget { throw 'Unexpected system Python installation' }
"""
    )
    script += _part(
        "studio/setup.ps1", "$CondaSkipPattern = ", "# Add user-scheme Python Scripts dir"
    )
    script += (
        '\nif (-not $PythonOk) { throw "Python prerequisite failed" }\n"$ValidatedSetupPython"'
    )
    output = _run(script, env, tmp_path, powershell = True).stdout.splitlines()
    assert output[-1] == str(python)


@WINDOWS
def test_windows_relative_master_survives_directory_change(tmp_path, env):
    env["UNSLOTH_HOME"] = "portable"
    (tmp_path / "elsewhere").mkdir()
    script = PS_PRELUDE + "function Get-PathState { param($Path, $PathType) return 'Missing' }\n"
    script += _part("install.ps1", "    function Get-CanonicalDir {", "\n    }\n") + "\n    }\n"
    script += _part(
        "install.ps1", "    function Get-MasterRootOverride {", "    # Explicit staging root,"
    )
    script += "\nSet-Location -LiteralPath 'elsewhere'\nGet-MasterRootOverride"
    assert (
        Path(_run(script, env, tmp_path, powershell = True).stdout.strip()) == tmp_path / "portable"
    )


@WINDOWS
def test_windows_install_root_uses_powershell_working_directory(tmp_path, env):
    caller = tmp_path / "caller space [1]"
    caller.mkdir()
    env["UNSLOTH_HOME"] = "portable"
    script = "Set-Location -LiteralPath 'caller space [1]'\n" + _root_script("install.ps1")
    output = _run(script, env, tmp_path, powershell = True).stdout.strip()
    assert Path(output.split("|")[0]) == caller / "portable" / "studio"
    assert not (tmp_path / "portable").exists()


@WINDOWS
@pytest.mark.parametrize("variable", ["UNSLOTH_HOME", "UNSLOTH_STUDIO_HOME", "STUDIO_HOME"])
@pytest.mark.parametrize("accepted", [False, True])
def test_windows_tauri_root_uses_powershell_working_directory(tmp_path, env, variable, accepted):
    caller = tmp_path / "caller space [1]"
    caller.mkdir()
    env["USERPROFILE"] = str(caller)
    env.pop("UNSLOTH_DESKTOP_CHOSEN_STUDIO_HOME", None)
    root = ".unsloth" if accepted else "custom"
    env[variable] = root if variable == "UNSLOTH_HOME" else root + "/studio"
    script = "$TauriMode = $true\nSet-Location -LiteralPath 'caller space [1]'\n"
    result = _run(
        script + _root_script("install.ps1"),
        env,
        tmp_path,
        powershell = True,
        expected_exit = 0 if accepted else 1,
    )
    if accepted:
        assert Path(result.stdout.strip().split("|")[0]) == caller / ".unsloth" / "studio"
    else:
        assert "not supported with --tauri" in result.stderr
        assert not any(caller.iterdir())
    assert not (tmp_path / root).exists()


@POSIX
@pytest.mark.parametrize("prefix", ["", "~/"])
def test_saved_launcher_keeps_runtime_path_without_master_env(tmp_path, env, prefix):
    master = tmp_path / "portable space's [1]"
    studio = master / "studio"
    (studio / "share").mkdir(parents = True)
    (master / "llama.cpp").mkdir()
    config = studio / "share" / "studio.conf"
    saved_env = dict(
        env,
        UNSLOTH_HOME = f" {prefix}{master.name} ",
        STUDIO_HOME = str(studio),
        _STUDIO_HOME_REDIRECT = "env",
        _css_data_dir = str(config.parent),
        _css_quoted_exe = "unused",
        _css_id_file = "unused",
    )
    block = _part(
        "install.sh",
        "    {\n        printf '%s\\n' \"UNSLOTH_EXE=",
        ' > "$_css_data_dir/studio.conf"',
    )
    _run(block + ' > "$_css_data_dir/studio.conf"', saved_env, tmp_path)
    env["TEST_CONFIG"] = str(config)
    for override in ("", str(tmp_path / "external llama")):
        env["UNSLOTH_LLAMA_CPP_PATH"] = override
        script = (
            '. "$TEST_CONFIG"\nprintf "%s\\n%s\\n" "$UNSLOTH_STUDIO_HOME" "$UNSLOTH_LLAMA_CPP_PATH"'
        )
        output = _run(script, env, tmp_path).stdout.splitlines()
        assert output == [str(studio), override or str(master / "llama.cpp")]

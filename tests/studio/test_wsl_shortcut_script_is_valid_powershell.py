# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The PowerShell install.sh generates for the WSL shortcut has to be valid PowerShell.

Nothing else checks. install.sh is shell, so `bash -n` covers the here-string as a here-string
and says nothing about its contents; the contents only ever run on a Windows machine, through
`powershell.exe -File`, with stdout and stderr both sent to /dev/null and the exit status folded
into a best-effort `&& _css_created=1`. A parse error there is not a crash and not a message: the
whole script fails to load, no shortcut appears, and the user gets the "WSL interop may be
disabled" notice, which points at the wrong thing entirely.

That path is one edit away at all times, and the icon refresh inside it sits in `try { } catch { }`,
so even at RUNTIME on the right machine a mistake there is swallowed.

The refresh tests are the other half: syntax is not enough. The per-item SHChangeNotify runs in a
Windows Python through ctypes, and both halves of that are checked here on Linux: the Python it
passes is run against a recording stand-in for shell32, and the PowerShell that launches it is run
under pwsh against a stand-in interpreter, so a quoting mistake in the command line, a lost path or
a wrong flag fails here instead of silently on a user's desktop.
"""

from __future__ import annotations

import re
import shlex
import sys
import shutil
import subprocess
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO / "install.sh"

needs_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")

# `_render` is a pure function of install.sh's bytes, so Linux and macOS cover it; on Windows
# `bash` is Git Bash or the WSL stub, neither of which ever runs install.sh.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason = "renders install.sh with bash; POSIX shell installer test",
)

# Awkward on purpose: a distro name with a space, and an apostrophe PowerShell must double.
_RENDER_VARS = {
    "_css_sc_target": "wt.exe",
    "_css_sc_args_ps": 'wsl.exe -d "O\'\'Brien 24.04" -- bash -l -c "exec /home/u/launch.sh"',
    "_css_lnk_name_ps": "Unsloth Studio (WSL - O''Brien 24.04).lnk",
    "_css_wsl_ico_win_ps": "C:\\Users\\ci\\unsloth.ico",
}


def _render() -> str:
    """The here-string as bash would actually write it, not a copy of it.

    Extracted from the shipped file and expanded by a real shell, so every backslash escape in
    it is resolved the way install.sh resolves it. A test holding its own copy of this script
    would keep passing after install.sh broke.
    """
    text = INSTALL_SH.read_text(encoding = "utf-8")
    body = re.search(
        r"(?ms)^            _css_ps1_body=\$\(cat << WSLPS1_EOF\n(.*?)^WSLPS1_EOF$",
        text,
    )
    assert body, (
        "could not find the WSLPS1_EOF here-string in install.sh. Either the WSL shortcut script "
        "moved or it is no longer generated, and until this locates it again nothing checks that "
        "what install.sh writes is PowerShell at all."
    )
    # shlex.quote, not repr: the values contain shell quotes and repr quotes for Python.
    assigns = "".join(f"{k}={shlex.quote(v)}\n" for k, v in _RENDER_VARS.items())
    script = assigns + "cat << WSLPS1_EOF\n" + body.group(1) + "WSLPS1_EOF\n"
    # check = False then assert: CalledProcessError drops the shell's own diagnostic.
    done = subprocess.run(["bash", "-c", script], capture_output = True, text = True)
    assert done.returncode == 0, (
        f"bash could not render the here-string (exit {done.returncode}):\n"
        f"{done.stderr.strip()}\n{done.stdout.strip()[:2000]}"
    )
    assert "SHChangeNotify" in done.stdout, done.stdout
    return done.stdout


@needs_pwsh
def test_the_generated_wsl_shortcut_script_parses() -> None:
    rendered = _render()
    probe = (
        "$errors = $null; $tokens = $null; "
        "$text = [Console]::In.ReadToEnd(); "
        "$null = [System.Management.Automation.Language.Parser]::ParseInput("
        "$text, [ref]$tokens, [ref]$errors); "
        "if ($errors -and $errors.Count) { "
        "  $errors | ForEach-Object { Write-Output ("
        "    'PARSE_ERROR line ' + $_.Extent.StartLineNumber + ': ' + $_.Message) }; "
        "} else { Write-Output 'PARSE_OK' }"
    )
    done = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", probe],
        input = rendered,
        capture_output = True,
        text = True,
        verdict = "PARSE_OK",
    )
    assert "PARSE_OK" in done.stdout, (
        "the PowerShell install.sh generates for the WSL shortcut does not parse, so on a real "
        "machine it would load nothing and the user would be told WSL interop is disabled:\n"
        f"{done.stdout.strip()}\n{done.stderr.strip()}"
    )


_SHELL32_STUB = """
import ctypes, json, os, sys
calls = []
class _Fn:
    def __call__(self, *a):
        calls.append(list(a))
class _Dll:
    def __init__(self, name):
        assert name == "shell32", name
        self.SHChangeNotify = _Fn()
ctypes.WinDLL = _Dll
exec(sys.argv[1])
sys.stderr.write(json.dumps(calls))
"""


def _refresh_code(rendered: str) -> str:
    match = re.search(r'(?m)^    \$refreshCode = "([^"\r\n]*)"$', rendered)
    assert match, "could not find the icon refresh's Python in the generated script"
    return match.group(1)


_LINKS = [
    "C:\\Users\\O'Brien\\Desktop\\Unsloth Studio (WSL - O'Brien 24.04).lnk",
    "C:\\Users\\O'Brien\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\x.lnk",
]
_EXPECTED_CALLS = [[0x2000, 0x1005, link, None] for link in _LINKS] + [
    [0x8000000, 0x1000, None, None]
]


def test_the_icon_refresh_python_notifies_each_shortcut_then_the_shell() -> None:
    """Per-item SHCNE_UPDATEITEM for every shortcut, then SHCNE_ASSOCCHANGED, all flushed."""
    import json

    done = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", _SHELL32_STUB, _refresh_code(_render())],
        capture_output = True,
        text = True,
        env = {"UNSLOTH_SHORTCUT_PATHS": "|".join(_LINKS)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok", done.stdout
    assert json.loads(done.stderr) == _EXPECTED_CALLS


@needs_pwsh
def test_the_icon_refresh_launch_runs_one_interpreter_with_every_shortcut(tmp_path: Path) -> None:
    """The PowerShell half: candidate order, the WindowsApps skip, argv quoting, stop on success.

    The block is lifted from the rendered script as written; only the elevation probe, which has
    no answer off Windows, is pinned to "not elevated".
    """
    import json

    rendered = _render()
    match = re.search(r"(?ms)^(if \(\$created\.Count -gt 0\) \{\n.*?^\})\n", rendered)
    assert match, "could not find the icon refresh block in the generated script"
    block = match.group(1)
    elevation = re.search(r"(?m)^        \$isAdmin = \(\[Security.*$", block)
    assert elevation, "the refresh no longer checks elevation before launching an interpreter"
    block = block.replace(elevation.group(0), "        $isAdmin = $false")

    log = tmp_path / "calls.jsonl"
    stub = tmp_path / "stub.py"
    stub.write_text(
        _SHELL32_STUB.replace(
            "sys.stderr.write(json.dumps(calls))",
            "open(os.environ['STUB_LOG'], 'a').write(json.dumps("
            "{'argv': sys.argv[2:], 'cwd': os.getcwd(), 'calls': calls}) + '\\n')",
        )
    )
    venv_python = (
        tmp_path / "home" / ".unsloth" / "studio" / "unsloth_studio" / "Scripts" / "python.exe"
    )
    venv_python.parent.mkdir(parents = True)
    venv_python.write_text(
        "#!/bin/sh\n"
        f'[ "$1 $2 $3 $4" = "-I -S -B -c" ] || exit 7\n'
        f'exec {shlex.quote(sys.executable)} -I -S -B {shlex.quote(str(stub))} "$5"\n'
    )
    venv_python.chmod(0o755)
    path_dir = tmp_path / "bin"
    path_dir.mkdir()
    (path_dir / "python3").write_bytes(venv_python.read_bytes())
    (path_dir / "python3").chmod(0o755)
    links = ", ".join("'" + link.replace("'", "''") + "'" for link in _LINKS)
    script = (
        f"$env:USERPROFILE = '{tmp_path / 'home'}'\n"
        f"$env:STUB_LOG = '{log}'\n"
        f"$env:PATH = '{path_dir}' + [IO.Path]::PathSeparator + $env:PATH\n"
        f"$created = @({links})\n" + block + "\n"
        "Write-Output 'BLOCK_DONE'\n"
    )
    runner = tmp_path / "refresh.ps1"
    runner.write_text(script, encoding = "utf-8")
    done = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(runner)],
        capture_output = True,
        text = True,
        verdict = "BLOCK_DONE",
    )
    assert "BLOCK_DONE" in done.stdout, f"{done.stdout}\n{done.stderr}"
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(entries) == 1, entries
    assert entries[0]["calls"] == _EXPECTED_CALLS
    assert entries[0]["cwd"] == str(venv_python.parent)


def test_the_generated_script_defines_no_native_type() -> None:
    """install.sh held the last reflection-emit P/Invoke in the tree; it must not come back."""
    rendered = _render()
    for token in (
        "Add-Type",
        "DefinePInvoke" + "Method",
        "DefineDynamic" + "Assembly",
        "DllImport",
    ):
        assert (
            token not in rendered
        ), f"the WSL shortcut script declares a native type again: {token}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_the_wsl_install_is_watched_live_for_a_compiler() -> None:
    """A search run after the install cannot see the compile it is looking for.

    CodeDom deletes its whole intermediate directory once the assembly is loaded, which
    `.github/scripts/Watch-ForCompiler.ps1` records as a before-and-after diff seeing "nothing at
    all while 4688 recorded csc.exe". So a post-hoc listing of `*.cmdline` cannot fail for the case
    this lane exists to catch, and a positive control that plants persistent files only proves the
    search can traverse a directory. The install has to run INSIDE the watcher.
    """
    import yaml

    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "clean-machine-install-ci.yml").read_text(
            encoding = "utf-8"
        )
    )
    steps = [
        step
        for job in workflow["jobs"].values()
        for step in (job.get("steps") or [])
        # The WSL leg only: Linux container legs run the same piped install with no Windows side.
        if "cat install.sh | sh" in str(step.get("run", ""))
        and "wsl -d unsloth-ci" in str(step.get("run", ""))
    ]
    assert steps, "nothing runs the piped WSL install any more"
    for step in steps:
        run = str(step["run"])
        assert "Invoke-WithCompilerWatch" in run, (
            "the WSL install is not wrapped in the live compiler watch, so a compile that cleans up "
            "after itself would go unseen and the lane would report it clean"
        )
        assert "$seen.Compilers" in run, "the watch result is never inspected"
        assert "$seen.TempLibraries" in run, (
            "the dropped-library half is never inspected, and that is the half the Bitdefender "
            "report in #10540 keyed on"
        )


def test_the_wsl_lane_arms_the_4688_half_of_the_watch() -> None:
    """The watcher's process half was dead in this lane, and a dead half reads as a clean one.

    `Watch-ForCompiler.ps1` says outright that 4688 "needs auditing enabled by the caller". Its
    no-match path returns an empty compiler list whenever the Security log is merely readable, so
    with auditing off `$seen.Compilers` is empty no matter what ran, and the step still prints that
    no compiler process was seen. That leaves the FileSystemWatcher carrying the whole result alone
    while the output claims two detectors. Require the audit policy to be turned on, verified, and
    shown to produce a real detection before the install is judged.
    """
    import yaml

    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "clean-machine-install-ci.yml").read_text(
            encoding = "utf-8"
        )
    )
    job = next(
        job
        for job in workflow["jobs"].values()
        if any(
            "cat install.sh | sh" in str(step.get("run", ""))
            and "wsl -d unsloth-ci" in str(step.get("run", ""))
            for step in (job.get("steps") or [])
        )
    )
    runs = [str(step.get("run", "")) for step in (job.get("steps") or [])]
    index = next(i for i, run in enumerate(runs) if "cat install.sh | sh" in run)
    # Only steps before the install count: auditing turned on afterwards measures nothing.
    earlier = "\n".join(runs[:index])
    assert 'auditpol /set /subcategory:"Process Creation" /success:enable' in earlier, (
        "the WSL lane never enables process creation auditing, so the 4688 half of the watch is "
        "dead and an empty compiler list means unmeasured rather than clean"
    )
    assert (
        "ProcessCreationIncludeCmdLine_Enabled" in earlier
    ), "without the command line, 4688 cannot tell csc.exe ran for us from csc.exe ran"
    assert (
        "Process Creation\\s+Success" in earlier
    ), "the audit policy is set but never verified, and machine policy can silently override it"
    assert "$control.TempLibraries" in earlier, (
        "the control validates only the process half, so a FileSystemWatcher that never attached "
        "looks exactly like a run that dropped no library -- and that is the half the Bitdefender "
        "report in #10540 keyed on"
    )
    assert "$control.Compilers" in earlier, (
        "nothing proves the detector fires on this runner, so a clean verdict is indistinguishable "
        "from a detector that never attached"
    )

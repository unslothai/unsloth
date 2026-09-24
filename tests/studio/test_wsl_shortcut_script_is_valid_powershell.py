# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The PowerShell install.sh generates for the WSL shortcut has to be valid PowerShell.

Nothing else checks. install.sh is shell, so `bash -n` covers the here-string as a here-string
and says nothing about its contents; the contents only ever run on a Windows machine, through
`powershell.exe -File`, with stdout and stderr both sent to /dev/null and the exit status folded
into a best-effort `&& _css_created=1`. A parse error there is not a crash and not a message: the
whole script fails to load, no shortcut appears, and the user gets the "WSL interop may be
disabled" notice, which points at the wrong thing entirely.

That path is one edit away at all times. The block this guards was just rewritten from a
one-line `Add-Type -MemberDefinition` into a twenty-line reflection-emit sequence with nested
calls, an array literal and a here-string escaping every `$` it means to keep -- and it sits
inside `try { } catch { }`, so even at RUNTIME on the right machine a mistake inside it is
swallowed.

The emit test is the other half: syntax is not enough, the API sequence has to be real. shell32
is Windows-only, but DefineDynamicAssembly / DefineDynamicModule / DefineType /
DefinePInvokeMethod / CreateType is not, so the same sequence is pointed at a symbol this host
does export and the resulting method is called. A wrong call order or a wrong argument count
fails here, on Linux, instead of silently on a user's desktop.

What that is worth, measured rather than asserted. Mutating the shipped file five ways: a dropped
closing paren in DefinePInvokeMethod, an unterminated string literal in DefineType and a stray
closing brace all fail both tests or the parse one; giving the P/Invoke the wrong number of
parameters fails the emit one. Dropping SetImplementationFlags does NOT fail, and cannot: it sets
PreserveSig, whose only effect is on how a failing HRESULT is surfaced, and the retargeted symbol
here returns a plain int. That line is unguarded and is the one place a reviewer still has to
read for themselves.
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

# What `_render` produces is a pure function of install.sh's bytes and the four values below, so the
# Linux and macOS legs cover the subject completely and a Windows leg adds no coverage of it. What a
# Windows leg does add is a different shell: `bash` there is Git Bash or the WSL stub in System32,
# neither of which install.sh is ever run by -- its WSL arm is gated on /proc/version naming
# microsoft, so the shell that reads this here-string is always a POSIX one inside the distro. The
# repository already draws this line the same way for its other bash-driven installer tests
# (tests/python/test_install_uv_override_space.py:21,
# tests/studio/install/test_selection_logic.py:4185).
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason = "renders install.sh with bash; POSIX shell installer test",
)

# Values install.sh would interpolate, chosen to be awkward: a distro name with a space, and the
# apostrophe the surrounding single-quoted PowerShell literals have to be doubling.
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
        # The body is captured into a variable now, because it is either written to a file or piped
        # to powershell on stdin depending on whether a Windows directory is reachable.
        r"(?ms)^            _css_ps1_body=\$\(cat << WSLPS1_EOF\n(.*?)^WSLPS1_EOF$",
        text,
    )
    assert body, (
        "could not find the WSLPS1_EOF here-string in install.sh. Either the WSL shortcut script "
        "moved or it is no longer generated, and until this locates it again nothing checks that "
        "what install.sh writes is PowerShell at all."
    )
    # shlex.quote, not repr: these values deliberately contain the apostrophes and double quotes
    # the real ones do, and Python's repr quotes for Python.
    assigns = "".join(f"{k}={shlex.quote(v)}\n" for k, v in _RENDER_VARS.items())
    script = assigns + "cat << WSLPS1_EOF\n" + body.group(1) + "WSLPS1_EOF\n"
    # check = False, then asserted. CalledProcessError carries the command and the return code and
    # drops the shell's own diagnostic, so a rendering that fails to render reported a 6 KB repr of
    # the script and not the one line saying what was wrong with it.
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


@needs_pwsh
def test_the_icon_refresh_emit_sequence_builds_a_callable_type() -> None:
    rendered = _render()
    emit = re.search(r"(?ms)^try \{\n(    \$refreshType = .*?)\n\} catch \{\}", rendered)
    assert emit, (
        "could not find the icon-refresh emit block in the generated script; if SHChangeNotify "
        "is now declared some other way, update this test rather than deleting it"
    )
    # Retargeted, not rewritten: only the library, the symbol and its signature change, so the
    # call sequence under test is the shipped one character for character.
    body = emit.group(1)
    body = body.replace(
        "'SHChangeNotify', 'shell32.dll', 'SHChangeNotify'", "'getpid', 'libc', 'getpid'"
    )
    body = re.sub(r"\[System\.Void\],", "[int],", body)
    body = re.sub(r"@\(\[int\], \[uint32\], \[string\], \[IntPtr\]\),", "@(),", body)
    body = body.replace(
        "[System.Runtime.InteropServices.CharSet]::Unicode)",
        "[System.Runtime.InteropServices.CharSet]::Ansi)",
    )
    body = "\n".join(l for l in body.splitlines() if "SHChangeNotify(" not in l)
    body = body.replace("UnslothShellIconRefresh", "UnslothEmitProbe")
    assert "DefinePInvokeMethod" in body and "getpid" in body, body

    # Twice, because the block caches on `-as [type]`: a second pass must reuse the type rather
    # than throw on a duplicate name, which is what makes the guard safe to leave in.
    script = body + "\nWrite-Output ('EMIT_A ' + [string]$refreshType::getpid())\n"
    script += body + "\nWrite-Output ('EMIT_B ' + [string]$refreshType::getpid())\n"
    done = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        verdict = "EMIT_B",
    )
    pids = re.findall(r"EMIT_[AB] (\d+)", done.stdout)
    assert len(pids) == 2, (
        "the reflection-emit sequence in the generated script does not build a callable P/Invoke "
        "type. shell32 is Windows-only but this sequence is not, so this is the sequence itself "
        f"being wrong:\n{done.stdout.strip()}\n{done.stderr.strip()}"
    )
    assert pids[0] == pids[1], f"the cached type was rebuilt instead of reused: {pids}"


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
        # The WSL leg specifically. The Linux container legs run the same piped install and
        # have no Windows side to watch.
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
    # Only the steps BEFORE the install count: auditing turned on afterwards measures nothing.
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


def test_the_generated_emit_carries_both_spellings_of_define_dynamic_assembly() -> None:
    """The outer catch is empty, so guessing the wrong spelling costs the icon refresh silently.

    `install.ps1`'s `New-StudioDynamicAssembly` tries the static
    `AssemblyBuilder::DefineDynamicAssembly` and falls back to
    `AppDomain.CurrentDomain.DefineDynamicAssembly`, and says why: the static form is documented for
    .NET Framework 4.5 through 4.8.1, so the Windows PowerShell 5.1 host this script is launched
    under should take the first branch, but nothing in this repository can run a .NET Framework host
    to confirm it. The generated WSL script had only the static form under a bare `catch {}`, which
    is the combination that fails invisibly. Carry the same fallback.
    """
    script = _render()
    assert (
        "[System.Reflection.Emit.AssemblyBuilder]::DefineDynamicAssembly" in script
    ), "the generated script no longer tries the documented static spelling first"
    assert "[AppDomain]::CurrentDomain.DefineDynamicAssembly" in script, (
        "the generated script has no AppDomain fallback, so on a host without the static overload "
        "the empty outer catch swallows the failure and both icon refreshes stop happening"
    )
    static_at = script.index("[System.Reflection.Emit.AssemblyBuilder]::DefineDynamicAssembly")
    appdomain_at = script.index("[AppDomain]::CurrentDomain.DefineDynamicAssembly")
    assert (
        static_at < appdomain_at
    ), "AppDomain.CurrentDomain is absent on .NET Core, so leading with it would break pwsh"

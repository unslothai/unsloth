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
import shutil
import subprocess
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO / "install.sh"

needs_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "needs PowerShell")

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
        r'(?ms)^            cat > "\$_css_ps1_tmp" << WSLPS1_EOF\n(.*?)^WSLPS1_EOF$', text
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
    done = subprocess.run(["bash", "-c", script], capture_output = True, text = True, check = True)
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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`UNSLOTH_INSTALL_TIMING` prefixes installer output with elapsed seconds, and only then.

`Install Unsloth (--local, --no-torch)` is the largest step in every Windows CI job: 260s of
Windows API CI's 374s, 291s of Windows UI CI's 715s, 281s of Windows Update CI's 794s. The
same install on Linux is 88s. Across the ~11 Windows cells a commit triggers, that is roughly
50 minutes of Windows runner time per commit spent installing the same thing.

Which phase spends it was, until this switch existed, unknowable from a CI log: neither
`studio/setup.ps1` nor `studio/setup.sh` emits a timestamp anywhere, and the one Stopwatch in
setup.ps1 sits inside the llama.cpp source-build branch that CI never takes. Guessing would
have been actively misleading -- `unsloth studio update` over an already-complete install
costs 297s, MORE than the 281s full install it follows, which is the opposite of what a
download-bound install does.

Two things have to hold, and the first is the one that would annoy real users rather than
break CI, so it is asserted rather than trusted:

  * OFF by default. In PowerShell every non-empty string is truthy, so a bare
    `[bool]$env:UNSLOTH_INSTALL_TIMING` treats "0" as enabled; in bash an unquoted default
    does the same. Both halves must reject "" and "0" explicitly.
  * The Windows install steps must actually request it, or the breakdown never appears in the
    logs that motivated the switch.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SETUP_PS1 = REPO / "studio" / "setup.ps1"
SETUP_SH = REPO / "studio" / "setup.sh"
WORKFLOWS = REPO / ".github" / "workflows"

ENV_VAR = "UNSLOTH_INSTALL_TIMING"


def test_the_powershell_half_rejects_zero_rather_than_treating_it_as_truthy():
    src = SETUP_PS1.read_text(encoding = "utf-8")
    line = next(
        (l for l in src.splitlines() if "StudioTimingEnabled" in l and "=" in l and "if" not in l),
        None,
    )
    assert line, "setup.ps1 no longer sets $script:StudioTimingEnabled"
    assert f"$env:{ENV_VAR} -ne '0'" in line, (
        f'the guard against a truthy "0" is gone from: {line.strip()!r}. PowerShell treats '
        f"every non-empty string as true, so {ENV_VAR}=0 would switch timing ON."
    )


def _sh_function(name: str) -> str:
    """The body of a shell function in setup.sh, as text."""
    src = SETUP_SH.read_text(encoding = "utf-8")
    start = src.index(f"\n{name}() {{")
    depth, i = 0, start
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1


@pytest.mark.parametrize("fn", ["step", "substep"])
def test_the_bash_half_rejects_zero_and_empty(fn):
    body = _sh_function(fn)
    assert re.search(
        r'""\|0\)', body
    ), f'setup.sh\'s {fn}() no longer treats "" and 0 as off:\n{body}'


def _bash_runs_posix_scripts() -> bool:
    """Whether `bash` here is a real POSIX shell, rather than Windows' WSL launcher.

    On a windows-latest runner `bash` resolves to the WSL stub, which ignores the script
    entirely and exits 1 with a UTF-16 "no distributions installed, use `wsl --install`"
    message on stdout. That is not a finding about setup.sh, so these two tests skip there
    rather than failing: the cross-platform parity job runs this file on Windows too, and
    setup.sh is not the installer Windows uses.

    Probed rather than keyed off sys.platform, so a Windows box that does have a working
    git-bash still runs them.
    """
    try:
        probe = subprocess.run(
            ["bash", "-c", "printf ok"], capture_output = True, text = True, timeout = 30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and probe.stdout.strip() == "ok"


BASH_OK = _bash_runs_posix_scripts()


def test_the_bash_probe_still_finds_bash_where_bash_exists():
    """A skip condition that silently became always-true would disable the two tests below.

    setup.sh is the installer for Linux and macOS, so on those platforms the probe failing
    means the probe is broken, not that the environment lacks a shell.
    """
    if sys.platform.startswith("win"):
        pytest.skip("Windows has no POSIX bash by default; that is the case being skipped")
    assert BASH_OK, (
        "the POSIX-bash probe failed on a platform that ships bash, so the two tests that "
        "actually execute setup.sh's print helpers are being skipped everywhere"
    )


@pytest.mark.skipif(not BASH_OK, reason = "no POSIX bash here (Windows resolves it to WSL)")
@pytest.mark.parametrize("fn", ["step", "substep"])
def test_the_bash_helpers_are_silent_by_default_and_prefix_when_asked(fn):
    """Run the real function. A text check alone would not catch a broken printf."""
    script = "C_DIM= C_OK= C_RST= C_WARN=\n" + _sh_function(fn) + f'\n{fn} "lbl" "msg"\n'
    base = {"PATH": "/usr/bin:/bin"}

    for env in (base, {**base, ENV_VAR: "0"}, {**base, ENV_VAR: ""}):
        r = subprocess.run(["bash", "-c", script], capture_output = True, text = True, env = env)
        assert r.returncode == 0, r.stderr
        assert "s] " not in r.stdout, f"timing leaked into default output: {r.stdout!r}"

    on = subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, env = {**base, ENV_VAR: "1"}
    )
    assert on.returncode == 0, on.stderr
    assert re.search(
        r"\[ *\d+s\] ", on.stdout
    ), f"{ENV_VAR}=1 produced no elapsed prefix: {on.stdout!r}"


def test_both_print_helpers_carry_the_prefix():
    """Prefixing one sink and not the other would time half the install."""
    src = SETUP_PS1.read_text(encoding = "utf-8")
    for fn, var in (("function step {", "$Value"), ("function substep {", "$Message")):
        block = src[src.index(fn) :][:1600]
        assert (
            "Variable:script:StudioTimingEnabled" in block
        ), f"{fn.strip()} no longer consults the timing switch"
        assert (
            f'{var} = ("[{{0,7:N1}}s] "' in block
        ), f"{fn.strip()} does not prefix {var} with the elapsed time"
    for fn in ("step", "substep"):
        assert "SECONDS" in _sh_function(fn), f"setup.sh {fn}() carries no timing"


def test_the_print_helpers_call_nothing_defined_elsewhere_in_the_file():
    """They are dot-sourced ON THEIR OWN, so a call out of them is a crash, not a warning.

    tests/python/test_windows_setup_output_encoding.py builds a probe script containing only
    Get-StudioAnsi, Write-StudioLine, Write-StudioStdoutMirror, step and substep, and runs it.
    A first cut of this feature had `step` call a `Get-StudioElapsedPrefix` helper defined at
    the top of setup.ps1; the probe then died with "The term 'Get-StudioElapsedPrefix' is not
    recognized" and took 12 tests with it. The timing logic is inline for that reason, and
    reads its state through Test-Path so an unset $script: variable is empty rather than
    fatal under a caller's Set-StrictMode.
    """
    src = SETUP_PS1.read_text(encoding = "utf-8")
    allowed = {
        "Get-StudioAnsi",
        "Write-StudioLine",
        "Write-StudioStdoutMirror",
        "Write-Host",
        "Test-Path",
    }
    for fn in ("function step {", "function substep {"):
        block = src[src.index(fn) :]
        block = block[: block.index("\n}\n") + 3]
        # Comments out: the block explains itself by NAMING the cmdlets it must not call.
        block = "\n".join(l for l in block.splitlines() if not l.lstrip().startswith("#"))
        called = set(re.findall(r"\b([A-Z][a-z]+-[A-Za-z]+)\b", block))
        stray = called - allowed
        assert not stray, (
            f"{fn.strip()} calls {sorted(stray)}, which the encoding probe does not "
            f"dot-source alongside it. Inline it instead."
        )


def _windows_install_steps():
    for f in sorted(WORKFLOWS.glob("studio-windows-*.yml")):
        doc = yaml.safe_load(f.read_text(encoding = "utf-8"))
        for jid, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if "install.ps1 --local --no-torch" in str(step.get("run", "")):
                    yield f.name, jid, step


def test_every_windows_install_step_asks_for_the_breakdown():
    steps = list(_windows_install_steps())
    assert steps, "no Windows step runs install.ps1 --local --no-torch any more"
    missing = [
        f"{name}:{jid}"
        for name, jid, step in steps
        if (step.get("env") or {}).get(ENV_VAR) not in ("1", 1)
    ]
    assert not missing, (
        f"these Windows install steps do not set {ENV_VAR}, so their logs stay unreadable "
        f"about where the 260-291s goes: {missing}"
    )


@pytest.mark.parametrize("script", [SETUP_PS1, SETUP_SH])
def test_timing_is_never_enabled_unconditionally(script):
    """The switch must stay a switch. Hardcoding it on changes what every user sees."""
    # Comments out: both files name the variable in prose, and `#` starts a comment in
    # PowerShell and bash alike.
    code = "\n".join(
        l for l in script.read_text(encoding = "utf-8").splitlines() if not l.lstrip().startswith("#")
    )
    for bad in (f'{ENV_VAR}="1"', f"{ENV_VAR}='1'", f"{ENV_VAR}=1"):
        assert bad not in code, f"{script.name} sets {ENV_VAR} itself: {bad}"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CI prefixes installer output with elapsed seconds, without touching the installers.

`Install Unsloth (--local, --no-torch)` is the largest step in most jobs that run it:
260-291s of a Windows job, ~90s median on Linux across 40 jobs. Which phase spends it was,
until this filter existed, unknowable from a CI log -- neither `install.sh` nor
`studio/setup.ps1` emits a timestamp anywhere. Guessing has been actively misleading:
`unsloth studio update --local` over an already-complete install costs 297s, MORE than the
281s full install it follows, which is the opposite of what a download-bound install does.

The timing is a **display filter on a stream CI already pipes**, not a feature of the
installers. That distinction is the whole design and it is what these tests guard:

  * `install.sh`, `install.ps1`, `studio/setup.sh` and `studio/setup.ps1` are user-facing
    and are not modified. No environment variable, no switch, no truthiness rule, and no
    way for a real user's install to behave differently from a CI one.
  * The filter sits **downstream of the log write**. `logs/install.log` keeps byte-for-byte
    what the installer produced, so the ~30 places that read or grep that artifact are
    unaffected -- including `interrupted-install-ci.yml:185`, which matches
    `^\\[TAURI:STEP\\]` anchored at line start and would silently stop matching if a prefix
    reached the file.

Both properties fail SILENTLY when broken -- a reordered pipeline still goes green, and an
installer edit still installs -- so they are asserted rather than reviewed.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from unsloth_pwsh_runner import run_pwsh

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
ACTION = REPO / ".github" / "actions" / "install-unsloth-local" / "action.yml"

# The four scripts this feature deliberately does not touch.
INSTALLERS = (
    REPO / "install.sh",
    REPO / "install.ps1",
    REPO / "studio" / "setup.sh",
    REPO / "studio" / "setup.ps1",
)

# Markers of the two filter dialects, each paired with the log-writing stage that must come before it in the same
# pipeline.
POSIX_FILTER = "printf '[%4ds] %s\\n' \"$SECONDS\""
PWSH_FILTER = "$sw.Elapsed.TotalSeconds"


# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", INSTALLERS, ids = lambda p: p.name)
def test_the_installers_carry_no_timing_machinery(script):
    """The first cut of this feature put the clock inside the installers. It should not.

    That version needed a `UNSLOTH_INSTALL_TIMING` switch, an off-by-default rule that
    differs between PowerShell (every non-empty string is truthy, so "0" enabled it) and
    bash, and a `UNSLOTH_INSTALL_TIMING_T0` epoch handed from the outer installer to the
    inner one -- which then had to be bounds-checked, because a parseable but out-of-range
    long crashes `[System.DateTime]::new(ticks)` and a non-numeric value aborts POSIX
    `$(( ))` under `set -u`. None of that exists now, and this test is what keeps it from
    coming back one convenience at a time.
    """
    src = script.read_text(encoding = "utf-8")
    assert "UNSLOTH_INSTALL_TIMING" not in src, (
        f"{script.name} interprets UNSLOTH_INSTALL_TIMING. The install timing is a CI-side "
        f"display filter over a stream that is already piped; putting it back inside the "
        f"installer re-adds a user-facing switch, a shell-specific truthiness rule and a "
        f"cross-process epoch handoff, for output CI can prefix for free."
    )


# --------------------------------------------------------------------------------------
# Where the filter is, and what has to come before it
# --------------------------------------------------------------------------------------


def _run_bodies():
    """Every `run:` body in the workflows and in the composite action, with its origin."""
    paths = sorted(WORKFLOWS.glob("*.yml")) + [ACTION]
    for path in paths:
        doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
        if not isinstance(doc, dict):
            continue
        if path == ACTION:
            groups = [("runs", (doc.get("runs") or {}).get("steps") or [])]
        else:
            groups = [
                (jid, job.get("steps") or [])
                for jid, job in (doc.get("jobs") or {}).items()
                if isinstance(job, dict)
            ]
        for jid, steps in groups:
            for step in steps:
                if isinstance(step, dict) and step.get("run"):
                    yield path, jid, step.get("name") or "<unnamed>", str(step["run"])


def _prefixing_bodies():
    for path, jid, name, run in _run_bodies():
        if POSIX_FILTER in run or PWSH_FILTER in run:
            yield path, jid, name, run


def test_the_filter_is_actually_wired_somewhere():
    """A scan that found nothing would pass every check below on an empty set."""
    bodies = list(_prefixing_bodies())
    assert len(bodies) >= 7, (
        f"only {len(bodies)} steps prefix installer output with elapsed seconds. Expected "
        f"the composite POSIX action, five Windows install.ps1 pipelines and the two "
        f"`unsloth studio update` steps."
    )


def test_every_windows_install_pipeline_is_timed():
    """Five steps run install.ps1 directly; a sixth added later must not be missed."""
    untimed = [
        f"{path.name}:{jid}:{name}"
        for path, jid, name, run in _run_bodies()
        if "install.ps1 --local --no-torch" in run and PWSH_FILTER not in run
    ]
    assert not untimed, (
        f"these Windows install steps produce no phase breakdown, so their 260-291s stays "
        f"unattributable: {untimed}"
    )


def test_the_posix_install_action_is_timed():
    run = next(
        (r for p, _, _, r in _run_bodies() if p == ACTION and "install.sh" in r),
        None,
    )
    assert run, "the install-unsloth-local action no longer runs install.sh"
    assert POSIX_FILTER in run, (
        "the shared POSIX install action no longer prefixes elapsed seconds. It is the one "
        "definition behind 40 jobs, so the breakdown disappears from all of them at once."
    )


def _code_only(run: str) -> str:
    """``run`` with whole-line ``#`` comments dropped.

    Load-bearing for the ordering checks below, not tidiness. These steps carry a comment
    block that explains the design by NAMING the stages -- "Tee-Object writes
    logs/install.log upstream of this filter" -- so an ordering check over the raw body
    finds `Tee-Object` in the prose long before the pipeline and reports correct order no
    matter how the pipeline is actually written. Verified: without this the pwsh
    reorder-mutation goes green.

    Whole-line comments only, which is what these bodies use; `#` inside the format
    strings would otherwise be at risk, and neither dialect needs one here.
    """
    return "\n".join(l for l in run.splitlines() if not l.lstrip().startswith("#"))


@pytest.mark.parametrize(
    "marker,writer",
    [(POSIX_FILTER, "tee "), (PWSH_FILTER, "Tee-Object")],
    ids = ["posix", "pwsh"],
)
def test_the_prefix_is_applied_after_the_log_is_written(marker, writer):
    """Reordering to `| prefix | tee` is a one-character-class edit and stays green.

    It would put the prefix into `logs/install.log`, which roughly 30 steps read. Most
    grep it for substrings and would survive, but `interrupted-install-ci.yml:185` matches
    `^\\[TAURI:STEP\\]` anchored at line start: every line would gain a `[  12s] ` prefix,
    the grep would match nothing, and the step asserts on what it found. That is a silent
    false pass in a workflow this PR does not otherwise touch.
    """
    for path, jid, name, body in _prefixing_bodies():
        run = _code_only(body)
        if marker not in run:
            continue
        assert writer in run, (
            f"{path.name}:{jid}:{name} prefixes elapsed seconds but never writes the "
            f"unprefixed stream to a log at all"
        )
        assert run.index(writer) < run.index(marker), (
            f"{path.name}:{jid}:{name} applies the elapsed prefix BEFORE {writer.strip()}, "
            f"so the prefix lands in the log artifact rather than only in the step log. "
            f"Roughly 30 steps read those logs, and interrupted-install-ci.yml anchors a "
            f"pattern at line start against one of them."
        )


def test_the_powershell_clock_is_started_before_it_is_read():
    """`$sw` is an ordinary variable, and PowerShell does not require it to exist.

    Without `Set-StrictMode` an undefined `$sw` is `$null`, so `$sw.Elapsed.TotalSeconds`
    yields nothing and `-f` renders an empty field. The step log then shows `[    s] ` on
    every line: no error, no failure, and a breakdown that reads as a formatting quirk
    rather than as a broken measurement. Deleting the declaration is exactly the kind of
    edit a later cleanup makes.
    """
    for path, jid, name, body in _prefixing_bodies():
        run = _code_only(body)
        if PWSH_FILTER not in run:
            continue
        assert "Stopwatch]::StartNew()" in run, (
            f"{path.name}:{jid}:{name} reads $sw.Elapsed without starting a Stopwatch, so "
            f"every elapsed field renders empty and the step still passes"
        )
        assert run.index("Stopwatch]::StartNew()") < run.index(PWSH_FILTER), (
            f"{path.name}:{jid}:{name} starts its Stopwatch after the pipeline that reads " f"it"
        )


def test_a_failing_install_still_fails_its_step():
    """Adding pipeline stages is exactly how a `tee` idiom loses its exit status."""
    for path, jid, name, run in _prefixing_bodies():
        if POSIX_FILTER in run:
            assert "set -o pipefail" in run, (
                f"{path.name}:{jid}:{name} pipes the installer through two stages without "
                f"pipefail, so the step reports the status of the prefix loop -- always 0 "
                f"-- and a failed install passes"
            )
        if PWSH_FILTER in run:
            # The comparison, not the bare variable name: `$child` already ends with `exit $LASTEXITCODE`, so a
            # substring test for the name alone stays green after the outer check is deleted.
            assert re.search(r"\$LASTEXITCODE\s+-ne\s+0", run), (
                f"{path.name}:{jid}:{name} no longer throws on a non-zero $LASTEXITCODE "
                f"after the pipeline. PowerShell does not fail a step for a native "
                f"command's exit code, so a failing install.ps1 leaves the step green."
            )


def test_the_posix_filter_does_not_swallow_the_last_line():
    """`while read` drops a final line with no trailing newline, and that is often the error.

    Cheap to get wrong, invisible when wrong: the install still fails on its exit status,
    but the message explaining why is the line that disappeared.
    """
    for path, jid, name, run in _prefixing_bodies():
        if POSIX_FILTER not in run:
            continue
        assert '|| [ -n "$line" ]' in run, (
            f"{path.name}:{jid}:{name} reads with a bare `while IFS= read -r line`, which "
            f"discards output that ends without a newline"
        )


# --------------------------------------------------------------------------------------
# Run the real filters, rather than only reading them
# --------------------------------------------------------------------------------------


def _posix_filter_body() -> str:
    """The POSIX pipeline as the composite action actually declares it.

    Extracted rather than restated so this exercises the shipped text: a copy in the test
    would keep passing after the action was broken.
    """
    run = next(r for p, _, _, r in _run_bodies() if p == ACTION and "install.sh" in r)
    return run


def _bash_runs_posix_scripts() -> bool:
    """Whether `bash` here is a real POSIX shell rather than Windows' WSL launcher.

    On a windows-latest runner `bash` resolves to the WSL stub, which ignores the script
    and exits 1 with a UTF-16 "no distributions installed" message. That is not a finding
    about the filter, so the executing tests skip there. Probed rather than keyed off
    sys.platform, so a Windows box with a working git-bash still runs them.
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
    """A skip condition that quietly became always-true would disable the tests below."""
    if sys.platform.startswith("win"):
        pytest.skip("Windows has no POSIX bash by default; that is the case being skipped")
    assert BASH_OK, (
        "the POSIX-bash probe failed on a platform that ships bash, so the tests that "
        "actually execute the shipped filter are being skipped everywhere"
    )


def _run_posix_filter(tmp_path, fake_installer: str):
    """Run the action's real pipeline with install.sh swapped for a fake, and report both.

    Returns (returncode, stdout, log_bytes). The fake writes a phase line, sleeps, writes
    a second, then a final line with no trailing newline.
    """
    body = _posix_filter_body()
    log = tmp_path / "install.log"
    script = body.replace("bash install.sh --local --no-torch", fake_installer)
    script = script.replace("logs/install.log", str(log))
    script = script.replace("mkdir -p logs", ":")
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        cwd = tmp_path,
        env = {**os.environ, "SECONDS": ""},
    )
    return proc.returncode, proc.stdout, (log.read_bytes() if log.exists() else None)


@pytest.mark.skipif(not BASH_OK, reason = "no POSIX bash here (Windows resolves it to WSL)")
def test_the_shipped_posix_filter_leaves_the_log_byte_identical(tmp_path):
    """The load-bearing claim of the whole design, executed rather than argued."""
    payload = 'printf "phase one\\nphase two\\nno trailing newline"'
    rc, stdout, log = _run_posix_filter(tmp_path, f"bash -c '{payload}'")
    assert rc == 0, stdout
    assert log == b"phase one\nphase two\nno trailing newline", (
        f"the artifact is not what the installer wrote: {log!r}. Every reader of "
        f"logs/install.log depends on this."
    )
    assert re.search(r"\[ *\d+s\] phase one", stdout), f"no elapsed prefix on stdout: {stdout!r}"
    assert (
        "no trailing newline" in stdout
    ), f"the final unterminated line never reached the step log: {stdout!r}"


@pytest.mark.skipif(not BASH_OK, reason = "no POSIX bash here (Windows resolves it to WSL)")
def test_the_shipped_posix_filter_propagates_a_failed_install(tmp_path):
    """Two extra pipeline stages between the installer and the step's status."""
    rc, stdout, _ = _run_posix_filter(tmp_path, "bash -c 'echo boom; exit 7'")
    assert rc == 7, (
        f"a failing install exited {rc} through the filter, not 7. The step would pass on "
        f"a broken install.\n{stdout}"
    )


@pytest.mark.skipif(not BASH_OK, reason = "no POSIX bash here (Windows resolves it to WSL)")
def test_the_elapsed_prefix_tracks_real_time_rather_than_printing_a_constant(tmp_path):
    """`[   0s]` on every line would look exactly like a working feature in a CI log."""
    rc, stdout, _ = _run_posix_filter(tmp_path, "bash -c 'echo first; sleep 2; echo second'")
    assert rc == 0, stdout
    seconds = [int(m) for m in re.findall(r"\[ *(\d+)s\]", stdout)]
    assert len(seconds) >= 2, f"expected a prefix per line, got {stdout!r}"
    assert seconds[-1] > seconds[0], (
        f"the elapsed prefix never advanced across a 2s gap ({seconds}), so it is not "
        f"measuring anything and the breakdown it exists to give is fiction"
    )


PWSH = None
for _candidate in ("pwsh", "powershell"):
    try:
        if (
            subprocess.run([_candidate, "-NoProfile", "-Command", "exit 0"], timeout = 60).returncode
            == 0
        ):
            PWSH = _candidate
            break
    except (OSError, subprocess.SubprocessError):
        continue


def _run_pwsh(script: str, attempts: int = 2):
    """Run `script` under pwsh, retrying only an interpreter crash.

    Delegates to the shared `run_pwsh`, which was generalised out of this function: it keeps
    the crash banner (an interpreter that dies mid-run and still exits normally, seen here on
    a hosted ubuntu runner with completely empty stdout) and adds the SIGABRT case this file
    never covered, where .NET failfasts at pwsh startup and the process is killed by a signal
    instead of printing anything at all.

    A crash yields no verdict either way, so retrying it is not papering over a failure:
    there is nothing to paper over yet. A run that reaches `RC=` is returned as-is on the
    first attempt, whatever the value, so a real regression is never retried into green.
    That is what `verdict` says here. `PwshInterpreterCrash` is an `AssertionError`, so an
    exhausted retry loop still surfaces as a failure naming the interpreter rather than
    accusing install.ps1 of losing $LASTEXITCODE through the pipeline.
    """
    return run_pwsh(
        [PWSH, "-NoProfile", "-Command", script],
        attempts = attempts,
        verdict = "RC=",
        capture_output = True,
        text = True,
    )


@pytest.mark.skipif(PWSH is None, reason = "no PowerShell on this platform")
def test_the_pwsh_filter_keeps_the_log_clean_and_the_exit_code_intact(tmp_path):
    """Same two claims for the Windows dialect, which is where the 291s actually is.

    `Tee-Object` and `ForEach-Object` sit between the native command and the
    `$LASTEXITCODE` check; that variable surviving two extra pipeline stages is an
    assumption worth executing rather than believing.
    """
    log = tmp_path / "install.log"
    script = textwrap.dedent(
        f"""
        $child = 'Write-Host "phase one"; Start-Sleep 2; Write-Host "phase two"; exit 7'
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        {PWSH} -NoProfile -Command $child 2>&1 |
          Tee-Object -FilePath '{log.as_posix()}' |
          ForEach-Object {{ '[{{0,4:N0}}s] {{1}}' -f $sw.Elapsed.TotalSeconds, $_ }}
        Write-Output "RC=$LASTEXITCODE"
        """
    )
    proc = _run_pwsh(script)
    assert "RC=7" in proc.stdout, (
        f"$LASTEXITCODE did not survive the added pipeline stages, so a failing "
        f"install.ps1 would leave its step green:\n{proc.stdout}\n{proc.stderr}"
    )
    contents = log.read_text(encoding = "utf-8")
    assert (
        "phase one" in contents and "s]" not in contents
    ), f"the elapsed prefix leaked into logs/install.log: {contents!r}"
    seconds = [int(m) for m in re.findall(r"\[ *(\d+)s\]", proc.stdout)]
    assert (
        seconds and seconds[-1] > seconds[0]
    ), f"the PowerShell prefix did not advance across a 2s gap ({seconds})"

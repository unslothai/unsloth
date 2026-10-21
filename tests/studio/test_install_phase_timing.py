# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The elapsed-time prefix is a display filter after the log write, so install.log is byte-identical."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from unsloth_pwsh_runner import pwsh_env, run_pwsh

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
ACTION = REPO / ".github" / "actions" / "install-unsloth-local" / "action.yml"

INSTALLERS = (
    REPO / "install.sh",
    REPO / "install.ps1",
    REPO / "studio" / "setup.sh",
    REPO / "studio" / "setup.ps1",
)

# Each filter paired with the log-writing stage that must precede it in the same pipeline.
POSIX_FILTER = "printf '[%4ds] %s\\n' \"$SECONDS\""
PWSH_FILTER = "$sw.Elapsed.TotalSeconds"


@pytest.mark.parametrize("script", INSTALLERS, ids = lambda p: p.name)
def test_the_installers_carry_no_timing_machinery(script):
    """Installers must not read UNSLOTH_INSTALL_TIMING or any timing switch; the clock lives in CI only."""
    src = script.read_text(encoding = "utf-8")
    assert "UNSLOTH_INSTALL_TIMING" not in src, (
        f"{script.name} interprets UNSLOTH_INSTALL_TIMING. The install timing is a CI-side "
        f"display filter over a stream that is already piped; putting it back inside the "
        f"installer re-adds a user-facing switch, a shell-specific truthiness rule and a "
        f"cross-process epoch handoff, for output CI can prefix for free."
    )


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
    """Run body minus whole-line comments, so ordering checks cannot match the prose naming the stages."""
    return "\n".join(l for l in run.splitlines() if not l.lstrip().startswith("#"))


@pytest.mark.parametrize(
    "marker,writer",
    [(POSIX_FILTER, "tee "), (PWSH_FILTER, "Tee-Object")],
    ids = ["posix", "pwsh"],
)
def test_the_prefix_is_applied_after_the_log_is_written(marker, writer):
    """The prefix must follow the log write, or the anchored TAURI grep in install.log breaks."""
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
    """Start the stopwatch before reading it; an unset $sw renders blank timings with no error."""
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
            # The comparison, not the name: `$child` already ends with `exit $LASTEXITCODE`.
            assert re.search(r"\$LASTEXITCODE\s+-ne\s+0", run), (
                f"{path.name}:{jid}:{name} no longer throws on a non-zero $LASTEXITCODE "
                f"after the pipeline. PowerShell does not fail a step for a native "
                f"command's exit code, so a failing install.ps1 leaves the step green."
            )


def test_the_posix_filter_does_not_swallow_the_last_line():
    """A while read loop drops a final line with no newline; the guard is || [ -n "$line" ]."""
    for path, jid, name, run in _prefixing_bodies():
        if POSIX_FILTER not in run:
            continue
        assert '|| [ -n "$line" ]' in run, (
            f"{path.name}:{jid}:{name} reads with a bare `while IFS= read -r line`, which "
            f"discards output that ends without a newline"
        )


def _posix_filter_body() -> str:
    """Returns the real POSIX pipeline from the composite action, so the test exercises shipped text."""
    run = next(r for p, _, _, r in _run_bodies() if p == ACTION and "install.sh" in r)
    return run


def _bash_runs_posix_scripts() -> bool:
    """True only for a real POSIX bash; the WSL launcher stub on windows-latest is not a finding."""
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
    """Runs the action's pipeline with a fake installer; returns returncode, stdout and the log bytes."""
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
        # pwsh_env, not run_pwsh: an import-time probe must answer "no", not raise.
        if (
            subprocess.run(
                [_candidate, "-NoProfile", "-Command", "exit 0"], timeout = 60, env = pwsh_env()
            ).returncode
            == 0
        ):
            PWSH = _candidate
            break
    except (OSError, subprocess.SubprocessError):
        continue


def _run_pwsh(script: str, attempts: int = 2):
    """Retries only interpreter crashes; any run that reaches RC= is returned on the first attempt."""
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

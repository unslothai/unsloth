# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The installers turn uv's large-download announcements into markers the app reads.

These drive the real run_install_cmd out of install.sh and the real Invoke-InstallCommand
out of install.ps1, against recorded uv output, with only their collaborators stubbed.
"""

from __future__ import annotations

import os
import re
import select
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason = "drives install.sh under /bin/sh")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTALL_SH = _REPO_ROOT / "install.sh"
_INSTALL_PS1 = _REPO_ROOT / "install.ps1"

_THRESHOLD = 52428800  # 50 MiB, the shipped default in both installers.

# Shaped like real `uv pip install` output, but the two nvidia sizes are picked to bracket
# _THRESHOLD rather than to match those wheels (both are in fact far larger), so moving the
# default in either direction changes what is marked. transformers lands unannounced.
_UV_OUTPUT = """Resolved 38 packages in 2.60s
Downloading torch (2.4GiB)
Downloading nvidia-cudnn-cu12 (674.0MiB)
Downloading nvidia-cusparse-cu12 (51.0MiB)
Downloading nvidia-cublas-cu12 (49.0MiB)
Downloading transformers (11.2MiB)
Downloading idna (68.0KiB)
 Downloaded transformers
 Downloaded nvidia-cublas-cu12
 Downloaded nvidia-cusparse-cu12
 Downloaded nvidia-cudnn-cu12
 Downloaded torch
Installed 38 packages in 9.20s
+ torch==2.10.0+cu126
"""

# uv does not repeat a completion, but a repeat must not close a download twice: the app
# would see an end for something it never saw start.
_UV_REPEATED_COMPLETION = """Downloading torch (2.4GiB)
 Downloaded torch
 Downloaded torch
"""

_MARKED = ["torch 2.4GiB", "nvidia-cudnn-cu12 674.0MiB", "nvidia-cusparse-cu12 51.0MiB"]
_LANDED = ["DONE nvidia-cusparse-cu12", "DONE nvidia-cudnn-cu12", "DONE torch"]

_ENV = {"PATH": "/usr/bin:/bin:/usr/local/bin"}


def _expected(short: list[str]) -> list[str]:
    return [
        f"[TAURI:DL_DONE] {s.removeprefix('DONE ')}" if s.startswith("DONE ") else f"[TAURI:DL] {s}"
        for s in short
    ]


def _extract(name: str) -> str:
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", source, re.MULTILINE | re.DOTALL)
    assert match, f"install.sh no longer defines {name}"
    return match.group(0)


# run_install_cmd's collaborators, stubbed so the real function can run here.
_STUBS = """
_is_verbose() { [ -n "${VERBOSE_MODE:-}" ]; }
step() { :; }; substep() { :; }; tauri_stream_log() { :; }; tauri_clear_install_error() { :; }
_redact_install_output() { cat "$@"; }
"""


def _extract_default() -> str:
    """install.sh's own threshold default, so changing the shipped value changes the test."""
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    match = re.search(r'^: "\$\{UNSLOTH_DL_MARKER_MIN_BYTES:=\d+\}"$', source, re.MULTILINE)
    assert match, "install.sh no longer defaults UNSLOTH_DL_MARKER_MIN_BYTES"
    return match.group(0)


def _sh_harness(child: str) -> str:
    return f"""
{_STUBS}
{_extract_default()}
{_extract("_uv_download_markers")}
{_extract("run_install_cmd")}
run_install_cmd "install PyTorch" sh -c '{child}'
printf 'RC=%s\\n' "$?"
"""


def _run(
    *,
    tauri: bool = True,
    exit_code: int = 0,
    min_bytes: str | None = None,
    verbose: bool = False,
    output: str = _UV_OUTPUT,
    path: str | None = None,
):
    """Drive the real run_install_cmd, with only its collaborators stubbed."""
    env = dict(_ENV, UV_OUTPUT = output)
    if path:
        env["PATH"] = path
    # Left unset by default so install.sh's own shipped threshold is what runs.
    if min_bytes:
        env["UNSLOTH_DL_MARKER_MIN_BYTES"] = min_bytes
    if tauri:
        env["TAURI_MODE"] = "true"
    if verbose:
        env["VERBOSE_MODE"] = "1"
    done = subprocess.run(
        ["/bin/sh", "-c", _sh_harness(f'printf "%s" "$UV_OUTPUT"; exit {exit_code}')],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
    )
    assert done.returncode == 0, done.stderr
    # Markers ride stderr so the verbose path's block-buffering redactor cannot delay them.
    markers = [line for line in done.stderr.splitlines() if line.startswith("[TAURI:")]
    rc = next(line for line in done.stdout.splitlines() if line.startswith("RC="))
    # Both streams: the quiet arm prints the log to stderr, the verbose arm pipes to stdout.
    shown = "\n".join(
        [l for l in done.stdout.splitlines() if not l.startswith("RC=")]
        + [l for l in done.stderr.splitlines() if not l.startswith("[TAURI:")]
    )
    return markers, rc.removeprefix("RC="), shown


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, _MARKED + _LANDED),
        ({"min_bytes": "1000000000"}, ["torch 2.4GiB", "DONE torch"]),
        # The verbose arm pipes output instead of logging it, and marks just the same.
        ({"verbose": True}, _MARKED + _LANDED),
        ({"tauri": False}, []),
        ({"output": _UV_REPEATED_COMPLETION}, ["torch 2.4GiB", "DONE torch"]),
    ],
)
def test_only_downloads_worth_waiting_for_become_markers(options, expected):
    markers, _, _ = _run(**options)
    assert markers == _expected(expected)


def test_a_failure_still_shows_the_childs_whole_output():
    # The added pipe must not cost the failure path the log it exists to print.
    _, rc, shown = _run(exit_code = 42)
    assert rc == "42"
    assert _UV_OUTPUT.strip() in shown


@pytest.mark.parametrize("exit_code", [0, 42])
def test_the_exit_code_survives_the_added_pipe(exit_code):
    _, rc, _ = _run(exit_code = exit_code)
    assert rc == str(exit_code)


@pytest.mark.parametrize("verbose", [False, True])
def test_a_host_without_awk_still_installs(verbose):
    # install.sh supports minimal images that ship no awk, and this pipe now carries every
    # install command, so losing awk must cost the markers and nothing else. Without the
    # fallback the pipeline closes and the child dies of SIGPIPE, reporting exit 141.
    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "bin"
        stub.mkdir()
        for tool in ("sh", "cat", "mktemp", "rm", "sed", "printf"):
            found = shutil.which(tool)
            if found:
                (stub / tool).symlink_to(found)
        assert shutil.which("awk", path = str(stub)) is None, "the sandbox must still have no awk"
        ok_markers, ok_rc, ok_shown = _run(path = str(stub), verbose = verbose)
        markers, rc, shown = _run(exit_code = 42, path = str(stub), verbose = verbose)
    assert (ok_rc, rc) == ("0", "42"), "a missing awk turned into SIGPIPE"
    assert markers == [] and ok_markers == [], "markers cannot be produced without awk"
    assert _UV_OUTPUT.strip() in shown, "the failure path lost the child's output"
    # The sink still has to differ by arm: quiet holds output in the log until something
    # fails, verbose passes it straight through.
    assert (_UV_OUTPUT.strip() in ok_shown) is verbose


def test_a_marker_arrives_while_its_download_is_still_running():
    # Every other test reads output after the child exits, so a buffered marker still
    # shows up -- just too late. Verbose is the arm with the block-buffering redactor.
    proc = subprocess.Popen(
        ["/bin/sh", "-c", _sh_harness('printf "Downloading torch (2.4GiB)\\n"; sleep 30')],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.PIPE,
        text = True,
        env = dict(_ENV, TAURI_MODE = "true", VERBOSE_MODE = "1"),
        start_new_session = True,
    )
    try:
        assert select.select([proc.stderr], [], [], 15)[0], "no marker while the download ran"
        assert proc.stderr.readline().startswith("[TAURI:DL] torch"), "not the marker"
        assert proc.poll() is None, "the download must still be running when it arrives"
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


_PWSH = shutil.which("pwsh") or shutil.which("powershell")


def _ps1_block(first: str, last: str) -> str:
    """install.ps1 source, from the definition of `first` through the end of `last`."""
    ps1 = _INSTALL_PS1.read_text(encoding = "utf-8")
    start = ps1.find(f"    function {first} {{")
    tail = ps1.find(f"    function {last} {{")
    assert start >= 0 and tail >= start, f"install.ps1 no longer defines {first}..{last}"
    return ps1[start : ps1.index("\n    }\n", tail) + len("\n    }\n")]


def _run_ps1(
    *,
    tauri: bool = True,
    min_bytes: str | None = None,
    verbose: bool = False,
    output: str = _UV_OUTPUT,
) -> list[str]:
    """Drive the real Invoke-InstallCommand, so both its arms are exercised as shipped."""
    markers = _ps1_block("Write-TauriLog", "Write-UvDownloadMarker")
    invoke = _ps1_block("Invoke-InstallCommand", "Invoke-InstallCommand")
    assert "Write-UvDownloadMarker" in invoke, "Invoke-InstallCommand no longer marks at all"
    lines = ", ".join("'{}'".format(line.replace("'", "''")) for line in output.splitlines())
    probe = (
        f"$TauriMode = ${str(tauri).lower()}\n"
        f"$script:UnslothVerbose = ${str(verbose).lower()}\n"
        "function Write-StudioLine { param([string]$Text, $ForegroundColor)"
        " [Console]::Out.WriteLine($Text) }\n"
        "function Redact-InstallOutput { param([string]$Text) $Text }\n"
        "function Clear-TauriInstallError { param([string]$Message) }\n"
        f"{markers}\n{invoke}\n"
        f"Invoke-InstallCommand -Command {{ @({lines}) | Write-Output }} -Label 'install PyTorch'\n"
    )
    # Not os.environ: an ambient UNSLOTH_DL_MARKER_MIN_BYTES would override the default.
    env = dict(_ENV)
    if min_bytes:
        env["UNSLOTH_DL_MARKER_MIN_BYTES"] = min_bytes
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "probe.ps1"
        script.write_text(probe, encoding = "utf-8")
        done = subprocess.run(
            [_PWSH, "-NoProfile", "-File", str(script)],
            capture_output = True,
            text = True,
            env = env,
            timeout = 120,
        )
    assert done.returncode == 0, done.stderr
    return [line for line in done.stdout.splitlines() if line.startswith("[TAURI:DL")]


@pytest.mark.skipif(_PWSH is None, reason = "needs PowerShell to run install.ps1")
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"verbose": True},
        {"min_bytes": "1000000000"},
        {"tauri": False},
        {"output": _UV_REPEATED_COMPLETION},
    ],
)
def test_both_installers_emit_the_same_markers(options):
    # The sh expectations are pinned above, so agreeing with sh pins PowerShell too.
    expected, _, _ = _run(**options)
    assert _run_ps1(**options) == expected


def test_both_installers_ship_the_same_threshold():
    # Execution only brackets the default between two fixture sizes; the literal pins it
    # exactly, and pins the two installers to each other so they cannot drift apart.
    assert _extract_default() == f': "${{UNSLOTH_DL_MARKER_MIN_BYTES:={_THRESHOLD}}}"'
    ps1 = _INSTALL_PS1.read_text(encoding = "utf-8")
    assert f"$script:UvDownloadMarkerMinBytes = {_THRESHOLD}" in ps1

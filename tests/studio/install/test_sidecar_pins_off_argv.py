# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sidecar pins never reach a child's argv: macOS XProtect checks the argv of a pasted install with
Safe Browsing, reads transformers==5.13.1 as the listed IP 5.13.0.1 and kills the session (#18699)."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
SHIM = REPO_ROOT / "studio" / "install_manifest.py"
PINS = ("transformers==5.13.1", "huggingface_hub==1.8.0", "hf_xet==1.4.2")


def _function(name: str) -> str:
    match = re.search(
        rf"^{name}\(\) \{{\n.*?^\}}\n", SETUP_SH.read_text(encoding = "utf-8"), re.S | re.M
    )
    assert match is not None, name
    return match.group()


def _bash(script: str, tmp_path: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, cwd = tmp_path, timeout = 60
    )


def test_the_install_passes_the_pin_in_a_requirements_file(tmp_path: pathlib.Path) -> None:
    script = (
        'fast_install_sidecar() { printf "argv=%s\\n" "$*"; '
        '[ "$4" = -r ] && sed "s/^/file=/" "$5"; return 3; }\n'
        f"{_function('_sidecar_install_pin')}"
        '_sidecar_install_pin /t "transformers==5.13.1"; echo "rc=$?"\n'
    )
    result = _bash(script, tmp_path)
    lines = result.stdout.splitlines()
    assert lines[0].startswith("argv=--target /t --no-deps -r ")
    assert "5.13.1" not in lines[0]
    assert lines[1:] == ["file=transformers==5.13.1", "rc=3"]
    # The requirements directory is removed whatever the install returned.
    assert not pathlib.Path(lines[0].split()[-1]).exists()


def test_the_audit_sends_the_pins_on_stdin(tmp_path: pathlib.Path) -> None:
    recorder = tmp_path / "python"
    recorder.write_text(
        '#!/bin/sh\nprintf "argv=%s\\n" "$*" >> "$LOG"\nsed "s/^/stdin=/" >> "$LOG"\n'
        'echo "sidecar: current"\n'
    )
    recorder.chmod(0o755)
    (tmp_path / "install_manifest.py").write_text("")
    (tmp_path / "side").mkdir()
    log = tmp_path / "log"
    script = (
        f'SCRIPT_DIR="{tmp_path}"; VENV_DIR="{tmp_path}/venv"; mkdir -p "$VENV_DIR/bin"\n'
        f'cp "{recorder}" "$VENV_DIR/bin/python"\n'
        '_SIDECAR_COMMON_PINS="huggingface_hub==1.8.0 hf_xet==1.4.2"\n'
        "verbose_substep() { :; }\n"
        "_target_has_pkg_version() { return 9; }\n"
        f"{_function('_sidecar_current')}"
        f'LOG="{log}" _sidecar_current "{tmp_path}/side" 5.13.1; echo "rc=$?"\n'
    )
    result = _bash(script, tmp_path)
    assert result.stdout.strip() == "rc=0", result.stderr
    lines = log.read_text().splitlines()
    assert lines[0].startswith("argv=") and lines[0].endswith(f"sidecar {tmp_path}/side -")
    assert "==" not in lines[0]
    assert [line.removeprefix("stdin=") for line in lines[1:]] == list(PINS)


def test_the_shim_reads_pins_from_stdin(tmp_path: pathlib.Path) -> None:
    run = lambda *args, stdin = "": subprocess.run(
        [sys.executable, str(SHIM), "sidecar", str(tmp_path / "gone"), *args],
        input = stdin,
        capture_output = True,
        text = True,
    )
    from_stdin = run("-", stdin = "\n".join(PINS) + "\n")
    from_argv = run(*PINS)
    assert (from_stdin.returncode, from_stdin.stdout) == (from_argv.returncode, from_argv.stdout)
    assert from_stdin.stdout.startswith("sidecar: ")
    # No pins at all is still a usage error, not "current".
    assert run("-").returncode == 2

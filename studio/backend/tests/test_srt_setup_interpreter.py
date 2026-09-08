# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Execute setup's real interpreter-selection/helper block with benign helpers."""

import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(os.name == "nt", reason = "POSIX setup block; run on Linux/macOS")
@pytest.mark.parametrize("route", ["standalone", "staged", "main", "missing", "colab"])
def test_setup_srt_uses_selected_python(tmp_path, route):
    studio = Path(__file__).resolve().parents[2]
    source = (studio / "setup.sh").read_text(encoding = "utf-8")
    start = source.index("_COLAB_NO_VENV=false")
    end = source.index("install_python_stack()", start)
    block = source[start:end]
    # The pre-fix helper precedes selection; retain that ordering in the witness.
    helper_start = source.index('if ! python "$SCRIPT_DIR/install_srt_runtime.py"; then')
    if helper_start < start:
        block = source[helper_start:source.index("\nfi", helper_start) + 3] + "\n" + block
    venv = tmp_path / "selected venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents = True)
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    log = tmp_path / "calls"
    selected = fallback if route == "colab" else bin_dir
    if route != "missing":
        python = selected / "python"
        python.write_text('#!/bin/sh\nprintf "%s\\n" "$0" "$@" >> "$CALL_LOG"\n')
        python.chmod(0o755)
    (bin_dir / "activate").write_text('export PATH="$VENV_DIR/bin:$PATH"\n')
    script_dir = tmp_path / "studio files"
    script_dir.mkdir()
    (script_dir / "backend/requirements").mkdir(parents = True)
    (script_dir / "backend/requirements/studio.txt").write_text("fixture-package>=1\n")
    env = {**os.environ, "VENV_DIR": str(venv), "SCRIPT_DIR": str(script_dir),
           "CALL_LOG": str(log), "STAGE_ROOT": "stage" if route == "staged" else "",
           "IS_COLAB": "true" if route == "colab" else "false",
           "PATH": str(selected if route in ("main", "colab") else fallback)}
    # Colab's no-venv branch uses ordinary POSIX utilities; no pip call is needed.
    if route == "colab":
        env["PATH"] += ":/usr/bin:/bin"
    result = subprocess.run(
        ["/bin/bash", "-c", 'set -e\nstep() { :; }; substep() { :; }; run_quiet_no_exit() { :; }; setup_fail() { exit "$1"; };\n' + block],
        env = env, text = True, capture_output = True, timeout = 10,
    )
    if route == "missing":
        assert result.returncode == 1
        assert not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert log.read_text().splitlines() == [str(selected / "python"), str(script_dir / "install_srt_runtime.py")]

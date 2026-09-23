# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""JupyterLab must not be started on Studio's port.

Studio's port is 8000 unless `UNSLOTH_STUDIO_PORT` says otherwise, so with `JUPYTER_PORT=8000`
JupyterLab wins the bind and Studio falls back to an unpublished 8001. Both report
RUNNING and the summary still points at 8000, where Jupyter answers 404.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "docker" / "studio_launch.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(port: str) -> subprocess.CompletedProcess:
    # check-only: past the settings checks the launcher writes /etc/profile.d,
    # /root/.jupyter and /workspace, which on a root test host it really would
    env = dict(os.environ, JUPYTER_PORT=port, UNSLOTH_STUDIO_LAUNCH_CHECK_ONLY="1")
    return subprocess.run(
        ["bash", str(LAUNCH)], capture_output=True, text=True, env=env, timeout=120
    )


def test_jupyter_on_studios_port_is_refused_with_a_remedy():
    res = _run("8000")
    assert res.returncode != 0, "the container started with Studio unreachable"
    assert "JUPYTER_PORT=8000" in res.stderr, res.stderr
    assert "-p 9000:8888" in res.stderr, "the remedy must be printed:\n" + res.stderr


@pytest.mark.parametrize("spelling", ["08000", " 8000", "8000 ", "+8000", "8_000"])
def test_other_spellings_of_8000_are_refused_too(spelling: str):
    """Jupyter's port is a traitlets Integer, read with int(): whitespace, leading zeros,
    a leading + and digit-group underscores all give 8000 as well."""
    res = _run(spelling)
    assert res.returncode != 0, spelling
    assert "JUPYTER_PORT=8000" in res.stderr, res.stderr


@pytest.mark.parametrize("port", ["8899", "8001", "8000.0", "0x1f40", "8000/tcp", "-8000"])
def test_other_ports_and_values_jupyter_rejects_itself_pass_the_guard(port: str):
    """Only what int() reads as 8000 is ours to refuse; "8000.0" or "0x1f40" fail in
    Jupyter with its own message."""
    res = _run(port)
    assert res.returncode == 0, res.stderr
    assert "JUPYTER_PORT=8000" not in res.stderr, res.stderr


@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: -1)() == 0,
    reason="as root the launcher would write to /etc and /root",
)
def test_check_only_set_to_zero_does_not_stop_the_launcher():
    """`=0` must mean off: the launcher goes on past the guard. On a non-root test host
    the next step, writing /etc/profile.d, fails, which is the proof that it went on."""
    env = dict(os.environ, JUPYTER_PORT="8899", UNSLOTH_STUDIO_LAUNCH_CHECK_ONLY="0")
    res = subprocess.run(
        ["bash", str(LAUNCH)], capture_output=True, text=True, env=env, timeout=120
    )
    assert res.returncode != 0, "check-only=0 exited 0 before touching anything"
    assert "/etc/profile.d/unsloth_env.sh" in res.stderr, res.stderr


def test_the_check_only_exit_comes_after_the_guard():
    """The guard is the point; check-only must not skip it, and nothing before the
    check-only exit may touch the host."""
    body = LAUNCH.read_text(encoding="utf-8")
    guard = body.index("jupyter_port_digits == UNSLOTH_STUDIO_PORT")
    check = body.index("UNSLOTH_STUDIO_LAUNCH_CHECK_ONLY:-")
    assert guard < check
    assert "> /etc/profile.d/unsloth_env.sh" not in body[:check]
    assert "> /etc/profile.d/unsloth_env.sh" in body[check:]

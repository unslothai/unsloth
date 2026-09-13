# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""JupyterLab must not be started on Studio's port.

Studio's port is fixed at 8000 (`studio_run.sh`), so with `JUPYTER_PORT=8000`
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

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


def _run(port: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, JUPYTER_PORT = port)
    return subprocess.run(
        ["bash", str(LAUNCH)], capture_output = True, text = True, env = env, timeout = 120
    )


def test_jupyter_on_studios_port_is_refused_with_a_remedy():
    res = _run("8000")
    assert res.returncode != 0, "the container started with Studio unreachable"
    assert "JUPYTER_PORT=8000" in res.stderr, res.stderr
    assert "-p 9000:8888" in res.stderr, "the remedy must be printed:\n" + res.stderr


def test_another_port_is_not_refused_here():
    res = _run("8899")
    assert "JUPYTER_PORT=8000" not in res.stderr, res.stderr

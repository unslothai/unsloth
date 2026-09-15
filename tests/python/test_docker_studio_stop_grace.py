# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""docker stop must leave a training run the time to save its checkpoint.

Studio stops the run cooperatively on SIGTERM and waits up to
UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S for the save. supervisord and docker stop both
default to a 10 second budget, so the image, the launcher and run.sh carry a matching one.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"
LAUNCH = DOCKER / "studio_launch.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


def _program(name: str) -> str:
    conf = (DOCKER / "supervisord.conf").read_text(encoding = "utf-8")
    match = re.search(rf"^\[program:{name}\]\n(.*?)(?=^\[|\Z)", conf, re.M | re.S)
    assert match, name
    return match.group(1)


def _launch(**env: str) -> subprocess.CompletedProcess:
    e = dict(os.environ, UNSLOTH_STUDIO_LAUNCH_CHECK_ONLY = "1", JUPYTER_PORT = "8888", **env)
    return subprocess.run(["bash", str(LAUNCH)], capture_output = True, text = True, env = e, timeout = 120)


def test_supervisord_waits_for_the_save_and_then_kills_the_whole_tree():
    studio = _program("studio")
    assert "stopwaitsecs=%(ENV_UNSLOTH_STUDIO_STOP_WAIT_S)s" in studio
    assert "killasgroup=true" in studio
    # a raw SIGTERM to the group would kill the worker before Studio can ask it to save
    assert "stopasgroup" not in studio


def test_the_image_defaults_resolve_the_placeholder_without_the_launcher():
    dockerfile = (DOCKER / "Dockerfile.studio").read_text(encoding = "utf-8")
    assert "UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S=120" in dockerfile
    assert "UNSLOTH_STUDIO_STOP_WAIT_S=150" in dockerfile


def test_the_launcher_derives_supervisords_wait_from_the_budget():
    body = LAUNCH.read_text(encoding = "utf-8")
    assert (
        "UNSLOTH_STUDIO_STOP_WAIT_S=$(( 10#$UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S + 30 ))" in body
    )
    assert body.index("UNSLOTH_STUDIO_STOP_WAIT_S=") < body.index(
        "UNSLOTH_STUDIO_LAUNCH_CHECK_ONLY:-"
    )


@pytest.mark.parametrize("value", ["soon", "-1", "1.5", ""])
def test_a_budget_that_is_not_a_number_of_seconds_is_refused(value: str):
    res = _launch(UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S = value)
    if value == "":
        assert res.returncode == 0, res.stderr
        return
    assert res.returncode != 0
    assert f"UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S={value}" in res.stderr, res.stderr


@pytest.mark.parametrize("value", ["0", "120", "0600"])
def test_a_number_of_seconds_passes(value: str):
    res = _launch(UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S = value)
    assert res.returncode == 0, res.stderr


def test_run_sh_gives_docker_stop_the_same_budget():
    body = (DOCKER / "run.sh").read_text(encoding = "utf-8")
    assert "STOP_TIMEOUT=$(( ${UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S:-120} + 30 ))" in body
    assert '--stop-timeout "$STOP_TIMEOUT"' in body
    assert "ENV_FORWARD+=(-e UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S)" in body


def test_the_server_asks_for_a_save_before_it_kills_the_worker():
    body = (REPO_ROOT / "studio" / "backend" / "run.py").read_text(encoding = "utf-8")
    block = body[body.index("from core.training.training import _training_backend") :]
    assert block.index("stop_for_shutdown()") < block.index("force_terminate()")

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


# Sourced under an EXIT trap so the wait the launcher exports for supervisord can be read
# back; the check-only exit still runs the trap, and nothing is written to the host.
_PROBE = r"""trap 'printf "WAIT=%s\n" "${UNSLOTH_STUDIO_STOP_WAIT_S-unset}"' EXIT; . "$1" """


def _launch(**env: str) -> subprocess.CompletedProcess:
    e = dict(os.environ, UNSLOTH_STUDIO_LAUNCH_CHECK_ONLY = "1", JUPYTER_PORT = "8888", **env)
    return subprocess.run(
        ["bash", "-c", _PROBE, "_", str(LAUNCH)],
        capture_output = True,
        text = True,
        env = e,
        timeout = 120,
    )


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


# "0600" and "08" would read as octal, and an unset budget must still produce a wait.
@pytest.mark.parametrize(
    "value, expected", [(None, "150"), ("0", "30"), ("120", "150"), ("0600", "630"), ("08", "38")]
)
def test_the_launcher_derives_supervisords_wait_from_the_budget(value, expected):
    res = _launch(**({} if value is None else {"UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S": value}))
    assert res.returncode == 0, res.stderr
    assert f"WAIT={expected}" in res.stdout, res.stdout


@pytest.mark.parametrize("value", ["soon", "-1", "1.5"])
def test_a_budget_that_is_not_a_number_of_seconds_is_refused(value: str):
    res = _launch(UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S = value)
    assert res.returncode != 0
    assert f"UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S={value}" in res.stderr, res.stderr
    assert "WAIT=unset" in res.stdout, res.stdout


def test_an_empty_budget_falls_back_to_the_default():
    res = _launch(UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S = "")
    assert res.returncode == 0, res.stderr
    assert "WAIT=150" in res.stdout, res.stdout


def _run_sh(
    tmp_path: Path, budget: "str | None", **extra: str
) -> "tuple[subprocess.CompletedProcess, list[str]]":
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv = tmp_path / "argv"
    stubs = {
        "docker": 'if [ "$1" = "info" ]; then echo " Runtimes: runc"; exit 0; fi\n'
        f'printf "%s\\n" "$@" > {argv}\nexit 0\n',
        "nvidia-smi": "exit 1\n",
    }
    for name, body in stubs.items():
        (bindir / name).write_text("#!/usr/bin/env bash\n" + body)
        (bindir / name).chmod(0o755)
    (tmp_path / "root" / "dev").mkdir(parents = True)
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("UNSLOTH_", "HF_TOKEN", "WANDB_"))
    }
    env.update(
        PATH = f"{bindir}:/usr/bin:/bin",
        HOME = str(tmp_path / "home"),
        UNSLOTH_DEV_ROOT = str(tmp_path / "root"),
        UNSLOTH_WORKDIR = str(tmp_path),
        UNSLOTH_STUDIO_VOLUME = "",
    )
    if budget is not None:
        env["UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S"] = budget
    env.update(extra)
    res = subprocess.run(
        [shutil.which("bash"), str(DOCKER / "run.sh"), "true"],
        capture_output = True,
        text = True,
        env = env,
        timeout = 120,
    )
    return res, argv.read_text().splitlines() if argv.exists() else []


_posix_only = pytest.mark.skipif(os.name != "posix", reason = "run.sh needs a POSIX shell")


@_posix_only
@pytest.mark.parametrize(
    "budget, expected",
    [(None, "150"), ("120", "150"), ("0", "30"), ("0600", "630"), ("08", "38")],
)
def test_run_sh_gives_docker_stop_the_launchers_budget(tmp_path, budget, expected):
    res, argv = _run_sh(tmp_path, budget)
    assert res.returncode == 0, res.stderr
    assert argv[argv.index("--stop-timeout") + 1] == expected


@_posix_only
@pytest.mark.parametrize("budget", ["soon", "-1", "1.5"])
def test_run_sh_refuses_a_budget_that_is_not_a_number_of_seconds(tmp_path, budget):
    res, argv = _run_sh(tmp_path, budget)
    assert res.returncode != 0
    assert f"UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S={budget}" in res.stderr, res.stderr
    assert argv == []


# The watchdog cap travels too: a budget raised past it alone waits on a save it kills.
_BUDGET_VARS = ("UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S", "UNSLOTH_STUDIO_TRAINING_STOP_TIMEOUT_S")


@_posix_only
def test_run_sh_forwards_both_budgets_into_the_container(tmp_path):
    res, argv = _run_sh(tmp_path, "900", UNSLOTH_STUDIO_TRAINING_STOP_TIMEOUT_S = "900")
    assert res.returncode == 0, res.stderr
    for name in _BUDGET_VARS:
        assert argv[argv.index(name) - 1] == "-e", name


@_posix_only
def test_run_sh_forwards_nothing_the_host_did_not_set(tmp_path):
    res, argv = _run_sh(tmp_path, None)
    assert res.returncode == 0, res.stderr
    for name in _BUDGET_VARS:
        assert name not in argv

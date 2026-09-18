# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""UNSLOTH_STUDIO_PORT moves Studio off 8000 inside the container.

Every place that named the port (the CLI launch, the JupyterLab port guard, the ready
summary and its health probe, the in-place updater's health wait) reads the variable,
and the image sets the default so a bare docker run keeps 8000.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"
LAUNCH = DOCKER / "studio_launch.sh"
RUN = DOCKER / "studio_run.sh"
PASSWORD = DOCKER / "studio_password.sh"
UPDATE = DOCKER / "unsloth_studio_update.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


def _stub(bin_dir: Path, name: str, body: str) -> None:
    bin_dir.mkdir(parents = True, exist_ok = True)
    path = bin_dir / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding = "utf-8")
    path.chmod(0o755)


def _clean_env(bin_dir: Path) -> dict:
    e = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_STUDIO")}
    e["PATH"] = f"{bin_dir}{os.pathsep}" + e["PATH"]
    return e


def _launch(**env: str) -> subprocess.CompletedProcess:
    e = dict(_clean_env(Path("/nonexistent")), UNSLOTH_STUDIO_LAUNCH_CHECK_ONLY = "1", **env)
    return subprocess.run(["bash", str(LAUNCH)], capture_output = True, text = True, env = e, timeout = 120)


def test_the_default_stays_8000():
    res = _launch(JUPYTER_PORT = "8000")
    assert res.returncode != 0
    assert "JUPYTER_PORT=8000" in res.stderr, res.stderr
    assert _launch(JUPYTER_PORT = "8888").returncode == 0


def test_the_jupyter_guard_follows_the_studio_port():
    res = _launch(UNSLOTH_STUDIO_PORT = "9000", JUPYTER_PORT = "9000")
    assert res.returncode != 0
    assert "JUPYTER_PORT=9000" in res.stderr, res.stderr
    assert "-p 9000:8888" in res.stderr
    assert _launch(UNSLOTH_STUDIO_PORT = "9000", JUPYTER_PORT = "8000").returncode == 0


@pytest.mark.parametrize("spelling", ["09000", " 9000", "9_000", "+9000"])
def test_jupyter_spellings_of_the_studio_port_are_still_caught(spelling: str):
    res = _launch(UNSLOTH_STUDIO_PORT = "9000", JUPYTER_PORT = spelling)
    assert res.returncode != 0, spelling
    assert "JUPYTER_PORT=9000" in res.stderr, res.stderr


def test_leading_zeros_on_the_studio_port_normalize():
    res = _launch(UNSLOTH_STUDIO_PORT = "09000", JUPYTER_PORT = "9000")
    assert res.returncode != 0
    assert "JUPYTER_PORT=9000" in res.stderr, res.stderr


@pytest.mark.parametrize(
    "value", ["abc", "0", "65536", "8000.0", "-1", "80 00", "18446744073709551617"]
)
def test_a_studio_port_that_is_not_a_port_is_refused(value: str):
    res = _launch(UNSLOTH_STUDIO_PORT = value, JUPYTER_PORT = "8888")
    assert res.returncode != 0, value
    assert f"UNSLOTH_STUDIO_PORT={value}" in res.stderr, res.stderr


def test_many_leading_zeros_still_normalize():
    res = _launch(UNSLOTH_STUDIO_PORT = "000000000000000000009000", JUPYTER_PORT = "9000")
    assert "JUPYTER_PORT=9000" in res.stderr, res.stderr


@pytest.mark.parametrize("key_var", ["SSH_KEY", "PUBLIC_KEY"])
def test_studio_on_sshd_port_is_refused_when_ssh_is_enabled(key_var: str):
    res = _launch(
        UNSLOTH_STUDIO_PORT = "22", JUPYTER_PORT = "8888", **{key_var: "ssh-ed25519 AAAA test"}
    )
    assert res.returncode != 0
    assert "UNSLOTH_STUDIO_PORT=22 is sshd's port" in res.stderr, res.stderr
    assert _launch(UNSLOTH_STUDIO_PORT = "22", JUPYTER_PORT = "8888").returncode == 0


def test_studio_run_hands_the_port_to_the_cli(tmp_path: Path):
    _stub(tmp_path / "bin", "unsloth", 'printf "%s\\n" "$*"\n')
    e = _clean_env(tmp_path / "stub-bin")
    e["UNSLOTH_STUDIO_HOME"] = str(tmp_path)
    e["UNSLOTH_STUDIO_INITIAL_PASSWORD_FILE"] = str(tmp_path / "initial")
    res = subprocess.run(["bash", str(RUN)], capture_output = True, text = True, env = e, timeout = 60)
    assert res.stdout.strip() == "studio -H 0.0.0.0 -p 8000", res.stdout + res.stderr
    e["UNSLOTH_STUDIO_PORT"] = "9000"
    res = subprocess.run(["bash", str(RUN)], capture_output = True, text = True, env = e, timeout = 60)
    assert res.stdout.strip() == "studio -H 0.0.0.0 -p 9000", res.stdout + res.stderr


def test_the_ready_summary_and_its_health_probe_use_the_port(tmp_path: Path):
    bin_dir = tmp_path / "stub-bin"
    log = tmp_path / "curl.log"
    _stub(bin_dir, "curl", f'echo "$*" >> "{log}"\nexit 0\n')
    _stub(bin_dir, "unsloth-studio-run", "exit 0\n")
    e = _clean_env(bin_dir)
    e.update(
        UNSLOTH_STUDIO_HOME = str(tmp_path),
        UNSLOTH_STUDIO_PASSWORD_STATE = "stored",
        UNSLOTH_STUDIO_PASSWORD_WAIT = "3",
        UNSLOTH_STUDIO_READY_WAIT = "2",
        UNSLOTH_STUDIO_PORT = "9000",
        JUPYTER_PORT = "8888",
        NO_COLOR = "1",
    )
    res = subprocess.run(["bash", str(PASSWORD)], capture_output = True, text = True, env = e, timeout = 60)
    assert res.returncode == 0, res.stderr
    assert "Unsloth     http://localhost:9000" in res.stdout, res.stdout
    assert "http://127.0.0.1:9000/api/health" in log.read_text(encoding = "utf-8")


def test_the_updater_waits_on_the_port():
    body = UPDATE.read_text(encoding = "utf-8")
    assert '_port="${UNSLOTH_STUDIO_PORT:-8000}"' in body
    assert '"http://127.0.0.1:${_port}/api/health"' in body
    assert "answering on port ${_port}" in body


def test_the_image_default_and_the_host_wrapper_carry_the_variable():
    assert "UNSLOTH_STUDIO_PORT=8000" in (DOCKER / "Dockerfile.studio").read_text(encoding = "utf-8")
    assert "ENV_FORWARD+=(-e UNSLOTH_STUDIO_PORT)" in (DOCKER / "run.sh").read_text(
        encoding = "utf-8"
    )
    assert "`UNSLOTH_STUDIO_PORT`" in (DOCKER / "DOCKERHUB.md").read_text(encoding = "utf-8")


@pytest.mark.parametrize("script", [LAUNCH, RUN, PASSWORD, UPDATE])
def test_no_script_names_the_port_literally_any_more(script: Path):
    lines = script.read_text(encoding = "utf-8").splitlines()
    body = "\n".join(l for l in lines if not l.lstrip().startswith("#"))
    for literal in ("127.0.0.1:8000", "localhost:8000", "-p 8000", "port 8000"):
        assert literal not in body, f"{script.name} still hardcodes {literal}"

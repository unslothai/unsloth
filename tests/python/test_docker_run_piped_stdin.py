# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

RUN_SH = Path(__file__).resolve().parents[2] / "docker" / "run.sh"

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason = "POSIX shell required",
)


def _docker_flags(
    tmp_path,
    stdin,
    stdout = subprocess.DEVNULL,
    close_stdin = False,
):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok = True)
    argv_log = tmp_path / "argv"
    docker = bindir / "docker"
    docker.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{argv_log}"\n', encoding = "utf-8"
    )
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    env = dict(
        os.environ,
        PATH = f"{bindir}:/usr/bin:/bin",
        HOME = str(tmp_path / "home"),
        UNSLOTH_WORKDIR = str(tmp_path),
        UNSLOTH_GPUS = "none",
    )
    proc = subprocess.run(
        [shutil.which("bash"), "-c", f'exec bash "{RUN_SH}" python <&-']
        if close_stdin
        else [shutil.which("bash"), str(RUN_SH), "python"],
        cwd = tmp_path,
        env = env,
        stdin = stdin,
        stdout = stdout,
        stderr = subprocess.PIPE,
        timeout = 60,
    )
    assert proc.returncode == 0, proc.stderr
    argv = argv_log.read_text(encoding = "utf-8").splitlines()
    return argv[: argv.index("--ipc=host")]


def test_piped_stdin_reaches_the_container(tmp_path):
    r, w = os.pipe()
    os.write(w, b'print("hi")\n')
    os.close(w)
    try:
        assert _docker_flags(tmp_path, r) == ["run", "--rm", "-i"]
    finally:
        os.close(r)


def test_redirected_file_reaches_the_container(tmp_path):
    script = tmp_path / "s.py"
    script.write_text('print("hi")\n', encoding = "utf-8")
    with script.open("rb") as f:
        assert _docker_flags(tmp_path, f) == ["run", "--rm", "-i"]


@pytest.mark.skipif(not hasattr(os, "openpty"), reason = "pty required")
def test_terminal_stdin(tmp_path):
    main, tty = os.openpty()
    try:
        assert _docker_flags(tmp_path, tty, tty) == ["run", "--rm", "-it"]
        # stdout redirected, e.g. `run.sh > log &`: -i would read the terminal and stop on SIGTTIN
        assert _docker_flags(tmp_path, tty) == ["run", "--rm"]
    finally:
        os.close(main)
        os.close(tty)


def test_dev_null_stdin(tmp_path):
    with open(os.devnull, "rb") as f:
        assert _docker_flags(tmp_path, f) == ["run", "--rm", "-i"]


def test_closed_stdin(tmp_path):
    assert _docker_flags(tmp_path, None, close_stdin = True) == ["run", "--rm"]

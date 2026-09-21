# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[2] / "docker"
RUN_SH = DOCKER / "run.sh"

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason = "POSIX shell required",
)


def _documented(page, heading):
    table = (DOCKER / page).read_text().split(f"\n{heading}\n", 1)[1]
    names = []
    for row in table.split("\n## ", 1)[0].splitlines():
        if row.startswith("| `"):
            names += re.findall(r"`([A-Z][A-Z0-9_]+)(?:=[^`]*)?`", row.split("|")[1])
    return names


ROCM_DOCUMENTED = _documented("DOCKERHUB-ROCM.md", "## Environment")
DOCUMENTED = list(
    dict.fromkeys(_documented("DOCKERHUB.md", "## Environment variables") + ROCM_DOCUMENTED)
)


def _forwarded(tmp_path, **env_extra):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "argv"
    docker = bindir / "docker"
    docker.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > {argv_log}\n')
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    env = {k: v for k, v in os.environ.items() if k not in DOCUMENTED}
    env.update(
        PATH = f"{bindir}:/usr/bin:/bin",
        HOME = str(tmp_path / "home"),
        UNSLOTH_WORKDIR = str(tmp_path),
        UNSLOTH_GPUS = "none",
        **env_extra,
    )
    proc = subprocess.run(
        [shutil.which("bash"), str(RUN_SH), "true"],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert proc.returncode == 0, proc.stderr
    argv = argv_log.read_text().splitlines()
    return [spec for flag, spec in zip(argv, argv[1:]) if flag == "-e"]


def test_the_hub_pages_still_have_a_variable_table():
    assert "JUPYTER_PORT" in DOCUMENTED and len(DOCUMENTED) >= 10, DOCUMENTED
    assert "UNSLOTH_SKIP_GPU_CHECK" in ROCM_DOCUMENTED, ROCM_DOCUMENTED


@pytest.mark.parametrize("name", DOCUMENTED)
def test_every_documented_variable_reaches_the_container(tmp_path, name):
    assert name in _forwarded(tmp_path, **{name: "1"})


@pytest.mark.parametrize("name", DOCUMENTED)
def test_an_unset_variable_is_not_forwarded(tmp_path, name):
    assert name not in _forwarded(tmp_path)

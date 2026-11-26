# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

RUN_SH = Path(__file__).resolve().parents[2] / "docker" / "run.sh"
HF_VARS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME")

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason = "POSIX shell required",
)


def _mounts(tmp_path, **env_extra):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "argv"
    docker = bindir / "docker"
    docker.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{argv_log}"\n', encoding = "utf-8"
    )
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    env = {k: v for k, v in os.environ.items() if k not in HF_VARS}
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
    argv = argv_log.read_text(encoding = "utf-8").splitlines()
    mounts = {}
    for flag, spec in zip(argv, argv[1:]):
        if flag == "-v" and ":/workspace/.cache/huggingface" in spec:
            host, target = spec.rsplit(":", 1)
            mounts[target] = host
    return mounts


def test_default_cache_is_mounted_alone(tmp_path):
    assert _mounts(tmp_path) == {
        "/workspace/.cache/huggingface": str(tmp_path / "home" / ".cache" / "huggingface")
    }
    # the container creates hub itself, group root writable for Studio's sandbox
    assert not (tmp_path / "home" / ".cache" / "huggingface" / "hub").exists()


def test_hf_home_is_mounted_alone(tmp_path):
    hf_home = tmp_path / "hf"
    assert _mounts(tmp_path, HF_HOME = str(hf_home)) == {
        "/workspace/.cache/huggingface": str(hf_home)
    }


@pytest.mark.parametrize("name", ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"])
def test_host_hub_cache_is_mounted_as_the_container_hub(tmp_path, name):
    hub = tmp_path / "models" / "hub"
    mounts = _mounts(tmp_path, **{name: str(hub)})
    assert mounts["/workspace/.cache/huggingface/hub"] == str(hub)
    assert hub.is_dir()


def test_xdg_cache_home_moves_the_default_cache(tmp_path):
    xdg = tmp_path / "xdg"
    assert _mounts(tmp_path, XDG_CACHE_HOME = str(xdg)) == {
        "/workspace/.cache/huggingface": str(xdg / "huggingface")
    }


def test_home_relative_paths_are_expanded(tmp_path):
    home = tmp_path / "home"
    mounts = _mounts(tmp_path, HF_HOME = "~/hf", HF_HUB_CACHE = "~/models/hub")
    assert mounts == {
        "/workspace/.cache/huggingface": str(home / "hf"),
        "/workspace/.cache/huggingface/hub": str(home / "models" / "hub"),
    }
    assert (home / "models" / "hub").is_dir()
    assert not (tmp_path / "~").exists()


def test_only_a_hub_created_here_is_flagged(tmp_path):
    hub = tmp_path / "models" / "hub"
    _mounts(tmp_path, HF_HUB_CACHE = str(hub))
    assert "UNSLOTH_HF_HUB_CREATED=1" in (tmp_path / "argv").read_text(encoding = "utf-8")
    (tmp_path / "bin").rename(tmp_path / "bin1")
    _mounts(tmp_path, HF_HUB_CACHE = str(hub))
    assert "UNSLOTH_HF_HUB_CREATED" not in (tmp_path / "argv").read_text(encoding = "utf-8")


def test_a_differently_spelled_default_hub_is_not_mounted_twice(tmp_path):
    hf = tmp_path / "hf"
    mounts = _mounts(tmp_path, HF_HOME = f"{hf}/", HF_HUB_CACHE = f"{tmp_path}//hf/hub/")
    assert mounts == {"/workspace/.cache/huggingface": str(hf)}

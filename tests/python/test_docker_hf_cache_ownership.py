# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import shutil
import signal
import stat
import subprocess
import sys
import time

import pytest

_DOCKER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docker"
)
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bash") is None,
    reason = "container entrypoints need Linux bash and GNU stat/chown",
)


@pytest.fixture(params = ["entrypoint.sh", "entrypoint-rocm.sh"])
def entrypoint(request):
    return os.path.join(_DOCKER, request.param)


def _stub(path, body):
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _boot(entrypoint, tmp_path, hf_home, bindir, until):
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "HF_HOME": str(hf_home),
        "UNSLOTH_SKIP_GPU_CHECK": "1",
        "UNSLOTH_STUDIO_HOME": str(tmp_path / "studio"),
    }
    proc = subprocess.Popen(
        ["bash", entrypoint, "true"],
        env = env,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        start_new_session = True,
    )
    try:
        assert proc.wait(timeout = 30) == 0
        deadline = time.monotonic() + 5
        while not until() and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _download_as_root(hub):
    for d in (
        hub / "models--x" / "blobs",
        hub / "models--x" / "refs",
        hub / "models--x" / "snapshots" / "abc",
        hub / ".locks" / "models--x",
    ):
        d.mkdir(parents = True, exist_ok = True)
    (hub / "models--x" / "blobs" / "f00").write_text("{}")
    (hub / "models--x" / "refs" / "main").write_text("abc")
    (hub / ".locks" / "models--x" / "f00.lock").write_text("")
    os.symlink("../../blobs/f00", hub / "models--x" / "snapshots" / "abc" / "config.json")


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() != 0, reason = "chown to another uid needs root"
)
def test_a_model_root_downloaded_into_a_host_owned_cache_is_handed_to_the_host_user(
    entrypoint, tmp_path
):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    os.chown(hf_home, 1000, 1000)
    _download_as_root(hf_home / "hub")
    entries = [hf_home / "hub", *(hf_home / "hub").rglob("*")]
    bindir = tmp_path / "bin"
    bindir.mkdir()

    _boot(
        entrypoint,
        tmp_path,
        hf_home,
        bindir,
        lambda: all(os.lstat(p).st_uid == 1000 for p in entries),
    )

    assert [
        str(p) for p in entries if (os.lstat(p).st_uid, os.lstat(p).st_gid) != (1000, 1000)
    ] == []


def _recording_chown(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "chown.calls"
    _stub(str(bindir / "id"), "echo 0\n")
    _stub(str(bindir / "chown"), f'echo "$*" >> "{calls}"\n')
    return bindir, calls


@pytest.mark.skipif(
    os.name == "posix" and os.geteuid() == 0, reason = "root runs the real chown test above"
)
def test_the_host_owned_cache_is_re_owned_to_its_owner_at_startup(entrypoint, tmp_path):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    bindir, calls = _recording_chown(tmp_path)
    st = os.stat(hf_home)

    _boot(entrypoint, tmp_path, hf_home, bindir, calls.exists)

    assert calls.exists(), "nothing re-owned the root-written downloads"
    call = calls.read_text().splitlines()[0].split()
    assert call[-2:] == [f"{st.st_uid}:{st.st_gid}", str(hf_home)]
    assert "-R" in call and "--from=0" in call, call


def test_a_root_owned_cache_is_left_alone(entrypoint, tmp_path):
    bindir, calls = _recording_chown(tmp_path)

    _boot(entrypoint, tmp_path, "/", bindir, lambda: False)

    assert not calls.exists(), calls.read_text()

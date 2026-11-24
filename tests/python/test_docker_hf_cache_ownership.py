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
        assert proc.wait(timeout = 120) == 0
        deadline = time.monotonic() + 5
        while not until() and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _download_as_root(hub, outside):
    old = os.umask(0o077)
    try:
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
        (hub / "models--x" / "blobs" / "f01.incomplete").write_text("partial")
        outside.write_text("secret")
    finally:
        os.umask(old)
    os.symlink("../../blobs/f00", hub / "models--x" / "snapshots" / "abc" / "config.json")
    os.symlink(outside, hub / "models--x" / "escape")
    os.link(outside, hub / "models--x" / "hardlink")
    acl = hub / "models--x" / "blobs" / "acl"
    acl.write_text("{}")
    if shutil.which("setfacl"):
        subprocess.run(["setfacl", "-m", "g:2000:rw", str(acl)], check = False)
    every = [hub, *hub.rglob("*")]
    kept = {p for p in every if p.is_symlink() or p.name == "hardlink" or p.suffix == ".incomplete"}
    if os.listxattr(acl, follow_symlinks = False).count("system.posix_acl_access"):
        kept.add(acl)
    return [p for p in every if p not in kept], sorted(kept)


def _handed_over(path, uid):
    st = os.lstat(path)
    if st.st_uid != uid or st.st_gid != 0:
        return False
    want = 0o070 if stat.S_ISDIR(st.st_mode) else 0o060
    return st.st_mode & want == want


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() != 0, reason = "chown to another uid needs root"
)
def test_a_model_root_downloaded_into_a_host_owned_cache_is_handed_to_the_host_user(
    entrypoint, tmp_path
):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    os.chown(hf_home, 1000, 1000)
    outside = tmp_path / "outside"
    entries, kept = _download_as_root(hf_home / "hub", outside)
    before = {p: os.lstat(p) for p in kept}
    shared = hf_home / "hub" / "shared"
    shared.mkdir()
    os.chown(shared, 1000, 2000)
    os.chmod(shared, 0o2750)
    (shared / "blob").write_text("{}")
    os.chmod(shared / "blob", 0o640)
    bindir = tmp_path / "bin"
    bindir.mkdir()

    _boot(
        entrypoint, tmp_path, hf_home, bindir, lambda: all(_handed_over(p, 1000) for p in entries)
    )

    assert [str(p) for p in entries if not _handed_over(p, 1000)] == []
    assert os.lstat(hf_home).st_gid == 1000
    assert [str(p) for p in kept if os.lstat(p) != before[p]] == []
    shared = hf_home / "hub" / "shared"
    blob = os.lstat(shared / "blob")
    assert (blob.st_uid, blob.st_gid, stat.S_IMODE(blob.st_mode)) == (0, 2000, 0o640)
    st = os.stat(outside)
    assert (st.st_uid, stat.S_IMODE(st.st_mode)) == (0, 0o600)


_BOOT = """
import os, sys
log = open(sys.argv[1], "a", buffering = 1)
here = lambda fd: os.readlink(f"/proc/self/fd/{fd}")
os.fchown = lambda fd, u, g: log.write(f"{here(fd)} {u}:{g}\\n")
real_open, out = os.open, os.path.join(os.path.dirname(sys.argv[1]), "racy_out")


def racy_open(name, flags, mode = 0o777, *, dir_fd = None):
    fd = real_open(name, flags, mode, dir_fd = dir_fd)
    if name == "racy":  # moves the cache name outside right after it is opened
        os.link(name, out, src_dir_fd = dir_fd)
        os.unlink(name, dir_fd = dir_fd)
    return fd


os.open = racy_open
sys.argv = ["-", sys.argv[2], sys.argv[3], str(os.getuid()), str(os.getgid())]
exec(compile(sys.stdin.read(), "<entrypoint>", "exec"), {"__name__": "__main__"})
"""


def _recording_python(tmp_path):
    """Runs the entrypoint's own sweep with our uid standing in for root's, logging its chowns."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "chown.calls"
    (tmp_path / "boot.py").write_text(_BOOT)
    _stub(str(bindir / "id"), "echo 0\n")
    _stub(
        str(bindir / "python3"),
        f'exec "{sys.executable}" -I -S "{tmp_path / "boot.py"}" "{calls}" "$4" "$5"\n',
    )
    return bindir, calls


@pytest.mark.skipif(
    os.name == "posix" and os.geteuid() == 0, reason = "root runs the real chown test above"
)
def test_root_written_entries_go_to_the_cache_owner_and_stay_group_root_usable(
    entrypoint, tmp_path
):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    outside = tmp_path / "outside"
    entries, _ = _download_as_root(hf_home / "hub", outside)
    racy = hf_home / "hub" / "models--x" / "blobs" / "racy"
    racy.write_text("{}")
    racy.chmod(0o600)
    bindir, calls = _recording_python(tmp_path)

    _boot(entrypoint, tmp_path, hf_home, bindir, lambda: len(_logged(calls)) >= len(entries))

    owner = f"{os.getuid()}:-1"
    logged = _logged(calls)
    assert {str(p) for p in entries} == set(logged), sorted(logged)
    assert set(logged.values()) == {owner}
    weak = [str(p) for p in entries if not _handed_over_mode(p)]
    assert stat.S_IMODE(os.stat(tmp_path / "racy_out").st_mode) == 0o600
    assert [p for p in logged if "racy" in p] == []
    assert weak == []
    assert stat.S_IMODE(os.stat(outside).st_mode) == 0o600


def _logged(calls):
    if not calls.exists():
        return {}
    out = {}
    for line in calls.read_text().splitlines():
        path, who = line.split(" ")
        out[path] = who
    return out


def _handed_over_mode(path):
    st = os.lstat(path)
    want = 0o070 if stat.S_ISDIR(st.st_mode) else 0o060
    return st.st_mode & want == want and not st.st_mode & 0o6000


def test_a_root_owned_cache_is_left_alone(entrypoint, tmp_path):
    bindir, calls = _recording_python(tmp_path)

    _boot(entrypoint, tmp_path, "/", bindir, lambda: False)

    assert not calls.exists(), calls.read_text()


def test_a_root_user_outside_group_root_hands_nothing_over(entrypoint, tmp_path):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    _download_as_root(hf_home / "hub", tmp_path / "outside")
    bindir, calls = _recording_python(tmp_path)
    _stub(str(bindir / "id"), '[[ "$1" == -g ]] && echo 2000 || echo 0\n')

    _boot(entrypoint, tmp_path, hf_home, bindir, lambda: False)

    assert not calls.exists(), calls.read_text()

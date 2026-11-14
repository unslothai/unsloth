# SPDX-License-Identifier: AGPL-3.0-only

import os
import subprocess
import sys
import time

import pytest

from core.inference import diffusion_te_release as snapshots
from hub.utils import download_registry as downloads
from utils import process_lifetime

PROBES = [(snapshots, "_pid_alive"), (downloads, "_process_alive")]


@pytest.fixture
def exited_owner():
    if sys.platform == "win32":
        pytest.skip("Windows does not retain POSIX zombies.")
    process = subprocess.Popen([sys.executable, "-c", "pass"], stdout = subprocess.PIPE)
    try:
        process.stdout.read()
        deadline = time.monotonic() + 5
        while not process_lifetime._pid_is_zombie(process.pid):
            if time.monotonic() >= deadline:
                pytest.fail("The exited child did not become a zombie.")
        yield process.pid
    finally:
        process.stdout.close()
        process.wait(timeout = 10)


@pytest.fixture
def live_owner():
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin = subprocess.PIPE,
    )
    try:
        yield process.pid
    finally:
        process.stdin.close()
        process.wait(timeout = 10)


@pytest.mark.parametrize("module, name", PROBES, ids = ["snapshots", "downloads"])
def test_exited_owner_is_not_running(module, name, exited_owner):
    assert not getattr(module, name)(exited_owner)


@pytest.mark.parametrize("module, name", PROBES, ids = ["snapshots", "downloads"])
def test_live_owner_is_still_running(module, name, live_owner):
    assert getattr(module, name)(live_owner)


@pytest.mark.parametrize("module, name", PROBES, ids = ["snapshots", "downloads"])
def test_owner_checks_delegate_without_signaling(module, name, monkeypatch):
    signals = []
    checked = []
    monkeypatch.setattr(os, "kill", lambda *args: signals.append(args))
    monkeypatch.setattr(
        module,
        "pid_is_running",
        lambda pid: checked.append(pid) or False,
        raising = False,
    )

    assert not getattr(module, name)(12345)
    assert checked == [12345]
    assert not signals


def test_snapshot_cleanup_removes_exited_owners_but_keeps_live_ones(
    tmp_path, exited_owner, live_owner
):
    dead = tmp_path / f"{exited_owner}-dead"
    live = tmp_path / f"{live_owner}-live"
    own = tmp_path / f"{os.getpid()}-own"
    for directory in (dead, live, own):
        directory.mkdir()
        (directory / "weights.bin").write_bytes(b"saved weights")

    snapshots._sweep_stale(tmp_path)

    assert not dead.exists()
    assert (live / "weights.bin").read_bytes() == b"saved weights"
    assert (own / "weights.bin").read_bytes() == b"saved weights"


def test_snapshot_owner_pid_one_is_kept_without_signaling(monkeypatch):
    signals = []
    monkeypatch.setattr(os, "kill", lambda *args: signals.append(args))

    assert snapshots._pid_alive(1)
    assert not signals

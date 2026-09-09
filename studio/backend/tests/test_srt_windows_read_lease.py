# SPDX-License-Identifier: AGPL-3.0-only
import threading
import shutil
import sys
import time
from types import SimpleNamespace

import pytest
from core.inference import srt_adapter, srt_probe, srt_windows_read_lease as leases


@pytest.fixture
def owners(monkeypatch):
    created = []
    monkeypatch.setattr(leases, "_owner", None)
    monkeypatch.setattr(leases, "_stopping", False)
    monkeypatch.setattr(srt_adapter, "installation_identity", lambda: "installed")
    monkeypatch.setattr(srt_probe, "runtime_inputs", lambda: ("selected-python",))

    class Owner:
        def __init__(self, identity, deadline, **kwargs):
            self.identity = identity
            self.port = 12345
            self.token = "fixture"
            self.closed = 0
            self.proc = SimpleNamespace(poll = lambda: None)
            self.bootstrap = kwargs["bootstrap"]
            created.append(self)

        def close(self):
            self.closed += 1

    monkeypatch.setattr(leases, "_Broker", Owner)
    yield created
    leases.shutdown()


def test_workdirs_and_commands_share_only_runtime_reads(owners):
    first = leases.acquire({"readRoots": ["runtime"], "cwd": "one", "writeRoots": ["one"]})
    second = leases.acquire({"readRoots": ["runtime"], "cwd": "two", "writeRoots": ["two"]})
    assert first == second
    assert len(owners) == 1
    assert owners[0].bootstrap == {"readRoots": ["runtime"]}


def test_runtime_change_releases_before_new_grants(owners, monkeypatch):
    leases.acquire({"readRoots": ["runtime"]})
    monkeypatch.setattr(srt_probe, "runtime_inputs", lambda: ("other-python",))
    leases.acquire({"readRoots": ["runtime"]})
    assert owners[0].closed == 1
    assert len(owners) == 2


def test_cancelled_admission_starts_no_owner(owners):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(srt_adapter.SrtError):
        leases.acquire({"readRoots": ["runtime"]}, cancel)
    assert not owners


def test_shutdown_blocks_further_admission(owners):
    leases.acquire({"readRoots": ["runtime"]})
    leases.shutdown()
    with pytest.raises(srt_adapter.SrtError):
        leases.acquire({"readRoots": ["runtime"]})
    assert len(owners) == 1


def test_failed_release_does_not_admit_changed_runtime(owners):
    leases.acquire({"readRoots": ["runtime"]})
    original = owners[0].close

    def failed():
        raise RuntimeError("revoke failed")

    owners[0].close = failed
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                leases.acquire({"readRoots": ["other-runtime"]})
        assert len(owners) == 1
    finally:
        owners[0].close = original


def test_default_windows_dispatch_uses_verified_lease(owners, monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_STUDIO_SRT_READ_LEASE", raising = False)
    monkeypatch.setattr(srt_adapter.sys, "platform", "win32")
    monkeypatch.setattr(srt_adapter, "RUNTIME", tmp_path)
    (tmp_path / "bridge.mjs").touch()
    dispatched = []
    monkeypatch.setattr(
        srt_adapter, "_spawn_windows", lambda request, **kwargs: dispatched.append(kwargs)
    )
    srt_adapter.spawn({"readRoots": ["runtime"]})
    assert len(owners) == 1
    assert dispatched[0]["read_transport"]["token"] == owners[0].token


def test_failed_default_lease_never_falls_back(owners, monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_STUDIO_SRT_READ_LEASE", raising = False)
    monkeypatch.setattr(srt_adapter.sys, "platform", "win32")
    monkeypatch.setattr(srt_adapter, "RUNTIME", tmp_path)
    (tmp_path / "bridge.mjs").touch()
    dispatched = []
    monkeypatch.setattr(
        srt_adapter, "_spawn_windows", lambda *args, **kwargs: dispatched.append(kwargs)
    )
    leases.shutdown()
    with pytest.raises(srt_adapter.SrtError):
        srt_adapter.spawn({"readRoots": ["runtime"]})
    assert dispatched == []


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows owned helper lifecycle")
def test_unresponsive_owner_reports_timeout_and_is_reaped(monkeypatch, tmp_path):
    from core.inference.srt_windows_owner import _Broker

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the real helper")
    # This real helper withholds readiness, then exits on its controller's EOF.
    # It does not install grants or execute a sandbox payload.
    (tmp_path / "windows-read-owner.mjs").write_text(
        "process.stdin.resume();process.stdin.on('end',()=>process.exit(0));"
    )
    monkeypatch.setattr(srt_adapter, "RUNTIME", tmp_path)
    monkeypatch.setattr(srt_adapter, "node_executable", lambda: node)
    original = srt_adapter.subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(srt_adapter.subprocess, "Popen", launch)
    with pytest.raises(srt_adapter.SrtError) as error:
        _Broker("fixture", time.monotonic() + 0.05, bootstrap = {"readRoots": []})
    assert error.value.diagnostic.code == "probe_timeout"
    assert len(children) == 1
    assert children[0].poll() == 0
    assert children[0].stdin.closed

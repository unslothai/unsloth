# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Scoped real worker cleanup and bounded failure, not full backend qualification."""

import os
from pathlib import Path
import sys
import time
import subprocess

import pytest

from test_content_access import snapshot, identities, grants, READ_EXECUTE
from test_preparation import BACKEND, _worker_handle, lpac
from core.inference.windows_sandbox import preparation, content_access
from core.inference.windows_sandbox.content import RuntimeContentStore
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows reader cleanup")


@pytest.fixture
def observe_worker(monkeypatch):
    original = lpac.WindowsLpacProcess.__init__
    handles = []

    def capture(process, *args):
        original(process, *args)
        handle = _worker_handle(process.pid)
        assert handle
        handles.append(handle)

    monkeypatch.setattr(lpac.WindowsLpacProcess, "__init__", capture)
    yield handles
    try:
        assert handles
        assert all(lpac._api().kernel32.WaitForSingleObject(handle, 0) == 0 for handle in handles)
    finally:
        for handle in handles:
            assert lpac._api().kernel32.CloseHandle(handle)


def test_bounded_cleanup_revokes_only_the_exact_reader_in_the_worker(
    snapshot, identities, monkeypatch, observe_worker
):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    one = store.read_access(digest, identities[0].sid_string).__enter__()
    two = store.read_access(digest, identities[1].sid_string).__enter__()
    one.cleanup_in_worker = True
    path = one.generation.files[0]
    try:
        with monkeypatch.context() as context:
            context.setattr(
                store.api,
                "set_owned_dacl",
                lambda *args: pytest.fail("Broker changed a runtime ACL"),
            )
            one.close()
        assert one.closed and not one.pins.handles
        assert grants(store, path) == {two.sid: READ_EXECUTE}
        assert not (store.root / ".readers" / (one.name + ".json")).exists()
        assert (store.root / ".readers" / (two.name + ".json")).exists()
        # Exact cleanup is idempotent and does not recover other stale readers.
        two.pins.close()
        preparation.cleanup_runtime_reader(store.root, one.name, one.sid)
        assert grants(store, path) == {two.sid: READ_EXECUTE}
    finally:
        one.close()
        two.close()
    assert not list((store.root / ".readers").iterdir())


@pytest.mark.parametrize("case", ["live", "wrong_sid"])
def test_live_or_different_reader_cannot_be_reclaimed(snapshot, identities, observe_worker, case):
    store, spec, _ = snapshot
    digest = store.publish(spec)
    with store.read_access(digest, identities[0].sid_string) as lease:
        sid = lease.sid if case == "live" else identities[1].sid_string
        with pytest.raises(WindowsRuntimeError):
            preparation.cleanup_runtime_reader(store.root, lease.name, sid)
        assert grants(store, lease.generation.files[0]) == {lease.sid: READ_EXECUTE}
        assert lease.pins.handles


def test_cleanup_never_initializes_a_missing_store(tmp_path, observe_worker):
    target = tmp_path / "not-a-store"
    with pytest.raises(WindowsRuntimeError):
        preparation.cleanup_runtime_reader(target, "a" * 32, "S-1-15-2-1-2-3-4-5-6-7")
    assert not target.exists()


@pytest.mark.parametrize("phase", ["before_revoke", "after_revoke"])
def test_deadline_kills_stuck_cleanup_worker_and_retry_does_not_replay(
    snapshot, identities, tmp_path, monkeypatch, observe_worker, phase
):
    store, spec, _ = snapshot
    lease = store.read_access(store.publish(spec), identities[0].sid_string).__enter__()
    lease.pins.close()
    original = preparation._run_worker
    wrapper = tmp_path / "fixed-cleanup-fixture.py"
    wrapper.write_text(
        f"""
import sys, os
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import content_access
from core.inference.windows_sandbox.reader_cleanup_worker import main
original = content_access._change
def change(*args, **kwargs):
    if {phase!r} == 'after_revoke':
        original(*args, **kwargs)
    r,w = os.pipe()
    os.read(r,1)
content_access._change = change
raise SystemExit(main())
""",
        encoding = "utf-8",
    )

    def worker(argv, *args, **kwargs):
        assert argv[1:4] == ["-I", "-S", "-B"]
        assert argv[4].endswith("reader_cleanup_worker.py")
        return original([*argv[:4], str(wrapper), *argv[5:]], *args, **kwargs)

    try:
        started = time.monotonic()
        with monkeypatch.context() as context:
            context.setattr(preparation, "_run_worker", worker)
            with pytest.raises(WindowsRuntimeError, match = "TIMEOUT"):
                preparation.cleanup_runtime_reader(store.root, lease.name, lease.sid, timeout = 0.75)
        assert time.monotonic() - started < 8
        assert (store.root / ".readers" / (lease.name + ".json")).exists()
        expected = {lease.sid: READ_EXECUTE} if phase == "before_revoke" else {}
        assert grants(store, lease.generation.files[0]) == expected
        preparation.cleanup_runtime_reader(store.root, lease.name, lease.sid)
        assert not (store.root / ".readers" / (lease.name + ".json")).exists()
        assert grants(store, lease.generation.files[0]) == {}
    finally:
        lease.cleanup_in_worker = True
        lease.close()


def test_grant_failure_preserves_unreaped_cleanup_worker(
    snapshot, identities, monkeypatch, observe_worker
):
    store, spec, _ = snapshot
    lease = store.read_access(store.publish(spec), identities[0].sid_string)
    lease.cleanup_in_worker = True
    run = preparation._run_worker
    workers = []
    original_init = lpac.WindowsLpacProcess.__init__

    def capture(worker, *args):
        original_init(worker, *args)
        workers.append(worker)

    def blocked(argv, *args, **kwargs):
        return run(
            [*argv[:4], "-c", "import os; r,w=os.pipe(); os.read(r,1)"],
            {},
            str(store.root),
            deadline = time.monotonic() + 0.5,
            cancel = None,
        )

    def failed_grant(*args, **kwargs):
        raise OSError("Injected grant failure before payload")

    def failed_wait(worker, timeout = None):
        raise subprocess.TimeoutExpired(worker.args, timeout)

    try:
        with monkeypatch.context() as context:
            context.setattr(content_access, "_change", failed_grant)
            context.setattr(preparation, "_run_worker", blocked)
            context.setattr(lpac.WindowsLpacProcess, "__init__", capture)
            context.setattr(lpac.WindowsLpacProcess, "wait", failed_wait)
            context.setattr(lpac._WindowsJob, "terminate", lambda self: False)
            with pytest.raises(WindowsRuntimeError) as caught:
                lease.__enter__()
            assert len(workers) == 1 and workers[0].poll() is None
            assert getattr(caught.value, "retained_process", None) is workers[0]
            assert not lease.closed and lease.journaled
    finally:
        for worker in workers:
            worker.reap(timeout = 5)
            worker.close()
        lease.close()
    assert not list((store.root / ".readers").iterdir())


@pytest.mark.parametrize("owner", [(), ("x", "y"), (1, "y"), ["a" * 32, "S-1-15-2-1-2-3-4-5-6-7"]])
def test_scoped_recovery_rejects_malformed_owners_before_filesystem_access(owner):
    class NoStore:
        def _mutation(self):
            pytest.fail("Invalid cleanup touched storage")

    with pytest.raises(WindowsRuntimeError):
        content_access.recover_readers(NoStore(), owner = owner)

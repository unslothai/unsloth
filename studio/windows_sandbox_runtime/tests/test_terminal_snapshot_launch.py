# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Copied Git Bash preparation ownership without payload execution or qualification."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from core.inference.os_sandbox import ToolLaunchPlan
from core.inference.windows_sandbox import identity, terminal_launch, terminal_runtime
from core.inference.windows_sandbox.content_files import PathLease, native_files
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


def _acl_state(source):
    api = native_files()
    lease = PathLease()
    try:
        state = {}
        for root in map(Path, source.runtime_roots):
            state[str(root)] = api.security_text(lease.directory(root))
        executable = Path(source.argv[0])
        state[str(executable)] = api.security_text(lease.file(executable))
        return state
    finally:
        lease.close()


def _normalized(paths):
    return {os.path.normcase(os.path.realpath(path)) for path in paths}


@pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows snapshot preparation")
def test_selected_git_bash_prepares_copied_runtime_without_source_acl_grants(tmp_path, monkeypatch):
    from core.inference.tools import _get_shell_cmd

    argv = tuple(_get_shell_cmd("printf SNAPSHOT_PAYLOAD_MUST_NOT_RUN"))
    if Path(argv[0]).name.lower() not in ("bash", "bash.exe"):
        pytest.skip("Studio did not select an installed Git Bash")
    local, workdir, store = tmp_path / "LocalAppData", tmp_path / "work", tmp_path / "store"
    local.mkdir()
    workdir.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    spec = ToolLaunchPlan(argv, str(workdir), {}, execution_kind = "terminal")
    source = terminal_runtime.inspect_terminal_runtime(spec)
    source_bytes = Path(source.argv[0]).read_bytes()
    source_acl = _acl_state(source)
    worker_requests = []
    original_worker = terminal_launch._run_worker

    def run_worker(argv, *args, **kwargs):
        worker_requests.append(json.loads(argv[-1]))
        return original_worker(argv, *args, **kwargs)

    monkeypatch.setattr(terminal_launch, "_run_worker", run_worker)
    prepared = owner = None
    profile = journal = None
    reader_paths = ()
    try:
        prepared = terminal_launch.prepare_terminal_launch(spec, timeout = 120, store_root = str(store))
        owner = prepared.spawn_callback.__self__
        profile, journal = Path(owner.identity.profile_folder), owner.reservation.path
        reader = owner.reservation.reader
        reader_root = Path(reader.store_root) / ".readers" / reader.name
        reader_paths = (reader_root.with_suffix(".json"), reader_root.with_suffix(".lock"))
        record = identity._read(journal)

        assert worker_requests == [
            {
                "schema": 2,
                "nonce": owner.nonce,
                "input": owner.request,
                "recipe": asdict(owner.reservation.recipe),
                "content": {"store_root": str(store), "reader": owner.reader_name},
            }
        ]
        assert prepared.execution_record is None
        assert owner.started is owner.native.attempted is False
        assert owner.job_owner.process is None
        assert owner.snapshot.source == source
        assert prepared.argv == owner.selected.argv
        assert prepared.argv[1:] == source.argv[1:] == argv[1:]
        assert os.path.normcase(prepared.argv[0]) != os.path.normcase(source.argv[0])
        assert Path(prepared.argv[0]).read_bytes() == source_bytes
        assert record["version"] == 6 and record["reader"] == asdict(reader)
        assert record["purpose"] == "terminal" and "runtime_roots" not in record
        assert all(path.exists() for path in reader_paths)

        source_paths = _normalized(source.runtime_roots)
        prepared_paths = _normalized(prepared.env["PATH"].split(os.pathsep))
        copied_paths = _normalized(owner.selected.runtime_roots)
        assert not source_paths & prepared_paths
        assert copied_paths <= prepared_paths
        assert _acl_state(source) == source_acl

        base_handles = set(owner.pins.handles.values())
        reader_handles = set(owner.access.pins.handles.values())
        assert base_handles and reader_handles and base_handles.isdisjoint(reader_handles)
        assert all(
            not os.get_handle_inheritable(handle) for handle in base_handles | reader_handles
        )
    finally:
        if prepared is not None:
            prepared.cleanup()
            assert not prepared.cleanup_diagnostics
        assert _acl_state(source) == source_acl

    assert owner is not None and owner.closed and owner.job_owner.closed
    assert owner.native.closed and not owner.pins.handles and owner.access.closed
    assert profile is not None and not profile.exists()
    assert journal is not None and not journal.exists()
    assert reader_paths and all(not path.exists() for path in reader_paths)
    assert store.exists()  # The immutable cache may outlive this reader.


def _cleanup_owner(events, job_cleanup):
    owner = object.__new__(terminal_launch._TerminalLaunch)
    owner.lock = threading.Lock()
    owner.closed = False
    owner.retained_processes = []
    owner.retained_raw = []
    owner.handles = set()
    owner.job_owner = SimpleNamespace(cleanup = job_cleanup)
    owner.native = SimpleNamespace(cleanup = lambda: events.append("native"))
    owner.pins = SimpleNamespace(close = lambda: events.append("base-pins"))
    owner.access = SimpleNamespace(close = lambda: events.append("reader"))
    owner.reservation = SimpleNamespace(
        cleanup = lambda *, in_worker: events.append(("identity", in_worker))
    )
    return owner


def test_terminal_cleanup_orders_job_native_pins_reader_then_identity():
    events = []
    owner = _cleanup_owner(events, lambda: events.append("job"))
    try:
        owner.cleanup()
        assert events == ["job", "native", "base-pins", "reader", ("identity", True)]
        assert owner.closed and owner not in terminal_launch._pending_cleanup
    finally:
        terminal_launch._pending_cleanup.discard(owner)


def test_job_cleanup_failure_retains_reader_and_retries_before_release():
    events = []
    fail = [True]

    def job_cleanup():
        events.append("job-failed" if fail else "job")
        if fail:
            fail.pop()
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_TERMINAL_JOB_INVALID", "fixed descendant failure"
            )

    owner = _cleanup_owner(events, job_cleanup)
    try:
        with pytest.raises(WindowsRuntimeError, match = "fixed descendant failure"):
            owner.cleanup()
        assert events == ["job-failed"]
        assert owner.access is not None and not owner.closed
        assert owner in terminal_launch._pending_cleanup

        owner.cleanup()
        assert events == [
            "job-failed",
            "job",
            "native",
            "base-pins",
            "reader",
            ("identity", True),
        ]
        assert owner.closed and owner not in terminal_launch._pending_cleanup
    finally:
        terminal_launch._pending_cleanup.discard(owner)

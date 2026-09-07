# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Installed-artifact publication, forced worker interruption and owned recovery."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
from types import SimpleNamespace

import pefile
import pytest

import test_artifacts as artifact_fixtures
from test_preparation import assert_dead, _worker_handle

runtime_wheel = artifact_fixtures.runtime_wheel
installed_runtime = artifact_fixtures.installed_runtime

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox.content import RuntimeContentStore
from core.inference.windows_sandbox.preparation import PublishedRuntime, _decode
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows publication lane")


def harness_source(body):
    return f"""
import sys, os, json, time
sys.path.insert(0, {str(BACKEND)!r})
sys.path.extend([{str(Path(pefile.__file__).parent)!r}, {sysconfig.get_path("purelib")!r}])
from dataclasses import asdict
from core.inference.windows_sandbox.preparation import prepare_runtime_snapshot
{body}
"""


def publish(prefix, store):
    source = harness_source(f"""
from pathlib import Path
from core.inference.windows_sandbox.content_files import PathLease, native_files
from core.inference.windows_sandbox.artifacts import PACKAGE
sources = [Path(sys.base_prefix) / f'python{{sys.version_info.major}}{{sys.version_info.minor}}.dll', Path(sys.prefix) / 'Lib/site-packages' / PACKAGE / f'bin/python_host-cpython-{{sys.version_info.major}}{{sys.version_info.minor}}-x64-release.exe']
def acls():
    with PathLease() as pins:
        return [native_files().security_text(pins.file(path)) for path in sources]
before = acls()
value = prepare_runtime_snapshot(sys.executable, {str(store)!r})
assert before == acls(), 'Publication changed installed runtime ACLs'
print(json.dumps(asdict(value)))
""")
    result = subprocess.run(
        [str(prefix / "Scripts/python.exe"), "-I", "-c", source], capture_output = True, timeout = 120
    )
    assert result.returncode == 0, result.stderr
    return _decode(PublishedRuntime, json.loads(result.stdout))


@pytest.fixture(scope = "module")
def published(installed_runtime, tmp_path_factory):
    root = tmp_path_factory.mktemp("worker-publication") / "cache"
    value = publish(installed_runtime, root)
    yield value
    store = RuntimeContentStore(root)
    assert not list((root / ".readers").iterdir())
    store.collect(value.content_digest)


def test_publication_matches_its_complete_content_manifest(published):
    store = RuntimeContentStore(published.store_root)
    with store.lease(published.content_digest) as generation:
        assert len(generation.files) == len(published.spec().files)
        assert len(generation.files) > 500
        assert (generation.directory / "manifest.json").read_bytes() == published.spec().manifest()
        assert (generation.directory / "files/trusted/python_host.exe").is_file()
    assert not list(store.root.glob(".build-*"))
    assert not hasattr(published, "qualified")


def test_published_installed_artifacts_run_through_native_host(published, tmp_path):
    from test_python_host import python_launch, run

    store = RuntimeContentStore(published.store_root)
    paths = {item.source.path: item.relative_path for item in published.spec().files}
    with store.lease(published.content_digest) as generation:
        runtime = SimpleNamespace(
            store = store,
            digest = published.content_digest,
            descriptor = published.core.runtime,
            binary = generation.directory / "files/trusted/python_host.exe",
            stdlib_relative = "Lib",
            native_images = tuple(paths[path] for path in published.core.dependencies.ordered_loads),
        )
        with python_launch(
            runtime,
            tmp_path,
            "import asyncio, sqlite3; assert asyncio.run(asyncio.sleep(0, result=7)) == 7; print(sqlite3.connect(':memory:').execute('select 17').fetchone())",
            production_creation = True,
        ) as launch:
            assert "(17,)" in run(launch)


def blocking_intercept(stage, marker):
    # Only test code replaces the fixed worker's invocation with an instrumented
    # copy. No production environment switch, callback or alternate worker path.
    return f"""
import os, sys
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox.content_files import NativeFiles
def block():
    open({str(marker)!r}, 'w').write(str(os.getpid()))
    r, w = os.pipe()
    os.read(r, 1)
original_create = NativeFiles.create
def create(self, path, data):
    if {stage!r} == 'partial_copy' and str(path).endswith('.py'):
        original_create(self, path, data[:16])
        block()
    else:
        original_create(self, path, data)
        if {stage!r} == 'journal' and path.name == '.build.json':
            block()
NativeFiles.create = create
original_rename = os.rename
def rename(source, target):
    if {stage!r} == 'before_publish':
        block()
    return original_rename(source, target)
os.rename = rename
import runpy
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def wait_pid(path, process):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if path.exists() and (value := path.read_text()).isdigit():
            return int(value)
        if process.poll() is not None:
            break
        time.sleep(0.01)
    raise AssertionError(f"Fixture did not reach {path.name}; parent exit={process.poll()}")


@contextmanager
def interrupted_publication(prefix, root, stage, mode):
    marker = root.parent / "blocked-worker"
    broker_marker = root.parent / "broker-pid"
    cancel_marker = root.parent / "observer-ready"
    interception = blocking_intercept(stage, marker)
    body = f"""
from core.inference.windows_sandbox import preparation
from core.inference.windows_sandbox.profiles import WindowsRuntimeError
import threading
open({str(broker_marker)!r}, 'w').write(str(os.getpid()))
original = preparation._run_worker
def worker(argv, *args, **kwargs):
    return original([argv[0], '-I', '-S', '-B', '-c', {interception!r}, *argv[4:]], *args, **kwargs)
preparation._run_worker = worker
cancel = threading.Event()
def request_cancel():
    while not os.path.exists({str(cancel_marker)!r}):
        time.sleep(0.01)
    cancel.set()
if {mode!r} == 'cancel':
    threading.Thread(target=request_cancel, daemon=True).start()
try:
    prepare_runtime_snapshot(sys.executable, {str(root)!r}, timeout={15 if mode == "timeout" else 60}, cancel=cancel)
except WindowsRuntimeError as error:
    print(json.dumps({{'error':error.code}}), flush=True)
else:
    raise AssertionError('interrupted publication returned success')
"""
    parent = subprocess.Popen(
        [str(prefix / "Scripts/python.exe"), "-I", "-c", harness_source(body)],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    broker_handle = worker_handle = None
    try:
        broker_pid = wait_pid(broker_marker, parent)
        broker_handle = _worker_handle(broker_pid)
        worker_pid = wait_pid(marker, parent)
        worker_handle = _worker_handle(worker_pid)
        assert worker_handle
        if mode == "cancel":
            # Retain the exact native process before the broker can reap it.
            cancel_marker.write_text("ready", encoding = "ascii")
        if mode == "parent_death":
            # Kill the actual Python broker, not the venv redirector PID.
            assert lpac._api().kernel32.TerminateProcess(broker_handle, 1)
        out, error = parent.communicate(timeout = 25)
        if mode != "parent_death":
            assert parent.returncode == 0, error
            expected = (
                "WINDOWS_SANDBOX_PREPARATION_TIMEOUT"
                if mode == "timeout"
                else "WINDOWS_SANDBOX_CANCELLED"
            )
            assert json.loads(out) == {"error": expected}
        assert lpac._api().kernel32.WaitForSingleObject(worker_handle, 5000) == 0
        assert_dead(worker_pid)
        yield root
    finally:
        api = lpac._api().kernel32
        for handle in (broker_handle, worker_handle):
            if handle:
                api.TerminateProcess(handle, 1)
                api.WaitForSingleObject(handle, 5000)
                api.CloseHandle(handle)
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout = 5)


def test_publication_cancel_waits_for_native_observation_handle(
    installed_runtime, tmp_path, monkeypatch
):
    original = _worker_handle
    observed = []

    def delayed_observation(pid):
        observed.append(pid)
        if len(observed) == 2:
            # Delay only the worker observation, not broker acquisition or
            # the later dead-process check. Cancellation must wait for it.
            time.sleep(0.25)
        handle = original(pid)
        if len(observed) == 2:
            try:
                assert handle, "Cancellation reaped the worker before observation"
                assert lpac._api().kernel32.WaitForSingleObject(handle, 0) == 258
            except BaseException:
                if handle:
                    lpac._api().kernel32.CloseHandle(handle)
                raise
        return handle

    monkeypatch.setitem(globals(), "_worker_handle", delayed_observation)
    with interrupted_publication(installed_runtime, tmp_path / "cache", "before_publish", "cancel"):
        assert len(observed) == 2


@pytest.mark.parametrize("stage", ["journal", "partial_copy", "before_publish"])
@pytest.mark.parametrize("mode", ["timeout", "cancel", "parent_death"])
def test_interrupted_publication_is_quarantined_and_recovers(
    installed_runtime, tmp_path, stage, mode
):
    root = tmp_path / "cache"
    with interrupted_publication(installed_runtime, root, stage, mode):
        stages = list(root.glob(".build-*"))
        assert len(stages) == 1
        assert not any(len(path.name) == 64 for path in root.iterdir())
        # Recovery happens under the released exclusive store handle, never by PID
        # guesses or recursive deletion. A fresh worker can then publish safely.
        value = publish(installed_runtime, root)
        store = RuntimeContentStore(root)
        assert not list(root.glob(".build-*"))
        with store.lease(value.content_digest):
            pass
        store.collect(value.content_digest)


def test_recovery_does_not_delete_an_unowned_file(installed_runtime, tmp_path):
    root = tmp_path / "cache"
    with interrupted_publication(installed_runtime, root, "journal", "cancel"):
        stage = next(root.glob(".build-*"))
        from core.inference.windows_sandbox.content_files import native_files

        foreign = stage / "not-owned"
        native_files().create(foreign, b"preserve this")
        with pytest.raises(WindowsRuntimeError, match = "unowned"):
            RuntimeContentStore(root)
        assert foreign.read_bytes() == b"preserve this"
        foreign.unlink()  # remove only this test-created sentinel, after proving preservation
        store = RuntimeContentStore(root)
        assert not list(root.glob(".build-*"))
        assert sorted(path.name for path in store.root.iterdir()) == [".lock", ".readers", ".store"]


def test_publication_io_failure_retains_diagnostic_without_success(installed_runtime, tmp_path):
    root = tmp_path / "never-published"
    interception = f"""
import sys, runpy
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import content
def unavailable_store(root):
    raise OSError(5, 'fixed publication IO control')
content.RuntimeContentStore = unavailable_store
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    source = harness_source(f"""
from core.inference.windows_sandbox import preparation
from core.inference.windows_sandbox.profiles import WindowsRuntimeError
original = preparation._run_worker
def worker(argv, *args, **kwargs):
    return original([argv[0], '-I', '-S', '-B', '-c', {interception!r}, *argv[4:]], *args, **kwargs)
preparation._run_worker = worker
try:
    prepare_runtime_snapshot(sys.executable, {str(root)!r})
except WindowsRuntimeError as error:
    assert error.code == 'WINDOWS_SANDBOX_PREPARATION_IO_FAILED'
    assert 'fixed publication IO control' in str(error)
else:
    raise AssertionError('Failed publication returned success')
""")
    result = subprocess.run(
        [str(installed_runtime / "Scripts/python.exe"), "-I", "-c", source],
        capture_output = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors = "replace")
    assert not root.exists()

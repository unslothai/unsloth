# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Trace the real static publisher; this is not payload/backend qualification."""

import json
import subprocess
import sys

import pytest

from test_publication import harness_source, installed_runtime, runtime_wheel

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows publication trace")


# Test-only instrumentation surrounds the existing fixed worker entrypoint. It
# changes neither the production worker request nor the preparation/spawn API.
TRACE = r"""
import ctypes, json, os, runpy, sys
from pathlib import Path
worker, report, mode = sys.argv[1:4]
sys.argv = [worker, *sys.argv[4:]]
sys.path.insert(0, str(Path(worker).parents[3]))
from core.inference.windows_sandbox.content_files import native_files
api = native_files()
create, close, rename = api.kernel.CreateFileW, api.kernel.CloseHandle, os.rename
active, publications = {}, []
opened = 0
def save(phase):
    Path(report).write_text(json.dumps({'pid': os.getpid(), 'opened': opened,
        'publications': publications, 'remaining': list(active.values()), 'phase': phase}), encoding='utf-8')
def tracked_create(*args):
    global opened
    handle = create(*args)
    error = ctypes.get_last_error()
    if handle != ctypes.c_void_p(-1).value:
        active[handle] = str(args[0]).removeprefix('\\\\?\\')
        opened += 1
    ctypes.set_last_error(error)
    return handle
def tracked_close(handle):
    result = close(handle)
    error = ctypes.get_last_error()
    if result:
        active.pop(handle, None)
    ctypes.set_last_error(error)
    return result
def tracked_rename(source, destination):
    held = api.open(Path(source) / 'manifest.json') if mode == 'conflict' or (mode == 'released' and not publications) else None
    row = {'source': str(source), 'destination': str(destination),
           'active': list(active.values()), 'error': None, 'succeeded': False}
    publications.append(row)
    try:
        save('before_rename')
        result = rename(source, destination)
        row['succeeded'] = True
        return result
    except OSError as error:
        row['error'] = error.winerror
        raise
    finally:
        if held is not None:
            assert api.kernel.CloseHandle(held)
        save('after_rename')
api.kernel.CreateFileW, api.kernel.CloseHandle, os.rename = tracked_create, tracked_close, tracked_rename
try:
    runpy.run_path(worker, run_name='__main__')
finally:
    save('worker_exit')
"""


@pytest.mark.parametrize("transfer_pins", [False, True], ids = ["publish", "handoff"])
@pytest.mark.parametrize("lock_mode", ["normal", "persistent", "released"])
def test_repeated_real_worker_publication_closes_owned_handles(
    installed_runtime, tmp_path, transfer_pins, lock_mode
):
    force_conflict = lock_mode == "persistent"
    wrapper = tmp_path / "fixed-trace.py"
    wrapper.write_text(TRACE, encoding = "utf-8")
    body = f"""
from pathlib import Path
from core.inference.windows_sandbox import preparation
from core.inference.windows_sandbox.content import RuntimeContentStore
from core.inference.windows_sandbox.content_files import PathLease
from core.inference.windows_sandbox.profiles import WindowsRuntimeError
from core.inference.windows_sandbox import native_compat as lpac
root = Path({str(tmp_path)!r})
original = preparation._run_worker
process_init = lpac.WindowsLpacProcess.__init__
api = lpac._api().kernel32
workers = []
def observe_process(process, *args, **kwargs):
    process_init(process, *args, **kwargs)
    handle = api.OpenProcess(0x100000, False, process.pid)
    assert handle
    workers.append((process.pid, handle))
lpac.WindowsLpacProcess.__init__ = observe_process
observed = []
def trace_worker(argv, *args, **kwargs):
    assert argv[1:4] == ['-I', '-S', '-B']
    assert Path(argv[4]).name == 'preparation_worker.py'
    report = root / f'worker-{{len(observed)}}.json'
    try:
        result = original([*argv[:4], {str(wrapper)!r}, argv[4], str(report), {"conflict" if force_conflict else lock_mode!r}, *argv[5:]], *args, **kwargs)
    except BaseException:
        if report.exists():
            value = json.loads(report.read_text(encoding='utf-8'))
            observed.append(value)
            print(json.dumps(value), file=sys.stderr)
        raise
    finally:
        assert len(workers) == 1
        worker_pid, handle = workers.pop()
        try:
            assert api.WaitForSingleObject(handle, 0) == 0, 'Preparation worker still alive'
        finally:
            assert api.CloseHandle(handle)
    value = json.loads(report.read_text(encoding='utf-8'))
    assert value['pid'] == result[1] == worker_pid
    observed.append(value)
    return result
preparation._run_worker = trace_worker
for index in range({10 if lock_mode == "normal" else 1}):
    cache = root / f'cache-{{index}}'
    with PathLease() as pins:
        try:
            value = prepare_runtime_snapshot(sys.executable, cache, pins=pins if {transfer_pins!r} else None)
        except WindowsRuntimeError as error:
            assert {force_conflict!r}, str(error)
            assert error.code == 'WINDOWS_SANDBOX_PREPARATION_IO_FAILED', str(error)
            assert not pins.handles
        else:
            assert not {force_conflict!r}, 'Locked publication returned success'
            assert len(value.spec().files) > 500
            assert bool(pins.handles) == {transfer_pins!r}
    report = observed[-1]
    assert report['opened'] > 500, report
    rows = report['publications']
    row = rows[-1]
    assert 1 <= len(rows) <= 4, report
    assert all(item['source'] == row['source'] and item['destination'] == row['destination'] for item in rows), report
    assert all(not item['succeeded'] and item['error'] in (5, 32) for item in rows[:-1]), report
    for attempt, item in enumerate(rows):
        staging = Path(item['source'])
        staged_handles = {{Path(path) for path in item['active'] if Path(path).is_relative_to(staging)}}
        intentionally_held = {{staging / 'manifest.json'}} if {force_conflict!r} or ({lock_mode == "released"!r} and attempt == 0) else set()
        assert staged_handles == intentionally_held, item
    if {force_conflict!r}:
        assert row['error'] in (5, 32) and not row['succeeded'], row
        assert str(Path(row['source']) / 'manifest.json') in row['active'], row
        assert report['phase'] in ('after_rename', 'worker_exit'), report
        assert sorted(path.name for path in cache.iterdir()) == ['.lock', '.readers', '.store']
        continue
    if {lock_mode == "released"!r}:
        assert len(rows) == 2 and rows[0]['error'] in (5, 32) and not rows[0]['succeeded'], report
    assert report['phase'] == 'worker_exit' and not report['remaining'], report
    assert row['succeeded'] and row['error'] is None, row
    staging = Path(row['source'])
    assert staging.name.startswith('.build-') and staging.parent == cache
    assert Path(row['destination']) == cache / value.content_digest
    assert not any(Path(path).is_relative_to(staging) for path in row['active']), row
    for item in rows:
        assert not {{file.source.path for file in value.spec().files}}.intersection(item['active']), item
    assert not list(cache.glob('.build-*'))
    # Parent pins and worker pins are gone: exact owned collection must succeed.
    RuntimeContentStore(cache).collect(value.content_digest)
    assert not (cache / value.content_digest).exists()
print('REAL_WORKER_PUBLICATION_HANDLES_OK')
"""
    result = subprocess.run(
        [str(installed_runtime / "Scripts/python.exe"), "-I", "-c", harness_source(body)],
        capture_output = True,
        timeout = 180,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors = "replace")
    assert b"REAL_WORKER_PUBLICATION_HANDLES_OK" in result.stdout
    reports = [
        json.loads(path.read_text(encoding = "utf-8")) for path in tmp_path.glob("worker-*.json")
    ]
    assert len(reports) == (10 if lock_mode == "normal" else 1)

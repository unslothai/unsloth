# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Actual post-drop registry denial; host controls are never qualification labels."""

import ctypes
from ctypes import wintypes as W
from contextlib import contextmanager, ExitStack
import secrets
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_launch import run_harness, installed_runtime, runtime_wheel
from core.inference.windows_sandbox import probe
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows registry probe")


def test_host_registry_positive_control_and_unrestricted_negative_control():
    probe._verify_host_registry()
    source = (
        "checks=[]\nHOST_REGISTRY_PATH="
        + repr(probe.native_files().owner + "\\Software")
        + "\n"
        + probe._REGISTRY_SOURCE
    )
    result = subprocess.run([sys.executable, "-I", "-c", source], capture_output = True, timeout = 10)
    assert result.returncode != 0
    assert b"Host registry access was not denied" in result.stderr
    assert b"host_registry_read_denied', 0" in result.stderr


@pytest.mark.parametrize("code", [0, 2, 6, 87])
def test_registry_probe_does_not_accept_other_results(monkeypatch, code):
    closed = []

    def opened(*args):
        return code

    def close(handle):
        closed.append(True)
        return 0

    api = SimpleNamespace(RegOpenKeyExW = opened, RegCloseKey = close)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: api)
    with pytest.raises(AssertionError, match = "Host registry access was not denied"):
        exec(probe._REGISTRY_SOURCE, {"checks": [], "HOST_REGISTRY_PATH": "fixed test path"})
    assert len(closed) == int(code == 0)


@pytest.mark.parametrize("stage", ["query", "set", "close"])
def test_registry_host_control_checks_every_result(monkeypatch, stage):
    probe.native_files()
    opened, closed = [], []

    def open_key(root, path, options, access, output):
        opened.append(access)
        return 5 if (stage == "query" or stage == "set" and access == 0x102) else 0

    def close_key(handle):
        closed.append(True)
        return 5 if stage == "close" else 0

    api = SimpleNamespace(RegOpenKeyExW = open_key, RegCloseKey = close_key)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: api)
    with pytest.raises(WindowsRuntimeError, match = "Host registry"):
        probe._verify_host_registry()
    assert opened == ([0x101, 0x102] if stage == "set" else [0x101])
    assert len(closed) == int(stage != "query")


def test_registry_control_failure_prevents_private_payload(installed_runtime, tmp_path):
    worker_source = """
import sys
sys.path.insert(0,sys.argv.pop(1))
from core.inference.windows_sandbox import probe
def fail_registry():
    raise probe.WindowsRuntimeError('WINDOWS_SANDBOX_PROBE_FAILED','Host registry positive control failed')
probe._verify_host_registry = fail_registry
from core.inference.windows_sandbox.preparation_worker import main
raise SystemExit(main())
"""
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
from core.inference.windows_sandbox import launch,preparation,probe,identity
worker = root/'fixed-registry-failure.py'
worker.write_text({worker_source!r},encoding='utf-8')
original_worker, original_init = preparation._run_worker, launch._PythonLaunch.__init__
owners=[]
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def run(argv,*args,**kwargs):
    if argv[4].endswith('preparation_worker.py'):
        assert argv[1:4] == ['-I','-S','-B']
        argv = [*argv[:4],str(worker),str(Path(launch.__file__).parents[3]),*argv[5:]]
    return original_worker(argv,*args,**kwargs)
preparation._run_worker, launch._PythonLaunch.__init__ = run,capture
try:
    try:
        probe.prepare_python_probe(sys.executable,root/'cache')
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_PROBE_FAILED',str(error)
        assert 'Host registry' in str(error),str(error)
    else:
        raise AssertionError('Failed positive control returned a payload launch')
finally:
    preparation._run_worker, launch._PythonLaunch.__init__ = original_worker,original_init
    for owner in owners:
        owner.cleanup()
assert owners and all(owner.closed and not owner.started for owner in owners)
assert all(not owner.handles and not owner.pins.handles and not owner.file_pins.handles for owner in owners)
assert not launch._pending_cleanup and not list(work.iterdir())
with identity._journal_root() as journals:
    for owner in owners:
        path=journals/owner.reservation.recipe.filename()
        assert not path.exists() and not Path(str(path)+'.tmp').exists()
        with identity._derived_sid(owner.reservation.recipe.moniker) as (_,sid):
            assert identity._profile_path(sid) is None
print('REGISTRY_FAILURE_NO_PAYLOAD')
""",
    )
    assert "REGISTRY_FAILURE_NO_PAYLOAD" in output


@contextmanager
def _registry_sentinel():
    """The pytest parent owns the key even if the installed-runtime harness dies."""
    reg = ctypes.WinDLL("advapi32", use_last_error = True, winmode = 0x800)
    declarations = {
        "RegOpenKeyExW": [W.HKEY, W.LPCWSTR, W.DWORD, W.DWORD, ctypes.POINTER(W.HKEY)],
        "RegCloseKey": [W.HKEY],
        "RegCreateKeyExW": [
            W.HKEY,
            W.LPCWSTR,
            W.DWORD,
            W.LPWSTR,
            W.DWORD,
            W.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(W.HKEY),
            ctypes.POINTER(W.DWORD),
        ],
        "RegSetValueExW": [W.HKEY, W.LPCWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, W.DWORD],
        "RegQueryValueExW": [
            W.HKEY,
            W.LPCWSTR,
            ctypes.c_void_p,
            ctypes.POINTER(W.DWORD),
            ctypes.c_void_p,
            ctypes.POINTER(W.DWORD),
        ],
        "RegDeleteKeyExW": [W.HKEY, W.LPCWSTR, W.DWORD, W.DWORD],
    }
    for name, args in declarations.items():
        getattr(reg, name).argtypes = args
        getattr(reg, name).restype = W.LONG
    users = W.HKEY(ctypes.c_int32(0x80000003).value)
    parent, key = W.HKEY(), W.HKEY()
    parent_name = probe.native_files().owner + "\\Software"
    leaf = "UnslothSandboxProbe." + secrets.token_hex(32)
    path = parent_name + "\\" + leaf
    marker = secrets.token_bytes(32)

    def close_key():
        nonlocal key
        if key:
            assert reg.RegCloseKey(key) == 0
            key = W.HKEY()

    def remove_owned_key():
        assert reg.RegDeleteKeyExW(parent, leaf, 0x100, 0) == 0
        absent = W.HKEY()
        result = reg.RegOpenKeyExW(users, path, 0, 0x101, ctypes.byref(absent))
        if result == 0:
            assert reg.RegCloseKey(absent) == 0
        assert result == 2, result

    def close_parent():
        assert reg.RegCloseKey(parent) == 0

    def positive_controls():
        for access in (1, 2):
            assert reg.RegOpenKeyExW(users, path, 0, access | 0x100, ctypes.byref(key)) == 0
            try:
                if access == 1:
                    data, size, kind = ctypes.create_string_buffer(32), W.DWORD(32), W.DWORD()
                    assert (
                        reg.RegQueryValueExW(
                            key, "sentinel", None, ctypes.byref(kind), data, ctypes.byref(size)
                        )
                        == 0
                    )
                    assert data.raw == marker and kind.value == 3 and size.value == 32
            finally:
                close_key()

    with ExitStack() as cleanup:
        assert reg.RegOpenKeyExW(users, parent_name, 0, 0x104, ctypes.byref(parent)) == 0
        cleanup.callback(close_parent)
        disposition = W.DWORD()
        result = reg.RegCreateKeyExW(
            parent, leaf, 0, None, 1, 0x103, None, ctypes.byref(key), ctypes.byref(disposition)
        )
        if result == 0 and disposition.value == 1:
            cleanup.callback(remove_owned_key)
        cleanup.callback(close_key)
        assert result == 0 and disposition.value == 1, (result, disposition.value)
        data = ctypes.create_string_buffer(marker)
        assert reg.RegSetValueExW(key, "sentinel", 0, 3, data, len(marker)) == 0
        close_key()
        positive_controls()
        yield path, positive_controls


def test_postdrop_denies_host_registry_sentinel_and_cleans_it(installed_runtime, tmp_path):
    with _registry_sentinel() as (path, positive_controls):
        output = run_harness(
            installed_runtime,
            tmp_path,
            f"""
from core.inference.windows_sandbox import probe
source='checks=[]'+chr(10)+'HOST_REGISTRY_PATH='+repr({path!r})+chr(10)+probe._REGISTRY_SOURCE+chr(10)+'print(checks)'
try:
    script.write_text(source,encoding='utf-8')
    spec=ToolLaunchPlan(argv=(sys.executable,'-u',str(script)),workdir=str(work),env={{}},execution_kind='python')
    prepared=prepare_python_launch(spec,root/'cache')
    owner=prepared.spawn_callback.__self__
    process=spawn_prepared_launch(prepared,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,text=True,encoding='utf-8',errors='replace',
        cwd=prepared.workdir,env=prepared.env,close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW)
    output=process.stdout.read()
    assert process.wait(timeout=10)==0,output
    assert output.strip()==repr(['host_registry_read_denied','host_registry_write_denied']),output
finally:
    if 'owner' in globals():
        owner.cleanup()
assert owner.closed and not owner.reservation.path.exists()
print('HOST_REGISTRY_SENTINEL_DENIED')
""",
        )
        positive_controls()
    assert "HOST_REGISTRY_SENTINEL_DENIED" in output


def test_registry_sentinel_cleanup_runs_after_test_failure():
    with pytest.raises(RuntimeError, match = "injected harness failure"):
        with _registry_sentinel() as (_, positive_controls):
            positive_controls()
            raise RuntimeError("injected harness failure")

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source contract and mandatory Windows-live tests for the LPAC backend."""

from __future__ import annotations

import ast
import ctypes
from contextlib import contextmanager
from ctypes import wintypes
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from core.inference import os_sandbox
from core.inference import tools as inference_tools
from core.inference import windows_lpac


def _spec(
    workdir: Path,
    *argv: str,
    env: dict[str, str] | None = None,
):
    return os_sandbox.ToolLaunchPlan(
        argv = tuple(argv) or (sys.executable, "-I", "-S", "-c", "pass"),
        workdir = str(workdir),
        env = env
        or {
            "HOME": str(workdir),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
        },
    )


def test_source_only_public_api_and_profile_are_narrow_and_unique():
    path = Path(windows_lpac.__file__).resolve()
    source = path.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    public = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }

    assert path.suffix == ".py"
    assert public == {"WindowsLpacBackend", "WindowsLpacProcess"}
    assert windows_lpac.__all__ == ["WindowsLpacBackend", "WindowsLpacProcess"]
    assert "subprocess.Popen(" not in source
    assert "secrets.token_hex(16)" in source
    assert "_SECURITY_CAPABILITIES(identity.sid, None, 0, 0)" in source
    profiles = {
        os_sandbox.LinuxBubblewrapBackend.profile_id,
        os_sandbox.MacOSSeatbeltBackend.profile_id,
        windows_lpac.WindowsLpacBackend.profile_id,
    }
    assert len(profiles) == 3


def test_network_probe_rejects_an_unrestricted_process():
    # A real host launch is the negative control: these endpoints must be reachable.
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "host network endpoint"):
        with windows_lpac._probe_network_endpoints() as endpoints:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", windows_lpac._probe_network_payload(endpoints)],
                capture_output = True,
                text = True,
                timeout = 10,
                check = False,
            )
            assert result.returncode != 0
            assert "LPAC network operation was not denied" in result.stderr


@pytest.mark.parametrize("endpoint_index", [1, 3])
def test_network_probe_rejects_udp_delivery(endpoint_index):
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "host network endpoint"):
        with windows_lpac._probe_network_endpoints() as endpoints:
            family, kind, address = endpoints[endpoint_index]
            with socket.socket(family, kind) as sender:
                sender.sendto(b"escape", address)


@pytest.mark.parametrize("winerror", [10061, 10060, 10047])
def test_network_probe_does_not_accept_refusal_timeout_or_missing_family(monkeypatch, winerror):
    error = OSError("diagnostic error")
    error.winerror = winerror

    def failing_socket(*_args):
        raise error

    monkeypatch.setattr(socket, "socket", failing_socket)
    with pytest.raises(AssertionError, match = "unexpected network error"):
        exec(windows_lpac._probe_network_payload([(2, 1, ("127.0.0.1", 1234))]), {})


def test_network_probe_closes_endpoints_after_failure(monkeypatch):
    opened = []
    real_socket = socket.socket

    def tracked_socket(*args, **kwargs):
        sock = real_socket(*args, **kwargs)
        opened.append(sock)
        return sock

    monkeypatch.setattr(socket, "socket", tracked_socket)
    with pytest.raises(RuntimeError, match = "launch failed"):
        with windows_lpac._probe_network_endpoints():
            raise RuntimeError("launch failed")
    assert opened
    assert all(sock.fileno() == -1 for sock in opened)


@pytest.mark.parametrize(
    "canonical,split",
    [
        ("C:\\", ("C:", "\\")),
        (r"\\server\share\work", (r"\\server\share", r"\work")),
        ("relative", ("", "relative")),
    ],
)
def test_local_directory_rejects_roots_unc_and_drive_relative_paths(canonical, split, monkeypatch):
    monkeypatch.setattr(windows_lpac.os.path, "abspath", lambda _path: canonical)
    monkeypatch.setattr(windows_lpac.os.path, "realpath", lambda path: path)
    monkeypatch.setattr(windows_lpac.os.path, "splitdrive", lambda _path: split)
    monkeypatch.setattr(windows_lpac.os.path, "isdir", lambda _path: True)

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "non-root directory"):
        windows_lpac._canonical_local_directory("ignored")


def test_local_directory_rejects_non_fixed_drives(monkeypatch):
    kernel32 = SimpleNamespace(GetDriveTypeW = lambda _drive: 4)
    monkeypatch.setattr(windows_lpac, "_api", lambda: SimpleNamespace(kernel32 = kernel32))
    monkeypatch.setattr(windows_lpac.os.path, "abspath", lambda _path: r"C:\work")
    monkeypatch.setattr(windows_lpac.os.path, "realpath", lambda path: path)
    monkeypatch.setattr(windows_lpac.os.path, "splitdrive", lambda _path: ("C:", r"\work"))
    monkeypatch.setattr(windows_lpac.os.path, "isdir", lambda _path: True)

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "network.*removable"):
        windows_lpac._canonical_local_directory("ignored")


def test_workdir_rejects_root_and_nested_reparse_points(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    nested = workdir / "junction"
    monkeypatch.setattr(windows_lpac, "_canonical_local_directory", lambda _path: str(workdir))
    monkeypatch.setattr(windows_lpac, "_manifest_root", lambda: str(tmp_path / "manifests"))
    monkeypatch.setattr(
        windows_lpac.os,
        "walk",
        lambda *_a, **_k: [(str(workdir), [nested.name], [])],
    )
    original_lstat = windows_lpac.os.lstat

    def fake_lstat(path):
        if os.fspath(path) == str(nested):
            return SimpleNamespace(st_file_attributes = 0x400)
        info = original_lstat(path)
        return SimpleNamespace(
            st_file_attributes = 0,
            st_mode = info.st_mode,
            st_dev = info.st_dev,
            st_ino = info.st_ino,
            st_nlink = info.st_nlink,
        )

    monkeypatch.setattr(windows_lpac.os, "lstat", fake_lstat)

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "reparse point"):
        windows_lpac._validate_workdir(str(workdir))


def test_workdir_rejects_a_hardlink_crossing_its_boundary(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding = "utf-8")
    os.link(outside, workdir / "crossing-link")
    monkeypatch.setattr(windows_lpac, "_canonical_local_directory", lambda _path: str(workdir))
    monkeypatch.setattr(windows_lpac, "_manifest_root", lambda: str(tmp_path / "manifests"))

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "hardlink crossing"):
        windows_lpac._validate_workdir(str(workdir))


def test_runtime_tree_rejects_reparse_points_and_boundary_hardlinks(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    original_lstat = windows_lpac.os.lstat

    def reparse_root(path):
        info = original_lstat(path)
        if os.fspath(path) == str(runtime):
            return SimpleNamespace(st_file_attributes = 0x400, st_mode = info.st_mode)
        return info

    monkeypatch.setattr(windows_lpac.os, "lstat", reparse_root)
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "unsafe reparse"):
        windows_lpac._validate_runtime_trees((str(runtime),))
    monkeypatch.setattr(windows_lpac.os, "lstat", original_lstat)
    outside = tmp_path / "outside"
    outside.write_text("host", encoding = "utf-8")
    os.link(outside, runtime / "crossing-link")
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "hardlink crossing"):
        windows_lpac._validate_runtime_trees((str(runtime),))


def test_runtime_tree_allows_an_internal_file_reparse_alias(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    alias = runtime / "python3.exe"
    target = runtime / "python.exe"
    alias.touch()
    target.touch()
    original_lstat = windows_lpac.os.lstat
    original_realpath = windows_lpac.os.path.realpath

    def lstat_with_alias(path):
        info = original_lstat(path)
        if os.fspath(path) == str(alias):
            return SimpleNamespace(
                st_file_attributes = 0x400,
                st_mode = info.st_mode,
                st_dev = info.st_dev,
                st_ino = info.st_ino,
                st_nlink = info.st_nlink,
            )
        return info

    monkeypatch.setattr(windows_lpac.os, "lstat", lstat_with_alias)
    monkeypatch.setattr(
        windows_lpac.os.path,
        "realpath",
        lambda path: str(target) if os.fspath(path) == str(alias) else original_realpath(path),
    )

    windows_lpac._validate_runtime_trees((str(runtime),))


def test_environment_strips_host_channels_and_uses_private_temp(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    private = tmp_path / "profile" / "Temp" / "private"
    workdir.mkdir()
    private.mkdir(parents = True)
    identity = SimpleNamespace(private_temp = str(private), profile_folder = str(private.parent.parent))
    executable = os.path.realpath(sys.executable)
    monkeypatch.setattr(windows_lpac.sys, "executable", executable)

    safe = windows_lpac._safe_environment(
        {
            "KEEP": "yes",
            "APPDATA": "host-appdata",
            "DOCKER_HOST": "npipe://host",
            "HOMEDRIVE": "C:",
            "HOMEPATH": r"\Users\operator",
            "SSH_AUTH_SOCK": "host-agent",
            "USERPROFILE": "host-profile",
        },
        str(workdir),
        identity,
        (executable,),
    )

    assert safe["KEEP"] == "yes"
    assert safe["HOME"] == str(workdir)
    assert safe["USERPROFILE"] == str(workdir)
    assert safe["LOCALAPPDATA"] == identity.profile_folder
    assert {safe[name] for name in ("APPDATA", "TEMP", "TMP")} == {str(private)}
    assert safe.keys().isdisjoint({"DOCKER_HOST", "HOMEDRIVE", "HOMEPATH", "SSH_AUTH_SOCK"})


def test_initial_environment_uses_unredirected_prefix_without_mutating_plan(tmp_path):
    moniker = "unsloth.studio.test"
    profile = tmp_path / "Packages" / moniker / "AC"
    identity = SimpleNamespace(profile_folder = str(profile), moniker = moniker)
    env = {"LOCALAPPDATA": str(profile), "localappdata": "untrusted", "TEMP": str(profile / "Temp")}
    original = env.copy()
    initial = windows_lpac._initial_appcontainer_environment(env, identity)
    assert initial["LOCALAPPDATA"] == str(tmp_path)
    assert "localappdata" not in initial
    assert initial["TEMP"] == str(profile / "Temp")
    assert env == original


@pytest.mark.parametrize("layout", ["elsewhere/AC", "Packages/foreign/AC", "Packages/owned/other"])
def test_initial_environment_rejects_unexpected_profile_layout(tmp_path, layout):
    identity = SimpleNamespace(profile_folder = str(tmp_path / layout), moniker = "owned")
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "profile directory layout"):
        windows_lpac._initial_appcontainer_environment({}, identity)


def test_private_temp_validation_accepts_current_and_legacy_owned_layouts(tmp_path):
    profile = tmp_path / "profile"
    current = profile / "Temp"
    legacy = current / ("a" * 24)
    assert windows_lpac._validated_private_temp(str(profile), str(current)) == str(current)
    assert windows_lpac._validated_private_temp(str(profile), str(legacy)) == str(legacy)
    for unsafe in (profile, profile / "foreign", tmp_path / "Temp", current / "foreign"):
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "outside its profile"):
            windows_lpac._validated_private_temp(str(profile), str(unsafe))


def test_legacy_private_temp_rejects_reparse_parent(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    parent = profile / "Temp"
    parent.mkdir(parents = True)
    legacy = parent / ("a" * 24)
    real_lstat = windows_lpac.os.lstat

    def reparse_parent(path, *args, **kwargs):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(parent)):
            return SimpleNamespace(st_file_attributes = 0x400)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(windows_lpac.os, "lstat", reparse_parent)
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "reparse point"):
        windows_lpac._validated_private_temp(str(profile), str(legacy))


@pytest.mark.skipif(sys.platform != "win32", reason = "native Windows launcher regression")
def test_native_shell_temp_matches_launch_plan_and_is_removed(tmp_path):
    # This tests the real zero-capability launcher without claiming that Python
    # or the backend as a whole qualifies. No capability or probe is overridden.
    workdir = tmp_path / "work"
    workdir.mkdir()
    backend = windows_lpac.WindowsLpacBackend()
    previous = None
    for _ in range(2):
        prepared = backend.prepare(_spec(workdir, os.environ["COMSPEC"], "/d", "/c", "echo %TEMP%"))
        identity = prepared.spawn_callback._lpac_identity
        private = Path(identity.private_temp)
        profile = Path(identity.profile_folder)
        manifest = Path(identity.manifest_path)
        process = None
        try:
            assert private != previous
            assert not (private / "previous-invocation").exists()
            process = os_sandbox.spawn_prepared_launch(
                prepared,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                stdin = subprocess.DEVNULL,
                text = True,
                close_fds = True,
                creationflags = subprocess.CREATE_NO_WINDOW,
            )
            process.wait(timeout = 10)
            output = process.stdout.read().strip()
            assert process.returncode == 0, output
            assert output == prepared.env["TEMP"] == str(private)
            (private / "previous-invocation").write_text("owned sentinel")
        finally:
            prepared.cleanup()
        assert not prepared.cleanup_diagnostics
        assert not private.exists() and not profile.exists() and not manifest.exists()
        previous = private


def test_process_termination_does_not_depend_on_leader_liveness():
    events = []
    process = windows_lpac.WindowsLpacProcess(
        (),
        None,
        None,
        123,
        None,
        SimpleNamespace(terminate = lambda: events.append("terminate-job")),
    )
    process.returncode = 0
    process.terminate()
    assert events == ["terminate-job"]


def test_process_cleanup_closes_job_before_output_stream():
    events = []
    process = windows_lpac.WindowsLpacProcess(
        (),
        None,
        None,
        123,
        SimpleNamespace(close = lambda: events.append("close-stream")),
        SimpleNamespace(close = lambda: events.append("close-job")),
    )
    process.close()
    assert events == ["close-job", "close-stream"]


@pytest.mark.skipif(sys.platform != "win32", reason = "native Windows launcher regression")
def test_native_cleanup_releases_blocked_output_reader(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    prepared = windows_lpac.WindowsLpacBackend().prepare(
        _spec(
            workdir,
            os.environ["COMSPEC"],
            "/d",
            "/c",
            "for /l %i in (1,1,1000000000) do @rem waiting",
        )
    )
    process = None
    reader = None
    closer = None
    reading = threading.Event()
    read_errors = []
    stuck = False
    try:
        process = os_sandbox.spawn_prepared_launch(
            prepared,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            stdin = subprocess.DEVNULL,
            text = True,
            close_fds = True,
            creationflags = subprocess.CREATE_NO_WINDOW,
        )
        stream = process.stdout

        def drain():
            reading.set()
            try:
                stream.read()
            except Exception as exc:
                read_errors.append(exc)

        reader = threading.Thread(target = drain, daemon = True)
        reader.start()
        assert reading.wait(2)
        time.sleep(0.1)
        assert process.poll() is None and reader.is_alive()
        # Time the process-close callback separately from potentially slow ACL
        # reconciliation; the latter still runs in the finally block below.
        closer = threading.Thread(target = process.close, daemon = True)
        closer.start()
        closer.join(3)
        stuck = closer.is_alive()
    finally:
        if process is not None and (closer is None or closer.is_alive()):
            process._unsloth_job.terminate()
        if closer is not None:
            closer.join(5)
        if reader is not None:
            reader.join(5)
        prepared.cleanup()
    assert not stuck, "cleanup blocked on the reader before closing the Job Object"
    assert reader is not None and not reader.is_alive()
    assert not read_errors
    assert not prepared.cleanup_diagnostics
    assert windows_lpac._process_identity(process.pid) is None


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows runtime path validation")
def test_runtime_roots_include_sandbox_site_and_reject_network_drives(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    executable = runtime / "python.exe"
    executable.touch()
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(windows_lpac.sys, "executable", str(executable))
    monkeypatch.setattr(windows_lpac.sys, "prefix", str(runtime))
    monkeypatch.setattr(windows_lpac.sys, "base_prefix", str(runtime))
    monkeypatch.setattr(windows_lpac.sysconfig, "get_paths", lambda: {})
    roots = windows_lpac._runtime_roots(str(workdir), (str(executable),))
    sandbox_site = Path(windows_lpac.__file__).parent / "sandbox_site"
    assert os.path.realpath(sandbox_site) in roots
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "overlap"):
        windows_lpac._runtime_roots(str(runtime), (str(executable),))
    monkeypatch.setattr(
        windows_lpac,
        "_api",
        lambda: SimpleNamespace(kernel32 = SimpleNamespace(GetDriveTypeW = lambda _drive: 4)),
    )
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "network"):
        windows_lpac._runtime_roots(str(workdir), (str(executable),))


@contextmanager
def _diagnostic_sid():
    api = windows_lpac._api()
    derive = api.userenv.DeriveAppContainerSidFromAppContainerName
    derive.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    derive.restype = ctypes.c_long
    sid = ctypes.c_void_p()
    assert derive("unsloth.studio.acl-test." + uuid.uuid4().hex, ctypes.byref(sid)) == 0
    try:
        yield sid
    finally:
        api.advapi32.FreeSid(sid)


def _path_contains_sid(path, sid):
    api = windows_lpac._api()
    acl, descriptor = ctypes.c_void_p(), ctypes.c_void_p()
    assert (
        api.advapi32.GetNamedSecurityInfoW(
            str(path),
            windows_lpac._SE_FILE_OBJECT,
            windows_lpac._DACL_SECURITY_INFORMATION,
            None,
            None,
            ctypes.byref(acl),
            None,
            ctypes.byref(descriptor),
        )
        == 0
    )
    try:
        return windows_lpac._acl_contains_sid(acl, sid)
    finally:
        api.kernel32.LocalFree(descriptor)


@pytest.mark.skipif(sys.platform != "win32", reason = "native Windows ACL regression")
def test_revoke_absent_sid_never_writes_read_only_host_acl(monkeypatch):
    api = windows_lpac._api()
    path = os.path.join(os.environ["SystemRoot"], "System32")

    def unexpected_write(*_args):
        pytest.fail("cleanup attempted to change an ACL without its SID")

    # Real ACL reads; writes are forbidden, so this test cannot modify System32.
    monkeypatch.setattr(api.advapi32, "SetEntriesInAclW", unexpected_write)
    monkeypatch.setattr(api.advapi32, "SetNamedSecurityInfoW", unexpected_write)
    monkeypatch.setattr(api.advapi32, "SetFileSecurityW", unexpected_write)
    with _diagnostic_sid() as sid:
        assert not _path_contains_sid(path, sid)
        windows_lpac._revoke_sid(path, sid)
        windows_lpac._revoke_sid(path, sid, exact = True)


@pytest.mark.skipif(sys.platform != "win32", reason = "native Windows ACL regression")
def test_revoke_present_sid_preserves_another_invocation_grant(tmp_path):
    with _diagnostic_sid() as first, _diagnostic_sid() as second:
        try:
            windows_lpac._grant_read_execute(str(tmp_path), first)
            windows_lpac._grant_read_execute(str(tmp_path), second)
            assert _path_contains_sid(tmp_path, first)
            assert _path_contains_sid(tmp_path, second)
            windows_lpac._revoke_sid(str(tmp_path), first)
            assert not _path_contains_sid(tmp_path, first)
            assert _path_contains_sid(tmp_path, second)
            windows_lpac._revoke_sid(str(tmp_path), first)
            assert _path_contains_sid(tmp_path, second)
        finally:
            windows_lpac._revoke_sid(str(tmp_path), first)
            windows_lpac._revoke_sid(str(tmp_path), second)
        assert not _path_contains_sid(tmp_path, second)


def test_identity_cleanup_is_lifo_and_removes_only_its_sid(monkeypatch):
    events: list[tuple[object, ...]] = []
    sid = ctypes.c_void_p(1234)
    api = SimpleNamespace(
        userenv = SimpleNamespace(
            DeleteAppContainerProfile = lambda moniker: events.append(("profile", moniker)) or 0
        ),
        advapi32 = SimpleNamespace(FreeSid = lambda value: events.append(("sid", value.value))),
    )
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)
    monkeypatch.setattr(
        windows_lpac,
        "_revoke_sid",
        lambda path, value, *, exact = False: events.append(("acl", path, value.value, exact)),
    )
    monkeypatch.setattr(
        windows_lpac.shutil, "rmtree", lambda path, **_kwargs: events.append(("temp", path))
    )
    monkeypatch.setattr(windows_lpac.os, "unlink", lambda path: events.append(("manifest", path)))
    monkeypatch.setattr(windows_lpac, "_validated_private_temp", lambda _profile, private: private)
    identity = windows_lpac._InvocationIdentity(
        "unsloth.studio.test",
        sid,
        "S-1-15-2-1234",
        "profile",
        "private",
        "owned.json",
        ("runtime", "work", "ancestor"),
        ("ancestor",),
        100,
        200,
    )

    identity.cleanup()

    assert events == [
        ("acl", "ancestor", 1234, True),
        ("acl", "work", 1234, False),
        ("acl", "runtime", 1234, False),
        ("temp", "private"),
        ("profile", "unsloth.studio.test"),
        ("manifest", "owned.json"),
        ("sid", 1234),
    ]
    assert identity.cleaned is True


def test_reconciliation_accepts_only_owned_well_formed_manifests(monkeypatch, tmp_path):
    sid_text = "S-1-15-2-4242"
    valid = tmp_path / "unsloth.studio.valid.json"
    invalid = tmp_path / "unsloth.studio.forged.json"
    escaped = tmp_path / "unsloth.studio.escaped.json"
    active = tmp_path / "unsloth.studio.active.json"
    profile = tmp_path / "profile"
    payload = {
        "version": 1,
        "moniker": "unsloth.studio.valid",
        "sid": sid_text,
        "profile_folder": str(profile),
        "private_temp": str(profile / "Temp" / ("a" * 24)),
        "granted_roots": [str(tmp_path / "work")],
        "traverse_roots": [str(tmp_path)],
        "owner_pid": 111,
        "owner_created": 222,
    }
    valid.write_text(json.dumps(payload), encoding = "utf-8")
    invalid.write_text(json.dumps({**payload, "moniker": "foreign.profile"}), encoding = "utf-8")
    escaped.write_text(
        json.dumps(
            {
                **payload,
                "moniker": "unsloth.studio.escaped",
                "private_temp": str(tmp_path / ("b" * 24)),
            }
        ),
        encoding = "utf-8",
    )
    active.write_text(
        json.dumps(
            {**payload, "moniker": "unsloth.studio.active", "owner_pid": 333, "owner_created": 444}
        ),
        encoding = "utf-8",
    )

    class Derive:
        argtypes = None
        restype = None

        def __call__(self, _moniker, output):
            output._obj.value = 4242
            return 0

    api = SimpleNamespace(
        userenv = SimpleNamespace(DeriveAppContainerSidFromAppContainerName = Derive())
    )
    cleaned: list[str] = []
    monkeypatch.setattr(windows_lpac, "_manifest_root", lambda: str(tmp_path))
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)
    monkeypatch.setattr(windows_lpac, "_sid_string", lambda _api, _sid: sid_text)
    monkeypatch.setattr(
        windows_lpac, "_process_identity", lambda pid = None: (333, 444) if pid == 333 else None
    )
    monkeypatch.setattr(
        windows_lpac, "_profile_folder", lambda _api, _sid: payload["profile_folder"]
    )
    monkeypatch.setattr(
        windows_lpac._InvocationIdentity,
        "cleanup",
        lambda self: cleaned.append(self.moniker),
    )

    windows_lpac.WindowsLpacBackend().reconcile_stale_manifests()

    assert cleaned == ["unsloth.studio.valid"]
    assert invalid.exists() and escaped.exists() and active.exists()


@pytest.fixture(scope = "module")
def live_lpac_backend():
    if sys.platform != "win32":
        pytest.skip("native LPAC tests run only on Windows")
    capability = os_sandbox.capability_snapshot(force = True)
    assert capability.available is True, capability.reason
    assert capability.qualified is True, capability.reason
    assert capability.backend == "windows-lpac"
    assert capability.protection_state == "preview"
    assert capability.profile_id == windows_lpac.WindowsLpacBackend.profile_id
    assert "zero-capability LPAC live enforcement probe passed" in capability.reason
    backend = os_sandbox._platform_backend()
    assert isinstance(backend, windows_lpac.WindowsLpacBackend)
    return backend


def _run_native(
    backend,
    workdir: Path,
    code: str,
    *,
    timeout: int = 30,
    env: dict[str, str] | None = None,
    script: bool = False,
) -> tuple[str, float]:
    started = time.perf_counter()
    argv: tuple[str, ...]
    if script:
        script_path = workdir / f"lpac-{uuid.uuid4().hex}.py"
        script_path.write_text(code, encoding = "utf-8")
        argv = (sys.executable, "-I", "-S", str(script_path))
    else:
        argv = (sys.executable, "-I", "-S", "-c", code)
    prepared = backend.prepare(_spec(workdir, *argv, env = env))
    identity = prepared.spawn_callback._lpac_identity
    private_temp = Path(identity.private_temp)
    manifest = Path(identity.manifest_path)
    process = None
    output = ""
    try:
        process = os_sandbox.spawn_prepared_launch(
            prepared,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            stdin = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            cwd = prepared.workdir,
            env = prepared.env,
            close_fds = prepared.close_fds,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        process.wait(timeout = timeout)
        output = process.stdout.read()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout = 10)
        prepared.cleanup()
        assert not private_temp.exists()
        assert not manifest.exists()
    elapsed = time.perf_counter() - started
    assert process is not None and process.returncode == 0, output
    return output, elapsed


def _acl_text(path: Path) -> str:
    result = subprocess.run(
        ["icacls", str(path)],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        check = False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_live_profiles_sids_acl_and_owned_artifacts_are_per_invocation(live_lpac_backend, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    first = live_lpac_backend.prepare(_spec(workdir))
    second = live_lpac_backend.prepare(_spec(workdir))
    one = first.spawn_callback._lpac_identity
    two = second.spawn_callback._lpac_identity
    try:
        assert one.moniker != two.moniker
        assert one.sid_string != two.sid_string
        assert Path(one.manifest_path).is_file()
        assert Path(two.manifest_path).is_file()
        assert Path(one.private_temp).is_dir()
        assert Path(two.private_temp).is_dir()
        acl = _acl_text(workdir)
        assert one.sid_string in acl and two.sid_string in acl
        first.cleanup()
        assert not Path(one.manifest_path).exists()
        assert not Path(one.private_temp).exists()
        deleted = windows_lpac._api().userenv.DeleteAppContainerProfile(one.moniker)
        assert ctypes.c_uint32(deleted).value == 0x80070002
        assert Path(two.manifest_path).is_file()
        assert Path(two.private_temp).is_dir()
        acl = _acl_text(workdir)
        assert one.sid_string not in acl and two.sid_string in acl
    finally:
        first.cleanup()
        second.cleanup()
    acl = _acl_text(workdir)
    assert one.sid_string not in acl and two.sid_string not in acl
    deleted = windows_lpac._api().userenv.DeleteAppContainerProfile(two.moniker)
    assert ctypes.c_uint32(deleted).value == 0x80070002


def test_live_workdir_home_runtime_native_extension_and_fresh_temp(live_lpac_backend, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "input.txt").write_text("inside", encoding = "utf-8")
    outside = tmp_path / "host-secret"
    outside.write_text("secret", encoding = "utf-8")
    outside_write = tmp_path / "host-write"
    runtime_write = Path(sys.executable).parent / f"lpac-write-{uuid.uuid4().hex}"
    code = f"""
import _ssl, os, pathlib, tempfile
def denied(path, mode):
    try:
        with open(path, mode) as stream:
            stream.read(1) if 'r' in mode else stream.write('escape')
    except OSError:
        return
    raise AssertionError('escaped filesystem boundary: ' + path)
assert open('input.txt', encoding='utf-8').read() == 'inside'
open('output.txt', 'w', encoding='utf-8').write('written')
assert pathlib.Path.home().resolve() == pathlib.Path.cwd().resolve()
open(pathlib.Path.home() / 'home-output', 'w', encoding='utf-8').write('home')
denied({str(outside)!r}, 'r')
denied({str(outside_write)!r}, 'w')
denied({str(runtime_write)!r}, 'w')
assert _ssl.__file__ and open(_ssl.__file__, 'rb').read(1)
private = pathlib.Path(os.environ['TEMP'])
assert private == pathlib.Path(os.environ['TMP'])
assert not list(private.iterdir())
handle = tempfile.NamedTemporaryFile(dir=private, delete=False)
handle.write(b'private'); handle.close()
assert pathlib.Path(handle.name).read_bytes() == b'private'
print('LPAC_FILESYSTEM_OK')
"""

    output, _elapsed = _run_native(live_lpac_backend, workdir, code)

    assert "LPAC_FILESYSTEM_OK" in output
    assert (workdir / "output.txt").read_text(encoding = "utf-8") == "written"
    assert (workdir / "home-output").read_text(encoding = "utf-8") == "home"
    assert outside.read_text(encoding = "utf-8") == "secret"
    assert not outside_write.exists()
    assert not runtime_write.exists()


def _listener(
    family,
    address,
    *,
    udp = False,
):
    kind = socket.SOCK_DGRAM if udp else socket.SOCK_STREAM
    server = socket.socket(family, kind)
    server.bind(address)
    if not udp:
        server.listen(1)
    return server, server.getsockname()


def test_live_ipv4_ipv6_udp_dns_loopback_and_host_pipe_are_denied(live_lpac_backend, tmp_path):
    from multiprocessing.connection import Client, Listener

    workdir = tmp_path / "work"
    workdir.mkdir()
    ipv4, address4 = _listener(socket.AF_INET, ("127.0.0.1", 0))
    ipv6, address6 = _listener(socket.AF_INET6, ("::1", 0))
    udp, udp_address = _listener(socket.AF_INET, ("127.0.0.1", 0), udp = True)
    pipe_address = rf"\\.\pipe\unsloth-host-{uuid.uuid4().hex}"
    pipe = Listener(pipe_address, family = "AF_PIPE", authkey = None)
    pipe_reached = threading.Event()

    def accept_pipe():
        try:
            connection = pipe.accept()
            pipe_reached.set()
            connection.close()
        except OSError:
            pass

    waiter = threading.Thread(target = accept_pipe, daemon = True)
    waiter.start()
    code = f"""
import socket, threading
from multiprocessing.connection import Client
def denied(family, address):
    client = socket.socket(family); client.settimeout(1)
    try:
        client.connect(address)
    except OSError:
        return
    finally:
        client.close()
    raise AssertionError('host endpoint reachable: ' + repr(address))
denied(socket.AF_INET, {address4!r})
denied(socket.AF_INET6, {address6!r})
packet = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    try: packet.sendto(b'LPAC_ESCAPE', {udp_address!r})
    except OSError: pass
finally: packet.close()
try:
    socket.getaddrinfo('example.com', 443)
except OSError:
    pass
else:
    raise AssertionError('DNS escaped LPAC')
outcome = []
def reach_pipe():
    try:
        Client({pipe_address!r}, family='AF_PIPE', authkey=None).close()
        outcome.append('reached')
    except OSError:
        outcome.append('denied')
attempt = threading.Thread(target=reach_pipe, daemon=True); attempt.start(); attempt.join(3)
assert outcome == ['denied'], outcome
print('LPAC_NETWORK_OK')
"""
    try:
        output, _elapsed = _run_native(live_lpac_backend, workdir, code)
        udp.settimeout(0.2)
        with pytest.raises((TimeoutError, socket.timeout)):
            udp.recvfrom(128)
        assert not pipe_reached.is_set()
    finally:
        for server in (ipv4, ipv6, udp):
            server.close()
        if not pipe_reached.is_set():
            try:
                Client(pipe_address, family = "AF_PIPE", authkey = None).close()
            except OSError:
                pass
        waiter.join(2)
        pipe.close()

    assert "LPAC_NETWORK_OK" in output


def test_live_unexpected_inheritable_handles_are_absent(live_lpac_backend, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.CreateEventW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateEventW.restype = wintypes.HANDLE
    kernel32.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
    kernel32.SetHandleInformation.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = []
    try:
        for _ in range(4):
            handle = kernel32.CreateEventW(None, True, False, None)
            assert handle
            assert kernel32.SetHandleInformation(handle, 1, 1)
            handles.append(int(handle))
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "UNSLOTH_TEST_HANDLES": ",".join(map(str, handles)),
        }
        code = """
import ctypes, os
from ctypes import wintypes
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetHandleInformation.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
k.GetHandleInformation.restype = wintypes.BOOL
for raw in os.environ['UNSLOTH_TEST_HANDLES'].split(','):
    flags = wintypes.DWORD()
    ctypes.set_last_error(0)
    assert not k.GetHandleInformation(wintypes.HANDLE(int(raw)), ctypes.byref(flags))
    assert ctypes.get_last_error() == 6
print('LPAC_HANDLES_OK')
"""
        output, _elapsed = _run_native(live_lpac_backend, workdir, code, env = env)
    finally:
        for handle in handles:
            kernel32.CloseHandle(handle)

    assert "LPAC_HANDLES_OK" in output


def test_live_private_multiprocessing_and_resource_sharing(live_lpac_backend, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import multiprocessing as mp
from multiprocessing.resource_sharer import DupSocket, stop
import os, socket

def child(queue):
    queue.put(('child-ok', os.environ['TEMP']))

if __name__ == '__main__':
    raw = socket.socket(socket.AF_UNIX)
    duplicate = DupSocket(raw).detach()
    assert duplicate.family == socket.AF_UNIX
    duplicate.close(); raw.close(); stop()
    context = mp.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=child, args=(queue,))
    process.start()
    value, child_temp = queue.get(timeout=15)
    process.join(15)
    assert process.exitcode == 0
    assert value == 'child-ok'
    assert child_temp == os.environ['TEMP']
    queue.close(); queue.join_thread()
    print('LPAC_MULTIPROCESSING_OK')
"""

    output, _elapsed = _run_native(live_lpac_backend, workdir, code, timeout = 40, script = True)

    assert "LPAC_MULTIPROCESSING_OK" in output


def test_live_pytorch_tensor_transfer_when_installed(live_lpac_backend, tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import torch
import torch.multiprocessing as mp

def child(queue):
    queue.put(torch.arange(4))

if __name__ == '__main__':
    context = mp.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=child, args=(queue,))
    process.start()
    tensor = queue.get(timeout=30)
    process.join(30)
    assert process.exitcode == 0
    assert torch.equal(tensor, torch.arange(4))
    queue.close(); queue.join_thread()
    print('LPAC_PYTORCH_OK')
"""

    output, _elapsed = _run_native(live_lpac_backend, workdir, code, timeout = 60, script = True)

    assert "LPAC_PYTORCH_OK" in output


def test_live_python_and_terminal_share_launcher_and_stream(
    live_lpac_backend, monkeypatch, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    real_spawn = os_sandbox.spawn_prepared_launch
    backends: list[str] = []

    def traced_spawn(prepared, **kwargs):
        backends.append(prepared.backend)
        return real_spawn(prepared, **kwargs)

    monkeypatch.setattr(inference_tools, "spawn_prepared_launch", traced_spawn)
    chunks: list[str] = []
    records: list[os_sandbox.ToolExecutionRecord] = []

    python_result = inference_tools._python_exec(
        "print('python-lpac-stream')",
        timeout = 20,
        output_callback = chunks.append,
        launch_record_callback = records.append,
    )
    terminal_result = inference_tools._bash_exec(
        "echo terminal-lpac-stream",
        timeout = 20,
        output_callback = chunks.append,
        launch_record_callback = records.append,
    )

    assert python_result.strip() == "python-lpac-stream"
    assert terminal_result.strip() == "terminal-lpac-stream"
    assert "python-lpac-stream\n" in chunks
    assert "terminal-lpac-stream\n" in chunks
    assert backends == ["windows-lpac", "windows-lpac"]
    assert len(records) == 2
    assert all(record.os_isolation and record.backend == "windows-lpac" for record in records)


def test_live_production_timeout_and_cancellation(live_lpac_backend, monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))

    timed_out = inference_tools._python_exec("import time; time.sleep(30)", timeout = 1)
    cancel = threading.Event()
    timer = threading.Timer(0.5, cancel.set)
    timer.start()
    try:
        command = (
            "sleep 30"
            if inference_tools._get_shell_cmd("sleep 30")[0].lower().endswith("bash.exe")
            else "timeout /t 30 /nobreak >NUL"
        )
        cancelled = inference_tools._bash_exec(command, cancel_event = cancel, timeout = 20)
    finally:
        timer.cancel()

    assert timed_out == "Execution timed out after 1 seconds."
    assert cancelled == "Execution cancelled."


def test_live_job_termination_reaps_descendants(live_lpac_backend, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    marker = workdir / "descendant-survived"
    child = workdir / "child.py"
    child.write_text(
        f"import pathlib,time; time.sleep(2); pathlib.Path({str(marker)!r}).write_text('escaped')",
        encoding = "utf-8",
    )
    parent = workdir / "parent.py"
    parent.write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}], close_fds=True)\n"
        "print('READY', flush=True)\n"
        "time.sleep(30)\n",
        encoding = "utf-8",
    )
    prepared = live_lpac_backend.prepare(_spec(workdir, sys.executable, "-I", "-S", str(parent)))
    process = None
    try:
        process = os_sandbox.spawn_prepared_launch(
            prepared,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            stdin = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            cwd = prepared.workdir,
            env = prepared.env,
            close_fds = True,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert process.stdout.readline().strip() == "READY"
        process.terminate()
        process.wait(timeout = 10)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout = 10)
        prepared.cleanup()

    time.sleep(2.5)
    assert not marker.exists()


def test_live_twenty_startup_samples(live_lpac_backend, tmp_path, record_property):
    workdir = tmp_path / "work"
    workdir.mkdir()
    samples = []
    for _ in range(20):
        output, elapsed = _run_native(live_lpac_backend, workdir, "pass")
        assert output == ""
        samples.append(elapsed * 1000)
    ordered = sorted(samples)
    median = statistics.median(samples)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]

    record_property("windows_lpac_startup_samples_ms", json.dumps(samples))
    record_property("windows_lpac_startup_median_ms", f"{median:.3f}")
    record_property("windows_lpac_startup_p95_ms", f"{p95:.3f}")
    assert median > 0
    assert p95 >= median

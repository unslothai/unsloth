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
import re
import socket
import statistics
import subprocess
import sys
import sysconfig
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from core.inference import os_sandbox
from core.inference import tools as inference_tools
from core.inference import windows_lpac
from core.inference import windows_restricted_token


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
    assert "launch_id = secrets.token_hex(16)" in source
    # Windows rewrites TEMP/TMP for an AppContainer child to the package
    # redirected <profile>\Temp, so the launch owns that directory itself and
    # empties it rather than removing a per-launch subdirectory of it.
    assert "private_temp = temp_root" in source
    assert "_empty_directory(private_temp)" in source
    assert "secrets.token_hex(12)" not in source
    assert "_SECURITY_CAPABILITIES(identity.sid, None, 0, 0)" in source
    # Both container kinds are zero-capability; only the opt-out attribute differs.
    assert source.count("_SECURITY_CAPABILITIES(identity.sid, None, 0, 0)") == 1
    assert "_PROC_THREAD_ATTRIBUTE_ALL_APPLICATION_PACKAGES_POLICY" in source
    assert windows_lpac._PROFILE_ID == "windows-lpac-preview-v1"
    assert windows_lpac._APPCONTAINER_PROFILE_ID == "windows-appcontainer-preview-v1"
    profiles = {
        os_sandbox.LinuxBubblewrapBackend.profile_id,
        os_sandbox.MacOSSeatbeltBackend.profile_id,
        windows_lpac.WindowsLpacBackend.profile_id,
        windows_lpac._APPCONTAINER_PROFILE_ID,
        windows_restricted_token.WindowsRestrictedTokenBackend.profile_id,
    }
    assert len(profiles) == 5
    backend = windows_lpac.WindowsLpacBackend()
    assert backend.active_profile == "lpac"
    assert backend.active_profile_id() == windows_lpac.WindowsLpacBackend.profile_id


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
def test_native_shell_temp_matches_launch_plan_and_is_emptied(tmp_path):
    # This tests the real zero-capability launcher without claiming that Python
    # or the backend as a whole qualifies. No capability or probe is overridden.
    #
    # Windows builds the package environment for an AppContainer child and
    # overwrites TEMP/TMP with the package redirected <profile>\Temp, so what the
    # plan promises has to be that directory: a per-launch subdirectory of it was
    # planned, granted and cleaned while the child wrote to <profile>\Temp
    # anyway. The isolation the temp is there for is unchanged - it is inside the
    # container profile and is not the operator's TEMP - and it is emptied when
    # the last launch of the installation finishes.
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
            assert previous is None or private == previous
            # The previous launch emptied it, so nothing of that launch is left.
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
            # What the child sees is what the plan and the record say, and it is
            # the container's own temp, never the operator's.
            assert output == prepared.env["TEMP"] == str(private) == str(profile / "Temp")
            assert os.path.normcase(output) != os.path.normcase(os.environ["TEMP"])
            assert os.path.normcase(str(private)).startswith(os.path.normcase(str(profile)))
            (private / "previous-invocation").write_text("owned sentinel")
        finally:
            prepared.cleanup()
        assert not prepared.cleanup_diagnostics
        assert not manifest.exists()
        # The container is per installation now, so its profile and the temp
        # Windows names for every child of it outlive the launch; only what the
        # launch left inside the temp is removed.
        assert private.is_dir() and not list(private.iterdir())
        assert identity.moniker == windows_lpac._install_moniker()
        assert profile.is_dir()
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


_CONTAINER_SID = "S-1-15-2-11-22-33-44-55-66-77"
_INSTALL_MONIKER = "unsloth.studio.sandbox.0123456789abcdef"


def _cleanup_recorder(monkeypatch, events):
    sid = ctypes.c_void_p(1234)
    api = SimpleNamespace(
        userenv = SimpleNamespace(
            DeleteAppContainerProfile = lambda moniker: events.append(("profile", moniker)) or 0
        ),
        advapi32 = SimpleNamespace(FreeSid = lambda value: events.append(("sid", value.value))),
    )
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)
    monkeypatch.setattr(windows_lpac, "_SHARED_GRANTS", {})
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
    return sid


def test_launch_cleanup_is_lifo_and_leaves_the_shared_profile_and_sid_alone(monkeypatch):
    events: list[tuple[object, ...]] = []
    sid = _cleanup_recorder(monkeypatch, events)
    identity = windows_lpac._InvocationIdentity(
        _INSTALL_MONIKER,
        sid,
        _CONTAINER_SID,
        "profile",
        "private",
        "owned.json",
        ("work", "ancestor"),
        ("ancestor",),
        100,
        200,
        workdir = "work",
        launch_id = "a" * 32,
        shared_roots = ("work", "ancestor", "private"),
    )
    windows_lpac._hold_shared_grants(identity.shared_roots)

    identity.cleanup()

    assert events == [
        ("acl", "ancestor", 1234, True),
        ("acl", "work", 1234, False),
        ("temp", "private"),
        ("manifest", "owned.json"),
    ]
    # The profile and its SID belong to the installation, not to this launch.
    assert identity.cleaned is True
    assert identity.sid.value == 1234
    assert not windows_lpac._SHARED_GRANTS


def test_a_single_use_identity_still_deletes_its_profile_and_frees_its_sid(monkeypatch):
    events: list[tuple[object, ...]] = []
    sid = _cleanup_recorder(monkeypatch, events)
    identity = windows_lpac._InvocationIdentity(
        "unsloth.studio.deadbeef",
        sid,
        _CONTAINER_SID,
        "profile",
        "private",
        "owned.json",
        ("runtime", "work"),
        (),
        100,
        200,
        delete_profile = True,
        free_sid = True,
    )

    identity.cleanup()

    assert events == [
        ("acl", "work", 1234, False),
        ("acl", "runtime", 1234, False),
        ("temp", "private"),
        ("profile", "unsloth.studio.deadbeef"),
        ("manifest", "owned.json"),
        ("sid", 1234),
    ]
    assert identity.cleaned is True


def _manifest_payloads(tmp_path):
    # The layout Windows builds: <LocalAppData>\Packages\<moniker>\AC.
    profile = tmp_path / "Packages" / _INSTALL_MONIKER / "AC"
    work = tmp_path / "work"
    single_use = {
        "version": 1,
        "moniker": "unsloth.studio.valid",
        "sid": _CONTAINER_SID,
        "profile_folder": str(profile),
        "private_temp": str(profile / "Temp" / ("a" * 24)),
        "granted_roots": [str(work)],
        "traverse_roots": [str(tmp_path)],
        "owner_pid": 111,
        "owner_created": 222,
    }
    launch = {
        "version": 1,
        "kind": "lpac-launch",
        "moniker": _INSTALL_MONIKER,
        "launch_id": "b" * 32,
        "sid": _CONTAINER_SID,
        "profile_folder": str(profile),
        "private_temp": str(profile / "Temp" / ("c" * 24)),
        "workdir": str(work),
        "granted_roots": [str(work), str(tmp_path)],
        "traverse_roots": [str(tmp_path)],
        "owner_pid": 111,
        "owner_created": 222,
    }
    return single_use, launch


def test_reconciliation_accepts_only_owned_well_formed_manifests(monkeypatch, tmp_path):
    single_use, launch = _manifest_payloads(tmp_path)
    written = {
        "unsloth.studio.valid.json": single_use,
        f"unsloth.studio.launch.{'b' * 32}.json": launch,
        # A moniker outside the Studio namespace, so its SID is not ours to revoke.
        "foreign.profile.json": {**single_use, "moniker": "foreign.profile"},
        # A private temp outside the profile the SID resolves to.
        "unsloth.studio.escaped.json": {
            **single_use,
            "moniker": "unsloth.studio.escaped",
            "private_temp": str(tmp_path / ("b" * 24)),
        },
        # A live owner.
        "unsloth.studio.active.json": {
            **single_use,
            "moniker": "unsloth.studio.active",
            "owner_pid": 333,
            "owner_created": 444,
        },
        # An account SID, which the reconciler must never revoke anything for.
        "unsloth.studio.account.json": {
            **single_use,
            "moniker": "unsloth.studio.account",
            "sid": "S-1-5-21-1-2-3-1001",
        },
        # A launch manifest naming a path that is neither its workdir, its
        # private temp, nor an ancestor of either.
        f"unsloth.studio.launch.{'d' * 32}.json": {
            **launch,
            "launch_id": "d" * 32,
            "granted_roots": [*launch["granted_roots"], str(tmp_path / "victim")],
        },
        # A launch manifest whose file name does not match its launch id.
        f"unsloth.studio.launch.{'e' * 32}.json": launch,
    }
    for name, payload in written.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding = "utf-8")

    class Derive:
        def __call__(self, _moniker, output):
            output._obj.value = 4242
            return 0

    api = SimpleNamespace(
        userenv = SimpleNamespace(DeriveAppContainerSidFromAppContainerName = Derive()),
        advapi32 = SimpleNamespace(FreeSid = lambda _value: None),
    )
    cleaned: list[tuple[str, bool, bool]] = []
    monkeypatch.setattr(windows_lpac, "_manifest_root", lambda: str(tmp_path))
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)
    monkeypatch.setattr(windows_lpac, "_sid_string", lambda _api, _sid: _CONTAINER_SID)
    monkeypatch.setattr(
        windows_lpac, "_process_identity", lambda pid = None: (333, 444) if pid == 333 else None
    )
    monkeypatch.setattr(
        windows_lpac, "_profile_folder", lambda _api, _sid: single_use["profile_folder"]
    )
    monkeypatch.setattr(
        windows_lpac._InvocationIdentity,
        "cleanup",
        lambda self: cleaned.append((self.moniker, self.delete_profile, self.free_sid)),
    )

    windows_lpac.WindowsLpacBackend().reconcile_stale_manifests()

    # The single-use manifest owns the profile it names; the launch manifest
    # shares the installation's, so only the first deletes one.
    assert sorted(cleaned) == [
        (_INSTALL_MONIKER, False, True),
        ("unsloth.studio.valid", True, True),
    ]
    for name in written:
        if name not in ("unsloth.studio.valid.json", f"unsloth.studio.launch.{'b' * 32}.json"):
            assert (tmp_path / name).exists()


def test_orphan_temporary_manifests_are_swept_by_age(monkeypatch, tmp_path):
    fresh = tmp_path / "unsloth.studio.sandbox.0123456789abcdef.json.1234.tmp"
    stale = tmp_path / f"unsloth.studio.launch.{'b' * 32}.json.5678.tmp"
    unrelated = tmp_path / "other.json.tmp"
    for path in (fresh, stale, unrelated):
        path.write_text("{}", encoding = "utf-8")
    old = time.time() - windows_lpac._ORPHAN_TEMPORARY_MANIFEST_SECONDS - 60
    os.utime(stale, (old, old))
    os.utime(unrelated, (old, old))

    windows_lpac._remove_orphan_temporary_manifests(str(tmp_path))

    assert fresh.exists()
    assert not stale.exists()
    assert unrelated.exists()


def _sid_bytes(*subauthorities: int) -> bytes:
    """A syntactically valid SID: revision 1, authority 15 (package), given subauthorities."""
    import struct

    return bytes([1, len(subauthorities)]) + (15).to_bytes(6, "big") + b"".join(
        struct.pack("<I", value) for value in subauthorities
    )


def _fake_acl(*aces: tuple[int, int, int, bytes]) -> tuple[ctypes.Array, list[int]]:
    """Build an in-memory ACL of (type, flags, mask, sid) ACEs and return the buffer."""
    import struct

    body = b""
    offsets: list[int] = []
    for kind, flags, mask, sid in aces:
        size = 4 + 4 + len(sid)
        offsets.append(8 + len(body))
        body += bytes([kind, flags]) + struct.pack("<H", size) + struct.pack("<I", mask) + sid
    header = struct.pack("<BBHHH", 2, 0, 8 + len(body), len(aces), 0)
    buffer = ctypes.create_string_buffer(header + body, len(header) + len(body))
    return buffer, offsets


@contextmanager
def _fake_acl_api(monkeypatch, buffer, offsets):
    base = ctypes.addressof(buffer)

    def get_ace(_acl, index, entry):
        entry._obj.value = base + offsets[index]
        return 1

    def equal_sid(left, right):
        left_len = ctypes.string_at(left, 2)[1] * 4 + 8
        right_len = ctypes.string_at(right, 2)[1] * 4 + 8
        return int(ctypes.string_at(left, left_len) == ctypes.string_at(right, right_len))

    api = SimpleNamespace(
        advapi32 = SimpleNamespace(GetAce = get_ace, EqualSid = equal_sid),
        kernel32 = SimpleNamespace(LocalFree = lambda _value: None),
    )
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)
    yield ctypes.c_void_p(base)


def _sid_pointer(raw: bytes):
    buffer = ctypes.create_string_buffer(raw, len(raw))
    return buffer, ctypes.c_void_p(ctypes.addressof(buffer))


def test_acl_walk_accepts_ambient_read_execute_and_rejects_other_sids(monkeypatch):
    package = _sid_bytes(2, 111, 222)
    ambient = _sid_bytes(2, 1)  # S-1-15-2-1
    stranger = _sid_bytes(2, 999)
    buffer, offsets = _fake_acl(
        (0, 0, windows_lpac._FILE_GENERIC_READ | windows_lpac._FILE_GENERIC_EXECUTE, ambient),
        (0, 0, windows_lpac._GENERIC_ALL, stranger),
    )
    keep_a, package_ptr = _sid_pointer(package)
    keep_b, ambient_ptr = _sid_pointer(ambient)
    keep_c, stranger_ptr = _sid_pointer(_sid_bytes(2, 555))
    with _fake_acl_api(monkeypatch, buffer, offsets) as acl:
        required = windows_lpac._FILE_GENERIC_READ | windows_lpac._FILE_GENERIC_EXECUTE
        assert windows_lpac._acl_grants(acl, (package_ptr, ambient_ptr), required)
        assert windows_lpac._acl_grants(acl, (ambient_ptr,), windows_lpac._FILE_TRAVERSE)
        # The package SID alone holds nothing; a stranger's GENERIC_ALL never counts for us.
        assert not windows_lpac._acl_grants(acl, (package_ptr,), required)
        assert not windows_lpac._acl_grants(acl, (stranger_ptr,), windows_lpac._FILE_TRAVERSE)


def test_acl_walk_ignores_inherit_only_and_honours_deny(monkeypatch):
    ambient = _sid_bytes(2, 2)  # S-1-15-2-2
    buffer, offsets = _fake_acl(
        (0, windows_lpac._INHERIT_ONLY_ACE, windows_lpac._GENERIC_ALL, ambient),
    )
    keep, ambient_ptr = _sid_pointer(ambient)
    with _fake_acl_api(monkeypatch, buffer, offsets) as acl:
        assert not windows_lpac._acl_grants(acl, (ambient_ptr,), windows_lpac._FILE_TRAVERSE)
    buffer, offsets = _fake_acl(
        (1, 0, windows_lpac._FILE_TRAVERSE, ambient),
        (0, 0, windows_lpac._GENERIC_ALL, ambient),
    )
    with _fake_acl_api(monkeypatch, buffer, offsets) as acl:
        assert not windows_lpac._acl_grants(acl, (ambient_ptr,), windows_lpac._FILE_TRAVERSE)
    # Rights accumulate across allow ACEs the way the access check applies them.
    buffer, offsets = _fake_acl(
        (0, 0, windows_lpac._GENERIC_READ, ambient),
        (0, 0, windows_lpac._GENERIC_EXECUTE, ambient),
    )
    with _fake_acl_api(monkeypatch, buffer, offsets) as acl:
        required = windows_lpac._FILE_GENERIC_READ | windows_lpac._FILE_GENERIC_EXECUTE
        assert windows_lpac._acl_grants(acl, (ambient_ptr,), required)


def test_ambient_sid_matches_the_container_kind():
    assert windows_lpac._ambient_sid_text("lpac") == "S-1-15-2-2"
    assert windows_lpac._ambient_sid_text("appcontainer") == "S-1-15-2-1"


class _FakeCreateProfile:
    """``CreateAppContainerProfile`` recording its calls and its queued results."""

    def __init__(self, events, results = None):
        self._events = events
        self._results = list(results or [])

    def __call__(self, moniker, _display, _description, _capabilities, _count, output):
        self._events.append(("create-profile", moniker))
        result = self._results.pop(0) if self._results else 0
        if result == 0:
            output._obj.value = 4242
        return result


class _FakeDeriveSid:
    def __init__(self, events):
        self._events = events

    def __call__(self, moniker, output):
        self._events.append(("derive-sid", moniker))
        output._obj.value = 4242
        return 0


@contextmanager
def _lpac_fakes(
    monkeypatch,
    tmp_path,
    *,
    events,
    existing = (),
    create_results = None,
    runtime_roots = None,
):
    """Drive the real prepare and cleanup paths with every Windows API recorded.

    The install profile, the persistent grant bookkeeping, the manifests and the
    private temp are real; only the Win32 calls and the tree validators are fakes.
    """
    monkeypatch.setattr(windows_lpac, "_INSTALL_PROFILE", None)
    monkeypatch.setattr(windows_lpac, "_ACCESS_MEMO", {})
    monkeypatch.setattr(windows_lpac, "_SHARED_GRANTS", {})
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    # The layout Windows builds for a profile: the package directory it ACLs for
    # the container, with the AC storage inside it.
    profile = tmp_path / "Packages" / windows_lpac._install_moniker() / "AC"
    profile.mkdir(parents = True)
    runtime = str(tmp_path / "Program Files" / "Python")
    os.makedirs(runtime)
    roots = [runtime] if runtime_roots is None else list(runtime_roots)
    sid_text = "S-1-15-2-11-22-33-44-55-66-77"
    api = SimpleNamespace(
        userenv = SimpleNamespace(
            CreateAppContainerProfile = _FakeCreateProfile(events, create_results),
            DeriveAppContainerSidFromAppContainerName = _FakeDeriveSid(events),
            DeleteAppContainerProfile = lambda moniker: (
                events.append(("delete-profile", moniker)) or 0
            ),
        ),
        advapi32 = SimpleNamespace(
            FreeSid = lambda value: events.append(("free-sid", getattr(value, "value", value)))
        ),
        kernel32 = SimpleNamespace(LocalFree = lambda _value: None),
    )
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)
    monkeypatch.setattr(windows_lpac, "_sid_string", lambda _api, _sid: sid_text)
    monkeypatch.setattr(windows_lpac, "_profile_folder", lambda _api, _sid: str(profile))
    monkeypatch.setattr(windows_lpac, "_manifest_root", lambda: str(manifests))
    monkeypatch.setattr(windows_lpac, "_process_identity", lambda pid = None: (1, 2))
    monkeypatch.setattr(windows_lpac, "_validate_workdir", lambda value: value)
    monkeypatch.setattr(windows_lpac, "_canonical_inner_argv", lambda argv, env: tuple(argv))
    monkeypatch.setattr(windows_lpac, "_runtime_roots", lambda _workdir, _argv: tuple(roots))
    monkeypatch.setattr(windows_lpac, "_needs_explicit_acl", lambda _path: True)
    monkeypatch.setattr(
        windows_lpac,
        "_validate_runtime_trees",
        lambda roots: events.append(("validate-runtime", tuple(roots))),
    )
    monkeypatch.setattr(windows_lpac, "_machine_wide", lambda path: "Program Files" in path)
    monkeypatch.setattr(windows_lpac, "_safe_environment", lambda env, *_a: dict(env))

    @contextmanager
    def fake_well_known(text):
        events.append(("ambient", text))
        yield ctypes.c_void_p(7)

    monkeypatch.setattr(windows_lpac, "_well_known_sid", fake_well_known)
    # A small model of the DACL: a grant makes the later access check succeed, a
    # revoke undoes it, and revoking a SID no ACE names is the no-op _revoke_sid
    # short-circuits to.
    acl: dict[tuple[str, int], bool] = {}

    def check(path, _sids, required):
        events.append(("check", path, required))
        return path in existing or acl.get((path, int(required)), False)

    monkeypatch.setattr(windows_lpac, "_existing_access", check)

    def grant(kind, required):
        def _grant(path, value, **_kwargs):
            events.append((kind, path))
            acl[(path, required)] = True

        return _grant

    read_execute = windows_lpac._FILE_GENERIC_READ | windows_lpac._FILE_GENERIC_EXECUTE
    monkeypatch.setattr(windows_lpac, "_grant_read_execute", grant("read_execute", read_execute))
    monkeypatch.setattr(windows_lpac, "_grant_modify", grant("modify", read_execute))
    monkeypatch.setattr(
        windows_lpac, "_grant_traverse", grant("traverse", windows_lpac._FILE_TRAVERSE)
    )

    def revoke(path, _value, *, exact = False):
        held = [key for key in acl if key[0] == path]
        if not held:
            events.append(("revoke-absent", path))
            return
        for key in held:
            acl.pop(key)
        events.append(("revoke", path, exact))

    monkeypatch.setattr(windows_lpac, "_revoke_sid", revoke)
    workdir = tmp_path / "chats" / "work"
    workdir.mkdir(parents = True)
    yield SimpleNamespace(
        backend = windows_lpac.WindowsLpacBackend(),
        spec = _spec(workdir),
        workdir = str(workdir),
        runtime = runtime,
        runtime_roots = tuple(roots),
        manifests = manifests,
        profile = profile,
        sid_text = sid_text,
        moniker = windows_lpac._install_moniker(),
        acl = acl,
    )


def _kinds(events, *names):
    return [event for event in events if event[0] in set(names)]


def test_install_moniker_is_stable_and_within_the_appcontainer_name_rules():
    first = windows_lpac._install_moniker()
    assert first == windows_lpac._install_moniker()
    assert first.startswith("unsloth.studio.sandbox.")
    assert len(first) <= 64
    assert re.fullmatch(r"[-_. A-Za-z0-9]+", first)
    digest = first[len("unsloth.studio.sandbox.") :]
    assert len(digest) == 16 and set(digest) <= set("0123456789abcdef")


def test_install_moniker_follows_the_interpreter_and_the_account(monkeypatch):
    baseline = windows_lpac._install_moniker()
    monkeypatch.setattr(windows_lpac.sys, "executable", str(Path(sys.executable).parent / "other"))
    assert windows_lpac._install_moniker() != baseline
    monkeypatch.undo()
    monkeypatch.setattr(windows_lpac.os, "getlogin", lambda: "somebody-else")
    assert windows_lpac._install_moniker() != baseline


def test_container_sid_text_accepts_only_derived_profile_sids():
    assert windows_lpac._is_container_sid_text("S-1-15-2-11-22-33-44-55-66-77")
    # The ambient package groups and any account SID must never name a manifest.
    assert not windows_lpac._is_container_sid_text("S-1-15-2-1")
    assert not windows_lpac._is_container_sid_text("S-1-15-2-2")
    assert not windows_lpac._is_container_sid_text("S-1-5-21-1-2-3-1001")
    assert not windows_lpac._is_container_sid_text("S-1-15-2-11-22-33-44-55-66-77-88")
    assert not windows_lpac._is_container_sid_text("S-1-15-2-a-b-c-d-e-f-g")
    assert not windows_lpac._is_container_sid_text(None)


def test_two_launches_in_one_process_share_the_profile_name_and_sid(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        first = fakes.backend.prepare(fakes.spec)
        second = fakes.backend.prepare(fakes.spec)
        one = first.spawn_callback._lpac_identity
        two = second.spawn_callback._lpac_identity
        assert one.moniker == two.moniker == windows_lpac._install_moniker()
        assert one.sid_string == two.sid_string == fakes.sid_text
        assert one.sid.value == two.sid.value
        # Per launch and never shared: the manifest. The container temp is
        # shared, because Windows points every child's TEMP at it, and it is
        # emptied only once the last launch holding it has finished.
        assert one.manifest_path != two.manifest_path
        assert one.private_temp == two.private_temp == str(fakes.profile / "Temp")
        assert Path(one.private_temp).is_dir()
        scratch = Path(one.private_temp) / "scratch"
        scratch.write_text("owned by the second launch", encoding = "utf-8")
        first.cleanup()
        assert scratch.is_file()  # the second launch is still running
        second.cleanup()
        assert first.cleanup_diagnostics == [] and second.cleanup_diagnostics == []
    assert [event for event in events if event[0] == "create-profile"] == [
        ("create-profile", one.moniker)
    ]
    # The profile outlives both launches.
    assert not [event for event in events if event[0] == "delete-profile"]
    assert Path(one.private_temp).is_dir()
    assert not list(Path(one.private_temp).iterdir())
    assert not Path(one.manifest_path).exists()


def test_an_existing_profile_is_reused_through_the_derived_sid(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(
        monkeypatch,
        tmp_path,
        events = events,
        create_results = [windows_lpac._HRESULT_ALREADY_EXISTS],
    ) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        identity = prepared.spawn_callback._lpac_identity
        prepared.cleanup()
    assert ("derive-sid", identity.moniker) in events
    assert identity.sid_string == fakes.sid_text
    assert not [event for event in events if event[0] == "delete-profile"]


def test_the_interpreter_is_granted_once_and_the_workdir_on_every_launch(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        first = fakes.backend.prepare(fakes.spec)
        first_events = list(events)
        first.cleanup()
        second = fakes.backend.prepare(fakes.spec)
        second.cleanup()
        runtime = fakes.runtime
        workdir = fakes.workdir
    assert ("read_execute", runtime) in first_events
    assert ("validate-runtime", (runtime,)) in first_events
    # The second launch does not touch the runtime tree at all: no DACL read, no
    # walk, no SetNamedSecurityInfoW.
    after_first = events[len(first_events) :]
    assert not [event for event in after_first if event[0] in {"read_execute", "validate-runtime"}]
    assert not [event for event in after_first if len(event) > 1 and event[1] == runtime]
    assert [event for event in events if event[0] == "modify" and event[1] == workdir] == [
        ("modify", workdir)
    ] * 2
    assert [event for event in events if event[0] == "revoke" and event[1] == workdir] == [
        ("revoke", workdir, False)
    ] * 2
    # The interpreter grant is never revoked.
    assert not [event for event in events if event[0] == "revoke" and event[1] == runtime]


def test_the_persistent_manifest_is_written_before_the_grant_and_kept_by_reconcile(
    monkeypatch, tmp_path
):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        manifest = fakes.manifests / (windows_lpac._install_moniker() + ".json")

        recorded_grant = windows_lpac._grant_read_execute

        def watched_grant(path, value, **kwargs):
            events.append(("read_execute", path, manifest.is_file()))
            recorded_grant(path, value, **kwargs)

        monkeypatch.setattr(windows_lpac, "_grant_read_execute", watched_grant)
        prepared = fakes.backend.prepare(fakes.spec)
        prepared.cleanup()
        assert ("read_execute", fakes.runtime, True) in events
        payload = json.loads(manifest.read_text(encoding = "utf-8"))
        assert payload["kind"] == "lpac-persistent"
        assert payload["moniker"] == windows_lpac._install_moniker()
        assert payload["interpreter"] == os.path.realpath(sys.executable)
        assert fakes.runtime in payload["granted_roots"]
        assert "owner_pid" not in payload

        fakes.backend.reconcile_stale_manifests()

        assert manifest.is_file()
        assert not [event for event in events if event[0] == "revoke" and event[1] == fakes.runtime]


def test_a_persistent_manifest_whose_interpreter_is_gone_is_revoked(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        prepared.cleanup()
        manifest = fakes.manifests / (windows_lpac._install_moniker() + ".json")
        payload = json.loads(manifest.read_text(encoding = "utf-8"))
        payload["interpreter"] = str(tmp_path / "removed" / "python.exe")
        manifest.write_text(json.dumps(payload), encoding = "utf-8")
        events.clear()

        fakes.backend.reconcile_stale_manifests()

        assert ("revoke", fakes.runtime, False) in events
        assert ("free-sid", 4242) in events
        assert not manifest.exists()
        # A persistent manifest owns no profile and no private temp.
        assert not [event for event in events if event[0] == "delete-profile"]


def test_remove_persistent_grants_revokes_everything_and_drops_the_profile(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        prepared.cleanup()
        manifest = fakes.manifests / (windows_lpac._install_moniker() + ".json")
        assert manifest.is_file()
        events.clear()

        removed = fakes.backend.remove_persistent_grants()

        assert removed == (windows_lpac._install_moniker(),)
        assert ("revoke", fakes.runtime, False) in events
        assert ("delete-profile", windows_lpac._install_moniker()) in events
        assert not manifest.exists()
        assert windows_lpac._INSTALL_PROFILE is None
        # A later launch builds the profile again and pays the grant once more.
        again = fakes.backend.prepare(fakes.spec)
        again.cleanup()
        assert ("read_execute", fakes.runtime) in events
        assert manifest.is_file()


def test_a_concurrent_launch_keeps_the_shared_workdir_grant(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        first = fakes.backend.prepare(fakes.spec)
        second = fakes.backend.prepare(fakes.spec)
        events.clear()
        first.cleanup()
        # Every launch of this installation carries the same SID, so the first
        # cleanup must not revoke an ACE the second launch is still using.
        assert not [event for event in events if event[1] == fakes.workdir]
        second.cleanup()
        assert ("revoke", fakes.workdir, False) in events


def test_a_stale_launch_manifest_is_reconciled_without_deleting_the_profile(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        live = fakes.backend.prepare(fakes.spec)
        identity = live.spawn_callback._lpac_identity
        manifest = Path(identity.manifest_path)
        assert manifest.name.startswith("unsloth.studio.launch.")
        payload = json.loads(manifest.read_text(encoding = "utf-8"))
        assert payload["kind"] == "lpac-launch"
        assert set(payload["granted_roots"]) == {identity.workdir, *payload["traverse_roots"]}
        assert identity.private_temp not in payload["granted_roots"]

        # What a Studio that crashed mid-launch left behind: the same workdir, a
        # private temp of its own, and an owner that is gone.
        stale_temp = fakes.profile / "Temp" / ("d" * 24)
        stale_temp.mkdir(parents = True)
        stale = fakes.manifests / f"unsloth.studio.launch.{'e' * 32}.json"
        stale.write_text(
            json.dumps(
                {
                    **payload,
                    "launch_id": "e" * 32,
                    "private_temp": str(stale_temp),
                    "owner_pid": 99,
                }
            ),
            encoding = "utf-8",
        )
        monkeypatch.setattr(
            windows_lpac, "_process_identity", lambda pid = None: None if pid == 99 else (1, 2)
        )
        events.clear()

        fakes.backend.reconcile_stale_manifests()

        # The crashed launch's own state is gone.
        assert not stale.exists()
        assert not stale_temp.exists()
        # A profile shared with this installation is never deleted, and the
        # workdir ACE a live launch is still using is not revoked with it.
        assert not [event for event in events if event[0] == "delete-profile"]
        assert not [event for event in events if event[0] == "revoke"]
        assert manifest.is_file()
        assert Path(identity.private_temp).is_dir()
        assert ("free-sid", 4242) in events

        # A manifest that names a running launch's private temp is left whole.
        impostor = fakes.manifests / f"unsloth.studio.launch.{'f' * 32}.json"
        impostor.write_text(
            json.dumps({**payload, "launch_id": "f" * 32, "owner_pid": 99}),
            encoding = "utf-8",
        )
        fakes.backend.reconcile_stale_manifests()
        assert impostor.is_file()
        assert Path(identity.private_temp).is_dir()

        live.cleanup()
        assert ("revoke", identity.workdir, False) in events
        assert not manifest.exists()


def test_a_stale_manifest_empties_the_container_temp_only_when_nothing_holds_it(
    monkeypatch, tmp_path
):
    """The temp a crashed Studio shared with this one is emptied, never removed.

    Windows points every AppContainer child of this installation at
    ``<profile>\\Temp``, so the directory belongs to the container and not to one
    launch: reconciliation may only clear its contents, and only once no live
    launch of this process is using it.
    """
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        live = fakes.backend.prepare(fakes.spec)
        identity = live.spawn_callback._lpac_identity
        payload = json.loads(Path(identity.manifest_path).read_text(encoding = "utf-8"))
        assert payload["private_temp"] == identity.private_temp == str(fakes.profile / "Temp")
        leftover = Path(identity.private_temp) / "crashed-launch-output"
        leftover.write_text("left behind", encoding = "utf-8")
        stale = fakes.manifests / f"unsloth.studio.launch.{'e' * 32}.json"
        stale.write_text(
            json.dumps({**payload, "launch_id": "e" * 32, "owner_pid": 99}), encoding = "utf-8"
        )
        monkeypatch.setattr(
            windows_lpac, "_process_identity", lambda pid = None: None if pid == 99 else (1, 2)
        )

        fakes.backend.reconcile_stale_manifests()

        # A live launch of this process holds the container temp, so the dead
        # owner's record is kept for the next reconciliation instead.
        assert stale.is_file() and leftover.is_file()

        live.cleanup()
        assert live.cleanup_diagnostics == []
        # The last live launch emptied it; the directory itself stays, because a
        # later launch's TEMP is this same path.
        assert Path(identity.private_temp).is_dir() and not leftover.exists()

        leftover.write_text("left behind", encoding = "utf-8")
        fakes.backend.reconcile_stale_manifests()
        assert not stale.exists()
        assert Path(identity.private_temp).is_dir() and not leftover.exists()


def test_another_live_studios_launch_keeps_its_grants_and_the_container_temp(
    monkeypatch, tmp_path
):
    """The refcount is per process; the SID and the profile are per installation.

    A second Studio process of the same installation grants the same ACEs to the
    same SID and shares the container temp, and its manifest is the only thing
    this process can see it through. Revoking on top of it would leave a running
    container unable to reach its own workdir.
    """
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        mine = fakes.backend.prepare(fakes.spec)
        identity = mine.spawn_callback._lpac_identity
        payload = json.loads(Path(identity.manifest_path).read_text(encoding = "utf-8"))
        # What the other Studio wrote before it granted anything: the same roots,
        # the same container temp, its own launch id, and an owner still alive.
        other = fakes.manifests / f"unsloth.studio.launch.{'a' * 32}.json"
        other.write_text(json.dumps({**payload, "launch_id": "a" * 32}), encoding = "utf-8")
        leftover = Path(identity.private_temp) / "the-other-launch-is-using-this"
        leftover.write_text("live", encoding = "utf-8")
        assert os.path.normcase(identity.workdir) in windows_lpac._live_launch_holds("a" * 32)
        events.clear()

        mine.cleanup()

        assert mine.cleanup_diagnostics == []
        # Nothing was revoked and nothing was emptied: the other launch is live.
        assert not [event for event in events if event[0] == "revoke"]
        assert leftover.is_file()
        # This launch's own record is gone, so the other one is now alone.
        assert not Path(identity.manifest_path).exists()
        assert windows_lpac._live_launch_holds(payload["launch_id"])

        # The installation-wide release refuses while that manifest is live, and
        # the refusal is decided under the lock the launch path holds.
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "still running"):
            fakes.backend.remove_persistent_grants()

        other.unlink()
        assert not windows_lpac._live_launch_holds()


def test_a_live_owner_keeps_its_launch_manifest(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        identity = prepared.spawn_callback._lpac_identity
        events.clear()

        fakes.backend.reconcile_stale_manifests()

        assert Path(identity.manifest_path).is_file()
        assert not [event for event in events if event[0] == "revoke"]
        prepared.cleanup()


def test_a_launch_never_revokes_what_the_installation_holds(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        first = fakes.backend.prepare(fakes.spec)
        persistent = json.loads(
            (fakes.manifests / (fakes.moniker + ".json")).read_text(encoding = "utf-8")
        )
        kept = {*persistent["granted_roots"], *persistent["traverse_roots"]}
        assert kept  # the runtime root and the ancestors it needed
        events.clear()
        first.cleanup()
        second = fakes.backend.prepare(fakes.spec)
        second.cleanup()

    revoked = {event[1] for event in events if event[0] == "revoke"}
    # Revoking one of these would leave the container unable to reach the
    # interpreter, or its own profile, on the very next launch.
    assert revoked.isdisjoint(kept), sorted(revoked & kept)
    assert fakes.workdir in revoked


def test_the_container_own_storage_is_never_granted_or_revoked(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        identity = prepared.spawn_callback._lpac_identity
        prepared.cleanup()
    package = str(fakes.profile.parent)
    touched = [
        event
        for event in events
        if event[0] in {"read_execute", "traverse", "revoke"}
        and (event[1] == package or event[1].startswith(package + os.sep))
    ]
    # Windows puts the package SID on this directory itself. The only thing this
    # module does inside it is grant the launch its own temp, and delete it.
    assert touched == [], touched
    assert ("modify", identity.private_temp) in events
    assert windows_lpac._container_owned(str(fakes.profile / "Temp"), str(fakes.profile))
    assert not windows_lpac._container_owned(str(tmp_path / "Packages"), str(fakes.profile))


def test_a_new_runtime_root_is_granted_once_and_appended_to_the_manifest(monkeypatch, tmp_path):
    events: list[tuple] = []
    shell = str(tmp_path / "Program Files" / "Git" / "bin")
    os.makedirs(shell)
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        first = fakes.backend.prepare(fakes.spec)
        first.cleanup()
        manifest = fakes.manifests / (fakes.moniker + ".json")
        assert json.loads(manifest.read_text(encoding = "utf-8"))["granted_roots"] == [
            fakes.runtime
        ]
        # A Terminal call reaching Git bash after a Python call.
        monkeypatch.setattr(
            windows_lpac, "_runtime_roots", lambda _workdir, _argv: (fakes.runtime, shell)
        )
        events.clear()
        second = fakes.backend.prepare(fakes.spec)
        second.cleanup()
        third = fakes.backend.prepare(fakes.spec)
        third.cleanup()

    assert [event for event in events if event[0] == "read_execute"] == [("read_execute", shell)]
    assert ("validate-runtime", (shell,)) in events
    assert json.loads(manifest.read_text(encoding = "utf-8"))["granted_roots"] == [
        fakes.runtime,
        shell,
    ]


def test_a_persistent_manifest_whose_sid_is_not_its_moniker_is_left_alone(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        prepared.cleanup()
        manifest = fakes.manifests / (fakes.moniker + ".json")
        payload = json.loads(manifest.read_text(encoding = "utf-8"))
        payload["interpreter"] = str(tmp_path / "removed" / "python.exe")
        payload["sid"] = "S-1-15-2-99-88-77-66-55-44-33"
        manifest.write_text(json.dumps(payload), encoding = "utf-8")
        events.clear()

        fakes.backend.reconcile_stale_manifests()

    # The moniker derives to a different SID, so nothing may be revoked for it.
    assert not [event for event in events if event[0] == "revoke"]
    assert manifest.is_file()


def test_the_access_memo_follows_the_root_and_never_caches_a_missing_one(monkeypatch, tmp_path):
    monkeypatch.setattr(windows_lpac, "_ACCESS_MEMO", {})
    root = tmp_path / "runtime"
    root.mkdir()
    calls: list[str] = []

    def check(path, _sids, _required):
        calls.append(path)
        return True

    monkeypatch.setattr(windows_lpac, "_existing_access", check)
    read = windows_lpac._FILE_GENERIC_READ

    def resolve(path):
        return windows_lpac._memoized_existing_access(
            str(path), (), read, sid_text = "S-1-15-2-1-2-3-4-5-6-7", ambient_text = "S-1-15-2-2"
        )

    assert resolve(root) is True
    assert resolve(root) is True
    assert calls == [str(root)]
    # An interpreter upgrade writes into the root directory. A Windows last
    # write time is stamped from the system clock, which advances about every
    # 15.6 ms, so a write this soon after the mkdir leaves the directory on the
    # very stamp the memo pinned and the memo answers from its cache. Move the
    # stamp by hand, so what this asserts is the memo following the root and not
    # a filesystem's timestamp resolution.
    (root / "python.exe").write_text("new", encoding = "utf-8")
    stamped = os.stat(root)
    os.utime(root, ns = (stamped.st_atime_ns, stamped.st_mtime_ns + 2_000_000_000))
    assert os.stat(root).st_mtime_ns != stamped.st_mtime_ns
    assert resolve(root) is True
    assert calls == [str(root), str(root)]
    # A root that cannot be stat'ed is asked about every time and never recorded.
    missing = tmp_path / "gone"
    assert resolve(missing) is True
    assert resolve(missing) is True
    assert calls[-2:] == [str(missing), str(missing)]
    assert not [key for key in windows_lpac._ACCESS_MEMO if key[0] == str(missing).lower()]
    assert not [key for key in windows_lpac._ACCESS_MEMO if key[0] == str(missing)]


def test_the_access_memo_expires_so_an_external_acl_change_is_seen(monkeypatch, tmp_path):
    monkeypatch.setattr(windows_lpac, "_ACCESS_MEMO", {})
    root = tmp_path / "runtime"
    root.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(
        windows_lpac, "_existing_access", lambda path, _s, _r: calls.append(path) or True
    )

    def resolve():
        return windows_lpac._memoized_existing_access(
            str(root),
            (),
            windows_lpac._FILE_GENERIC_READ,
            sid_text = "S-1-15-2-1-2-3-4-5-6-7",
            ambient_text = "S-1-15-2-2",
        )

    assert resolve() and resolve()
    assert calls == [str(root)]
    # Another Studio process of this installation, icacls, or a repair install
    # can drop the grant without touching a single timestamp, so the answer has
    # to go stale on its own.
    monkeypatch.setattr(windows_lpac, "_ACCESS_MEMO_SECONDS", 0.0)
    assert resolve()
    assert calls == [str(root), str(root)]



def test_a_manifest_write_survives_a_leftover_temporary_file(tmp_path):
    target = str(tmp_path / "unsloth.studio.sandbox.0123456789abcdef.json")
    windows_lpac._atomic_write_manifest(target, {"version": 1, "round": 1})
    # The crash case: temporary files a previous write never replaced, including
    # the name a fixed-suffix writer would reach for. That writer would fail here
    # with FileExistsError and never be able to update this manifest again.
    for leftover in (target + ".tmp", target + ".abcd1234.tmp"):
        Path(leftover).write_text("{}", encoding = "utf-8")
    windows_lpac._atomic_write_manifest(target, {"version": 1, "round": 2})
    assert json.loads(Path(target).read_text(encoding = "utf-8"))["round"] == 2
    # Only what was planted is left; the successful write cleaned up after itself.
    assert len(list(tmp_path.glob("*.tmp"))) == 2


def test_a_profile_whose_storage_was_deleted_is_created_again(monkeypatch, tmp_path):
    events: list[tuple] = []
    folders = [str(tmp_path / "gone" / "AC"), str(tmp_path / "Packages" / "owned" / "AC")]
    os.makedirs(folders[1])
    monkeypatch.setattr(windows_lpac, "_INSTALL_PROFILE", None)
    api = SimpleNamespace(
        userenv = SimpleNamespace(
            CreateAppContainerProfile = _FakeCreateProfile(events),
            DeriveAppContainerSidFromAppContainerName = _FakeDeriveSid(events),
            DeleteAppContainerProfile = lambda moniker: (
                events.append(("delete-profile", moniker)) or 0
            ),
        ),
        advapi32 = SimpleNamespace(
            FreeSid = lambda value: events.append(("free-sid", value.value))
        ),
    )
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)
    monkeypatch.setattr(windows_lpac, "_sid_string", lambda _api, _sid: "S-1-15-2-1-2-3-4-5-6-7")
    monkeypatch.setattr(windows_lpac, "_profile_folder", lambda _api, _sid: folders.pop(0))

    install = windows_lpac._install_profile()

    # Registered but with no storage: only CreateAppContainerProfile puts the
    # package ACL on that directory, so the registration is dropped and remade.
    moniker = windows_lpac._install_moniker()
    assert [event for event in events if event[0] != "free-sid"] == [
        ("create-profile", moniker),
        ("delete-profile", moniker),
        ("create-profile", moniker),
    ]
    assert install.profile_folder == str(tmp_path / "Packages" / "owned" / "AC")


def _planted(tmp_path, name, payload):
    (tmp_path / name).write_text(json.dumps(payload), encoding = "utf-8")
    return windows_lpac._parse_manifest(tmp_path / name)


def test_manifest_parsing_refuses_planted_records(tmp_path):
    single_use, launch = _manifest_payloads(tmp_path)
    persistent = {
        "version": 1,
        "kind": "lpac-persistent",
        "moniker": _INSTALL_MONIKER,
        "sid": _CONTAINER_SID,
        "profile_folder": str(tmp_path / "profile"),
        "interpreter": str(tmp_path / "python.exe"),
        "granted_roots": [str(tmp_path / "runtime")],
        "traverse_roots": [str(tmp_path)],
    }
    launch_name = f"unsloth.studio.launch.{'b' * 32}.json"
    assert _planted(tmp_path, launch_name, launch)["kind"] == "lpac-launch"
    # The container temp itself is what a launch owns now: Windows points the
    # child's TEMP there, so reconciliation empties it instead of removing a
    # per-launch subdirectory. The subdirectory shape an earlier build wrote
    # (``launch`` above) still parses, so its manifest reconciles.
    container_temp = str(Path(launch["profile_folder"]) / "Temp")
    assert (
        _planted(tmp_path, launch_name, {**launch, "private_temp": container_temp})["kind"]
        == "lpac-launch"
    )
    assert _planted(tmp_path, "unsloth.studio.valid.json", single_use)["kind"] == "lpac-single-use"
    assert _planted(tmp_path, _INSTALL_MONIKER + ".json", persistent)["kind"] == "lpac-persistent"

    victim = str(tmp_path / "victim")
    rejected = {
        # The ambient package groups: revoking them would strip every application
        # package's access to whatever the manifest names.
        "ambient sid": (launch_name, {**launch, "sid": "S-1-15-2-1"}),
        "account sid": (launch_name, {**launch, "sid": "S-1-5-21-1-2-3-1001"}),
        "child container sid": (
            launch_name,
            {**launch, "sid": "S-1-15-2-1-2-3-4-5-6-7-8-9-10-11"},
        ),
        # A traverse root that is not an ancestor of anything this launch owns,
        # smuggled in so that the granted-roots set equality still holds.
        "traverse root that is not an ancestor": (
            launch_name,
            {
                **launch,
                "granted_roots": [*launch["granted_roots"], victim],
                "traverse_roots": [*launch["traverse_roots"], victim],
            },
        ),
        "granted root nobody owns": (
            launch_name,
            {**launch, "granted_roots": [*launch["granted_roots"], victim]},
        ),
        # The container temp of a different profile: emptying it would take a
        # foreign installation's live launches with it.
        "private temp is another profile's temp": (
            launch_name,
            {
                **launch,
                "private_temp": str(tmp_path / "profile" / "Temp"),
                "granted_roots": launch["granted_roots"],
            },
        ),
        "private temp outside the profile": (
            launch_name,
            {**launch, "private_temp": str(tmp_path / ("c" * 24))},
        ),
        "launch id that is not the file name": (
            launch_name,
            {**launch, "launch_id": "a" * 32},
        ),
        "launch id of the wrong length": (
            f"unsloth.studio.launch.{'b' * 30}.json",
            {**launch, "launch_id": "b" * 30},
        ),
        "launch moniker outside the installation namespace": (
            launch_name,
            {**launch, "moniker": "unsloth.studio.other"},
        ),
        "single use manifest wearing a kind": (
            "unsloth.studio.valid.json",
            {**single_use, "kind": "lpac-launch"},
        ),
        "single use manifest named after another moniker": (
            "unsloth.studio.other.json",
            single_use,
        ),
        "persistent manifest named after another moniker": (
            "unsloth.studio.sandbox.fedcba9876543210.json",
            persistent,
        ),
        "persistent manifest without an interpreter": (
            _INSTALL_MONIKER + ".json",
            {**persistent, "interpreter": None},
        ),
        "persistent manifest with a relative interpreter": (
            _INSTALL_MONIKER + ".json",
            {**persistent, "interpreter": "python.exe"},
        ),
        "unknown kind": (launch_name, {**launch, "kind": "lpac-something-else"}),
        "wrong version": (launch_name, {**launch, "version": 2}),
        "relative granted root": (
            launch_name,
            {**launch, "granted_roots": ["relative"]},
        ),
    }
    for label, (name, payload) in rejected.items():
        assert _planted(tmp_path, name, payload) is None, label


def test_a_cleanup_cannot_revoke_between_a_concurrent_launch_check_and_its_grant(
    monkeypatch, tmp_path
):
    """The ledger is only worth anything if the revoke cannot slip past a grant.

    Two chats at once: one launch's cleanup releases the ancestor both share, the
    other launch starts, reads the ACE as present and skips its own grant, and the
    first launch then revokes it. The second container is left unable to reach its
    own workdir. The window is real without a lock spanning release and revoke, so
    the second launch is driven from another thread while the first is inside its
    revoke.
    """
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        first = fakes.backend.prepare(fakes.spec)
        entered = threading.Event()
        second: list = []
        failures: list[BaseException] = []

        def run_second():
            entered.wait(10)
            try:
                second.append(fakes.backend.prepare(fakes.spec))
            except BaseException as exc:  # noqa: BLE001 - reported below
                failures.append(exc)

        recorded_revoke = windows_lpac._revoke_sid

        def slow_revoke(path, value, *, exact = False):
            # Wide open: the other launch gets its whole prepare in here.
            entered.set()
            time.sleep(0.4)
            return recorded_revoke(path, value, exact = exact)

        worker = threading.Thread(target = run_second)
        worker.start()
        try:
            monkeypatch.setattr(windows_lpac, "_revoke_sid", slow_revoke)
            first.cleanup()
        finally:
            entered.set()
            worker.join(20)
        assert not failures, failures
        assert second, "the concurrent launch never completed"

        prepared = second[0]
        workdir_key = os.path.normcase(fakes.workdir)
        assert windows_lpac._SHARED_GRANTS.get(workdir_key) == 1
        # Whatever order the two took, the live launch's workdir is granted.
        assert fakes.acl.get(
            (fakes.workdir, windows_lpac._FILE_GENERIC_READ | windows_lpac._FILE_GENERIC_EXECUTE)
        ), events
        prepared.cleanup()
        assert not fakes.acl.get(
            (fakes.workdir, windows_lpac._FILE_GENERIC_READ | windows_lpac._FILE_GENERIC_EXECUTE)
        )


def test_a_launch_manifest_may_not_name_the_container_own_storage(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        identity = prepared.spawn_callback._lpac_identity
        payload = json.loads(Path(identity.manifest_path).read_text(encoding = "utf-8"))
        prepared.cleanup()
        package = str(fakes.profile.parent)
        # The one path guaranteed to carry this SID, and the one Windows alone can
        # put back. A propagating revoke here would end every later launch.
        planted = fakes.manifests / f"unsloth.studio.launch.{'a' * 32}.json"
        planted.write_text(
            json.dumps(
                {
                    **payload,
                    "launch_id": "a" * 32,
                    "workdir": package,
                    "granted_roots": [package],
                    "traverse_roots": [],
                    "owner_pid": 99,
                }
            ),
            encoding = "utf-8",
        )
        assert windows_lpac._parse_manifest(planted) is None
        monkeypatch.setattr(
            windows_lpac, "_process_identity", lambda pid = None: None if pid == 99 else (1, 2)
        )
        events.clear()

        fakes.backend.reconcile_stale_manifests()

        assert not [event for event in events if event[0] == "revoke"]


def test_a_launch_manifest_may_not_release_what_the_installation_owns(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        identity = prepared.spawn_callback._lpac_identity
        payload = json.loads(Path(identity.manifest_path).read_text(encoding = "utf-8"))
        prepared.cleanup()
        # A workdir is only checked for being absolute, so a planted manifest can
        # name the interpreter tree. The persistent record is what says that this
        # is not a launch's to revoke.
        planted = fakes.manifests / f"unsloth.studio.launch.{'a' * 32}.json"
        planted.write_text(
            json.dumps(
                {
                    **payload,
                    "launch_id": "a" * 32,
                    "workdir": fakes.runtime,
                    "granted_roots": [fakes.runtime],
                    "traverse_roots": [],
                    "owner_pid": 99,
                }
            ),
            encoding = "utf-8",
        )
        assert windows_lpac._parse_manifest(planted) is not None
        monkeypatch.setattr(
            windows_lpac, "_process_identity", lambda pid = None: None if pid == 99 else (1, 2)
        )
        events.clear()

        fakes.backend.reconcile_stale_manifests()

        assert not [event for event in events if event[0] == "revoke"]
        assert not planted.exists()  # the record is still cleaned up
        assert fakes.acl.get(
            (fakes.runtime, windows_lpac._FILE_GENERIC_READ | windows_lpac._FILE_GENERIC_EXECUTE)
        )


def test_a_renamed_installation_releases_its_previous_identity(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        prepared.cleanup()
        previous = fakes.manifests / (fakes.moniker + ".json")
        payload = json.loads(previous.read_text(encoding = "utf-8"))
        assert payload["interpreter"] == os.path.realpath(sys.executable)

        # The machine is renamed: same interpreter, same account, new moniker.
        monkeypatch.setattr(windows_lpac.platform, "node", lambda: "renamed-host")
        monkeypatch.setattr(windows_lpac, "_INSTALL_PROFILE", None)
        assert windows_lpac._install_moniker() != fakes.moniker
        events.clear()

        fakes.backend.reconcile_stale_manifests()

        # Nothing else would ever release an inheritable grant on a tree that is
        # still there, under a name this installation no longer answers to.
        assert ("revoke", fakes.runtime, False) in events
        assert not previous.exists()


def test_another_installation_persistent_grant_is_left_alone(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        other = fakes.manifests / "unsloth.studio.sandbox.fedcba9876543210.json"
        other.write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "lpac-persistent",
                    "moniker": "unsloth.studio.sandbox.fedcba9876543210",
                    "sid": "S-1-15-2-99-88-77-66-55-44-33",
                    "profile_folder": str(tmp_path / "Packages" / "other" / "AC"),
                    "interpreter": str(tmp_path / "other-python" / "python.exe"),
                    "granted_roots": [str(tmp_path / "other-python")],
                    "traverse_roots": [],
                }
            ),
            encoding = "utf-8",
        )
        os.makedirs(tmp_path / "other-python")
        (tmp_path / "other-python" / "python.exe").write_text("", encoding = "utf-8")
        events.clear()

        fakes.backend.reconcile_stale_manifests()

        assert other.is_file()
        assert not [event for event in events if event[0] == "revoke"]
        # And it is not this installation's to release either.
        assert fakes.backend.remove_persistent_grants() == ()
        assert other.is_file()


def test_the_container_cannot_be_released_while_a_launch_is_running(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "still running"):
            fakes.backend.remove_persistent_grants()
        assert (fakes.manifests / (fakes.moniker + ".json")).is_file()
        prepared.cleanup()
        assert fakes.backend.remove_persistent_grants() == (fakes.moniker,)


def test_a_profile_directory_that_cannot_be_rebuilt_refuses_the_launch(monkeypatch, tmp_path):
    events: list[tuple] = []
    monkeypatch.setattr(windows_lpac, "_INSTALL_PROFILE", None)
    api = SimpleNamespace(
        userenv = SimpleNamespace(
            CreateAppContainerProfile = _FakeCreateProfile(events),
            DeriveAppContainerSidFromAppContainerName = _FakeDeriveSid(events),
            # The delete fails, which is what another process holding the profile
            # looks like, so the second attempt finds it registered again.
            DeleteAppContainerProfile = lambda moniker: (
                events.append(("delete-profile", moniker)) or 0
            ),
        ),
        advapi32 = SimpleNamespace(
            FreeSid = lambda value: events.append(("free-sid", value.value))
        ),
    )
    monkeypatch.setattr(windows_lpac, "_api", lambda: api)
    monkeypatch.setattr(windows_lpac, "_sid_string", lambda _api, _sid: "S-1-15-2-1-2-3-4-5-6-7")
    monkeypatch.setattr(windows_lpac, "_profile_folder", lambda _api, _sid: str(tmp_path / "gone"))

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "profile directory is missing"):
        windows_lpac._install_profile()

    # Never cached: os.makedirs would otherwise rebuild the container's storage
    # by hand, without the package ACE only Windows can put there.
    assert windows_lpac._INSTALL_PROFILE is None
    assert [event[0] for event in events].count("create-profile") == 2


def test_a_reconciled_identity_frees_its_sid_even_when_cleanup_fails(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        identity = prepared.spawn_callback._lpac_identity
        manifest = Path(identity.manifest_path)
        payload = json.loads(manifest.read_text(encoding = "utf-8"))
        prepared.cleanup()
        manifest.write_text(json.dumps({**payload, "owner_pid": 99}), encoding = "utf-8")
        os.makedirs(identity.private_temp, exist_ok = True)
        monkeypatch.setattr(
            windows_lpac, "_process_identity", lambda pid = None: None if pid == 99 else (1, 2)
        )
        monkeypatch.setattr(
            windows_lpac,
            "_revoke_sid",
            lambda path, value, *, exact = False: (_ for _ in ()).throw(OSError(5, "denied")),
        )
        events.clear()

        fakes.backend.reconcile_stale_manifests()

        # The record is kept for the next probe, which derives the SID again, so
        # the failing path must not leak one allocation per probe.
        assert manifest.is_file()
        assert [event for event in events if event[0] == "free-sid"] == [("free-sid", 4242)]


def test_prepare_skips_grants_the_dacl_already_covers(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(
        monkeypatch, tmp_path, events = events, existing = {str(tmp_path)}
    ) as fakes:
        prepared = fakes.backend.prepare(fakes.spec)
        identity = prepared.spawn_callback._lpac_identity
        prepared.cleanup()
    kinds = _kinds(events, "read_execute", "traverse", "modify")
    assert ("read_execute", fakes.runtime) in kinds
    assert ("traverse", str(tmp_path / "Program Files")) in kinds
    # USERPROFILE is already covered by the ambient package group, so it is
    # neither granted nor revoked.
    assert ("traverse", str(tmp_path)) not in kinds
    assert not [event for event in events if event[0] == "revoke" and event[1] == str(tmp_path)]
    # What was granted for the installation stays granted after the launch ends.
    assert not [
        event
        for event in events
        if event[0] == "revoke" and event[1] in (fakes.runtime, str(tmp_path / "Program Files"))
    ]
    ambient = [event for event in events if event[0] == "ambient"]
    assert ambient[0] == ("ambient", "S-1-15-2-2")
    assert identity.unverified_access == ()
    assert identity.profile == "lpac"
    assert prepared.backend == "windows-lpac"


def test_prepare_records_access_denied_on_machine_wide_paths_instead_of_failing(
    monkeypatch, tmp_path
):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        runtime = fakes.runtime
        monkeypatch.setattr(
            windows_lpac,
            "_grant_read_execute",
            lambda path, value, **_k: (_ for _ in ()).throw(OSError(5, "denied")),
        )
        monkeypatch.setattr(
            windows_lpac,
            "_grant_traverse",
            lambda path, value, **_k: events.append(("traverse", path))
            or (
                (_ for _ in ()).throw(OSError(5, "denied"))
                if "Program Files" in path
                else None
            ),
        )
        fakes.backend._profile = "appcontainer"
        prepared = fakes.backend.prepare(fakes.spec)
        identity = prepared.spawn_callback._lpac_identity
        assert set(identity.unverified_access) == {runtime, str(tmp_path / "Program Files")}
        assert identity.profile == "appcontainer"
        ambient = [event for event in events if event[0] == "ambient"]
        assert ambient[0] == ("ambient", "S-1-15-2-1")
        assert prepared.spawn_callback._lpac_identity is identity
        # A refusal Windows owns is recorded once, not retried on every launch.
        events.clear()
        second = fakes.backend.prepare(fakes.spec)
        assert not [
            event for event in events if event[0] in {"read_execute", "validate-runtime"}
        ]
        # It is not retried, and it is still disclosed: a launch must not report a
        # tree as verified because an earlier launch was refused on it.
        assert set(second.spawn_callback._lpac_identity.unverified_access) == {
            runtime,
            str(tmp_path / "Program Files"),
        }
        prepared.cleanup()
        second.cleanup()


def test_prepare_still_fails_when_a_user_owned_grant_is_denied(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        monkeypatch.setattr(windows_lpac, "_machine_wide", lambda _path: False)
        monkeypatch.setattr(
            windows_lpac,
            "_grant_read_execute",
            lambda path, value, **_k: (_ for _ in ()).throw(OSError(5, "denied")),
        )
        with pytest.raises(OSError, match = "denied"):
            fakes.backend.prepare(fakes.spec)
        # The failure is before any per-launch state exists, so nothing is held.
        assert not windows_lpac._SHARED_GRANTS
        assert not list(fakes.manifests.glob("unsloth.studio.launch.*"))


def test_prepare_releases_its_hold_when_the_workdir_grant_fails(monkeypatch, tmp_path):
    events: list[tuple] = []
    with _lpac_fakes(monkeypatch, tmp_path, events = events) as fakes:
        monkeypatch.setattr(
            windows_lpac,
            "_grant_modify",
            lambda path, value, **_k: (_ for _ in ()).throw(OSError(5, "denied")),
        )
        with pytest.raises(OSError, match = "denied"):
            fakes.backend.prepare(fakes.spec)
        assert not windows_lpac._SHARED_GRANTS
        assert not list(fakes.manifests.glob("unsloth.studio.launch.*"))
        assert not list((fakes.profile / "Temp").iterdir())


def test_probe_falls_back_to_appcontainer_when_lpac_cannot_start_python(monkeypatch):
    backend = windows_lpac.WindowsLpacBackend()
    calls: list[str] = []

    def probe_profile(profile):
        calls.append(profile)
        if profile == "lpac":
            return (-1073741790, "the LPAC live probe failed (-1073741790): STATUS_ACCESS_DENIED")
        backend._last_probe_limitations = ()
        return None

    monkeypatch.setattr(windows_lpac, "_is_windows", lambda: True)
    monkeypatch.setattr(windows_lpac, "_api", lambda: SimpleNamespace())
    monkeypatch.setattr(backend, "reconcile_stale_manifests", lambda: None)
    monkeypatch.setattr(backend, "_probe_profile", probe_profile)

    capability = backend.probe()

    assert calls == ["lpac", "appcontainer"]
    assert capability.available is True and capability.qualified is True
    assert capability.protection_state == "preview"
    assert capability.profile_id == "windows-appcontainer-preview-v1"
    assert capability.limitations == (
        "all_application_packages_ambient_read",
        "null_device_and_named_pipes_denied",
        "concurrent_launches_share_the_container",
    )
    assert "AppContainer fallback passed" in capability.reason
    assert backend.active_profile == "appcontainer"
    assert backend.active_profile_id() == "windows-appcontainer-preview-v1"
    assert backend.profile_id == "windows-appcontainer-preview-v1"
    assert windows_lpac.WindowsLpacBackend.profile_id == "windows-lpac-preview-v1"


def test_probe_does_not_fall_back_for_other_failures_and_reports_both(monkeypatch):
    backend = windows_lpac.WindowsLpacBackend()
    backend._profile = "appcontainer"  # a stale fallback from a previous fingerprint
    calls: list[str] = []

    def probe_profile(profile):
        calls.append(profile)
        return (1, f"the {profile} live probe failed (1): boom")

    monkeypatch.setattr(windows_lpac, "_is_windows", lambda: True)
    monkeypatch.setattr(windows_lpac, "_api", lambda: SimpleNamespace())
    monkeypatch.setattr(backend, "reconcile_stale_manifests", lambda: None)
    monkeypatch.setattr(backend, "_probe_profile", probe_profile)

    capability = backend.probe()

    assert calls == ["lpac"]
    assert capability.available is False
    assert backend.active_profile == "lpac"

    def probe_profile_denied(profile):
        calls.append(profile)
        return (-1073741515, f"the {profile} live probe failed: dll")

    monkeypatch.setattr(backend, "_probe_profile", probe_profile_denied)
    capability = backend.probe()
    assert calls[-2:] == ["lpac", "appcontainer"]
    assert capability.available is False
    assert "AppContainer fallback probe also failed" in capability.reason
    assert backend.active_profile == "lpac"


def test_probe_success_under_lpac_keeps_the_strong_profile(monkeypatch):
    backend = windows_lpac.WindowsLpacBackend()
    monkeypatch.setattr(windows_lpac, "_is_windows", lambda: True)
    monkeypatch.setattr(windows_lpac, "_api", lambda: SimpleNamespace())
    monkeypatch.setattr(backend, "reconcile_stale_manifests", lambda: None)
    backend._last_probe_limitations = ("ipv6_unavailable_on_host",)
    monkeypatch.setattr(backend, "_probe_profile", lambda profile: None)
    capability = backend.probe()
    assert capability.profile_id == "windows-lpac-preview-v1"
    assert capability.limitations == (
        "null_device_and_named_pipes_denied",
        "concurrent_launches_share_the_container",
        "ipv6_unavailable_on_host",
    )
    assert backend.active_profile == "lpac"
    assert backend.profile_id == "windows-lpac-preview-v1"


def test_probe_payload_asserts_the_token_kind_for_each_profile():
    lpac = windows_lpac._probe_payload("wd", "ext", "S-1-15-2-1", [], less_privileged = True)
    plain = windows_lpac._probe_payload("wd", "ext", "S-1-15-2-1", [], less_privileged = False)
    assert "assert token_dword(46) == 1" in lpac
    assert "assert token_dword_or_zero(46) == 0" in plain
    # Zero capabilities and the file policy are asserted under both kinds.
    for payload in (lpac, plain):
        assert "assert token_dword(29) == 1" in payload
        assert "ctypes.POINTER(wintypes.DWORD)).contents.value == 0" in payload
        assert "LPAC escaped file policy" in payload


def test_network_probe_skips_ipv6_when_the_host_cannot_bind_it(monkeypatch):
    real_socket = socket.socket

    class NoIpv6Socket(real_socket):
        def bind(self, address):
            if self.family == socket.AF_INET6:
                raise OSError(99, "Cannot assign requested address")
            return super().bind(address)

    monkeypatch.setattr(socket, "socket", NoIpv6Socket)
    with windows_lpac._probe_network_endpoints() as endpoints:
        assert len(endpoints) == 2
        assert {family for family, _kind, _address in endpoints} == {int(socket.AF_INET)}
        assert endpoints.limitations == ("ipv6_unavailable_on_host",)


def test_network_probe_reports_no_limitation_when_ipv6_binds():
    try:
        with socket.socket(socket.AF_INET6) as check:
            check.bind(("::1", 0))
    except OSError:
        pytest.skip("this host has no IPv6 loopback")
    with windows_lpac._probe_network_endpoints() as endpoints:
        assert len(endpoints) == 4
        assert endpoints.limitations == ()


def test_environment_keeps_the_trusted_git_directory(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    private = tmp_path / "profile" / "Temp"
    git_dir = tmp_path / "Program Files" / "Git" / "cmd"
    workdir.mkdir()
    private.mkdir(parents = True)
    git_dir.mkdir(parents = True)
    identity = SimpleNamespace(private_temp = str(private), profile_folder = str(private.parent))
    monkeypatch.setattr(inference_tools, "_resolve_trusted_windows_git", lambda: (str(git_dir), ".exe"))
    safe = windows_lpac._safe_environment({}, str(workdir), identity, (sys.executable,))
    assert safe["PATH"].split(os.pathsep)[-1] == str(git_dir)

    monkeypatch.setattr(inference_tools, "_resolve_trusted_windows_git", lambda: ("", ""))
    safe = windows_lpac._safe_environment({}, str(workdir), identity, (sys.executable,))
    assert str(git_dir) not in safe["PATH"]


def test_spawn_attribute_count_follows_the_profile():
    source = Path(windows_lpac.__file__).read_text(encoding = "utf-8")
    # Security capabilities, the optional AAP opt-out, the handle list, plus the
    # Job Object attached at creation.
    assert "attribute_count = (3 if less_privileged else 2) + 1" in source
    assert "_PROC_THREAD_ATTRIBUTE_JOB_LIST" in source[source.index("def _spawn_lpac") :]
    spawn = source[source.index("def _spawn_lpac") :]
    assert spawn.index("job = _job_object_with_limits()") < spawn.index("CreateProcessW(")
    assert spawn.index("AssignProcessToJobObject") < spawn.index("ResumeThread(")
    assert 'less_privileged = identity.profile != _PROFILE_APPCONTAINER' in source


def test_limited_windows_job_requests_kill_on_close_and_is_terminated_after_drain():
    source = Path(inference_tools.__file__).read_text(encoding = "utf-8")
    assert "LimitFlags = 0x2 | 0x8 | 0x100 | 0x200 | 0x2000" in source
    call = "        _terminate_limited_windows_job(pgid, effective_execution_mode)\n"
    assert source.count(call) == 2  # _python_exec and _bash_exec, right after the drain
    events = []
    job = SimpleNamespace(terminate = lambda: events.append("terminated") or True)
    inference_tools._terminate_limited_windows_job(("windows-job", job), "limited")
    inference_tools._terminate_limited_windows_job(("windows-job", job), "os_isolation_required")
    inference_tools._terminate_limited_windows_job(None, "limited")
    inference_tools._terminate_limited_windows_job(("windows-tree", 1, None), "limited")
    assert events == ["terminated"]


@pytest.fixture(scope = "module")
def live_lpac_backend():
    if sys.platform != "win32":
        pytest.skip("native LPAC tests run only on Windows")
    capability = os_sandbox.capability_snapshot(force = True)
    assert capability.available is True, capability.reason
    assert capability.qualified is True, capability.reason
    assert capability.backend == "windows-lpac"
    assert capability.protection_state == "preview"
    backend = os_sandbox._platform_backend()
    assert isinstance(backend, windows_lpac.WindowsLpacBackend)
    assert capability.profile_id == backend.active_profile_id()
    if backend.active_profile == "appcontainer":
        assert capability.profile_id == windows_lpac._APPCONTAINER_PROFILE_ID
        assert "AppContainer fallback passed" in capability.reason
        assert windows_lpac._LIMITATION_AMBIENT_READ in capability.limitations
    else:
        assert capability.profile_id == windows_lpac.WindowsLpacBackend.profile_id
        assert "zero-capability LPAC live enforcement probe passed" in capability.reason
    print(f"live Windows container profile: {backend.active_profile}")
    yield backend
    # The interpreter grant and the container profile are meant to outlive a
    # launch, so a test run that does not release them leaves read+execute ACEs
    # for this installation on the runner. This is also the production caller of
    # the uninstall path.
    released = backend.remove_persistent_grants()
    assert released == (windows_lpac._install_moniker(),), released
    assert not Path(windows_lpac._persistent_manifest_path(released[0])).exists()


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
        # The container temp is shared with every launch of this installation,
        # so cleanup empties it and leaves the directory itself in place.
        assert private_temp.is_dir() and not list(private_temp.iterdir())
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


def test_live_launches_share_one_profile_and_own_their_manifest_and_temp(
    live_lpac_backend, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    first = live_lpac_backend.prepare(_spec(workdir))
    second = live_lpac_backend.prepare(_spec(workdir))
    one = first.spawn_callback._lpac_identity
    two = second.spawn_callback._lpac_identity
    profile_folder = Path(one.profile_folder)
    try:
        # One identity per installation, so the interpreter grant is made once.
        assert one.moniker == two.moniker == windows_lpac._install_moniker()
        assert one.sid_string == two.sid_string
        # The manifest is a launch's own. The temp is the container's, because
        # Windows points every AppContainer child of this profile at it.
        assert one.manifest_path != two.manifest_path
        assert one.private_temp == two.private_temp == str(profile_folder / "Temp")
        assert Path(one.manifest_path).is_file() and Path(two.manifest_path).is_file()
        assert Path(one.private_temp).is_dir()
        scratch = Path(two.private_temp) / "second-launch-scratch"
        scratch.write_text("owned by the second launch", encoding = "utf-8")
        assert one.sid_string in _acl_text(workdir)
        first.cleanup()
        assert first.cleanup_diagnostics == [], first.cleanup_diagnostics
        assert not Path(one.manifest_path).exists()
        # The second launch still holds the container temp, so nothing in it is
        # removed under it.
        assert scratch.is_file()
        assert Path(two.manifest_path).is_file()
        assert Path(two.private_temp).is_dir()
        # The second launch is still live, so the shared workdir grant stays.
        assert two.sid_string in _acl_text(workdir)
        assert profile_folder.is_dir()
    finally:
        first.cleanup()
        second.cleanup()
    assert two.sid_string not in _acl_text(workdir)
    # The last launch of the installation emptied the container temp; the
    # directory itself is the next launch's TEMP and stays.
    assert Path(two.private_temp).is_dir()
    assert not list(Path(two.private_temp).iterdir())
    # The profile and the interpreter grant outlive every launch of it.
    assert profile_folder.is_dir()
    manifest = Path(windows_lpac._persistent_manifest_path(one.moniker))
    assert manifest.is_file()
    persistent = json.loads(manifest.read_text(encoding = "utf-8"))
    verified = [
        root for root in persistent["granted_roots"] if root not in one.unverified_access
    ]
    # Not just the record: the ACE itself is still on the runtime tree.
    for root in verified[:1]:
        assert one.sid_string in _acl_text(Path(root))

    deleted = windows_lpac._api().userenv.DeleteAppContainerProfile(one.moniker)
    assert ctypes.c_uint32(deleted).value in (0, 0x80070002)
    # Deleting the profile does not change the identity: it is derived from the
    # name, so this rebuilds the same container with the same SID, and the later
    # tests in this module keep working against it.
    windows_lpac._INSTALL_PROFILE = None
    rebuilt = windows_lpac._install_profile()
    assert rebuilt.sid_string == one.sid_string
    assert rebuilt.moniker == one.moniker
    assert Path(rebuilt.profile_folder).is_dir()


def test_live_consecutive_launches_reuse_the_grant_and_stay_fast(
    live_lpac_backend, monkeypatch, tmp_path, record_property
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    install = windows_lpac._install_profile()
    granted: list[str] = []
    walked: list[tuple[str, ...]] = []
    real_grant = windows_lpac._grant_read_execute
    real_validate = windows_lpac._validate_runtime_trees

    def traced_grant(path, sid):
        granted.append(path)
        return real_grant(path, sid)

    def traced_validate(roots):
        walked.append(tuple(roots))
        return real_validate(roots)

    monkeypatch.setattr(windows_lpac, "_grant_read_execute", traced_grant)
    monkeypatch.setattr(windows_lpac, "_validate_runtime_trees", traced_validate)

    _first, first_elapsed = _run_native(live_lpac_backend, workdir, "pass")
    granted_after_first = list(granted)
    walked_after_first = list(walked)
    _second, second_elapsed = _run_native(live_lpac_backend, workdir, "pass")

    record_property("windows_lpac_first_launch_s", f"{first_elapsed:.3f}")
    record_property("windows_lpac_second_launch_s", f"{second_elapsed:.3f}")
    # The wall clock is a regression guard; this is the mechanism. Whatever the
    # first launch still had to grant, the second one repeats none of it, and it
    # walks no runtime tree either.
    assert granted == granted_after_first, granted[len(granted_after_first) :]
    assert walked == walked_after_first, walked[len(walked_after_first) :]
    # Propagating a fresh package SID over the interpreter tree, then revoking it
    # and deleting the profile, cost 25 to 45 s per call on these runners. The
    # grant is now made once per installation, so a later launch pays only the
    # workdir work and the DACL reads.
    assert second_elapsed < 8.0, (first_elapsed, second_elapsed)
    assert windows_lpac._install_profile() is install
    assert Path(install.profile_folder).is_dir()
    assert Path(windows_lpac._persistent_manifest_path(install.moniker)).is_file()


def test_live_workdir_home_runtime_native_extension_and_container_temp(live_lpac_backend, tmp_path):
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
assert private != pathlib.Path({str(os.environ.get("TEMP", ""))!r})
parts = [part.lower() for part in private.parts]
assert 'packages' in parts and parts[-2:] == ['ac', 'temp']
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
def denied(family, address, kind=socket.SOCK_STREAM):
    # A zero-capability AppContainer cannot even initialize Winsock
    # (WSAEPROVIDERFAILEDINIT), so socket() itself may be the denial.
    client = None
    try:
        client = socket.socket(family, kind); client.settimeout(1)
        if kind == socket.SOCK_STREAM:
            client.connect(address)
        else:
            client.sendto(b'LPAC_ESCAPE', address)
    except OSError:
        return
    finally:
        if client is not None:
            client.close()
    if kind == socket.SOCK_STREAM:
        raise AssertionError('host endpoint reachable: ' + repr(address))
denied(socket.AF_INET, {address4!r})
denied(socket.AF_INET6, {address6!r})
denied(socket.AF_INET, {udp_address!r}, socket.SOCK_DGRAM)
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
k.SetEvent.argtypes = [wintypes.HANDLE]
k.SetEvent.restype = wintypes.BOOL
# The parent's handle values are only meaningful in the child if they were
# inherited. Signalling through them and letting the parent look at its own
# events is the definitive check: a value that happens to name some unrelated
# handle in this process (numbering shifts with every object the launcher
# creates) cannot signal the parent's event, so no false failure either way.
for raw in os.environ['UNSLOTH_TEST_HANDLES'].split(','):
    try:
        k.SetEvent(wintypes.HANDLE(int(raw)))
    except OSError:
        pass
print('LPAC_HANDLES_OK')
"""
        output, _elapsed = _run_native(live_lpac_backend, workdir, code, env = env)
        assert "LPAC_HANDLES_OK" in output
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        for handle in handles:
            # WAIT_TIMEOUT: the event is still unsignalled, so the child never held it.
            assert kernel32.WaitForSingleObject(handle, 0) == 0x102, handle
    finally:
        for handle in handles:
            kernel32.CloseHandle(handle)


def test_live_null_device_and_named_pipes_are_denied_and_disclosed(live_lpac_backend, tmp_path):
    """The container cannot open NUL or create named pipes; the capability says so.

    multiprocessing.Pipe() and therefore Queue(), Process() with the spawn
    context, and torch (through dill, which opens os.devnull at import) cannot
    work inside the sandbox. The limitation is advertised so the UI and the
    model can route such work to Limited or Full mode instead of guessing.
    """
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import multiprocessing.connection as connection
import os
outcomes = {}
try:
    open(os.devnull, 'rb').close()
    outcomes['devnull'] = 'opened'
except PermissionError:
    outcomes['devnull'] = 'denied'
try:
    reader, writer = connection.Pipe(duplex=False)
    reader.close(); writer.close()
    outcomes['pipe'] = 'created'
except PermissionError:
    outcomes['pipe'] = 'denied'
print('LPAC_NULL_PIPES ' + repr(outcomes))
"""
    output, _elapsed = _run_native(live_lpac_backend, workdir, code)
    assert "LPAC_NULL_PIPES {'devnull': 'denied', 'pipe': 'denied'}" in output, output
    capability = os_sandbox.capability_snapshot()
    assert windows_lpac._LIMITATION_NULL_DEVICE_PIPES in capability.limitations


def test_live_pytorch_import_failure_is_the_disclosed_null_device_limit(live_lpac_backend, tmp_path):
    """torch cannot import inside the container, and the failure is the advertised one."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    workdir = tmp_path / "work"
    workdir.mkdir()
    site_dirs = [
        path for path in (sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")) if path
    ]
    code = f"import sys; sys.path[:0] = {site_dirs!r}\n" + """
try:
    import torch
except PermissionError as exc:
    print('LPAC_TORCH_DENIED ' + repr(exc.filename))
else:
    print('LPAC_TORCH_IMPORTED')
"""
    output, _elapsed = _run_native(live_lpac_backend, workdir, code, timeout = 120)
    assert "LPAC_TORCH_DENIED 'nul'" in output or "LPAC_TORCH_IMPORTED" in output, output


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
    # Every sample here is a later launch of one installation, so none of them
    # pays the interpreter grant. Before the stable identity the median of this
    # loop was 25 s and over on the hosted runners.
    assert median < 8000, samples

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Write-restricted token launcher for Limited mode on Windows.

The first half runs everywhere and pins the launcher's contract (flags, restricting
SIDs, probe evaluation, manifest reconciliation, how os_sandbox selects it and how it
falls back). The second half runs only on Windows against a real restricted token.
"""

from __future__ import annotations

import ast
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from core.inference import os_sandbox
from core.inference import tool_isolation
from core.inference import tools as inference_tools
from core.inference import windows_lpac
from core.inference import windows_restricted_token as token_launcher


SOURCE = Path(token_launcher.__file__).read_text(encoding = "utf-8")


@pytest.fixture
def isolated_capability_cache():
    before = dict(os_sandbox._capability_cache)
    os_sandbox._capability_cache.clear()
    try:
        yield
    finally:
        os_sandbox._capability_cache.clear()
        os_sandbox._capability_cache.update(before)


def _good_findings(sid_text: str) -> dict:
    return {
        "restricted": True,
        "restricted_sids": [sid_text, "S-1-5-5-0-1234", "S-1-1-0"],
        "privileges": 1,
        "in_job": True,
        "secret_readable": True,
        "secret_writable": False,
        "sibling_writable": False,
        "workdir_writable": True,
        "temp_writable": True,
        "temp_is_private": True,
        "devnull": True,
        "pipe": True,
    }


# ── contract pins (every platform) ───────────────────────────────────────────


def test_public_api_profile_and_token_flags_are_pinned():
    assert token_launcher.__all__ == ["WindowsRestrictedTokenBackend"]
    backend = token_launcher.WindowsRestrictedTokenBackend()
    assert backend.identity == "windows-restricted-token"
    assert backend.profile_id == "windows-restricted-token-write-isolation-v1"
    assert backend.limitations == (
        "user_profile_readable",
        "network_unrestricted",
        "everyone_writable_objects_writable",
    )
    # DISABLE_MAX_PRIVILEGE | LUA_TOKEN | WRITE_RESTRICTED, the Codex / DeepSeek flag set.
    assert token_launcher._RESTRICTED_TOKEN_FLAGS == 0x1 | 0x4 | 0x8
    assert token_launcher._RESTRICTED_TOKEN_FLAGS & 0x2 == 0  # never SANDBOX_INERT
    # The token is a copy of Studio's own primary token; no privilege is needed to assign it.
    assert "OpenProcessToken(" in SOURCE
    assert "CreateProcessAsUserW(" in SOURCE
    assert "subprocess.Popen(" not in SOURCE
    assert "_PROC_THREAD_ATTRIBUTE_JOB_LIST" in SOURCE
    assert "Limited processes may not break away from their Job Object" in SOURCE
    assert '_ADMINISTRATORS_SID = "S-1-5-32-544"' in SOURCE
    # The restricting set, the default DACL, the job attachment order and the
    # failure paths are pinned behaviourally below, against a fake _api().


def test_no_failure_this_module_raises_mentions_lpac():
    # Limited mode is not an AppContainer; messages reused from the shared
    # helpers are restated before they reach a user or a log.
    for node in ast.walk(ast.parse(SOURCE)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") not in {"SandboxUnavailableError", "OSError"}:
            continue
        for text in ast.walk(node):
            if isinstance(text, ast.Constant) and isinstance(text.value, str):
                assert "LPAC" not in text.value, ast.dump(node)


def test_shared_validation_failures_are_restated_for_limited_mode():
    assert token_launcher._limited_wording(
        "the LPAC workdir contains a reparse point: C:\\w\\link"
    ) == "the Limited mode workdir contains a reparse point: C:\\w\\link"
    assert token_launcher._limited_wording("LPAC requires a non-root directory on a local drive") == (
        "Limited mode requires a non-root directory on a local drive"
    )
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "Limited mode workdir") as caught:
        with token_launcher._limited_mode_wording():
            raise os_sandbox.SandboxUnavailableError("the LPAC workdir root is a reparse point")
    assert "LPAC" not in str(caught.value)
    # A message that never mentioned LPAC is re-raised untouched, transient flag included.
    original = os_sandbox.SandboxUnavailableError("a Limited mode private temp path is outside its root")
    with pytest.raises(os_sandbox.SandboxUnavailableError) as unchanged:
        with token_launcher._limited_mode_wording():
            raise original
    assert unchanged.value is original


def test_random_launch_sid_is_a_fresh_domain_sid():
    seen = {token_launcher._random_domain_sid_text() for _ in range(64)}
    assert len(seen) == 64
    for text in seen:
        assert token_launcher._is_launch_sid_text(text)
        parts = text.split("-")
        assert parts[:4] == ["S", "1", "5", "21"]
        assert all(0 <= int(part) < 2**32 for part in parts[4:])
    for bad in ("S-1-5-32-544", "S-1-15-2-1", "S-1-5-21-1-2-3", "S-1-5-21-a-b-c-d", 5, None):
        assert not token_launcher._is_launch_sid_text(bad)


def test_probe_evaluation_requires_every_observation():
    sid = "S-1-5-21-1-2-3-4"
    assert token_launcher._evaluate_probe(_good_findings(sid), sid_text = sid) is None
    assert token_launcher._evaluate_probe({}, sid_text = sid) == "the token is not restricted"
    flips = {
        "restricted": (False, "not restricted"),
        "privileges": (3, "kept privileges"),
        "in_job": (False, "not inside its Job Object"),
        "secret_readable": (False, "stronger than modelled"),
        "secret_writable": (True, "outside the workdir was writable"),
        "sibling_writable": (True, "another launch's temp"),
        "workdir_writable": (False, "workdir was not writable"),
        "temp_writable": (False, "private temp was not writable"),
        "temp_is_private": (False, "TEMP was not redirected"),
        "devnull": ("PermissionError", "NUL device was unavailable"),
        "pipe": ("PermissionError", "named pipes were unavailable"),
    }
    for key, (value, fragment) in flips.items():
        findings = _good_findings(sid)
        findings[key] = value
        reason = token_launcher._evaluate_probe(findings, sid_text = sid)
        assert reason is not None and fragment in reason, (key, reason)
    without_launch_sid = _good_findings(sid)
    without_launch_sid["restricted_sids"] = ["S-1-1-0"]
    assert "launch SID" in token_launcher._evaluate_probe(without_launch_sid, sid_text = sid)
    without_everyone = _good_findings(sid)
    without_everyone["restricted_sids"] = [sid]
    assert "Everyone" in token_launcher._evaluate_probe(without_everyone, sid_text = sid)


@pytest.mark.skipif(sys.platform == "win32", reason = "the launcher is Windows-only")
def test_launcher_is_unavailable_off_windows(tmp_path):
    backend = token_launcher.WindowsRestrictedTokenBackend()
    capability = backend.probe()
    assert capability.available is False
    assert capability.qualified is False
    assert "require Windows" in capability.reason
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "require Windows"):
        backend.prepare(os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(tmp_path), env = {}))
    assert os_sandbox._limited_isolation_backend() is None


def test_launch_environment_redirects_temp_and_keeps_system_root(monkeypatch):
    identity = SimpleNamespace(private_temp = r"C:\Users\u\AppData\Local\Unsloth\Studio\limited-temp\a")
    monkeypatch.setenv("SystemRoot", r"C:\WINDOWS")
    env = token_launcher._launch_environment(
        {"PATH": "p", "Temp": r"C:\old", "TMP": r"C:\old", "PYTHONIOENCODING": "utf-8"}, identity
    )
    assert env["TEMP"] == env["TMP"] == identity.private_temp
    assert "Temp" not in env
    assert env["PATH"] == "p"
    assert env["SystemRoot"] == r"C:\WINDOWS"
    kept = token_launcher._launch_environment({"SYSTEMROOT": r"D:\W"}, identity)
    assert kept["SYSTEMROOT"] == r"D:\W" and "SystemRoot" not in kept


def _manifest(root: Path, sid: str, **overrides) -> Path:
    payload = {
        "version": 1,
        "kind": "restricted-token",
        "sid": sid,
        "workdir": str(root / "work"),
        "private_temp": str(root / "temp" / ("a" * 24)),
        "granted_roots": [str(root / "work"), str(root / "temp" / ("a" * 24))],
        "owner_pid": 4242,
        "owner_created": 7,
    }
    payload.update(overrides)
    path = root / (token_launcher._MANIFEST_PREFIX + sid + ".json")
    path.write_text(json.dumps(payload), encoding = "utf-8")
    return path


def _unmapped_sid_api(freed: list, *, resolves: bool = False) -> SimpleNamespace:
    """A fake ``_api()`` whose LookupAccountSidW resolves no SID unless asked to."""

    def lookup(system, sid, name, name_length, domain, domain_length, use):
        return 1 if resolves else 0

    return SimpleNamespace(
        kernel32 = SimpleNamespace(LocalFree = freed.append),
        advapi32 = SimpleNamespace(LookupAccountSidW = lookup),
    )


def test_manifest_parser_accepts_only_this_launchers_records(tmp_path):
    sid = "S-1-5-21-1-2-3-4"
    good = _manifest(tmp_path, sid)
    assert token_launcher._parse_manifest(good)["sid"] == sid
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, version = 2)) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, kind = "lpac")) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, "S-1-15-2-1-2-3-4-5")) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, granted_roots = ["rel"])) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, owner_pid = "1")) is None
    # granted_roots drives an ACL revoke, so it must be exactly the two roots
    # _create_identity grants: the manifest's own workdir and private temp.
    workdir = str(tmp_path / "work")
    private_temp = str(tmp_path / "temp" / ("a" * 24))
    for planted in (
        [workdir, private_temp, str(tmp_path / "documents")],
        [workdir],
        [str(tmp_path / "documents"), private_temp],
        [workdir, workdir],
        [],
    ):
        assert token_launcher._parse_manifest(
            _manifest(tmp_path, sid, granted_roots = planted)
        ) is None, planted
    assert token_launcher._parse_manifest(
        _manifest(tmp_path, sid, granted_roots = [private_temp, workdir])
    )["granted_roots"] == [private_temp, workdir]
    foreign = tmp_path / "unsloth.studio.deadbeef.json"  # an AppContainer manifest
    foreign.write_text(json.dumps({"version": 1, "moniker": "unsloth.studio.deadbeef"}))
    assert token_launcher._parse_manifest(foreign) is None
    broken = tmp_path / (token_launcher._MANIFEST_PREFIX + sid + "x.json")
    broken.write_text("{not json")
    assert token_launcher._parse_manifest(broken) is None


def test_reconcile_cleans_only_manifests_whose_owner_is_gone(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    temp_root = tmp_path / "temp"
    manifests.mkdir()
    temp_root.mkdir()
    (tmp_path / "work").mkdir()
    dead_sid = "S-1-5-21-1-1-1-1"
    live_sid = "S-1-5-21-2-2-2-2"
    dead_temp = temp_root / ("a" * 24)
    live_temp = temp_root / ("b" * 24)
    dead_temp.mkdir()
    live_temp.mkdir()
    (dead_temp / "scratch.txt").write_text("x")
    dead = _manifest(manifests, dead_sid, workdir = str(tmp_path / "work"), private_temp = str(dead_temp),
                     granted_roots = [str(tmp_path / "work"), str(dead_temp)], owner_pid = 4242)
    live = _manifest(manifests, live_sid, workdir = str(tmp_path / "work"), private_temp = str(live_temp),
                     granted_roots = [str(tmp_path / "work"), str(live_temp)], owner_pid = 5151)
    # A manifest whose file name does not match its SID is evidence, never acted on.
    mismatched = manifests / (token_launcher._MANIFEST_PREFIX + "S-1-5-21-3-3-3-3.json")
    mismatched.write_text(dead.read_text(encoding = "utf-8"), encoding = "utf-8")

    revoked: list[tuple[str, str]] = []
    freed: list[int] = []
    monkeypatch.setattr(token_launcher, "_manifest_root", lambda: str(manifests))
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(hash(text) & 0xFFFF or 1))
    monkeypatch.setattr(
        windows_lpac, "_process_identity", lambda pid = None: (5151, 7) if pid == 5151 else None
    )
    monkeypatch.setattr(windows_lpac, "_revoke_sid", lambda path, sid, **kw: revoked.append((path, sid.value)))
    monkeypatch.setattr(windows_lpac, "_api", lambda: _unmapped_sid_api(freed))
    monkeypatch.setattr(token_launcher, "_last_error", lambda: token_launcher._ERROR_NONE_MAPPED)

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    assert not dead.exists()
    assert not dead_temp.exists()
    assert live.exists() and live_temp.exists()
    assert mismatched.exists()
    assert [path for path, _ in revoked] == [str(dead_temp), str(tmp_path / "work")]
    assert len(freed) == 1


def test_identity_cleanup_is_idempotent_and_reports_every_failure(tmp_path, monkeypatch):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    private = temp_root / ("c" * 24)
    private.mkdir()
    manifest = tmp_path / "m.json"
    manifest.write_text("{}")
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    calls: list[str] = []

    def revoke(path, sid, **kw):
        calls.append(path)
        if path.endswith("work"):
            raise OSError(5, "denied")

    monkeypatch.setattr(windows_lpac, "_revoke_sid", revoke)
    freed: list = []
    monkeypatch.setattr(
        windows_lpac, "_api", lambda: SimpleNamespace(kernel32 = SimpleNamespace(LocalFree = freed.append))
    )
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(2))
    identity = token_launcher._LaunchIdentity(
        ctypes.c_void_p(1), "S-1-5-21-9-9-9-9", str(tmp_path / "work"), str(private), str(manifest),
        (str(tmp_path / "work"), str(private)), 1, 1,
    )
    with pytest.raises(OSError, match = "ACL .*work"):
        identity.cleanup()
    # The temp went even though a revoke failed, and the manifest stayed as evidence.
    assert not private.exists()
    assert manifest.exists()
    assert identity.cleaned is False
    # The SID allocation is released even by the failing path, and the retry
    # converts the recorded SID text again rather than reusing freed memory.
    assert [item.value for item in freed] == [1]
    calls.clear()
    monkeypatch.setattr(windows_lpac, "_revoke_sid", lambda path, sid, **kw: calls.append(path))
    identity.cleanup()
    assert identity.cleaned is True
    assert not manifest.exists()
    identity.cleanup()  # idempotent
    assert calls == [str(private), str(tmp_path / "work")]
    assert [item.value for item in freed] == [1, 2]


# ── Win32 call behaviour (every platform, against a fake _api()) ─────────────


class _WinApiRecorder:
    """A windows_lpac._api() stand-in that records the ordered Win32 calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.closed: list[int] = []
        self.freed: list[int] = []
        self.create_token_result = 1
        self.create_process_result = 1
        self.resume_result = 0
        self.job_list_supported = True
        self.active_processes = [0]
        self.last_error = token_launcher._ERROR_INSUFFICIENT_BUFFER
        self.kernel32 = SimpleNamespace(
            GetCurrentProcess = lambda: 11,
            CloseHandle = self._close,
            LocalFree = self._free,
            InitializeProcThreadAttributeList = self._initialize_attributes,
            UpdateProcThreadAttribute = self._update_attribute,
            DeleteProcThreadAttributeList = lambda attributes: self._note(
                "DeleteProcThreadAttributeList"
            ),
            AssignProcessToJobObject = lambda job, process: self._note("AssignProcessToJobObject"),
            ResumeThread = self._resume,
            TerminateProcess = self._terminate,
            QueryInformationJobObject = self._query_job,
        )
        self.advapi32 = SimpleNamespace(
            OpenProcessToken = self._open_process_token,
            CreateRestrictedToken = self._create_restricted_token,
            CreateProcessAsUserW = self._create_process,
        )

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]

    def call(self, name: str) -> tuple:
        return next(call for call in self.calls if call[0] == name)

    def _note(self, name: str, *args) -> int:
        self.calls.append((name, *args))
        return 1

    @staticmethod
    def _value(handle):
        return getattr(handle, "value", handle)

    def _close(self, handle) -> int:
        self.closed.append(self._value(handle))
        return self._note("CloseHandle", self._value(handle))

    def _free(self, sid) -> int:
        self.freed.append(self._value(sid))
        self._note("LocalFree", self._value(sid))
        return 0

    def _open_process_token(self, process, access, out) -> int:
        out._obj.value = 4321
        return self._note("OpenProcessToken", access)

    def _create_restricted_token(
        self, source, flags, disable_count, disable, privilege_count, privileges,
        restrict_count, restrict, out,
    ) -> int:
        self.calls.append((
            "CreateRestrictedToken",
            flags,
            disable_count,
            None if disable is None else len(disable),
            [] if disable is None else [(entry.Sid, entry.Attributes) for entry in disable],
            privilege_count,
            privileges,
            restrict_count,
            len(restrict),
            [(entry.Sid, entry.Attributes) for entry in restrict],
        ))
        if not self.create_token_result:
            return 0
        out._obj.value = 9999
        return 1

    def _initialize_attributes(self, attributes, count, flags, size) -> int:
        self._note("InitializeProcThreadAttributeList", count)
        if attributes is None:
            size._obj.value = 64
            return 0
        return 1

    def _update_attribute(self, attributes, flags, attribute, value, size, a, b) -> int:
        self._note("UpdateProcThreadAttribute", attribute)
        if attribute == windows_lpac._PROC_THREAD_ATTRIBUTE_JOB_LIST and not self.job_list_supported:
            self.last_error = token_launcher._ERROR_NOT_SUPPORTED
            return 0
        return 1

    def _create_process(
        self, token, application, command_line, process_attributes, thread_attributes,
        inherit, flags, environment, workdir, startup, process_information,
    ) -> int:
        self.calls.append(("CreateProcessAsUserW", self._value(token), flags, workdir))
        if not self.create_process_result:
            return 0
        info = process_information._obj
        info.hProcess = 1234
        info.hThread = 5678
        info.dwProcessId = 4242
        return 1

    def _resume(self, thread) -> int:
        self._note("ResumeThread")
        return self.resume_result

    def _terminate(self, process, code) -> int:
        return self._note("TerminateProcess", self._value(process))

    def _query_job(self, handle, kind, info, length, returned) -> int:
        self._note("QueryInformationJobObject", kind)
        remaining = self.active_processes.pop(0) if len(self.active_processes) > 1 else (
            self.active_processes[0]
        )
        info._obj.ActiveProcesses = remaining
        return 1


class _FakeJob:
    """The _WindowsJob surface _spawn_restricted and the drain wait use."""

    def __init__(self, recorder: _WinApiRecorder, handle: int = 55) -> None:
        self._handle = handle
        self._recorder = recorder

    def terminate(self) -> bool:
        self._recorder._note("job.terminate")
        return True

    def close(self) -> None:
        self._recorder._note("job.close")
        self._handle = None


_STDIO_PLAN = {
    "stdout": subprocess.PIPE,
    "stderr": subprocess.STDOUT,
    "stdin": subprocess.DEVNULL,
    "text": True,
    "encoding": "utf-8",
    "errors": "replace",
    "close_fds": True,
}


def _launch_identity(tmp_path) -> object:
    return token_launcher._LaunchIdentity(
        ctypes.c_void_p(77), "S-1-5-21-1-1-1-1", str(tmp_path), str(tmp_path / "temp"),
        str(tmp_path / "m.json"), (str(tmp_path),), 1, 1,
    )


def _prepared(tmp_path) -> os_sandbox.PreparedSandboxLaunch:
    return os_sandbox.PreparedSandboxLaunch(
        argv = (sys.executable, "-c", "pass"),
        workdir = str(tmp_path),
        env = {"PATH": "p", "TEMP": str(tmp_path)},
        preexec_fn = None,
        backend = token_launcher._BACKEND_IDENTITY,
    )


def test_create_restricted_token_pins_the_flags_and_the_restricting_set(monkeypatch):
    recorder = _WinApiRecorder()
    sids: dict[str, int] = {}

    def sid_from_text(text):
        sids.setdefault(text, len(sids) + 1)
        return ctypes.c_void_p(sids[text])

    groups = {
        token_launcher._TOKEN_LOGON_SID: ["S-1-5-5-0-9999"],
        token_launcher._TOKEN_GROUPS: ["S-1-5-32-544", "S-1-5-21-1-2-3-513"],
    }
    dacl: list[tuple[int, int]] = []
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(token_launcher, "_sid_from_text", sid_from_text)
    monkeypatch.setattr(token_launcher, "_token_group_sids", lambda api, token, kind: groups[kind])
    monkeypatch.setattr(
        token_launcher, "_set_default_dacl",
        lambda api, token, sid: dacl.append((recorder._value(token), sid.value)),
    )
    identity = SimpleNamespace(sid = ctypes.c_void_p(77), sid_string = "S-1-5-21-1-1-1-1")

    handle = token_launcher._create_restricted_token(identity)

    assert handle.value == 9999
    (
        _, flags, disable_count, disable_length, disable_entries,
        privilege_count, privileges, restrict_count, restrict_length, restrict_entries,
    ) = recorder.call("CreateRestrictedToken")
    assert flags == token_launcher._RESTRICTED_TOKEN_FLAGS == 0x1 | 0x4 | 0x8
    assert privilege_count == 0 and privileges is None
    # Counts must match the arrays they describe: they are adjacent DWORDs.
    assert disable_count == disable_length == 1
    assert disable_entries == [(sids["S-1-5-32-544"], 0)]
    assert restrict_count == restrict_length == 3
    # The launch SID first, then the logon SID, then Everyone; Attributes zero.
    assert [entry[0] for entry in restrict_entries] == [
        sids["S-1-5-21-1-1-1-1"], sids["S-1-5-5-0-9999"], sids["S-1-1-0"]
    ]
    assert [entry[1] for entry in restrict_entries] == [0, 0, 0]
    # The default DACL is widened to the launch SID, on the restricted token.
    assert dacl == [(9999, 77)]
    # Every converted SID is released and the source token handle is closed.
    assert sorted(recorder.freed) == sorted(sids.values())
    assert 4321 in recorder.closed


def test_create_restricted_token_declines_when_the_logon_sid_is_unavailable(monkeypatch):
    recorder = _WinApiRecorder()
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(1))
    monkeypatch.setattr(token_launcher, "_set_default_dacl", lambda api, token, sid: None)
    identity = SimpleNamespace(sid = ctypes.c_void_p(77), sid_string = "S-1-5-21-1-1-1-1")

    # An empty logon SID list would silently drop it from the restricting set.
    monkeypatch.setattr(token_launcher, "_token_group_sids", lambda api, token, kind: [])
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "logon SID"):
        token_launcher._create_restricted_token(identity)

    # A failed query is not swallowed either; both fall back to the process guard.
    def failing(api, token, kind):
        raise OSError(5, "GetTokenInformation(28)")

    monkeypatch.setattr(token_launcher, "_token_group_sids", failing)
    with pytest.raises(OSError):
        token_launcher._create_restricted_token(identity)
    assert "CreateRestrictedToken" not in recorder.names()
    assert recorder.closed == [4321, 4321]  # the source token is released both times


def test_administrators_lookup_failure_stays_a_swallow(monkeypatch):
    recorder = _WinApiRecorder()
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(1))
    monkeypatch.setattr(token_launcher, "_set_default_dacl", lambda api, token, sid: None)

    def groups(api, token, kind):
        if kind == token_launcher._TOKEN_GROUPS:
            raise OSError(5, "GetTokenInformation(2)")
        return ["S-1-5-5-0-9999"]

    monkeypatch.setattr(token_launcher, "_token_group_sids", groups)
    identity = SimpleNamespace(sid = ctypes.c_void_p(77), sid_string = "S-1-5-21-1-1-1-1")
    assert token_launcher._create_restricted_token(identity).value == 9999
    # LUA_TOKEN already covers the Administrators gap, so the token is still built.
    assert recorder.call("CreateRestrictedToken")[2] == 0


def _spawn_with_fake_api(tmp_path, monkeypatch, recorder, job):
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle = lambda fd: fd))
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(token_launcher, "_last_error", lambda: recorder.last_error)
    resources = token_launcher._LaunchResources(token = wintypes.HANDLE(4711), job = job)
    prepared = _prepared(tmp_path)
    return prepared, resources


def test_spawn_restricted_attaches_the_job_before_it_resumes_the_child(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    recorder.active_processes = [1, 0]
    job = _FakeJob(recorder)
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, recorder, job)

    process = token_launcher._spawn_restricted(
        prepared, dict(_STDIO_PLAN), _launch_identity(tmp_path), resources
    )

    names = recorder.names()
    created = names.index("CreateProcessAsUserW")
    # The job is attached at creation through PROC_THREAD_ATTRIBUTE_JOB_LIST, so
    # no instruction ever runs outside it, and the resume comes last.
    assert ("UpdateProcThreadAttribute", windows_lpac._PROC_THREAD_ATTRIBUTE_JOB_LIST) in recorder.calls
    assert names.index("UpdateProcThreadAttribute") < created < names.index("ResumeThread")
    assert "AssignProcessToJobObject" not in names
    token_value, flags, workdir = recorder.call("CreateProcessAsUserW")[1:]
    assert token_value == 4711
    assert flags & windows_lpac._CREATE_SUSPENDED
    assert workdir == prepared.workdir
    assert process.pid == 4242
    # The token handle is released once the child holds its own reference, and
    # the job belongs to the process from here.
    assert 4711 in recorder.closed
    assert resources.token is None and resources.job is None

    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []
    names = recorder.names()
    # Kill the job, wait for it to actually drain, only then release the handles.
    assert names.index("job.terminate") < names.index("QueryInformationJobObject")
    assert names.index("QueryInformationJobObject") < names.index("job.close")
    assert names.count("QueryInformationJobObject") == 2  # one process left, then none


def test_spawn_restricted_falls_back_to_assigning_the_job_before_the_resume(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    recorder.job_list_supported = False
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, recorder, _FakeJob(recorder))

    token_launcher._spawn_restricted(
        prepared, dict(_STDIO_PLAN), _launch_identity(tmp_path), resources
    )

    names = recorder.names()
    assert names.index("CreateProcessAsUserW") < names.index("AssignProcessToJobObject")
    assert names.index("AssignProcessToJobObject") < names.index("ResumeThread")
    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []


def test_spawn_restricted_terminates_the_suspended_child_when_the_resume_fails(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    recorder.resume_result = 0xFFFFFFFF
    job = _FakeJob(recorder)
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, recorder, job)
    monkeypatch.setattr(windows_lpac, "_winerror", lambda prefix, code = None: OSError(5, prefix))

    with pytest.raises(OSError, match = "ResumeThread"):
        token_launcher._spawn_restricted(
            prepared, dict(_STDIO_PLAN), _launch_identity(tmp_path), resources
        )

    # Nothing ever ran: the still-suspended child is terminated and every handle
    # this launch owns is released.
    assert ("TerminateProcess", 1234) in recorder.calls
    assert 5678 in recorder.closed and 1234 in recorder.closed
    assert 4711 in recorder.closed  # the token was consumed by CreateProcessAsUserW
    assert prepared.cleanup_callbacks == []
    assert resources.job is job  # still the launch's, released by prepare's cleanup
    resources.close()
    assert recorder.names().count("job.close") == 1


def test_spawn_restricted_refuses_a_launch_whose_resources_are_gone(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    prepared, resources = _spawn_with_fake_api(tmp_path, monkeypatch, recorder, _FakeJob(recorder))
    resources.close()
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "already released"):
        token_launcher._spawn_restricted(
            prepared, dict(_STDIO_PLAN), _launch_identity(tmp_path), resources
        )


def _prepare_environment(tmp_path, monkeypatch, recorder):
    """Everything prepare() touches on a host, replaced by recorders."""
    manifests = tmp_path / "manifests"
    temp_root = tmp_path / "temp"
    work = tmp_path / "work"
    for path in (manifests, temp_root, work):
        path.mkdir()
    granted: list[str] = []
    revoked: list[str] = []
    monkeypatch.setattr(token_launcher, "_is_windows", lambda: True)
    monkeypatch.setattr(token_launcher, "_manifest_root", lambda: str(manifests))
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(3))
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    monkeypatch.setattr(windows_lpac, "_validate_workdir", lambda path: str(work))
    monkeypatch.setattr(windows_lpac, "_canonical_inner_argv", lambda argv, env: tuple(argv))
    monkeypatch.setattr(windows_lpac, "_process_identity", lambda pid = None: (os.getpid(), 5))
    monkeypatch.setattr(windows_lpac, "_grant_modify", lambda path, sid: granted.append(path))
    monkeypatch.setattr(windows_lpac, "_revoke_sid", lambda path, sid, **kw: revoked.append(path))
    return SimpleNamespace(
        manifests = manifests, temp_root = temp_root, work = work,
        granted = granted, revoked = revoked,
    )


def test_prepare_builds_the_token_and_the_job_before_it_returns(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    job = _FakeJob(recorder)
    monkeypatch.setattr(
        token_launcher, "_create_restricted_token", lambda identity: wintypes.HANDLE(4711)
    )
    monkeypatch.setattr(windows_lpac, "_job_object_with_limits", lambda: job)

    prepared = token_launcher.WindowsRestrictedTokenBackend().prepare(
        os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
    )

    identity = prepared.spawn_callback._launch_identity
    assert host.granted == [str(host.work), identity.private_temp]
    assert Path(identity.manifest_path).exists()
    prepared.cleanup()
    assert prepared.cleanup_diagnostics == []
    # LIFO: the job dies and drains, then the ACEs and the private temp go.
    names = recorder.names()
    assert names.index("job.close") < names.index("LocalFree")
    assert 4711 in recorder.closed
    assert host.revoked == [identity.private_temp, str(host.work)]
    assert not Path(identity.manifest_path).exists()
    assert list(host.temp_root.iterdir()) == []


def test_prepare_declines_and_cleans_up_when_the_token_cannot_be_created(tmp_path, monkeypatch):
    recorder = _WinApiRecorder()
    host = _prepare_environment(tmp_path, monkeypatch, recorder)
    jobs: list[object] = []

    def refuse(identity):
        raise OSError(5, "CreateRestrictedToken denied")

    monkeypatch.setattr(token_launcher, "_create_restricted_token", refuse)
    monkeypatch.setattr(
        windows_lpac, "_job_object_with_limits", lambda: jobs.append(_FakeJob(recorder))
    )

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "restricted token and job"):
        token_launcher.WindowsRestrictedTokenBackend().prepare(
            os_sandbox.ToolLaunchPlan(argv = ("x",), workdir = str(host.work), env = {})
        )

    # A token failure is a fallback, not a failed tool call, and it leaves no
    # grant, manifest, private temp or handle behind.
    # The grants were applied before the token failed, and all of them were undone.
    assert len(host.granted) == 2 and host.granted[0] == str(host.work)
    assert os.path.dirname(host.granted[1]) == str(host.temp_root)
    assert host.revoked == list(reversed(host.granted))
    assert list(host.manifests.iterdir()) == []
    assert list(host.temp_root.iterdir()) == []
    assert jobs == []  # the job is never created once the token failed
    assert recorder.freed == [3]


def test_wait_for_job_drain_polls_until_the_job_is_empty(monkeypatch):
    recorder = _WinApiRecorder()
    recorder.active_processes = [2, 1, 0]
    monkeypatch.setattr(windows_lpac, "_api", lambda: recorder)
    assert token_launcher._wait_for_job_drain(_FakeJob(recorder)) is True
    assert recorder.names().count("QueryInformationJobObject") == 3
    # A job that never drains gives up instead of blocking cleanup forever.
    recorder.active_processes = [3]
    started = time.perf_counter()
    assert token_launcher._wait_for_job_drain(_FakeJob(recorder), timeout = 0.05) is False
    assert time.perf_counter() - started < 2
    # A host without QueryInformationJobObject is reported, not spun on.
    del recorder.kernel32.QueryInformationJobObject
    assert token_launcher._wait_for_job_drain(_FakeJob(recorder)) is False
    # A job whose handle is already closed has nothing left to wait for.
    assert token_launcher._wait_for_job_drain(SimpleNamespace(_handle = None)) is True


def test_read_only_files_are_cleared_and_the_removal_retried(tmp_path, monkeypatch):
    handler = token_launcher._rmtree_error_handler()
    assert list(handler) == ["onexc" if sys.version_info >= (3, 12) else "onerror"]
    victim = tmp_path / "readonly.txt"
    victim.write_text("x", encoding = "utf-8")
    chmods: list[tuple[str, int]] = []
    retried: list[str] = []
    monkeypatch.setattr(os, "chmod", lambda path, mode: chmods.append((str(path), mode)))

    token_launcher._force_removable(retried.append, str(victim), PermissionError(13, "denied"))
    assert chmods == [(str(victim), stat.S_IWRITE)]
    assert retried == [str(victim)]
    # The pre-3.12 onerror shape reaches the same handler.
    token_launcher._rmtree_onerror(
        retried.append, str(victim), (PermissionError, PermissionError(13, "denied"), None)
    )
    assert retried == [str(victim), str(victim)]
    # A path that vanished under the walk is not a failure.
    token_launcher._force_removable(retried.append, str(tmp_path / "gone"), FileNotFoundError())
    assert len(retried) == 2 and len(chmods) == 2


def test_private_temp_removal_retries_a_sharing_violation_then_gives_up(tmp_path, monkeypatch):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    private = temp_root / ("d" * 24)
    private.mkdir()
    (private / "scratch.txt").write_text("x", encoding = "utf-8")
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    sleeps: list[float] = []
    monkeypatch.setattr(token_launcher.time, "sleep", sleeps.append)
    attempts: list[str] = []
    real_rmtree = shutil.rmtree

    def flaky(path, **kwargs):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError(13, "the file is in use by another process")
        return real_rmtree(path, **kwargs)

    monkeypatch.setattr(token_launcher.shutil, "rmtree", flaky)
    token_launcher._remove_private_temp(str(private))
    assert len(attempts) == 3 and not private.exists()
    assert sleeps == [0.05, 0.1]

    private.mkdir()
    attempts.clear()
    monkeypatch.setattr(
        token_launcher.shutil, "rmtree",
        lambda path, **kwargs: attempts.append(path) or (_ for _ in ()).throw(
            PermissionError(13, "still in use")
        ),
    )
    with pytest.raises(PermissionError):
        token_launcher._remove_private_temp(str(private))
    assert len(attempts) == token_launcher._TEMP_REMOVAL_ATTEMPTS


def test_a_junction_in_the_private_temp_is_unlinked_instead_of_followed(tmp_path, monkeypatch):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    private = temp_root / ("e" * 24)
    (private / "nested").mkdir(parents = True)
    (private / "nested" / "scratch.txt").write_text("x", encoding = "utf-8")
    junction = private / "junction"
    junction.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding = "utf-8")
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(
        token_launcher, "_is_reparse_point", lambda path: str(path) == str(junction)
    )
    scanned: list[str] = []
    removed: list[str] = []
    real_scandir = os.scandir
    real_rmdir = os.rmdir

    def scandir(path = "."):
        scanned.append(str(path))
        return real_scandir(path)

    def rmdir(path, **kwargs):
        removed.append(str(path))
        return real_rmdir(path, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(os, "rmdir", rmdir)

    token_launcher._remove_private_temp(str(private))

    assert not private.exists()
    assert (outside / "keep.txt").exists()
    # The reparse point is removed as a link (RemoveDirectoryW), never walked.
    assert str(junction) in removed
    assert str(junction) not in scanned


def test_reconcile_refuses_a_manifest_whose_sid_names_a_real_account(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    temp_root = tmp_path / "temp"
    manifests.mkdir()
    temp_root.mkdir()
    (tmp_path / "work").mkdir()
    private = temp_root / ("a" * 24)
    private.mkdir()
    planted = _manifest(
        manifests, "S-1-5-21-1004336348-1177238915-682003330-1001",
        workdir = str(tmp_path / "work"), private_temp = str(private),
        granted_roots = [str(tmp_path / "work"), str(private)], owner_pid = 4242,
    )
    revoked: list[str] = []
    freed: list = []
    monkeypatch.setattr(token_launcher, "_manifest_root", lambda: str(manifests))
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(token_launcher, "_sid_from_text", lambda text: ctypes.c_void_p(9))
    monkeypatch.setattr(windows_lpac, "_process_identity", lambda pid = None: None)
    monkeypatch.setattr(windows_lpac, "_revoke_sid", lambda path, sid, **kw: revoked.append(path))
    monkeypatch.setattr(windows_lpac, "_api", lambda: _unmapped_sid_api(freed, resolves = True))

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    # A planted manifest must not turn into an ACL revoke on a real account's tree.
    assert planted.exists() and private.exists()
    assert revoked == []
    assert [item.value for item in freed] == [9]


def test_reconcile_removes_orphaned_temporary_manifests(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    temp_root = tmp_path / "temp"
    manifests.mkdir()
    temp_root.mkdir()
    prefix = token_launcher._MANIFEST_PREFIX
    dead = manifests / (prefix + "S-1-5-21-1-1-1-1.json.tmp")
    dead.write_text(json.dumps({"owner_pid": 4242, "owner_created": 7}), encoding = "utf-8")
    live = manifests / (prefix + "S-1-5-21-2-2-2-2.json.tmp")
    live.write_text(json.dumps({"owner_pid": 5151, "owner_created": 7}), encoding = "utf-8")
    partial = manifests / (prefix + "S-1-5-21-3-3-3-3.json.tmp")
    partial.write_text('{"owner_pid": 51', encoding = "utf-8")
    stale = manifests / (prefix + "S-1-5-21-4-4-4-4.json.tmp")
    stale.write_text('{"owner_pid": 51', encoding = "utf-8")
    old = time.time() - token_launcher._ORPHAN_TEMPORARY_MANIFEST_SECONDS - 60
    os.utime(stale, (old, old))
    foreign = manifests / "unrelated.json.tmp"
    foreign.write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(token_launcher, "_manifest_root", lambda: str(manifests))
    monkeypatch.setattr(token_launcher, "_temp_root", lambda: str(temp_root))
    monkeypatch.setattr(
        windows_lpac, "_process_identity", lambda pid = None: (5151, 7) if pid == 5151 else None
    )

    token_launcher.WindowsRestrictedTokenBackend().reconcile_stale_manifests()

    assert not dead.exists()  # its writer is gone, so os.replace will never run
    assert not stale.exists()  # unreadable, but older than any write can be
    assert live.exists() and partial.exists() and foreign.exists()


# ── os_sandbox selection and fallback (every platform, via a fake launcher) ──


class _FakeLauncher:
    identity = "windows-restricted-token"
    profile_id = "windows-restricted-token-write-isolation-v1"
    limitations = token_launcher._LIMITATIONS

    def __init__(self, *, available: bool = True, decline: Exception | None = None):
        self.available = available
        self.decline = decline
        self.probe_calls: list[bool] = []
        self.prepared: list[os_sandbox.ToolLaunchPlan] = []

    def probe(self, *, force: bool = False):
        self.probe_calls.append(force)
        if self.available:
            return os_sandbox.SandboxCapability(
                self.identity, True, "fake token probe passed", available = True,
                protection_state = "preview", profile_id = self.profile_id,
                limitations = self.limitations,
            )
        return os_sandbox.SandboxCapability(
            self.identity, False, "fake token probe failed", available = False
        )

    def prepare(self, spec):
        self.prepared.append(spec)
        if self.decline is not None:
            raise self.decline
        return os_sandbox.PreparedSandboxLaunch(
            argv = spec.argv, workdir = spec.workdir, env = {**spec.env, "TEMP": "private"},
            preexec_fn = None, backend = self.identity, timeout_seconds = spec.timeout_seconds,
            close_fds = spec.close_fds, terminate_descendants = spec.terminate_descendants,
            spawn_callback = lambda prepared, kwargs: None,
        )


def _limited_plan(tmp_path, monkeypatch, launcher):
    store = tool_isolation.LimitedGrantStore(ttl_seconds = 60, max_entries = 4)
    monkeypatch.setattr(tool_isolation, "_LIMITED_GRANTS", store)
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    monkeypatch.setattr(os_sandbox, "_environment_fingerprint", lambda _backend: "env-token")
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: launcher)
    capability = os_sandbox.capability_snapshot()
    grant = tool_isolation.issue_limited_grant(
        current_subject = "actor", tool_ui_session_id = "page",
        probe_generation = capability.probe_generation,
    )
    return os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", "pass"), workdir = str(tmp_path), env = {"PATH": "p"},
        requested_mode = "limited", current_subject = "actor", tool_ui_session_id = "page",
        limited_grant = grant.token,
    )


def test_limited_mode_runs_under_the_token_launcher_once_it_qualified(
    tmp_path, monkeypatch, isolated_capability_cache
):
    launcher = _FakeLauncher()
    plan = _limited_plan(tmp_path, monkeypatch, launcher)
    prepared = os_sandbox.prepare_tool_launch(plan)
    record = prepared.execution_record
    assert prepared.backend == "windows-restricted-token"
    assert prepared.spawn_callback is not None
    assert prepared.env["TEMP"] == "private"
    assert record.effective_mode == "limited"
    assert record.os_isolation is False
    assert record.backend == "windows-restricted-token"
    assert record.profile_id == "windows-restricted-token-write-isolation-v1"
    assert record.limitations == token_launcher._LIMITATIONS
    # The token leaves the host network reachable, which the record must not
    # contradict: "deny" would read as no network path out of the sandbox.
    assert record.network_policy == "unrestricted"
    assert set(os_sandbox._LIMITED_SAFEGUARDS) <= set(record.retained_safeguards)
    assert {"write_restricted_token", "job_object"} <= set(record.retained_safeguards)
    assert record.probe_generation
    assert launcher.prepared[0].workdir == os.path.realpath(str(tmp_path))
    assert launcher.prepared[0].requested_mode == "limited"


def test_limited_mode_falls_back_to_the_process_guard_when_the_launcher_declines(
    tmp_path, monkeypatch, isolated_capability_cache
):
    launcher = _FakeLauncher(decline = os_sandbox.SandboxUnavailableError("workdir too large"))
    plan = _limited_plan(tmp_path, monkeypatch, launcher)
    prepared = os_sandbox.prepare_tool_launch(plan)
    record = prepared.execution_record
    assert prepared.backend == "process-guard"
    assert prepared.spawn_callback is None
    assert record.backend == "process-guard"
    assert record.profile_id == "limited-software-safeguards-v1"
    assert record.retained_safeguards == os_sandbox._LIMITED_SAFEGUARDS
    assert "restricted_token_unavailable" in record.limitations
    assert record.network_policy == "unrestricted"
    assert record.os_isolation is False
    # An OS-level refusal (an ACL API error) falls back the same way; anything else propagates.
    launcher.decline = OSError(5, "denied")
    assert "restricted_token_unavailable" in os_sandbox.prepare_tool_launch(plan).execution_record.limitations
    launcher.decline = RuntimeError("bug")
    with pytest.raises(RuntimeError):
        os_sandbox.prepare_tool_launch(plan)


def test_limited_mode_ignores_a_launcher_that_did_not_qualify(
    tmp_path, monkeypatch, isolated_capability_cache
):
    launcher = _FakeLauncher(available = False)
    plan = _limited_plan(tmp_path, monkeypatch, launcher)
    prepared = os_sandbox.prepare_tool_launch(plan)
    record = prepared.execution_record
    assert record.backend == "process-guard"
    assert "restricted_token_unavailable" not in record.limitations
    assert launcher.prepared == []
    if sys.platform != "linux":
        assert "detached_descendant_cleanup_unverified" in record.limitations


def test_limited_grant_is_not_bound_to_the_launcher_probe(tmp_path, monkeypatch, isolated_capability_cache):
    # The token launcher qualifying (or not) must not invalidate grants: it is
    # deliberately outside probe_generation, so a grant issued before the launcher
    # probe still authorises the launch that now runs under the token.
    plan = _limited_plan(tmp_path, monkeypatch, _FakeLauncher(available = False))
    before = os_sandbox.capability_snapshot()
    os_sandbox._capability_cache.clear()
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: _FakeLauncher())
    after = os_sandbox.capability_snapshot()
    assert before.probe_generation == after.probe_generation
    assert before.limited_backend == "process-guard"
    assert after.limited_backend == "windows-restricted-token"
    prepared = os_sandbox.prepare_tool_launch(plan)
    assert prepared.execution_record.backend == "windows-restricted-token"


def test_capability_snapshot_reports_what_limited_runs_under(monkeypatch, isolated_capability_cache):
    os_backend = SimpleNamespace(
        identity = "fake-os",
        profile_id = "fake-os-v1",
        probe = lambda: os_sandbox.SandboxCapability("fake-os", False, "no bwrap", available = False),
    )
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: os_backend)
    monkeypatch.setattr(os_sandbox, "_environment_fingerprint", lambda _backend: "env-cap")
    launcher = _FakeLauncher()
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: launcher)

    first = os_sandbox.capability_snapshot()
    assert first.available is False
    assert first.limited_backend == "windows-restricted-token"
    assert first.limited_profile_id == "windows-restricted-token-write-isolation-v1"
    assert first.limited_limitations == token_launcher._LIMITATIONS
    assert "passed" in first.limited_reason
    assert launcher.probe_calls == [False]
    assert os_sandbox.capability_snapshot() is first  # cached with its limited fields
    forced = os_sandbox.capability_snapshot(force = True)
    assert launcher.probe_calls == [False, True]
    assert forced.limited_backend == "windows-restricted-token"

    os_sandbox._capability_cache.clear()
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: _FakeLauncher(available = False))
    unavailable = os_sandbox.capability_snapshot()
    assert unavailable.limited_backend == "process-guard"
    assert unavailable.limited_profile_id == "limited-software-safeguards-v1"
    assert unavailable.limited_limitations == ()
    assert unavailable.limited_reason == "fake token probe failed"

    os_sandbox._capability_cache.clear()
    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: None)
    plain = os_sandbox.capability_snapshot()
    assert plain.limited_backend == "process-guard"
    assert plain.limited_reason == ""


def test_capability_snapshot_survives_a_crashing_launcher_probe(monkeypatch, isolated_capability_cache):
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    monkeypatch.setattr(os_sandbox, "_environment_fingerprint", lambda _backend: "env-crash")

    class _Crashing(_FakeLauncher):
        def probe(self, *, force: bool = False):
            raise RuntimeError("ctypes exploded")

    monkeypatch.setattr(os_sandbox, "_limited_isolation_backend", lambda: _Crashing())
    unsupported = os_sandbox.capability_snapshot()
    assert unsupported.limited_backend == "process-guard"
    assert "RuntimeError: ctypes exploded" in unsupported.limited_reason
    os_backend = SimpleNamespace(
        identity = "fake-os", profile_id = "fake-os-v1",
        probe = lambda: os_sandbox.SandboxCapability("fake-os", False, "no", available = False),
    )
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: os_backend)
    os_sandbox._capability_cache.clear()
    capability = os_sandbox.capability_snapshot()
    assert capability.limited_backend == "process-guard"
    assert "RuntimeError: ctypes exploded" in capability.limited_reason


def test_tools_treats_both_job_owning_backends_alike():
    assert inference_tools._WINDOWS_JOB_OWNING_BACKENDS == {"windows-lpac", "windows-restricted-token"}
    source = Path(inference_tools.__file__).read_text(encoding = "utf-8")
    assert source.count("or prepared_launch.backend in _WINDOWS_JOB_OWNING_BACKENDS") == 2
    assert 'prepared_launch.backend == "windows-lpac"' not in source


def test_windows_lpac_exposes_the_shared_job_factory():
    lpac_source = Path(windows_lpac.__file__).read_text(encoding = "utf-8")
    assert "def _job_object_with_limits()" in lpac_source
    assert "_PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D" in lpac_source
    for name in ("OpenProcessToken", "CreateRestrictedToken", "SetTokenInformation",
                 "CreateProcessAsUserW", "IsTokenRestricted", "IsProcessInJob"):
        assert f"{name}.argtypes" in lpac_source, name
    assert windows_lpac.__all__ == ["WindowsLpacBackend", "WindowsLpacProcess"]


# ── live Windows tests ───────────────────────────────────────────────────────


@pytest.fixture(scope = "module")
def live_token_backend():
    if sys.platform != "win32":
        pytest.skip("native restricted-token tests run only on Windows")
    backend = os_sandbox._limited_isolation_backend()
    assert isinstance(backend, token_launcher.WindowsRestrictedTokenBackend)
    started = time.perf_counter()
    capability = backend.probe(force = True)
    print(f"restricted-token probe: {time.perf_counter() - started:.2f}s {capability.reason}")
    assert capability.available is True, capability.reason
    assert capability.profile_id == backend.profile_id
    assert capability.limitations == token_launcher._LIMITATIONS
    return backend


def _run_live(backend, workdir: Path, code: str, *argv: str, timeout: int = 60) -> tuple[str, int, float]:
    started = time.perf_counter()
    prepared = backend.prepare(
        os_sandbox.ToolLaunchPlan(
            argv = (sys.executable, "-I", "-S", "-c", code, *argv),
            workdir = str(workdir),
            env = {
                "PATH": os.pathsep.join(
                    (os.path.dirname(sys.executable), os.path.join(os.environ["SystemRoot"], "System32"))
                ),
                "PYTHONIOENCODING": "utf-8",
            },
        )
    )
    identity = prepared.spawn_callback._launch_identity
    process = None
    output = ""
    try:
        process = os_sandbox.spawn_prepared_launch(
            prepared, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, stdin = subprocess.DEVNULL,
            text = True, encoding = "utf-8", errors = "replace", cwd = prepared.workdir,
            env = prepared.env, close_fds = True,
        )
        process.wait(timeout = timeout)
        output = process.stdout.read()
        returncode = process.returncode
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout = 10)
        prepared.cleanup()
        assert prepared.cleanup_diagnostics == []
        assert not Path(identity.private_temp).exists()
        assert not Path(identity.manifest_path).exists()
        acl = subprocess.run(["icacls", str(workdir)], capture_output = True, text = True).stdout
        assert identity.sid_string not in acl
    return output, returncode, time.perf_counter() - started


def test_live_token_child_writes_only_the_workdir_and_private_temp(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    secret_root = Path(token_launcher._private_root("limited-probe-tests"))
    secret = secret_root / f"secret-{os.getpid()}.txt"
    secret.write_text("secret", encoding = "utf-8")
    try:
        output, returncode, elapsed = _run_live(
            live_token_backend, work,
            "import json, os, sys\n"
            "def w(p):\n"
            "    try:\n"
            "        open(p, 'a').write('x'); return True\n"
            "    except OSError as e:\n"
            "        return type(e).__name__\n"
            "print(json.dumps({'secret_read': open(sys.argv[1]).read(), 'secret_write': w(sys.argv[1]),"
            " 'work': w('out.txt'), 'temp': w(os.path.join(os.environ['TEMP'], 't.txt')),"
            " 'user_temp': w(os.path.join(sys.argv[2], 'u.txt')), 'exe': sys.executable}))",
            str(secret), str(secret_root),
        )
        assert returncode == 0, output
        report = json.loads(output.strip().splitlines()[-1])
        assert report["secret_read"] == "secret"  # reads keep the user's access (disclosed)
        assert report["secret_write"] == "PermissionError"
        assert report["user_temp"] == "PermissionError"
        assert report["work"] is True and (work / "out.txt").exists()
        assert report["temp"] is True
        assert os.path.normcase(report["exe"]) == os.path.normcase(sys.executable)
        print(f"restricted-token launch round trip: {elapsed:.2f}s")
        assert elapsed < 30
    finally:
        secret.unlink(missing_ok = True)


def test_live_token_child_keeps_nul_pipes_and_multiprocessing(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    output, returncode, _ = _run_live(
        live_token_backend, work,
        "import os\n"
        "open(os.devnull, 'r+b').close()\n"
        "import multiprocessing.connection as c\n"
        "a, b = c.Pipe(); a.send('ping'); assert b.recv() == 'ping'\n"
        "import subprocess, sys\n"
        "print(subprocess.run([sys.executable, '-I', '-S', '-c', 'print(\"grandchild ok\")'],"
        " capture_output=True, text=True).stdout.strip())\n"
        "print('ok')",
    )
    assert returncode == 0, output
    assert "grandchild ok" in output and output.strip().endswith("ok")


def test_live_token_is_restricted_lua_and_privilege_stripped(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    output, returncode, _ = _run_live(
        live_token_backend, work,
        token_launcher._PROBE_PAYLOAD, str(work / "missing-secret"), str(work), "",
    )
    assert returncode == 0, output
    findings = json.loads(output.strip().splitlines()[-1])
    assert findings["restricted"] is True
    assert findings["privileges"] <= 1
    assert findings["in_job"] is True
    assert "S-1-1-0" in findings["restricted_sids"]
    assert any(token_launcher._is_launch_sid_text(s) for s in findings["restricted_sids"])
    assert findings["devnull"] is True and findings["pipe"] is True


def test_live_detached_grandchild_dies_with_the_job(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    output, returncode, _ = _run_live(
        live_token_backend, work,
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-I', '-S', '-c', 'import time; time.sleep(300)'],"
        " creationflags=0x8 | 0x200)\n"  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        "print(p.pid)",
    )
    assert returncode == 0, output
    grandchild = int(output.strip().splitlines()[-1])
    deadline = time.time() + 10
    while time.time() < deadline and windows_lpac._process_identity(grandchild) is not None:
        time.sleep(0.2)
    assert windows_lpac._process_identity(grandchild) is None


def test_live_breakaway_is_refused_before_any_process_exists(live_token_backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    prepared = live_token_backend.prepare(
        os_sandbox.ToolLaunchPlan(argv = (sys.executable, "-c", "pass"), workdir = str(work), env = {})
    )
    try:
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "break away"):
            prepared.spawn_callback(
                prepared,
                {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                 "creationflags": 0x01000000},
            )
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "stdio plan"):
            prepared.spawn_callback(prepared, {"stdout": subprocess.PIPE})
    finally:
        prepared.cleanup()
    assert prepared.cleanup_diagnostics == []


def test_live_limited_tool_call_records_the_token_launcher(live_token_backend, tmp_path, monkeypatch):
    workdir = tmp_path / "limited-work"
    workdir.mkdir()
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    before = dict(os_sandbox._capability_cache)
    os_sandbox._capability_cache.clear()
    try:
        capability = os_sandbox.capability_snapshot(force = True)
        assert capability.available is False
        assert capability.limited_backend == "windows-restricted-token"
        grant = tool_isolation.issue_limited_grant(
            current_subject = "test:limited-user", tool_ui_session_id = "test-page-token",
            probe_generation = capability.probe_generation,
        )
        records = []
        result = inference_tools._python_exec(
            "import os\nopen('made-here.txt', 'w').write('x')\nprint(os.environ['TEMP'])",
            timeout = 60, tool_execution_mode = "limited", current_subject = "test:limited-user",
            tool_ui_session_id = "test-page-token", limited_grant = grant.token,
            launch_record_callback = records.append,
        )
    finally:
        os_sandbox._capability_cache.clear()
        os_sandbox._capability_cache.update(before)
    assert (workdir / "made-here.txt").exists(), result
    assert "limited-temp" in result
    assert len(records) == 1
    assert records[0].backend == "windows-restricted-token"
    assert records[0].effective_mode == "limited"
    assert "write_restricted_token" in records[0].retained_safeguards
    assert records[0].limitations == token_launcher._LIMITATIONS
    assert not Path(result.strip().splitlines()[-1]).exists()  # private temp removed after the call

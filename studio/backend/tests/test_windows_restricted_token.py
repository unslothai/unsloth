# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Write-restricted token launcher for Limited mode on Windows.

The first half runs everywhere and pins the launcher's contract (flags, restricting
SIDs, probe evaluation, manifest reconciliation, how os_sandbox selects it and how it
falls back). The second half runs only on Windows against a real restricted token.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
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
    # The restricting SIDs: the launch SID first, then the logon SID, then Everyone.
    assert "[identity.sid_string, *logon_sids[:1], _EVERYONE_SID]" in SOURCE
    # Administrators is disabled outright, and the default DACL is widened to the
    # launch SID so anonymous pipes the child creates stay writable to it.
    assert '_ADMINISTRATORS_SID = "S-1-5-32-544"' in SOURCE
    assert "_set_default_dacl(api, restricted, identity.sid)" in SOURCE
    # A process never runs outside its job: assignment precedes ResumeThread.
    spawn = SOURCE[SOURCE.index("def _spawn_restricted") :]
    assert spawn.index("AssignProcessToJobObject") < spawn.index("ResumeThread(")
    assert spawn.index("_job_object_with_limits()") < spawn.index("CreateProcessAsUserW(")


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


def test_manifest_parser_accepts_only_this_launchers_records(tmp_path):
    sid = "S-1-5-21-1-2-3-4"
    good = _manifest(tmp_path, sid)
    assert token_launcher._parse_manifest(good)["sid"] == sid
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, version = 2)) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, kind = "lpac")) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, "S-1-15-2-1-2-3-4-5")) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, granted_roots = ["rel"])) is None
    assert token_launcher._parse_manifest(_manifest(tmp_path, sid, owner_pid = "1")) is None
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
    dead = _manifest(manifests, dead_sid, private_temp = str(dead_temp),
                     granted_roots = [str(tmp_path / "work"), str(dead_temp)], owner_pid = 4242)
    live = _manifest(manifests, live_sid, private_temp = str(live_temp),
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
    monkeypatch.setattr(
        windows_lpac, "_api", lambda: SimpleNamespace(kernel32 = SimpleNamespace(LocalFree = freed.append))
    )

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
    monkeypatch.setattr(
        windows_lpac, "_api", lambda: SimpleNamespace(kernel32 = SimpleNamespace(LocalFree = lambda s: None))
    )
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
    calls.clear()
    monkeypatch.setattr(windows_lpac, "_revoke_sid", lambda path, sid, **kw: calls.append(path))
    identity.cleanup()
    assert identity.cleaned is True
    assert not manifest.exists()
    identity.cleanup()  # idempotent
    assert calls == [str(private), str(tmp_path / "work")]


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

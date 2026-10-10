# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-only; writes need a UI session, prepare also a loopback client (the UAC prompt is local)."""

from pathlib import Path
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
from core.inference import (
    mxc_host_prep_job,
    mxc_policy,
    mxc_probe,
    mxc_read_grants,
    mxc_runtime,
    os_sandbox,
    tools,
)
from utils import client_ip, mxc_isolation_settings
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")


def _cap(available = True, backend = "bubblewrap"):
    return os_sandbox.SandboxCapability(
        backend = backend,
        available = available,
        reason = "probe passed" if available else "probe failed",
        protection_state = "preview" if available else "unavailable",
        limitations = ("network_not_confined",),
    )


@pytest.fixture
def host(monkeypatch):
    calls = {"snapshot": 0, "invalidate": 0, "profile_reset": 0, "revoke": 0, "start": 0}
    saved = {"dacl": False, "grants": True}

    def snapshot(**_kw):
        calls["snapshot"] += 1
        return _cap()

    monkeypatch.setattr(os_sandbox, "capability_snapshot", snapshot)
    monkeypatch.setattr(settings, "_sandbox_windows_status", lambda: _windows_block(saved))
    monkeypatch.setattr(settings, "_sandbox_terminal_target", lambda: ("/bin/bash", None))
    monkeypatch.setattr(
        mxc_probe,
        "invalidate_cache",
        lambda: calls.__setitem__("invalidate", calls["invalidate"] + 1),
    )
    monkeypatch.setattr(
        tools,
        "reset_terminal_profile_cache",
        lambda: calls.__setitem__("profile_reset", calls["profile_reset"] + 1),
    )

    def revoke():
        calls["revoke"] += 1
        return ("C:\\runtime",)

    monkeypatch.setattr(mxc_read_grants, "revoke_recorded", revoke)
    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: saved["dacl"])
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: saved["grants"])
    monkeypatch.setattr(
        mxc_isolation_settings, "set_dacl_fallback_setting", lambda v: saved.__setitem__("dacl", v)
    )
    monkeypatch.setattr(
        mxc_isolation_settings,
        "set_persistent_grants_setting",
        lambda v: saved.__setitem__("grants", v),
    )
    monkeypatch.setattr(mxc_isolation_settings, "locked_by_environment", lambda _name: False)
    monkeypatch.setattr(mxc_runtime, "installation_identity", lambda: "identity")

    def start():
        calls["start"] += 1
        return mxc_host_prep_job.HostPrepJob(id = "job1")

    monkeypatch.setattr(mxc_host_prep_job, "start", start)
    monkeypatch.setattr(mxc_host_prep_job, "current", lambda: None)
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: True)
    settings._forget_sandbox_status()
    return calls, saved


def _windows_block(saved):
    return settings.SandboxWindowsStatus(
        runtime_installed = True,
        allow_dacl_fallback = saved["dacl"],
        allow_dacl_fallback_saved = saved["dacl"],
        dacl_locked_by_environment = False,
        persistent_read_grants = saved["grants"],
        persistent_read_grants_saved = saved["grants"],
        grants_locked_by_environment = False,
        host_prep_missing = ["prepare-null-device"],
    )


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")


@pytest.fixture
def posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")


def _client(account, via_api_key = False):
    app = FastAPI()
    app.include_router(settings.router)

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[settings.get_current_subject] = subject
    app.dependency_overrides[settings.authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app, raise_server_exceptions = False)


def test_owner_reads_status_and_it_is_cached_until_refresh(host, posix):
    calls, _saved = host
    with _client(OWNER) as client:
        first = client.get("/sandbox")
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["python"]["backend"] == "bubblewrap" and body["python"]["available"] is True
        assert body["terminal"]["limitations"] == ["network_not_confined"]
        assert body["windows"] is None
        client.get("/sandbox")
        assert calls["snapshot"] == 2  # python + terminal, once
        client.get("/sandbox", params = {"refresh": "true"})
        assert calls["snapshot"] == 4


def test_a_windows_refresh_picks_the_terminal_from_fresh_verdicts(host, windows, monkeypatch):
    from core.inference import sandbox_probe

    calls, _saved = host
    order = []
    monkeypatch.setattr(sandbox_probe, "reset_probe_cache", lambda: order.append("reset"))
    monkeypatch.setattr(
        settings, "_sandbox_terminal_target", lambda: order.append("target") or ("cmd.exe", None)
    )
    forced = []

    def snapshot(force = False, **kw):
        forced.append((kw["execution_kind"], force))
        return _cap()

    monkeypatch.setattr(os_sandbox, "capability_snapshot", snapshot)
    with _client(OWNER) as client:
        client.get("/sandbox", params = {"refresh": "true"})
    assert order == ["reset", "target"]
    assert calls["invalidate"] == 1 and calls["profile_reset"] == 1
    assert forced == [("python", True), ("terminal", False)]


def test_a_windows_refresh_still_records_what_it_just_checked(host, windows, monkeypatch):
    monkeypatch.setattr(settings, "_sandbox_terminal_target", lambda: ("cmd.exe", "cmd_isolated"))
    monkeypatch.setattr(settings, "_sandbox_setup_status", lambda _ready: None)
    os_sandbox.forget_tool_isolation()
    with _client(OWNER) as client:
        assert client.get("/sandbox", params = {"refresh": "true"}).status_code == 200
    assert os_sandbox.cached_tool_isolation("python") is True
    assert os_sandbox.cached_tool_isolation("terminal") is True
    os_sandbox.forget_tool_isolation()


def test_windows_status_carries_the_opt_in_block(host, windows):
    with _client(OWNER) as client:
        body = client.get("/sandbox").json()
    assert body["platform"] == "win32"
    assert body["windows"]["host_prep_missing"] == ["prepare-null-device"]
    assert body["windows"]["prepare_repeats_after_restart"] is True


@pytest.mark.parametrize(
    "dacl, capability, expected",
    [
        # With the fallback off wxc-exec runs only in BaseContainer, so a working MXC is that tier.
        (False, _cap(backend = "mxc-processcontainer"), True),
        (
            False,
            os_sandbox.SandboxCapability(
                backend = "mxc-processcontainer",
                available = False,
                reason = mxc_probe.NO_BUILTIN_CONTAINER_REASON,
            ),
            False,
        ),
        (False, _cap(available = False, backend = "mxc-processcontainer"), None),
        # With the fallback on, wxc-exec may have used either tier.
        (True, _cap(backend = "mxc-processcontainer"), None),
    ],
)
def test_windows_status_names_the_builtin_container(
    host, windows, monkeypatch, dacl, capability, expected
):
    _calls, saved = host
    saved["dacl"] = dacl
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_kw: capability)
    with _client(OWNER) as client:
        body = client.get("/sandbox").json()
    assert body["windows"]["builtin_container"] is expected


@pytest.mark.parametrize("dacl_before, dacl_after", [(True, False), (False, True)])
def test_a_save_during_the_probe_leaves_the_builtin_container_unknown(
    host, windows, monkeypatch, dacl_before, dacl_after
):
    """A verdict measured under one fallback setting says nothing about the tier under the other."""
    _calls, saved = host
    saved["dacl"] = dacl_before

    def snapshot(**_kw):
        saved["dacl"] = dacl_after  # the owner saved the switch while the probe ran
        return _cap(backend = "mxc-processcontainer")

    monkeypatch.setattr(os_sandbox, "capability_snapshot", snapshot)
    with _client(OWNER) as client:
        body = client.get("/sandbox").json()
    assert body["windows"]["allow_dacl_fallback"] is dacl_after
    assert body["windows"]["builtin_container"] is None


def test_without_the_runtime_the_builtin_container_is_unknown(host, windows, monkeypatch):
    block = _windows_block({"dacl": False, "grants": True}).model_copy(
        update = {"runtime_installed": False}
    )
    monkeypatch.setattr(settings, "_sandbox_windows_status", lambda: block)
    monkeypatch.setattr(
        os_sandbox, "capability_snapshot", lambda **_kw: _cap(backend = "mxc-processcontainer")
    )
    with _client(OWNER) as client:
        assert client.get("/sandbox").json()["windows"]["builtin_container"] is None


def test_managed_account_is_refused(host, windows):
    with _client(ALICE) as client:
        assert client.get("/sandbox").status_code == 403
        assert client.put("/sandbox", json = {"allow_dacl_fallback": True}).status_code == 403
        assert client.post("/sandbox/prepare").status_code == 403
    calls, saved = host
    assert saved["dacl"] is False and calls["start"] == 0


def test_api_key_cannot_change_or_prepare(host, windows):
    calls, saved = host
    with _client(OWNER, via_api_key = True) as client:
        assert client.put("/sandbox", json = {"allow_dacl_fallback": True}).status_code == 403
        assert client.post("/sandbox/prepare").status_code == 403
    assert saved["dacl"] is False and calls["start"] == 0


def test_turning_on_saves_and_resets_every_cache(host, windows):
    calls, saved = host
    with _client(OWNER) as client:
        client.get("/sandbox")
        response = client.put("/sandbox", json = {"allow_dacl_fallback": True})
    assert response.status_code == 200, response.text
    assert saved["dacl"] is True
    assert response.json()["windows"]["allow_dacl_fallback"] is True
    assert calls["invalidate"] == 1 and calls["profile_reset"] == 1
    assert calls["revoke"] == 0 and response.json()["grants_restored"] is None


def test_turning_off_takes_the_read_grants_back(host, windows):
    calls, saved = host
    saved["dacl"] = True
    with _client(OWNER) as client:
        response = client.put("/sandbox", json = {"allow_dacl_fallback": False})
    assert response.status_code == 200, response.text
    assert calls["revoke"] == 1 and response.json()["grants_restored"] == 1


def test_a_setting_held_by_the_environment_is_refused(host, windows, monkeypatch):
    monkeypatch.setattr(
        mxc_isolation_settings,
        "locked_by_environment",
        lambda name: name == mxc_policy.DACL_FALLBACK_ENV,
    )
    _calls, saved = host
    with _client(OWNER) as client:
        response = client.put("/sandbox", json = {"allow_dacl_fallback": True})
        assert response.status_code == 409
        assert mxc_policy.DACL_FALLBACK_ENV in response.json()["detail"]
        assert client.put("/sandbox", json = {"persistent_read_grants": False}).status_code == 200
    assert saved["dacl"] is False and saved["grants"] is False


@pytest.mark.parametrize("body", [{"allow_dacl_fallback": "yes"}, {"unknown": True}])
def test_malformed_bodies_are_refused(host, windows, body):
    with _client(OWNER) as client:
        assert client.put("/sandbox", json = body).status_code == 422


def test_settings_are_windows_only(host, posix):
    with _client(OWNER) as client:
        assert client.put("/sandbox", json = {"allow_dacl_fallback": True}).status_code == 409
        assert client.post("/sandbox/prepare").status_code == 409


def test_prepare_starts_one_job_from_the_local_console(host, windows):
    calls, _saved = host
    with _client(OWNER) as client:
        response = client.post("/sandbox/prepare")
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "running" and response.json()["id"] == "job1"
        assert client.get("/sandbox/prepare").json() == {
            "state": "idle",
            "id": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "output_tail": [],
            "steps": [],
        }
    assert calls["start"] == 1


def test_prepare_refused_while_another_setup_runs_is_a_conflict(host, windows, monkeypatch):
    from core.inference import mxc_host_prep_job, sandbox_setup_job

    def busy():
        raise sandbox_setup_job.SetupUnavailable(
            "Another sandbox setup (windows-runtime) is still running"
        )

    monkeypatch.setattr(mxc_host_prep_job, "start", busy)
    with _client(OWNER) as client:
        response = client.post("/sandbox/prepare")
    assert response.status_code == 409 and "windows-runtime" in response.json()["detail"]


def test_prepare_is_refused_from_a_remote_browser(host, windows, monkeypatch):
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: False)
    calls, _saved = host
    with _client(OWNER) as client:
        response = client.post("/sandbox/prepare")
    assert response.status_code == 403
    assert "computer running Unsloth" in response.json()["detail"]
    assert calls["start"] == 0


def test_a_loopback_peer_relaying_a_remote_browser_is_not_local():
    from types import SimpleNamespace

    def request(
        peer,
        host = "127.0.0.1:8888",
        **headers,
    ):
        return SimpleNamespace(
            client = SimpleNamespace(host = peer, port = 0), headers = {"host": host, **headers}
        )

    assert client_ip.is_direct_local_request(request("127.0.0.1")) is True
    # A reverse proxy or tunnel on this machine: the peer is loopback, the browser is not.
    for headers in ({"x-forwarded-for": "203.0.113.7"}, {"cf-connecting-ip": "203.0.113.7"}):
        assert client_ip.is_direct_local_request(request("127.0.0.1", **headers)) is False
    assert client_ip.is_direct_local_request(request("127.0.0.1", "studio.example.com")) is False
    assert client_ip.is_direct_local_request(request("192.168.1.10")) is False


def test_prepare_without_the_runtime_is_refused(host, windows, monkeypatch):
    def missing():
        raise mxc_runtime.MxcRuntimeUnavailable("the managed MXC runtime is not installed")

    monkeypatch.setattr(mxc_runtime, "installation_identity", missing)
    calls, _saved = host
    with _client(OWNER) as client:
        assert client.post("/sandbox/prepare").status_code == 409
    assert calls["start"] == 0


def test_a_status_built_before_a_save_is_not_cached(host, windows, monkeypatch):
    real_build = settings._build_sandbox_status

    def build_then_save(force):
        status = real_build(force)
        settings._forget_sandbox_status()  # a PUT landed while this status was being built
        return status

    monkeypatch.setattr(settings, "_build_sandbox_status", build_then_save)
    settings._sandbox_status()
    assert settings._sandbox_status_cache is None


@pytest.fixture
def memory(monkeypatch):
    from utils import sandbox_memory_limit

    stored = {}
    monkeypatch.delenv(sandbox_memory_limit.MEMORY_LIMIT_ENV, raising = False)
    monkeypatch.setattr(
        sandbox_memory_limit,
        "saved_memory_limit_gb",
        lambda: stored.get("gb", sandbox_memory_limit.DEFAULT_MEMORY_LIMIT_GB),
    )
    monkeypatch.setattr(
        sandbox_memory_limit, "set_memory_limit_gb", lambda v: stored.__setitem__("gb", v)
    )
    return stored


def test_posix_status_carries_the_memory_limit(host, posix, memory):
    with _client(OWNER) as client:
        body = client.get("/sandbox").json()
    assert body["memory"] == {
        "limit_gb": 8,
        "saved_gb": 8,
        "default_gb": 8,
        "min_gb": 1,
        "max_gb": 4096,
        "locked_by_environment": False,
    }


def test_saving_the_memory_limit_shows_at_once_and_repeats_cleanly(host, posix, memory):
    calls, _saved = host
    with _client(OWNER) as client:
        client.get("/sandbox")
        for _ in range(2):
            response = client.put("/sandbox", json = {"memory_limit_gb": 32})
            assert response.status_code == 200, response.text
            assert response.json()["memory"]["limit_gb"] == 32
        assert client.get("/sandbox").json()["memory"]["saved_gb"] == 32
    assert memory == {"gb": 32}
    # Memory-only saves never touch the Windows MXC state.
    assert calls["invalidate"] == 0 and calls["revoke"] == 0


def test_the_memory_limit_env_var_wins_and_refuses_a_save(host, posix, memory, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_AS_GB", "16")
    with _client(OWNER) as client:
        body = client.get("/sandbox").json()["memory"]
        assert body["limit_gb"] == 16 and body["locked_by_environment"] is True
        response = client.put("/sandbox", json = {"memory_limit_gb": 32})
    assert response.status_code == 409
    assert "UNSLOTH_STUDIO_SANDBOX_AS_GB" in response.json()["detail"]
    assert memory == {}


@pytest.mark.parametrize("value", [0, 4097, "16", 1.5, True])
def test_out_of_range_memory_limits_are_refused(host, posix, memory, value):
    with _client(OWNER) as client:
        assert client.put("/sandbox", json = {"memory_limit_gb": value}).status_code == 422
    assert memory == {}


def test_the_memory_limit_is_owner_and_ui_session_only(host, posix, memory):
    with _client(ALICE) as client:
        assert client.put("/sandbox", json = {"memory_limit_gb": 32}).status_code == 403
    with _client(OWNER, via_api_key = True) as client:
        assert client.put("/sandbox", json = {"memory_limit_gb": 32}).status_code == 403
    assert memory == {}


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_only_linux_has_the_memory_limit(host, memory, monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    with _client(OWNER) as client:
        assert client.get("/sandbox").json()["memory"] is None
        assert client.put("/sandbox", json = {"memory_limit_gb": 32}).status_code == 409
    assert memory == {}


def test_an_empty_windows_save_still_resets_every_cache(host, windows, memory):
    calls, _saved = host
    with _client(OWNER) as client:
        assert client.put("/sandbox", json = {}).status_code == 200
    assert calls["invalidate"] == 1 and calls["profile_reset"] == 1

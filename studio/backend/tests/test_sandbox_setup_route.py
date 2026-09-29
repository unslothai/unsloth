# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route policy for /api/settings/sandbox/setup: owner, UI session, and the computer running Unsloth."""

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
    os_sandbox,
    sandbox_setup_job,
    sandbox_setup_plan,
    tools,
)
from utils import client_ip, mxc_isolation_settings
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")


def _cap(available):
    return os_sandbox.SandboxCapability(
        backend = "bubblewrap",
        available = available,
        reason = "probe passed" if available else "probe failed",
        protection_state = "preview" if available else "unavailable",
        limitations = (),
    )


@pytest.fixture
def host(monkeypatch):
    calls = {"start": [], "revoke": 0}
    saved = {"dacl": False, "locked": False, "available": False}
    plan = {
        "value": sandbox_setup_plan.SetupPlan(
            platform = "linux",
            action = sandbox_setup_plan.LINUX_INSTALL,
            elevation = "sudo",
            steps = (("apt-get", "install", "-y", "bubblewrap"),),
            manual_command = "sudo apt-get install -y bubblewrap",
            reason = "bubblewrap (bwrap) is not installed.",
        )
    }
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_kw: _cap(saved["available"]))
    monkeypatch.setattr(settings, "_sandbox_terminal_target", lambda: ("/bin/bash", None))
    monkeypatch.setattr(settings, "_sandbox_windows_status", lambda: None)
    monkeypatch.setattr(sandbox_setup_plan, "detect", lambda *a, **k: plan["value"])
    calls["elevation_checks"] = 0

    def elevation(**_kw):
        calls["elevation_checks"] += 1
        return saved.get("elevation", ("sudo", "/usr/bin/sudo"))

    monkeypatch.setattr(sandbox_setup_plan, "linux_elevation", elevation)
    monkeypatch.setattr(mxc_probe, "invalidate_cache", lambda: None)
    monkeypatch.setattr(tools, "reset_terminal_profile_cache", lambda: None)
    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: saved["dacl"])
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: True)

    def revoke():
        calls["revoke"] += 1
        return ()

    monkeypatch.setattr(mxc_read_grants, "revoke_recorded", revoke)
    monkeypatch.setattr(
        mxc_isolation_settings, "set_dacl_fallback_setting", lambda v: saved.__setitem__("dacl", v)
    )
    monkeypatch.setattr(
        mxc_isolation_settings, "locked_by_environment", lambda _name: saved["locked"]
    )

    def start(operation):
        calls["start"].append(operation)
        return sandbox_setup_job.SetupJob(id = "job1", operation = operation)

    monkeypatch.setattr(sandbox_setup_job, "start", start)
    monkeypatch.setattr(sandbox_setup_job, "current", lambda: None)
    monkeypatch.setattr(mxc_host_prep_job, "current", lambda: None)
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: True)
    settings._forget_sandbox_status()
    return calls, saved, plan


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")


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


def test_owner_starts_the_linux_install_from_this_computer(host, linux):
    calls, _saved, _plan = host
    with _client(OWNER) as client:
        response = client.post("/sandbox/setup", json = {"operation": "linux-install"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["state"] == "running" and body["operation"] == "linux-install"
        assert client.get("/sandbox/setup").json()["state"] == "idle"
    assert calls["start"] == ["linux-install"]


def test_status_names_the_setup_for_this_host(host, linux):
    with _client(OWNER) as client:
        setup = client.get("/sandbox").json()["setup"]
    assert setup["action"] == "linux-install" and setup["elevation"] == "sudo"
    assert setup["manual_command"] == "sudo apt-get install -y bubblewrap"


def test_status_offers_the_button_only_to_this_computer(host, linux, monkeypatch):
    with _client(OWNER) as client:
        assert client.get("/sandbox").json()["setup"]["can_run"] is True
        monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: False)
        setup = client.get("/sandbox").json()["setup"]
    # The command still shows; only the button is local.
    assert setup["can_run"] is False and setup["manual_command"]


def _capability_client(account):
    from routes.sandbox_capability import router as capability_router

    app = FastAPI()
    app.include_router(capability_router, prefix = "/api/sandbox")

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[settings.get_current_subject] = subject
    return TestClient(app, raise_server_exceptions = False)


def test_capability_offers_setup_to_the_local_owner(host, linux):
    body = _capability_client(OWNER).get("/api/sandbox/capability").json()
    assert body["setup_action"] == "linux-install" and body["can_run_setup"] is True
    assert body["manual_command"] == "sudo apt-get install -y bubblewrap"


@pytest.mark.parametrize("who", ["other_account", "remote_owner"])
def test_capability_gives_everyone_else_only_the_command(host, linux, monkeypatch, who):
    calls, _saved, _plan = host
    account = ALICE
    if who == "remote_owner":
        account = OWNER
        monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: False)
    body = _capability_client(account).get("/api/sandbox/capability").json()
    assert body["setup_action"] is None and body["can_run_setup"] is False
    assert body["manual_command"] == "sudo apt-get install -y bubblewrap"
    # Nobody but the owner on this computer makes Unsloth run the elevation check.
    assert calls["elevation_checks"] == 0


def test_a_remote_status_read_never_checks_elevation(host, linux, monkeypatch):
    calls, _saved, _plan = host
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: False)
    with _client(OWNER) as client:
        setup = client.get("/sandbox").json()["setup"]
    assert setup["can_run"] is False and calls["elevation_checks"] == 0


def test_no_elevation_here_means_the_command_only(host, linux):
    calls, saved, _plan = host
    saved["elevation"] = (None, None)
    with _client(OWNER) as client:
        setup = client.get("/sandbox").json()["setup"]
    assert setup["can_run"] is False and setup["manual_command"]
    body = _capability_client(OWNER).get("/api/sandbox/capability").json()
    assert body["setup_action"] is None and body["can_run_setup"] is False


def test_the_capability_read_checks_a_tool_it_has_no_answer_for(host, linux, monkeypatch):
    from core.inference import os_sandbox

    monkeypatch.setenv(os_sandbox.WARMUP_DISABLE_ENV, "0")
    refreshed = []

    def refresh(tool, *, force = False):
        refreshed.append((tool, force))
        os_sandbox.note_tool_isolation(tool, True, backend = "bubblewrap", reason = "passed")
        return True

    monkeypatch.setattr(os_sandbox, "refresh_tool_isolation", refresh)
    os_sandbox.forget_tool_isolation()
    body = _capability_client(OWNER).get("/api/sandbox/capability").json()
    assert body["python_os_isolated"] is True and body["terminal_os_isolated"] is True
    assert refreshed == [("python", False), ("terminal", False)]
    refreshed.clear()
    # Answered now: no check; refresh=1 from the owner re-checks both, forced.
    _capability_client(OWNER).get("/api/sandbox/capability")
    assert refreshed == []
    _capability_client(OWNER).get("/api/sandbox/capability", params = {"refresh": "1"})
    assert refreshed == [("python", True), ("terminal", True)]
    refreshed.clear()
    _capability_client(ALICE).get("/api/sandbox/capability", params = {"refresh": "1"})
    assert refreshed == []
    os_sandbox.forget_tool_isolation()


def test_a_remote_browser_is_refused(host, linux, monkeypatch):
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: False)
    calls, _saved, _plan = host
    with _client(OWNER) as client:
        response = client.post("/sandbox/setup", json = {"operation": "linux-install"})
    assert response.status_code == 403 and "computer running Unsloth" in response.json()["detail"]
    assert calls["start"] == []


def test_api_keys_and_other_accounts_are_refused(host, linux):
    calls, _saved, _plan = host
    with _client(OWNER, via_api_key = True) as client:
        assert client.post("/sandbox/setup", json = {"operation": "linux-install"}).status_code == 403
    with _client(ALICE) as client:
        assert client.post("/sandbox/setup", json = {"operation": "linux-install"}).status_code == 403
        assert client.get("/sandbox/setup").status_code == 403
    assert calls["start"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"operation": "rm -rf /"},
        {"operation": "linux-install", "command": "id"},
        {"operation": "windows-setup", "consent_dacl_fallback": "yes"},
    ],
)
def test_nothing_but_a_known_operation_is_accepted(host, linux, body):
    calls, _saved, _plan = host
    with _client(OWNER) as client:
        assert client.post("/sandbox/setup", json = body).status_code == 422
    assert calls["start"] == []


def test_the_operation_must_match_the_platform(host, linux, monkeypatch):
    calls, _saved, _plan = host
    with _client(OWNER) as client:
        assert client.post("/sandbox/setup", json = {"operation": "windows-setup"}).status_code == 409
        assert (
            client.post("/sandbox/setup", json = {"operation": "windows-runtime"}).status_code == 409
        )
    monkeypatch.setattr(sys, "platform", "win32")
    with _client(OWNER) as client:
        assert client.post("/sandbox/setup", json = {"operation": "linux-install"}).status_code == 409
    monkeypatch.setattr(sys, "platform", "darwin")
    with _client(OWNER) as client:
        assert client.post("/sandbox/setup", json = {"operation": "linux-install"}).status_code == 409
    assert calls["start"] == []


def test_nothing_to_set_up_is_a_conflict(host, linux, monkeypatch):
    def unavailable(_operation):
        raise sandbox_setup_job.SetupUnavailable("OS isolation already works on this computer.")

    monkeypatch.setattr(sandbox_setup_job, "start", unavailable)
    with _client(OWNER) as client:
        response = client.post("/sandbox/setup", json = {"operation": "linux-install"})
    assert response.status_code == 409 and "already works" in response.json()["detail"]


def test_windows_consent_turns_the_opt_in_on_once_the_setup_is_accepted(host, windows, monkeypatch):
    calls, saved, _plan = host
    seen = []

    def start(operation):
        # Turned on first, the Python check could pass before preparation and the plan would
        # read "already works": the opt-in must still be off while the job is decided.
        seen.append(saved["dacl"])
        return sandbox_setup_job.SetupJob(id = "job1", operation = operation)

    monkeypatch.setattr(sandbox_setup_job, "start", start)
    with _client(OWNER) as client:
        response = client.post(
            "/sandbox/setup", json = {"operation": "windows-setup", "consent_dacl_fallback": True}
        )
    assert response.status_code == 200, response.text
    assert seen == [False] and saved["dacl"] is True


def test_windows_consent_is_not_saved_when_the_setup_is_refused(host, windows, monkeypatch):
    _calls, saved, _plan = host

    def unavailable(_operation):
        raise sandbox_setup_job.SetupUnavailable("OS isolation already works on this computer.")

    monkeypatch.setattr(sandbox_setup_job, "start", unavailable)
    with _client(OWNER) as client:
        response = client.post(
            "/sandbox/setup", json = {"operation": "windows-setup", "consent_dacl_fallback": True}
        )
    assert response.status_code == 409 and saved["dacl"] is False


def test_windows_runtime_only_is_a_windows_operation(host, windows):
    calls, saved, _plan = host
    with _client(OWNER) as client:
        response = client.post(
            "/sandbox/setup", json = {"operation": "windows-runtime", "consent_dacl_fallback": True}
        )
    assert response.status_code == 200 and calls["start"] == ["windows-runtime"]
    # Consent belongs to the full setup only.
    assert saved["dacl"] is False


def test_windows_setup_without_consent_leaves_the_opt_in_alone(host, windows):
    calls, saved, _plan = host
    with _client(OWNER) as client:
        assert client.post("/sandbox/setup", json = {"operation": "windows-setup"}).status_code == 200
    assert saved["dacl"] is False and calls["start"] == ["windows-setup"]


def test_windows_consent_held_off_by_the_environment_is_refused(host, windows):
    calls, saved, _plan = host
    saved["locked"] = True
    with _client(OWNER) as client:
        response = client.post(
            "/sandbox/setup", json = {"operation": "windows-setup", "consent_dacl_fallback": True}
        )
    assert response.status_code == 409 and mxc_policy.DACL_FALLBACK_ENV in response.json()["detail"]
    assert saved["dacl"] is False and calls["start"] == []


def test_the_prepare_job_is_reported_by_the_setup_route(host, windows, monkeypatch):
    prep = mxc_host_prep_job.HostPrepJob(id = "prep1")
    monkeypatch.setattr(mxc_host_prep_job, "current", lambda: prep)
    with _client(OWNER) as client:
        body = client.get("/sandbox/setup").json()
    assert (
        body["id"] == "prep1"
        and body["operation"] == "windows-setup"
        and body["state"] == "running"
    )


def test_a_setup_run_is_reported_by_the_prepare_route(host, windows, monkeypatch):
    # A Prepare click while setup runs is answered with that run; polling must keep seeing it.
    setup = sandbox_setup_job.SetupJob(id = "setup1", operation = "windows-setup")
    monkeypatch.setattr(sandbox_setup_job, "current", lambda: setup)
    with _client(OWNER) as client:
        body = client.get("/sandbox/prepare").json()
    assert body["id"] == "setup1" and body["state"] == "running"

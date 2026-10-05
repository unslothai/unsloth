# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Sandbox > Install runtime: the plan, the one-at-a-time job, and its route."""

from pathlib import Path
import sys
import time
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import platform

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
from core.inference import (
    mxc_host_prep_job,
    mxc_probe,
    mxc_runtime,
    sandbox_probe,
    sandbox_setup_job as job_mod,
    sandbox_setup_plan as plan_mod,
    tools,
)
from utils import client_ip
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")


class _Proc:
    def __init__(self, lines, code):
        self.stdout = iter(line + "\n" for line in lines)
        self._code = code

    def wait(self):
        return self._code


@pytest.fixture
def x64_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        sys, "getwindowsversion", lambda: _types.SimpleNamespace(build = 26100), raising = False
    )
    monkeypatch.setattr(plan_mod, "windows_runtime_installed", lambda: False)
    monkeypatch.setattr(
        mxc_runtime, "_installed_package_root", lambda: Path("C:/studio/mxc-runtime")
    )
    monkeypatch.setattr(mxc_host_prep_job, "current", lambda: None)
    resets = []
    monkeypatch.setattr(sandbox_probe, "reset_probe_cache", lambda: resets.append("probe"))
    monkeypatch.setattr(mxc_probe, "invalidate_cache", lambda: resets.append("mxc"))
    monkeypatch.setattr(tools, "reset_terminal_profile_cache", lambda: resets.append("terminal"))
    monkeypatch.setattr(job_mod, "_current", None)
    monkeypatch.setattr(job_mod, "_on_finish", [])
    return resets


def _wait(job, limit = 5.0):
    deadline = time.monotonic() + limit
    while job.state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    return job


@pytest.mark.parametrize(
    "machine, build, expected",
    [
        ("AMD64", 26100, None),
        ("x86_64", 26200, None),
        ("ARM64", 26100, "arch"),
        ("AMD64", 22631, "build"),
    ],
)
def test_only_x64_windows_11_24h2_or_newer_can_run_mxc(monkeypatch, machine, build, expected):
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(
        sys, "getwindowsversion", lambda: _types.SimpleNamespace(build = build), raising = False
    )
    assert plan_mod.windows_runtime_unsupported() == expected
    assert plan_mod.windows_runtime_supported() is (expected is None)


def test_the_plan_is_the_setup_ps1_install_call(x64_windows):
    plan = plan_mod.windows_runtime_plan()
    assert plan.action == plan_mod.WINDOWS_RUNTIME
    (step,) = plan.steps
    assert step[0] == sys.executable
    assert step[1].endswith("install_mxc_prebuilt.py") and Path(step[1]).is_file()
    assert step[2:] == ("--install-dir", str(Path("C:/studio/mxc-runtime")))
    assert plan.manual_command.startswith("& '")


def test_every_powershell_single_quote_stays_inside_the_pasted_path():
    path = "C:\\Users\\O’Neil‘s ‚x‛ 'y'\\python.exe"
    line = plan_mod.powershell_command([(path, "--flag")])
    assert line == "& 'C:\\Users\\O’’Neil‘‘s ‚‚x‛‛ ''y''\\python.exe' '--flag'"


def test_no_plan_when_installed_or_unsupported(x64_windows, monkeypatch):
    monkeypatch.setattr(plan_mod, "windows_runtime_installed", lambda: True)
    assert plan_mod.windows_runtime_plan().action is None
    monkeypatch.setattr(plan_mod, "windows_runtime_installed", lambda: False)
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    plan = plan_mod.windows_runtime_plan()
    assert plan.action is None and "x64" in plan.reason


def test_a_successful_install_resets_every_cache(x64_windows, monkeypatch):
    spawned = []
    monkeypatch.setattr(
        job_mod, "_spawn", lambda argv, *_env: spawned.append(argv) or _Proc(["installed"], 0)
    )
    finished = []
    job_mod.add_finish_hook(lambda: finished.append(True))
    job = _wait(job_mod.start(plan_mod.WINDOWS_RUNTIME))
    assert job.state == "succeeded" and job.exit_code == 0 and job.output_tail == ["installed"]
    assert len(spawned) == 1 and spawned[0][1].endswith("install_mxc_prebuilt.py")
    assert set(x64_windows) == {"probe", "mxc", "terminal"} and finished == [True]


def test_a_runtime_in_use_says_so(x64_windows, monkeypatch):
    monkeypatch.setattr(job_mod, "_spawn", lambda argv, *_env: _Proc(["busy"], 3))
    job = _wait(job_mod.start(plan_mod.WINDOWS_RUNTIME))
    assert job.state == "failed" and job.exit_code == 3
    assert "using the runtime" in job.note


def test_a_spawn_failure_is_a_failed_job(x64_windows, monkeypatch):
    def boom(argv, *_env):
        raise OSError("no python")

    monkeypatch.setattr(job_mod, "_spawn", boom)
    job = _wait(job_mod.start(plan_mod.WINDOWS_RUNTIME))
    assert job.state == "failed" and "no python" in job.output_tail[-1]


def test_one_install_at_a_time(x64_windows, monkeypatch):
    running = job_mod.SetupJob(id = "j1", operation = plan_mod.WINDOWS_RUNTIME)
    monkeypatch.setattr(job_mod, "_current", running)
    assert job_mod.start(plan_mod.WINDOWS_RUNTIME) is running


def test_never_beside_prepare_this_pc(x64_windows, monkeypatch):
    monkeypatch.setattr(
        mxc_host_prep_job, "current", lambda: mxc_host_prep_job.HostPrepJob(id = "prep")
    )
    with pytest.raises(job_mod.SetupUnavailable, match = "still running"):
        job_mod.start(plan_mod.WINDOWS_RUNTIME)


def test_unknown_operation_and_nothing_to_do_are_refused(x64_windows, monkeypatch):
    with pytest.raises(job_mod.SetupUnavailable):
        job_mod.start("format-the-disk")
    monkeypatch.setattr(plan_mod, "windows_runtime_installed", lambda: True)
    with pytest.raises(job_mod.SetupUnavailable, match = "already installed"):
        job_mod.start(plan_mod.WINDOWS_RUNTIME)


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


@pytest.fixture
def route(x64_windows, monkeypatch):
    started = []

    def start(operation):
        started.append(operation)
        return job_mod.SetupJob(id = "rt1", operation = operation)

    monkeypatch.setattr(job_mod, "start", start)
    return started


def test_owner_starts_the_install_even_from_a_remote_browser(route, monkeypatch):
    # No administrator prompt is involved, so the local-console rule of Prepare this PC does not apply.
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: False)
    with _client(OWNER) as client:
        response = client.post("/sandbox/setup", json = {"operation": "windows-runtime"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "running" and body["operation"] == "windows-runtime"
    assert route == ["windows-runtime"]


def test_reading_the_job_when_none_ran(route):
    with _client(OWNER) as client:
        assert client.get("/sandbox/setup").json()["state"] == "idle"


def test_api_keys_and_other_accounts_cannot_install(route):
    with _client(OWNER, via_api_key = True) as client:
        assert (
            client.post("/sandbox/setup", json = {"operation": "windows-runtime"}).status_code == 403
        )
    with _client(ALICE) as client:
        assert (
            client.post("/sandbox/setup", json = {"operation": "windows-runtime"}).status_code == 403
        )
        assert client.get("/sandbox/setup").status_code == 403
    assert route == []


@pytest.mark.parametrize(
    "body", [{"operation": "format-the-disk"}, {"operation": "windows-runtime", "x": 1}, {}]
)
def test_only_the_fixed_operation_is_accepted(route, body):
    with _client(OWNER) as client:
        assert client.post("/sandbox/setup", json = body).status_code == 422
    assert route == []


def test_not_on_other_platforms(route, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with _client(OWNER) as client:
        assert (
            client.post("/sandbox/setup", json = {"operation": "windows-runtime"}).status_code == 409
        )
    assert route == []


def test_an_unavailable_install_is_a_conflict(x64_windows, monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    with _client(OWNER) as client:
        response = client.post("/sandbox/setup", json = {"operation": "windows-runtime"})
    assert response.status_code == 409 and "x64" in response.json()["detail"]

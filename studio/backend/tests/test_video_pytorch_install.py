# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from utils import video_pytorch_install as install


@pytest.fixture(autouse = True)
def reset_job(monkeypatch):
    monkeypatch.setattr(install, "_job", {"status": "idle", "message": None})


def test_start_is_idempotent_and_does_not_block(monkeypatch):
    worker = Mock()
    monkeypatch.setattr(install, "_missing", lambda: True)
    monkeypatch.setattr(install.threading, "Thread", worker)
    assert install.start()["status"] == "installing"
    assert install.start()["status"] == "installing"
    worker.assert_called_once()
    worker.return_value.start.assert_called_once()


def test_other_capability_failures_cannot_install(monkeypatch):
    monkeypatch.setattr(install, "_missing", lambda: False)
    with pytest.raises(RuntimeError, match = "only available"):
        install.start()
    assert install.status()["status"] == "idle"


@pytest.mark.parametrize("exit_code,expected", [(0, "restart_required"), (1, "failed")])
def test_worker_targets_running_python_and_reports_result(monkeypatch, exit_code, expected):
    from utils import mlx_repair

    monkeypatch.setattr(mlx_repair, "_uv_executable", lambda: "/tools/uv")
    run = Mock(return_value = SimpleNamespace(returncode = exit_code))
    monkeypatch.setattr(install.subprocess, "run", run)
    install._run()
    assert install.status()["status"] == expected
    args, kwargs = run.call_args
    assert args[0] == [sys.executable, "-u", str(install._installer()), "--install-video-pytorch"]
    assert kwargs["env"]["VIRTUAL_ENV"] == sys.prefix
    assert kwargs["env"]["UNSLOTH_NO_TORCH"] == "0"
    assert kwargs["env"]["PATH"].startswith("/tools")
    if exit_code == 0:
        assert install.start()["status"] == "restart_required"


def test_worker_spawn_failure_allows_retry(monkeypatch):
    from utils import mlx_repair

    monkeypatch.setattr(mlx_repair, "_uv_executable", lambda: None)
    monkeypatch.setattr(install.subprocess, "run", Mock(side_effect = OSError("cannot launch")))
    install._run()
    assert install.status()["status"] == "failed"


def test_managed_account_cannot_install(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth import policy
    from auth.authentication import get_current_subject
    from utils.account_context import AccountContext, bind_account, reset_account
    from routes.video import router

    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    start = Mock()
    monkeypatch.setattr(install, "start", start)
    app = FastAPI()
    app.include_router(router, prefix = "/api/inference")

    async def subject():
        token = bind_account(AccountContext("a" * 32, "alice"))
        try:
            yield "alice"
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    with TestClient(app) as client:
        assert client.post("/api/inference/video/pytorch-install").status_code == 403
    start.assert_not_called()


def test_owner_can_install_and_read_progress(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth.authentication import get_current_subject
    from utils.account_context import OWNER, bind_account, reset_account
    from routes.video import router

    monkeypatch.setattr(install, "start", lambda: {"status": "installing", "message": None})
    app = FastAPI()
    app.include_router(router, prefix = "/api/inference")

    async def subject():
        token = bind_account(OWNER)
        try:
            yield OWNER.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    with TestClient(app) as client:
        assert client.post("/api/inference/video/pytorch-install").json()["status"] == "installing"
        assert client.get("/api/inference/video/pytorch-install").status_code == 200

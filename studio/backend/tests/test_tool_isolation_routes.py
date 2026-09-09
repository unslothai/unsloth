# SPDX-License-Identifier: AGPL-3.0-only
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.parametrize("force", [False, True])
def test_capability_http_route(monkeypatch, force):
    from routes import inference
    from core.inference import os_sandbox

    calls = []

    def capability_snapshot(*, force):
        calls.append(force)
        assert threading.current_thread().name != "MainThread"
        return os_sandbox.SandboxCapability("none", False, "test unavailable")

    monkeypatch.setattr(os_sandbox, "capability_snapshot", capability_snapshot)
    app = FastAPI()
    app.dependency_overrides[inference.get_current_subject] = lambda: "test-user"
    app.add_api_route("/capability", inference.tool_isolation_capability)
    with TestClient(app) as client:
        response = client.get("/capability", params = {"force": force})
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert calls == [force]


@pytest.mark.parametrize("repair", [False, True])
def test_setup_http_route(monkeypatch, repair):
    from routes import inference
    from core.inference import srt_setup

    calls = []

    def setup(*, repair_existing):
        calls.append(repair_existing)
        return {"success": True, "message": "test setup"}

    monkeypatch.setattr(inference.sys, "platform", "win32")
    monkeypatch.setattr(srt_setup, "install_windows_sandbox", setup)
    app = FastAPI()
    app.dependency_overrides[inference.get_current_subject] = lambda: "test-user"
    app.add_api_route("/setup", inference.setup_windows_tool_isolation, methods = ["POST"])
    with TestClient(app) as client:
        response = client.post("/setup", params = {"repair_existing": repair})
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert calls == [repair]

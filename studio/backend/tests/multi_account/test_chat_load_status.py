# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load's preflight identity must stay private before it obtains GPU residency."""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio/backend"))

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from auth.authentication import get_current_subject
from core.inference import gpu_arbiter
from hub.services.models import account_access as access
from routes import inference
from utils.account_context import bind_account, reset_account


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(inference.router, prefix = "/api/inference")
    return TestClient(app)


def test_foreign_preflight_model_is_not_in_status(monkeypatch, accounts):
    entered, finish = threading.Event(), threading.Event()
    secret_model = "alice/private-unreleased-checkpoint"
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(
        inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    monkeypatch.setattr(inference, "_probe_llama_cpp_status", lambda _: (False, {}))
    monkeypatch.setattr(inference, "_running_load_attempt", None)
    monkeypatch.setattr(inference, "_pending_load_attempts", {})
    inference.begin_load_lifecycle()

    def slow_access(reference):
        assert reference == secret_model
        entered.set()
        assert finish.wait(10)
        # End the simulated slow Hub authorization without downloading anything.
        raise HTTPException(status_code = 404, detail = "Model not found")

    monkeypatch.setattr(access, "require_model_access", slow_access)
    load_result = {}

    def load():
        with client_for(accounts["alice"]) as client:
            load_result["response"] = client.post(
                "/api/inference/load", json = {"model_path": secret_model}
            )

    worker = threading.Thread(target = load)
    worker.start()
    try:
        assert entered.wait(10), "real load handler never reached preflight"
        assert gpu_arbiter.current_owner() is None
        with client_for(accounts["alice"]) as client:
            mine = client.get("/api/inference/status")
            assert mine.status_code == 200
            assert secret_model in mine.json()["loading"]
        with client_for(accounts["unsloth"]) as client:
            assert secret_model in client.get("/api/inference/status").json()["loading"]
        with client_for(accounts["bob"]) as client:
            response = client.get("/api/inference/status")
        assert response.status_code == 200
        assert secret_model not in response.text, response.text
    finally:
        finish.set()
        worker.join(10)
        assert not worker.is_alive()

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.systemone import catalog, laya_runtime
from utils import systemone_settings
from utils.account_context import OWNER, run_as
from utils.paths import outputs_root

from .support import make_app

QUESTIONS = {"urgent": {"type": "noul", "instructions": "Does this need a reply now?"}}


def _answer(agent, state, questions):
    answers = {name: {"type": "noul", "noul": 0.5, "confidence": 0.5} for name in questions}
    return {"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}, False


@pytest.fixture
def client(accounts, monkeypatch):
    from routes import systemone
    from routes.settings import router as settings_router

    for name in (
        "UNSLOTH_SYSTEMONE_MODEL",
        "UNSLOTH_SYSTEMONE_DISABLE",
        "UNSLOTH_SYSTEMONE_DEVICE",
    ):
        monkeypatch.delenv(name, raising = False)
    for name in ("_agent", "_loaded", "_device_name", "_loader", "_loading", "_failure"):
        monkeypatch.setattr(laya_runtime, name, None)
    monkeypatch.setattr(systemone_settings, "runtime_unavailable_reason", lambda: None)
    monkeypatch.setattr(
        laya_runtime, "_load_checkpoint", lambda checkpoint: (SimpleNamespace(), "cpu")
    )
    monkeypatch.setattr(laya_runtime, "_predict", _answer)
    app = make_app()
    app.include_router(systemone.router, prefix = "/v1")
    app.include_router(settings_router, prefix = "/api/settings")
    with TestClient(app) as client:
        yield client
    if laya_runtime._loader is not None:
        laya_runtime._loader.join(5)


def _fine_tune(folder):
    path = run_as(OWNER, outputs_root) / folder
    for name in ("encoder", "tokenizer"):
        (path / name).mkdir(parents = True)
    (path / "model.safetensors").write_bytes(b"x")
    (path / "rl_agent_config.json").write_text("{}", encoding = "utf-8")
    return catalog.FINE_TUNE_PREFIX + folder


def _login(client, username):
    response = client.post(
        "/api/auth/login", json = {"username": username, "password": "account-password"}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_a_managed_account_sees_only_the_configured_fine_tune(client):
    served = _fine_tune("laya_served_1")
    private = _fine_tune("laya_hr_layoffs_q4_2")
    owner, bob = _login(client, "unsloth"), _login(client, "bob")
    saved = client.put(
        "/api/settings/systemone", json = {"enabled": True, "model": served}, headers = owner
    )
    assert saved.status_code == 200, saved.text
    # The owner's own API call makes the other fine-tune the resident model.
    decided = client.post(
        "/v1/systemone",
        json = {"model": private, "state": "x", "questions": QUESTIONS},
        headers = owner,
    )
    assert decided.status_code == 200, decided.text

    def view(headers):
        settings = client.get("/api/settings/systemone", headers = headers).json()
        listed = [m["name"] for m in settings["models"] if m["kind"] == "fine_tune"]
        return settings["model"], listed, settings["loaded_model"]

    assert view(owner) == (served, [private, served], private)
    assert view(bob) == (served, [served], None)


def test_a_managed_account_cannot_swap_in_another_catalog_checkpoint(client, monkeypatch):
    loads = []
    monkeypatch.setattr(
        laya_runtime,
        "_load_checkpoint",
        lambda checkpoint: (loads.append(checkpoint.name), (SimpleNamespace(), "cpu"))[1],
    )
    owner, alice = _login(client, "unsloth"), _login(client, "alice")
    saved = client.put(
        "/api/settings/systemone",
        json = {"enabled": True, "model": "laya-multilingual"},
        headers = owner,
    )
    assert saved.status_code == 200, saved.text

    def decide(model, headers):
        return client.post(
            "/v1/systemone",
            json = {"model": model, "state": "x", "questions": QUESTIONS},
            headers = headers,
        )

    assert decide("laya-multilingual", owner).status_code == 200
    # The configured model, by name or alias, still answers a managed account.
    assert decide("laya-multilingual", alice).status_code == 200
    assert decide("default", alice).status_code == 200
    swap = decide("laya-english", alice)
    assert swap.status_code == 403, swap.text
    assert swap.json()["detail"]["error_type"] == "permission_error"
    assert laya_runtime._loaded.name == "laya-multilingual"
    assert "laya-english" not in loads
    # The owner still picks any catalog entry.
    assert decide("laya-english", owner).status_code == 200
    assert laya_runtime._loaded.name == "laya-english"

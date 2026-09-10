# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed user's valid HF token must also work for capability and size probes."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from studio.backend.tests.test_account_lifecycle import auth_env, headers
from auth import storage
from hub.services.models import account_access as access
from utils.account_context import run_as

REPO = "org/private-model"
TOKEN = "hf_alice_private_access"


@pytest.fixture
def probe_env(auth_env, monkeypatch):
    from routes import models
    from core.inference import llama_cpp

    client, _, _ = auth_env
    storage.create_initial_user(
        "alice", "alice-password", "alice-jwt-secret-for-test-at-least-32-bytes"
    )
    storage.create_initial_user("bob", "bob-password", "bob-jwt-secret-for-test-at-least-32-bytes")
    client.app.include_router(models.router, prefix = "/api/models")
    monkeypatch.setattr(access, "_public_repos", {})
    calls = []

    def repo_info(
        repo_id,
        *,
        token = None,
        **kwargs,
    ):
        calls.append(token)
        if token != TOKEN:
            exc = Exception("Repository not found")
            exc.response = SimpleNamespace(status_code = 404)
            raise exc
        return SimpleNamespace(private = True, gated = False)

    monkeypatch.setattr(access, "HfApi", lambda: SimpleNamespace(repo_info = repo_info))
    monkeypatch.setattr(llama_cpp, "_hf_offline_if_unreachable_for", lambda _: nullcontext())
    monkeypatch.setattr(models, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(models, "is_vision_model", lambda *a, **k: True)
    monkeypatch.setattr(models, "is_embedding_model", lambda *a, **k: True)
    monkeypatch.setattr(models, "_export_size_cached", lambda *a, **k: (123, 45, "hub"))
    return client, calls


def _token_headers(username = "alice"):
    return {**headers(username), "X-Unsloth-HF-Token": TOKEN, "X-HF-Token": TOKEN}


def test_check_vision_accepts_callers_valid_token(probe_env):
    client, _ = probe_env
    alice = storage.get_account("alice")
    assert run_as(alice, access.model_grants) == set()
    response = client.get(f"/api/models/check-vision/{REPO}", headers = _token_headers())
    print(f"check-vision: HTTP {response.status_code} body={response.text}")
    assert response.status_code == 200, response.text
    assert response.json()["is_vision"] is True
    assert run_as(alice, access.model_grants) == set()


def test_check_embedding_accepts_callers_valid_token(probe_env):
    client, _ = probe_env
    response = client.get(f"/api/models/check-embedding/{REPO}", headers = _token_headers())
    print(f"check-embedding: HTTP {response.status_code} body={response.text}")
    assert response.status_code == 200, response.text
    assert response.json()["is_embedding"] is True


def test_export_size_accepts_callers_valid_token(probe_env):
    client, _ = probe_env
    response = client.get(
        "/api/models/export-size", params = {"model": REPO}, headers = _token_headers()
    )
    print(f"export-size: HTTP {response.status_code} body={response.text}")
    assert response.status_code == 200, response.text
    assert response.json()["fp16_bytes"] == 123


@pytest.mark.parametrize("token", [None, "hf_wrong"])
@pytest.mark.parametrize("path", ["check-vision/" + REPO, "check-embedding/" + REPO, "export-size"])
def test_missing_or_wrong_token_remains_hidden(probe_env, token, path):
    client, _ = probe_env
    request_headers = headers("bob")
    if token:
        request_headers["X-Unsloth-HF-Token"] = token
        request_headers["X-HF-Token"] = token
    params = {"model": REPO} if path == "export-size" else None
    response = client.get(f"/api/models/{path}", params = params, headers = request_headers)
    assert response.status_code == 404, response.text


def test_owner_is_unaffected(probe_env):
    client, _ = probe_env
    for path, params in (
        ("check-vision/" + REPO, None),
        ("check-embedding/" + REPO, None),
        ("export-size", {"model": REPO}),
    ):
        response = client.get(
            f"/api/models/{path}", params = params, headers = _token_headers("unsloth")
        )
        assert response.status_code == 200, response.text

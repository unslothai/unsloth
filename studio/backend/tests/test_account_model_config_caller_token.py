# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed user's valid HF token must work before their first model download."""

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
def config_env(auth_env, monkeypatch):
    from routes import models
    from core.inference import llama_cpp
    from utils.models import model_config

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

    # Stand in only for Hub metadata and model introspection; real HTTP auth,
    # account policy, grants, and authorization checks remain in use.
    monkeypatch.setattr(access, "HfApi", lambda: SimpleNamespace(repo_info = repo_info))
    monkeypatch.setattr(llama_cpp, "_hf_offline_if_unreachable_for", lambda _: nullcontext())
    monkeypatch.setattr(models, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(models, "load_model_defaults", lambda _: {})
    monkeypatch.setattr(models, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(models, "is_embedding_model", lambda *a, **k: False)
    monkeypatch.setattr(model_config, "detect_audio_type_checked", lambda *a, **k: (None, True))
    monkeypatch.setattr(
        models.ModelConfig, "from_identifier", lambda *a, **k: SimpleNamespace(is_lora = False)
    )
    monkeypatch.setattr(models, "_get_max_position_embeddings", lambda _: 4096)
    monkeypatch.setattr(models, "_get_model_size_bytes", lambda *a, **k: 123)
    return client, calls


def test_first_private_model_config_accepts_callers_valid_token(config_env):
    client, calls = config_env
    alice = storage.get_account("alice")
    assert run_as(alice, access.model_grants) == set()
    response = client.get(
        f"/api/models/config/{REPO}",
        headers = {
            **headers("alice"),
            "X-Unsloth-HF-Token": TOKEN,
        },
    )
    print(
        f"Valid caller token: HTTP {response.status_code}, body={response.json()}, Hub tokens={calls}"
    )
    assert response.status_code == 200, response.text
    assert response.json()["max_position_embeddings"] == 4096
    assert TOKEN in calls
    assert (
        run_as(alice, access.model_grants) == set()
    ), "metadata preflight must not grant cached content"


@pytest.mark.parametrize("token", [None, "hf_wrong"])
def test_missing_or_wrong_token_remains_hidden(config_env, token):
    client, _ = config_env
    request_headers = headers("bob")
    if token:
        request_headers["X-Unsloth-HF-Token"] = token
    assert client.get(f"/api/models/config/{REPO}", headers = request_headers).status_code == 404


def test_caller_token_does_not_authorize_foreign_local_path(config_env, tmp_path):
    client, _ = config_env
    foreign = tmp_path / "foreign-model"
    foreign.mkdir()
    response = client.get(
        f"/api/models/config/{REPO}",
        params = {"local_path": str(foreign)},
        headers = {
            **headers("alice"),
            "X-Unsloth-HF-Token": TOKEN,
        },
    )
    assert response.status_code == 404


def test_owner_and_managed_account_with_grant_can_read_config(config_env):
    client, _ = config_env
    run_as(storage.get_account("alice"), access.record_model_grant, REPO)
    for username in ("unsloth", "alice"):
        response = client.get(
            f"/api/models/config/{REPO}",
            headers = {
                **headers(username),
                "X-Unsloth-HF-Token": TOKEN,
            },
        )
        assert response.status_code == 200, response.text


def test_cache_only_selection_still_needs_account_grant(config_env):
    client, _ = config_env
    response = client.get(
        f"/api/models/config/{REPO}",
        params = {"prefer_local_cache": "true"},
        headers = {
            **headers("alice"),
            "X-Unsloth-HF-Token": TOKEN,
        },
    )
    assert response.status_code == 404

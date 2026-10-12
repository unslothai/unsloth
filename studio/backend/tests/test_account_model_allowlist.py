# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-set per-account model allowlists (#11650)."""

import importlib.util
import secrets
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth import authentication, policy, storage
from hub.services.models import account_access as access
from state import active_generations
from utils.account_context import OWNER, bind_account, reset_account, run_as
from utils.paths import storage_roots


def _load_route(name):
    path = Path(__file__).resolve().parents[1] / "routes" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_allowlist_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(storage_roots.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "studio" / "auth" / "auth.db")
    monkeypatch.setattr(
        storage, "_BOOTSTRAP_PW_PATH", storage.DB_PATH.parent / ".bootstrap_password"
    )
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    monkeypatch.setattr(access, "_public_repos", {})
    monkeypatch.setattr(access, "_allowlists", {})
    # Every repo is public and readable, so only the allowlist can refuse.
    monkeypatch.setattr(
        access,
        "HfApi",
        lambda **_k: SimpleNamespace(
            repo_info = lambda *a, **k: SimpleNamespace(private = False, gated = False),
            auth_check = lambda *a, **k: None,
        ),
    )
    storage._reset_api_key_hash_cache()
    token = bind_account(OWNER)
    policy.invalidate_account_cache()
    active_generations.reset_for_tests()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(64))
    for username in ("alice", "bob"):
        storage.create_initial_user(username, f"{username}-password", secrets.token_urlsafe(64))
    accounts = _load_route("accounts")
    app = FastAPI()
    app.include_router(accounts.router, prefix = "/api/accounts")
    with TestClient(app) as client:
        yield client
    active_generations.reset_for_tests()
    reset_account(token)
    policy.invalidate_account_cache()
    storage._reset_api_key_hash_cache()


def headers(username = "unsloth"):
    return {"Authorization": "Bearer " + authentication.create_access_token(username)}


def put_access(
    client,
    target,
    allowlist,
    actor = "unsloth",
):
    account = storage.get_account(target)
    return client.put(
        f"/api/accounts/{account.account_id}/model-access",
        headers = headers(actor),
        json = {"model_allowlist": allowlist},
    )


def as_user(username, fn, *args, **kwargs):
    return run_as(storage.get_account(username), fn, *args, **kwargs)


def test_normalize_accepts_ids_urls_quants_and_patterns():
    assert storage.normalize_model_allowlist(None) is None
    assert storage.normalize_model_allowlist([]) is None
    assert storage.normalize_model_allowlist(["", "  "]) is None
    assert storage.normalize_model_allowlist(
        [
            " Unsloth/Qwen3-4B-GGUF ",
            "https://huggingface.co/unsloth/qwen3-4b-gguf",
            "unsloth/Qwen3-4B-GGUF:Q4_K_M",
            "meta-llama/*",
            "GPT2",
        ]
    ) == ["unsloth/qwen3-4b-gguf", "meta-llama/*", "gpt2"]
    for bad in ("*", "../etc/passwd", "*/model", "a/b/c", "org/na me"):
        with pytest.raises(ValueError):
            storage.normalize_model_allowlist([bad])


def test_schema_upgrade_is_idempotent_and_defaults_to_unrestricted(env):
    conn = storage.get_connection()
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(auth_user)")]
    finally:
        conn.close()
    assert columns.count("model_allowlist") == 1
    # A pre-existing install: drop the column, reopen twice, data intact and unrestricted.
    raw = sqlite3.connect(storage.DB_PATH)
    raw.execute("ALTER TABLE auth_user DROP COLUMN model_allowlist")
    raw.commit()
    raw.close()
    storage._auth_schema_ready.clear()
    for _ in range(2):
        storage.get_connection().close()
    accounts = {row["username"]: row for row in storage.list_accounts()}
    assert set(accounts) == {"unsloth", "alice", "bob"}
    assert all(row["model_allowlist"] is None for row in accounts.values())
    assert storage.account_model_allowlist(storage.get_account("alice").account_id) is None


@pytest.mark.parametrize("actor", ["alice", "bob"])
def test_only_the_owner_can_set_model_access(env, actor):
    response = put_access(env, "alice", ["unsloth/*"], actor = actor)
    assert response.status_code == 403
    assert storage.account_model_allowlist(storage.get_account("alice").account_id) is None


def test_owner_cannot_be_restricted_and_bad_input_is_refused(env):
    assert put_access(env, "unsloth", ["unsloth/*"]).status_code == 400
    assert put_access(env, "alice", ["not a repo"]).status_code == 422
    response = env.put(
        "/api/accounts/missing/model-access", headers = headers(), json = {"model_allowlist": None}
    )
    assert response.status_code == 404


def test_set_is_idempotent_and_listed(env):
    for _ in range(2):
        response = put_access(env, "alice", ["Unsloth/*", "unsloth/*"])
        assert response.status_code == 200
        assert response.json()["model_allowlist"] == ["unsloth/*"]
    listed = {
        row["username"]: row
        for row in env.get("/api/accounts", headers = headers()).json()["accounts"]
    }
    assert listed["alice"]["model_allowlist"] == ["unsloth/*"]
    assert listed["bob"]["model_allowlist"] is None
    assert put_access(env, "alice", []).json()["model_allowlist"] is None


def test_allowlist_gates_listing_loading_and_downloading(env):
    assert put_access(env, "alice", ["unsloth/*", "google/gemma-3-1b-it"]).status_code == 200
    for repo in ("unsloth/Qwen3-4B-GGUF", "Google/Gemma-3-1B-it"):
        assert as_user("alice", access.repo_visible, repo)
        as_user("alice", access.require_model_access, repo)
        as_user("alice", access.authorize_download, repo, "model", None)
    as_user("alice", access.require_model_access, "unsloth/Qwen3-4B-GGUF:Q4_K_M")

    assert not as_user("alice", access.repo_visible, "meta-llama/Llama-3.2-1B")
    for gate in (
        lambda: as_user("alice", access.require_model_access, "meta-llama/Llama-3.2-1B"),
        lambda: as_user(
            "alice", access.authorize_download, "meta-llama/Llama-3.2-1B", "model", None
        ),
    ):
        with pytest.raises(HTTPException) as refused:
            gate()
        assert refused.value.status_code == 403
        assert refused.value.detail == access.MODEL_NOT_ALLOWED

    rows = [{"id": "unsloth/Qwen3-4B-GGUF"}, {"id": "meta-llama/Llama-3.2-1B"}]
    assert as_user("alice", access.filter_model_rows, rows) == rows[:1]
    # Datasets are not models; the allowlist leaves them alone.
    assert as_user("alice", access.repo_visible, "meta-llama/some-dataset", "dataset")
    # Other accounts and the owner are unaffected.
    assert as_user("bob", access.filter_model_rows, rows) == rows
    assert run_as(OWNER, access.filter_model_rows, rows) == rows


def test_allowlist_applies_to_cached_snapshot_paths(env, tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    snapshot = cache / "models--meta-llama--Llama-3.2-1B" / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [cache])
    assert as_user("alice", access.model_visible, str(snapshot))
    assert put_access(env, "alice", ["unsloth/*"]).status_code == 200
    assert not as_user("alice", access.model_visible, str(snapshot))
    with pytest.raises(HTTPException) as refused:
        as_user("alice", access.require_model_access, str(snapshot))
    assert refused.value.status_code == 403


def test_changes_apply_on_the_next_request(env):
    rows = [{"id": "meta-llama/Llama-3.2-1B"}]
    assert as_user("alice", access.filter_model_rows, rows) == rows
    put_access(env, "alice", ["unsloth/*"])
    assert as_user("alice", access.filter_model_rows, rows) == []
    put_access(env, "alice", None)
    assert as_user("alice", access.filter_model_rows, rows) == rows


def test_a_grant_from_an_earlier_download_does_not_outrank_the_allowlist(env):
    as_user("alice", access.record_model_grant, "meta-llama/Llama-3.2-1B")
    assert as_user("alice", access.repo_visible, "meta-llama/Llama-3.2-1B")
    put_access(env, "alice", ["unsloth/*"])
    assert not as_user("alice", access.repo_visible, "meta-llama/Llama-3.2-1B")


def test_setting_access_drops_resident_shares(env):
    alice = storage.get_account("alice").account_id
    access._resident_sharers["chat"] = {alice, "someone-else"}
    try:
        put_access(env, "alice", ["unsloth/*"])
        assert access._resident_sharers["chat"] == {"someone-else"}
    finally:
        access._resident_sharers.pop("chat", None)


def test_training_jobs_refuse_a_base_model_outside_the_allowlist(env):
    from core.training import account_jobs as jobs

    put_access(env, "alice", ["unsloth/*"])
    as_user("alice", jobs.validate_job_paths, {"model_name": "unsloth/Qwen3-4B", "hf_token": "t"})
    for key, model in (
        ("model_name", "meta-llama/Llama-3.2-1B"),
        ("model_name", "gpt2"),
        ("base_model", "meta-llama/Llama-3.2-1B"),
        ("base_model_id", "meta-llama/Llama-3.2-1B"),
    ):
        with pytest.raises(HTTPException) as refused:
            as_user("alice", jobs.validate_job_paths, {key: model, "hf_token": "t"})
        assert refused.value.status_code == 403
    as_user(
        "bob", jobs.validate_job_paths, {"model_name": "meta-llama/Llama-3.2-1B", "hf_token": "t"}
    )
    # A local load is named by its path; the display name is not a Hub fetch.
    own = storage_roots.workspace_root
    local = run_as(storage.get_account("alice"), own) / "outputs" / "run1"
    local.mkdir(parents = True)
    as_user(
        "alice",
        jobs.validate_job_paths,
        {"model_name": "run1", "model_local_path": str(local), "hf_token": "t"},
    )

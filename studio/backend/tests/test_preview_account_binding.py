# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A public preview link opens the outputs of the account that minted it.

The owner's tokens keep their shape, so links handed out before this change
still work. A managed account's token names the account and is served inside
its outputs; a run of the same name in the owner's outputs stays closed.
"""

import json
import secrets
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.preview as preview
import utils.preview_token as preview_token
from auth import policy, storage
from utils.account_context import OWNER, AccountContext, run_as
from utils.paths import outputs_root

_SECRET = b"unit-test-preview-secret-0123456789"


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "studio" / "auth" / "auth.db")
    (tmp_path / "studio" / "auth").mkdir(parents = True)
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(preview_token, "get_or_create_preview_link_secret", lambda: _SECRET)
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: True)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield tmp_path
    policy.invalidate_account_cache()


def _managed(username: str) -> AccountContext:
    account = storage.issue_account_setup_code(username = username)["account"]
    return AccountContext(account["account_id"], account["username"], "user")


def _make_run(account: AccountContext, name: str, marker: str) -> Path:
    run = Path(run_as(account, outputs_root)) / name
    run.mkdir(parents = True)
    (run / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": marker}))
    return run


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(preview.router, prefix = "/p")
    return TestClient(app)


def test_owner_tokens_keep_their_shape_and_open_the_owner_root(studio):
    token = run_as(OWNER, preview_token.sign_preview_ref, "demorun")
    assert "." not in token
    assert preview_token.preview_token_account("demorun", token) == OWNER
    assert preview_token.verify_preview_ref("demorun", token)
    assert not preview_token.verify_preview_ref("otherrun", token)


def test_managed_token_names_its_account_and_only_that_account(studio):
    alice = _managed("alice")
    bob = _managed("bob")
    token = run_as(alice, preview_token.sign_preview_ref, "demorun")
    account_id, _, signature = token.partition(".")
    assert account_id == alice.account_id and signature
    assert preview_token.preview_token_account("demorun", token) == alice
    # Same secret, same ref, other account: a different token.
    assert run_as(bob, preview_token.sign_preview_ref, "demorun") != token
    assert run_as(OWNER, preview_token.sign_preview_ref, "demorun") != token
    # Relabelling the id does not carry the signature over.
    forged = f"{bob.account_id}.{signature}"
    assert preview_token.preview_token_account("demorun", forged) is None
    # The owner's signature is not accepted for a managed id either.
    owner_mac = run_as(OWNER, preview_token.sign_preview_ref, "demorun")
    assert preview_token.preview_token_account("demorun", f"{alice.account_id}.{owner_mac}") is None
    assert preview_token.preview_token_account("demorun", owner_mac) == OWNER


def test_deactivated_or_deleted_account_links_stop_working(studio):
    alice = _managed("alice")
    token = run_as(alice, preview_token.sign_preview_ref, "demorun")
    storage.set_account_active(alice.account_id, False)
    assert preview_token.preview_token_account("demorun", token) is None
    storage.set_account_active(alice.account_id, True)
    assert preview_token.preview_token_account("demorun", token) == alice
    storage.delete_account(alice.account_id, lambda account: None)
    assert preview_token.preview_token_account("demorun", token) is None


def test_public_request_is_served_in_the_minting_accounts_outputs(studio):
    alice = _managed("alice")
    owner_run = _make_run(OWNER, "victim-run", "OWNER_ADAPTER")
    alice_run = _make_run(alice, "victim-run", "ALICE_ADAPTER")
    assert owner_run != alice_run
    alice_token = run_as(alice, preview_token.sign_preview_ref, "victim-run")
    owner_token = run_as(OWNER, preview_token.sign_preview_ref, "victim-run")
    client = _client()

    # Each capability resolves its own run.
    owner_models = client.get("/p/victim-run/v1/models", params = {"k": owner_token})
    alice_models = client.get("/p/victim-run/v1/models", params = {"k": alice_token})
    assert owner_models.status_code == 200 and alice_models.status_code == 200
    assert client.get("/p/victim-run", params = {"k": alice_token}).status_code == 200

    # Remove alice's copy: her capability must not fall through to the owner's run.
    (alice_run / "adapter_config.json").unlink()
    alice_run.rmdir()
    assert owner_run.is_dir()
    assert client.get("/p/victim-run/v1/models", params = {"k": alice_token}).status_code == 404
    assert client.get("/p/victim-run", params = {"k": alice_token}).status_code == 404
    assert client.get("/p/victim-run/v1/models", params = {"k": owner_token}).status_code == 200

    # And the owner's capability never opens alice's run.
    alice_run.mkdir()
    (alice_run / "adapter_config.json").write_text("{}")
    owner_run.joinpath("adapter_config.json").unlink()
    owner_run.rmdir()
    assert client.get("/p/victim-run/v1/models", params = {"k": owner_token}).status_code == 404
    assert client.get("/p/victim-run/v1/models", params = {"k": alice_token}).status_code == 200


def test_chat_route_binds_the_minting_account_for_the_whole_request(studio, monkeypatch):
    alice = _managed("alice")
    _make_run(alice, "demorun", "ALICE_ADAPTER")
    token = run_as(alice, preview_token.sign_preview_ref, "demorun")
    seen = {}

    async def fake_load(load_request, request, subject):
        from utils.account_context import current_account
        seen["load_account"] = current_account().account_id
        seen["path"] = load_request.model_path

    async def fake_chat(payload, request, subject):
        from utils.account_context import current_account
        seen["chat_account"] = current_account().account_id
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    monkeypatch.setattr(preview, "load_model_for_preview", fake_load)
    monkeypatch.setattr(preview, "openai_chat_completions", fake_chat)
    monkeypatch.setattr(preview, "check_rate_limit", lambda ip: None)
    client = _client()
    response = client.post(
        "/p/demorun/v1/chat/completions",
        params = {"k": token},
        json = {"model": "demorun", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text
    assert seen["load_account"] == alice.account_id
    assert seen["chat_account"] == alice.account_id
    assert str(Path(run_as(alice, outputs_root)) / "demorun") in seen["path"]

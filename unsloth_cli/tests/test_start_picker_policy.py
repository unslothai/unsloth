# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Managed-account pickers omit cached models the server refuses to load."""

import secrets
import urllib.error

import pytest
from typer.testing import CliRunner

import unsloth_cli.commands.start as start
from unsloth_cli import _model_catalog as catalog
from unsloth_cli._inference import ensure_studio_backend_path
from unsloth_cli.tests.test_start import fake_studio
from unsloth_cli.tests.test_start_model_picker import picker, _empty_server, _loads

REAL_CATALOG = catalog.list_chat_models
KEY = "sk-unsloth-policy-example"


@pytest.fixture()
def policy_catalog(tmp_path, monkeypatch, picker, fake_studio, request):
    ensure_studio_backend_path(seed_cache_env = False)
    from auth import policy, storage
    from fastapi import HTTPException
    from hub.services.models import account_access
    from utils import hf_cache_settings
    from utils.account_context import run_as

    fmt, restriction = request.param
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "web-ui"))
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    account_id = storage.issue_account_setup_code(username = "alice")["account"]["account_id"]
    account = storage.get_account("alice")
    suffix = "-GGUF" if fmt == "gguf" else ""
    denied_id, allowed_id = (f"org/{name}{suffix}" for name in ("Denied", "Allowed"))
    if restriction == "allowlist":
        storage.set_account_model_allowlist(account_id, [allowed_id])
    monkeypatch.setattr(
        account_access,
        "repo_is_public",
        lambda repo, *args, **kwargs: restriction == "allowlist"
        or repo.lower() == allowed_id.lower(),
    )
    cache = tmp_path / "hub"
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    rows = []
    for repo_id in (denied_id, allowed_id):
        repo = cache / ("models--" + repo_id.replace("/", "--"))
        snapshot = repo / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents = True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("a" * 40)
        if fmt == "gguf":
            (snapshot / "model-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 4092)
        rows.append({"repo_id": repo_id, "cache_path": str(repo), "model_format": fmt})
    monkeypatch.setattr(
        catalog, "_cached_catalog_rows", lambda: (rows, []) if fmt == "gguf" else ([], rows)
    )
    monkeypatch.setattr(catalog, "trained_entries", lambda: [])
    monkeypatch.setattr(catalog, "exported_entries", lambda: [])
    monkeypatch.setattr(catalog, "local_folder_entries", lambda: [])
    monkeypatch.setattr(catalog, "list_chat_models", REAL_CATALOG)
    expected = catalog._cached_gguf_load_id(rows[1]) if fmt == "gguf" else allowed_id
    _empty_server(fake_studio, monkeypatch)
    previous = start._http_json
    checked = []

    def http(
        method,
        url,
        key,
        payload = None,
        **kwargs,
    ):
        if url.endswith("/api/auth/account"):
            return {
                "account_id": account.account_id,
                "username": account.username,
                "role": account.role,
            }
        if url.endswith("/api/auth/model-access"):
            checked.append(payload)
            if fake_studio.state.get("access_error"):
                raise urllib.error.HTTPError(
                    url, fake_studio.state["access_error"], "unavailable", None, None
                )
            if "access_reply" in fake_studio.state:
                return fake_studio.state["access_reply"]

            def allowed():
                grants = account_access.model_grants()
                return [
                    reference
                    for reference in payload
                    if account_access.model_visible(reference, grants = grants)
                ]

            return {"allowed": run_as(account, allowed)}
        if url.endswith("/api/inference/load"):
            try:
                run_as(account, account_access.require_model_access, payload["model_path"])
            except HTTPException as error:
                raise urllib.error.HTTPError(url, error.status_code, error.detail, None, None)
        return previous(method, url, key, payload = payload, **kwargs)

    monkeypatch.setattr(start, "_http_json", http)
    picker["answers"] = [""]
    try:
        yield fmt, expected, checked
    finally:
        policy.invalidate_account_cache()


@pytest.mark.parametrize(
    "policy_catalog",
    [
        ("gguf", "allowlist"),
        ("safetensors", "allowlist"),
        ("gguf", "private-repo"),
        ("safetensors", "private-repo"),
    ],
    indirect = True,
)
def test_managed_picker_excludes_forbidden_cached_models(policy_catalog, fake_studio):
    fmt, expected, checked = policy_catalog
    agent = "codex" if fmt == "gguf" else "opencode"
    result = CliRunner().invoke(start.start_app, [agent, "--no-launch", "--api-key", KEY])
    assert result.exit_code == 0, result.output
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": expected}]
    assert len(checked) == 1
    assert "1. Denied" not in result.output


@pytest.mark.parametrize("policy_catalog", [("gguf", "allowlist")], indirect = True)
@pytest.mark.parametrize("access_mode", ["404", "legacy", "empty"])
def test_unavailable_or_empty_model_access_never_offers_unverified_choices(
    policy_catalog, fake_studio, access_mode
):
    if access_mode == "404":
        fake_studio.state["access_error"] = 404
    elif access_mode == "legacy":
        fake_studio.state["access_reply"] = {"error": "API endpoint not found"}
    else:
        fake_studio.state["access_reply"] = {"allowed": []}
    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch", "--api-key", KEY])
    assert result.exit_code == 1
    assert _loads(fake_studio) == []
    assert "Your models" not in result.output
    assert len(policy_catalog[2]) == 1

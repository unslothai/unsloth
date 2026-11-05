# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""An idle-server picker scans the account authenticated by the selected API key."""

import json
import urllib.error

import pytest
import typer

import unsloth_cli.commands.start as start
from unsloth_cli import _model_catalog as catalog
from unsloth_cli._inference import ensure_studio_backend_path
from unsloth_cli.tests.test_start import BASE, fake_studio
from unsloth_cli.tests.test_start_model_picker import picker, _empty_server, _loads

ALICE = {"account_id": "a" * 32, "username": "alice", "role": "user"}
OWNER = {"account_id": "owner", "username": "unsloth", "role": "owner"}
KEY = "sk-unsloth-alice-example"


@pytest.fixture()
def account_catalog(tmp_path, monkeypatch, picker, fake_studio):
    ensure_studio_backend_path(seed_cache_env = False)
    from utils.account_context import current_account

    root = tmp_path / "web-ui"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(root))
    targets = {}
    for account, workspace in (("owner", root), ("alice", root / "accounts" / ALICE["account_id"])):
        path = workspace / "outputs" / f"{account}-private"
        path.mkdir(parents = True)
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "unsloth/gemma-3-270m-it",
                    "peft_type": "LORA",
                }
            )
        )
        (path / "adapter_model.safetensors").write_bytes(b"placeholder-not-loaded")
        targets[account] = str(path)
    monkeypatch.setattr(catalog, "_runs_by_output_dir", lambda: {})
    scans = []

    def inventory():
        scans.append(current_account().account_id)
        return catalog.trained_entries()

    monkeypatch.setattr(catalog, "list_chat_models", inventory)
    _empty_server(fake_studio, monkeypatch)
    previous = start._http_json
    fake_studio.state["picker_account"] = ALICE

    def http(method, url, key, *args, **kwargs):
        if url.endswith("/api/auth/model-access"):
            return {"allowed": args[0]}
        if url.endswith("/api/auth/account"):
            assert key == KEY
            code = fake_studio.state.get("account_error")
            if code:
                raise urllib.error.HTTPError(url, code, "test account error", None, None)
            return fake_studio.state["picker_account"]
        return previous(method, url, key, *args, **kwargs)

    monkeypatch.setattr(start, "_http_json", http)
    picker["answers"] = [""]
    return targets, scans


@pytest.mark.parametrize("key_source", ["explicit", "saved"])
def test_idle_picker_scopes_model_scan_to_authenticated_account(
    account_catalog, fake_studio, key_source
):
    from utils.account_context import current_account

    targets, scans = account_catalog
    if key_source == "saved":
        start._remember_key(start._key_cache_path(), BASE, KEY, "saved")
    start._connect(KEY if key_source == "explicit" else None, None, launch = False)
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": targets["alice"]}]
    assert scans == [ALICE["account_id"]]
    assert current_account().is_owner


def test_idle_picker_still_lists_owner_models(account_catalog, fake_studio):
    targets, scans = account_catalog
    fake_studio.state["picker_account"] = OWNER
    start._connect(KEY, None, launch = False)
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": targets["owner"]}]
    assert scans == ["owner"]


def test_idle_picker_restores_context_after_cancel(account_catalog, monkeypatch):
    from utils.account_context import AccountContext, bind_account, current_account, reset_account

    previous = AccountContext("b" * 32, "bob")
    token = bind_account(previous)

    def cancel(prompt):
        assert current_account().account_id == ALICE["account_id"]
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", cancel)
    try:
        with pytest.raises(typer.Exit):
            start._connect(KEY, None, launch = False)
        assert current_account() == previous
    finally:
        reset_account(token)


def test_idle_picker_restores_context_after_scan_error(account_catalog, monkeypatch):
    from utils.account_context import current_account

    def fail():
        assert current_account().account_id == ALICE["account_id"]
        raise RuntimeError("scan failed")

    monkeypatch.setattr(catalog, "list_chat_models", fail)
    with pytest.raises(RuntimeError, match = "scan failed"):
        start._connect(KEY, None, launch = False)
    assert current_account().is_owner


@pytest.mark.parametrize("code", [404, 500])
def test_idle_picker_does_not_scan_an_unknown_account(account_catalog, fake_studio, code):
    _, scans = account_catalog
    fake_studio.state["account_error"] = code
    with pytest.raises(typer.Exit):
        start._connect(KEY, None, launch = False)
    assert scans == []
    assert _loads(fake_studio) == []


def test_idle_picker_accepts_legacy_missing_route_response(account_catalog, fake_studio):
    _, scans = account_catalog
    fake_studio.state["picker_account"] = {"error": "API endpoint not found"}
    with pytest.raises(typer.Exit):
        start._connect(KEY, None, launch = False)
    assert scans == []
    assert _loads(fake_studio) == []

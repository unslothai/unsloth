# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
import json
import secrets
import urllib.error

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli.commands.start as start
from unsloth_cli import _model_catalog as catalog
from unsloth_cli._inference import ensure_studio_backend_path
from unsloth_cli.tests.test_start import BASE, fake_studio  # noqa: F401
from unsloth_cli.tests.test_start_model_picker import _empty_server, _loads, _route, picker  # noqa: F401

ALICE = {"account_id": "a" * 32, "username": "alice", "role": "user"}
OWNER = {"account_id": "owner", "username": "unsloth", "role": "owner"}
KEY = "sk-unsloth-alice-example"
REAL_CATALOG = catalog.list_chat_models


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
        adapter = {"base_model_name_or_path": "unsloth/gemma-3-270m-it", "peft_type": "LORA"}
        (path / "adapter_config.json").write_text(json.dumps(adapter))
        (path / "adapter_model.safetensors").write_bytes(b"placeholder-not-loaded")
        targets[account] = str(path)
    monkeypatch.setattr(catalog, "_runs_by_output_dir", lambda: {})
    scans = []

    def inventory():
        scans.append(current_account().account_id)
        return catalog.trained_entries()

    monkeypatch.setattr(catalog, "list_chat_models", inventory)
    _empty_server(fake_studio, monkeypatch)
    fake_studio.state["picker_account"] = ALICE

    def account(url, key, payload):
        assert key == KEY
        code = fake_studio.state.get("account_error")
        if code:
            raise urllib.error.HTTPError(url, code, "test account error", None, None)
        return fake_studio.state["picker_account"]

    _route(
        monkeypatch,
        {
            "/api/auth/model-access": lambda url, key, payload: {"allowed": payload},
            "/api/auth/account": account,
        },
    )
    picker["answers"] = [""]
    return targets, scans


@pytest.mark.parametrize(
    "key_source, account", [("explicit", ALICE), ("saved", ALICE), ("explicit", OWNER)]
)
def test_idle_picker_scans_the_authenticated_account(
    account_catalog, fake_studio, key_source, account
):
    from utils.account_context import current_account

    targets, scans = account_catalog
    fake_studio.state["picker_account"] = account
    if key_source == "saved":
        start._remember_key(start._key_cache_path(), BASE, KEY, "saved")

    start._connect(KEY if key_source == "explicit" else None, None, launch = False)

    name = "owner" if account is OWNER else "alice"
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": targets[name]}]
    assert scans == [account["account_id"]]
    assert current_account().is_owner


@pytest.mark.parametrize("root_name", ["MTP", "dspark"])
def test_picked_file_validation_uses_the_authenticated_scan_root(
    account_catalog, fake_studio, monkeypatch, root_name
):
    from storage import studio_db
    from utils.account_context import AccountContext, current_account, run_as
    from utils.models.model_config import _direct_gguf_for_variant
    from utils.paths import workspace_root

    account = AccountContext(ALICE["account_id"], ALICE["username"], ALICE["role"])
    root = run_as(account, workspace_root) / root_name
    root.mkdir(parents = True)
    # macOS tmp_path lives under /private/var, which the deny list refuses.
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    file = root / "model-Q8_0.gguf"
    file.write_bytes(b"GGUF" + bytes(4092))
    run_as(account, studio_db.add_scan_folder, str(root))
    assert run_as(account, _direct_gguf_for_variant, str(file), "Q8_0") == str(file)
    assert _direct_gguf_for_variant(str(file), "Q8_0") is None

    def inventory(*, gguf_variant):
        assert current_account().account_id == account.account_id
        return [catalog._gguf_variant_entry(root, gguf_variant, "Downloaded", "model")]

    monkeypatch.setattr(catalog, "list_chat_models", inventory)
    load = start.LoadOptions(gguf_variant = "Q8_0")
    start._connect(KEY, None, load = load, launch = False, gguf_only = True)

    assert _loads(fake_studio) == [
        {"require_idle": True, "model_path": str(file), "gguf_variant": "Q8_0"}
    ]
    assert current_account().is_owner


@pytest.mark.parametrize("failure", ["cancel", "scan-error"])
def test_idle_picker_restores_the_callers_account(account_catalog, monkeypatch, failure):
    from utils.account_context import AccountContext, bind_account, current_account, reset_account

    def inside_alice():
        assert current_account().account_id == ALICE["account_id"]

    def cancel(prompt):
        inside_alice()
        raise KeyboardInterrupt

    def scan_error():
        inside_alice()
        raise RuntimeError("scan failed")

    if failure == "cancel":
        monkeypatch.setattr("builtins.input", cancel)
    else:
        monkeypatch.setattr(catalog, "list_chat_models", scan_error)
    previous = AccountContext("b" * 32, "bob")
    token = bind_account(previous)
    try:
        with pytest.raises((typer.Exit, RuntimeError)):
            start._connect(KEY, None, launch = False)
        assert current_account() == previous
    finally:
        reset_account(token)


@pytest.mark.parametrize("reply", [404, 500, {"error": "API endpoint not found"}])
def test_idle_picker_does_not_scan_an_unknown_account(account_catalog, fake_studio, reply):
    _, scans = account_catalog
    if isinstance(reply, int):
        fake_studio.state["account_error"] = reply
    else:
        fake_studio.state["picker_account"] = reply

    with pytest.raises(typer.Exit):
        start._connect(KEY, None, launch = False)

    assert scans == []
    assert _loads(fake_studio) == []


@pytest.fixture()
def policy_catalog(tmp_path, monkeypatch, picker, fake_studio, request):
    """Alice may load only org/Allowed*, through an allowlist or because org/Denied* is private."""
    ensure_studio_backend_path(seed_cache_env = False)
    from auth import policy, storage
    from fastapi import HTTPException
    from hub.services.models import account_access
    from utils import hf_cache_settings
    from utils.account_context import run_as

    fmt, restriction = request.param
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "web-ui"))
    for name, value in [
        ("DB_PATH", tmp_path / "auth.db"),
        ("_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password"),
        ("_bootstrap_password", None),
    ]:
        monkeypatch.setattr(storage, name, value)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    storage.issue_account_setup_code(username = "alice")
    account = storage.get_account("alice")
    suffix = "-GGUF" if fmt == "gguf" else ""
    denied_id, allowed_id = f"org/Denied{suffix}", f"org/Allowed{suffix}"
    if restriction == "allowlist":
        storage.set_account_model_allowlist(account.account_id, [allowed_id])
    monkeypatch.setattr(
        account_access,
        "repo_is_public",
        lambda repo, *a, **k: restriction == "allowlist" or repo.lower() == allowed_id.lower(),
    )
    cache = tmp_path / "hub"
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    rows = []
    for repo_id in (denied_id, allowed_id):
        repo = cache / ("models--" + repo_id.replace("/", "--"))
        (repo / "snapshots" / ("a" * 40)).mkdir(parents = True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("a" * 40)
        if fmt == "gguf":
            (repo / "snapshots" / ("a" * 40) / "model-Q4_K_M.gguf").write_bytes(
                b"GGUF" + bytes(4092)
            )
        rows.append({"repo_id": repo_id, "cache_path": str(repo), "model_format": fmt})
    monkeypatch.setattr(
        catalog, "_cached_catalog_rows", lambda: (rows, []) if fmt == "gguf" else ([], rows)
    )
    for source in ("trained_entries", "exported_entries", "local_folder_entries"):
        monkeypatch.setattr(catalog, source, lambda: [])
    monkeypatch.setattr(catalog, "list_chat_models", REAL_CATALOG)
    expected = catalog._cached_gguf_load_id(rows[1]) if fmt == "gguf" else allowed_id
    _empty_server(fake_studio, monkeypatch)
    checked = []

    def model_access(url, key, payload):
        checked.append(payload)
        if fake_studio.state.get("access_error"):
            raise urllib.error.HTTPError(url, fake_studio.state["access_error"], "", None, None)
        if "access_reply" in fake_studio.state:
            return fake_studio.state["access_reply"]

        def allowed():
            grants = account_access.model_grants()
            return [m for m in payload if account_access.model_visible(m, grants = grants)]

        return {"allowed": run_as(account, allowed)}

    previous = start._http_json

    def load(url, key, payload):
        try:
            run_as(account, account_access.require_model_access, payload["model_path"])
        except HTTPException as error:
            raise urllib.error.HTTPError(url, error.status_code, error.detail, None, None)
        return previous("POST", url, key, payload)

    _route(
        monkeypatch,
        {
            "/api/auth/account": lambda url, key, payload: {
                "account_id": account.account_id,
                "username": account.username,
                "role": account.role,
            },
            "/api/auth/model-access": model_access,
            "/api/inference/load": load,
        },
    )
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
@pytest.mark.parametrize(
    "state",
    [
        {"access_error": 404},
        {"access_reply": {"error": "API endpoint not found"}},
        {"access_reply": {"allowed": []}},
    ],
)
def test_unverified_model_access_never_offers_a_choice(policy_catalog, fake_studio, state):
    fake_studio.state.update(state)

    result = CliRunner().invoke(start.start_app, ["codex", "--no-launch", "--api-key", KEY])

    assert result.exit_code == 1
    assert _loads(fake_studio) == []
    assert "Your models" not in result.output
    assert len(policy_catalog[2]) == 1

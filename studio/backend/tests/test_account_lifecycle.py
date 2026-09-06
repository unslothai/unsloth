# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real SQLite and HTTP coverage for owner-managed account lifecycle and setup."""

import importlib.util
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from auth import authentication, hashing, policy, storage
from state import active_generations
from utils.account_context import OWNER, bind_account, reset_account, run_as
from utils.paths import storage_roots
from unsloth_cli.commands import studio as studio_cli


def _load_route(name):
    path = Path(__file__).resolve().parents[1] / "routes" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_lifecycle_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(storage_roots.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "studio" / "auth" / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", storage.DB_PATH.parent / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    token = bind_account(OWNER)
    policy.invalidate_account_cache()
    active_generations.reset_for_tests()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(64))
    auth = _load_route("auth")
    accounts = _load_route("accounts")
    app = FastAPI()
    app.include_router(auth.router, prefix = "/api/auth")
    app.include_router(accounts.router, prefix = "/api/accounts")
    app.state.bootstrap_password = "owner-bootstrap"
    with TestClient(app) as client:
        yield client, auth, accounts
    active_generations.reset_for_tests()
    reset_account(token)
    policy.invalidate_account_cache()
    storage._reset_api_key_hash_cache()


@pytest.fixture
def matrix(auth_env):
    for username in ("alice", "bob"):
        storage.create_initial_user(username, f"{username}-password", secrets.token_urlsafe(64))
    return auth_env


def headers(username = "unsloth"):
    return {"Authorization": "Bearer " + authentication.create_access_token(username)}


def login(client, username, password):
    return client.post("/api/auth/login", json = {"username": username, "password": password})


def create(client, username = "carol"):
    return client.post("/api/accounts", headers = headers(), json = {"username": username})


def db_rows(table):
    conn = storage.get_connection()
    try:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id")]
    finally:
        conn.close()


@pytest.mark.parametrize("actor", ["unsloth", "alice", "bob"])
@pytest.mark.parametrize("operation", ["list", "create", "setup-code", "deactivate", "reactivate", "delete"])
@pytest.mark.parametrize("target", ["unsloth", "alice", "bob"])
def test_owner_alice_bob_endpoint_matrix(matrix, actor, operation, target):
    client, _, _ = matrix
    account = storage.get_account(target)
    url = f"/api/accounts/{account.account_id}"
    before = db_rows("auth_user")
    if operation == "list":
        response = client.get("/api/accounts", headers = headers(actor))
    elif operation == "create":
        response = client.post("/api/accounts", headers = headers(actor), json = {"username": "carol"})
    elif operation == "setup-code":
        response = client.post(url + "/setup-code", headers = headers(actor))
    elif operation in ("deactivate", "reactivate"):
        response = client.patch(url, headers = headers(actor), json = {"is_active": operation == "reactivate"})
    else:
        response = client.delete(url, headers = headers(actor))
    if actor != "unsloth":
        assert response.status_code == 403
        assert response.json() == {"detail": "Only the installation owner can do this"}
        assert db_rows("auth_user") == before
    elif target == "unsloth" and operation not in ("list", "create"):
        assert response.status_code == 400
        assert db_rows("auth_user") == before
    else:
        assert response.status_code == {"create": 201, "delete": 204}.get(operation, 200)
    if response.status_code == 200 and operation == "list":
        assert {r["username"] for r in response.json()["accounts"]} == {"unsloth", "alice", "bob"}
        assert all(set(row) == {"account_id", "username", "role", "is_active", "created_at", "setup_code_pending"} for row in response.json()["accounts"])


@pytest.mark.parametrize("method,suffix,payload", [
    ("GET", "", None), ("POST", "", {"username": "carol"}),
    ("POST", "/missing/setup-code", None),
    ("PATCH", "/missing", {"is_active": False}), ("DELETE", "/missing", None),
])
def test_accounts_requires_authentication(auth_env, method, suffix, payload):
    response = auth_env[0].request(method, "/api/accounts" + suffix, json = payload)
    assert response.status_code in (401, 403)
    assert "unsloth" not in response.text and "alice" not in response.text


@pytest.mark.parametrize("username,valid,canonical", [
    ("abc", True, "abc"), ("A_L-I9", True, "a_l-i9"), ("a" * 32, True, "a" * 32),
    ("ab", False, None), ("a" * 33, False, None), ("", False, None),
    ("alice ", False, None), (" ali", False, None), ("a.b", False, None),
    ("a/b", False, None), ("a\\b", False, None), ("álice", False, None),
    ("a\nb", False, None), ("UNSLOTH", False, None), ("Owner", False, None),
    ("ADMIN", False, None), ("root", False, None), ("system", False, None),
])
def test_username_validation(auth_env, username, valid, canonical):
    response = create(auth_env[0], username)
    assert response.status_code == (201 if valid else 422)
    if valid:
        assert response.json()["account"]["username"] == canonical
    else:
        assert storage.count_active_accounts() == 1


def test_create_code_hash_expiry_duplicate_and_public_fields(auth_env):
    client, _, _ = auth_env
    assert policy.login_mode() == "single"
    before = datetime.now(timezone.utc)
    response = create(client, "ALICE")
    assert response.status_code == 201
    body = response.json()
    after = datetime.now(timezone.utc)
    expiry = datetime.fromisoformat(body["setup_code_expires_at"])
    assert before + timedelta(minutes = 60) <= expiry <= after + timedelta(minutes = 60)
    assert policy.login_mode() == "multi"
    record = next(row for row in db_rows("auth_user") if row["username"] == "alice")
    assert record["setup_code_hash"] == storage._hash_token(body["setup_code"])
    assert body["setup_code"] not in str(record)
    assert record["must_change_password"] == 1
    listing = client.get("/api/accounts", headers = headers())
    assert body["setup_code"] not in listing.text
    assert "hash" not in listing.text
    assert body["account"]["setup_code_pending"] is True
    assert create(client, "Alice").status_code == 409


@pytest.mark.parametrize("username", ["alice", "bob"])
def test_setup_single_use_then_change_password_without_owner_side_effects(auth_env, username):
    client, _, _ = auth_env
    desktop = storage.create_desktop_secret()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"owner-bootstrap\n")
    owner_before = storage.get_user_record("unsloth")
    body = create(client, username).json()
    response = login(client, username.upper(), body["setup_code"])
    assert response.status_code == 200
    session = response.json()
    assert session["must_change_password"] is True
    assert login(client, username, body["setup_code"]).status_code == 401
    auth_header = {"Authorization": "Bearer " + session["access_token"]}
    assert client.get("/api/auth/api-keys", headers = auth_header).status_code == 403
    response = client.post("/api/auth/change-password", headers = auth_header, json = {
        "current_password": body["setup_code"], "new_password": "permanent-password",
    })
    assert response.status_code == 200
    assert response.json()["must_change_password"] is False
    assert login(client, username, "permanent-password").status_code == 200
    assert client.post("/api/auth/refresh", json = {"refresh_token": session["refresh_token"]}).status_code == 401
    assert storage.get_user_record("unsloth") == owner_before
    assert storage.validate_desktop_secret(desktop) == "unsloth"
    assert storage._BOOTSTRAP_PW_PATH.read_bytes() == b"owner-bootstrap\n"
    assert client.app.state.bootstrap_password == "owner-bootstrap"


def test_expired_setup_code_is_the_same_generic_401_as_wrong_password(auth_env):
    client, _, _ = auth_env
    body = create(client, "alice").json()
    with storage.get_connection() as conn:
        conn.execute("UPDATE auth_user SET setup_code_expires_at = ? WHERE account_id = ?", (
            (datetime.now(timezone.utc) - timedelta(seconds = 1)).isoformat(), body["account"]["account_id"],
        ))
    responses = [login(client, "alice", body["setup_code"]), login(client, "alice", "wrong"), login(client, "nobody", "wrong")]
    assert all(r.status_code == 401 for r in responses)
    assert len({r.content for r in responses}) == 1


def test_setup_consumption_is_atomic(auth_env):
    client, _, _ = auth_env
    body = create(client, "alice").json()
    barrier = threading.Barrier(8)

    def consume(_):
        barrier.wait()
        return storage.authenticate_account_login("alice", body["setup_code"])

    with ThreadPoolExecutor(max_workers = 8) as pool:
        results = list(pool.map(consume, range(8)))
    assert sum(result is not None for result in results) == 1


def test_setup_login_uses_existing_casefolded_rate_limit_buckets(auth_env):
    client, auth, _ = auth_env
    alice = create(client, "alice").json()
    bob = create(client, "bob").json()
    for i in range(auth._LOGIN_MAX_FAILS):
        assert login(client, "ALICE" if i % 2 else "alice", "wrong").status_code == 401
    blocked = login(client, "alice", alice["setup_code"])
    assert blocked.status_code == 429 and "Retry-After" in blocked.headers
    assert login(client, "bob", bob["setup_code"]).status_code == 200
    assert login(client, "unsloth", "owner-password").status_code == 200


def test_multi_login_requires_username(matrix):
    client, _, _ = matrix
    assert client.post("/api/auth/login", json = {"password": "owner-password"}).status_code == 422
    assert login(client, "", "owner-password").status_code == 401
    assert login(client, "UNSLOTH", "owner-password").status_code == 200


@pytest.mark.parametrize("must_change", [False, True])
def test_single_owner_login_response_bytes_and_lookup_cost_are_unchanged(auth_env, monkeypatch, must_change):
    client, auth, _ = auth_env
    with storage.get_connection() as conn:
        conn.execute("UPDATE auth_user SET must_change_password = ?", (int(must_change),))
    assert policy.login_mode() == "single"
    monkeypatch.setattr(policy, "active_account_count", lambda: 1)
    monkeypatch.setattr(auth, "create_access_token", lambda **kwargs: "access")
    monkeypatch.setattr(auth, "create_refresh_token", lambda **kwargs: "refresh")
    calls = []
    lookup = storage.get_user_and_secret

    def record_lookup(username):
        calls.append(username)
        return lookup(username)

    monkeypatch.setattr(storage, "get_user_and_secret", record_lookup)
    monkeypatch.setattr(storage, "authenticate_account_login", lambda *_: pytest.fail("owner entered managed login"))
    response = login(client, "unsloth", "owner-password")
    assert response.status_code == 200
    assert response.content == b'{"access_token":"access","refresh_token":"refresh","token_type":"bearer","must_change_password":' + (b"true}" if must_change else b"false}")
    assert calls == ["unsloth"]
    assert login(client, "UNSLOTH", "owner-password").status_code == 401


def seed_credentials():
    return {
        name: (authentication.create_access_token(name), authentication.create_refresh_token(name), storage.create_api_key(name, "test")[0])
        for name in ("unsloth", "alice", "bob")
    }


@pytest.mark.parametrize("operation", ["setup-code", "deactivate", "delete"])
@pytest.mark.parametrize("target", ["alice", "bob"])
def test_credential_revocation_is_scoped(matrix, operation, target):
    client, _, _ = matrix
    credentials = seed_credentials()
    account = storage.get_account(target)
    url = f"/api/accounts/{account.account_id}"
    if operation == "setup-code":
        response = client.post(url + "/setup-code", headers = headers())
    elif operation == "deactivate":
        response = client.patch(url, headers = headers(), json = {"is_active": False})
    else:
        response = client.delete(url, headers = headers())
    assert response.status_code in (200, 204)
    for name, (access, refresh, key) in credentials.items():
        response = client.get("/api/auth/api-keys", headers = {"Authorization": "Bearer " + access})
        assert response.status_code == (401 if name == target else 200)
        assert (storage.verify_refresh_token(refresh) is None) == (name == target)
        assert (storage.validate_api_key(key) is None) == (name == target)


def test_regeneration_invalidates_old_setup_and_its_session(auth_env):
    client, _, _ = auth_env
    first = create(client, "alice").json()
    session = login(client, "alice", first["setup_code"]).json()
    second = client.post(f'/api/accounts/{first["account"]["account_id"]}/setup-code', headers = headers()).json()
    assert first["setup_code"] != second["setup_code"]
    assert login(client, "alice", first["setup_code"]).status_code == 401
    assert client.post("/api/auth/change-password", headers = {"Authorization": "Bearer " + session["access_token"]}, json = {
        "current_password": first["setup_code"], "new_password": "new-password",
    }).status_code == 401
    assert login(client, "alice", second["setup_code"]).status_code == 200


def test_activity_updates_policy_and_deactivated_login_stays_refused_in_single_mode(auth_env):
    client, _, _ = auth_env
    body = create(client, "alice").json()
    url = f'/api/accounts/{body["account"]["account_id"]}'
    assert policy.login_mode() == "multi"
    assert client.patch(url, headers = headers(), json = {"is_active": False}).status_code == 200
    assert policy.login_mode() == "single"
    assert login(client, "alice", body["setup_code"]).status_code == 401
    assert client.patch(url, headers = headers(), json = {"is_active": True}).status_code == 200
    assert policy.login_mode() == "multi"
    assert login(client, "alice", body["setup_code"]).status_code == 200


def test_delete_retires_all_roots_and_recreated_username_inherits_nothing(matrix):
    client, _, _ = matrix
    alice = storage.get_account("alice")
    bob = storage.get_account("bob")
    roots = (storage_roots.workspace_root, storage_roots.project_workspaces_root, storage_roots.tmp_root)
    old_paths = []
    untouched = []
    for account in (OWNER, alice, bob):
        for root in roots:
            path = run_as(account, root)
            path.mkdir(parents = True, exist_ok = True)
            (path / "private.txt").write_text(account.username)
            (old_paths if account == alice else untouched).append(path)
    events = {account.username: threading.Event() for account in (OWNER, alice, bob)}
    with run_as(OWNER, active_generations.ActiveGeneration, events["unsloth"]), run_as(alice, active_generations.ActiveGeneration, events["alice"]), run_as(bob, active_generations.ActiveGeneration, events["bob"]):
        assert client.delete(f"/api/accounts/{alice.account_id}", headers = headers()).status_code == 204
        assert events["alice"].is_set()
        assert not events["bob"].is_set() and not events["unsloth"].is_set()
    for path in old_paths:
        assert not path.exists()
        retired = list(path.parent.glob(path.name + "-deleted-*"))
        assert len(retired) == 1
        assert (retired[0] / "private.txt").read_text() == "alice"
    for path in untouched:
        assert (path / "private.txt").exists()
    fresh = create(client, "alice").json()
    assert fresh["account"]["account_id"] != alice.account_id
    for root in roots:
        assert not run_as(storage.get_account("alice"), root).exists()
    assert client.delete(f"/api/accounts/{alice.account_id}", headers = headers()).status_code == 404


def test_failed_retirement_leaves_disabled_retryable_account(matrix, monkeypatch):
    client, _, accounts = matrix
    account = storage.get_account("alice")

    def fail(_):
        raise PermissionError("private host path must not be exposed")

    retire = accounts.retire_account_roots
    monkeypatch.setattr(accounts, "retire_account_roots", fail)
    response = client.delete(f"/api/accounts/{account.account_id}", headers = headers())
    assert response.status_code == 409
    assert "private host path" not in response.text
    assert storage.get_user_record("alice")["is_active"] == 0
    monkeypatch.setattr(accounts, "retire_account_roots", retire)
    assert client.delete(f"/api/accounts/{account.account_id}", headers = headers()).status_code == 204


def test_managed_password_change_cannot_overwrite_a_rotated_credential(matrix):
    record = storage.get_user_record("alice")
    storage.set_account_active(record["account_id"], False)
    storage.set_account_active(record["account_id"], True)
    assert storage.update_account_password("alice", "replacement-password", expect_password_hash = record["password_hash"], expect_secret = record["jwt_secret"]) is None
    current = storage.get_user_record("alice")
    assert hashing.verify_password("alice-password", current["password_salt"], current["password_hash"])


@pytest.fixture
def reset_cli(auth_env, monkeypatch):
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", storage.DB_PATH.parent.parent)
    monkeypatch.setattr(studio_cli, "_generate_reset_password", lambda: "reset-password")
    return CliRunner()


def test_cli_single_account_default_output_and_desktop_cleanup_unchanged(auth_env, reset_cli):
    client, _, _ = auth_env
    desktop = storage.create_desktop_secret()
    auth_dir = storage.DB_PATH.parent
    for filename in (studio_cli.BOOTSTRAP_PASSWORD_FILE, studio_cli.DESKTOP_SECRET_FILE):
        (auth_dir / filename).write_text("old-secret")
    token = authentication.create_refresh_token("unsloth")
    key = storage.create_api_key("unsloth", "test")[0]
    result = reset_cli.invoke(studio_cli.studio_app, ["reset-password"])
    assert result.exit_code == 0, result.output
    assert result.output == (
        "New password for 'unsloth': reset-password\n"
        "Sessions and API keys revoked. A running Unsloth takes it on the next request, "
        "though repeated failed logins can hold the rate limit shut for up to a minute.\n"
    )
    assert storage.validate_desktop_secret(desktop) is None
    assert storage.verify_refresh_token(token) is None
    assert storage.validate_api_key(key) is None
    assert login(client, "unsloth", "reset-password").status_code == 200
    for filename in (studio_cli.BOOTSTRAP_PASSWORD_FILE, studio_cli.DESKTOP_SECRET_FILE):
        assert not (auth_dir / filename).exists()


def test_cli_multi_requires_username_without_listing_or_changing_accounts(matrix, reset_cli):
    seed_credentials()
    before = {table: db_rows(table) for table in ("auth_user", "refresh_tokens", "api_keys")}
    result = reset_cli.invoke(studio_cli.studio_app, ["reset-password"])
    assert result.exit_code == 1
    assert "--username is required" in result.output
    assert all(name not in result.output for name in ("unsloth", "alice", "bob"))
    assert {table: db_rows(table) for table in before} == before


@pytest.mark.parametrize("target", ["unsloth", "alice", "bob"])
def test_cli_reset_only_target_and_owner_only_desktop_rotation(matrix, reset_cli, target):
    client, _, _ = matrix
    credentials = seed_credentials()
    desktop = storage.create_desktop_secret()
    auth_dir = storage.DB_PATH.parent
    for filename in (studio_cli.BOOTSTRAP_PASSWORD_FILE, studio_cli.DESKTOP_SECRET_FILE):
        (auth_dir / filename).write_text("owner-only")
    before = {name: storage.get_user_record(name) for name in credentials}
    result = reset_cli.invoke(studio_cli.studio_app, ["reset-password", "--username", target.upper()])
    assert result.exit_code == 0, result.output
    assert f"New password for '{target}'" in result.output
    for name, (_, refresh, key) in credentials.items():
        after = storage.get_user_record(name)
        if name == target:
            assert after["jwt_secret"] != before[name]["jwt_secret"]
            assert hashing.verify_password("reset-password", after["password_salt"], after["password_hash"])
        else:
            assert after == before[name]
        assert (storage.verify_refresh_token(refresh) is None) == (name == target)
        assert (storage.validate_api_key(key) is None) == (name == target)
    assert (storage.validate_desktop_secret(desktop) is None) == (target == "unsloth")
    for filename in (studio_cli.BOOTSTRAP_PASSWORD_FILE, studio_cli.DESKTOP_SECRET_FILE):
        assert (auth_dir / filename).exists() == (target != "unsloth")
    assert login(client, target, "reset-password").status_code == 200


def test_cli_reset_clears_pending_setup_and_never_reactivates(auth_env, reset_cli):
    client, _, _ = auth_env
    body = create(client, "alice").json()
    storage.set_account_active(body["account"]["account_id"], False)
    result = reset_cli.invoke(studio_cli.studio_app, ["reset-password", "--username", "alice"])
    assert result.exit_code == 0
    row = next(row for row in db_rows("auth_user") if row["username"] == "alice")
    assert row["setup_code_hash"] is None and row["setup_code_expires_at"] is None
    assert row["is_active"] == 0 and row["must_change_password"] == 0
    assert login(client, "alice", body["setup_code"]).status_code == 401
    assert login(client, "alice", "reset-password").status_code == 401
    storage.set_account_active(body["account"]["account_id"], True)
    assert login(client, "alice", "reset-password").status_code == 200


def test_cli_unknown_target_never_creates_account_or_changes_credentials(matrix, reset_cli):
    before = db_rows("auth_user")
    result = reset_cli.invoke(studio_cli.studio_app, ["reset-password", "--username", "missing"])
    assert result.exit_code == 1
    assert result.output == "Error: account not found.\n"
    assert db_rows("auth_user") == before


@pytest.mark.parametrize("actor", ["unsloth", "alice", "bob"])
def test_logout_revokes_only_actor_and_only_owner_clears_bootstrap_state(matrix, actor):
    client, _, _ = matrix
    tokens = {name: authentication.create_refresh_token(name) for name in ("unsloth", "alice", "bob")}
    response = client.post("/api/auth/logout", headers = headers(actor))
    assert response.status_code == 204
    for name, token in tokens.items():
        assert (storage.verify_refresh_token(token) is None) == (name == actor)
    assert client.app.state.bootstrap_password == (None if actor == "unsloth" else "owner-bootstrap")


@pytest.mark.parametrize("method,suffix,payload", [
    ("POST", "/setup-code", None), ("PATCH", "", {"is_active": False}), ("DELETE", "", None),
])
def test_unknown_account_id_returns_404(matrix, method, suffix, payload):
    response = matrix[0].request(method, "/api/accounts/missing" + suffix, headers = headers(), json = payload)
    assert response.status_code == 404
    assert response.json() == {"detail": "Account not found"}


def test_retirement_preserves_symlink_target_and_collision(auth_env, monkeypatch):
    _, _, accounts = auth_env
    storage.create_initial_user("alice", "alice-password", secrets.token_urlsafe(64))
    account = storage.get_account("alice")
    root = run_as(account, storage_roots.workspace_root)
    owner_file = storage_roots.workspace_root() / "owner-data"
    owner_file.mkdir()
    (owner_file / "private").write_text("owner")
    root.parent.mkdir(parents = True, exist_ok = True)
    root.symlink_to(owner_file, target_is_directory = True)

    class FrozenDatetime:
        @staticmethod
        def now(tz):
            return datetime(2026, 1, 1, tzinfo = tz)

    monkeypatch.setattr(accounts, "datetime", FrozenDatetime)
    existing = root.with_name(root.name + "-deleted-20260101T000000000000Z")
    existing.mkdir()
    (existing / "preserved").write_text("old")
    accounts.retire_account_roots(account)
    assert (owner_file / "private").read_text() == "owner"
    assert (existing / "preserved").read_text() == "old"
    assert existing.with_name(existing.name + "-1").is_symlink()
    assert not root.is_symlink()
    with pytest.raises(ValueError):
        accounts.retire_account_roots(OWNER)

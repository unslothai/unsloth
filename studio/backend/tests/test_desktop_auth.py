import importlib.util
import asyncio
import hashlib
import json
import os
import platform
import secrets
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from auth import storage


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    yield


def seed_user(*, must_change_password = False):
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = must_change_password,
    )


def auth_client():
    route_path = Path(__file__).resolve().parents[1] / "routes" / "auth.py"
    spec = importlib.util.spec_from_file_location("_desktop_auth_route", route_path)
    auth_route = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(auth_route)

    app = FastAPI()
    app.include_router(auth_route.router, prefix = "/api/auth")
    return TestClient(app)


def data_recipe_jobs_module():
    route_path = (
        Path(__file__).resolve().parents[1] / "routes" / "data_recipe" / "jobs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_desktop_data_recipe_jobs", route_path
    )
    jobs_route = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(jobs_route)
    return jobs_route


def local_recipe():
    return {
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [{"alias": "local-model", "provider": "local"}],
        "columns": [{"column_type": "llm-text", "model_alias": "local-model"}],
    }


def local_recipe_request(token):
    return SimpleNamespace(
        headers = {"authorization": f"Bearer {token}"},
        app = SimpleNamespace(state = SimpleNamespace(server_port = 8888)),
        scope = {},
        base_url = "http://testserver/",
    )


@pytest.fixture
def loaded_local_model(monkeypatch):
    inference_module = SimpleNamespace(
        get_llama_cpp_backend = lambda: SimpleNamespace(is_loaded = True),
    )
    monkeypatch.setitem(sys.modules, "routes.inference", inference_module)


def test_desktop_secret_round_trip_uses_real_admin_subject():
    seed_user()
    raw = storage.create_desktop_secret()

    assert raw.startswith("desktop-")
    assert storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME
    assert storage.validate_desktop_secret(raw + "x") is None


def test_create_desktop_secret_rotates_old_secret():
    seed_user()
    old = storage.create_desktop_secret()
    new = storage.create_desktop_secret()

    assert old != new
    assert storage.validate_desktop_secret(old) is None
    assert storage.validate_desktop_secret(new) == storage.DEFAULT_ADMIN_USERNAME


def test_clear_desktop_secret_invalidates_secret():
    seed_user()
    raw = storage.create_desktop_secret()

    storage.clear_desktop_secret()

    assert storage.validate_desktop_secret(raw) is None


def test_ensure_default_admin_does_not_recreate_bootstrap_for_existing_admin():
    seed_user()

    created = storage.ensure_default_admin()

    assert created is False
    assert not storage._BOOTSTRAP_PW_PATH.exists()


def test_ensure_default_admin_loads_existing_bootstrap_after_restart(monkeypatch):
    created = storage.ensure_default_admin()
    bootstrap_pw = storage._BOOTSTRAP_PW_PATH.read_text().strip()

    monkeypatch.setattr(storage, "_bootstrap_password", None)
    created_again = storage.ensure_default_admin()

    assert created is True
    assert storage._BOOTSTRAP_PW_PATH.exists()
    assert created_again is False
    assert storage.get_bootstrap_password() == bootstrap_pw


def test_ensure_default_admin_does_not_generate_for_empty_existing_bootstrap():
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_text(" \n")

    created = storage.ensure_default_admin()

    assert created is False
    assert storage._BOOTSTRAP_PW_PATH.read_text() == " \n"
    assert storage.get_bootstrap_password() is None


def test_web_login_token_has_no_desktop_marker_and_keeps_password_gate():
    seed_user(must_change_password = True)
    client = auth_client()

    response = client.post(
        "/api/auth/login",
        json = {
            "username": storage.DEFAULT_ADMIN_USERNAME,
            "password": "human-password-123",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["must_change_password"] is True
    payload = jwt.decode(
        body["access_token"],
        storage.get_jwt_secret(storage.DEFAULT_ADMIN_USERNAME),
        algorithms = ["HS256"],
    )
    assert payload["sub"] == storage.DEFAULT_ADMIN_USERNAME
    assert "desktop" not in payload

    gated = client.post(
        "/api/auth/api-keys",
        headers = {"Authorization": f"Bearer {body['access_token']}"},
        json = {"name": "web"},
    )
    assert gated.status_code == 403


def test_desktop_login_mints_admin_token_without_clearing_web_password_change():
    seed_user(must_change_password = True)
    raw = storage.create_desktop_secret()
    client = auth_client()

    response = client.post("/api/auth/desktop-login", json = {"secret": raw})

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["token_type"] == "bearer"
    assert body["must_change_password"] is False
    assert storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME) is True

    payload = jwt.decode(
        body["access_token"],
        storage.get_jwt_secret(storage.DEFAULT_ADMIN_USERNAME),
        algorithms = ["HS256"],
    )
    assert payload["sub"] == storage.DEFAULT_ADMIN_USERNAME
    assert payload["desktop"] is True


def test_desktop_refresh_preserves_desktop_marker():
    seed_user(must_change_password = True)
    raw = storage.create_desktop_secret()
    client = auth_client()
    login_body = client.post("/api/auth/desktop-login", json = {"secret": raw}).json()

    response = client.post(
        "/api/auth/refresh",
        json = {"refresh_token": login_body["refresh_token"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["must_change_password"] is False
    payload = jwt.decode(
        body["access_token"],
        storage.get_jwt_secret(storage.DEFAULT_ADMIN_USERNAME),
        algorithms = ["HS256"],
    )
    assert payload["sub"] == storage.DEFAULT_ADMIN_USERNAME
    assert payload["desktop"] is True


def test_consume_refresh_token_second_call_returns_none():
    """Single-use rotation rejects the same token on a second consume."""
    seed_user()
    from datetime import datetime, timedelta, timezone

    raw = secrets.token_urlsafe(48)
    expires = (datetime.now(timezone.utc) + timedelta(days = 30)).isoformat()
    storage.save_refresh_token(raw, storage.DEFAULT_ADMIN_USERNAME, expires)

    first = storage.consume_refresh_token(raw)
    assert first == (storage.DEFAULT_ADMIN_USERNAME, False)
    second = storage.consume_refresh_token(raw)
    assert second is None


def test_consume_refresh_token_concurrent_only_one_succeeds(tmp_path, monkeypatch):
    """64-thread pile-up against one token; DELETE RETURNING permits one winner."""
    seed_user()
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timedelta, timezone

    raw = secrets.token_urlsafe(48)
    expires = (datetime.now(timezone.utc) + timedelta(days = 30)).isoformat()
    storage.save_refresh_token(raw, storage.DEFAULT_ADMIN_USERNAME, expires)

    workers = 64

    def attempt(_idx: int):
        try:
            return storage.consume_refresh_token(raw)
        except sqlite3.OperationalError:
            # "database is locked" under heavy contention; treat as losing the race.
            return None

    with ThreadPoolExecutor(max_workers = workers) as pool:
        results = list(pool.map(attempt, range(workers)))

    successes = [r for r in results if r is not None]
    assert (
        len(successes) == 1
    ), f"expected exactly one consumer to win, got {len(successes)}"
    assert successes[0] == (storage.DEFAULT_ADMIN_USERNAME, False)


def test_consume_refresh_token_expired_returns_none():
    seed_user()
    from datetime import datetime, timedelta, timezone

    raw = secrets.token_urlsafe(48)
    expires = (datetime.now(timezone.utc) - timedelta(hours = 1)).isoformat()
    storage.save_refresh_token(raw, storage.DEFAULT_ADMIN_USERNAME, expires)
    assert storage.consume_refresh_token(raw) is None


def test_desktop_session_uses_real_admin_identity_for_api_keys():
    seed_user(must_change_password = True)
    raw = storage.create_desktop_secret()
    client = auth_client()
    token = client.post("/api/auth/desktop-login", json = {"secret": raw}).json()[
        "access_token"
    ]

    response = client.post(
        "/api/auth/api-keys",
        headers = {"Authorization": f"Bearer {token}"},
        json = {"name": "desktop"},
    )

    assert response.status_code == 200
    rows = storage.list_api_keys(storage.DEFAULT_ADMIN_USERNAME)
    assert [row["name"] for row in rows] == ["desktop"]


def test_local_recipe_token_authenticates_as_admin_for_desktop_user(loaded_local_model):
    # _inject_local_providers mints an internal sk-unsloth-* API key (not a
    # forwarded JWT). The unified API-key path validates as the real admin
    # user regardless of whether the incoming session was desktop or web.
    from auth.authentication import create_access_token, get_current_subject

    seed_user(must_change_password = True)
    jobs_route = data_recipe_jobs_module()
    incoming_token = create_access_token(
        subject = storage.DEFAULT_ADMIN_USERNAME,
        desktop = True,
    )
    recipe = local_recipe()

    jobs_route._inject_local_providers(recipe, local_recipe_request(incoming_token))

    local_token = recipe["model_providers"][0]["api_key"]
    assert local_token.startswith(storage.API_KEY_PREFIX)
    credentials = HTTPAuthorizationCredentials(
        scheme = "Bearer",
        credentials = local_token,
    )
    assert (
        asyncio.run(get_current_subject(credentials)) == storage.DEFAULT_ADMIN_USERNAME
    )


def test_local_recipe_token_authenticates_as_admin_for_web_user(loaded_local_model):
    # Mirror of the desktop variant: API-key issuance is identical for web
    # and desktop incoming tokens; auth via get_current_subject works the same.
    from auth.authentication import create_access_token, get_current_subject

    seed_user(must_change_password = False)
    jobs_route = data_recipe_jobs_module()
    incoming_token = create_access_token(subject = storage.DEFAULT_ADMIN_USERNAME)
    recipe = local_recipe()

    jobs_route._inject_local_providers(recipe, local_recipe_request(incoming_token))

    local_token = recipe["model_providers"][0]["api_key"]
    assert local_token.startswith(storage.API_KEY_PREFIX)
    credentials = HTTPAuthorizationCredentials(
        scheme = "Bearer",
        credentials = local_token,
    )
    assert (
        asyncio.run(get_current_subject(credentials)) == storage.DEFAULT_ADMIN_USERNAME
    )


def test_desktop_login_rejects_invalid_secret():
    seed_user(must_change_password = False)
    client = auth_client()

    response = client.post(
        "/api/auth/desktop-login",
        json = {"secret": "desktop-invalid"},
    )

    assert response.status_code == 401


def test_write_desktop_secret_file_is_0600_on_unix(tmp_path):
    from unsloth_cli.commands import studio as studio_cli

    path = tmp_path / ".desktop_secret"
    if platform.system() != "Windows":
        path.write_text("old-secret")
        os.chmod(path, 0o644)

    studio_cli._write_auth_secret(path, "desktop-secret")

    assert path.read_text() == "desktop-secret"
    if platform.system() != "Windows":
        assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_reset_password_removes_desktop_secret_files(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from unsloth_cli.commands import studio as studio_cli

    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / "auth.db").write_text("db")
    (auth_dir / ".bootstrap_password").write_text("boot")
    (auth_dir / ".desktop_secret").write_text("new")
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", tmp_path)

    result = CliRunner().invoke(studio_cli.studio_app, ["reset-password"])

    assert result.exit_code == 0
    assert not (auth_dir / "auth.db").exists()
    assert not (auth_dir / ".bootstrap_password").exists()
    assert not (auth_dir / ".desktop_secret").exists()


def test_reset_password_removes_desktop_secret_files_without_db(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from unsloth_cli.commands import studio as studio_cli

    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / ".desktop_secret").write_text("new")
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", tmp_path)

    result = CliRunner().invoke(studio_cli.studio_app, ["reset-password"])

    assert result.exit_code == 0
    assert not (auth_dir / ".desktop_secret").exists()


def test_desktop_capabilities_json_reports_rollout_safe_flags():
    from typer.testing import CliRunner
    import unsloth_cli.commands.studio as studio_cli

    result = CliRunner().invoke(
        studio_cli.studio_app,
        ["desktop-capabilities", "--json"],
    )

    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["desktop_protocol_version"] == 1
    assert body["supports_provision_desktop_auth"] is True
    assert body["supports_api_only"] is True
    assert isinstance(body["version"], str)


def test_health_response_reports_desktop_capability_fields(monkeypatch):
    router_stub = SimpleNamespace(
        auth_router = APIRouter(),
        data_recipe_router = APIRouter(),
        datasets_router = APIRouter(),
        export_router = APIRouter(),
        inference_router = APIRouter(),
        inference_studio_router = APIRouter(),
        models_router = APIRouter(),
        training_history_router = APIRouter(),
        training_router = APIRouter(),
    )
    monkeypatch.setitem(sys.modules, "routes", router_stub)

    import studio.backend.main as backend_main

    monkeypatch.setattr(backend_main._hw_module, "CHAT_ONLY", False)

    seed_user()
    from auth.authentication import create_access_token

    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)

    app = FastAPI()
    app.add_api_route("/api/health", backend_main.health_check, methods = ["GET"])
    client = TestClient(app)

    response = client.get(
        "/api/health",
        headers = {"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["desktop_protocol_version"] == 1
    assert body["supports_desktop_auth"] is True


def test_provision_desktop_auth_writes_secret_and_creates_db_without_backend_deps(
    tmp_path,
    monkeypatch,
):
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()

    code = """
import builtins
import sys
from pathlib import Path
from typer.testing import CliRunner

studio_home = Path(sys.argv[1])
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    blocked = ("auth", "fastapi", "structlog", "utils")
    if name in blocked or name.startswith(("auth.", "utils.")):
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from unsloth_cli.commands import studio as studio_cli

studio_cli.STUDIO_HOME = studio_home
result = CliRunner().invoke(studio_cli.studio_app, ["provision-desktop-auth"])
if result.exit_code != 0:
    print(result.output)
    if result.exception is not None:
        raise result.exception
    raise SystemExit(result.exit_code)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd = Path(__file__).resolve().parents[3],
        env = {**os.environ, "PYTHONPATH": "."},
        text = True,
        capture_output = True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    secret = (auth_dir / ".desktop_secret").read_text()
    assert secret.startswith("desktop-")

    conn = sqlite3.connect(auth_dir / "auth.db")
    conn.row_factory = sqlite3.Row
    try:
        user = conn.execute(
            """
            SELECT username, password_salt, password_hash, must_change_password
            FROM auth_user
            """
        ).fetchone()
        app_secrets = {
            row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM app_secrets")
        }
        refresh_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(refresh_tokens)")
        }
    finally:
        conn.close()

    bootstrap_password = (auth_dir / ".bootstrap_password").read_text().strip()
    bootstrap_hash = hashlib.pbkdf2_hmac(
        "sha256",
        bootstrap_password.encode("utf-8"),
        user["password_salt"].encode("utf-8"),
        100_000,
    ).hex()

    assert bootstrap_password
    assert user["username"] == "unsloth"
    assert user["must_change_password"] == 1
    assert bootstrap_hash == user["password_hash"]
    assert len(app_secrets["api_key_pbkdf2_salt"]) == 64
    assert len(app_secrets["desktop_secret_hash"]) == 64
    assert app_secrets["desktop_secret_created_at"]
    assert "is_desktop" in refresh_columns

    monkeypatch.setattr(storage, "DB_PATH", auth_dir / "auth.db")
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    assert storage.validate_desktop_secret(secret) == storage.DEFAULT_ADMIN_USERNAME
    assert storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME) is True


def test_provision_desktop_auth_keeps_existing_admin_password(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from unsloth_cli.commands import studio as studio_cli

    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", tmp_path)

    conn = sqlite3.connect(auth_dir / "auth.db")
    try:
        conn.execute(
            """
            CREATE TABLE auth_user (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                jwt_secret TEXT NOT NULL,
                must_change_password INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO auth_user (
                username, password_salt, password_hash, jwt_secret, must_change_password
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("unsloth", "existing-salt", "existing-hash", "existing-jwt", 0),
        )
        conn.commit()
    finally:
        conn.close()

    result = CliRunner().invoke(studio_cli.studio_app, ["provision-desktop-auth"])

    assert result.exit_code == 0
    assert not (auth_dir / ".bootstrap_password").exists()
    conn = sqlite3.connect(auth_dir / "auth.db")
    conn.row_factory = sqlite3.Row
    try:
        user = conn.execute(
            """
            SELECT password_salt, password_hash, jwt_secret, must_change_password
            FROM auth_user WHERE username = ?
            """,
            ("unsloth",),
        ).fetchone()
    finally:
        conn.close()

    assert dict(user) == {
        "password_salt": "existing-salt",
        "password_hash": "existing-hash",
        "jwt_secret": "existing-jwt",
        "must_change_password": 0,
    }


def test_update_password_clears_desktop_secret():
    seed_user()
    raw = storage.create_desktop_secret()
    assert storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME

    changed = storage.update_password(
        storage.DEFAULT_ADMIN_USERNAME, "new-admin-password"
    )
    assert changed is True
    assert storage.validate_desktop_secret(raw) is None


def test_update_password_on_unknown_user_leaves_desktop_secret_intact():
    seed_user()
    raw = storage.create_desktop_secret()

    changed = storage.update_password("not-a-user", "irrelevant")
    assert changed is False
    assert storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME


def test_desktop_auth_provision_has_bounded_timeout():
    rs_path = (
        Path(__file__).resolve().parents[3]
        / "studio"
        / "src-tauri"
        / "src"
        / "desktop_auth.rs"
    )
    src = rs_path.read_text()
    start = src.index("async fn provision_desktop_auth(")
    depth = 0
    body_start = src.index("{", start)
    body_end = None
    for i in range(body_start, len(src)):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                body_end = i + 1
                break
    assert body_end is not None
    body = src[start:body_end]
    assert "tokio::time::timeout" in body
    import re

    m = re.search(r"Duration::from_secs\(\s*(\d+)\s*\)", body)
    assert m is not None
    seconds = int(m.group(1))
    assert 5 <= seconds <= 120

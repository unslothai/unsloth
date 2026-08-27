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
from types import ModuleType, SimpleNamespace

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

    monkeypatch.setattr(storage, "_credential_encryption_key_cache", None)
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
    route_path = Path(__file__).resolve().parents[1] / "routes" / "data_recipe" / "jobs.py"
    spec = importlib.util.spec_from_file_location("_desktop_data_recipe_jobs", route_path)
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
    bootstrap_pw = storage._BOOTSTRAP_PW_PATH.read_text(encoding = "utf-8").strip()

    monkeypatch.setattr(storage, "_bootstrap_password", None)
    created_again = storage.ensure_default_admin()

    assert created is True
    assert storage._BOOTSTRAP_PW_PATH.exists()
    assert created_again is False
    assert storage.get_bootstrap_password() == bootstrap_pw


def test_bootstrap_password_file_ends_with_a_newline():
    # Otherwise `cat` welds the passphrase onto the shell prompt.
    storage.ensure_default_admin()

    # Bytes: read_text would decode CRLF back to "\n" and hide a CR.
    raw = storage._BOOTSTRAP_PW_PATH.read_bytes()

    assert raw == storage.get_bootstrap_password().encode("utf-8") + b"\n"


def test_bootstrap_password_round_trips_across_a_restart_with_the_newline():
    storage.ensure_default_admin()
    original = storage.get_bootstrap_password()

    storage._bootstrap_password = None

    assert storage.generate_bootstrap_password() == original


def test_upgrade_normalises_the_bootstrap_file():
    # Upgrade path: the admin row exists, so generate_bootstrap_password() never runs.
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"legacy-bootstrap-secret")

    storage.ensure_default_admin()

    assert storage._BOOTSTRAP_PW_PATH.read_bytes() == b"legacy-bootstrap-secret\n"
    assert storage.get_bootstrap_password() == "legacy-bootstrap-secret"


@pytest.mark.parametrize(
    "other",
    [
        b"legacy-bootstrap-secret\r\n",  # only an unreleased build wrote this
        b"legacy-bootstrap-secret\r",
        b"legacy-bootstrap-secret   ",
    ],
)
def test_only_an_exactly_unterminated_bootstrap_file_is_touched(other):
    # Appending is safe only because it is restricted to the one released shape.
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(other)

    storage.ensure_default_admin()

    assert storage.get_bootstrap_password() == "legacy-bootstrap-secret"
    assert storage._BOOTSTRAP_PW_PATH.read_bytes() == other


def test_upgrade_normalises_when_the_admin_row_is_missing():
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"legacy-bootstrap-secret")

    assert storage.generate_bootstrap_password() == "legacy-bootstrap-secret"
    assert storage._BOOTSTRAP_PW_PATH.read_bytes() == b"legacy-bootstrap-secret\n"


def test_a_well_formed_bootstrap_file_is_not_rewritten():
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"legacy-bootstrap-secret\n")
    mtime = storage._BOOTSTRAP_PW_PATH.stat().st_mtime_ns

    storage.ensure_default_admin()

    assert storage._BOOTSTRAP_PW_PATH.stat().st_mtime_ns == mtime


def test_migration_failure_does_not_break_startup(monkeypatch):
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"legacy-bootstrap-secret")

    real_open = storage.os.open

    def refuse(path, flags, *args, **kwargs):
        if str(path) == str(storage._BOOTSTRAP_PW_PATH):
            raise PermissionError("read-only auth dir")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", refuse)

    storage.ensure_default_admin()

    assert storage.get_bootstrap_password() == "legacy-bootstrap-secret"
    assert storage._BOOTSTRAP_PW_PATH.read_bytes() == b"legacy-bootstrap-secret"


def test_normalising_never_recreates_a_cleared_bootstrap_file(monkeypatch):
    # A rename would resurrect revoked plaintext if the password changed after the read.
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"legacy-bootstrap-secret")

    real_open = storage.os.open

    def clear_then_open(path, flags, *args, **kwargs):
        if str(path) == str(storage._BOOTSTRAP_PW_PATH):
            storage._BOOTSTRAP_PW_PATH.unlink(missing_ok = True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", clear_then_open)

    assert storage._read_persisted_bootstrap_password() == "legacy-bootstrap-secret"
    assert not storage._BOOTSTRAP_PW_PATH.exists()


def test_normalising_does_not_overwrite_a_rotated_bootstrap_file(monkeypatch):
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"legacy-bootstrap-secret")

    real_open = storage.os.open

    def rotate_then_open(path, flags, *args, **kwargs):
        if str(path) == str(storage._BOOTSTRAP_PW_PATH):
            storage._BOOTSTRAP_PW_PATH.write_bytes(b"brand-new-secret\n")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", rotate_then_open)

    storage._read_persisted_bootstrap_password()

    # The append may add a second newline; the rotated credential must survive.
    raw = storage._BOOTSTRAP_PW_PATH.read_bytes()
    assert raw.strip() == b"brand-new-secret"
    storage._bootstrap_password = None
    assert storage._load_bootstrap_password() == "brand-new-secret"


def test_leading_whitespace_bootstrap_file_is_left_alone(monkeypatch):
    # An in-place rewrite is not atomic, so only the exact unterminated shape is touched.
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"  legacy-bootstrap-secret  ")

    storage.ensure_default_admin()

    assert storage.get_bootstrap_password() == "legacy-bootstrap-secret"
    assert storage._BOOTSTRAP_PW_PATH.read_bytes() == b"  legacy-bootstrap-secret  "


def test_normalising_opens_the_file_in_binary_mode(monkeypatch):
    # Without O_BINARY, Windows text mode turns the written LF back into CRLF.
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"legacy-bootstrap-secret")
    monkeypatch.setattr(storage.os, "O_BINARY", 0x8000, raising = False)
    seen = []
    real_open = storage.os.open

    def spy(path, flags, *args, **kwargs):
        if str(path) == str(storage._BOOTSTRAP_PW_PATH):
            seen.append(flags)
        return real_open(path, flags & ~0x8000, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", spy)

    storage.ensure_default_admin()

    assert seen and all(f & 0x8000 for f in seen), seen


def test_clearing_by_truncation_mid_normalisation_is_not_undone(monkeypatch):
    # clear_bootstrap_password() truncates through its own descriptor when the unlink
    # fails (Windows, while ours is open); the append must not restore the plaintext.
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"legacy-bootstrap-secret")

    real_open = storage.os.open

    def truncate_then_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if str(path) == str(storage._BOOTSTRAP_PW_PATH):
            storage._BOOTSTRAP_PW_PATH.write_text("", encoding = "utf-8")
        return fd

    monkeypatch.setattr(storage.os, "open", truncate_then_open)

    storage._read_persisted_bootstrap_password()

    # A lone newline over a cleared file still reads back as no password.
    assert storage._BOOTSTRAP_PW_PATH.read_bytes().strip() == b""
    storage._bootstrap_password = None
    assert storage._load_bootstrap_password() is None


def test_normalising_works_without_fchmod(monkeypatch):
    # os.fchmod only reached Windows in 3.13; its absence must not raise.
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_bytes(b"legacy-bootstrap-secret")
    monkeypatch.delattr(storage.os, "fchmod", raising = False)

    storage.ensure_default_admin()

    assert storage._BOOTSTRAP_PW_PATH.read_bytes() == b"legacy-bootstrap-secret\n"
    assert storage.get_bootstrap_password() == "legacy-bootstrap-secret"


def test_persisting_the_bootstrap_password_is_atomic(monkeypatch, tmp_path):
    # A partial write would destroy the only plaintext recovery credential.
    storage._persist_bootstrap_password("original-secret")

    def boom(src, dst):
        raise OSError("crash before replace")

    monkeypatch.setattr(storage.os, "replace", boom)
    with pytest.raises(OSError):
        storage._persist_bootstrap_password("new-secret")

    assert storage._BOOTSTRAP_PW_PATH.read_bytes() == b"original-secret\n"
    leftovers = [
        p.name
        for p in storage._BOOTSTRAP_PW_PATH.parent.iterdir()
        if "bootstrap_password." in p.name
    ]
    assert leftovers == []


def test_ensure_default_admin_does_not_generate_for_empty_existing_bootstrap():
    seed_user()
    storage._BOOTSTRAP_PW_PATH.write_text(" \n", encoding = "utf-8")

    created = storage.ensure_default_admin()

    assert created is False
    assert storage._BOOTSTRAP_PW_PATH.read_text(encoding = "utf-8") == " \n"
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
    assert first[:2] == (storage.DEFAULT_ADMIN_USERNAME, False)
    second = storage.consume_refresh_token(raw)
    assert second is None


def test_consume_refresh_token_concurrent_only_one_succeeds(tmp_path, monkeypatch):
    """64-thread pile-up on one token; DELETE RETURNING permits one winner."""
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
            # "database is locked" under contention; treat as losing the race.
            return None

    with ThreadPoolExecutor(max_workers = workers) as pool:
        results = list(pool.map(attempt, range(workers)))

    successes = [r for r in results if r is not None]
    assert len(successes) == 1, f"expected exactly one consumer to win, got {len(successes)}"
    assert successes[0][:2] == (storage.DEFAULT_ADMIN_USERNAME, False)


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
    token = client.post("/api/auth/desktop-login", json = {"secret": raw}).json()["access_token"]

    response = client.post(
        "/api/auth/api-keys",
        headers = {"Authorization": f"Bearer {token}"},
        json = {"name": "desktop"},
    )

    assert response.status_code == 200
    rows = storage.list_api_keys(storage.DEFAULT_ADMIN_USERNAME)
    assert [row["name"] for row in rows] == ["desktop"]


def web_bearer(client) -> str:
    payload = {"username": storage.DEFAULT_ADMIN_USERNAME, "password": "human-password-123"}
    return client.post("/api/auth/login", json = payload).json()["access_token"]


def is_desktop_token(token: str) -> bool:
    payload = jwt.decode(
        token, storage.get_jwt_secret(storage.DEFAULT_ADMIN_USERNAME), algorithms = ["HS256"]
    )
    return payload.get("desktop") is True


def test_desktop_sets_the_remote_password_without_the_seeded_one():
    seed_user(must_change_password = True)
    raw = storage.create_desktop_secret()
    client = auth_client()
    token = client.post("/api/auth/desktop-login", json = {"secret": raw}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    body = {"new_password": "remote-password-123"}

    response = client.post("/api/auth/desktop-initial-password", headers = headers, json = body)

    assert response.status_code == 200
    assert response.json()["must_change_password"] is False
    assert is_desktop_token(response.json()["access_token"])
    assert storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME) is False
    # Desktop auto-auth survives the change the desktop itself made.
    assert storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME
    remote_login = client.post(
        "/api/auth/login",
        json = {"username": storage.DEFAULT_ADMIN_USERNAME, "password": "remote-password-123"},
    )
    assert remote_login.status_code == 200
    # The seeded credential is gone, so change-password owns every later change.
    repeat = client.post(
        "/api/auth/desktop-initial-password",
        headers = {"Authorization": f"Bearer {response.json()['access_token']}"},
        json = body,
    )
    assert repeat.status_code == 409


def test_remote_password_refuses_credentials_that_are_not_the_desktop_app():
    seed_user(must_change_password = True)
    client = auth_client()
    api_key, _row = storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, "cli")

    for bearer in (web_bearer(client), api_key):
        response = client.post(
            "/api/auth/desktop-initial-password",
            headers = {"Authorization": f"Bearer {bearer}"},
            json = {"new_password": "remote-password-123"},
        )
        # Distinguishable from the pre-existing "Password change required" refusal.
        assert response.status_code == 403
        assert response.json()["detail"] == "This action requires the Unsloth desktop app."
    assert storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME) is True


@pytest.mark.parametrize("desktop", [True, False])
def test_change_password_revokes_the_desktop_secret_only_for_browsers(desktop):
    seed_user(must_change_password = False)
    raw = storage.create_desktop_secret()
    client = auth_client()
    bearer = (
        client.post("/api/auth/desktop-login", json = {"secret": raw}).json()["access_token"]
        if desktop
        else web_bearer(client)
    )

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    credential_key = storage.get_or_create_credential_encryption_key()
    nonce = os.urandom(12)
    encrypted = AESGCM(credential_key).encrypt(nonce, b"hf-survives-password-change", b"test")

    response = client.post(
        "/api/auth/change-password",
        headers = {"Authorization": f"Bearer {bearer}"},
        json = {"current_password": "human-password-123", "new_password": "remote-password-123"},
    )

    assert response.status_code == 200
    assert is_desktop_token(response.json()["access_token"]) is desktop
    assert (storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME) is desktop

    storage._credential_encryption_key_cache = None
    reloaded_key = storage.get_or_create_credential_encryption_key()
    assert AESGCM(reloaded_key).decrypt(nonce, encrypted, b"test") == b"hf-survives-password-change"


def test_local_recipe_token_authenticates_as_admin_for_desktop_user(loaded_local_model):
    # _inject_local_providers mints an internal sk-unsloth-* API key (not a
    # forwarded JWT) that validates as admin whether the session was desktop or web.
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
    assert asyncio.run(get_current_subject(credentials)) == storage.DEFAULT_ADMIN_USERNAME


def test_local_recipe_token_authenticates_as_admin_for_web_user(loaded_local_model):
    # Mirror of the desktop variant: API-key issuance is identical for web/desktop tokens.
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
    assert asyncio.run(get_current_subject(credentials)) == storage.DEFAULT_ADMIN_USERNAME


def test_rotated_credential_job_start_is_401_not_500(loaded_local_model):
    # A reset-password landing mid-request makes the workflow-key mint refuse.
    # That must reach the client as a revoked credential, not an unhandled error.
    from fastapi import HTTPException

    seed_user()
    jobs_route = data_recipe_jobs_module()
    stale_gen = storage.credential_generation(secrets.token_urlsafe(64))

    with pytest.raises(storage.CredentialRotated):
        jobs_route._inject_local_providers(local_recipe(), local_recipe_request("t"), stale_gen)

    def _boom(*_a, **_k):
        raise storage.CredentialRotated("revoked")

    jobs_route._inject_local_providers = _boom
    payload = SimpleNamespace(recipe = local_recipe(), run = {})
    with pytest.raises(HTTPException) as excinfo:
        jobs_route.create_job(payload, local_recipe_request("t"), ("unsloth", stale_gen))
    assert excinfo.value.status_code == 401


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

    assert path.read_bytes() == b"desktop-secret\n"
    if platform.system() != "Windows":
        assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_reset_password_removes_desktop_secret_files(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from unsloth_cli.commands import studio as studio_cli

    from auth import storage as auth_storage
    from storage import credential_secrets

    auth_dir = tmp_path / "auth"
    studio_db = tmp_path / "studio.db"
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(auth_storage, "DB_PATH", auth_dir / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: studio_db)
    monkeypatch.setattr(credential_secrets, "ensure_dir", lambda _path: None)
    monkeypatch.setattr(
        credential_secrets,
        "get_or_create_credential_encryption_key",
        auth_storage.get_or_create_credential_encryption_key,
    )
    credential_secrets._schema_ready = False

    secret = studio_cli._create_desktop_secret_in_cli()
    studio_cli._write_auth_secret(auth_dir / studio_cli.DESKTOP_SECRET_FILE, secret)
    credential_secrets.save_hf_token("hf_survives_reset")
    (auth_dir / studio_cli.BOOTSTRAP_PASSWORD_FILE).write_text("boot")

    result = CliRunner().invoke(studio_cli.studio_app, ["reset-password"])

    assert result.exit_code == 0, result.output
    # The DB survives on purpose: a running server keeps serving from its admin row.
    assert (auth_dir / "auth.db").exists()
    assert not (auth_dir / studio_cli.BOOTSTRAP_PASSWORD_FILE).exists()
    assert not (auth_dir / studio_cli.DESKTOP_SECRET_FILE).exists()

    conn = studio_cli._connect_auth_db()
    try:
        surviving = conn.execute(
            "SELECT COUNT(*) FROM app_secrets WHERE key IN (?, ?)",
            (
                studio_cli.DESKTOP_SECRET_HASH_KEY,
                studio_cli.DESKTOP_SECRET_CREATED_AT_KEY,
            ),
        ).fetchone()[0]

        credential_key = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?",
            ("credential_encryption_key_v1",),
        ).fetchone()
    finally:
        conn.close()
    assert surviving == 0

    assert credential_key is not None
    auth_storage._credential_encryption_key_cache = None
    assert credential_secrets.get_hf_token() == "hf_survives_reset"
    credential_secrets._schema_ready = False
    auth_storage._credential_encryption_key_cache = None


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


def _routers_main_imports() -> set[str]:
    """Every ``*_router`` name in main.py's ``from routes import (...)`` block, read textually:
    importing ``routes`` would pull in torch, transformers and llama.cpp, which is what the stub
    below exists to avoid."""
    import re
    from pathlib import Path

    backend = Path(__file__).resolve().parent.parent
    main_src = (backend / "main.py").read_text(encoding = "utf-8")
    block = re.search(r"from routes import \(([^)]*)\)", main_src, re.S)
    assert block is not None, "main.py no longer has a parenthesised routes import; re-derive this"
    names = set(re.findall(r"(\w+_router)", block.group(1)))
    assert names, "the routes import block named no routers; re-derive this"
    return names


def test_health_response_reports_desktop_capability_fields(monkeypatch):
    routes_module = ModuleType("routes")
    routes_module.__path__ = []
    settings_module = ModuleType("routes.settings")
    settings_module.router = APIRouter()
    llama_module = ModuleType("routes.llama")
    llama_module.router = APIRouter()
    prompts_module = ModuleType("routes.prompts")
    prompts_module.router = APIRouter()
    preview_module = ModuleType("routes.preview")
    preview_module.router = APIRouter()
    whisper_module = ModuleType("routes.whisper")
    whisper_module.router = APIRouter()
    profile_stats_module = ModuleType("routes.profile_stats")
    profile_stats_module.router = APIRouter()

    # Derived from main.py's import block, not hand-listed: the old hardcoded dict went stale
    # twice (#8511's openai_codex_auth_router, #8648's youtube_router), each time killing every
    # test in this file with an ImportError.
    for name in _routers_main_imports():
        setattr(
            routes_module,
            name,
            # The health payload reads the real settings router.
            settings_module.router if name == "settings_router" else APIRouter(),
        )
    routes_module.settings = settings_module
    routes_module.llama = llama_module

    monkeypatch.setitem(sys.modules, "routes", routes_module)
    monkeypatch.setitem(sys.modules, "routes.settings", settings_module)
    monkeypatch.setitem(sys.modules, "routes.llama", llama_module)
    monkeypatch.setitem(sys.modules, "routes.prompts", prompts_module)
    monkeypatch.setitem(sys.modules, "routes.preview", preview_module)
    monkeypatch.setitem(sys.modules, "routes.whisper", whisper_module)
    monkeypatch.setitem(sys.modules, "routes.profile_stats", profile_stats_module)

    import studio.backend.main as backend_main

    # DEVICE alongside the two it is set with, since /api/health waits on
    # ensure_hardware_detected(). CPU + "mlx_unavailable" is an MLX-less Apple Silicon.
    monkeypatch.setattr(backend_main._hw_module, "DEVICE", backend_main._hw_module.DeviceType.CPU)
    monkeypatch.setattr(backend_main._hw_module, "CHAT_ONLY", True)
    monkeypatch.setattr(backend_main._hw_module, "CHAT_ONLY_REASON", "mlx_unavailable")
    # _hardware_snapshot() returns None until detection settles, so chat_only never publishes.
    import threading

    _settled = threading.Event()
    _settled.set()
    monkeypatch.setattr(backend_main._hw_module, "DETECTION_COMPLETE", _settled)
    # On a Mac the MLX self-heal overturns "mlx_unavailable" and health_check() drops the snapshot.
    monkeypatch.setattr(backend_main._hw_module, "is_apple_silicon", lambda: False)

    seed_user()
    from auth.authentication import create_access_token

    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)

    app = FastAPI()
    app.add_api_route("/api/health", backend_main.health_check, methods = ["GET"])
    client = TestClient(app)

    unauthenticated = client.get("/api/health")
    assert unauthenticated.status_code == 200
    unauthenticated_body = unauthenticated.json()
    assert unauthenticated_body["chat_only"] is True
    assert "chat_only_reason" not in unauthenticated_body

    response = client.get(
        "/api/health",
        headers = {"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["desktop_protocol_version"] == 1
    assert body["supports_desktop_auth"] is True
    assert body["chat_only_reason"] == "mlx_unavailable"


def test_provision_desktop_auth_writes_secret_and_creates_db_without_backend_deps(
    tmp_path, monkeypatch
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

def guarded_import(name, globals = None, locals = None, fromlist = (), level = 0):
    # Only gate absolute imports; relative `from .utils import x` inside
    # third-party packages (e.g. typer._click.decorators) hits level > 0
    # with name="utils" and must pass through.
    blocked = ("auth", "fastapi", "structlog", "utils")
    if level == 0 and (name in blocked or name.startswith(("auth.", "utils."))):
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)

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
    # Strip like the src-tauri readers do.
    secret = (auth_dir / ".desktop_secret").read_text().strip()
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
            row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM app_secrets")
        }
        refresh_columns = {row["name"] for row in conn.execute("PRAGMA table_info(refresh_tokens)")}
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

    changed = storage.update_password(storage.DEFAULT_ADMIN_USERNAME, "new-admin-password")
    assert changed
    assert storage.validate_desktop_secret(raw) is None


def test_update_password_revokes_desktop_secret_in_the_same_transaction(monkeypatch):
    # The desktop secret authenticates as this user WITHOUT the password, so it has
    # to die in the SAME transaction as the rotation. It used to be revoked after
    # the commit, on a second connection: anything that failed in between (a locked
    # or busy database, or the bootstrap-file cleanup raising first, as simulated
    # here) left a pre-change desktop credential live against the new password.
    seed_user()
    raw = storage.create_desktop_secret()
    assert storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME

    def _boom():
        raise OSError("auth dir is read-only")

    monkeypatch.setattr(storage, "clear_bootstrap_password", _boom)

    with pytest.raises(OSError):
        storage.update_password(storage.DEFAULT_ADMIN_USERNAME, "new-admin-password")

    # The rotation committed, and the desktop secret went with it.
    assert storage.validate_desktop_secret(raw) is None
    salt, pwd_hash, _jwt, _must_change = storage.get_user_and_secret(storage.DEFAULT_ADMIN_USERNAME)
    from auth import hashing

    assert hashing.verify_password("new-admin-password", salt, pwd_hash) is True


def test_update_password_on_unknown_user_leaves_desktop_secret_intact():
    seed_user()
    raw = storage.create_desktop_secret()

    changed = storage.update_password("not-a-user", "irrelevant")
    assert not changed
    assert storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME


def test_desktop_auth_provision_has_bounded_timeout():
    rs_path = (
        Path(__file__).resolve().parents[3] / "studio" / "src-tauri" / "src" / "desktop_auth.rs"
    )
    src = rs_path.read_text(encoding = "utf-8")
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


def test_the_router_stub_covers_every_router_main_imports():
    """The stub is derived from main.py, so it cannot go stale as it did for #8511's
    ``openai_codex_auth_router`` and #8648's ``youtube_router``. What can still break is the
    derivation: a reshaped import block would silently yield a short list. So pin it against the
    real package's ``__all__``, read textually for the same no-import reason."""
    import re
    from pathlib import Path

    backend = Path(__file__).resolve().parent.parent
    imported = _routers_main_imports()

    init_src = (backend / "routes" / "__init__.py").read_text(encoding = "utf-8")
    all_block = re.search(r"__all__\s*=\s*\[([^\]]*)\]", init_src, re.S)
    assert all_block is not None, "routes/__init__.py no longer has a literal __all__"
    exported = set(re.findall(r'"(\w+_router)"', all_block.group(1)))
    assert exported, "routes/__init__.py's __all__ named no routers; re-derive this"

    unknown = sorted(imported - exported)
    assert not unknown, (
        f"main.py imports {unknown} from routes, but routes/__init__.py does not export them: "
        f"the app itself would fail to start"
    )

    # Same drift for submodule imports, which need a sys.modules entry and are still hand-listed.
    import inspect

    main_src = (backend / "main.py").read_text(encoding = "utf-8")
    own_src = inspect.getsource(test_health_response_reports_desktop_capability_fields)
    submodules = set(re.findall(r"^from routes\.(\w+) import", main_src, re.M))
    assert submodules, "main.py imports no routes submodule; re-derive this"
    registered = set(re.findall(r'setitem\(\s*sys\.modules,\s*"routes\.(\w+)"', own_src))
    assert registered, "the health test registered no routes submodule; re-derive this"
    unstubbed = sorted(submodules - registered)
    assert not unstubbed, (
        f"main.py imports from routes.{{{','.join(unstubbed)}}}, which this file never "
        f"registers in sys.modules, so the real package would be imported instead"
    )


def test_update_password_discards_the_rotation_if_desktop_revocation_raises(monkeypatch):
    """A failing desktop revoke must roll the password back, not commit half of it.

    Moving clear_desktop_secret INSIDE the transaction changed this case. It used
    to run post-commit on a second connection, so an OperationalError from a busy
    database left the password CHANGED and a pre-change desktop credential LIVE --
    a credential that authenticates as this user without the password, still valid
    against the new one. Now the raise unwinds the whole transaction: old password,
    old desktop secret, consistent either way. Fail closed beats half-applied.
    """
    seed_user()
    raw = storage.create_desktop_secret()
    assert storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME
    salt_before, hash_before, _jwt_before, _mc = storage.get_user_and_secret(
        storage.DEFAULT_ADMIN_USERNAME
    )

    def _boom(conn = None):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(storage, "clear_desktop_secret", _boom)

    with pytest.raises(sqlite3.OperationalError):
        storage.update_password(storage.DEFAULT_ADMIN_USERNAME, "should-not-land")

    salt_after, hash_after, _jwt_after, _mc2 = storage.get_user_and_secret(
        storage.DEFAULT_ADMIN_USERNAME
    )
    assert (salt_after, hash_after) == (
        salt_before,
        hash_before,
    ), "the password was committed even though the transaction could not finish"
    from auth import hashing

    assert hashing.verify_password("should-not-land", salt_after, hash_after) is False
    # The desktop secret is still the pre-change one, matching the un-rotated password.
    assert storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME


def test_update_password_still_applies_when_desktop_secret_is_preserved(monkeypatch):
    """preserve_desktop_secret must skip the revoke entirely, not merely tolerate it.

    The desktop app authenticates WITH that secret and then sets its first
    password; revoking it would break the auto-auth for a change the desktop
    itself made. If the in-transaction move ever stopped honouring the flag, the
    revoke would run and this would fail.
    """
    seed_user()
    raw = storage.create_desktop_secret()

    called = []
    real = storage.clear_desktop_secret
    monkeypatch.setattr(
        storage,
        "clear_desktop_secret",
        lambda conn = None: called.append(conn) or real(conn),
    )
    _salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(storage.DEFAULT_ADMIN_USERNAME)
    assert (
        storage.update_password(
            storage.DEFAULT_ADMIN_USERNAME,
            "desktop-chosen-password",
            expect_password_hash = pwd_hash,
            preserve_desktop_secret = True,
        )
        is not None
    )
    assert called == [], "clear_desktop_secret ran despite preserve_desktop_secret"
    assert storage.validate_desktop_secret(raw) == storage.DEFAULT_ADMIN_USERNAME

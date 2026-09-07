# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Upgrade, concurrency, clock and hostile-input contracts for the setup token.

The setup token replaced a credential that had been embedded in the served page
for many releases, so the interesting question is not whether the happy path
works (test_link_token.py and test_link_initial_password.py cover that) but
whether anything that used to work stops working: an auth.db written by an older
Studio, the other credential routes, a machine in a non-UTC timezone, a Windows
filesystem that refuses chmod, two browsers racing, and input chosen to break the
parser.
"""

from __future__ import annotations

import importlib.util
import os
import secrets
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth import authentication, hashing, storage  # noqa: E402

_SEED = "seeded-bootstrap-123"
_NEW = "a-real-password-456"


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    yield


def _seed_admin(*, must_change_password: bool = True) -> str:
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = _SEED,
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = must_change_password,
    )
    return storage.DEFAULT_ADMIN_USERNAME


def _load_auth_route():
    spec = importlib.util.spec_from_file_location(
        "unsloth_test_auth_route_compat", _BACKEND / "routes" / "auth.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(_load_auth_route().router, prefix = "/api/auth")
    return TestClient(app)


def _authenticates(username: str, candidate: str) -> bool:
    record = storage.get_user_and_secret(username)
    if record is None:
        return False
    salt, pwd_hash, _jwt, _must_change = record
    return hashing.verify_password(candidate, salt, pwd_hash)


# ── upgrading an install that predates link tokens ───────────────────
#
# There is no migration framework and no PRAGMA user_version in this repo; the
# schema is (re)declared by get_connection() on every open. These tests pin that,
# because it is the entire reason an old auth.db keeps working.


_PRE_LINK_TOKEN_SCHEMA = """
CREATE TABLE auth_user (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    jwt_secret TEXT NOT NULL
);
CREATE TABLE refresh_tokens (
    id INTEGER PRIMARY KEY,
    token_hash TEXT NOT NULL,
    username TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE api_keys (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    key_hash   TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    expires_at TEXT,
    is_active  INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE app_secrets (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _write_ancient_auth_db(path: Path) -> None:
    """An auth.db as an older Studio left it: no link_tokens, older columns."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_PRE_LINK_TOKEN_SCHEMA)
        salt, pwd_hash = hashing.hash_password(_SEED)
        conn.execute(
            "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret) "
            "VALUES (?, ?, ?, ?)",
            (storage.DEFAULT_ADMIN_USERNAME, salt, pwd_hash, secrets.token_urlsafe(64)),
        )
        conn.execute(
            "INSERT INTO refresh_tokens (token_hash, username, expires_at) VALUES (?, ?, ?)",
            ("deadbeef", storage.DEFAULT_ADMIN_USERNAME, "2099-01-01T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()


def test_an_old_auth_db_gains_the_table_on_first_open():
    _write_ancient_auth_db(storage.DB_PATH)
    probe = sqlite3.connect(storage.DB_PATH)
    tables = {r[0] for r in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    probe.close()
    assert "link_tokens" not in tables, "fixture is not actually an old database"

    conn = storage.get_connection()
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "link_tokens" in tables


def test_an_old_auth_db_gains_the_back_filled_columns_too():
    # The same open runs the ALTER ladder; a token flow reads must_change_password
    # and a desktop session reads is_desktop, so a half-migrated DB would fail
    # somewhere less obvious than here.
    _write_ancient_auth_db(storage.DB_PATH)
    conn = storage.get_connection()
    try:
        auth_cols = {r["name"] for r in conn.execute("PRAGMA table_info(auth_user)")}
        refresh_cols = {r["name"] for r in conn.execute("PRAGMA table_info(refresh_tokens)")}
        api_cols = {r["name"] for r in conn.execute("PRAGMA table_info(api_keys)")}
    finally:
        conn.close()
    assert "must_change_password" in auth_cols
    assert {"is_desktop", "secret_gen"} <= refresh_cols
    assert "is_internal" in api_cols


def test_first_login_completes_on_a_migrated_old_database():
    """The whole point: an upgraded install can still finish first boot."""
    _write_ancient_auth_db(storage.DB_PATH)
    admin = storage.DEFAULT_ADMIN_USERNAME
    # An old row has no must_change_password, so the column back-fills to 0 and
    # the install reads as already set up. Put it back into first-boot state the
    # way an operator forced to change would be.
    conn = storage.get_connection()
    try:
        conn.execute("UPDATE auth_user SET must_change_password = 1 WHERE username = ?", (admin,))
        conn.commit()
    finally:
        conn.close()
    assert storage.requires_password_change(admin) is True

    client = _client()
    token = authentication.create_link_token(admin)
    exchanged = client.post("/api/auth/link-exchange", json = {"link_token": token})
    assert exchanged.status_code == 200, exchanged.text
    resp = client.post(
        "/api/auth/link-initial-password",
        json = {"new_password": _NEW},
        headers = {"Authorization": f"Bearer {exchanged.json()['access_token']}"},
    )
    assert resp.status_code == 200, resp.text
    assert _authenticates(admin, _NEW)


def test_an_old_refresh_token_row_survives_the_upgrade():
    _write_ancient_auth_db(storage.DB_PATH)
    conn = storage.get_connection()
    try:
        rows = conn.execute("SELECT COUNT(*) FROM refresh_tokens").fetchone()[0]
    finally:
        conn.close()
    assert rows == 1, "the migration dropped an existing refresh token row"


# ── the other credential routes still work ───────────────────────────


def test_password_login_is_unaffected_by_the_link_surface():
    admin = _seed_admin(must_change_password = False)
    client = _client()
    resp = client.post("/api/auth/login", json = {"username": admin, "password": _SEED})
    assert resp.status_code == 200, resp.text
    assert authentication.is_link_access_token(resp.json()["access_token"]) is False


def test_a_link_session_cannot_reach_change_password():
    """The link claim opens exactly one route, not the ordinary rotation one."""
    admin = _seed_admin()
    client = _client()
    token = authentication.create_link_token(admin)
    access = client.post("/api/auth/link-exchange", json = {"link_token": token}).json()[
        "access_token"
    ]
    resp = client.post(
        "/api/auth/change-password",
        json = {"current_password": "not-the-seed", "new_password": _NEW},
        headers = {"Authorization": f"Bearer {access}"},
    )
    assert resp.status_code >= 400
    assert _authenticates(admin, _SEED), "the password moved without the current one"


def test_a_link_session_cannot_reach_the_desktop_route():
    admin = _seed_admin()
    client = _client()
    token = authentication.create_link_token(admin)
    access = client.post("/api/auth/link-exchange", json = {"link_token": token}).json()[
        "access_token"
    ]
    resp = client.post(
        "/api/auth/desktop-initial-password",
        json = {"new_password": _NEW},
        headers = {"Authorization": f"Bearer {access}"},
    )
    assert resp.status_code == 403
    assert _authenticates(admin, _SEED)


def test_refreshing_a_link_session_drops_the_privilege():
    admin = _seed_admin()
    client = _client()
    token = authentication.create_link_token(admin)
    body = client.post("/api/auth/link-exchange", json = {"link_token": token}).json()
    refreshed = client.post("/api/auth/refresh", json = {"refresh_token": body["refresh_token"]})
    assert refreshed.status_code == 200, refreshed.text
    assert authentication.is_link_access_token(refreshed.json()["access_token"]) is False


# ── two browsers, one first boot ─────────────────────────────────────


def test_parallel_exchanges_of_one_token_yield_exactly_one_session():
    admin = _seed_admin()
    token = authentication.create_link_token(admin)
    results: list[object] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        outcome = authentication.exchange_link_token(token)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target = worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for r in results if r == admin) == 1, results
    conn = storage.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM link_tokens").fetchone()[0] == 0
    finally:
        conn.close()


def test_two_setup_tabs_race_and_exactly_one_sets_the_password():
    """Both tabs got their own token, so both hold a valid session."""
    admin = _seed_admin()
    client = _client()
    sessions = []
    for _ in range(2):
        token = authentication.create_link_token(admin)
        resp = client.post("/api/auth/link-exchange", json = {"link_token": token})
        assert resp.status_code == 200, resp.text
        sessions.append(resp.json()["access_token"])

    statuses = [
        client.post(
            "/api/auth/link-initial-password",
            json = {"new_password": f"{_NEW}-{index}"},
            headers = {"Authorization": f"Bearer {access}"},
        ).status_code
        for index, access in enumerate(sessions)
    ]
    assert statuses.count(200) == 1, statuses
    assert storage.requires_password_change(admin) is False


# ── clocks and timezones ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "tz", ["UTC", "America/New_York", "Asia/Kolkata", "Pacific/Chatham", "Etc/GMT+12"]
)
def test_mint_and_exchange_survive_a_non_utc_local_timezone(tz, monkeypatch):
    """Expiries are stored as ISO-8601 and compared lexicographically.

    A naive local timestamp would sort against an aware UTC one and either expire
    a fresh token or keep a dead one alive, and would do it only for operators
    outside UTC. Pacific/Chatham is +12:45, so it also catches assumptions that
    offsets are whole hours.
    """
    monkeypatch.setenv("TZ", tz)
    if hasattr(os, "tzset"):
        os.tzset()
    try:
        admin = _seed_admin()
        token = authentication.create_link_token(admin)
        assert authentication.exchange_link_token(token) == admin
    finally:
        monkeypatch.delenv("TZ", raising = False)
        if hasattr(os, "tzset"):
            os.tzset()


def test_a_token_past_its_expiry_is_refused():
    import time

    admin = _seed_admin()
    token = authentication.create_link_token(admin, expires_in = 1)
    time.sleep(1.2)
    assert authentication.exchange_link_token(token) is None


def test_a_negative_ttl_cannot_mint_a_token_that_is_already_expired():
    """create_link_token clamps to a minimum of one second.

    Worth pinning: the alternative is minting something whose expiry is already
    in the past, which then depends on whichever of the two expiry checks (the
    signed exp, or the nonce row) runs first for its behaviour.
    """
    admin = _seed_admin()
    token = authentication.create_link_token(admin, expires_in = -1)
    assert authentication.exchange_link_token(token) == admin


def test_a_token_just_inside_its_expiry_is_accepted():
    admin = _seed_admin()
    token = authentication.create_link_token(admin, expires_in = 30)
    assert authentication.exchange_link_token(token) == admin


def test_a_naive_expiry_is_rejected_rather_than_raising(monkeypatch):
    """A signed but timezone-naive exp must not escape as a TypeError.

    Not reachable through the real issuer, which writes an aware timestamp, but
    the comparison is the kind of thing a future caller gets wrong and the
    failure mode should be "no" rather than a 500.
    """
    admin = _seed_admin()
    real = authentication.datetime

    class _NaiveNow(real):
        @classmethod
        def now(cls, tz = None):
            return real.now(timezone.utc).replace(tzinfo = None)

    token = authentication.create_link_token(admin)
    monkeypatch.setattr(authentication, "datetime", _NaiveNow)
    assert authentication.exchange_link_token(token) is None


def test_a_far_future_expiry_still_needs_a_live_nonce_row():
    """The signature is not the single-use control; the row is."""
    admin = _seed_admin()
    token = authentication.create_link_token(
        admin, expires_in = int(timedelta(days = 3650).total_seconds())
    )
    conn = storage.get_connection()
    try:
        conn.execute("DELETE FROM link_tokens")
        conn.commit()
    finally:
        conn.close()
    assert authentication.exchange_link_token(token) is None


# ── hostile input ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        ".",
        "..",
        "no-separator",
        "a.b.c",
        "eyJzdWIiOiJ1bnNsb3RoIn0",
        "\x00\x00",
        "sub=unsloth&sig=x",
        "\x00.\x00",
        "aGVsbG8=.aGVsbG8=",
        "%2e%2e",
        "x" * 4096,
    ],
)
def test_malformed_tokens_are_refused_without_raising(candidate):
    _seed_admin()
    assert authentication.exchange_link_token(candidate) is None


def test_a_tampered_signature_is_refused():
    admin = _seed_admin()
    payload, _sig = authentication.create_link_token(admin).split(".", 1)
    assert authentication.exchange_link_token(f"{payload}.{'A' * 43}") is None


def test_a_swapped_payload_is_refused():
    """Re-pointing a valid signature at a different subject must not verify."""
    _seed_admin()
    storage.create_initial_user(
        username = "someone-else",
        password = _SEED,
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = True,
    )
    _payload, sig = authentication.create_link_token("someone-else").split(".", 1)
    victim_payload, _ = authentication.create_link_token(storage.DEFAULT_ADMIN_USERNAME).split(
        ".", 1
    )
    assert authentication.exchange_link_token(f"{victim_payload}.{sig}") is None


def test_an_access_token_is_not_accepted_as_a_link_token():
    admin = _seed_admin()
    assert (
        authentication.exchange_link_token(authentication.create_access_token(subject = admin))
        is None
    )


def test_a_link_token_is_not_accepted_as_a_bearer_token():
    admin = _seed_admin()
    assert authentication.is_link_access_token(authentication.create_link_token(admin)) is False


# ── filesystems that are not POSIX ───────────────────────────────────


def test_the_flow_works_when_chmod_is_refused(monkeypatch):
    """Windows has no POSIX mode bits and os.chmod can fail outright.

    storage swallows those failures deliberately; this pins that the token flow
    still completes rather than merely that the chmod call is wrapped.
    """
    real_chmod = os.chmod

    def _refuse(path, mode, *args, **kwargs):
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chmod", _refuse)
    try:
        admin = _seed_admin()
        token = authentication.create_link_token(admin)
        assert authentication.exchange_link_token(token) == admin
    finally:
        monkeypatch.setattr(os, "chmod", real_chmod)


def test_the_bootstrap_file_is_written_with_a_single_lf(tmp_path, monkeypatch):
    """Text mode would write CRLF on Windows and the CR would join the secret."""
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    storage._persist_bootstrap_password("passphrase-here")
    assert (tmp_path / ".bootstrap_password").read_bytes() == b"passphrase-here\n"


# ── the auth body cap ────────────────────────────────────────────────


def test_every_auth_payload_fits_well_inside_the_cap():
    """The cap is new, so nothing legitimate may now be over it."""
    import main as studio_main

    admin = _seed_admin()
    biggest = max(
        len(authentication.create_link_token(admin)),
        len(authentication.create_access_token(subject = admin)),
        len(authentication.create_refresh_token(subject = admin)),
    )
    assert biggest * 4 < studio_main.AUTH_REQUEST_BODY_MAX_BYTES, (
        f"the largest auth credential is {biggest} bytes against a "
        f"{studio_main.AUTH_REQUEST_BODY_MAX_BYTES} byte cap"
    )


# ── no hardware coupling ─────────────────────────────────────────────


def test_the_auth_import_graph_pulls_in_no_hardware_module():
    """Auth must not drag torch or device detection in.

    main.py's lifespan deliberately seeds auth before starting hardware
    detection, with a comment that detection used to hold the login screen up.
    An import edge from auth into that machinery would quietly undo it.
    """
    before = set(sys.modules)
    for name in ("torch", "utils.hardware", "utils.torch_device_probe"):
        sys.modules.pop(name, None)

    importlib.util.find_spec("auth.authentication")
    import auth.authentication  # noqa: F401
    import auth.storage  # noqa: F401

    added = set(sys.modules) - before
    forbidden = {m for m in added if m.split(".")[0] in {"torch"} or "hardware" in m}
    assert not forbidden, f"auth now imports hardware machinery: {sorted(forbidden)}"

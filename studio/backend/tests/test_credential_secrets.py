# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from auth import storage as auth_storage
from storage import credential_secrets


@pytest.fixture(autouse = True)
def isolated_databases(tmp_path, monkeypatch):
    auth_db = tmp_path / "auth.db"
    studio_db = tmp_path / "studio.db"
    monkeypatch.setattr(auth_storage, "DB_PATH", auth_db)
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: studio_db)
    monkeypatch.setattr(
        credential_secrets, "ensure_dir", lambda path: path.mkdir(parents = True, exist_ok = True)
    )
    monkeypatch.setattr(
        credential_secrets,
        "get_or_create_credential_encryption_key",
        auth_storage.get_or_create_credential_encryption_key,
    )
    credential_secrets._schema_ready = False
    yield studio_db
    credential_secrets._schema_ready = False
    auth_storage._credential_encryption_key_cache = None


def test_round_trip_isolation_and_ciphertext(isolated_databases):
    secret = "hf_example-secret-value"
    credential_secrets.save_hf_token("alice", secret)

    assert credential_secrets.get_hf_token("alice") == secret
    assert credential_secrets.get_hf_token("bob") is None
    assert secret.encode() not in isolated_databases.read_bytes()

    credential_secrets.save_provider_api_key("alice", "provider-1", "sk-one")
    credential_secrets.save_provider_api_key("alice", "provider-2", "sk-two")
    assert credential_secrets.get_provider_api_key("alice", "provider-1") == "sk-one"
    assert credential_secrets.get_provider_api_key("alice", "provider-2") == "sk-two"
    assert credential_secrets.get_provider_api_key("bob", "provider-1") is None


def test_upsert_and_delete_are_idempotent():
    credential_secrets.save_provider_api_key("alice", "provider-1", "first")
    credential_secrets.save_provider_api_key("alice", "provider-1", "second")
    assert credential_secrets.get_provider_api_key("alice", "provider-1") == "second"
    assert credential_secrets.delete_provider_api_key("alice", "provider-1") is True
    assert credential_secrets.delete_provider_api_key("alice", "provider-1") is False


def test_tampering_and_key_loss_fail_closed(isolated_databases):
    credential_secrets.save_hf_token("alice", "hf_private")
    conn = sqlite3.connect(isolated_databases)
    try:
        row = conn.execute("SELECT ciphertext FROM credential_secrets").fetchone()
        damaged = bytearray(row[0])
        damaged[-1] ^= 1
        conn.execute("UPDATE credential_secrets SET ciphertext = ?", (bytes(damaged),))
        conn.commit()
    finally:
        conn.close()
    assert credential_secrets.get_hf_token("alice") is None

    credential_secrets.save_hf_token("alice", "hf_replaced-key")
    auth_storage._credential_encryption_key_cache = None
    conn = auth_storage.get_connection()
    try:
        conn.execute(
            "UPDATE app_secrets SET value = ? WHERE key = ?",
            ("00" * 32, auth_storage._CREDENTIAL_ENCRYPTION_KEY_DB_KEY),
        )
        conn.commit()
    finally:
        conn.close()
    assert credential_secrets.get_hf_token("alice") is None


def test_repeated_schema_initialization_and_concurrent_upserts():
    credential_secrets.get_connection().close()
    credential_secrets._schema_ready = False
    credential_secrets.get_connection().close()

    with ThreadPoolExecutor(max_workers = 4) as pool:
        list(
            pool.map(
                lambda value: credential_secrets.save_provider_api_key(
                    "alice", "provider-1", value
                ),
                ["one", "two", "three", "four"],
            )
        )
    assert credential_secrets.get_provider_api_key("alice", "provider-1") in {
        "one",
        "two",
        "three",
        "four",
    }


def test_credential_key_persists_independently_of_password_changes():
    before = auth_storage.get_or_create_credential_encryption_key()
    auth_storage._credential_encryption_key_cache = None
    after = auth_storage.get_or_create_credential_encryption_key()
    assert before == after


def test_credentials_survive_process_restart_simulation():
    credential_secrets.save_hf_token("alice", "hf_restart")
    credential_secrets.save_provider_api_key("alice", "provider-1", "sk_restart")

    auth_storage._credential_encryption_key_cache = None
    credential_secrets._schema_ready = False

    assert credential_secrets.get_hf_token("alice") == "hf_restart"
    assert credential_secrets.get_provider_api_key("alice", "provider-1") == "sk_restart"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import secrets
import sqlite3

import pytest

from auth import oidc_storage, policy, storage


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    storage._auth_schema_ready.clear()
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield
    storage._auth_schema_ready.clear()
    policy.invalidate_account_cache()


def test_external_identity_migration_is_idempotent(auth_db):
    storage.get_connection().close()
    storage._auth_schema_ready.clear()
    storage.get_connection().close()

    conn = sqlite3.connect(storage.DB_PATH)
    try:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "external_identities" in tables
    finally:
        conn.close()


def test_oidc_account_is_created_as_normal_non_owner_and_reused(auth_db):
    created = oidc_storage.create_external_account(
        issuer = "https://auth.example/realms/company",
        subject = "subject-1",
        preferred_username = "Bernardo",
        email = "bernardo@example.com",
        display_name = "Bernardo Example",
    )
    repeated = oidc_storage.create_external_account(
        issuer = "https://auth.example/realms/company",
        subject = "subject-1",
        preferred_username = "changed-name",
        email = "changed@example.com",
    )

    assert created["account_id"] == repeated["account_id"]
    assert created["username"] == "bernardo"
    assert created["role"] == "user"
    assert storage.requires_password_change(created["username"]) is False


def test_same_email_with_different_subject_never_links_accounts(auth_db):
    first = oidc_storage.create_external_account(
        issuer = "https://auth.example/realms/company",
        subject = "subject-1",
        preferred_username = "bernardo",
        email = "shared@example.com",
    )
    second = oidc_storage.create_external_account(
        issuer = "https://auth.example/realms/company",
        subject = "subject-2",
        preferred_username = "bernardo",
        email = "shared@example.com",
    )

    assert first["account_id"] != second["account_id"]
    assert first["username"] != second["username"]


def test_local_username_collision_never_links_oidc_identity(auth_db):
    local = storage.issue_account_setup_code(username = "alice")["account"]
    external = oidc_storage.create_external_account(
        issuer = "https://auth.example/realms/company",
        subject = "alice-at-idp",
        preferred_username = "alice",
        email = "alice@example.com",
    )

    assert local["account_id"] != external["account_id"]
    assert external["username"].startswith("alice-")


def test_deleting_account_removes_external_mapping(auth_db):
    account = oidc_storage.create_external_account(
        issuer = "https://auth.example/realms/company",
        subject = "subject-1",
        preferred_username = "alice",
    )

    storage.delete_account(account["account_id"], lambda _account: lambda: None)

    assert (
        oidc_storage.get_external_identity("https://auth.example/realms/company", "subject-1")
        is None
    )

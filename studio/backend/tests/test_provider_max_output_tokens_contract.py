# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Upgrade, downgrade and contract tests for the per-connection ``max_output_tokens``.

Three things a happy-path test cannot reach:

* an existing studio.db, written before this column existed, opened by this code;
* the same database opened AGAIN by a build that has never heard of the column,
  which is what a user who reverts to the previous release does;
* the route contract, where an explicit null has to be accepted on every provider type:
  the dialog sends null for a blank field rather than omitting it, so rejecting it broke
  every unrelated edit of a row that carries no override.

No network, no GPU, no server: the routes are driven as plain coroutines and every
database is a per-test temporary file.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

from auth import storage as auth_storage
from core.inference.providers import PROVIDER_REGISTRY
from models.providers import ProviderCreate, ProviderUpdate
from storage import credential_secrets, providers_db


# routes/providers.py imports its siblings as ``routes.*``. Loading it by path under
# a private name is the pattern test_credential_routes.py already uses.
def _load_route_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_routes_dir = Path(__file__).resolve().parents[1] / "routes"
_previous_routes = sys.modules.get("routes")
_routes_package = types.ModuleType("routes")
_routes_package.__path__ = [str(_routes_dir)]
sys.modules["routes"] = _routes_package
try:
    _load_route_module("routes.provider_credentials", _routes_dir / "provider_credentials.py")
    providers_route = _load_route_module(
        "_max_output_contract_providers_route", _routes_dir / "providers.py"
    )
finally:
    sys.modules.pop("routes.provider_credentials", None)
    if _previous_routes is None:
        sys.modules.pop("routes", None)
    else:
        sys.modules["routes"] = _previous_routes


CREDENTIAL = ("alice", None)

# From the registry, so a provider added later is covered without an edit here.
NON_CUSTOM_PROVIDER_TYPES = tuple(t for t in PROVIDER_REGISTRY if t != "custom")
OVERRIDABLE_PROVIDER_TYPES = tuple(t for t in PROVIDER_REGISTRY if t != "openai_codex")

# The schema as it stood before this column, including the two columns earlier
# releases added by ALTER. A database in this shape is what an upgrading user has.
_PRE_PR_TABLE_DDL = """
    CREATE TABLE llm_providers (
        id TEXT NOT NULL PRIMARY KEY,
        provider_type TEXT NOT NULL,
        display_name TEXT NOT NULL,
        base_url TEXT NOT NULL,
        is_enabled INTEGER NOT NULL DEFAULT 1,
        models_json TEXT NOT NULL DEFAULT '[]',
        available_models_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
"""


@pytest.fixture()
def isolated_providers_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A per-test studio.db for direct ``providers_db`` calls."""
    db_path = tmp_path / "studio.db"
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: db_path)
    monkeypatch.setattr(providers_db, "ensure_dir", lambda _path: None)
    providers_db._schema_ready = False
    yield db_path
    providers_db._schema_ready = False


@pytest.fixture()
def provider_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every database the provider routes touch. Yields the studio.db path."""
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    studio_db = tmp_path / "studio.db"
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: studio_db)
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: studio_db)
    monkeypatch.setattr(providers_db, "ensure_dir", lambda _path: None)
    monkeypatch.setattr(credential_secrets, "ensure_dir", lambda _path: None)
    monkeypatch.setattr(
        credential_secrets,
        "get_or_create_credential_encryption_key",
        auth_storage.get_or_create_credential_encryption_key,
    )
    providers_db._schema_ready = False
    credential_secrets._schema_ready = False
    yield studio_db
    providers_db._schema_ready = False
    credential_secrets._schema_ready = False
    auth_storage._credential_encryption_key_cache = None


def _columns(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [row[1] for row in conn.execute("PRAGMA table_info(llm_providers)").fetchall()]
    finally:
        conn.close()


def _raw_override(db_path: Path, provider_id: str):
    """Read the column straight out of SQLite, bypassing every layer under test."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT max_output_tokens FROM llm_providers WHERE id = ?", (provider_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, f"no row for {provider_id!r}"
    return row[0]


def _write_pre_pr_database(db_path: Path) -> None:
    """Build a studio.db in the pre-column shape and put two rows in it."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_PRE_PR_TABLE_DDL)
        for provider_id, provider_type, name in (
            ("old-custom", "custom", "Old Custom"),
            ("old-openai", "openai", "Old OpenAI"),
        ):
            conn.execute(
                "INSERT INTO llm_providers (id, provider_type, display_name, base_url, "
                "is_enabled, models_json, available_models_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)",
                (
                    provider_id,
                    provider_type,
                    name,
                    "https://example.com/v1",
                    '["vendor/model"]',
                    '["vendor/model", "vendor/other"]',
                    "2020-01-01T00:00:00+00:00",
                    "2020-01-01T00:00:00+00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ── Upgrade ───────────────────────────────────────────────────────


def test_a_pre_column_database_migrates_and_keeps_its_rows(isolated_providers_db: Path):
    """The upgrade case: an existing home opened by this build."""
    _write_pre_pr_database(isolated_providers_db)
    assert "max_output_tokens" not in _columns(isolated_providers_db)

    row = providers_db.get_provider("old-custom")
    assert row["max_output_tokens"] is None, "a pre-existing row must read as no override"
    assert row["models"] == ["vendor/model"]
    assert row["available_models"] == ["vendor/model", "vendor/other"]
    assert {p["id"] for p in providers_db.list_providers()} == {"old-custom", "old-openai"}

    # And the migrated row now accepts one.
    assert providers_db.update_provider(id = "old-custom", max_output_tokens = 262144)
    assert _raw_override(isolated_providers_db, "old-custom") == 262144
    assert _raw_override(isolated_providers_db, "old-openai") is None


def test_the_migration_is_idempotent(isolated_providers_db: Path):
    """ALTER TABLE has no IF NOT EXISTS, so a second run must not raise."""
    _write_pre_pr_database(isolated_providers_db)
    for _ in range(3):
        providers_db._schema_ready = False
        assert providers_db.get_provider("old-custom")["max_output_tokens"] is None
    assert _columns(isolated_providers_db).count("max_output_tokens") == 1


def test_a_missing_table_is_created_then_migrated(isolated_providers_db: Path):
    """A fresh home has no table at all: CREATE TABLE never lists the column, so the
    new install reaches it through the same ALTER an upgrade does."""
    assert providers_db.list_providers() == []
    assert "max_output_tokens" in _columns(isolated_providers_db)


# ── Downgrade ─────────────────────────────────────────────────────


def test_the_previous_release_still_reads_and_writes_a_migrated_database(
    isolated_providers_db: Path,
):
    """The revert case. The old code selects * and inserts without naming the column,
    so a migrated file must stay readable and writable by it."""
    providers_db.create_provider(
        id = "new-custom",
        provider_type = "custom",
        display_name = "New Custom",
        base_url = "https://example.com/v1",
        models = ["vendor/model"],
        max_output_tokens = 384000,
    )

    conn = sqlite3.connect(str(isolated_providers_db))
    conn.row_factory = sqlite3.Row
    try:
        # Exactly what the previous release's get_provider does.
        row = dict(
            conn.execute("SELECT * FROM llm_providers WHERE id = ?", ("new-custom",)).fetchone()
        )
        assert row["display_name"] == "New Custom"
        assert row["models_json"] == '["vendor/model"]'
        # An INSERT that never mentions the column, as the old code writes.
        conn.execute(
            "INSERT INTO llm_providers (id, provider_type, display_name, base_url, "
            "is_enabled, models_json, available_models_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, '[]', '[]', ?, ?)",
            (
                "downgrade-written",
                "openai",
                "Written By Old Code",
                "https://api.openai.com/v1",
                "2020-01-01T00:00:00+00:00",
                "2020-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Back on this build: the old row reads as no override, the new one kept its value.
    assert providers_db.get_provider("downgrade-written")["max_output_tokens"] is None
    assert providers_db.get_provider("new-custom")["max_output_tokens"] == 384000


# ── Route contract ────────────────────────────────────────────────


def _create(payload: ProviderCreate):
    return asyncio.run(
        providers_route.create_provider_config(payload, credential = CREDENTIAL, via_api_key = False)
    )


def _update(provider_id: str, payload: ProviderUpdate):
    return asyncio.run(
        providers_route.update_provider_config(
            provider_id, payload, credential = CREDENTIAL, via_api_key = False
        )
    )


@pytest.mark.parametrize("provider_type", NON_CUSTOM_PROVIDER_TYPES)
def test_a_non_custom_provider_accepts_an_explicit_null_override(
    provider_routes: Path, provider_type: str
):
    """A blank Max Tokens limit field serialises as null rather than as an omission, so
    an unrelated edit of a row with no override -- a rename, a model change, a key
    rotation -- sends the null along and rejecting it failed the whole edit."""
    providers_db.create_provider(
        id = f"{provider_type}-1",
        provider_type = provider_type,
        display_name = provider_type,
        base_url = "https://example.com/v1",
    )
    updated = _update(
        f"{provider_type}-1", ProviderUpdate(display_name = "Renamed", max_output_tokens = None)
    )
    assert updated.display_name == "Renamed"
    assert updated.max_output_tokens is None
    assert _raw_override(provider_routes, f"{provider_type}-1") is None


@pytest.mark.parametrize("provider_type", OVERRIDABLE_PROVIDER_TYPES)
def test_every_provider_type_but_codex_takes_a_real_override(
    provider_routes: Path, provider_type: str
):
    """A documented per-model cap still wins in the frontend; the override replaces the
    32,768-token fallback every provider reaches for an unlisted model."""
    providers_db.create_provider(
        id = f"{provider_type}-1",
        provider_type = provider_type,
        display_name = provider_type,
        base_url = "https://example.com/v1",
    )
    assert (
        _update(f"{provider_type}-1", ProviderUpdate(max_output_tokens = 262144)).max_output_tokens
        == 262144
    )
    assert _raw_override(provider_routes, f"{provider_type}-1") == 262144


def test_a_chatgpt_subscription_rejects_a_real_override(provider_routes: Path):
    """Codex routing, model list and output cap are fixed, so a stored override would
    never be read."""
    providers_db.create_provider(
        id = "openai_codex-1",
        provider_type = "openai_codex",
        display_name = "ChatGPT",
        base_url = "https://chatgpt.com/backend-api/codex",
    )
    with pytest.raises(HTTPException) as error:
        _update("openai_codex-1", ProviderUpdate(max_output_tokens = 65536))
    assert error.value.status_code == 400
    assert _raw_override(provider_routes, "openai_codex-1") is None

    # Create takes the same contract, and reaches it before the auth one, so the caller
    # is told which rule stopped them.
    with pytest.raises(HTTPException) as created:
        _create(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT",
                max_output_tokens = 65536,
            )
        )
    assert created.value.status_code == 400
    assert "fixed Max Tokens limit" in created.value.detail


def test_a_custom_connection_can_set_preserve_and_clear_its_override(provider_routes: Path):
    """The whole lifecycle, asserted against the stored row rather than the response."""
    created = _create(
        ProviderCreate(
            provider_type = "custom",
            display_name = "Custom",
            base_url = "https://example.com/v1",
            models = ["vendor/model"],
            max_output_tokens = 131072,
        )
    )
    assert _raw_override(provider_routes, created.id) == 131072

    assert _update(created.id, ProviderUpdate(max_output_tokens = 65536)).max_output_tokens == 65536

    # An unrelated edit must leave it alone: omitted is not the same as null.
    preserved = _update(created.id, ProviderUpdate(display_name = "Renamed Custom"))
    assert preserved.display_name == "Renamed Custom"
    assert _raw_override(provider_routes, created.id) == 65536

    assert _update(created.id, ProviderUpdate(max_output_tokens = None)).max_output_tokens is None
    assert _raw_override(provider_routes, created.id) is None


def test_an_override_only_update_is_recognised_as_a_metadata_request(provider_routes: Path):
    """A request carrying nothing but the override must not be turned away as
    "No fields to update"."""
    created = _create(
        ProviderCreate(
            provider_type = "custom",
            display_name = "Custom",
            base_url = "https://example.com/v1",
            models = ["vendor/model"],
        )
    )
    assert _update(created.id, ProviderUpdate(max_output_tokens = 200000)).max_output_tokens == 200000


def test_the_largest_accepted_value_round_trips_exactly(provider_routes: Path):
    """SQLite INTEGER is 8 bytes, so the top of the accepted range must come back
    identical rather than as a float."""
    value = 9007199254740991
    created = _create(
        ProviderCreate(
            provider_type = "custom",
            display_name = "Custom",
            base_url = "https://example.com/v1",
            models = ["vendor/model"],
            max_output_tokens = value,
        )
    )
    stored = _raw_override(provider_routes, created.id)
    assert stored == value and isinstance(stored, int)


def test_a_failed_credential_write_restores_the_previous_override(
    provider_routes: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed key write rolls the metadata update back in the shared transaction."""
    created = _create(
        ProviderCreate(
            provider_type = "custom",
            display_name = "Custom",
            base_url = "https://example.com/v1",
            models = ["vendor/model"],
            max_output_tokens = 131072,
        )
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("keyring is unavailable")

    monkeypatch.setattr(providers_route.credential_secrets, "save_provider_api_key", _boom)
    with pytest.raises(Exception):
        _update(created.id, ProviderUpdate(max_output_tokens = 262144, encrypted_api_key = "x"))
    assert _raw_override(provider_routes, created.id) == 131072

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.user_assets_validation import UserAssetValidationError
from storage import studio_db, user_assets_db
from utils.paths import studio_db_path


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(user_assets_db, "_now_ms", lambda: 1_800_000_000_000)


def recipe(asset_id="r1", payload=None):
    return {"id": asset_id, "name": "Recipe", "payload": payload or {"nodes": []}}


def execution(asset_id="e1", **extra):
    return {
        "id": asset_id,
        "recipeId": "r1",
        "status": "completed",
        "createdAt": 1_700_000_000_000,
        "finishedAt": 1_700_000_001_000,
        **extra,
    }


def test_owner_isolation_and_secrets_never_reach_sqlite():
    user_assets_db.create_recipe("owner-a", recipe())
    user_assets_db.create_recipe("owner-b", {**recipe(), "name": "Other"})
    assert user_assets_db.get_recipe("owner-a", "r1")["name"] == "Recipe"
    assert user_assets_db.get_recipe("owner-b", "r1")["name"] == "Other"
    assert user_assets_db.get_recipe("owner-c", "r1") is None

    marker = "never-store-this-token"
    with pytest.raises(UserAssetValidationError, match="secret fields"):
        user_assets_db.create_recipe("owner-a", recipe("secret", {"apiKey": marker}))
    path = studio_db_path()
    assert marker.encode() not in path.read_bytes()


def test_execution_timestamps_remain_monotonic_across_clock_rollback(monkeypatch):
    user_assets_db.create_recipe("owner", recipe())
    future = 1_900_000_000_000
    inserted = user_assets_db.upsert_recipe_execution(
        "owner", "r1", "e1", execution(createdAt=future, finishedAt=future + 1)
    )
    monkeypatch.setattr(user_assets_db, "_now_ms", lambda: 1)
    updated = user_assets_db.upsert_recipe_execution(
        "owner",
        "r1",
        "e1",
        execution(createdAt=future, finishedAt=future + 2),
        expected_revision=inserted["revision"],
    )
    assert inserted["updatedAt"] >= future
    assert updated["updatedAt"] > inserted["updatedAt"]
    with pytest.raises(UserAssetValidationError, match="finishedAt"):
        user_assets_db.upsert_recipe_execution(
            "owner", "r1", "bad", execution("bad", createdAt=100, finishedAt=99)
        )


def test_corrected_legacy_rejection_retries_after_restart(monkeypatch):
    source = "recipe-indexeddb-v1"
    rejected = user_assets_db.import_legacy_assets(
        "owner", source, [{**recipe("retry-me"), "name": "", "createdAt": 1}], []
    )
    assert rejected["recipes"][0]["outcome"] == "rejected"
    assert user_assets_db.list_legacy_imports("owner", source)["recipes"] == []

    monkeypatch.setattr(studio_db, "_schema_ready", False)
    corrected = user_assets_db.import_legacy_assets(
        "owner", source, [{**recipe("retry-me"), "createdAt": 1}], []
    )
    assert corrected["recipes"][0]["outcome"] == "imported"


def test_route_unsafe_ids_are_never_persisted():
    with pytest.raises(UserAssetValidationError, match="URL path segment"):
        user_assets_db.create_recipe("owner", recipe("folder/recipe"))
    assert user_assets_db.list_recipes("owner") == []

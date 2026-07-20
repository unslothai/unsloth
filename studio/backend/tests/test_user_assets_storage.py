# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.user_assets_validation import UserAssetValidationError
from models.user_assets import RecipeUpdateRequest
from routes.user_assets.recipes import _recipe_input
from storage import studio_db, user_assets_db
from utils.paths import studio_db_path


@pytest.fixture(autouse = True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(user_assets_db, "_now_ms", lambda: 1_800_000_000_000)


def recipe(asset_id = "r1", payload = None):
    return {"id": asset_id, "name": "Recipe", "payload": payload or {"nodes": []}}


def execution(asset_id = "e1", **extra):
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
    with pytest.raises(UserAssetValidationError, match = "secret fields"):
        user_assets_db.create_recipe("owner-a", recipe("secret", {"apiKey": marker}))
    path = studio_db_path()
    assert marker.encode() not in path.read_bytes()


def test_execution_timestamps_remain_monotonic_across_clock_rollback(monkeypatch):
    user_assets_db.create_recipe("owner", recipe())
    future = 1_900_000_000_000
    inserted = user_assets_db.upsert_recipe_execution(
        "owner", "r1", "e1", execution(createdAt = future, finishedAt = future + 1)
    )
    monkeypatch.setattr(user_assets_db, "_now_ms", lambda: 1)
    updated = user_assets_db.upsert_recipe_execution(
        "owner",
        "r1",
        "e1",
        execution(createdAt = future, finishedAt = future + 2),
        expected_revision = inserted["revision"],
    )
    assert inserted["updatedAt"] >= future
    assert updated["updatedAt"] > inserted["updatedAt"]
    with pytest.raises(UserAssetValidationError, match = "finishedAt"):
        user_assets_db.upsert_recipe_execution(
            "owner", "r1", "bad", execution("bad", createdAt = 100, finishedAt = 99)
        )


def test_recipe_update_preserves_omitted_learning_linkage_and_can_clear_it():
    inserted = user_assets_db.create_recipe(
        "owner",
        {
            **recipe(),
            "learningRecipeId": "learning-1",
            "learningRecipeTitle": "Learning Recipe",
        },
    )

    updated = user_assets_db.update_recipe(
        "owner",
        "r1",
        {"name": "Edited", "payload": {"nodes": [{"id": "node-1"}]}},
        inserted["revision"],
    )
    assert updated["learningRecipeId"] == "learning-1"
    assert updated["learningRecipeTitle"] == "Learning Recipe"

    cleared = user_assets_db.update_recipe(
        "owner",
        "r1",
        {
            "name": "Unlinked",
            "payload": updated["payload"],
            "learningRecipeId": None,
            "learningRecipeTitle": None,
        },
        updated["revision"],
    )
    assert cleared["learningRecipeId"] is None
    assert cleared["learningRecipeTitle"] is None


def test_recipe_update_request_distinguishes_omitted_links_from_explicit_nulls():
    common = {"name": "Edited", "payload": {"nodes": []}, "revision": 1}
    omitted = _recipe_input(RecipeUpdateRequest(**common))
    cleared = _recipe_input(
        RecipeUpdateRequest(
            **common,
            learningRecipeId = None,
            learningRecipeTitle = None,
        )
    )

    assert "learningRecipeId" not in omitted
    assert "learningRecipeTitle" not in omitted
    assert cleared["learningRecipeId"] is None
    assert cleared["learningRecipeTitle"] is None


def test_legacy_recipe_updated_at_is_validated_preserved_and_ordered():
    imported = user_assets_db.import_legacy_assets(
        "owner",
        "recipe-indexeddb-v1",
        [
            {**recipe("exact"), "createdAt": 100, "updatedAt": 200},
            {**recipe("before-created"), "createdAt": 300, "updatedAt": 200},
            {**recipe("fallback"), "createdAt": 400},
            {**recipe("invalid"), "createdAt": 100, "updatedAt": "200"},
        ],
        [],
    )

    assert [result["outcome"] for result in imported["recipes"]] == [
        "imported",
        "imported",
        "imported",
        "rejected",
    ]
    assert imported["recipes"][-1]["reason"] == "invalid_timestamp"
    assert user_assets_db.get_recipe("owner", "exact")["updatedAt"] == 200
    assert user_assets_db.get_recipe("owner", "before-created")["updatedAt"] == 300
    assert user_assets_db.get_recipe("owner", "fallback")["updatedAt"] == 1_800_000_000_000
    assert [record["id"] for record in user_assets_db.list_recipes("owner")] == [
        "fallback",
        "before-created",
        "exact",
    ]


def test_corrected_legacy_rejection_retries_after_restart(monkeypatch):
    source = "recipe-indexeddb-v1"
    rejected = user_assets_db.import_legacy_assets(
        "owner", source, [{**recipe("retry-me"), "name": "", "createdAt": 1}], []
    )
    assert rejected["recipes"][0]["outcome"] == "rejected"

    assert user_assets_db.list_legacy_imports("owner", source)["recipes"] == []

    # Old builds persisted validation failures as rejected rows.
    conn = studio_db.get_connection()
    conn.execute(
        """
        INSERT INTO user_asset_legacy_imports
            (owner_subject, source, entity_kind, legacy_id, outcome, reason, imported_at)
        VALUES (?, ?, 'recipe', 'retry-me', 'rejected', 'invalid_name', ?)
        """,
        ("owner", source, 1_700_000_000_000),
    )
    conn.commit()
    conn.close()
    assert user_assets_db.list_legacy_imports("owner", source)["recipes"] == []

    monkeypatch.setattr(studio_db, "_schema_ready", False)
    corrected = user_assets_db.import_legacy_assets(
        "owner", source, [{**recipe("retry-me"), "createdAt": 1}], []
    )
    assert corrected["recipes"][0]["outcome"] == "imported"


def test_route_unsafe_ids_are_never_persisted():
    with pytest.raises(UserAssetValidationError, match = "URL path segment"):
        user_assets_db.create_recipe("owner", recipe("folder/recipe"))
    assert user_assets_db.list_recipes("owner") == []

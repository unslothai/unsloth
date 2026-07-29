# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi import HTTPException

from core.data_recipe.jobs.manager import JobManager
from core.user_assets_validation import MAX_TIMESTAMP_MS, UserAssetValidationError
from models.user_assets import ExecutionUpsertRequest, RecipeUpdateRequest
from models.data_recipe import PublishDatasetRequest
from routes.data_recipe import jobs as data_recipe_jobs
from routes.user_assets import legacy_import as user_assets_legacy_import
from routes.user_assets import recipes as user_assets_recipes
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


def test_recipe_summaries_are_payload_free_and_paginated():
    for asset_id in ("r1", "r2", "r3"):
        user_assets_db.create_recipe(
            "owner",
            recipe(asset_id, {"nodes": [], "large": "payload-must-not-be-listed"}),
        )

    first = user_assets_db.list_recipe_summaries("owner", limit = 2)
    second = user_assets_db.list_recipe_summaries("owner", cursor = first["nextCursor"], limit = 2)

    assert [item["id"] for item in first["recipes"]] == ["r1", "r2"]
    assert [item["id"] for item in second["recipes"]] == ["r3"]
    assert all("payload" not in item for item in first["recipes"] + second["recipes"])
    assert second["nextCursor"] is None


def test_recipe_pagination_does_not_drop_a_recipe_updated_between_pages(monkeypatch):
    timestamps = iter((100, 200, 300, 400))
    monkeypatch.setattr(user_assets_db, "_now_ms", lambda: next(timestamps))
    for asset_id in ("r1", "r2", "r3"):
        user_assets_db.create_recipe("owner", recipe(asset_id))

    first = user_assets_db.list_recipe_summaries("owner", limit = 2)
    updated = user_assets_db.get_recipe("owner", "r3")
    user_assets_db.update_recipe(
        "owner", "r3", {**recipe("r3"), "name": "Updated"}, updated["revision"]
    )
    second = user_assets_db.list_recipe_summaries("owner", cursor = first["nextCursor"], limit = 2)

    assert [item["id"] for item in first["recipes"] + second["recipes"]] == ["r1", "r2", "r3"]


def test_unicode_ids_generate_reusable_bounded_cursors():
    ids = ["a", "界" * 128, "🙂" * 128]
    for asset_id in ids:
        user_assets_db.create_recipe("owner", recipe(asset_id))

    first = user_assets_db.list_recipe_summaries("owner", limit = 2)
    assert 0 < len(first["nextCursor"]) <= user_assets_db.MAX_CURSOR_CHARS
    second = user_assets_db.list_recipe_summaries("owner", cursor = first["nextCursor"], limit = 2)

    assert len(first["recipes"]) == 2
    assert len(second["recipes"]) == 1
    for encode, decode in (
        (user_assets_db._encode_recipe_cursor, user_assets_db._decode_recipe_cursor),
        (user_assets_db._encode_execution_cursor, user_assets_db._decode_execution_cursor),
    ):
        cursor = encode(123, "🙂" * 128)
        assert len(cursor) <= user_assets_db.MAX_CURSOR_CHARS
        assert decode(cursor) == (123, "🙂" * 128)


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


def test_completed_artifact_handoff_survives_manager_restart(monkeypatch):
    user_assets_db.create_recipe("owner", recipe())
    inserted = user_assets_db.upsert_recipe_execution(
        "owner",
        "r1",
        "e1",
        execution(
            status = "active",
            finishedAt = None,
            artifact_path = None,
            jobId = "completed-job",
            kind = "full",
        ),
    )
    user_assets_db.record_completed_artifact_handoff(
        "owner", "completed-job", "recipes/recipe_r1", "full"
    )
    restarted_manager = JobManager()
    monkeypatch.setattr(data_recipe_jobs, "get_job_manager", lambda: restarted_manager)
    monkeypatch.setattr(user_assets_recipes, "get_job_manager", lambda: restarted_manager)

    status = data_recipe_jobs.job_status("completed-job", "owner")
    saved = user_assets_recipes.upsert_recipe_execution(
        "r1",
        "e1",
        ExecutionUpsertRequest(
            **execution(
                artifact_path = "recipes/recipe_r1",
                jobId = "completed-job",
                kind = "full",
                revision = inserted["revision"],
            )
        ),
        "owner",
    )

    assert status["status"] == "completed"
    assert status["artifact_path"] == "recipes/recipe_r1"
    assert saved["artifact_path"] == "recipes/recipe_r1"
    assert restarted_manager.get_completed_artifact_status("completed-job", "owner") is None


def test_execution_update_uses_immutable_stored_creation_timestamp():
    user_assets_db.create_recipe("owner", recipe())
    inserted = user_assets_db.upsert_recipe_execution(
        "owner",
        "r1",
        "e1",
        execution(createdAt = 1_000, finishedAt = None, status = "active"),
    )

    with pytest.raises(UserAssetValidationError, match = "stored createdAt"):
        user_assets_db.upsert_recipe_execution(
            "owner",
            "r1",
            "e1",
            execution(createdAt = 0, finishedAt = 500),
            expected_revision = inserted["revision"],
        )

    unchanged = user_assets_db.get_recipe_execution("owner", "r1", "e1")
    assert unchanged["revision"] == inserted["revision"]
    assert unchanged["createdAt"] == 1_000
    normalized = user_assets_db.upsert_recipe_execution(
        "owner",
        "r1",
        "e1",
        execution(createdAt = 0, finishedAt = 1_500),
        expected_revision = inserted["revision"],
    )
    assert normalized["createdAt"] == 1_000
    assert normalized["finishedAt"] == 1_500


def test_monotonic_updates_stay_within_the_timestamp_limit():
    imported = user_assets_db.import_legacy_assets(
        "owner",
        "recipe-indexeddb-v1",
        [
            {
                **recipe(),
                "createdAt": MAX_TIMESTAMP_MS,
                "updatedAt": MAX_TIMESTAMP_MS,
            }
        ],
        [execution(createdAt = MAX_TIMESTAMP_MS, finishedAt = MAX_TIMESTAMP_MS)],
    )
    assert imported["recipes"][0]["outcome"] == "imported"
    assert imported["executions"][0]["outcome"] == "imported"

    updated = user_assets_db.update_recipe("owner", "r1", recipe(), expected_revision = 1)
    updated_execution = user_assets_db.upsert_recipe_execution(
        "owner",
        "r1",
        "e1",
        execution(createdAt = MAX_TIMESTAMP_MS, finishedAt = MAX_TIMESTAMP_MS),
        expected_revision = 1,
    )

    assert updated["updatedAt"] == MAX_TIMESTAMP_MS
    assert updated_execution["updatedAt"] == MAX_TIMESTAMP_MS
    assert (
        user_assets_db.list_recipe_summaries("owner")["recipes"][0]["updatedAt"] == MAX_TIMESTAMP_MS
    )


def test_execution_upsert_omits_absent_optional_metadata(monkeypatch):
    user_assets_db.create_recipe("owner", recipe())
    monkeypatch.setattr(
        user_assets_recipes,
        "get_job_manager",
        lambda: object(),
    )

    saved = user_assets_recipes.upsert_recipe_execution(
        "r1",
        "e1",
        ExecutionUpsertRequest(createdAt = 1),
        "owner",
    )

    assert saved["createdAt"] == 1
    assert "kind" not in saved
    assert "status" not in saved


def test_execution_artifact_reference_is_owner_scoped_and_durable():
    user_assets_db.create_recipe("owner-a", recipe())
    user_assets_db.create_recipe("owner-b", recipe())
    saved = user_assets_db.upsert_recipe_execution(
        "owner-a",
        "r1",
        "e1",
        execution(artifact_path = "recipes/recipe_r1"),
    )

    assert saved["artifact_path"] == "recipes/recipe_r1"
    assert (
        user_assets_db.get_recipe_execution("owner-a", "r1", "e1")["artifact_path"]
        == "recipes/recipe_r1"
    )
    assert user_assets_db.get_recipe_execution("owner-b", "r1", "e1") is None


def test_persisted_completed_execution_remains_publishable_after_job_replacement(monkeypatch):
    artifact_path = "recipes/recipe_r1"
    user_assets_db.create_recipe("owner", recipe())
    user_assets_db.upsert_recipe_execution(
        "owner",
        "r1",
        "e1",
        execution(
            artifact_path = artifact_path,
            jobId = "completed-job",
            kind = "full",
        ),
    )
    monkeypatch.setattr(
        data_recipe_jobs,
        "get_job_manager",
        lambda: type("ReplacedJobManager", (), {"get_status": lambda *_args: None})(),
    )
    published = {}

    def fake_publish_recipe_dataset(**kwargs):
        published.update(kwargs)
        return "https://huggingface.co/datasets/owner/dataset"

    monkeypatch.setattr(data_recipe_jobs, "publish_recipe_dataset", fake_publish_recipe_dataset)

    status = data_recipe_jobs.job_status("completed-job", "owner")
    result = data_recipe_jobs.publish_job_dataset(
        "completed-job",
        PublishDatasetRequest(repo_id = "owner/dataset", description = "Dataset"),
        "owner",
    )

    assert status == {
        "job_id": "completed-job",
        "status": "completed",
        "execution_type": "full",
        "artifact_path": artifact_path,
    }
    assert result["success"] is True
    assert published["artifact_path"] == artifact_path
    assert (
        user_assets_db.get_completed_recipe_execution_by_job_id("other-owner", "completed-job")
        is None
    )


def test_completed_execution_persists_when_replaced_job_cannot_verify_artifact(monkeypatch):
    user_assets_db.create_recipe("owner", recipe())
    monkeypatch.setattr(
        user_assets_recipes,
        "get_job_manager",
        lambda: type(
            "ReplacedJobManager",
            (),
            {"get_owned_completed_artifact_path": lambda *_args: None},
        )(),
    )

    saved = user_assets_recipes.upsert_recipe_execution(
        "r1",
        "e1",
        ExecutionUpsertRequest(
            **execution(
                artifact_path = "recipes/recipe_r1",
                jobId = "replaced-job",
                kind = "full",
                run_name = "Finished run",
                recipeSignature = "signature",
                rows = 0,
                datasetTotal = 0,
                completed_columns = [],
            )
        ),
        "owner",
    )

    assert saved["status"] == "completed"
    assert user_assets_db.get_recipe_execution("owner", "r1", "e1").get("artifact_path") is None


def test_completed_execution_preserves_previously_verified_artifact(monkeypatch):
    user_assets_db.create_recipe("owner", recipe())
    artifact_path = "recipes/recipe_r1"
    released = []
    monkeypatch.setattr(
        user_assets_recipes,
        "get_job_manager",
        lambda: type(
            "CompletedJobManager",
            (),
            {
                "get_owned_completed_artifact_path": lambda *_args: artifact_path,
                "release_completed_artifact_path": (
                    lambda _self, job_id, owner, path: released.append((job_id, owner, path))
                ),
            },
        )(),
    )
    saved = user_assets_recipes.upsert_recipe_execution(
        "r1",
        "e1",
        ExecutionUpsertRequest(
            **execution(
                artifact_path = artifact_path,
                jobId = "completed-job",
                kind = "full",
                run_name = "Finished run",
                recipeSignature = "signature",
                rows = 1,
                datasetTotal = 1,
                completed_columns = [],
            )
        ),
        "owner",
    )
    assert released == [("completed-job", "owner", artifact_path)]

    monkeypatch.setattr(
        user_assets_recipes,
        "get_job_manager",
        lambda: type(
            "ReplacedJobManager",
            (),
            {
                "get_owned_completed_artifact_path": lambda *_args: None,
                "release_completed_artifact_path": lambda *_args: None,
            },
        )(),
    )
    enriched = user_assets_recipes.upsert_recipe_execution(
        "r1",
        "e1",
        ExecutionUpsertRequest(
            **execution(
                artifact_path = artifact_path,
                jobId = "completed-job",
                kind = "full",
                run_name = "Finished run",
                recipeSignature = "signature",
                rows = 2,
                datasetTotal = 2,
                completed_columns = [],
                revision = saved["revision"],
            )
        ),
        "owner",
    )

    assert enriched["artifact_path"] == artifact_path
    assert enriched["rows"] == 2

    spoofed = user_assets_recipes.upsert_recipe_execution(
        "r1",
        "e1",
        ExecutionUpsertRequest(
            **execution(
                artifact_path = artifact_path,
                jobId = "unverified-job",
                kind = "full",
                run_name = "Finished run",
                recipeSignature = "signature",
                rows = 3,
                datasetTotal = 3,
                completed_columns = [],
                revision = enriched["revision"],
            )
        ),
        "owner",
    )

    assert spoofed["artifact_path"] == artifact_path
    assert spoofed["jobId"] == "completed-job"
    assert (
        user_assets_db.get_completed_recipe_execution_by_job_id("owner", "unverified-job") is None
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


def test_legacy_execution_import_discards_client_supplied_artifact_path():
    user_assets_db.create_recipe("owner", recipe())

    imported = user_assets_db.import_legacy_assets(
        "owner",
        "recipe-indexeddb-v1",
        [],
        [execution(artifact_path = "recipes/recipe_other-account")],
    )

    assert imported["executions"][0]["outcome"] == "imported"
    saved = user_assets_db.get_recipe_execution("owner", "r1", "e1")
    assert saved is not None
    assert "artifact_path" not in saved


def test_legacy_timestamp_overflow_rejects_only_the_invalid_item():
    imported = user_assets_db.import_legacy_assets(
        "owner",
        "recipe-indexeddb-v1",
        [
            {**recipe("overflow"), "createdAt": 2**63},
            {**recipe("valid"), "createdAt": 100},
        ],
        [],
    )

    assert imported["recipes"][0] == {
        "id": "overflow",
        "outcome": "rejected",
        "reason": "invalid_timestamp",
    }
    assert imported["recipes"][1]["outcome"] == "imported"
    assert user_assets_db.get_recipe("owner", "overflow") is None
    assert user_assets_db.get_recipe("owner", "valid") is not None


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


def test_legacy_import_ledger_is_keyset_paginated():
    source = "recipe-indexeddb-v1"
    conn = studio_db.get_connection()
    conn.executemany(
        """
        INSERT INTO user_asset_legacy_imports
            (owner_subject, source, entity_kind, legacy_id, outcome, reason, imported_at)
        VALUES (?, ?, ?, ?, 'imported', NULL, ?)
        """,
        [
            ("owner", source, "execution", "e1", 1),
            ("owner", source, "execution", "e2", 2),
            ("owner", source, "recipe", "r1", 3),
            ("owner", source, "recipe", "r2", 4),
            ("owner", source, "recipe", "r3", 5),
        ],
    )
    conn.commit()
    conn.close()

    pages = []
    cursor = None
    while True:
        page = user_assets_db.list_legacy_imports("owner", source, cursor = cursor, limit = 2)
        pages.append(page)
        cursor = page["nextCursor"]
        if cursor is None:
            break

    assert [len(page["recipes"]) + len(page["executions"]) for page in pages] == [2, 2, 1]
    assert [item for page in pages for item in page["executions"]] == ["e1", "e2"]
    assert [item for page in pages for item in page["recipes"]] == ["r1", "r2", "r3"]
    assert pages[-1]["nextCursor"] is None
    with pytest.raises(UserAssetValidationError, match = "cursor"):
        user_assets_db.list_legacy_imports("owner", source, cursor = "not-a-cursor")
    with pytest.raises(HTTPException) as route_error:
        user_assets_legacy_import.bootstrap(cursor = "not-a-cursor", limit = 2, current_subject = "owner")
    assert route_error.value.status_code == 422
    assert route_error.value.detail["code"] == "invalid_cursor"


def test_route_unsafe_ids_are_never_persisted():
    with pytest.raises(UserAssetValidationError, match = "URL path segment"):
        user_assets_db.create_recipe("owner", recipe("folder/recipe"))
    with pytest.raises(UserAssetValidationError, match = "URL path segment"):
        user_assets_db.create_recipe("owner", recipe("\x00recipe"))
    with pytest.raises(UserAssetValidationError, match = "non-empty"):
        user_assets_db.create_recipe("owner", {**recipe(), "name": "\x00Recipe"})
    assert user_assets_db.list_recipes("owner") == []

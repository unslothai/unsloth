# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.data_recipe.library import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _recipe(
    recipe_id = "r1",
    name = "Recipe",
    updated_at = 2000,
    **payload,
) -> dict:
    return {
        "id": recipe_id,
        "name": name,
        "payload": {"recipe": {}, **payload},
        "createdAt": 1000,
        "updatedAt": updated_at,
    }


def _execution(
    execution_id = "e1",
    recipe_id = "r1",
    **extra,
) -> dict:
    return {"id": execution_id, "recipeId": recipe_id, "createdAt": 1000, **extra}


def test_recipe_round_trip_keeps_payload_and_learning_fields():
    client = _client()
    providers = [{"name": "p", "api_key": "sk-inline", "api_key_env": "OPENAI_API_KEY"}]
    body = _recipe(model_providers = providers)
    body["learningRecipeId"] = "text-to-sql"
    assert client.put("/recipes/r1", json = body).status_code == 200

    stored = client.get("/recipes/r1").json()
    assert stored["payload"]["model_providers"] == providers
    assert stored["learningRecipeId"] == "text-to-sql"
    assert "learningRecipeTitle" not in stored
    assert [r["id"] for r in client.get("/recipes").json()["recipes"]] == ["r1"]


def test_recipes_list_newest_first_and_update_keeps_created_at():
    client = _client()
    client.put("/recipes/old", json = _recipe("old", updated_at = 1000))
    client.put("/recipes/new", json = _recipe("new", updated_at = 3000))
    edited = {**_recipe("old", name = "Edited", updated_at = 4000), "createdAt": 9999}
    client.put("/recipes/old", json = edited)

    recipes = client.get("/recipes").json()["recipes"]
    assert [r["id"] for r in recipes] == ["old", "new"]
    assert recipes[0]["name"] == "Edited"
    assert recipes[0]["createdAt"] == 1000


def test_deleted_recipe_cannot_be_revived_by_a_stale_save_or_legacy_import():
    client = _client()
    client.put("/recipes/r1", json = _recipe())
    client.put("/recipes/r1/executions/e1", json = _execution())
    assert client.delete("/recipes/r1").status_code == 204

    assert client.get("/recipes/r1").status_code == 404
    assert client.get("/recipes/r1/executions").json() == {"executions": []}
    assert client.put("/recipes/r1", json = _recipe()).status_code == 410
    imported = client.post(
        "/recipes/import", json = {"recipes": [_recipe()], "executions": [_execution()]}
    )
    assert imported.json() == {"recipes": 0, "executions": 0}
    assert client.get("/recipes").json() == {"recipes": []}


def test_execution_needs_its_recipe_and_cannot_move_to_another():
    client = _client()
    assert client.put("/recipes/r1/executions/e1", json = _execution()).status_code == 404
    client.put("/recipes/r1", json = _recipe())
    client.put("/recipes/r2", json = _recipe("r2"))
    assert (
        client.put("/recipes/r1/executions/e1", json = _execution(status = "running")).status_code
        == 204
    )
    assert (
        client.put("/recipes/r1/executions/e1", json = _execution(status = "completed")).status_code
        == 204
    )
    assert client.put("/recipes/r1/executions/e2", json = _execution()).status_code == 400

    assert (
        client.put("/recipes/r2/executions/e1", json = _execution(recipe_id = "r2")).status_code == 404
    )
    assert client.get("/recipes/r1/executions").json()["executions"] == [
        _execution(status = "completed")
    ]
    assert client.get("/recipes/r2/executions").json() == {"executions": []}


def test_legacy_import_is_insert_only_and_idempotent():
    client = _client()
    client.put("/recipes/r1", json = _recipe(name = "Server copy", updated_at = 5000))
    body = {
        "recipes": [_recipe(name = "Browser copy"), _recipe("r2")],
        "executions": [_execution(), _execution("e2", "r2"), _execution("orphan", "missing")],
    }
    assert client.post("/recipes/import", json = body).json() == {"recipes": 1, "executions": 2}
    assert client.post("/recipes/import", json = body).json() == {"recipes": 0, "executions": 0}

    assert client.get("/recipes/r1").json()["name"] == "Server copy"
    assert client.get("/recipes/r2").status_code == 200
    assert [e["id"] for e in client.get("/recipes/r2/executions").json()["executions"]] == ["e2"]


def test_out_of_range_timestamp_is_a_validation_error_not_a_server_error():
    client = _client()
    huge = {**_recipe(), "createdAt": 10**20}
    assert client.put("/recipes/r1", json = huge).status_code == 422
    assert client.post("/recipes/import", json = {"recipes": [huge]}).status_code == 422


def test_stale_save_is_a_conflict_and_a_current_one_wins():
    client = _client()
    client.put("/recipes/r1", json = _recipe(updated_at = 2000))
    other_tab = {**_recipe(name = "Other tab", updated_at = 3000), "baseUpdatedAt": 2000}
    assert client.put("/recipes/r1", json = other_tab).status_code == 200

    stale = {**_recipe(name = "Stale tab", updated_at = 4000), "baseUpdatedAt": 2000}
    assert client.put("/recipes/r1", json = stale).status_code == 409
    assert client.get("/recipes/r1").json()["name"] == "Other tab"

    current = {**_recipe(name = "Reloaded", updated_at = 5000), "baseUpdatedAt": 3000}
    assert client.put("/recipes/r1", json = current).json()["name"] == "Reloaded"


def test_older_run_snapshot_from_another_tab_does_not_replace_a_newer_one():
    client = _client()
    client.put("/recipes/r1", json = _recipe())
    put = lambda **extra: client.put("/recipes/r1/executions/e1", json = _execution(**extra))
    assert put(status = "running", lastEventId = 5).status_code == 204
    assert put(status = "completed", lastEventId = 9).status_code == 204
    assert put(status = "running", lastEventId = 7).status_code == 204
    assert put(status = "running", lastEventId = 12).status_code == 204
    assert put(status = "cancelled", lastEventId = 12).status_code == 204

    stored = client.get("/recipes/r1/executions").json()["executions"][0]
    assert (stored["status"], stored["lastEventId"]) == ("cancelled", 12)


def test_unenriched_snapshot_at_the_same_event_keeps_the_enriched_one():
    client = _client()
    client.put("/recipes/r1", json = _recipe())
    put = lambda **extra: client.put("/recipes/r1/executions/e1", json = _execution(**extra))
    assert put(status = "completed", lastEventId = 9, analysis = {"num_records": 3}).status_code == 204
    assert put(status = "completed", lastEventId = 9, analysis = None).status_code == 204

    stored = client.get("/recipes/r1/executions").json()["executions"][0]
    assert stored["analysis"] == {"num_records": 3}

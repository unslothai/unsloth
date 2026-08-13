# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from auth.authentication import get_current_subject
from routes import settings as settings_routes
from storage import studio_db


def _client(monkeypatch, stored = None) -> TestClient:
    stored = {} if stored is None else stored
    monkeypatch.setattr(
        studio_db,
        "get_app_setting",
        lambda key, default = None: stored.get(key, default),
    )
    monkeypatch.setattr(studio_db, "upsert_app_settings", stored.update)
    app = FastAPI()
    app.include_router(settings_routes.router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "preset-test-user"
    return TestClient(app)


def test_image_and_video_presets_persist_independently(monkeypatch):
    client = _client(monkeypatch)
    image_state = {
        "currentParams": {"negativePrompt": "blurry", "width": 1280, "steps": 24},
        "activePreset": "Landscape",
    }
    image_preset = {
        "name": "Landscape",
        "params": image_state["currentParams"],
    }
    video_state = {
        "currentParams": {"negativePrompt": "flicker", "steps": 40},
        "activePreset": "Default",
    }

    assert client.put("/api/settings/generation-presets/image", json = image_state).status_code == 200
    assert (
        client.put("/api/settings/generation-presets/image/custom", json = image_preset).status_code
        == 200
    )
    assert client.put("/api/settings/generation-presets/video", json = video_state).status_code == 200
    image = client.get("/api/settings/generation-presets/image").json()
    video = client.get("/api/settings/generation-presets/video").json()
    assert image["saved"] is True
    assert image["currentParams"]["steps"] == 24
    assert image["customPresets"][0]["name"] == "Landscape"
    assert video["saved"] is True
    assert video["currentParams"]["steps"] == 40
    assert video["customPresets"] == []


def test_preset_payload_rejects_per_run_or_unknown_fields(monkeypatch):
    client = _client(monkeypatch)
    payload = {
        "currentParams": {"prompt": "must not be persisted"},
        "activePreset": "Default",
    }
    response = client.put("/api/settings/generation-presets/image", json = payload)
    assert response.status_code == 422


def test_preset_payload_bounds_names_and_values(monkeypatch):
    client = _client(monkeypatch)
    response = client.put(
        "/api/settings/generation-presets/video",
        json = {
            "currentParams": {"durationSeconds": 0},
            "activePreset": "Default",
        },
    )
    assert response.status_code == 422


def test_image_preset_payload_accepts_generation_control_limits(monkeypatch):
    client = _client(monkeypatch)
    response = client.put(
        "/api/settings/generation-presets/image",
        json = {
            "currentParams": {"batchSize": 32, "runs": 129},
            "activePreset": "Default",
        },
    )
    assert response.status_code == 200


def test_preset_negative_prompts_match_the_unbounded_generation_contract(monkeypatch):
    client = _client(monkeypatch)
    negative_prompt = "x" * 20_001
    for kind in ("image", "video"):
        response = client.put(
            f"/api/settings/generation-presets/{kind}",
            json = {
                "currentParams": {"negativePrompt": negative_prompt},
                "activePreset": "Default",
            },
        )
        assert response.status_code == 200


def test_state_writes_cannot_add_or_remove_named_presets(monkeypatch):
    client = _client(monkeypatch)
    params = {"steps": 9}
    settings = {
        "currentParams": params,
        "activePreset": "Default",
    }
    preset_x = {"name": "X", "params": params}

    assert client.put("/api/settings/generation-presets/image", json = settings).status_code == 200
    assert (
        client.put("/api/settings/generation-presets/image/custom", json = preset_x).status_code
        == 200
    )
    stale_snapshot = {**settings, "customPresets": []}
    assert (
        client.put("/api/settings/generation-presets/image", json = stale_snapshot).status_code == 422
    )
    saved = client.get("/api/settings/generation-presets/image").json()
    assert [preset["name"] for preset in saved["customPresets"]] == ["X"]


def test_custom_preset_schema_is_selected_from_the_path_kind(monkeypatch):
    client = _client(monkeypatch)
    assert (
        client.put(
            "/api/settings/generation-presets/video/custom",
            json = {"name": "Video", "params": {}},
        ).status_code
        == 200
    )
    assert (
        client.put(
            "/api/settings/generation-presets/image/custom",
            json = {"name": "Image", "params": {}},
        ).status_code
        == 200
    )
    video = client.get("/api/settings/generation-presets/video").json()
    image = client.get("/api/settings/generation-presets/image").json()
    assert video["customPresets"][0]["params"]["durationSeconds"] == 3
    assert image["customPresets"][0]["params"]["batchSize"] == 1


def test_custom_preset_names_are_canonical_and_cannot_replace_default(monkeypatch):
    client = _client(monkeypatch)
    assert (
        client.put(
            "/api/settings/generation-presets/image/custom",
            json = {"name": "  Landscape  ", "params": {}},
        ).status_code
        == 200
    )
    assert (
        client.put(
            "/api/settings/generation-presets/image/custom",
            json = {"name": "Default", "params": {}},
        ).status_code
        == 422
    )
    saved = client.get("/api/settings/generation-presets/image").json()
    assert [preset["name"] for preset in saved["customPresets"]] == ["Landscape"]


def test_the_preset_list_is_capped_and_says_why(monkeypatch):
    client = _client(monkeypatch)
    for index in range(100):
        assert (
            client.put(
                "/api/settings/generation-presets/image/custom",
                json = {"name": f"P{index}", "params": {}},
            ).status_code
            == 200
        )

    refused = client.put(
        "/api/settings/generation-presets/image/custom",
        json = {"name": "P100", "params": {}},
    )
    assert refused.status_code == 409
    assert refused.json()["detail"] == "Delete a preset before saving another one"

    # Overwriting one of the existing entries is not a new entry, so it still goes through.
    assert (
        client.put(
            "/api/settings/generation-presets/image/custom",
            json = {"name": "P0", "params": {"steps": 24}},
        ).status_code
        == 200
    )


def test_a_store_larger_than_the_write_cap_still_reads(monkeypatch):
    # The cap belongs to the write path, which answers 409. Refusing to report a store that
    # somehow holds more would turn the GET into a 500 and take the whole preset UI with it.
    client = _client(
        monkeypatch,
        {
            "image_generation_presets": {
                "activePreset": "Default",
                "currentParams": {"steps": 24},
                "customPresets": [
                    {"name": f"P{index}", "params": {"steps": 20}} for index in range(101)
                ],
            }
        },
    )

    body = client.get("/api/settings/generation-presets/image").json()

    assert len(body["customPresets"]) == 101
    assert body["currentParams"]["steps"] == 24


def test_unreadable_presets_do_not_consume_the_write_cap(monkeypatch):
    stored = {
        "image_generation_presets": {
            "activePreset": "Default",
            "currentParams": {"steps": 24},
            "customPresets": [
                {"name": f"Unreadable {index}", "params": {"steps": 999_999}}
                for index in range(100)
            ],
        }
    }
    client = _client(monkeypatch, stored)

    response = client.put(
        "/api/settings/generation-presets/image/custom",
        json = {"name": "Readable", "params": {"steps": 20}},
    )

    assert response.status_code == 200
    body = client.get("/api/settings/generation-presets/image").json()
    assert [preset["name"] for preset in body["customPresets"]] == ["Readable"]
    assert len(stored["image_generation_presets"]["customPresets"]) == 101


def test_unreadable_presets_do_not_expand_the_readable_write_cap(monkeypatch):
    stored = {
        "image_generation_presets": {
            "activePreset": "Default",
            "currentParams": {"steps": 24},
            "customPresets": [
                {"name": f"Readable {index}", "params": {"steps": 20}} for index in range(100)
            ]
            + [{"name": "Unreadable", "params": {"steps": 999_999}}],
        }
    }
    client = _client(monkeypatch, stored)

    response = client.put(
        "/api/settings/generation-presets/image/custom",
        json = {"name": "One too many", "params": {"steps": 20}},
    )

    assert response.status_code == 409


def test_a_blob_written_by_another_build_still_reads(monkeypatch):
    # extra = "forbid" is right for a submitted payload and wrong for reading storage back: a single
    # field from a newer build must not cost the user every recipe this build can still render.
    client = _client(
        monkeypatch,
        {
            "image_generation_presets": {
                "activePreset": "Landscape",
                "currentParams": {"steps": 24, "aFieldFromLater": 1},
                "somethingElseEntirely": True,
                "customPresets": [
                    {
                        "name": "Landscape",
                        "params": {"steps": 24, "aFieldFromLater": 1},
                        "alsoNew": "x",
                    }
                ],
            }
        },
    )

    body = client.get("/api/settings/generation-presets/image").json()

    assert body["saved"] is True
    assert body["activePreset"] == "Landscape"
    assert body["currentParams"]["steps"] == 24
    assert [preset["name"] for preset in body["customPresets"]] == ["Landscape"]
    assert body["customPresets"][0]["params"]["steps"] == 24


def test_one_unreadable_preset_does_not_discard_the_rest(monkeypatch):
    client = _client(
        monkeypatch,
        {
            "image_generation_presets": {
                "activePreset": "Broken",
                "currentParams": {"steps": 24},
                "customPresets": [
                    {"name": "Broken", "params": {"steps": 999_999}},
                    {"name": "Readable", "params": {"steps": 24}},
                ],
            }
        },
    )

    body = client.get("/api/settings/generation-presets/image").json()

    # Still "saved", so the client does not mistake this for a fresh install and overwrite it.
    assert body["saved"] is True
    assert [preset["name"] for preset in body["customPresets"]] == ["Readable"]
    # The state validated on its own, so one bad entry in the list does not reset it.
    assert body["activePreset"] == "Broken"
    assert body["currentParams"]["steps"] == 24


def test_a_store_holding_only_presets_is_not_a_saved_recipe(monkeypatch):
    # A named-preset write can land while the debounced state write does not. Reporting that as
    # saved hands the client schema defaults as if the user had chosen them, and the client stops
    # seeding the resident model's own defaults for as long as it believes that.
    client = _client(
        monkeypatch,
        {
            "image_generation_presets": {
                "customPresets": [{"name": "Landscape", "params": {"steps": 24}}],
            }
        },
    )

    body = client.get("/api/settings/generation-presets/image").json()

    assert body["saved"] is False
    # The library is still the user's; only the recipe falls back to the model's defaults.
    assert [preset["name"] for preset in body["customPresets"]] == ["Landscape"]


def test_a_preset_collection_that_is_not_a_list_reads_as_empty(monkeypatch):
    # Recovery exists so a store this build cannot represent still reads. Iterating a scalar here
    # would answer 500 and take the whole preset UI down with it.
    client = _client(
        monkeypatch,
        {
            "image_generation_presets": {
                "activePreset": "Default",
                "currentParams": {"steps": 24},
                "customPresets": 42,
            }
        },
    )

    body = client.get("/api/settings/generation-presets/image").json()

    assert body["saved"] is True
    assert body["customPresets"] == []
    assert body["currentParams"]["steps"] == 24


def test_one_unreadable_state_field_does_not_reset_the_recipe(monkeypatch):
    # The response is still "saved", so the client applies and autosaves it. Handing back schema
    # defaults for a recipe that read fine would overwrite the stored one.
    stored = {
        "image_generation_presets": {
            "activePreset": "X" * 200,
            "currentParams": {"steps": 24, "guidance": 3.5},
            "customPresets": [{"name": "Keep", "params": {"steps": 20}}],
        }
    }
    client = _client(monkeypatch, stored)

    body = client.get("/api/settings/generation-presets/image").json()

    assert body["activePreset"] == "Default"
    assert body["currentParams"]["steps"] == 24
    assert body["currentParams"]["guidance"] == 3.5
    assert [preset["name"] for preset in body["customPresets"]] == ["Keep"]

    echoed = {"activePreset": body["activePreset"], "currentParams": body["currentParams"]}
    assert client.put("/api/settings/generation-presets/image", json = echoed).status_code == 200
    assert stored["image_generation_presets"]["activePreset"] == "X" * 200

    echoed["activePreset"] = "Keep"
    assert client.put("/api/settings/generation-presets/image", json = echoed).status_code == 200
    assert stored["image_generation_presets"]["activePreset"] == "Keep"


@pytest.mark.parametrize(("kind", "recovered_guidance"), (("image", 0), ("video", 1)))
def test_one_unreadable_nested_recipe_field_does_not_reset_its_siblings(
    monkeypatch, kind, recovered_guidance
):
    stored = {
        f"{kind}_generation_presets": {
            "activePreset": "Keep",
            "currentParams": {"steps": 24, "guidance": 100},
            "customPresets": [{"name": "Keep", "params": {"steps": 24}}],
        }
    }
    client = _client(monkeypatch, stored)

    body = client.get(f"/api/settings/generation-presets/{kind}").json()

    assert body["activePreset"] == "Keep"
    assert body["currentParams"]["steps"] == 24
    assert body["currentParams"]["guidance"] == recovered_guidance
    assert [preset["name"] for preset in body["customPresets"]] == ["Keep"]

    # The client echoes the recovered representation after any recipe change. Its synthesized
    # guidance default must not replace the raw value merely because this older schema was opened.
    body["currentParams"]["steps"] = 30
    echoed = {"activePreset": body["activePreset"], "currentParams": body["currentParams"]}
    assert client.put(f"/api/settings/generation-presets/{kind}", json = echoed).status_code == 200
    kept = stored[f"{kind}_generation_presets"]["currentParams"]
    assert kept["steps"] == 30
    assert kept["guidance"] == 100

    # An actual edit to the recovered field still replaces the unreadable value.
    echoed["currentParams"]["guidance"] = 5
    assert client.put(f"/api/settings/generation-presets/{kind}", json = echoed).status_code == 200
    assert stored[f"{kind}_generation_presets"]["currentParams"]["guidance"] == 5


def test_preset_bounds_match_the_generation_request(monkeypatch):
    # A preset the generate endpoint would refuse is not usable: selecting it would make every
    # following Generate fail validation.
    client = _client(monkeypatch)
    for params in (
        {"steps": 500},
        {"guidance": 100},
        {"width": 8192},
        {"height": 64},
        {"width": 257},
        {"height": 257},
    ):
        assert (
            client.put(
                "/api/settings/generation-presets/image/custom",
                json = {"name": "Out of bounds", "params": params},
            ).status_code
            == 422
        ), params
    for params in ({"steps": 100}, {"guidance": 20}, {"width": 2048}, {"height": 256}):
        assert (
            client.put(
                "/api/settings/generation-presets/image/custom",
                json = {"name": "In bounds", "params": params},
            ).status_code
            == 200
        ), params
    assert (
        client.put(
            "/api/settings/generation-presets/video/custom",
            json = {"name": "Video bounds", "params": {"steps": 500}},
        ).status_code
        == 422
    )


def test_a_downgraded_read_does_not_erase_newer_stored_fields(monkeypatch):
    # The GET drops what this build cannot validate, so the state the client echoes back is lossy.
    # Opening the store with an older build must not cost the newer build its fields.
    stored = {
        "image_generation_presets": {
            "activePreset": "Default",
            "currentParams": {"steps": 24, "aFieldFromLater": 7},
            "somethingElseEntirely": {"nested": True},
            "customPresets": [],
        }
    }
    client = _client(monkeypatch, stored)

    body = client.get("/api/settings/generation-presets/image").json()
    assert "aFieldFromLater" not in body["currentParams"]

    # The client autosaves the representation it was given.
    echoed = {
        "activePreset": body["activePreset"],
        "currentParams": body["currentParams"],
    }
    assert client.put("/api/settings/generation-presets/image", json = echoed).status_code == 200

    kept = stored["image_generation_presets"]
    assert kept["currentParams"]["aFieldFromLater"] == 7
    assert kept["somethingElseEntirely"] == {"nested": True}
    assert kept["currentParams"]["steps"] == 24


def test_a_store_holding_load_options_is_read_without_them_and_keeps_them(monkeypatch):
    """Presets are a generation recipe; load options belong to the resident build.

    A store written before that split still holds them, so the read must ignore them and the
    write must not throw them away.
    """
    stored = {
        "image_generation_presets": {
            "activePreset": "Landscape",
            "currentParams": {"steps": 24},
            "currentLoadConfig": {"speedMode": "max"},
            "customPresets": [
                {
                    "name": "Landscape",
                    "params": {"steps": 24},
                    "loadConfig": {"memoryMode": "low_vram"},
                }
            ],
        }
    }
    client = _client(monkeypatch, stored)

    body = client.get("/api/settings/generation-presets/image").json()
    assert "currentLoadConfig" not in body
    assert "loadConfig" not in body["customPresets"][0]
    assert body["activePreset"] == "Landscape"
    assert body["currentParams"]["steps"] == 24

    assert (
        client.put(
            "/api/settings/generation-presets/image",
            json = {"activePreset": "Landscape", "currentParams": body["currentParams"]},
        ).status_code
        == 200
    )

    kept = stored["image_generation_presets"]
    assert kept["currentLoadConfig"] == {"speedMode": "max"}
    assert kept["customPresets"][0]["loadConfig"] == {"memoryMode": "low_vram"}

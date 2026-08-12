# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import FastAPI
from fastapi.testclient import TestClient

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
            "currentParams": {"batchSize": 32, "runs": 128},
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


def test_out_of_order_writes_keep_the_latest_browser_operation(monkeypatch):
    client = _client(monkeypatch)
    settings = {
        "currentParams": {"steps": 9},
        "activePreset": "Default",
    }

    def ordered(timestamp: int, writer: str) -> dict[str, str]:
        return {
            "Preset-Timestamp": str(timestamp),
            "Preset-Writer": writer,
        }

    assert (
        client.put(
            "/api/settings/generation-presets/image",
            json = {**settings, "currentParams": {"steps": 24}},
            headers = ordered(200, "newer-tab"),
        ).status_code
        == 200
    )
    stale = client.put(
        "/api/settings/generation-presets/image",
        json = settings,
        headers = ordered(100, "older-tab"),
    )
    assert stale.status_code == 200
    assert stale.json() == {"saved": False}
    saved = client.get("/api/settings/generation-presets/image").json()
    assert saved["currentParams"]["steps"] == 24

    preset = {"name": "Transient", "params": {}}
    assert (
        client.put(
            "/api/settings/generation-presets/image/custom",
            json = preset,
            headers = ordered(250, "older-tab"),
        ).status_code
        == 200
    )
    assert (
        client.delete(
            "/api/settings/generation-presets/image/custom",
            params = {"name": "Transient"},
            headers = ordered(400, "newer-tab"),
        ).status_code
        == 200
    )
    assert (
        client.put(
            "/api/settings/generation-presets/image/custom",
            json = preset,
            headers = ordered(300, "older-tab"),
        ).status_code
        == 409
    )
    saved = client.get("/api/settings/generation-presets/image").json()
    assert saved["customPresets"] == []


def test_rejected_capacity_write_does_not_consume_its_version(monkeypatch):
    client = _client(monkeypatch)

    def ordered(timestamp: int) -> dict[str, str]:
        return {
            "Preset-Timestamp": str(timestamp),
            "Preset-Writer": "capacity-test",
        }

    for index in range(100):
        response = client.put(
            "/api/settings/generation-presets/image/custom",
            json = {"name": f"Preset {index}", "params": {}},
        )
        assert response.status_code == 200

    new_preset = {"name": "New", "params": {}}
    assert (
        client.put(
            "/api/settings/generation-presets/image/custom",
            json = new_preset,
            headers = ordered(500),
        ).status_code
        == 409
    )
    assert (
        client.delete(
            "/api/settings/generation-presets/image/custom",
            params = {"name": "Preset 0"},
            headers = ordered(400),
        ).status_code
        == 200
    )
    assert (
        client.put(
            "/api/settings/generation-presets/image/custom",
            json = new_preset,
            headers = ordered(500),
        ).status_code
        == 200
    )


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
                {"name": "Landscape", "params": {"steps": 24}, "loadConfig": {"memoryMode": "low_vram"}}
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

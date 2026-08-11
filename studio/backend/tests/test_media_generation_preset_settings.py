# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes import settings as settings_routes
from storage import studio_db


def _client(monkeypatch) -> TestClient:
    stored = {}
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
    image = {
        "currentParams": {
            "negativePrompt": "blurry",
            "width": 1280,
            "height": 720,
            "steps": 24,
            "guidance": 3.5,
            "batchSize": 2,
            "runs": 3,
        },
        "currentLoadConfig": {
            "speedMode": "max",
            "transformerQuant": "fp8",
            "attentionBackend": "flash3",
            "memoryMode": "balanced",
            "transformerCache": "fbcache",
            "cpuOffload": True,
        },
        "customPresets": [
            {
                "name": "Landscape",
                "params": {
                    "negativePrompt": "blurry",
                    "width": 1280,
                    "height": 720,
                    "steps": 24,
                    "guidance": 3.5,
                    "batchSize": 2,
                    "runs": 3,
                },
                "loadConfig": {
                    "speedMode": "max",
                    "transformerQuant": "fp8",
                    "attentionBackend": "flash3",
                    "memoryMode": "balanced",
                    "transformerCache": "fbcache",
                    "cpuOffload": True,
                },
            }
        ],
        "activePreset": "Landscape",
        "activePresetSource": "custom",
    }
    video = {
        "currentParams": {
            "negativePrompt": "flicker",
            "width": 1216,
            "height": 704,
            "durationSeconds": 4.9,
            "steps": 40,
            "guidance": 4,
            "flowShift": 12,
            "audioFlowShift": 3,
        },
        "currentLoadConfig": {
            "memoryMode": "low_vram",
            "speedMode": "off",
            "attentionBackend": "native",
            "transformerCache": "off",
            "transformerQuant": "none",
        },
        "customPresets": [],
        "activePreset": "Default",
        "activePresetSource": "modified",
    }

    image_preset = image["customPresets"][0]
    image_without_presets = {**image, "customPresets": []}
    assert (
        client.put("/api/settings/generation-presets/image", json = image_without_presets).status_code
        == 200
    )
    assert (
        client.put("/api/settings/generation-presets/image/custom", json = image_preset).status_code
        == 200
    )
    assert client.put("/api/settings/generation-presets/video", json = video).status_code == 200
    assert client.get("/api/settings/generation-presets/image").json() == {
        **image,
        "saved": True,
    }
    assert client.get("/api/settings/generation-presets/video").json() == {
        **video,
        "saved": True,
    }


def test_preset_payload_rejects_per_run_or_unknown_fields(monkeypatch):
    client = _client(monkeypatch)
    payload = {
        "currentParams": {
            "negativePrompt": "",
            "width": 1024,
            "height": 1024,
            "steps": 9,
            "guidance": 0,
            "batchSize": 1,
            "runs": 1,
            "prompt": "must not be persisted",
        },
        "customPresets": [],
        "activePreset": "Default",
        "activePresetSource": "builtin-default",
    }
    response = client.put("/api/settings/generation-presets/image", json = payload)
    assert response.status_code == 422


def test_preset_payload_bounds_names_and_values(monkeypatch):
    client = _client(monkeypatch)
    response = client.put(
        "/api/settings/generation-presets/video",
        json = {
            "currentParams": {
                "negativePrompt": "",
                "width": 768,
                "height": 512,
                "durationSeconds": 0,
                "steps": 8,
                "guidance": 1,
            },
            "customPresets": [],
            "activePreset": "Default",
            "activePresetSource": "builtin-default",
        },
    )
    assert response.status_code == 422


def test_image_preset_payload_accepts_generation_control_limits(monkeypatch):
    client = _client(monkeypatch)
    response = client.put(
        "/api/settings/generation-presets/image",
        json = {
            "currentParams": {
                "negativePrompt": "",
                "width": 1024,
                "height": 1024,
                "steps": 9,
                "guidance": 0,
                "batchSize": 32,
                "runs": 128,
            },
            "customPresets": [],
            "activePreset": "Default",
            "activePresetSource": "modified",
        },
    )
    assert response.status_code == 200


def test_stale_settings_writes_cannot_add_or_remove_named_presets(monkeypatch):
    client = _client(monkeypatch)
    params = {
        "negativePrompt": "",
        "width": 1024,
        "height": 1024,
        "steps": 9,
        "guidance": 0,
        "batchSize": 1,
        "runs": 1,
    }
    settings = {
        "currentParams": params,
        "customPresets": [],
        "activePreset": "Default",
        "activePresetSource": "builtin-default",
    }
    preset_x = {"name": "X", "params": params}
    preset_y = {"name": "Y", "params": {**params, "steps": 24}}

    assert client.put("/api/settings/generation-presets/image", json = settings).status_code == 200
    assert client.put("/api/settings/generation-presets/image/custom", json = preset_x).status_code == 200
    stale = {**settings, "customPresets": [preset_x]}
    assert client.put("/api/settings/generation-presets/image/custom", json = preset_y).status_code == 200
    assert client.put("/api/settings/generation-presets/image", json = stale).status_code == 200
    saved = client.get("/api/settings/generation-presets/image").json()
    assert {preset["name"] for preset in saved["customPresets"]} == {"X", "Y"}

    assert client.delete("/api/settings/generation-presets/image/custom", params = {"name": "X"}).status_code == 200
    assert client.put("/api/settings/generation-presets/image", json = stale).status_code == 200
    saved = client.get("/api/settings/generation-presets/image").json()
    assert [preset["name"] for preset in saved["customPresets"]] == ["Y"]


def test_out_of_order_writes_keep_the_latest_browser_operation(monkeypatch):
    client = _client(monkeypatch)
    settings = {
        "currentParams": {"steps": 9},
        "customPresets": [],
        "activePreset": "Default",
        "activePresetSource": "builtin-default",
    }
    def ordered(sequence: int) -> dict[str, str]:
        return {
            "Preset-Writer": "out-of-order-test",
            "Preset-Sequence": str(sequence),
        }

    assert client.put(
        "/api/settings/generation-presets/image",
        json = {**settings, "currentParams": {"steps": 24}},
        headers = ordered(2),
    ).status_code == 200
    assert client.put(
        "/api/settings/generation-presets/image",
        json = settings,
        headers = ordered(1),
    ).status_code == 200
    saved = client.get("/api/settings/generation-presets/image").json()
    assert saved["currentParams"]["steps"] == 24

    assert client.delete(
        "/api/settings/generation-presets/image/custom",
        params = {"name": "Transient"},
        headers = ordered(4),
    ).status_code == 200
    assert client.put(
        "/api/settings/generation-presets/image/custom",
        json = {"name": "Transient", "params": {}},
        headers = ordered(3),
    ).status_code == 200
    saved = client.get("/api/settings/generation-presets/image").json()
    assert saved["customPresets"] == []

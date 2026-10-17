# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from hub.services.models import account_access
from models.inference import AudioCatalogResponse
from utils.api_errors import install_api_error_handlers

UMBRELLA = "audio-cpp/audio.cpp-gguf"
KOKORO = {"id": f"{UMBRELLA}/Kokoro-82M-GGUF", "repo_id": UMBRELLA, "folder": "Kokoro-82M-GGUF"}
KOKORO.update(source = "umbrella", display_name = "Kokoro 82M", task = "tts", workflows = ["speak"])
DRAMA = {"id": "Jmica/Drama-3-GGUF", "repo_id": "Jmica/Drama-3-GGUF", "source": "community"}
DRAMA.update(display_name = "Drama 3", task = "tts", workflows = ["speak", "clone"])
CATALOG = AudioCatalogResponse(
    source = "hub", fetched_at = 1760720000.0, runtime_specs = True, models = [KOKORO, DRAMA]
)


@pytest.fixture
def client():
    from routes import inference

    app = FastAPI()
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.include_router(inference.router, prefix = "/api/inference")
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(inference.router, prefix = "/v1")
    yield TestClient(app)


def _builder(monkeypatch, build):
    # The route imports the builder at call time, so this stands in for the module.
    module = SimpleNamespace(audio_catalog_payload = build)
    monkeypatch.setitem(sys.modules, "core.inference.audio_cpp_catalog", module)


def test_the_catalog_is_served_on_the_studio_router_only(client, monkeypatch):
    _builder(monkeypatch, lambda: CATALOG)
    response = client.get("/api/inference/audio/catalog")
    assert response.status_code == 200
    body = AudioCatalogResponse.model_validate(response.json())
    assert body == CATALOG
    assert not any("path" in row for row in response.json()["models"])
    assert client.get("/v1/audio/catalog").status_code == 404


def test_a_managed_account_sees_only_the_rows_it_may_use(client, monkeypatch):
    _builder(monkeypatch, lambda: CATALOG)
    seen = []
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "model_grants", lambda: set())
    monkeypatch.setattr(account_access, "_warm_public_repos", lambda *a: None)
    monkeypatch.setattr(
        account_access,
        "model_visible",
        lambda reference, **_k: seen.append(reference) or reference != DRAMA["repo_id"],
    )
    response = client.get("/api/inference/audio/catalog")
    assert [row["id"] for row in response.json()["models"]] == [KOKORO["id"]]
    assert seen == [KOKORO["repo_id"], DRAMA["repo_id"]]
    # The cached payload stays whole for the next account.
    assert len(CATALOG.models) == 2


def test_a_builder_failure_answers_503_without_its_text(client, monkeypatch):
    def build():
        raise RuntimeError("/home/me/.cache/huggingface exploded")

    _builder(monkeypatch, build)
    response = client.get("/api/inference/audio/catalog")
    assert response.status_code == 503
    assert "Unsloth Studio could not list audio models right now." in response.text
    assert "exploded" not in response.text and "/home/me" not in response.text

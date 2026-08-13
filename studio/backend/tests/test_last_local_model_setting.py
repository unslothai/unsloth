# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level tests for the last-local-model setting endpoints."""

from pathlib import Path
import sys
import types as _types


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
import storage.studio_db as studio_db


@pytest.fixture
def client(monkeypatch):
    store: dict = {}

    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: store.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda values: store.update(values) or store
    )

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False), store


def test_get_with_nothing_stored(client):
    c, _ = client
    r = c.get("/last-local-model")
    assert r.status_code == 200
    assert r.json() == {"id": None, "kind": None, "gguf_variant": None, "loaded_at": None}


def test_put_then_get_round_trips(client):
    c, store = client
    payload = {"id": "unsloth/gemma-4-E2B-it-GGUF", "kind": "gguf", "gguf_variant": "UD-Q4_K_XL"}
    r = c.put("/last-local-model", json = payload)
    assert r.status_code == 200
    assert r.json() == {**payload, "loaded_at": None}
    assert store[settings.LAST_LOCAL_MODEL_SETTING_KEY] == {**payload, "loaded_at": None}

    r = c.get("/last-local-model")
    assert r.status_code == 200
    assert r.json() == {**payload, "loaded_at": None}


def test_put_accepts_path_qualified_variant(client):
    c, _ = client
    variant = "quants/" + "a" * 80 + "/model-UD-Q4_K_XL-00001-of-00002.gguf"
    payload = {"id": "unsloth/gemma-4-E2B-it-GGUF", "kind": "gguf", "gguf_variant": variant}
    r = c.put("/last-local-model", json = payload)
    assert r.status_code == 200
    assert r.json()["gguf_variant"] == variant


def test_put_round_trips_loaded_at(client):
    c, _ = client
    payload = {"id": "unsloth/Qwen3-4B", "kind": "model", "loaded_at": 1765432100000}
    r = c.put("/last-local-model", json = payload)
    assert r.status_code == 200
    assert r.json()["loaded_at"] == 1765432100000
    assert c.get("/last-local-model").json()["loaded_at"] == 1765432100000


def test_put_without_variant(client):
    c, _ = client
    r = c.put("/last-local-model", json = {"id": "unsloth/Qwen3-4B", "kind": "model"})
    assert r.status_code == 200
    assert r.json() == {
        "id": "unsloth/Qwen3-4B",
        "kind": "model",
        "gguf_variant": None,
        "loaded_at": None,
    }


def test_put_ignores_stale_timestamped_write(client):
    c, _ = client
    newer = {"id": "unsloth/Qwen3-4B", "kind": "model", "loaded_at": 2000}
    stale = {
        "id": "unsloth/gemma-4-E2B-it-GGUF",
        "kind": "gguf",
        "gguf_variant": "UD-Q4_K_XL",
        "loaded_at": 1000,
    }
    assert c.put("/last-local-model", json = newer).status_code == 200
    r = c.put("/last-local-model", json = stale)
    assert r.status_code == 200
    assert r.json()["id"] == "unsloth/Qwen3-4B"
    assert c.get("/last-local-model").json()["loaded_at"] == 2000


def test_put_unstamped_write_keeps_last_write_wins(client):
    c, _ = client
    stamped = {"id": "unsloth/Qwen3-4B", "kind": "model", "loaded_at": 2000}
    assert c.put("/last-local-model", json = stamped).status_code == 200
    assert (
        c.put("/last-local-model", json = {"id": "unsloth/OLMo-4-13B", "kind": "model"}).status_code
        == 200
    )
    assert c.get("/last-local-model").json()["id"] == "unsloth/OLMo-4-13B"


def test_put_rejects_bad_payloads(client):
    c, _ = client
    assert c.put("/last-local-model", json = {"id": "x", "kind": "lora"}).status_code == 422
    assert c.put("/last-local-model", json = {"id": "", "kind": "gguf"}).status_code == 422
    assert c.put("/last-local-model", json = {"kind": "gguf"}).status_code == 422


def test_get_tolerates_corrupt_stored_value(client):
    c, store = client
    store[settings.LAST_LOCAL_MODEL_SETTING_KEY] = {"id": "x", "kind": "lora"}
    r = c.get("/last-local-model")
    assert r.status_code == 200
    assert r.json() == {"id": None, "kind": None, "gguf_variant": None, "loaded_at": None}

    store[settings.LAST_LOCAL_MODEL_SETTING_KEY] = "not-a-dict"
    r = c.get("/last-local-model")
    assert r.status_code == 200
    assert r.json() == {"id": None, "kind": None, "gguf_variant": None, "loaded_at": None}

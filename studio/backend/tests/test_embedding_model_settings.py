# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the customizable RAG embedding model: settings module, effective
config resolution (incl. GGUF repo derivation), and the settings routes."""

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

import utils.embedding_model_settings as ems
from core.rag import config as rag_config


@pytest.fixture
def settings_store(monkeypatch):
    """In-memory app_settings store patched under the module's lazy imports."""
    import storage.studio_db as studio_db

    store: dict = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: store.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda settings: store.update(settings) or store
    )
    ems._invalidate_cache()
    yield store
    ems._invalidate_cache()


def test_default_when_nothing_stored(settings_store):
    assert ems.get_stored_embedding_model() is None
    assert ems.get_rag_embedding_model() == rag_config.EMBEDDING_MODEL


def test_set_get_roundtrip(settings_store):
    assert ems.set_rag_embedding_model("  org/my-embedder  ") == "org/my-embedder"
    assert ems.get_stored_embedding_model() == "org/my-embedder"
    assert ems.get_rag_embedding_model() == "org/my-embedder"


def test_reset_clears_override(settings_store):
    ems.set_rag_embedding_model("org/my-embedder")
    assert ems.reset_rag_embedding_model() == rag_config.EMBEDDING_MODEL
    assert ems.get_stored_embedding_model() is None
    assert ems.get_rag_embedding_model() == rag_config.EMBEDDING_MODEL


@pytest.mark.parametrize(
    "bad", ["", "   ", "a\nb", "x" * (ems.MAX_EMBEDDING_MODEL_LENGTH + 1), 7, None, True]
)
def test_validation_rejects_bad_values(settings_store, bad):
    with pytest.raises(ValueError):
        ems.set_rag_embedding_model(bad)


def test_invalid_stored_value_falls_back_to_default(settings_store):
    settings_store[ems.EMBEDDING_MODEL_SETTING_KEY] = 123  # corrupt/legacy value
    assert ems.get_rag_embedding_model() == rag_config.EMBEDDING_MODEL


def test_cache_invalidated_on_write(settings_store):
    assert ems.get_rag_embedding_model() == rag_config.EMBEDDING_MODEL  # warm cache
    ems.set_rag_embedding_model("org/my-embedder")
    assert ems.get_rag_embedding_model() == "org/my-embedder"


def test_effective_embedding_model_prefers_override(settings_store):
    assert rag_config.effective_embedding_model() == rag_config.EMBEDDING_MODEL
    ems.set_rag_embedding_model("org/my-embedder")
    assert rag_config.effective_embedding_model() == "org/my-embedder"


def test_effective_gguf_repo_default_pairing(settings_store, monkeypatch):
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    assert rag_config.effective_gguf_repo() == rag_config.EMBED_GGUF_REPO


def test_effective_gguf_repo_derives_companion(settings_store, monkeypatch):
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    ems.set_rag_embedding_model("org/my-embedder")
    assert rag_config.effective_gguf_repo() == "org/my-embedder-GGUF"


def test_effective_gguf_repo_uses_gguf_repo_as_is(settings_store, monkeypatch):
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    ems.set_rag_embedding_model("org/my-embedder-GGUF")
    assert rag_config.effective_gguf_repo() == "org/my-embedder-GGUF"


def test_effective_gguf_repo_env_wins(settings_store, monkeypatch):
    monkeypatch.setenv("RAG_EMBED_GGUF_REPO", "org/pinned-GGUF")
    ems.set_rag_embedding_model("org/my-embedder")
    # Explicit env pin keeps the import-time repo regardless of the override.
    assert rag_config.effective_gguf_repo() == rag_config.EMBED_GGUF_REPO


# ---------------------------------------------------------------------------
# Route-level tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client(settings_store, monkeypatch):
    import routes.settings as settings_routes

    # Verification is stubbed per-test via calls["is_embedding"].
    calls: dict = {"is_embedding": True}

    def _is_embedding_model(model, hf_token = None):
        calls["checked"] = model
        calls["token"] = hf_token
        return calls["is_embedding"]

    import utils.models as models_utils

    monkeypatch.setattr(models_utils, "is_embedding_model", _is_embedding_model)

    app = FastAPI()
    app.include_router(settings_routes.router)
    app.dependency_overrides[settings_routes.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False), calls


def test_get_embedding_model(client):
    c, _ = client
    r = c.get("/embedding-model")
    assert r.status_code == 200
    body = r.json()
    assert body["embedding_model"] == rag_config.EMBEDDING_MODEL
    assert body["default_embedding_model"] == rag_config.EMBEDDING_MODEL
    assert body["is_custom"] is False


def test_put_embedding_model_verified(client):
    c, calls = client
    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "org/my-embedder", "hf_token": "hf_abc"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["embedding_model"] == "org/my-embedder"
    assert body["is_custom"] is True
    assert calls["checked"] == "org/my-embedder"
    assert calls["token"] == "hf_abc"


def test_put_embedding_model_unverified_409(client):
    c, calls = client
    calls["is_embedding"] = False
    r = c.put("/embedding-model", json = {"embedding_model": "org/not-an-embedder"})
    assert r.status_code == 409
    # Not saved.
    assert c.get("/embedding-model").json()["is_custom"] is False


def test_put_embedding_model_force_bypasses_verification(client):
    c, calls = client
    calls["is_embedding"] = False
    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "org/offline-model", "force": True},
    )
    assert r.status_code == 200
    assert r.json()["embedding_model"] == "org/offline-model"
    assert "checked" not in calls  # verification skipped entirely


def test_put_default_model_skips_verification(client):
    c, calls = client
    calls["is_embedding"] = False  # would 409 if verification ran
    r = c.put("/embedding-model", json = {"embedding_model": rag_config.EMBEDDING_MODEL})
    assert r.status_code == 200
    assert "checked" not in calls


def test_put_embedding_model_rejects_blank(client):
    c, _ = client
    assert c.put("/embedding-model", json = {"embedding_model": ""}).status_code == 422
    assert c.put("/embedding-model", json = {"embedding_model": "   "}).status_code == 400


def test_delete_resets_to_default(client):
    c, _ = client
    c.put("/embedding-model", json = {"embedding_model": "org/my-embedder"})
    r = c.delete("/embedding-model")
    assert r.status_code == 200
    body = r.json()
    assert body["embedding_model"] == rag_config.EMBEDDING_MODEL
    assert body["is_custom"] is False

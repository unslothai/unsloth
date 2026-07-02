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


def test_saving_default_is_not_an_override(settings_store):
    ems.set_rag_embedding_model(rag_config.EMBEDDING_MODEL)
    assert ems.get_stored_embedding_model() is None
    assert ems.get_rag_embedding_model() == rag_config.EMBEDDING_MODEL


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


@pytest.mark.parametrize(
    "override, expected",
    [
        (None, rag_config.EMBED_GGUF_REPO),  # no override -> default pairing
        ("org/my-embedder", "org/my-embedder-GGUF"),  # derive companion
        ("org/my-embedder-GGUF", "org/my-embedder-GGUF"),  # already a GGUF repo
        ("org/bigguf-model", "org/bigguf-model-GGUF"),  # "gguf" only as a segment
        ("org/model.GGUF", "org/model.GGUF"),  # dot-separated GGUF segment
    ],
)
def test_effective_gguf_repo_derivation(settings_store, monkeypatch, override, expected):
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    if override is not None:
        ems.set_rag_embedding_model(override)
    assert rag_config.effective_gguf_repo() == expected


def test_effective_gguf_repo_env_wins(settings_store, monkeypatch):
    monkeypatch.setenv("RAG_EMBED_GGUF_REPO", "org/pinned-GGUF")
    ems.set_rag_embedding_model("org/my-embedder")
    # Explicit env pin keeps the import-time repo regardless of the override.
    assert rag_config.effective_gguf_repo() == rag_config.EMBED_GGUF_REPO


def test_env_custom_model_derives_companion(settings_store, monkeypatch):
    """RAG_EMBEDDING_MODEL set to a custom model without RAG_EMBED_GGUF_REPO:
    the llama backend derives <custom>-GGUF instead of staying on the default."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    monkeypatch.setattr(rag_config, "EMBEDDING_MODEL", "org/env-custom")
    assert rag_config.effective_gguf_repo() == "org/env-custom-GGUF"


# ---------------------------------------------------------------------------
# Route-level tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client(settings_store, monkeypatch):
    import core.rag.config as core_rag_config
    import routes.settings as settings_routes

    # Route tests are backend-neutral by default; llama-backend specifics
    # (local GGUF saves, companion-repo availability) pin it explicitly.
    monkeypatch.setattr(core_rag_config, "EMBED_BACKEND", "st")

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


def test_put_embedding_model_trims_hf_token(client):
    c, calls = client
    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "org/my-embedder", "hf_token": "  hf_abc  "},
    )
    assert r.status_code == 200
    assert calls["token"] == "hf_abc"
    # Whitespace-only tokens degrade to anonymous access, not a broken client.
    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "org/other-embedder", "hf_token": "   "},
    )
    assert r.status_code == 200
    assert calls["token"] is None


def test_put_local_dir_without_gguf_on_llama_backend_409(client, monkeypatch, tmp_path):
    """A sentence-transformers-only local folder verifies as an embedding model
    but cannot run on the llama-server backend; the save must warn via 409."""
    import core.rag.config as core_rag_config

    monkeypatch.setattr(core_rag_config, "EMBED_BACKEND", "llama-server")
    local = tmp_path / "st-only"
    local.mkdir()
    (local / "modules.json").write_text("{}")
    c, _ = client
    r = c.put("/embedding-model", json = {"embedding_model": str(local)})
    assert r.status_code == 409
    assert "gguf" in r.json()["detail"].lower()
    # force saves anyway (user may point env overrides at it later).
    r = c.put("/embedding-model", json = {"embedding_model": str(local), "force": True})
    assert r.status_code == 200


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
    # Saving the default does not flip the setting to custom.
    assert r.json()["is_custom"] is False


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


def test_put_local_gguf_saves_without_verification(client, monkeypatch, tmp_path):
    """A local GGUF file/dir is exactly what the llama-server backend loads, so
    saving it must not require HF verification or Save anyway."""
    import core.rag.config as core_rag_config

    monkeypatch.setattr(core_rag_config, "EMBED_BACKEND", "llama-server")
    local = tmp_path / "gguf-dir"
    local.mkdir()
    (local / "embedder-F16.gguf").write_bytes(b"GGUF")
    c, calls = client
    calls["is_embedding"] = False  # would 409 if the HF gate ran
    r = c.put("/embedding-model", json = {"embedding_model": str(local)})
    assert r.status_code == 200
    assert "checked" not in calls
    assert r.json()["is_custom"] is True


def test_put_hf_repo_without_gguf_on_llama_backend_409(client, monkeypatch):
    """On the llama-server backend an HF repo with no GGUF anywhere fails at
    first index; the save must warn via 409 (force still bypasses)."""
    import core.rag.config as core_rag_config
    import routes.settings as settings_routes

    monkeypatch.setattr(core_rag_config, "EMBED_BACKEND", "llama-server")
    listed: list[str] = []

    def _no_gguf(model, hf_token):
        listed.append(model)
        return "no GGUF weights found (stub)"

    monkeypatch.setattr(settings_routes, "_hf_gguf_backend_error", _no_gguf)
    c, _ = client
    r = c.put("/embedding-model", json = {"embedding_model": "org/st-only"})
    assert r.status_code == 409
    assert listed == ["org/st-only"]
    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "org/st-only", "force": True},
    )
    assert r.status_code == 200


def test_hf_gguf_backend_error_checks_candidates(client, monkeypatch):
    """The availability probe accepts a .gguf in the companion repo or the repo
    itself, and reports both when neither has one."""
    import sys
    import types

    import core.rag.config as core_rag_config
    import routes.settings as settings_routes

    monkeypatch.setattr(core_rag_config, "EMBED_BACKEND", "llama-server")
    repos = {
        "org/has-companion-GGUF": ["embedder-F16.gguf"],
        "org/self-hosted": ["weights.gguf"],
        "org/st-only": ["model.safetensors"],
        "org/st-only-GGUF": None,  # missing repo
    }

    def _list_repo_files(repo_id, token = None):
        files = repos.get(repo_id)
        if files is None:
            raise RuntimeError("404")
        return files

    hub = types.SimpleNamespace(list_repo_files = _list_repo_files)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert settings_routes._hf_gguf_backend_error("org/has-companion", None) is None
    # st-only-GGUF is missing and self-hosted has the .gguf in the repo itself.
    repos["org/self-hosted-GGUF"] = None
    assert settings_routes._hf_gguf_backend_error("org/self-hosted", None) is None
    err = settings_routes._hf_gguf_backend_error("org/st-only", None)
    assert err is not None and "org/st-only-GGUF" in err


def test_stored_value_survives_transient_read_error(settings_store, monkeypatch):
    """A settings-store hiccup must not flip the hot path to the default model
    mid-ingestion; the last known override wins until the store recovers."""
    ems.set_rag_embedding_model("org/my-embedder")
    assert ems.get_rag_embedding_model() == "org/my-embedder"

    import storage.studio_db as studio_db

    def _boom(key, default = None):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(studio_db, "get_app_setting", _boom)
    # Expire the TTL entry (keeping its value) so the next read hits the
    # broken store and must fall back to the last known override.
    ems._cached = (ems.time.monotonic() - 10.0, "org/my-embedder")
    assert ems.get_rag_embedding_model() == "org/my-embedder"
    # Sticky across repeated failures until the store recovers.
    ems._cached = (ems.time.monotonic() - 10.0, "org/my-embedder")
    assert ems.get_rag_embedding_model() == "org/my-embedder"

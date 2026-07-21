# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test for the customizable RAG embedding model: a saved override becomes the
effective model and derives its GGUF companion for the llama-server backend."""

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

import utils.embedding_model_settings as ems
from core.rag import config as rag_config


@pytest.fixture
def settings_store(monkeypatch):
    """In-memory app_settings store patched under the module's lazy imports."""
    import storage.studio_db as studio_db

    store: dict = {}
    monkeypatch.setattr(
        studio_db,
        "get_app_setting",
        lambda key, fallback = None: store.get(key, fallback),
    )
    monkeypatch.setattr(
        studio_db,
        "upsert_app_settings",
        lambda settings: store.update(settings) or store,
    )
    ems._invalidate_cache()
    yield store
    ems._invalidate_cache()


def test_custom_model_overrides_default_and_derives_gguf(settings_store, monkeypatch):
    """The core contract: with nothing stored the default is in effect; a saved
    custom model becomes the effective embedding model and derives its -GGUF
    companion (what the llama-server backend loads); reset clears the override."""
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    assert ems.get_rag_embedding_model() == rag_config.EMBEDDING_MODEL
    assert rag_config.effective_gguf_repo() == rag_config.EMBED_GGUF_REPO

    assert ems.set_rag_embedding_model("  org/my-embedder  ") == "org/my-embedder"
    assert rag_config.effective_embedding_model() == "org/my-embedder"
    assert rag_config.effective_gguf_repo() == "org/my-embedder-GGUF"

    assert ems.reset_rag_embedding_model() == rag_config.EMBEDDING_MODEL
    assert ems.get_stored_embedding_model() is None


def test_env_default_derives_its_gguf_companion(monkeypatch):
    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    monkeypatch.setattr(rag_config, "EMBEDDING_MODEL", "org/env-default-embedder")

    assert rag_config.default_gguf_repo() == "org/env-default-embedder-GGUF"

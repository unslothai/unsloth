# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The picker asks what saving a model would need fetched before it saves, so the
download can be offered up front instead of happening invisibly at the first index.

The endpoint must agree with the loader about which repo and which file that is,
and with the PUT about what is unusable here."""

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


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: False)
    monkeypatch.setattr(settings, "_local_gguf_backend_error", lambda m: None)

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False)


def _resolve(c, model = "unsloth/bge-small-en-v1.5"):
    return c.get("/embedding-model/resolve", params = {"model": model})


def test_sentence_transformers_backend_points_at_the_model_repo(client, monkeypatch):
    """No GGUF involved: ST loads the model repo itself, and the cache check is the
    ordinary snapshot probe."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: False)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: True)

    body = _resolve(client).json()
    assert body["backend"] == "sentence-transformers"
    assert body["download_repo"] == "unsloth/bge-small-en-v1.5"
    assert body["cached"] is True
    assert body["files"] is None
    assert body["error"] is None


def test_uncached_gguf_names_the_one_file_the_loader_would_open(client, monkeypatch):
    """The companion repo carries every quant; only the variant the embedder opens
    should be fetched, so the picker gets a file list rather than a whole repo."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: True)
    monkeypatch.setattr(settings, "_cached_embedding_gguf", lambda candidates: False)
    monkeypatch.setattr(
        settings,
        "_remote_embedding_gguf_plan",
        lambda candidates, token: (candidates[0], "bge-small-en-v1.5-F16.gguf"),
    )
    monkeypatch.setattr(settings, "_hf_files_size", lambda repo, files, token: 133_000_000)

    body = _resolve(client).json()
    assert body["backend"] == "llama"
    assert body["download_repo"] == "unsloth/bge-small-en-v1.5-GGUF"
    assert body["files"] == ["bge-small-en-v1.5-F16.gguf"]
    assert body["cached"] is False
    assert body["size_bytes"] == 133_000_000
    assert body["error"] is None


def test_cached_gguf_asks_for_no_download(client, monkeypatch):
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: True)
    monkeypatch.setattr(settings, "_cached_embedding_gguf", lambda candidates: True)

    def _unreachable(*args, **kwargs):  # pragma: no cover - the point of the test
        raise AssertionError("a cached model must not be listed against the hub")

    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", _unreachable)

    body = _resolve(client).json()
    assert body["cached"] is True
    assert body["files"] is None


def test_the_search_fallback_finds_an_off_convention_name(client, monkeypatch):
    """The companion may not be named "<model>-GGUF" at all, so the owner's repos
    are searched before giving up."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: True)
    monkeypatch.setattr(settings, "_cached_embedding_gguf", lambda candidates: False)
    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", lambda candidates, token: None)
    monkeypatch.setattr(
        settings,
        "_search_hub_for_gguf",
        lambda m, token: ("unsloth/embeddinggemma-300m-GGUF", "embeddinggemma-300M-F16.gguf"),
    )
    monkeypatch.setattr(settings, "_hf_files_size", lambda repo, files, token: None)

    body = _resolve(client, "unsloth/embeddinggemma-300m-qat-q8_0-unquantized").json()
    assert body["download_repo"] == "unsloth/embeddinggemma-300m-GGUF"
    assert body["files"] == ["embeddinggemma-300M-F16.gguf"]
    assert body["error"] is None


def test_the_search_never_leaves_the_model_owner(monkeypatch):
    """Picking unsloth/X must download unsloth's own weights. A repo name is not
    proof of provenance, so a third party's "X-GGUF" is not an acceptable source."""
    seen: dict = {}

    class _Hit:
        def __init__(self, repo_id):
            self.id = repo_id

    class _Api:
        def list_models(self, **kwargs):
            seen.update(kwargs)
            # The Hub can return neighbours; only the owner's own may be taken.
            return [_Hit("someone-else/Qwen3-Embedding-8B-GGUF"), _Hit("unsloth/unrelated")]

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda *a, **k: ["x.gguf"])

    assert settings._search_hub_for_gguf("unsloth/Qwen3-Embedding-8B", None) is None
    # And the query itself is scoped to the owner, not filtered only afterwards.
    assert seen["author"] == "unsloth"


def _no_gguf_anywhere(monkeypatch):
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: True)
    monkeypatch.setattr(settings, "_cached_embedding_gguf", lambda candidates: False)
    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", lambda candidates, token: None)
    monkeypatch.setattr(settings, "_search_hub_for_gguf", lambda m, token: None)


def test_no_gguf_falls_back_to_the_models_own_safetensors(client, monkeypatch):
    """Safetensors cost about 1 GB more memory but they load, so they beat both
    refusing the model and pulling a stranger's conversion."""
    _no_gguf_anywhere(monkeypatch)
    monkeypatch.setattr(
        settings, "_safetensors_plan", lambda m, token: (m, ["model.safetensors"])
    )
    monkeypatch.setattr(settings, "_hf_files_size", lambda repo, files, token: 4096)
    import utils.utils as utils

    monkeypatch.setattr(utils, "hf_cache_snapshot_is_loadable", lambda m: False)

    body = _resolve(client, "unsloth/Qwen3-Embedding-8B").json()
    assert body["backend"] == "sentence-transformers"
    assert body["download_repo"] == "unsloth/Qwen3-Embedding-8B"
    # No file list: ST needs the config and tokenizer too, not just the weights.
    assert body["files"] is None
    assert body["size_bytes"] == 4096
    assert body["error"] is None


def test_the_fallback_stays_in_the_models_own_repo(monkeypatch):
    """Same rule as the GGUF search: only the publisher is a source."""
    monkeypatch.setattr(settings, "_st_backend_available", lambda: True)
    monkeypatch.setattr(
        settings,
        "_st_weight_files",
        lambda m, token: ["model.safetensors"] if m == "acme/embedder" else None,
    )
    assert settings._safetensors_plan("acme/embedder", None) == (
        "acme/embedder",
        ["model.safetensors"],
    )
    assert settings._safetensors_plan("acme/no-weights", None) is None


def test_a_gguf_only_install_is_not_offered_safetensors(monkeypatch):
    """No torch here, so ST is not a working answer and the save reason stands."""
    monkeypatch.setattr(settings, "_st_backend_available", lambda: False)
    monkeypatch.setattr(settings, "_st_weight_files", lambda m, token: ["model.safetensors"])
    assert settings._safetensors_plan("acme/embedder", None) is None


def test_nothing_anywhere_still_reports_the_save_reason(client, monkeypatch):
    """Only when the name candidates, the Hub search AND safetensors come up empty."""
    _no_gguf_anywhere(monkeypatch)
    monkeypatch.setattr(settings, "_safetensors_plan", lambda m, token: None)
    monkeypatch.setattr(
        settings, "_hf_gguf_backend_error", lambda m, token: "No GGUF weights found in ..."
    )

    body = _resolve(client, "unsloth/nothing-like-this").json()
    assert body["download_repo"] is None
    assert body["files"] is None
    assert body["cached"] is False
    assert body["error"] == "No GGUF weights found in ..."


def test_a_local_gguf_is_already_the_artifact(client, monkeypatch):
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: True)
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: True)

    body = _resolve(client, "/models/my-embedder.gguf").json()
    assert body["cached"] is True
    assert body["download_repo"] is None
    assert body["error"] is None


def test_candidates_follow_the_loader_order(monkeypatch):
    """One helper builds them, so the picker cannot fetch from a repo the loader
    would not have opened."""
    assert settings._embedding_gguf_candidates("acme/embedder") == [
        "acme/embedder-GGUF",
        "acme/embedder",
    ]
    # A repo that already names GGUF is its own candidate, not "...-GGUF-GGUF".
    assert settings._embedding_gguf_candidates("acme/embedder-GGUF") == ["acme/embedder-GGUF"]
    # unsloth's unquantized re-uploads keep their GGUF on the base name.
    assert settings._embedding_gguf_candidates(
        "unsloth/embeddinggemma-300m-qat-q8_0-unquantized"
    ) == [
        "unsloth/embeddinggemma-300m-qat-q8_0-unquantized-GGUF",
        "unsloth/embeddinggemma-300m-GGUF",
        "unsloth/embeddinggemma-300m-qat-q8_0-unquantized",
    ]


def test_the_resolved_repo_is_what_the_loader_opens(monkeypatch):
    """A conversion under another owner follows no naming rule, so it is stored and
    read back rather than re-derived."""
    import storage.studio_db as db

    store: dict = {}
    monkeypatch.setattr(db, "get_app_setting", lambda k, fallback = None: store.get(k, fallback))
    monkeypatch.setattr(db, "upsert_app_settings", lambda s: store.update(s) or store)

    import utils.embedding_model_settings as ems
    from core.rag import config as rag_config

    ems._invalidate_cache()
    ems.set_rag_embedding_model(
        "unsloth/Qwen3-Embedding-4B", gguf_repo = "Qwen/Qwen3-Embedding-4B-GGUF"
    )
    assert rag_config.effective_gguf_repo() == "Qwen/Qwen3-Embedding-4B-GGUF"

    # A pair recorded for another model must never be served for this one.
    store[ems.EMBEDDING_GGUF_SETTING_KEY] = "Qwen/Qwen3-Embedding-4B-GGUF"
    ems._invalidate_cache()
    assert ems.get_stored_gguf_repo("unsloth/bge-m3") is None
    ems._invalidate_cache()


def test_the_chosen_backend_is_read_back_by_the_loader(monkeypatch):
    """A safetensors-only model must not be handed to llama-server, which would have
    nothing to open. The picker records the backend; ``auto`` honours it."""
    import storage.studio_db as db

    store: dict = {}
    monkeypatch.setattr(db, "get_app_setting", lambda k, fallback = None: store.get(k, fallback))
    monkeypatch.setattr(db, "upsert_app_settings", lambda s: store.update(s) or store)

    import utils.embedding_model_settings as ems
    from core.rag import embeddings as rag_embeddings

    ems._invalidate_cache()
    ems.set_rag_embedding_model(
        "unsloth/Qwen3-Embedding-8B", backend = "sentence-transformers"
    )
    assert ems.get_stored_backend("unsloth/Qwen3-Embedding-8B") == "sentence-transformers"
    assert rag_embeddings._resolve_auto_for_model() == "sentence-transformers"

    # A backend recorded for another model must not be served for this one.
    store[ems.EMBEDDING_MODEL_SETTING_KEY] = "unsloth/bge-m3"
    ems._invalidate_cache()
    assert ems.get_stored_backend("unsloth/Qwen3-Embedding-8B") is None
    ems._invalidate_cache()


def test_the_token_is_a_header_not_a_query_parameter(client, monkeypatch):
    """A gated repo's credential must stay out of URLs and access logs."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: True)
    monkeypatch.setattr(settings, "_cached_embedding_gguf", lambda candidates: False)
    seen: dict = {}

    def _plan(candidates, token):
        seen["token"] = token
        return candidates[0], "model-F16.gguf"

    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", _plan)
    monkeypatch.setattr(settings, "_hf_files_size", lambda repo, files, token: None)

    client.get(
        "/embedding-model/resolve",
        params = {"model": "acme/gated-embedder"},
        headers = {"X-Unsloth-HF-Token": "hf_secret"},
    )
    assert seen["token"] == "hf_secret"

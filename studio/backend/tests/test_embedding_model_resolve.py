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


def test_a_repo_with_no_gguf_reports_the_same_reason_the_save_would(client, monkeypatch):
    """The screenshot case: a model whose companion repo publishes no GGUF. The
    picker must not offer a download for it, and must say why in the PUT's words."""
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: True)
    monkeypatch.setattr(settings, "_cached_embedding_gguf", lambda candidates: False)
    monkeypatch.setattr(settings, "_remote_embedding_gguf_plan", lambda candidates, token: None)
    monkeypatch.setattr(
        settings, "_hf_gguf_backend_error", lambda m, token: "No GGUF weights found in ..."
    )

    body = _resolve(client, "unsloth/gte-modernbert-base").json()
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
    """Companion repo first, then the model repo itself: the order
    `_resolve_model_path` tries, so the picker cannot fetch from a repo the loader
    would not have used."""
    assert settings._embedding_gguf_candidates("acme/embedder") == [
        "acme/embedder-GGUF",
        "acme/embedder",
    ]
    # A repo that already names GGUF is its own candidate, not "…-GGUF-GGUF".
    assert settings._embedding_gguf_candidates("acme/embedder-GGUF") == ["acme/embedder-GGUF"]


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

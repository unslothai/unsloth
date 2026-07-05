# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The RAG embedding model must pass the malware/pickle gate before it is persisted or
loaded. A flagged repo (or any repo saved with force) previously reached
SentenceTransformer unscanned, bypassing the normal model-load protections."""

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


class _Decision:
    def __init__(self, blocked):
        self.blocked = blocked


def _security_stub(blocked):
    mod = _types.ModuleType("utils.security")
    mod.evaluate_file_security = lambda *a, **k: _Decision(blocked)
    mod.security_load_subdirs = lambda *a, **k: ()
    return mod


@pytest.fixture
def client(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(settings, "default_embedding_model", lambda: "unsloth/default-embed")
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(settings, "set_rag_embedding_model", lambda v: saved.setdefault("model", v))
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: False)
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: False)
    monkeypatch.setattr(settings, "get_rag_embedding_model", lambda: saved.get("model", ""))
    monkeypatch.setattr(settings, "get_stored_embedding_model", lambda: saved.get("model"))

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False), saved


def test_flagged_repo_is_blocked_even_with_force(client, monkeypatch):
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))
    r = c.put(
        "/embedding-model", json = {"embedding_model": "attacker/malicious-embed", "force": True}
    )
    assert r.status_code == 409
    assert "model" not in saved  # force must not persist a flagged repo


def test_flagged_repo_is_blocked_without_force(client, monkeypatch):
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))
    r = c.put("/embedding-model", json = {"embedding_model": "attacker/malicious-embed"})
    assert r.status_code == 409
    assert "model" not in saved


def test_clean_repo_saves_under_force(client, monkeypatch):
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    r = c.put("/embedding-model", json = {"embedding_model": "acme/clean-embed", "force": True})
    assert r.status_code == 200
    assert saved.get("model") == "acme/clean-embed"


def test_load_sink_refuses_flagged_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))
    import core.rag.embeddings as embeddings
    with pytest.raises(embeddings.UnsafeEmbeddingModelError):
        embeddings._guard_model_security("attacker/malicious-embed")


def test_load_sink_allows_clean_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    import core.rag.embeddings as embeddings
    embeddings._guard_model_security("acme/clean-embed")  # no raise


def test_sink_threads_ambient_token_into_scan(monkeypatch):
    # A gated repo set via env/default has no request token; the guard must feed the
    # loader's own token to the scan, or it fails open for the repo that still loads.
    seen = {}
    mod = _types.ModuleType("utils.security")
    mod.security_load_subdirs = (
        lambda name, token = None: seen.setdefault("subdirs_token", token) or ()
    )
    mod.evaluate_file_security = lambda *a, **k: seen.setdefault(
        "scan_token", k.get("hf_token")
    ) or _Decision(False)
    monkeypatch.setitem(sys.modules, "utils.security", mod)
    import core.rag.embeddings as embeddings

    monkeypatch.setattr(embeddings, "_ambient_hf_token", lambda: "hf_ambient")
    embeddings._guard_model_security("acme/gated-embed")
    assert seen["scan_token"] == "hf_ambient"
    assert seen["subdirs_token"] == "hf_ambient"


def test_security_block_is_not_swallowed_by_llama_fallback(monkeypatch):
    # The ST encode fallback must re-raise a security block, not swap to llama-server.
    import core.rag.embeddings as embeddings

    def _boom(*a, **k):
        raise embeddings.UnsafeEmbeddingModelError("flagged")

    monkeypatch.setattr(embeddings, "_st_encode", _boom)
    monkeypatch.setattr(
        embeddings,
        "_switch_to_llama_fallback",
        lambda err: pytest.fail("security block must not fall back to llama-server"),
    )
    with pytest.raises(embeddings.UnsafeEmbeddingModelError):
        embeddings._SentenceTransformersBackend().encode(["hi"])

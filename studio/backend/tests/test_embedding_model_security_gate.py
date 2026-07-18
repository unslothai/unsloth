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
    # The settings scan unions in the ST module dirs read from modules.json; keep it
    # offline and deterministic for the endpoint tests that use this fixture.
    import core.rag.embeddings as embeddings

    monkeypatch.setattr(embeddings, "_st_module_subdirs", lambda name, token = None: ())
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
    # 403, not the forceable 409, so the client does not offer "save anyway".
    assert r.status_code == 403
    assert "model" not in saved  # force must not persist a flagged repo


def test_flagged_repo_is_blocked_without_force(client, monkeypatch):
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))
    r = c.put("/embedding-model", json = {"embedding_model": "attacker/malicious-embed"})
    assert r.status_code == 403
    assert "model" not in saved


def test_hard_block_uses_non_forceable_status(client, monkeypatch):
    # The forceable verification path uses 409; the hard security block must be distinct
    # (403) so the frontend never routes it into the "save anyway" force flow.
    c, _saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))
    blocked = c.put("/embedding-model", json = {"embedding_model": "attacker/malicious-embed"})
    assert blocked.status_code == 403

    # A verification failure (not-an-embedding-model) stays forceable at 409.
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    monkeypatch.setattr(settings, "is_embedding_model", lambda *a, **k: False, raising = False)
    import utils.models as _models

    monkeypatch.setattr(_models, "is_embedding_model", lambda *a, **k: False)
    unverified = c.put("/embedding-model", json = {"embedding_model": "acme/not-an-embedder"})
    assert unverified.status_code == 409


def test_llama_backend_skips_the_st_pickle_scan(monkeypatch):
    # On the llama-server backend the embedder loads GGUF (inert), not the ST repo's
    # pickle, so a flagged ST repo with a clean GGUF companion must not be rejected here.
    saved: dict = {}
    monkeypatch.setattr(settings, "default_embedding_model", lambda: "unsloth/default-embed")
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(settings, "set_rag_embedding_model", lambda v: saved.setdefault("model", v))
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: True)
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: False)
    monkeypatch.setattr(settings, "get_rag_embedding_model", lambda: saved.get("model", ""))
    monkeypatch.setattr(settings, "get_stored_embedding_model", lambda: saved.get("model"))
    # force skips the GGUF availability checks; the ST pickle gate is what we assert is skipped.
    called = {"scanned": False}
    mod = _types.ModuleType("utils.security")

    def _fail(*a, **k):
        called["scanned"] = True
        return _Decision(True)

    mod.evaluate_file_security = _fail
    mod.security_load_subdirs = lambda *a, **k: ()
    monkeypatch.setitem(sys.modules, "utils.security", mod)

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    c = TestClient(app, raise_server_exceptions = False)
    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "attacker/flagged-st-clean-gguf", "force": True},
    )
    assert r.status_code == 200
    assert called["scanned"] is False  # the ST pickle scan never ran on the llama path
    assert saved.get("model") == "attacker/flagged-st-clean-gguf"


def test_runtime_llama_fallback_skips_the_st_pickle_scan(monkeypatch):
    # auto resolves to sentence-transformers (GPU present) but the embedder fell back to
    # llama-server at runtime (torch/CUDA load or encode failure), so the process now loads
    # only inert GGUF. The real _llama_backend_active() must reflect that cached fallback,
    # so a flagged ST repo with a clean GGUF companion must not be hard-blocked here.
    import core.rag.embeddings as embeddings
    from core.rag.embed_llama_server import LlamaServerBackend

    # Simulate the runtime fallback: the process-wide backend is a LlamaServerBackend even
    # though the auto resolver would still say sentence-transformers.
    monkeypatch.setattr(embeddings, "_backend", LlamaServerBackend())
    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "sentence-transformers")
    monkeypatch.setattr(embeddings, "_st_module_subdirs", lambda name, token = None: ())

    saved: dict = {}
    monkeypatch.setattr(settings, "default_embedding_model", lambda: "unsloth/default-embed")
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(settings, "set_rag_embedding_model", lambda v: saved.setdefault("model", v))
    # Deliberately do NOT monkeypatch settings._llama_backend_active: this test exercises the
    # real delegation to embeddings.active_backend_is_llama() so the cached fallback is honored.
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: False)
    monkeypatch.setattr(settings, "get_rag_embedding_model", lambda: saved.get("model", ""))
    monkeypatch.setattr(settings, "get_stored_embedding_model", lambda: saved.get("model"))

    called = {"scanned": False}
    mod = _types.ModuleType("utils.security")

    def _fail(*a, **k):
        called["scanned"] = True
        return _Decision(True)

    mod.evaluate_file_security = _fail
    mod.security_load_subdirs = lambda *a, **k: ()
    monkeypatch.setitem(sys.modules, "utils.security", mod)

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    c = TestClient(app, raise_server_exceptions = False)
    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "attacker/flagged-st-clean-gguf", "force": True},
    )
    assert r.status_code == 200
    assert called["scanned"] is False  # the ST pickle scan never ran on the llama fallback
    assert saved.get("model") == "attacker/flagged-st-clean-gguf"


def test_active_backend_is_llama_reflects_cache_and_resolver(monkeypatch):
    # active_backend_is_llama() reports the ACTUAL built backend when one exists, and defers
    # to the resolver (fresh-process behavior) when none has been built yet.
    import core.rag.embeddings as embeddings
    import core.rag.config as rag_config
    from core.rag.embed_llama_server import LlamaServerBackend

    # A cached llama backend wins even when auto would resolve to sentence-transformers.
    monkeypatch.setattr(rag_config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "sentence-transformers")
    monkeypatch.setattr(embeddings, "_backend", LlamaServerBackend())
    assert embeddings.active_backend_is_llama() is True

    # A cached ST backend reports False even when the resolver now picks llama, so its
    # pickle stays gated (the cached backend, not the resolver, is what actually embeds).
    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "llama-server")
    monkeypatch.setattr(embeddings, "_backend", embeddings._SentenceTransformersBackend())
    assert embeddings.active_backend_is_llama() is False

    # No cached backend -> the resolver decides, unchanged from before.
    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "sentence-transformers")
    monkeypatch.setattr(embeddings, "_backend", None)
    assert embeddings.active_backend_is_llama() is False  # auto -> sentence-transformers

    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "llama-server")
    assert embeddings.active_backend_is_llama() is True  # auto -> llama-server

    # An explicit (non-auto) key is honored verbatim without a cached backend.
    monkeypatch.setattr(rag_config, "EMBED_BACKEND", "llama-server")
    assert embeddings.active_backend_is_llama() is True


def test_settings_scan_scopes_module_subdirs(monkeypatch):
    # The settings scan must pass the ST module dirs (0_Transformer/) as load roots so a
    # pickle directly under one blocks; assert those subdirs reach evaluate_file_security.
    saved: dict = {}
    monkeypatch.setattr(settings, "default_embedding_model", lambda: "unsloth/default-embed")
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(settings, "set_rag_embedding_model", lambda v: saved.setdefault("model", v))
    monkeypatch.setattr(settings, "_llama_backend_active", lambda: False)
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: False)
    monkeypatch.setattr(settings, "get_rag_embedding_model", lambda: saved.get("model", ""))
    monkeypatch.setattr(settings, "get_stored_embedding_model", lambda: saved.get("model"))

    import core.rag.embeddings as embeddings

    monkeypatch.setattr(
        embeddings, "_st_module_subdirs", lambda name, token = None: ("0_Transformer",)
    )
    seen = {}

    def _capture(*a, **k):
        seen["subdirs"] = tuple(k.get("load_subdirs") or ())
        return _Decision(False)

    mod = _types.ModuleType("utils.security")
    mod.security_load_subdirs = lambda *a, **k: ()
    mod.evaluate_file_security = _capture
    monkeypatch.setitem(sys.modules, "utils.security", mod)

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    c = TestClient(app, raise_server_exceptions = False)
    r = c.put(
        "/embedding-model", json = {"embedding_model": "acme/embed-with-module-dir", "force": True}
    )
    assert r.status_code == 200
    assert "0_Transformer" in seen["subdirs"]


def test_clean_repo_saves_under_force(client, monkeypatch):
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    r = c.put("/embedding-model", json = {"embedding_model": "acme/clean-embed", "force": True})
    assert r.status_code == 200
    assert saved.get("model") == "acme/clean-embed"


def test_custom_model_saved_in_cache_casing(client, monkeypatch):
    # A custom repo is persisted in the casing its local HF cache dir uses, so an
    # offline exact-case SentenceTransformer load still finds it (baai/bge-m3 ->
    # BAAI/bge-m3). force=True bypasses the verification gate to isolate this.
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    import utils.paths as _paths

    monkeypatch.setattr(_paths, "resolve_cached_repo_id_case", lambda m: "BAAI/bge-m3")
    r = c.put("/embedding-model", json = {"embedding_model": "baai/bge-m3", "force": True})
    assert r.status_code == 200
    assert saved.get("model") == "BAAI/bge-m3"


def test_default_model_is_not_casing_normalized(client, monkeypatch):
    # Submitting the exact default must NOT be run through cache-casing
    # normalization: rewriting it would make set_rag_embedding_model()'s exact
    # default comparison treat it as a custom override, so later default changes
    # would stop applying. resolve_cached_repo_id_case must not be consulted.
    c, saved = client
    import utils.paths as _paths

    def _must_not_run(m):
        raise AssertionError("the default must not be casing-normalized")

    monkeypatch.setattr(_paths, "resolve_cached_repo_id_case", _must_not_run)
    r = c.put("/embedding-model", json = {"embedding_model": "unsloth/default-embed"})
    assert r.status_code == 200
    assert saved.get("model") == "unsloth/default-embed"


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


def test_sink_scopes_st_module_subdirs_into_scan(monkeypatch):
    # A flagged pickle directly under a Transformer module dir (0_Transformer/) must
    # reach the scan as a load root; assert the guard unions the module dirs into
    # load_subdirs so evaluate_file_security treats such a pickle as root-level.
    seen = {}

    def _capture(*a, **k):
        seen["subdirs"] = tuple(k.get("load_subdirs") or ())
        return _Decision(False)

    mod = _types.ModuleType("utils.security")
    mod.security_load_subdirs = lambda name, token = None: ()
    mod.evaluate_file_security = _capture
    monkeypatch.setitem(sys.modules, "utils.security", mod)
    import core.rag.embeddings as embeddings

    monkeypatch.setattr(embeddings, "_ambient_hf_token", lambda: None)
    monkeypatch.setattr(
        embeddings, "_st_module_subdirs", lambda name, token = None: ("0_Transformer",)
    )
    embeddings._guard_model_security("acme/embed-with-module-dir")
    assert "0_Transformer" in seen["subdirs"]


def test_st_module_subdirs_reads_local_modules_json(tmp_path, monkeypatch):
    # The helper must parse each module's non-empty "path" from a local repo's
    # modules.json and drop the root-level ("") Transformer entry.
    import json
    import core.rag.embeddings as embeddings

    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "name": "0", "path": "0_Transformer", "type": "..."},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "..."},
                {"idx": 2, "name": "2", "path": "", "type": "..."},
            ]
        )
    )
    subdirs = embeddings._st_module_subdirs(str(tmp_path), None)
    assert subdirs == ("0_Transformer", "1_Pooling")


def test_st_module_subdirs_swallows_errors(monkeypatch):
    # Any failure (no modules.json, offline, malformed) returns () so the guard never
    # bricks the embedder.
    import huggingface_hub
    import core.rag.embeddings as embeddings

    def _boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    assert embeddings._st_module_subdirs("acme/no-such-repo-xyz", None) == ()


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

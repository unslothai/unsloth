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


def _plan(model, backend):
    return settings.EmbeddingModelResolveResponse(
        embedding_model = model,
        backend = backend,
        download_repo = f"{model}-GGUF" if backend == "llama" else model,
    )


@pytest.fixture
def client(monkeypatch):
    # keep the modules.json ST dir scan offline and deterministic
    import core.rag.embeddings as embeddings

    monkeypatch.setattr(embeddings, "_st_module_subdirs", lambda name, token = None: ())
    saved: dict = {}
    monkeypatch.setattr(settings, "default_embedding_model", lambda: "unsloth/default-embed")
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(
        settings,
        "set_rag_embedding_model",
        lambda v, gguf_repo = None, backend = None, download_pending = False, gguf_files = None: (
            saved.update(
                model = v,
                gguf_repo = gguf_repo,
                backend = backend,
                download_pending = download_pending,
                gguf_files = gguf_files,
            )
        ),
    )
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(
        settings,
        "_resolve_embedding_model_plan",
        lambda model, token: _plan(
            model, "llama" if settings._llama_backend_active() else "sentence-transformers"
        ),
    )
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: False)
    monkeypatch.setattr(settings, "get_rag_embedding_model", lambda: saved.get("model", ""))
    monkeypatch.setattr(settings, "get_stored_embedding_model", lambda: saved.get("model"))
    monkeypatch.setattr(
        settings,
        "effective_gguf_repo_for_embedding_model",
        lambda model: f"{model or 'unsloth/default-embed'}-GGUF",
    )
    monkeypatch.setattr(
        settings,
        "default_gguf_repo",
        lambda: "unsloth/default-embed-GGUF",
    )

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings.allow_ambient_hf_token] = lambda: True
    return TestClient(app, raise_server_exceptions = False), saved


def test_flagged_repo_is_blocked_even_with_force(client, monkeypatch):
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))
    r = c.put(
        "/embedding-model", json = {"embedding_model": "attacker/malicious-embed", "force": True}
    )
    # 403, not the forceable 409, so the client does not offer "save anyway"
    assert r.status_code == 403
    assert "model" not in saved


def test_flagged_repo_is_blocked_without_force(client, monkeypatch):
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))
    r = c.put("/embedding-model", json = {"embedding_model": "attacker/malicious-embed"})
    assert r.status_code == 403
    assert "model" not in saved


def test_uncached_selection_is_marked_pending_so_loaders_stay_offline(client, monkeypatch):
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    import utils.models as models

    monkeypatch.setattr(models, "is_embedding_model", lambda *a, **k: True)
    response = c.put("/embedding-model", json = {"embedding_model": "acme/embedder"})

    assert response.status_code == 200
    assert saved["download_pending"] is True


def test_hard_block_uses_non_forceable_status(client, monkeypatch):
    c, _saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))
    blocked = c.put("/embedding-model", json = {"embedding_model": "attacker/malicious-embed"})
    assert blocked.status_code == 403

    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    monkeypatch.setattr(settings, "is_embedding_model", lambda *a, **k: False, raising = False)
    import utils.models as _models

    monkeypatch.setattr(_models, "is_embedding_model", lambda *a, **k: False)
    unverified = c.put("/embedding-model", json = {"embedding_model": "acme/not-an-embedder"})
    assert unverified.status_code == 409


def test_offline_cached_non_st_model_is_accepted(client, monkeypatch):
    # ST can load any cached encoder offline, so accept it without HF metadata
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import utils.models as _models
    import utils.utils as _uu

    monkeypatch.setattr(_models, "is_embedding_model", lambda *a, **k: False)
    monkeypatch.setattr(_uu, "hf_cache_snapshot_is_loadable", lambda name: True)
    r = c.put("/embedding-model", json = {"embedding_model": "acme/gte-modernbert"})
    assert r.status_code == 200
    assert saved.get("model") == "acme/gte-modernbert"


def test_offline_partial_or_uncached_model_still_409(client, monkeypatch):
    c, _saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import utils.models as _models
    import utils.utils as _uu

    monkeypatch.setattr(_models, "is_embedding_model", lambda *a, **k: False)
    monkeypatch.setattr(_uu, "hf_cache_snapshot_is_loadable", lambda name: False)
    r = c.put("/embedding-model", json = {"embedding_model": "acme/uncached-embedder"})
    assert r.status_code == 409


def test_offline_skips_remote_gguf_probe(client, monkeypatch):
    # offline llama path must skip list_repo_files so a dead-DNS session cannot hang
    c, _saved = client
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(settings, "_local_gguf_backend_error", lambda model: None)

    def _boom(*a, **k):
        raise AssertionError("hit the network for the GGUF probe")

    monkeypatch.setattr(settings, "_hf_gguf_backend_error", _boom)
    import utils.models as _models

    monkeypatch.setattr(_models, "is_embedding_model", lambda *a, **k: True)
    r = c.put("/embedding-model", json = {"embedding_model": "acme/embedder"})
    assert r.status_code == 200


def test_client_cannot_persist_an_unvalidated_gguf_repo(client, monkeypatch):
    c, saved = client
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))

    r = c.put(
        "/embedding-model",
        json = {
            "embedding_model": "acme/embedder",
            "backend": "llama",
            "gguf_repo": "attacker/unrelated-llm-GGUF",
        },
    )
    assert r.status_code == 400
    assert "model" not in saved


def test_security_scan_uses_the_resolved_destination_backend(client, monkeypatch):
    """The old backend may be llama while the selected model resolves to ST."""
    c, saved = client
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(
        settings,
        "_resolve_embedding_model_plan",
        lambda model, token: _plan(model, "sentence-transformers"),
    )
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))

    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "attacker/flagged-st", "backend": "sentence-transformers"},
    )
    assert r.status_code == 403
    assert "model" not in saved


def test_llama_backend_skips_the_st_pickle_scan(monkeypatch):
    # the llama backend loads inert GGUF, not the ST pickle, so a flagged ST repo is fine
    saved: dict = {}
    monkeypatch.setattr(settings, "default_embedding_model", lambda: "unsloth/default-embed")
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(
        settings,
        "set_rag_embedding_model",
        lambda v, gguf_repo = None, backend = None, download_pending = False, gguf_files = None: (
            saved.update(
                model = v,
                gguf_repo = gguf_repo,
                backend = backend,
                download_pending = download_pending,
                gguf_files = gguf_files,
            )
        ),
    )
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(
        settings,
        "_resolve_embedding_model_plan",
        lambda model, token: _plan(model, "llama"),
    )
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
    app.dependency_overrides[settings.allow_ambient_hf_token] = lambda: True
    c = TestClient(app, raise_server_exceptions = False)
    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "attacker/flagged-st-clean-gguf", "force": True},
    )
    assert r.status_code == 200
    assert called["scanned"] is False
    assert saved.get("model") == "attacker/flagged-st-clean-gguf"


def test_runtime_llama_fallback_skips_the_st_pickle_scan(monkeypatch):
    # the embedder fell back to llama-server at runtime; the real check must honor that
    import core.rag.embeddings as embeddings
    from core.rag.embed_llama_server import LlamaServerBackend

    monkeypatch.setattr(embeddings, "_backend", LlamaServerBackend())
    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "sentence-transformers")
    monkeypatch.setattr(embeddings, "_st_module_subdirs", lambda name, token = None: ())

    saved: dict = {}
    monkeypatch.setattr(settings, "default_embedding_model", lambda: "unsloth/default-embed")
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(
        settings,
        "set_rag_embedding_model",
        lambda v, gguf_repo = None, backend = None, download_pending = False, gguf_files = None: (
            saved.update(
                model = v,
                gguf_repo = gguf_repo,
                backend = backend,
                download_pending = download_pending,
                gguf_files = gguf_files,
            )
        ),
    )
    # deliberately not patching _llama_backend_active: exercises the real delegation
    monkeypatch.setattr(settings, "_resolves_as_local_gguf", lambda m: False)
    monkeypatch.setattr(settings, "get_rag_embedding_model", lambda: saved.get("model", ""))
    monkeypatch.setattr(settings, "get_stored_embedding_model", lambda: saved.get("model"))
    monkeypatch.setattr(
        settings,
        "_resolve_embedding_model_plan",
        lambda model, token: _plan(model, "llama"),
    )

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
    app.dependency_overrides[settings.allow_ambient_hf_token] = lambda: True
    c = TestClient(app, raise_server_exceptions = False)
    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "attacker/flagged-st-clean-gguf", "force": True},
    )
    assert r.status_code == 200
    assert called["scanned"] is False
    assert saved.get("model") == "attacker/flagged-st-clean-gguf"


def test_active_backend_is_llama_reflects_cache_and_resolver(monkeypatch):
    import core.rag.embeddings as embeddings
    import core.rag.config as rag_config
    from core.rag.embed_llama_server import LlamaServerBackend

    monkeypatch.setattr(rag_config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "sentence-transformers")
    monkeypatch.setattr(embeddings, "_backend", LlamaServerBackend())
    assert embeddings.active_backend_is_llama() is True

    # the cached backend, not the resolver, is what actually embeds
    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "llama-server")
    monkeypatch.setattr(embeddings, "_backend", embeddings._SentenceTransformersBackend())
    assert embeddings.active_backend_is_llama() is False

    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "sentence-transformers")
    monkeypatch.setattr(embeddings, "_backend", None)
    assert embeddings.active_backend_is_llama() is False

    monkeypatch.setattr(embeddings, "_resolve_auto", lambda: "llama-server")
    assert embeddings.active_backend_is_llama() is True

    monkeypatch.setattr(rag_config, "EMBED_BACKEND", "llama-server")
    assert embeddings.active_backend_is_llama() is True


def test_settings_scan_scopes_module_subdirs(monkeypatch):
    from utils import utils as studio_utils

    monkeypatch.setattr(studio_utils, "hf_env_offline", lambda: False)
    saved: dict = {}
    monkeypatch.setattr(settings, "default_embedding_model", lambda: "unsloth/default-embed")
    monkeypatch.setattr(settings, "validate_embedding_model", lambda v: v)
    monkeypatch.setattr(
        settings,
        "set_rag_embedding_model",
        lambda v, gguf_repo = None, backend = None, download_pending = False, gguf_files = None: (
            saved.update(
                model = v,
                gguf_repo = gguf_repo,
                backend = backend,
                download_pending = download_pending,
                gguf_files = gguf_files,
            )
        ),
    )
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: False)
    monkeypatch.setattr(
        settings,
        "_resolve_embedding_model_plan",
        lambda model, token: _plan(model, "sentence-transformers"),
    )
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
    app.dependency_overrides[settings.allow_ambient_hf_token] = lambda: True
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
    assert r.json() == {
        "embedding_model": "acme/clean-embed",
        "embedding_gguf_repo": "acme/clean-embed-GGUF",
        "default_embedding_model": "unsloth/default-embed",
        "default_embedding_gguf_repo": "unsloth/default-embed-GGUF",
        "is_custom": True,
        "loaded": False,
        "backend_loaded": False,
    }


def test_load_sink_refuses_flagged_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = True))
    import core.rag.embeddings as embeddings
    with pytest.raises(embeddings.UnsafeEmbeddingModelError):
        embeddings._guard_model_security("attacker/malicious-embed")


def test_load_sink_allows_clean_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    import core.rag.embeddings as embeddings
    embeddings._guard_model_security("acme/clean-embed")


def test_sink_threads_ambient_token_into_scan(monkeypatch):
    # a repo set via env has no request token; the scan must get the loader's token or fail open
    seen = {}
    mod = _types.ModuleType("utils.security")
    mod.security_load_subdirs = lambda name, token = None: (
        seen.setdefault("subdirs_token", token) or ()
    )
    mod.evaluate_file_security = lambda *a, **k: (
        seen.setdefault("scan_token", k.get("hf_token")) or _Decision(False)
    )
    monkeypatch.setitem(sys.modules, "utils.security", mod)
    import core.rag.embeddings as embeddings

    monkeypatch.setattr(embeddings, "_ambient_hf_token", lambda: "hf_ambient")
    embeddings._guard_model_security("acme/gated-embed")
    assert seen["scan_token"] == "hf_ambient"
    assert seen["subdirs_token"] == "hf_ambient"


def test_sink_scopes_st_module_subdirs_into_scan(monkeypatch):
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
    import huggingface_hub
    import core.rag.embeddings as embeddings

    def _boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    assert embeddings._st_module_subdirs("acme/no-such-repo-xyz", None) == ()


def test_security_block_is_not_swallowed_by_llama_fallback(monkeypatch):
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


def _erroring_plan(model, backend, error):
    return settings.EmbeddingModelResolveResponse(
        embedding_model = model, backend = backend, error = error
    )


def test_a_sentence_transformers_plan_error_is_refused_not_persisted(client, monkeypatch):
    """The PUT raised on plan.error only for llama destinations, so a repo passing
    the tag gate with no loadable checkpoint was persisted anyway."""
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    import utils.models as models

    monkeypatch.setattr(models, "is_embedding_model", lambda *a, **k: True)
    monkeypatch.setattr(
        settings,
        "_resolve_embedding_model_plan",
        lambda model, token: _erroring_plan(
            model, "sentence-transformers", "No sentence-transformers weights found."
        ),
    )

    r = c.put("/embedding-model", json = {"embedding_model": "acme/gguf-only"})
    assert r.status_code == 409
    assert "No sentence-transformers weights found." in r.json()["detail"]
    assert "model" not in saved


def test_forcing_over_a_failed_plan_stays_cache_only(client, monkeypatch):
    """Save anyway over a failed plan recorded no marker, so both loaders took
    their uncached path and fetched invisibly at the first index."""
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    monkeypatch.setattr(
        settings,
        "_resolve_embedding_model_plan",
        lambda model, token: _erroring_plan(model, "sentence-transformers", "cannot resolve"),
    )

    r = c.put("/embedding-model", json = {"embedding_model": "acme/embedder", "force": True})

    assert r.status_code == 200
    assert saved["model"] == "acme/embedder"
    assert saved["backend"] is None
    assert saved["gguf_repo"] is None
    assert saved["download_pending"] is True


def test_unload_is_offered_while_another_model_is_still_resident(client, monkeypatch):
    """Saving a new model does not release the old one, and `loaded` answers only
    about the selected one, so the previous model had no control to free it."""
    c, _saved = client
    import core.rag.embeddings as embeddings

    monkeypatch.setattr(embeddings, "backend_is_loaded", lambda model_name = None: model_name is None)

    body = c.get("/embedding-model").json()
    assert body["loaded"] is False
    assert body["backend_loaded"] is True

    monkeypatch.setattr(embeddings, "backend_is_loaded", lambda model_name = None: False)
    body = c.get("/embedding-model").json()
    assert body["loaded"] is False
    assert body["backend_loaded"] is False


def test_the_resolved_repo_is_what_gets_verified_and_scanned(client, monkeypatch):
    """A slashless alias resolves under sentence-transformers/, but the PUT ran
    is_embedding_model and the malware scan against the literal name: a repo that
    usually does not exist (fail-open, or a forceable 409) or, worse, a different
    top-level repo that does."""
    from utils import utils as studio_utils

    monkeypatch.setattr(studio_utils, "hf_env_offline", lambda: False)
    c, saved = client
    seen = {}

    def _subdirs(name, token = None):
        seen["subdirs"] = name
        return ()

    def _scan(name, **_kwargs):
        seen["scanned"] = name
        return _Decision(False)

    def _is_embedding(name, **_kwargs):
        seen["verified"] = name
        return True

    mod = _types.ModuleType("utils.security")
    mod.security_load_subdirs = _subdirs
    mod.evaluate_file_security = _scan
    monkeypatch.setitem(sys.modules, "utils.security", mod)
    import utils.models as models

    monkeypatch.setattr(models, "is_embedding_model", _is_embedding)
    monkeypatch.setattr(
        settings,
        "_resolve_embedding_model_plan",
        lambda model, token: settings.EmbeddingModelResolveResponse(
            embedding_model = model,
            backend = "sentence-transformers",
            download_repo = "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )

    r = c.put("/embedding-model", json = {"embedding_model": "all-MiniLM-L6-v2"})
    assert r.status_code == 200
    assert saved["model"] == "all-MiniLM-L6-v2"
    assert seen["scanned"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert seen["subdirs"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert seen["verified"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_a_llama_download_repo_is_not_used_as_the_scan_target(client, monkeypatch):
    """Only the ST path may diverge: a llama download_repo is the GGUF companion,
    which is not the repo whose pickles this gate is about."""
    c, _saved = client
    seen = {}

    def _scan(name, **_kwargs):
        seen["scanned"] = name
        return _Decision(False)

    mod = _types.ModuleType("utils.security")
    mod.security_load_subdirs = lambda name, token = None: ()
    mod.evaluate_file_security = _scan
    monkeypatch.setitem(sys.modules, "utils.security", mod)
    import utils.models as models

    monkeypatch.setattr(models, "is_embedding_model", lambda *a, **k: True)
    monkeypatch.setattr(settings, "_llama_backend_active", lambda *_: True)
    monkeypatch.setattr(
        settings,
        "_resolve_embedding_model_plan",
        lambda model, token: settings.EmbeddingModelResolveResponse(
            embedding_model = model, backend = "llama", download_repo = f"{model}-GGUF"
        ),
    )

    r = c.put("/embedding-model", json = {"embedding_model": "acme/embedder"})
    assert r.status_code == 200
    assert "scanned" not in seen


def test_offline_cached_acceptance_still_asks_who_is_asking(client, monkeypatch):
    """The offline branch above accepts a cached transformers-native embedder that HF
    metadata cannot verify. What makes that safe is the authorization check beside the
    loadable check: without it, an API key that cannot reach the repo learns the operator
    has it cached and gets it persisted as this deployment's embedder."""
    from hub.utils import hf_tokens

    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import utils.models as _models
    import utils.utils as _uu

    monkeypatch.setattr(_models, "is_embedding_model", lambda *a, **k: False)
    monkeypatch.setattr(_uu, "hf_cache_snapshot_is_loadable", lambda name: True)
    c.app.dependency_overrides[settings.allow_ambient_hf_token] = lambda: False
    hf_tokens.reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: False)

    r = c.put(
        "/embedding-model",
        json = {"embedding_model": "acme/private-embedder", "hf_token": "hf_dummy"},
    )

    assert r.status_code == 409
    assert saved.get("model") != "acme/private-embedder"
    hf_tokens.reset_repo_access_cache()


def _custom_module_repo(tmp_path):
    """A local embedding repo whose modules.json names a repo-hosted module class. Importing it
    drops a marker file, so a test can tell whether the repo's code ran."""
    import json

    marker = tmp_path / "ran.txt"
    (tmp_path / "custom_mod.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran')\n"
        "class Pooling:\n"
        "    pass\n"
    )
    (tmp_path / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "", "type": "custom_mod.Pooling"}])
    )
    return marker


def test_st_gate_refuses_repo_hosted_module_class_on_local_path(tmp_path):
    # sentence-transformers < 6 trusted repo code for any local path (CVE-2026-68770)
    st = pytest.importorskip("sentence_transformers")
    import core.rag.embeddings as embeddings

    marker = _custom_module_repo(tmp_path)
    embeddings._gate_st_custom_modules()
    with pytest.raises(ValueError, match = "not part of Sentence Transformers"):
        st.SentenceTransformer(str(tmp_path), device = "cpu")
    assert not marker.exists()


def test_st_gate_allows_stock_classes_and_explicit_trust(tmp_path):
    st = pytest.importorskip("sentence_transformers")
    import core.rag.embeddings as embeddings

    embeddings._gate_st_custom_modules()
    owner = next(
        c for c in st.SentenceTransformer.__mro__ if "_load_module_class_from_ref" in vars(c)
    )
    resolve = vars(owner)["_load_module_class_from_ref"]
    model = object.__new__(st.SentenceTransformer)
    pooling = resolve(
        model, "sentence_transformers.models.Pooling", str(tmp_path), False, None, None
    )
    assert pooling.__name__ == "Pooling"
    _custom_module_repo(tmp_path)
    resolve(model, "custom_mod.Pooling", str(tmp_path), True, None, None)


def test_st_gate_is_idempotent():
    st = pytest.importorskip("sentence_transformers")
    import core.rag.embeddings as embeddings

    embeddings._gate_st_custom_modules()
    owner = next(
        c for c in st.SentenceTransformer.__mro__ if "_load_module_class_from_ref" in vars(c)
    )
    first = vars(owner)["_load_module_class_from_ref"]
    embeddings._gate_st_custom_modules()
    assert vars(owner)["_load_module_class_from_ref"] is first
    if int(st.__version__.split(".")[0]) < 6:
        assert getattr(first, embeddings._ST_GATE_MARKER, False)
        assert not getattr(first.__wrapped__, embeddings._ST_GATE_MARKER, False)


def test_st_gate_covers_router_sub_module_types():
    # ST 5.0-5.4 Router resolves sub-module types via import_from_string, past the class resolvers
    st = pytest.importorskip("sentence_transformers")
    if int(st.__version__.split(".")[0]) >= 6:
        pytest.skip("sentence-transformers 6 gates this itself")
    import importlib

    import core.rag.embeddings as embeddings

    router = importlib.import_module("sentence_transformers.models.Router")
    if not hasattr(router, "import_from_string"):
        pytest.skip("this Router resolves sub-modules through the gated import_module_class")
    embeddings._gate_st_custom_modules()
    with pytest.raises(ValueError, match = "not part of Sentence Transformers"):
        router.import_from_string("custom_mod.Pooling")
    assert router.import_from_string("sentence_transformers.models.Pooling").__name__ == "Pooling"

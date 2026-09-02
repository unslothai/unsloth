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
    # The settings scan unions in the ST module dirs read from modules.json; keep it
    # offline and deterministic for the endpoint tests that use this fixture.
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


def _security_raises():
    mod = _types.ModuleType("utils.security")

    def _boom(*_a, **_k):
        raise RuntimeError("scan endpoint unreachable")

    mod.evaluate_file_security = _boom
    mod.security_load_subdirs = lambda *a, **k: ()
    return mod


def test_scan_error_fails_open_instead_of_500(client, monkeypatch):
    # A scan error is a gate failure, not a verdict; same policy as _guard_model_security.
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_raises())
    import utils.models as models

    monkeypatch.setattr(models, "is_embedding_model", lambda *a, **k: True)

    r = c.put("/embedding-model", json = {"embedding_model": "acme/embedder"})

    assert r.status_code == 200
    assert saved["model"] == "acme/embedder"


def test_uncached_selection_is_marked_pending_so_loaders_stay_offline(client, monkeypatch):
    c, saved = client
    monkeypatch.setitem(sys.modules, "utils.security", _security_stub(blocked = False))
    import utils.models as models

    monkeypatch.setattr(models, "is_embedding_model", lambda *a, **k: True)
    response = c.put("/embedding-model", json = {"embedding_model": "acme/embedder"})

    assert response.status_code == 200
    assert saved["download_pending"] is True


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


def test_offline_cached_non_st_model_is_accepted(client, monkeypatch):
    # Offline, a cached transformers-native embedder (no modules.json) is unverifiable via HF
    # metadata, but ST can load any cached encoder, so accept it (no 409).
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
    # Offline but not loadable (uncached or metadata-only partial cache): keep the forceable
    # 409, since the cache-only load would fail anyway.
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
    # Offline + llama backend: the remote GGUF probe (list_repo_files) must be skipped so a
    # dead-DNS session cannot hang.
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
    # On the llama-server backend the embedder loads GGUF (inert), not the ST repo's
    # pickle, so a flagged ST repo with a clean GGUF companion must not be rejected here.
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
    # Deliberately do NOT monkeypatch settings._llama_backend_active: this test exercises the
    # real delegation to embeddings.active_backend_is_llama() so the cached fallback is honored.
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
        # Nothing is held in this process, so Unload has nothing to offer.
        "loaded": False,
        # Nor is any other model, so the Unload control stays hidden too.
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
    embeddings._guard_model_security("acme/clean-embed")  # no raise


def test_sink_threads_ambient_token_into_scan(monkeypatch):
    # A gated repo set via env/default has no request token; the guard must feed the
    # loader's own token to the scan, or it fails open for the repo that still loads.
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
    # 409, so the client can still offer "save anyway" as it does for a GGUF error.
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
    # Nothing was validated, so nothing is claimed...
    assert saved["backend"] is None
    assert saved["gguf_repo"] is None
    # ...but the loader still may not download behind the user's back.
    assert saved["download_pending"] is True


def test_unload_is_offered_while_another_model_is_still_resident(client, monkeypatch):
    """Saving a new model does not release the old one, and `loaded` answers only
    about the selected one, so the previous model had no control to free it."""
    c, _saved = client
    import core.rag.embeddings as embeddings

    # A is resident; B is what Settings now names.
    monkeypatch.setattr(embeddings, "backend_is_loaded", lambda model_name = None: model_name is None)

    body = c.get("/embedding-model").json()
    assert body["loaded"] is False
    assert body["backend_loaded"] is True

    # Nothing resident at all: neither is claimed.
    monkeypatch.setattr(embeddings, "backend_is_loaded", lambda model_name = None: False)
    body = c.get("/embedding-model").json()
    assert body["loaded"] is False
    assert body["backend_loaded"] is False


def test_the_resolved_repo_is_what_gets_verified_and_scanned(client, monkeypatch):
    """A slashless alias resolves under sentence-transformers/, but the PUT ran
    is_embedding_model and the malware scan against the literal name: a repo that
    usually does not exist (fail-open, or a forceable 409) or, worse, a different
    top-level repo that does."""
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
    # The setting keeps what the user picked...
    assert saved["model"] == "all-MiniLM-L6-v2"
    # ...but every check ran against the repo the loader will open.
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
    # The llama path does not scan the ST repo at all, so nothing was scanned.
    assert "scanned" not in seen

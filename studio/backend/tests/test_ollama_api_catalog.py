# SPDX-License-Identifier: AGPL-3.0-only

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import local_model_resolver as resolver
from hub.services.models import ollama
from routes import inference as inf, models as models_route
from storage import studio_db
from utils import paths, hf_cache_settings
from studio.backend.tests.test_legacy_ollama_source import _write_ollama_store
from studio.backend.tests.test_openai_auto_switch import _reset_keepwarm


_NO_ORCHESTRATOR = SimpleNamespace(active_model_name = None, models = {})


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    _write_ollama_store(root)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(paths, "ollama_model_dirs", lambda: [root])
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    monkeypatch.setattr(paths, "lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr(paths, "hermes_model_dirs", lambda: [])
    monkeypatch.setattr(paths, "legacy_hf_cache_dir", lambda: empty)
    monkeypatch.setattr(paths, "hf_default_cache_dir", lambda: empty)
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: empty)
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [])
    monkeypatch.setattr(inf, "_CATALOG_CACHE", {"at": 0.0, "models": []})
    monkeypatch.setattr(inf, "_SERVABLE_SCAN_CACHE", {"entry": None})
    monkeypatch.setattr(inf, "_peek_inference_backend", lambda: None)
    for empty_index in ("_media_model_objects", "_stt_model_objects"):
        monkeypatch.setattr(inf, empty_index, lambda *args: [])
    resolver.invalidate_index()
    _reset_keepwarm()
    yield root
    resolver.invalidate_index()
    _reset_keepwarm()


def test_the_compat_inventory_still_hands_out_a_gguf_path(store):
    """Its readers treat a row's id and path as filenames; a suffixless blob is not loadable."""
    (row,) = models_route.collect_local_models(Path("models").resolve())
    assert row.model_id == "ollama/llama3:latest"
    assert row.id == row.path
    assert row.path.endswith(".gguf")
    assert Path(row.path).is_file()


def _store_contents(store):
    return sorted(
        (
            str(p.relative_to(store)),
            p.lstat().st_size,
            p.lstat().st_mtime_ns,
            os.readlink(p) if p.is_symlink() else None,
        )
        for p in store.rglob("*")
    )


def test_catalog_and_resolver_are_read_only_and_share_the_public_id(store, monkeypatch):
    before = _store_contents(store)
    (row,) = models_route.collect_local_models(
        Path("models").resolve(), materialize_ollama_links = False
    )
    assert row.model_id == "ollama/llama3:latest"
    assert row.id.startswith("ollama-manifest:")
    assert row.model_format == "gguf"
    assert resolver.local_servable_model(row) == (True, ())
    assert resolver.resolve_local_gguf(row.model_id) == (row.id, None, row.model_id)
    assert _store_contents(store) == before
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [{"path": str(store)}])
    assert (
        len(
            models_route.collect_local_models(
                Path("models").resolve(), materialize_ollama_links = False
            )
        )
        == 1
    )


def test_a_repulled_tag_is_not_reported_loaded_while_it_cannot_be_answered(store, monkeypatch):
    """A re-pulled tag no longer names the resident weights, so its entry is withdrawn."""
    from core.inference.llama_cpp import LlamaCppBackend

    ref = models_route._scan_ollama_dir(store, materialize_links = False)[0].id
    materialized = ollama.materialize_ollama_model_ref(ref)
    backend = SimpleNamespace(
        is_loaded = True,
        model_identifier = ref,
        _openai_advertised_id = "ollama/llama3:latest",
        hf_variant = None,
        gguf_path = materialized,
        context_length = 4096,
        max_context_length = 4096,
        native_context_length = 4096,
        _is_audio = False,
        _audio_type = None,
        _gguf_load_identity = LlamaCppBackend._gguf_load_source_identity(materialized),
    )
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _NO_ORCHESTRATOR)
    assert inf._loaded_satisfies("ollama/llama3:latest") is True
    assert [m["id"] for m in inf._openai_model_objects()] == ["ollama/llama3:latest"]

    manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
    data = json.loads(manifest.read_text())
    data["layers"][0]["digest"] = "sha256:" + "c" * 64
    (store / "blobs" / ("sha256-" + "c" * 64)).write_bytes(b"GGUF-repulled")
    manifest.write_text(json.dumps(data))

    assert inf._loaded_satisfies("ollama/llama3:latest") is False
    assert inf._openai_model_objects() == []

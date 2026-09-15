# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json
from inspect import getsource
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import local_model_resolver as resolver
from hub.services.models import account_access, ollama
from models.inference import LoadRequest
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
    for module in (paths, ollama):
        monkeypatch.setattr(module, "ollama_model_dirs", lambda: [root])
    for name in ("lmstudio_model_dirs", "hermes_model_dirs"):
        monkeypatch.setattr(paths, name, lambda: [])
    for name in ("legacy_hf_cache_dir", "hf_default_cache_dir"):
        monkeypatch.setattr(paths, name, lambda: empty)
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
    def _entry(p):
        stat = p.lstat()
        return str(p.relative_to(store)), stat.st_size, stat.st_mtime_ns, p.is_symlink()

    return sorted(_entry(p) for p in store.rglob("*"))


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
    rows = models_route.collect_local_models(
        Path("models").resolve(), materialize_ollama_links = False
    )
    assert len(rows) == 1


def _resident(ref: str) -> SimpleNamespace:
    from core.inference.llama_cpp import LlamaCppBackend
    materialized = ollama.materialize_ollama_model_ref(ref)
    return SimpleNamespace(
        is_loaded = True,
        model_identifier = ref,
        _openai_advertised_id = ollama.ollama_model_ref_public_id(ref),
        hf_variant = None,
        gguf_path = materialized,
        context_length = 4096,
        max_context_length = 4096,
        native_context_length = 4096,
        _is_audio = False,
        _audio_type = None,
        _gguf_load_identity = LlamaCppBackend._gguf_load_source_identity(materialized),
    )


def test_a_repulled_tag_is_not_reported_loaded_while_it_cannot_be_answered(store, monkeypatch):
    """A re-pulled tag no longer names the resident weights, so its entry is withdrawn."""
    ref = models_route._scan_ollama_dir(store, materialize_links = False)[0].id
    backend = _resident(ref)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _NO_ORCHESTRATOR)
    assert inf._loaded_satisfies("ollama/llama3:latest") is True
    assert [m["id"] for m in inf._openai_model_objects()] == ["ollama/llama3:latest"]
    # The load identity is recorded before the server answers and outlives a failed load.
    backend.is_loaded = False
    assert inf._resolves_to_resident(ref, llama_only = True) is False
    backend.is_loaded = True

    manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
    data = json.loads(manifest.read_text())
    data["layers"][0]["digest"] = "sha256:" + "c" * 64
    (store / "blobs" / ("sha256-" + "c" * 64)).write_bytes(b"GGUF-repulled")
    manifest.write_text(json.dumps(data))

    assert inf._loaded_satisfies("ollama/llama3:latest") is False
    assert inf._openai_model_objects() == []


def _rewritten(model_path: str) -> str:
    return inf._as_ollama_manifest_request(LoadRequest(model_path = model_path)).model_path


def test_a_link_is_loaded_as_the_tag_that_materialized_it(store, monkeypatch):
    """Its directory names the tag, so a selection stored under an older link name resolves too."""
    ref = models_route._scan_ollama_dir(store, materialize_links = False)[0].id
    link = Path(ollama.materialize_ollama_model_ref(ref))
    data = json.loads((store / "manifests/registry.ollama.ai/library/llama3/latest").read_text())
    for named in (link, link.with_name("llama3-latest-Q4_K_M.gguf")):
        assert _rewritten(str(named)) == ref
    assert _rewritten(str(store / "blobs")) == str(store / "blobs")
    # An authorized request keeps its own path: the tag may name blobs the authorization never
    # judged, and the resolver answers an Ollama reference before it ever verifies a lease.
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    assert _rewritten(str(link)) == str(link)
    monkeypatch.setattr(account_access, "managed_account", lambda: False)
    leased = LoadRequest(model_path = str(link), native_path_lease = "lease")
    assert inf._as_ollama_manifest_request(leased).model_path == str(link)
    # A tag that no longer reads leaves a link that still resolves alone.
    manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
    manifest.write_text("truncated")
    assert _rewritten(str(link)) == str(link)
    manifest.write_text(json.dumps(data))
    # Both routes bind the rewrite back, and only once the account has authorized the spelling
    # the caller actually sent: substituting an identity ahead of that check widens access.
    for route in (inf._load_model_impl, inf.validate_model):
        body = getsource(route)
        rewrite = body.index(
            "request = await asyncio.to_thread(_as_ollama_manifest_request, request)"
        )
        guard = body.index("account_access.require_model_access, request.model_path")
        assert guard < rewrite, route.__name__


def test_a_read_only_store_s_fallback_link_resolves_too(store, monkeypatch, tmp_path):
    """A read-only or sandboxed install puts its links beside the cache, not beside the blobs."""
    ref = models_route._scan_ollama_dir(store, materialize_links = False)[0].id
    monkeypatch.setattr(ollama, "cache_root", lambda: tmp_path / "cache")
    # A file, not a read-only directory: mkdir refuses it on every platform, root included.
    (store / ".studio_links").write_text("")
    link = Path(ollama.materialize_ollama_model_ref(ref))
    assert "ollama_links" in link.parts and ".studio_links" not in link.parts
    assert _rewritten(str(link)) == ref


def test_a_tag_loaded_by_its_link_is_listed_once(store, monkeypatch):
    """The load records the tag, so the scanned row and the resident are one entry, not two."""
    ref = models_route._scan_ollama_dir(store, materialize_links = False)[0].id
    link = ollama.materialize_ollama_model_ref(ref)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _resident(_rewritten(link)))
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _NO_ORCHESTRATOR)
    assert [(m["id"], m["loaded"]) for m in asyncio.run(inf._openai_catalog_objects())] == [
        ("ollama/llama3:latest", True)
    ]
    assert inf._loaded_satisfies("ollama/llama3:latest") is True


def test_a_symlinked_manifests_dir_still_resolves(tmp_path, monkeypatch):
    """A reference carries the canonical path, a scan the spelling it walked."""
    staging, elsewhere, root = (tmp_path / name for name in ("staging", "elsewhere", "ollama"))
    _write_ollama_store(staging)
    elsewhere.mkdir()
    root.mkdir()
    (staging / "manifests").rename(elsewhere / "manifests")
    (staging / "blobs").rename(root / "blobs")
    (root / "manifests").symlink_to(elsewhere / "manifests", target_is_directory = True)
    for module in (paths, ollama):
        monkeypatch.setattr(module, "ollama_model_dirs", lambda: [root])

    (row,) = models_route._scan_ollama_dir(root, materialize_links = False)
    assert ollama.ollama_model_ref_files(row.id)[0].startswith(str(root / "blobs"))

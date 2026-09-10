# SPDX-License-Identifier: AGPL-3.0-only

import json
import struct
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.inference import local_model_resolver as resolver
from hub.services.models import ollama
from routes import inference as inf, models as models_route
from storage import studio_db
from utils import paths, hf_cache_settings, openai_auto_switch_settings
from studio.backend.tests.test_legacy_ollama_source import _write_ollama_store
from studio.backend.tests.test_openai_auto_switch import (
    _FakeBackend,
    _LoadRecorder,
    _reset_keepwarm,
)


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
    monkeypatch.setattr(inf, "_media_model_objects", lambda *args: [])
    monkeypatch.setattr(inf, "_stt_model_objects", lambda *args: [])
    resolver.invalidate_index()
    _reset_keepwarm()
    yield root
    resolver.invalidate_index()
    _reset_keepwarm()


def test_catalog_and_resolver_are_read_only_and_share_the_public_id(store, monkeypatch):
    before = sorted(str(p.relative_to(store)) for p in store.rglob("*"))
    rows = models_route.collect_local_models(Path("models").resolve())
    assert len(rows) == 1
    row = rows[0]
    assert row.model_id == "ollama/llama3:latest"
    assert row.id.startswith("ollama-manifest:")
    assert row.model_format == "gguf"
    assert resolver.local_servable_model(row) == (True, ())
    assert resolver.resolve_local_gguf(row.model_id) == (row.id, None, row.model_id)
    assert sorted(str(p.relative_to(store)) for p in store.rglob("*")) == before
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [{"path": str(store)}])
    assert len(models_route.collect_local_models(Path("models").resolve())) == 1


@pytest.mark.parametrize("broken", ["runtime_layer", "missing_blob"])
def test_unsupported_ollama_models_are_withheld(store, broken):
    tag = store / "manifests/registry.ollama.ai/library/llama3/latest"
    manifest = json.loads(tag.read_text())
    if broken == "runtime_layer":
        manifest["layers"].append(
            {"mediaType": "application/vnd.ollama.image.adapter", "digest": "sha256:" + "c" * 64}
        )
        tag.write_text(json.dumps(manifest))
    else:
        next((store / "blobs").iterdir()).unlink()
    assert models_route.collect_local_models(Path("models").resolve()) == []
    assert resolver.resolve_local_gguf("ollama/llama3:latest") is None


@pytest.mark.parametrize("tag", ["latest", "Q8_0", "IQ3_M"])
def test_http_catalog_id_autoloads_without_prior_ui_load(store, monkeypatch, tag):
    manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
    manifest.rename(manifest.with_name(tag))
    model_id = f"ollama/llama3:{tag}"
    backend = _FakeBackend()
    backend.is_vision = False
    backend.supports_tools = False
    backend.supports_tool_passthrough = False
    backend.context_length = 4096
    backend.count_chat_tokens = lambda *args, **kwargs: 4
    backend.generate_chat_completion = lambda **kwargs: iter(["Fixture response."])
    recorder = _LoadRecorder(backend)
    seen = []

    async def load(request, *args, **kwargs):
        with ExitStack() as stack:
            linked = await inf._lease_ollama_model_ref(request, operation = "load", stack = stack)
            resolved, _, _ = inf._resolve_model_identifier_for_request(
                request, operation = "load", resolved_ollama_path = linked
            )
            assert Path(resolved).suffix == ".gguf"
            assert Path(resolved).read_bytes() == b"GGUF-not-really"
            seen.append((request.model_path, resolved))
            return await recorder(request, *args, **kwargs)

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inf, "_load_model_impl", load)
    monkeypatch.setattr(inf, "_openai_model_objects", lambda: [])
    monkeypatch.setattr(openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inf, "current_date_prompt_line", lambda **kwargs: "")
    monkeypatch.setattr(inf, "_auto_switch_waiters", {})
    app = FastAPI()
    app.include_router(inf.router, prefix = "/v1")
    app.dependency_overrides[inf.get_current_subject] = lambda: "test"
    with TestClient(app) as client:
        catalog = client.get("/v1/models")
        assert catalog.status_code == 200, catalog.text
        model = catalog.json()["data"][0]
        assert model["id"] == model_id
        assert model["loaded"] is False
        assert seen == []
        response = client.post(
            "/v1/chat/completions",
            json = {
                "model": model["id"],
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "Fixture response."
        assert response.json()["model"] == model["id"]
        repeated = client.post(
            "/v1/chat/completions",
            json = {
                "model": model["id"],
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Say hello again."}],
            },
        )
        assert repeated.status_code == 200, repeated.text
    assert len(seen) == 1
    assert seen[0][0].startswith("ollama-manifest:")
    assert backend._openai_advertised_id == model_id


@pytest.mark.parametrize("tag", ["latest", "Q8_0", "8b"])
def test_resident_ollama_identity_keeps_the_manifest_tag(monkeypatch, tag):
    model_id = f"ollama/llama3:{tag}"
    backend = _FakeBackend("ollama-manifest:fixture", advertised_id = model_id)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    for satisfies in (inf._loaded_satisfies, inf._loaded_identity_satisfies):
        assert satisfies(model_id)
        assert not satisfies("ollama/llama3:Q4_K_M")


def test_ollama_vision_preflight_reads_projector_without_materializing(store, monkeypatch):
    row = models_route._scan_ollama_dir(store)[0]
    assert inf._target_is_vision(row.id) is False
    tag = store / "manifests/registry.ollama.ai/library/llama3/latest"
    manifest = json.loads(tag.read_text())
    digest = "c" * 64
    projector = store / "blobs" / f"sha256-{digest}"
    projector.write_bytes(b"GGUF projector")
    manifest["layers"].append(
        {"mediaType": "application/vnd.ollama.image.projector", "digest": f"sha256:{digest}"}
    )
    tag.write_text(json.dumps(manifest))
    from utils.models import gguf_metadata

    monkeypatch.setattr(gguf_metadata, "mmproj_accepts_image", lambda path: path == str(projector))
    assert inf._target_is_vision(row.id) is True
    assert inf._resolve_target_gguf_file(row.id, None) == row.path
    assert not (store / ".studio_links").exists()


def test_removed_ollama_manifest_defers_vision_failure_to_load(store):
    load_path, _, _ = resolver.resolve_local_gguf("ollama/llama3:latest")
    (store / "manifests/registry.ollama.ai/library/llama3/latest").unlink()
    assert inf._target_accepts_request_input(
        load_path, is_gguf = True, needs_vision = True, needs_audio = False
    )
    with pytest.raises(ValueError):
        ollama.ollama_model_ref_files(load_path)


def test_ollama_speech_blob_keeps_task_and_codec_without_materializing(store):
    from hub.services.models import catalog_classification

    manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
    speech_manifest = store / "manifests/registry.ollama.ai/legraphista/Orpheus/3b-ft-q4_k_m"
    speech_manifest.parent.mkdir(parents = True)
    manifest.rename(speech_manifest)
    blob = store / "blobs" / ("sha256-" + "b" * 64)

    def gguf_string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    blob.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + gguf_string("general.architecture")
        + struct.pack("<I", 8)
        + gguf_string("llama")
    )
    row = models_route._scan_ollama_dir(store)[0]
    assert Path(row.path) == blob
    assert catalog_classification._local_model_task(row) == "text-to-speech"
    assert catalog_classification._local_model_audio_type(row) == "snac"
    assert not (store / ".studio_links").exists()

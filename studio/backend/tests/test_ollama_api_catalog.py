# SPDX-License-Identifier: AGPL-3.0-only

import json
import os
import struct
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

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


def test_cached_ollama_row_requires_its_manifest(store):
    row = models_route._scan_ollama_dir(store)[0]
    manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
    manifest.with_name("copy").write_bytes(manifest.read_bytes())
    manifest.unlink()
    assert Path(row.path).is_file()
    assert resolver.local_servable_model(row) is None


def test_ollama_catalog_and_resolver_share_root_precedence(store, tmp_path, monkeypatch):
    second = tmp_path / "second"
    _write_ollama_store(second)
    roots = [store, second]
    for root, timestamp, model_type in ((store, 100, "8B"), (second, 200, "70B")):
        manifest = root / "manifests/registry.ollama.ai/library/llama3/latest"
        data = json.loads(manifest.read_text())
        data["config"] = {"digest": "sha256:" + "c" * 64}
        (root / "blobs" / ("sha256-" + "c" * 64)).write_text(json.dumps({"model_type": model_type}))
        manifest.write_text(json.dumps(data))
        os.utime(manifest, (timestamp, timestamp))
    monkeypatch.setattr(paths, "ollama_model_dirs", lambda: roots)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: roots)
    rows = models_route.collect_local_models(Path("models").resolve())
    assert len(rows) == 1
    assert "8B" in rows[0].display_name
    assert resolver.resolve_local_gguf(rows[0].model_id)[0] == rows[0].id


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


@pytest.mark.parametrize("hardlinks", [False, True])
@pytest.mark.parametrize("disabled_projector", [False, True])
def test_http_catalog_id_autoloads_without_prior_ui_load(
    store, monkeypatch, hardlinks, disabled_projector
):
    projector = None
    if disabled_projector:
        manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
        data = json.loads(manifest.read_text())
        data["layers"].append(
            {"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:" + "d" * 64}
        )
        projector = store / "blobs" / ("sha256-" + "d" * 64)
        projector.write_bytes(b"GGUF projector")
        manifest.write_text(json.dumps(data))
    monkeypatch.setattr(
        openai_auto_switch_settings,
        "resolve_override_for_load",
        lambda *args: ("ollama/llama3:latest", {"disable_vision": disabled_projector}),
    )
    if hardlinks:

        def no_symlinks(*args, **kwargs):
            raise OSError("Symlinks unavailable")

        monkeypatch.setattr(Path, "symlink_to", no_symlinks)
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
            seen.append((request.model_path, resolved, Path(resolved).read_bytes()))
            result = await recorder(request, *args, **kwargs)
            from core.inference.llama_cpp import LlamaCppBackend

            backend._disable_vision = request.disable_vision
            backend._gguf_load_identity = LlamaCppBackend._gguf_load_source_identity(
                resolved, str(projector) if projector else None
            )
            return result

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
        assert model["id"] == "ollama/llama3:latest"
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
        manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
        data = json.loads(manifest.read_text())
        data["layers"][0]["digest"] = "sha256:" + "c" * 64
        (store / "blobs" / ("sha256-" + "c" * 64)).write_bytes(b"GGUF-updated")
        manifest.write_text(json.dumps(data))
        retagged = client.post(
            "/v1/chat/completions",
            json = {
                "model": model["id"],
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Say hello again."}],
            },
        )
        assert retagged.status_code == 200, retagged.text
        monkeypatch.setattr(
            openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False
        )
        data["layers"][0]["digest"] = "sha256:" + "e" * 64
        (store / "blobs" / ("sha256-" + "e" * 64)).write_bytes(b"GGUF-not-loaded")
        manifest.write_text(json.dumps(data))
        refused = client.post(
            "/v1/chat/completions",
            json = {
                "model": model["id"],
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Say hello again."}],
            },
        )
        assert refused.status_code == 404, refused.text
    assert len(seen) == 2
    assert [item[2] for item in seen] == [b"GGUF-not-really", b"GGUF-updated"]
    assert seen[0][0].startswith("ollama-manifest:")
    assert backend._openai_advertised_id == "ollama/llama3:latest"


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


def test_manual_ollama_load_advertises_one_loaded_catalog_id(store, monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()

    async def launch(_backend, intent, _cancel_event):
        backend._process = SimpleNamespace(poll = lambda: None)
        backend._healthy = True
        backend._model_identifier = intent.model_identifier
        backend._gguf_path = intent.gguf_path
        backend._gguf_load_identity = backend._gguf_load_source_identity(intent.gguf_path)
        backend._arch_gate_forced_cpu = True
        return True

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(
        inf, "get_inference_backend", lambda: SimpleNamespace(active_model_name = None, models = {})
    )
    monkeypatch.setattr(inf, "_run_gguf_load_attempt", launch)
    monkeypatch.setattr(
        openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *args, **kwargs: 4)
    monkeypatch.setattr(backend, "generate_chat_completion", lambda **kwargs: iter(["Hello."]))
    app = FastAPI()
    app.include_router(inf.router, prefix = "/v1")
    app.dependency_overrides[inf.get_current_subject] = lambda: "test"
    row = models_route._scan_ollama_dir(store)[0]
    with TestClient(app) as client:
        loaded = client.post(
            "/v1/load",
            json = {"model_path": row.id, "gpu_layers": 0, "max_seq_length": 512},
        )
        assert loaded.status_code == 200, loaded.text
        catalog = client.get("/v1/models").json()["data"]
        assert len(catalog) == 1
        assert catalog[0]["id"] == row.model_id
        assert catalog[0]["loaded"] is True
        response = client.post(
            "/v1/chat/completions",
            json = {"model": row.model_id, "messages": [{"role": "user", "content": "Hi."}]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["model"] == row.model_id


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


@pytest.mark.parametrize("size", ["0.6b", "1.7b"])
def test_ollama_asr_size_tag_is_excluded_from_chat(store, size, monkeypatch):
    from hub.services.models import catalog_classification

    manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
    speech_manifest = store / "manifests/registry.ollama.ai/library/qwen3-asr" / size
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
        + gguf_string("qwen3")
    )
    row = models_route._scan_ollama_dir(store)[0]
    assert catalog_classification._local_model_task(row) == "automatic-speech-recognition"
    monkeypatch.setattr(inf, "_openai_model_objects", lambda: [])
    app = FastAPI()
    app.include_router(inf.router, prefix = "/v1")
    app.dependency_overrides[inf.get_current_subject] = lambda: "test"
    with TestClient(app) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert response.json()["data"] == []
    assert not (store / ".studio_links").exists()

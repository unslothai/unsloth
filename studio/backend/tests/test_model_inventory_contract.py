# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.models as models_route
import routes.training as training_route
from utils.models import model_config


def test_local_gguf_folder_emits_inference_only_row(tmp_path):
    root = tmp_path / "models"
    folder = root / "foo"
    folder.mkdir(parents = True)
    (folder / "model.Q4_K_M.gguf").write_bytes(b"gguf")

    rows = models_route._scan_models_dir(root)

    assert len(rows) == 1
    row = rows[0]
    assert row.id == str(folder)
    assert row.load_id == str(folder)
    assert row.inventory_id != row.id
    assert row.model_format == "gguf"
    assert row.runtime == "llama_cpp"
    assert row.capabilities.can_train is False
    assert row.capabilities.can_chat is True
    assert row.capabilities.requires_variant is True


def test_local_safetensors_folder_emits_trainable_row(tmp_path):
    root = tmp_path / "models"
    folder = root / "foo"
    folder.mkdir(parents = True)
    (folder / "config.json").write_text('{"model_type":"llama"}')
    (folder / "model.safetensors").write_bytes(b"weights")

    rows = models_route._scan_models_dir(root)

    assert len(rows) == 1
    row = rows[0]
    assert row.model_format == "safetensors"
    assert row.runtime == "transformers"
    assert row.capabilities.can_train is True
    assert row.capabilities.can_chat is True


def test_local_safetensors_without_config_stays_unknown(tmp_path):
    root = tmp_path / "models"
    folder = root / "foo"
    folder.mkdir(parents = True)
    (folder / "model.safetensors").write_bytes(b"weights")

    rows = models_route._scan_models_dir(root)

    assert len(rows) == 1
    row = rows[0]
    assert row.model_format == "unknown"
    assert row.capabilities.can_train is False
    assert row.capabilities.can_chat is False


def test_hf_cache_standard_safetensors_without_config_emits_trainable_row(tmp_path):
    repo = tmp_path / "models--unsloth--Gemma"
    snapshot = repo / "snapshots" / ("a" * 40)
    blobs = repo / "blobs"
    snapshot.mkdir(parents = True)
    blobs.mkdir()
    (blobs / "weight").write_bytes(b"weights")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"weights")

    rows = models_route._scan_hf_cache(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.id == "unsloth/Gemma"
    assert row.load_id == "unsloth/Gemma"
    assert row.path == str(repo)
    assert row.model_format == "safetensors"
    assert row.runtime == "transformers"
    assert row.capabilities.can_train is True
    assert row.capabilities.can_chat is True


def test_hf_cache_broken_safetensors_symlink_emits_partial_safetensors_row(tmp_path):
    repo = tmp_path / "models--unsloth--Partial"
    snapshot = repo / "snapshots" / ("a" * 40)
    blobs = repo / "blobs"
    snapshot.mkdir(parents = True)
    blobs.mkdir()
    (blobs / "config").write_text('{"model_type":"llama"}')
    (snapshot / "config.json").write_text('{"model_type":"llama"}')
    try:
        (snapshot / "model.safetensors").symlink_to("../../blobs/missing")
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this filesystem")

    rows = models_route._scan_hf_cache(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.id == "unsloth/Partial"
    assert row.load_id == "unsloth/Partial"
    assert row.model_format == "safetensors"
    assert row.runtime == "transformers"
    assert row.partial is True
    assert row.capabilities.can_train is False
    assert row.capabilities.can_chat is False


def test_hf_cache_config_only_repo_emits_partial_safetensors_row(tmp_path):
    repo = tmp_path / "models--unsloth--ConfigOnly"
    snapshot = repo / "snapshots" / ("a" * 40)
    blobs = repo / "blobs"
    snapshot.mkdir(parents = True)
    blobs.mkdir()
    (blobs / "config").write_text('{"model_type":"llama"}')
    (snapshot / "config.json").write_text('{"model_type":"llama"}')

    rows = models_route._scan_hf_cache(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.id == "unsloth/ConfigOnly"
    assert row.load_id == "unsloth/ConfigOnly"
    assert row.model_format == "safetensors"
    assert row.runtime == "transformers"
    assert row.partial is True
    assert row.capabilities.can_train is False
    assert row.capabilities.can_chat is False


def test_local_adapter_folder_emits_adapter_row_with_base_model(tmp_path):
    root = tmp_path / "models"
    folder = root / "checkpoint-60"
    folder.mkdir(parents = True)
    (folder / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "unsloth/Llama-3.2-3B",
                "peft_type": "LORA",
                "unsloth_training_method": "qlora",
            }
        )
    )
    (folder / "adapter_model.safetensors").write_bytes(b"adapter")

    rows = models_route._scan_models_dir(root)

    assert len(rows) == 1
    row = rows[0]
    assert row.model_format == "adapter"
    assert row.runtime == "adapter"
    assert row.base_model == "unsloth/Llama-3.2-3B"
    assert row.base_model_source == "huggingface"
    assert row.adapter_type == "LORA"
    assert row.training_method == "qlora"
    assert row.capabilities.can_train is False
    assert row.capabilities.can_chat is True


def test_local_adapter_base_model_relative_output_path_is_local(tmp_path):
    root = tmp_path / "models"
    folder = root / "checkpoint-60"
    folder.mkdir(parents = True)
    (folder / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "outputs/checkpoint-40"})
    )
    (folder / "adapter_model.safetensors").write_bytes(b"adapter")

    rows = models_route._scan_models_dir(root)

    assert len(rows) == 1
    row = rows[0]
    assert row.model_format == "adapter"
    assert row.base_model == "outputs/checkpoint-40"
    assert row.base_model_source == "local"


def test_local_mixed_folder_emits_distinct_format_rows(tmp_path):
    root = tmp_path / "models"
    folder = root / "foo"
    folder.mkdir(parents = True)
    (folder / "config.json").write_text('{"model_type":"llama"}')
    (folder / "model.safetensors").write_bytes(b"weights")
    (folder / "model.Q4_K_M.gguf").write_bytes(b"gguf")

    rows = models_route._scan_models_dir(root)
    by_format = {row.model_format: row for row in rows}

    assert set(by_format) == {"gguf", "safetensors"}
    assert by_format["gguf"].load_id == by_format["safetensors"].load_id
    assert by_format["gguf"].inventory_id != by_format["safetensors"].inventory_id
    assert by_format["gguf"].capabilities.can_train is False
    assert by_format["safetensors"].capabilities.can_train is True


def test_list_local_models_handles_empty_custom_folder_registry(monkeypatch, tmp_path):
    import storage.studio_db as studio_db
    import utils.paths as paths

    model_folder = tmp_path / "foo"
    model_folder.mkdir()
    (model_folder / "config.json").write_text('{"model_type":"llama"}')
    (model_folder / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr(paths, "hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr(paths, "lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr(paths, "ollama_model_dirs", lambda: [])
    monkeypatch.setattr(paths, "studio_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "outputs_root", lambda: tmp_path / "outputs")
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [])

    response = asyncio.run(
        models_route.list_local_models(
            models_dir = str(tmp_path),
            current_subject = "test",
        )
    )

    assert len(response.models) == 1
    assert response.models[0].model_format == "safetensors"


def test_model_config_respects_non_gguf_format_for_mixed_local_folder(
    monkeypatch,
    tmp_path,
):
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "config.json").write_text('{"model_type":"llama"}')
    (folder / "model.safetensors").write_bytes(b"weights")
    (folder / "model.Q4_K_M.gguf").write_bytes(b"gguf")

    def fail_gguf_detection(_path):
        raise AssertionError("GGUF detection should be skipped")

    monkeypatch.setattr(model_config, "detect_gguf_model", fail_gguf_detection)
    monkeypatch.setattr(model_config, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(model_config, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(model_config, "is_audio_input_type", lambda _value: False)

    config = model_config.ModelConfig.from_identifier(
        str(folder),
        model_format = "safetensors",
    )

    assert config is not None
    assert config.is_gguf is False
    assert config.path == str(folder)


def test_model_config_requires_variant_for_multi_quant_local_gguf_folder(tmp_path):
    folder = tmp_path / "gguf"
    folder.mkdir()
    (folder / "model-Q4_K_M.gguf").write_bytes(b"q4")
    (folder / "model-Q8_0.gguf").write_bytes(b"q8")

    config = model_config.ModelConfig.from_identifier(
        str(folder),
        model_format = "gguf",
    )

    assert config is None


def test_model_config_resolves_explicit_local_gguf_variant(tmp_path):
    folder = tmp_path / "gguf"
    folder.mkdir()
    q4 = folder / "model-Q4_K_M.gguf"
    q8 = folder / "model-Q8_0.gguf"
    q4.write_bytes(b"q4")
    q8.write_bytes(b"q8")

    config = model_config.ModelConfig.from_identifier(
        str(folder),
        model_format = "gguf",
        gguf_variant = "Q4_K_M",
    )

    assert config is not None
    assert config.is_gguf is True
    assert config.gguf_variant == "Q4_K_M"
    assert config.gguf_file == str(q4.resolve())


def test_model_config_allows_single_variant_local_gguf_folder(tmp_path):
    folder = tmp_path / "gguf"
    folder.mkdir()
    q4 = folder / "model-Q4_K_M.gguf"
    q4.write_bytes(b"q4")

    config = model_config.ModelConfig.from_identifier(
        str(folder),
        model_format = "gguf",
    )

    assert config is not None
    assert config.is_gguf is True
    assert config.gguf_variant == "Q4_K_M"
    assert config.gguf_file == str(q4.resolve())


def test_training_rejects_explicit_gguf_format():
    request = SimpleNamespace(
        model_format = "gguf",
        model_local_path = None,
        model_name = "Org/Model-GGUF",
    )

    with pytest.raises(HTTPException) as exc:
        training_route._reject_untrainable_model_request(request)

    assert exc.value.status_code == 400


def test_training_rejects_explicit_adapter_format():
    request = SimpleNamespace(
        model_format = "adapter",
        model_local_path = None,
        model_name = "outputs/run/adapter",
    )

    with pytest.raises(HTTPException) as exc:
        training_route._reject_untrainable_model_request(request)

    assert exc.value.status_code == 400


def test_training_rejects_local_gguf_only_when_format_omitted(tmp_path):
    folder = tmp_path / "gguf-only"
    folder.mkdir()
    (folder / "model.Q4_K_M.gguf").write_bytes(b"gguf")
    request = SimpleNamespace(
        model_format = None,
        model_local_path = str(folder),
        model_name = str(folder),
    )

    with pytest.raises(HTTPException) as exc:
        training_route._reject_untrainable_model_request(request)

    assert exc.value.status_code == 400


def test_training_rejects_local_adapter_only_when_format_omitted(tmp_path):
    folder = tmp_path / "adapter-only"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}")
    (folder / "adapter_model.safetensors").write_bytes(b"adapter")
    request = SimpleNamespace(
        model_format = None,
        model_local_path = str(folder),
        model_name = str(folder),
    )

    with pytest.raises(HTTPException) as exc:
        training_route._reject_untrainable_model_request(request)

    assert exc.value.status_code == 400

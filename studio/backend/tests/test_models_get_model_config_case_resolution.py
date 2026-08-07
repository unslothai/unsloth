# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sys
import types

import pytest

# Keep this test runnable where optional logging deps are not installed.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
import utils.models.model_config as model_config_module


def test_get_model_config_resolves_cached_case_before_model_checks(monkeypatch):
    calls: dict[str, str] = {}

    class _DummyModelConfig:
        is_lora = False
        base_model = None

    def _record_load(model_name):
        calls["load_model_defaults"] = model_name
        return {}

    def _record_vision(
        model_name,
        hf_token = None,
        local_files_only = False,
    ):
        calls["is_vision_model"] = model_name
        return False

    def _record_embedding(model_name, hf_token = None):
        calls["is_embedding_model"] = model_name
        return False

    def _record_audio(
        model_name,
        hf_token = None,
        local_files_only = False,
    ):
        calls["detect_audio_type"] = model_name
        return None

    def _record_from_identifier(
        cls,
        model_name,
        hf_token = None,
    ):
        calls["from_identifier"] = model_name
        return _DummyModelConfig()

    def _record_remote_size(model_name, hf_token = None):
        calls["model_size"] = model_name
        return 123

    monkeypatch.setattr(models_route, "is_local_path", lambda _: False)
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda _: "Org/Model")
    monkeypatch.setattr(models_route, "load_model_defaults", _record_load)
    monkeypatch.setattr(models_route, "is_vision_model", _record_vision)
    monkeypatch.setattr(models_route, "is_embedding_model", _record_embedding)
    monkeypatch.setattr(model_config_module, "detect_audio_type", _record_audio)
    monkeypatch.setattr(
        models_route.ModelConfig,
        "from_identifier",
        classmethod(_record_from_identifier),
    )
    monkeypatch.setattr(models_route, "_get_max_position_embeddings", lambda _: 4096)
    monkeypatch.setattr(models_route, "_get_model_size_bytes", _record_remote_size)

    result = asyncio.run(
        models_route.get_model_config(
            model_name = "org/model",
            hf_token = None,
            current_subject = "test-subject",
        )
    )

    assert result.model_name == "Org/Model"
    assert calls["load_model_defaults"] == "Org/Model"
    assert calls["is_vision_model"] == "Org/Model"
    assert calls["is_embedding_model"] == "Org/Model"
    assert calls["detect_audio_type"] == "Org/Model"
    assert calls["from_identifier"] == "Org/Model"
    assert calls["model_size"] == "Org/Model"
    assert result.model_size_bytes == 123


@pytest.mark.parametrize(
    "tokenizer_relative_path",
    ["tokenizer_config.json", "LLM/tokenizer_config.json"],
)
def test_get_model_config_inspects_selected_cache_snapshot(
    tokenizer_relative_path, tmp_path, monkeypatch
):
    calls: dict[str, object] = {}
    cache_root = tmp_path / "hub"
    repo_root = cache_root / "models--org--model"
    snapshot_path = repo_root / "snapshots" / "selected"
    snapshot_path.mkdir(parents = True)
    (snapshot_path / "config.json").write_text("{}", encoding = "utf-8")
    tokenizer_path = snapshot_path / tokenizer_relative_path
    tokenizer_path.parent.mkdir(parents = True, exist_ok = True)
    tokenizer_path.write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "1": {"content": "<|bicodec_global_0|>"},
                }
            }
        ),
        encoding = "utf-8",
    )
    blobs_path = repo_root / "blobs"
    blobs_path.mkdir()
    weight_blob = blobs_path / "weight-blob"
    weight_blob.write_bytes(b"selected-weights")
    (snapshot_path / "model.safetensors").symlink_to(weight_blob)
    (snapshot_path / "adapter.bin").write_bytes(b"adapter")
    outside_weight = tmp_path / "outside.pt"
    outside_weight.write_bytes(b"outside-cache-root")
    (snapshot_path / "escaped.pt").symlink_to(outside_weight)
    other_snapshot = repo_root / "snapshots" / "other"
    other_snapshot.mkdir()
    (other_snapshot / "config.json").write_text("{}", encoding = "utf-8")
    (other_snapshot / "model.safetensors").write_bytes(b"wrong-revision" * 10)
    refs_path = repo_root / "refs"
    refs_path.mkdir()
    (refs_path / "main").write_text("selected", encoding = "utf-8")
    snapshot = str(snapshot_path.resolve())
    expected_model_size = len(b"selected-weights") + len(b"adapter")

    class _DummyModelConfig:
        is_lora = False
        base_model = None

    from hub.utils import hf_cache_state

    resolve_snapshot = hf_cache_state.latest_snapshot_from_cache_path

    def _resolve_snapshot(local_path, repo_type, repo_id, metadata_filenames):
        calls["snapshot"] = (
            local_path,
            repo_type,
            repo_id,
            metadata_filenames,
        )
        return resolve_snapshot(
            local_path,
            repo_type,
            repo_id,
            metadata_filenames,
        )

    def _record_load(model_name):
        calls["load_model_defaults"] = model_name
        return {}

    def _record_vision(
        model_name,
        hf_token = None,
        local_files_only = False,
    ):
        calls["is_vision_model"] = (model_name, local_files_only)
        return True

    def _record_embedding(model_name, hf_token = None):
        calls["is_embedding_model"] = model_name
        return False

    def _record_from_identifier(
        cls,
        model_name,
        hf_token = None,
    ):
        calls["from_identifier"] = model_name
        return _DummyModelConfig()

    def _reject_remote_size(*_args, **_kwargs):
        raise AssertionError("cached config must not query remote model size")

    monkeypatch.setattr(
        "hub.utils.hf_cache_state.latest_snapshot_from_cache_path",
        _resolve_snapshot,
    )
    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_caches",
        lambda: [cache_root],
    )
    monkeypatch.setattr("hub.utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("hub.utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr(
        models_route,
        "is_local_path",
        lambda value: str(value).startswith(str(cache_root)),
    )
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(models_route, "load_model_defaults", _record_load)
    monkeypatch.setattr(models_route, "is_vision_model", _record_vision)
    monkeypatch.setattr(models_route, "is_embedding_model", _record_embedding)
    monkeypatch.setattr(
        models_route.ModelConfig,
        "from_identifier",
        classmethod(_record_from_identifier),
    )
    monkeypatch.setattr(models_route, "_get_max_position_embeddings", lambda _: 4096)
    monkeypatch.setattr(models_route, "_get_model_size_bytes", _reject_remote_size)

    result = asyncio.run(
        models_route.get_model_config(
            model_name = "org/model",
            hf_token = None,
            prefer_local_cache = True,
            local_path = str(repo_root),
            current_subject = "test-subject",
        )
    )

    assert result.model_name == "org/model"
    assert calls["load_model_defaults"] == "org/model"
    assert calls["snapshot"] == (
        str(repo_root),
        "model",
        "org/model",
        ("config.json", "adapter_config.json"),
    )
    assert calls["is_vision_model"] == (snapshot, True)
    assert calls["is_embedding_model"] == snapshot
    assert calls["from_identifier"] == snapshot
    assert result.is_audio is True
    assert result.audio_type == "bicodec"
    assert result.model_size_bytes == expected_model_size


@pytest.mark.parametrize("path_kind", ["missing", "mismatched"])
def test_get_model_config_rejects_invalid_selected_cache_path(path_kind, tmp_path, monkeypatch):
    cache_root = tmp_path / "hub"
    cache_root.mkdir()
    if path_kind == "mismatched":
        local_path = cache_root / "models--other--model"
        snapshot = local_path / "snapshots" / "selected"
        snapshot.mkdir(parents = True)
        (snapshot / "config.json").write_text("{}", encoding = "utf-8")
    else:
        local_path = cache_root / "models--org--model"

    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_caches",
        lambda: [cache_root],
    )
    monkeypatch.setattr("hub.utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("hub.utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda name: name)

    with pytest.raises(models_route.HTTPException) as exc_info:
        asyncio.run(
            models_route.get_model_config(
                model_name = "org/model",
                hf_token = None,
                prefer_local_cache = True,
                local_path = str(local_path),
                current_subject = "test-subject",
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Selected cached model is no longer available."


def test_repo_in_any_hf_cache_matches_case_variant_in_legacy_cache(tmp_path, monkeypatch):
    # A case-variant in a legacy/default cache must read as present: case resolution only covers the
    # active cache, but discard deletes case-insensitively, so detection must too.
    import utils.paths as paths_pkg
    import hub.utils.paths as hub_paths

    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    default = tmp_path / "default"
    for d in (active, legacy, default):
        d.mkdir()
    # Differently-cased entry in the legacy cache only.
    (legacy / "models--Unsloth--Foo").mkdir()

    # No active-cache variant; case resolution is a no-op here.
    monkeypatch.setattr(paths_pkg, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(hub_paths, "legacy_hf_cache_dir", lambda: legacy)
    monkeypatch.setattr(hub_paths, "hf_default_cache_dir", lambda: default)
    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_caches",
        lambda: [active],
    )

    assert models_route._repo_in_any_hf_cache("unsloth/foo") is True
    # Absent from every cache -> reported absent.
    assert models_route._repo_in_any_hf_cache("unsloth/not-cached") is False
